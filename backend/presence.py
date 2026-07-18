"""
presence.py
============
MODULE 4: PRESENCE INTELLIGENCE

Instead of a flat "Present / Absent" flag, we track, per user per day:
  - entry_time          : first successful recognition today
  - last_seen           : most recent successful recognition today
  - presence_seconds    : total accumulated time considered "present"
  - ping_count          : how many recognition pings contributed
  - avg_trust_score     : running average trust score for the session
  - avg_confidence      : running average face-match similarity

PRESENCE ACCUMULATION LOGIC
------------------------------
Each time a face is recognized with decision == "auto_accept", this is a
"ping". We only add elapsed time to presence_seconds if the gap since the
last ping is below PRESENCE_TIMEOUT_SECONDS (meaning the person plausibly
stayed in frame/room the whole time). A large gap (e.g. they left and
came back an hour later) does NOT get counted as continuous presence.

PRESENCE PERCENTAGE
----------------------
    presence_percentage = presence_seconds / expected_session_seconds * 100
where expected_session_seconds is the configured session window
(SESSION_START_HOUR to SESSION_END_HOUR). This is intentionally simple -
NOT full multi-camera tracking - as scoped in the project brief.
"""

import datetime

from config import (
    PRESENCE_TIMEOUT_SECONDS, SESSION_START_HOUR, SESSION_END_HOUR, PRESENCE_BUCKET_MINUTES,
    PRESENCE_CATEGORY_BRIEF_MAX_PCT, PRESENCE_CATEGORY_CONTINUOUS_MIN_CONSISTENCY,
    PRESENCE_CATEGORY_SUSPICIOUS_TRANSITION_RATE,
)
import database as db


def _parse_iso(ts: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(ts)


def record_presence_ping(user_id: int, trust_score: float, similarity: float):
    """
    Call this whenever a user is recognized with decision == 'auto_accept'.
    Updates (or creates) today's attendance_sessions row.
    """
    now = datetime.datetime.now()
    session_date = now.date().isoformat()
    now_iso = now.isoformat()

    existing = db.get_today_session(user_id, session_date)

    if existing is None:
        db.upsert_session(
            user_id=user_id,
            session_date=session_date,
            entry_time=now_iso,
            last_seen=now_iso,
            presence_seconds=0.0,
            ping_count=1,
            avg_trust_score=trust_score,
            avg_confidence=similarity,
        )
        return

    last_seen = _parse_iso(existing["last_seen"])
    gap_seconds = (now - last_seen).total_seconds()

    presence_seconds = existing["presence_seconds"]
    if gap_seconds <= PRESENCE_TIMEOUT_SECONDS:
        presence_seconds += gap_seconds
    # else: treat as a re-entry, gap not counted toward presence

    ping_count = existing["ping_count"] + 1
    # running average update
    avg_trust = ((existing["avg_trust_score"] * existing["ping_count"]) + trust_score) / ping_count
    avg_conf = ((existing["avg_confidence"] * existing["ping_count"]) + similarity) / ping_count

    db.upsert_session(
        user_id=user_id,
        session_date=session_date,
        entry_time=existing["entry_time"],
        last_seen=now_iso,
        presence_seconds=presence_seconds,
        ping_count=ping_count,
        avg_trust_score=avg_trust,
        avg_confidence=avg_conf,
    )


def classify_session(presence_percentage: float, consistency: dict) -> str:
    """
    MODULE 4 UPGRADE (round 2): turn the raw numbers into a category a
    human reads instantly, instead of making every reader do the mental
    math from a percentage.

    WHY A TRANSITION RATE, NOT JUST THE CONSISTENCY %
    -------------------------------------------------------
    Two sessions can have the same consistency % with very different
    character: someone detected in buckets [1,2,3, ,5,6,7, ,9,10] (steady,
    occasional misses) vs. [1, ,3, ,5, ,7, ,9, ] (flickering every other
    bucket). We count DETECTED<->MISSING TRANSITIONS in the timeline and
    normalize by its length - a high transition rate means the presence
    signal is erratic/flickering rather than smoothly declining, which is
    exactly the pattern you'd also expect from camera issues, someone
    repeatedly walking in and out of frame, or a spoofing attempt being
    intermittently rejected - worth flagging distinctly from an ordinary
    "left early" case.

    CATEGORIES (checked in this priority order)
    ------------------------------------------------
    1. Brief Appearance        - presence_percentage below the "brief" floor,
                                  regardless of how consistent that brief
                                  window was (there just isn't much data).
    2. Continuous Attendance   - consistency at/above the "continuous" bar.
    3. Suspicious Intermittent - moderate/low consistency AND a high
                                  detected/missing transition rate.
    4. Frequent Exits          - moderate consistency, transitions are lower
                                  (fewer, longer absences rather than flicker).
    5. Interrupted Presence    - fallback for anything not caught above.
    """
    timeline = consistency.get("timeline", [])

    if presence_percentage < PRESENCE_CATEGORY_BRIEF_MAX_PCT:
        return "Brief Appearance"

    if consistency.get("consistency_percentage", 0) >= PRESENCE_CATEGORY_CONTINUOUS_MIN_CONSISTENCY:
        return "Continuous Attendance"

    if len(timeline) >= 2:
        transitions = sum(1 for i in range(1, len(timeline)) if timeline[i] != timeline[i - 1])
        transition_rate = transitions / (len(timeline) - 1)
    else:
        transition_rate = 0.0

    if transition_rate >= PRESENCE_CATEGORY_SUSPICIOUS_TRANSITION_RATE:
        return "Suspicious Intermittent Presence"

    if consistency.get("consistency_percentage", 0) >= 50:
        return "Frequent Exits"

    return "Interrupted Presence"


def expected_session_seconds() -> float:
    return (SESSION_END_HOUR - SESSION_START_HOUR) * 3600.0


def compute_presence_consistency(user_id: int, session_date: str, entry_time: str, last_seen: str) -> dict:
    """
    MODULE 4 UPGRADE: Presence Intelligence, not just presence logging.

    WHY THIS EXISTS
    -----------------
    `presence_seconds` alone can't distinguish "present continuously for
    2 hours" from "present for 5 minutes, gone for 90 minutes, present for
    5 more minutes" - both could accumulate similar totals depending on
    the timeout logic. CONSISTENCY fixes that by asking a different
    question: "of all the time windows between first and last detection,
    in how many was the person actually seen?"

    HOW IT WORKS
    --------------
    1. Take the span [entry_time, last_seen] and divide it into fixed-size
       buckets (PRESENCE_BUCKET_MINUTES, default 5 minutes) - like a
       timeline: [Detected][Detected][Missing][Detected][Detected]...
    2. For each bucket, check whether at least one auto_accept recognition
       timestamp falls inside it.
    3. consistency = (buckets with a detection) / (total buckets) * 100

    A person detected in every bucket scores 100% ("continuous"). Someone
    detected only at the start and end of a long session scores much lower
    ("interrupted") even if their raw presence_seconds total looks similar
    under a generous timeout window. This is a genuinely more informative
    signal than duration alone, and it's a good example of turning simple
    event logs into an actual analytical feature (Mistake #4 in the v1 review).
    """
    timestamps_str = db.get_accept_timestamps_for_session(user_id, session_date)
    if not timestamps_str:
        return {"consistency_percentage": 0.0, "buckets_total": 0, "buckets_detected": 0, "timeline": []}

    timestamps = [_parse_iso(t) for t in timestamps_str]
    # entry_time/last_seen are written by presence.record_presence_ping(),
    # which runs a few milliseconds AFTER the triggering recognition_logs
    # row is inserted - so the earliest/latest real detections can fall a
    # hair outside [entry_time, last_seen]. Widen the window to guarantee
    # every real detection is covered rather than dropped at a boundary.
    start = min(_parse_iso(entry_time), min(timestamps))
    end = max(_parse_iso(last_seen), max(timestamps))

    bucket_seconds = PRESENCE_BUCKET_MINUTES * 60
    total_span_seconds = max((end - start).total_seconds(), bucket_seconds)
    num_buckets = max(1, int(total_span_seconds // bucket_seconds) + 1)

    timeline = []
    detected_count = 0
    for i in range(num_buckets):
        bucket_start = start + datetime.timedelta(seconds=i * bucket_seconds)
        bucket_end = bucket_start + datetime.timedelta(seconds=bucket_seconds)
        hit = any(bucket_start <= ts < bucket_end for ts in timestamps)
        # the final bucket should also catch a timestamp exactly at `end`
        if i == num_buckets - 1:
            hit = hit or any(bucket_start <= ts <= end for ts in timestamps)
        timeline.append("detected" if hit else "missing")
        if hit:
            detected_count += 1

    consistency_pct = round((detected_count / num_buckets) * 100.0, 1)

    return {
        "consistency_percentage": consistency_pct,
        "buckets_total": num_buckets,
        "buckets_detected": detected_count,
        "bucket_minutes": PRESENCE_BUCKET_MINUTES,
        "timeline": timeline,
    }


def enrich_session_row(row: dict) -> dict:
    """Add human-friendly derived fields (presence %, formatted duration) to a raw DB row."""
    presence_seconds = row.get("presence_seconds", 0) or 0
    expected = expected_session_seconds()
    presence_pct = min(100.0, (presence_seconds / expected) * 100.0) if expected > 0 else 0.0

    hours = int(presence_seconds // 3600)
    minutes = int((presence_seconds % 3600) // 60)

    row["presence_percentage"] = round(presence_pct, 1)
    row["presence_duration_formatted"] = f"{hours}h {minutes}m"
    row["entry_time_formatted"] = _parse_iso(row["entry_time"]).strftime("%H:%M:%S")
    row["last_seen_formatted"] = _parse_iso(row["last_seen"]).strftime("%H:%M:%S")

    consistency = compute_presence_consistency(row["user_id"], row["session_date"], row["entry_time"], row["last_seen"])
    row["presence_consistency"] = consistency["consistency_percentage"]
    row["presence_timeline"] = consistency["timeline"]
    row["presence_category"] = classify_session(row["presence_percentage"], consistency)
    return row


def get_today_attendance_enriched():
    today = datetime.date.today().isoformat()
    rows = db.get_today_attendance(today)
    return [enrich_session_row(r) for r in rows]


def get_history_enriched(limit=200):
    rows = db.get_all_attendance_history(limit=limit)
    return [enrich_session_row(r) for r in rows]
