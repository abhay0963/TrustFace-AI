"""
05_presence_consistency_demo.py
==================================
CONCEPT: Turning raw event logs into an actual analytic - Presence
Consistency, not just Presence Duration.

WHAT THIS SCRIPT DOES
-----------------------
Picks a user who has attendance recorded today, buckets their session into
fixed-size time windows (config.PRESENCE_BUCKET_MINUTES), and prints/plots
an ASCII + graphical timeline of Detected vs Missing windows - exactly the
computation presence.compute_presence_consistency() performs inside the
live app, but shown step by step here so you can explain it in a viva.

WHY THIS MATTERS
-------------------
Two people can have similar "total presence_seconds" while having very
different actual behavior: one was there continuously, the other showed up
briefly at the start and end and was gone in between. Consistency captures
that difference; raw duration alone can't.

RUN
----
python learning_lab/05_presence_consistency_demo.py
"""

import sys
import os
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import database as db
import presence

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("LEARNING LAB 05: Presence Consistency Timeline")
    print("=" * 70)

    today = datetime.date.today().isoformat()
    sessions = db.get_today_attendance(today)

    if not sessions:
        print("\nNo attendance recorded today. Use Live Attendance for a bit, then re-run this.")
        return

    session = sessions[0]  # demo with the first user seen today
    print(f"\nUsing user: {session['name']} ({session['external_id']})")
    print(f"Entry time: {session['entry_time']}")
    print(f"Last seen:  {session['last_seen']}")

    result = presence.compute_presence_consistency(
        session["user_id"], today, session["entry_time"], session["last_seen"]
    )

    print(f"\nSession divided into {result['buckets_total']} buckets of "
          f"{result['bucket_minutes']} minutes each.")
    print(f"Detected in {result['buckets_detected']} / {result['buckets_total']} buckets.")
    print(f"Presence Consistency: {result['consistency_percentage']}%")

    print("\nTimeline (each block = one bucket):")
    symbols = "".join("■" if b == "detected" else "·" for b in result["timeline"])
    print(f"  [{symbols}]")
    print("  ■ = detected in that window   · = missing in that window")

    label = "continuous" if result["consistency_percentage"] >= 80 else \
            "partial" if result["consistency_percentage"] >= 50 else "interrupted"
    print(f"\nInterpretation: {result['consistency_percentage']}% -> '{label}' presence")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(max(6, len(result["timeline"]) * 0.4), 2))
    colors = ["#2dd4bf" if b == "detected" else "#212b35" for b in result["timeline"]]
    ax.bar(range(len(result["timeline"])), [1] * len(result["timeline"]), color=colors, width=1.0, edgecolor="#0a0e13")
    ax.set_yticks([])
    ax.set_xlabel(f"Time buckets ({result['bucket_minutes']} min each)")
    ax.set_title(f"{session['name']} - Presence Timeline ({result['consistency_percentage']}% consistent)")
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, "presence_timeline.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nSaved timeline plot to {filename}")


if __name__ == "__main__":
    main()
