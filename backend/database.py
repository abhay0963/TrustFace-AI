"""
database.py
============
All database access for TrustFace AI lives here.

WHY SQLite?
-----------
SQLite is a single-file, zero-configuration database that ships with
Python's standard library (via the `sqlite3` module). For a solo,
laptop-based MVP it is the right choice: no server to install, the whole
database is one file (database/trustface.db) you can back up, inspect
with any SQLite viewer, or wipe by deleting the file.

SCHEMA OVERVIEW
----------------
users               -> registered identities + ENCRYPTED embeddings
attendance_sessions -> one row per (user, day): entry time, last seen,
                       presence duration, running trust/confidence stats
recognition_logs    -> one row per recognition ATTEMPT (accepted, retried,
                       rejected, or unknown), used for audits + research
"""

import sqlite3
import datetime
from contextlib import contextmanager

from config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_cursor():
    """Context manager so every caller gets automatic commit/close."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables if they do not already exist. Safe to call every startup."""
    with db_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                external_id     TEXT UNIQUE NOT NULL,   -- roll no / employee id
                embedding_enc   BLOB NOT NULL,           -- AES-encrypted embedding
                created_at      TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER NOT NULL,
                session_date        TEXT NOT NULL,   -- YYYY-MM-DD
                entry_time          TEXT NOT NULL,   -- first-seen timestamp
                last_seen           TEXT NOT NULL,   -- most recent recognition
                presence_seconds    REAL NOT NULL DEFAULT 0,
                ping_count          INTEGER NOT NULL DEFAULT 0,
                avg_trust_score     REAL NOT NULL DEFAULT 0,
                avg_confidence      REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, session_date)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER,               -- NULL if unknown face
                timestamp       TEXT NOT NULL,
                similarity      REAL,
                trust_score     REAL,
                decision        TEXT NOT NULL,          -- auto_accept | retry | reject | unknown (RULE-BASED raw decision, always computed)
                blur_score      REAL,
                brightness_score REAL,
                pose_score      REAL,
                face_size_score REAL,
                spoof_score     REAL,
                human_label     TEXT,                    -- accept | retry | reject, set by a human reviewer (ground truth for ML training)
                ml_decision     TEXT,                     -- raw ML model prediction, if a model exists at the time (audit/comparison only)
                final_decision  TEXT,                      -- the SINGLE authoritative decision actually acted on (rule until ML is promoted, then ML)
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)

        # ---- lightweight migration for DBs created before human_label / ml_decision / final_decision existed ----
        existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(recognition_logs)").fetchall()}
        if "human_label" not in existing_cols:
            cur.execute("ALTER TABLE recognition_logs ADD COLUMN human_label TEXT")
        if "ml_decision" not in existing_cols:
            cur.execute("ALTER TABLE recognition_logs ADD COLUMN ml_decision TEXT")
        if "final_decision" not in existing_cols:
            cur.execute("ALTER TABLE recognition_logs ADD COLUMN final_decision TEXT")
            # Backfill: for any existing rows, the rule-based decision WAS the
            # authoritative one at the time (ML promotion didn't exist yet).
            cur.execute("UPDATE recognition_logs SET final_decision = decision WHERE final_decision IS NULL")


# -----------------------------------------------------------------------
# USER CRUD
# -----------------------------------------------------------------------
def create_user(name: str, external_id: str, embedding_enc: bytes) -> int:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO users (name, external_id, embedding_enc, created_at) VALUES (?, ?, ?, ?)",
            (name, external_id, embedding_enc, datetime.datetime.now().isoformat()),
        )
        return cur.lastrowid


def get_all_users():
    with db_cursor() as cur:
        cur.execute("SELECT id, name, external_id, created_at FROM users ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]


def get_all_users_with_embeddings():
    """Used by the recognition engine to build its in-memory matching gallery."""
    with db_cursor() as cur:
        cur.execute("SELECT id, name, external_id, embedding_enc FROM users")
        return [dict(row) for row in cur.fetchall()]


def get_user_by_id(user_id: int):
    with db_cursor() as cur:
        cur.execute("SELECT id, name, external_id, created_at FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def delete_user(user_id: int):
    with db_cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))


def external_id_exists(external_id: str) -> bool:
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE external_id = ?", (external_id,))
        return cur.fetchone() is not None


# -----------------------------------------------------------------------
# RECOGNITION LOGS
# -----------------------------------------------------------------------
def insert_recognition_log(user_id, similarity, trust_score, decision, quality: dict, ml_decision=None, final_decision=None):
    with db_cursor() as cur:
        cur.execute("""
            INSERT INTO recognition_logs
                (user_id, timestamp, similarity, trust_score, decision,
                 blur_score, brightness_score, pose_score, face_size_score, spoof_score,
                 ml_decision, final_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, datetime.datetime.now().isoformat(), similarity, trust_score, decision,
            quality.get("blur"), quality.get("brightness"), quality.get("pose"),
            quality.get("face_size"), quality.get("spoof"), ml_decision,
            final_decision or decision,  # default: rule decision was authoritative
        ))
        return cur.lastrowid


def set_human_label(log_id: int, label: str):
    """
    Human-in-the-loop labeling: a reviewer looks at a logged recognition
    attempt (and the frame it came from, in a real deployment) and confirms
    what the CORRECT decision actually was. This becomes ground truth for
    training the ML Trust Engine - the rule-based `decision` column is a
    heuristic guess, `human_label` is the trusted answer.
    """
    with db_cursor() as cur:
        cur.execute("UPDATE recognition_logs SET human_label = ? WHERE id = ?", (label, log_id))


def get_labeled_training_data():
    """
    Returns every recognition_log row that has a human_label set, with the
    six feature columns the ML Trust Engine trains on. This is literally
    the (X, y) dataset for supervised learning: X = quality/similarity
    features, y = human_label.
    """
    with db_cursor() as cur:
        cur.execute("""
            SELECT similarity, blur_score, brightness_score, pose_score,
                   face_size_score, spoof_score, human_label
            FROM recognition_logs
            WHERE human_label IS NOT NULL
        """)
        return [dict(row) for row in cur.fetchall()]


def count_labeled_examples() -> int:
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) as c FROM recognition_logs WHERE human_label IS NOT NULL")
        return cur.fetchone()["c"]


def get_accept_timestamps_for_session(user_id: int, session_date: str):
    """
    Every AUTHORITATIVE auto_accept timestamp (final_decision, not the raw
    rule-engine decision) for this user on this day, used by presence.py to
    bucket the session into windows and compute presence CONSISTENCY. Using
    final_decision keeps this correct even after the ML Trust Engine has
    been promoted to authoritative (see config.ML_ENGINE_PROMOTE_WHEN_READY).
    """
    with db_cursor() as cur:
        cur.execute("""
            SELECT timestamp FROM recognition_logs
            WHERE user_id = ? AND final_decision = 'auto_accept' AND date(timestamp) = ?
            ORDER BY timestamp ASC
        """, (user_id, session_date))
        return [row["timestamp"] for row in cur.fetchall()]


def get_recent_logs(limit=50):
    with db_cursor() as cur:
        cur.execute("""
            SELECT rl.*, u.name, u.external_id
            FROM recognition_logs rl
            LEFT JOIN users u ON rl.user_id = u.id
            ORDER BY rl.timestamp DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]


def get_recent_similarities_for_user(user_id: int, limit=10):
    """Used by the Trust Engine's 'historical stability' sub-score."""
    with db_cursor() as cur:
        cur.execute("""
            SELECT similarity FROM recognition_logs
            WHERE user_id = ? AND similarity IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (user_id, limit))
        return [row["similarity"] for row in cur.fetchall()]


# -----------------------------------------------------------------------
# ATTENDANCE / PRESENCE SESSIONS
# -----------------------------------------------------------------------
def get_today_session(user_id: int, session_date: str):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM attendance_sessions WHERE user_id = ? AND session_date = ?",
            (user_id, session_date),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def upsert_session(user_id, session_date, entry_time, last_seen,
                    presence_seconds, ping_count, avg_trust_score, avg_confidence):
    with db_cursor() as cur:
        existing = cur.execute(
            "SELECT id FROM attendance_sessions WHERE user_id = ? AND session_date = ?",
            (user_id, session_date),
        ).fetchone()

        if existing:
            cur.execute("""
                UPDATE attendance_sessions
                SET last_seen = ?, presence_seconds = ?, ping_count = ?,
                    avg_trust_score = ?, avg_confidence = ?
                WHERE user_id = ? AND session_date = ?
            """, (last_seen, presence_seconds, ping_count, avg_trust_score,
                  avg_confidence, user_id, session_date))
        else:
            cur.execute("""
                INSERT INTO attendance_sessions
                    (user_id, session_date, entry_time, last_seen, presence_seconds,
                     ping_count, avg_trust_score, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, session_date, entry_time, last_seen, presence_seconds,
                  ping_count, avg_trust_score, avg_confidence))


def get_today_attendance(session_date: str):
    with db_cursor() as cur:
        cur.execute("""
            SELECT a.*, u.name, u.external_id
            FROM attendance_sessions a
            JOIN users u ON a.user_id = u.id
            WHERE a.session_date = ?
            ORDER BY a.entry_time ASC
        """, (session_date,))
        return [dict(row) for row in cur.fetchall()]


def get_all_attendance_history(limit=200):
    with db_cursor() as cur:
        cur.execute("""
            SELECT a.*, u.name, u.external_id
            FROM attendance_sessions a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.session_date DESC, a.entry_time ASC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]
