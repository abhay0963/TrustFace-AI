# Architecture

## Request Flow (Live Attendance)

```
Browser (webcam)
   │  captures frame every 1.5s, JPEG → base64
   ▼
POST /api/recognize  (FastAPI, main.py)
   │
   ▼
recognition_service.recognize_frame()
   │
   ├─► face_engine.detect_faces()        [InsightFace: detect + align + embed]
   ├─► face_engine.compute_quality_metrics()  [blur, brightness, pose, size, spoof]
   ├─► privacy.decrypt_embedding()        [per registered user, in-memory only]
   ├─► face_engine.cosine_similarity()     [find best match in gallery]
   ├─► trust_engine.compute_trust_score()   [weighted sum → decision]
   ├─► database.insert_recognition_log()     [always logged, for audit]
   └─► presence.record_presence_ping()        [only if decision == auto_accept]
   │
   ▼
JSON response → browser renders bounding boxes + Trust Gauge
```

## Data Model

```
users
 ├─ id (PK)
 ├─ name
 ├─ external_id (unique - roll no / employee id)
 ├─ embedding_enc (BLOB - AES-256-GCM encrypted 512-d vector)
 └─ created_at

attendance_sessions   (one row per user per day)
 ├─ user_id (FK)
 ├─ session_date
 ├─ entry_time / last_seen
 ├─ presence_seconds / ping_count
 └─ avg_trust_score / avg_confidence

recognition_logs   (one row per recognition attempt, ALWAYS written)
 ├─ user_id (FK, nullable for unknown faces)
 ├─ timestamp
 ├─ similarity / trust_score / decision
 └─ blur_score / brightness_score / pose_score / face_size_score / spoof_score
```

## Why This Design

- **Separation of concerns**: each module (`face_engine`, `privacy`,
  `trust_engine`, `ml_trust_engine`, `spoof_model`, `presence`, `database`)
  has exactly one job and can be explained, tested, and modified
  independently — good for a viva walkthrough.
- **No black boxes**: the rule-based Trust Engine is a documented weighted
  sum. The ML Trust Engine IS a trained model, but every number it produces
  (accuracy, confusion matrix, feature importances) is computed on a
  held-out test split and saved to disk — also fully traceable.
- **Bootstrap-then-learn**: the system is usable from minute one on the
  rule-based engine, and gains a genuinely trained decision layer once
  enough human-labeled data accumulates — a standard cold-start pattern in
  real ML systems, not a workaround unique to this project.
- **Encrypt-then-store**: embeddings are only ever decrypted in-memory,
  never persisted unencrypted.
- **Everything logged**: even rejected/unknown attempts are recorded, which
  gives you real data to compute False Acceptance/Rejection Rates and to
  train the ML Trust Engine.

## Module 3b: ML Trust Engine — data flow

```
Logs page: reviewer clicks Accept/Retry/Reject on a logged attempt
   │
   ▼
database.set_human_label()  →  recognition_logs.human_label
   │
   ▼ (once ≥ MIN_TRAINING_SAMPLES rows are labeled)
ml_trust_engine.train_model()
   ├─► train/test split (75/25, stratified)
   ├─► fit DecisionTreeClassifier or RandomForestClassifier
   ├─► evaluate on held-out test split → accuracy, precision, recall, F1, confusion matrix
   └─► save model (joblib) + metrics (JSON) to models/
   │
   ▼
recognition_service.recognize_frame() calls ml_trust_engine.predict_decision()
   on every future frame, alongside the rule-based decision (shown side-by-side in the UI)
```

## Module 4: Presence Intelligence — duration vs. consistency

```
raw pings (auto_accept recognitions) → attendance_sessions (entry/last_seen/presence_seconds)
                                      → presence.compute_presence_consistency()
                                            bucket [entry_time, last_seen] into 5-min windows
                                            → % of windows with ≥1 detection = consistency
```
Duration alone can't distinguish "here the whole time" from "here at the
start and end only" — consistency can.
