# TrustFace AI — Complete Code Walkthrough

This walks through every file's actual logic, block by block. Read this
alongside the source — the source has the "why" comments, this has the
"what happens, in order" narrative. Backend first (where nearly all the
logic lives), then frontend.

---

## `backend/config.py` — the settings every other file imports

No logic here, just constants, but every constant matters:

- **Paths** (`BASE_DIR`, `DATABASE_PATH`, `KEY_FILE_PATH`, `MODELS_DIR`, `ASSETS_DIR`): computed relative to the file's own location (`os.path.dirname(os.path.dirname(...))`) so the app works no matter where it's cloned to. The `os.makedirs(..., exist_ok=True)` calls at the bottom of this section guarantee those folders exist before anything tries to write into them.
- **`INSIGHTFACE_MODEL_NAME = "buffalo_l"`**, **`INSIGHTFACE_CTX_ID = -1`** (CPU), **`INSIGHTFACE_DET_SIZE = (640, 640)`**: passed straight into `face_engine.get_face_app()`.
- **`FACE_MATCH_THRESHOLD = 0.45`**: the cosine similarity cutoff. Read `docs/THRESHOLDS.md` for how to tune it empirically.
- **`TRUST_WEIGHTS`**: a dict, must sum to 1.0, read directly by `trust_engine.compute_trust_score()`.
- **`TRUST_AUTO_ACCEPT = 85`, `TRUST_RETRY = 60`**: the two cut points that turn a 0–100 score into a three-way decision.
- **`SESSION_START_HOUR`/`SESSION_END_HOUR`**: define the 8-hour window `presence.expected_session_seconds()` divides duration by.
- **`PRESENCE_TIMEOUT_SECONDS = 120`**: the gap threshold `presence.record_presence_ping()` uses to decide "still the same visit" vs. "re-entry."
- **`ML_MODEL_PATH`, `ML_METRICS_PATH`, `MIN_TRAINING_SAMPLES = 30`, `ML_TEST_SIZE = 0.25`, `ML_RANDOM_STATE = 42`, `ML_ENGINE_PROMOTE_WHEN_READY = True`**: everything `ml_trust_engine.py` and `recognition_service.py`'s promotion logic need.
- **`ANTI_SPOOF_MODEL_PATH`, `ANTI_SPOOF_INPUT_SIZE`**: where `spoof_model.py` looks for a real pretrained model.
- **`PRESENCE_BUCKET_MINUTES = 5`** plus the three `PRESENCE_CATEGORY_*` constants: feed `presence.compute_presence_consistency()` and `presence.classify_session()`.
- **`SAVE_REGISTRATION_IMAGE`**: currently unused as a gate in code (documented as a future hook) — a flag you'd check in `recognition_service.register_face()` if you wanted to persist demo images.

---

## `backend/database.py` — schema + every query

**`get_connection()`**: opens a raw `sqlite3.connect()`, sets `row_factory = sqlite3.Row` (so rows behave like dicts — `row["name"]` instead of `row[0]`), and turns on `PRAGMA foreign_keys = ON` (SQLite has foreign keys off by default; this makes `ON DELETE CASCADE`/`SET NULL` actually work).

**`db_cursor()`**: a `@contextmanager` — every caller does `with db_cursor() as cur:`, and the connection auto-commits and auto-closes when the block exits. This is why no other function in the file manually calls `.commit()` or `.close()`.

**`init_db()`**: runs on every app startup (called once in `main.py`). Three `CREATE TABLE IF NOT EXISTS` statements:
- `users`: `id, name, external_id (UNIQUE), embedding_enc (BLOB), created_at`.
- `attendance_sessions`: one row per `(user_id, session_date)` — enforced by `UNIQUE(user_id, session_date)` — with `entry_time, last_seen, presence_seconds, ping_count, avg_trust_score, avg_confidence`.
- `recognition_logs`: every column a single recognition attempt produces — `similarity, trust_score, decision` (rule-based), the five quality sub-scores, `human_label`, `ml_decision`, `final_decision`.

Then the **migration block**: `PRAGMA table_info(recognition_logs)` lists existing columns; if `human_label`/`ml_decision`/`final_decision` are missing (an old database from before these existed), `ALTER TABLE ... ADD COLUMN` adds them — and for `final_decision` specifically, a one-time `UPDATE ... SET final_decision = decision WHERE final_decision IS NULL` backfills old rows correctly (their rule-based decision *was* authoritative at the time, since ML promotion didn't exist yet).

**User CRUD**: `create_user()` inserts and returns `cur.lastrowid`. `get_all_users()` excludes the embedding (for the Users page list). `get_all_users_with_embeddings()` includes it (for building the recognition gallery). `get_user_by_id()`, `delete_user()`, `external_id_exists()` are straightforward single-purpose queries.

**Recognition logs**: `insert_recognition_log()` takes the user, similarity, trust score, rule decision, the `quality` dict (pulling out `blur`/`brightness`/`pose`/`face_size`/`spoof` by key), plus optional `ml_decision` and `final_decision`. Note the line `final_decision or decision` — if the caller doesn't pass one explicitly, it defaults to the rule decision. `get_recent_logs()` does a `LEFT JOIN` to `users` so unknown-face rows (where `user_id IS NULL`) still return a row with `name = NULL` rather than being excluded. `get_recent_similarities_for_user()` powers the Trust Engine's historical-stability bonus. `set_human_label()` and `get_labeled_training_data()` and `count_labeled_examples()` are the three functions the ML pipeline runs on. `get_accept_timestamps_for_session()` filters specifically on `final_decision = 'auto_accept'` (not the raw rule `decision`) — this is the fix that keeps presence consistency correct after ML promotion.

**Attendance/presence**: `get_today_session()` looks up the one row for `(user_id, today)`. `upsert_session()` branches on whether that row exists — `UPDATE` if so, `INSERT` if not — this is what makes `presence.record_presence_ping()` idempotent-safe to call on every single recognition. `get_today_attendance()` and `get_all_attendance_history()` both `JOIN` to `users` for display and are what `presence.py`'s enrichment functions build on.

---

## `backend/privacy.py` — AES-256-GCM, 79 lines, every line matters

`_load_or_create_key()`: checks if `database/secret.key` exists; if yes, reads and returns its 32 bytes. If not, calls `get_random_bytes(32)` (a cryptographically secure random 256-bit key) and writes it to disk. This runs **once**, at import time (`_KEY = _load_or_create_key()` at module level) — every subsequent encrypt/decrypt call in the process reuses the same in-memory key.

`encrypt_embedding(embedding)`:
1. `embedding.astype(np.float32).tobytes()` — serializes the 512-float numpy array into raw bytes (2048 bytes: 512 × 4 bytes/float32).
2. `AES.new(_KEY, AES.MODE_GCM, nonce=get_random_bytes(12))` — a fresh random nonce every single call, critical for GCM's security guarantee.
3. `cipher.encrypt_and_digest(raw_bytes)` returns `(ciphertext, tag)` — the tag is the 16-byte authentication code.
4. Returns `cipher.nonce + tag + ciphertext` concatenated — one blob, laid out as `[12 bytes nonce][16 bytes tag][remaining bytes ciphertext]`.

`decrypt_embedding(blob)` reverses it exactly: slices out the first 12 bytes as nonce, next 16 as tag, rest as ciphertext; `AES.new(_KEY, AES.MODE_GCM, nonce=nonce)` then `cipher.decrypt_and_verify(ciphertext, tag)` — this call raises an exception if the tag doesn't match (tamper detection), and returns the raw bytes on success, which `np.frombuffer(raw_bytes, dtype=np.float32)` turns back into the original 512-float vector.

---

## `backend/face_engine.py` — InsightFace wrapper + quality metrics

`get_face_app()`: a lazy singleton. `_face_app` starts as `None`; the first call constructs `FaceAnalysis(name="buffalo_l")` and calls `.prepare(ctx_id=-1, det_size=(640,640))` (this is the slow step — loading ONNX weights into memory, downloading them the very first time). Every later call just returns the cached `_face_app`.

`detect_faces(image_bgr)`: one line, `app.get(image_bgr)` — returns a list of InsightFace `Face` objects, each with `.bbox`, `.kps` (5 landmarks), `.embedding` (512-d, already computed), `.det_score`.

`cosine_similarity(a, b)`: casts both to `float32`, computes `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`, guards against division by zero if either norm is 0.

The four quality functions all follow the same pattern — compute a raw signal, map it onto 0–100:
- `compute_blur_score`: `cv2.Laplacian(gray, cv2.CV_64F).var()`, then `min(100, (variance/150)*100)`. The `/150` is an empirically chosen "sharp enough" reference point.
- `compute_brightness_score`: `mean_val = np.mean(gray)`, then `100 - abs(mean_val - 130)/130 * 100` — a symmetric penalty around a 130/255 midpoint, clamped at 0.
- `compute_pose_score`: pulls `left_eye, right_eye, nose` from the 5 landmarks, computes `eye_mid_x`, the offset of `nose[0]` from that midpoint, normalizes by `eye_dist`, then `100 - offset_ratio*200` clamped at 0.
- `compute_face_size_score`: `face_area / frame_area`; a piecewise function — scales up linearly below 2% (too small), scales down above 60% (too close/occluding), flat 100 in between.

`estimate_spoof_risk(face_crop_bgr)`: resizes to 128×128 grayscale, runs `np.fft.fft2` then `fftshift`, builds a circular mask around the frequency-domain center (`radius = min(h,w)//6`), sums magnitude inside vs. total, computes `high_freq_ratio = 1 - (low_freq_energy/total_energy)`, then linearly remaps the range `[0.15, 0.55]` onto `[0, 100]` via `np.clip`. This is the pure-heuristic fallback — `spoof_model.py` calls this function by name when no ONNX model is present.

`compute_quality_metrics(image_bgr, face)`: crops the face from the bbox (clamped to image bounds), converts to grayscale, calls the four functions above plus `spoof_model.get_spoof_score(crop)` (imported inside the function to avoid a circular import at module load time, since `spoof_model.py` imports `face_engine` too), and returns one dict with all five scores plus `spoof_source` (`"model"` or `"heuristic"`).

---

## `backend/spoof_model.py` — the pluggable anti-spoof slot

`_try_load_session()`: checks `os.path.exists(ANTI_SPOOF_MODEL_PATH)` — if the file isn't there, returns `None` immediately (no error, no attempt). If it is there, tries `onnxruntime.InferenceSession(...)`; wraps the whole thing in `try/except` so a corrupt/incompatible model file degrades to the heuristic instead of crashing the app. `_session_checked` is a module-level flag so this only ever runs once per process, not once per frame.

`_run_model(face_crop_bgr, session)`: resizes to `ANTI_SPOOF_INPUT_SIZE` (80×80), converts BGR→RGB and normalizes to `[0,1]`, transposes `HWC→CHW` (the layout ONNX/PyTorch models expect), adds a batch dimension. Runs `session.run(...)`, applies a manual softmax to the raw output, and reads the **last** class as "real" (documented convention for this model family, whether it's a 2-class or 3-class output). Returns a 0–100 score.

`get_spoof_score(face_crop_bgr)`: the actual entry point `face_engine.py` calls. If a session loaded successfully, tries `_run_model()` — any inference-time exception falls back silently. Otherwise (or on failure), calls `face_engine.estimate_spoof_risk()`. Always returns `{"score": ..., "source": "model"|"heuristic"}`.

---

## `backend/trust_engine.py` — the rule-based formula

`similarity_to_score(similarity)`: piecewise-linear remap anchored at the configured threshold — below threshold, linearly stretches `[0, threshold] → [0, 50]`; at/above threshold, stretches `[threshold, 1.0] → [50, 100]`. This is why a similarity exactly at `FACE_MATCH_THRESHOLD` always scores ~50 regardless of what the threshold's numeric value is.

`historical_stability_bonus(recent_similarities)`: needs at least 3 prior values (returns 0.0 otherwise). Computes `coefficient_of_variation = std/mean`; below 0.05 → `+5`, below 0.15 → `+2`, above 0.35 → `-5`, otherwise `0`.

`compute_trust_score(similarity, quality, recent_similarities)`: builds a `sub_scores` dict (`similarity_to_score(similarity)` plus the four quality values passed through as-is — they're already 0–100), computes `base_score = sum(TRUST_WEIGHTS[k] * sub_scores[k] for k in TRUST_WEIGHTS)`, adds the stability bonus, clips to `[0, 100]` via `np.clip`, then maps the final number to `auto_accept`/`retry`/`reject` against `TRUST_AUTO_ACCEPT`/`TRUST_RETRY`. Returns everything — final score, decision, every sub-score, the bonus, and the weights used — so the frontend can render the full breakdown, not just a number.

---

## `backend/ml_trust_engine.py` — the trained classifier

`_build_feature_matrix(rows)`: turns DB rows into `(X, y)` numpy arrays. Skips any row with a `None` in any of the six feature columns. Multiplies `similarity` by 100 (it's stored 0–1, everything else is already 0–100) so magnitudes are comparable.

`training_status()`: returns `labeled_examples` (via `db.count_labeled_examples()`), `min_required`, `ready_to_train` (a simple `>=` comparison), and whether a model file already exists on disk.

`train_model(model_type)`:
1. Pulls all labeled rows, builds `X, y`.
2. If `len(X) < MIN_TRAINING_SAMPLES`, returns `{"success": False, "error": ...}` immediately — no training attempted.
3. `train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)` — stratify keeps class proportions similar in both splits.
4. Builds either a `DecisionTreeClassifier(max_depth=4)` or `RandomForestClassifier(n_estimators=100, max_depth=6)` based on the `model_type` argument.
5. `model.fit(X_train, y_train)`.
6. Predicts on `X_test`, computes `accuracy_score`, `precision_recall_fscore_support` (per class), `confusion_matrix` — all only on the held-out test set, never train.
7. If the model has `feature_importances_` (Random Forest does, Decision Tree does too actually — but the docstring notes it's most interpretable printed as text via `export_text` for a single tree), builds a dict mapping feature name → importance.
8. Saves the model with `joblib.dump()` and the metrics dict as JSON, returns `{"success": True, "metrics": {...}}`.

`_load_model()` / `model_available()`: a module-level `_cached_model` singleton, same pattern as `face_engine.get_face_app()` — load once, reuse.

`predict_decision(similarity, quality)`: builds the same 6-feature vector, returns `(None, None)` if no model is loaded (the cold-start signal `recognition_service.py` checks for), otherwise `model.predict()` plus `model.predict_proba()` (if available) for a confidence number.

`explain_tree_rules()`: only meaningful if the currently-saved model is a single `DecisionTreeClassifier` (`hasattr(model, "tree_")`); calls `sklearn.tree.export_text()` to produce the actual if/else rules as a string.

---

## `backend/presence.py` — duration, consistency, categories

`record_presence_ping(user_id, trust_score, similarity)`: called only when `decision == "auto_accept"`. Looks up today's session via `db.get_today_session()`. If none exists yet, creates one with `ping_count=1`, `presence_seconds=0`. If one exists: computes `gap_seconds = (now - last_seen).total_seconds()`; if that gap is `<= PRESENCE_TIMEOUT_SECONDS`, adds it to `presence_seconds` (continuous presence); if larger, doesn't add it (treated as a re-entry — the gap itself isn't counted, but a new `last_seen` still gets recorded). Updates `avg_trust_score`/`avg_confidence` as running averages: `(old_avg * old_count + new_value) / new_count`.

`compute_presence_consistency(user_id, session_date, entry_time, last_seen)`: pulls every `final_decision='auto_accept'` timestamp for that user/day via `db.get_accept_timestamps_for_session()`. Computes `start = min(entry_time, min(timestamps))` and `end = max(last_seen, max(timestamps))` — the min/max widening is specifically there to handle the sub-second gap between a recognition log's timestamp and the session update it triggers (see the "bugs I found while testing" note below). Divides `[start, end]` into `PRESENCE_BUCKET_MINUTES`-sized windows, checks each bucket for at least one timestamp inside it, returns the consistency percentage plus the raw `timeline` list (`"detected"`/`"missing"` per bucket).

`classify_session(presence_percentage, consistency)`: checked in strict priority order — brief appearance (presence % below the floor) beats everything else; then continuous (consistency at/above the bar); then it counts `detected↔missing` transitions in the timeline and computes a `transition_rate`; a high rate with moderate/low consistency → "Suspicious Intermittent Presence"; otherwise moderate consistency → "Frequent Exits"; anything left over → "Interrupted Presence".

`enrich_session_row(row)`: the function that turns a raw DB row into everything the dashboard displays — computes `presence_percentage`, formats duration as `"Xh Ym"`, formats entry/last-seen as `HH:MM:SS`, calls `compute_presence_consistency()` and `classify_session()`, attaches all of it to the row dict. `get_today_attendance_enriched()` and `get_history_enriched()` are just this applied to every row from the corresponding `database.py` query.

---

## `backend/recognition_service.py` — the orchestrator

`decode_base64_image(base64_str)`: strips the `data:image/jpeg;base64,` prefix if present, `base64.b64decode()`, `np.frombuffer(..., dtype=np.uint8)`, `cv2.imdecode(..., cv2.IMREAD_COLOR)` — turns a browser's data URL into an OpenCV BGR array.

`_load_gallery()`: pulls every user's encrypted embedding via `db.get_all_users_with_embeddings()`, decrypts each **in memory** with `privacy.decrypt_embedding()`, wraps decryption in `try/except` per-user (so one corrupted row doesn't crash the whole gallery load), returns a list of `{"id", "name", "external_id", "embedding"}` dicts. This runs on **every single recognition call** — a deliberate simplicity/performance tradeoff documented as acceptable at tens/hundreds of users, and the exact place a FAISS index would replace it at scale.

`find_best_match(embedding, gallery)`: a plain linear scan — computes `face_engine.cosine_similarity()` against every gallery entry, tracks the best.

`recognize_frame(image_bgr)` — the main pipeline, one iteration per detected face:
1. `quality = face_engine.compute_quality_metrics(...)`.
2. `best_user, similarity = find_best_match(...)`.
3. `is_match = best_user is not None and similarity >= FACE_MATCH_THRESHOLD`.
4. **If matched**: pulls `recent_sims` for that user, computes the rule-based `trust_result` via `trust_engine.compute_trust_score()`.
5. **If not matched**: still computes a trust breakdown (for transparency/logging) but hard-overrides `trust_result["decision"] = "unknown"`, `user_id = None`.
6. `ml_decision, ml_confidence = ml_trust_engine.predict_decision(...)` — `(None, None)` if no model trained yet.
7. **Authoritative decision logic**: if `is_match` and `ml_decision is not None` and `ML_ENGINE_PROMOTE_WHEN_READY`, the authoritative decision is the ML one — with a vocabulary translation (`"accept"` from the human-label vocabulary → `"auto_accept"` to match the rule engine's vocabulary, since `"retry"`/`"reject"` already match). Otherwise, the rule-based decision is authoritative.
8. `db.insert_recognition_log(...)` — always runs, logs the rule decision, the raw ML decision, and the authoritative `final_decision` all separately.
9. `if is_match and authoritative_decision == "auto_accept": presence.record_presence_ping(...)` — presence only updates off the authoritative decision, never the raw rule one directly.
10. Appends a result dict per face with everything the frontend needs — bbox, names, similarity, trust score, sub-scores, weights, both raw decisions, the authoritative one, and its source (`"rule"` or `"ml"`).

`register_face(image_bgr, name, external_id)`: checks `db.external_id_exists()` first (fails fast on duplicates). Runs detection; fails with a clear message if zero or more-than-one face is found. Computes quality metrics and applies two registration-time gates (`blur < 25` or `face_size < 20` → reject with a specific message — better data in means better matches out). On success, encrypts the embedding and calls `db.create_user()`.

---

## `backend/main.py` — the HTTP layer

Sets up the FastAPI app, mounts `/static`, configures `Jinja2Templates`, calls `db.init_db()` once at import time. Three Pydantic models (`RegisterRequest`, `RecognizeRequest`, `LabelRequest`, `TrainRequest`) give automatic request validation — malformed JSON gets rejected before any handler code runs.

**Page routes** (`/`, `/register`, `/attendance`, `/users`, `/logs`) are thin — they call a `database.py`/`presence.py` function for any server-rendered data (the Users page still renders its table server-side; the Logs page is now fully JS-driven so its route just returns the template shell) and hand off to `TemplateResponse`.

**API routes** are equally thin — each one calls exactly one function from `recognition_service`, `database`, `presence`, or `ml_trust_engine` and wraps the result in `JSONResponse`. The only routes with real branching are `api_register`/`api_recognize` (decode the image, check for `None` on decode failure) and `api_train_model`/`api_label_log` (validate the label value, pick the right HTTP status code based on `success`).

---

## Frontend — templates (`frontend/templates/*.html`)

All five content pages `{% extends "base.html" %}` and only fill the `content` and `extra_scripts` blocks — `base.html` owns the sidebar, nav highlighting (`{% if active == '...' %}`), the Google Fonts link, and the shared pipeline-footer label. `register.html` and `attendance.html` both have a `<video>` + overlay `<canvas>` + hidden capture `<canvas>` structure (webcam preview, drawing surface, off-screen frame grabber, respectively). `users.html` is the one page still doing server-side Jinja loops (`{% for u in users %}`) since it doesn't need live updates. `dashboard.html` and `logs.html` are just empty containers with `id`s — all their content is injected by JS on load.

## Frontend — JS (`frontend/static/js/*.js`)

`register.js`: `startCamera()` calls `navigator.mediaDevices.getUserMedia()` and pipes the stream into the `<video>` element. `btnCapture` draws the current video frame onto the hidden canvas and calls `.toDataURL()`. `checkSubmitEnabled()` gates the submit button on having both a captured image and non-empty name/ID fields. `btnSubmit`'s handler `POST`s to `/api/register` and shows a success/error flash.

`attendance.js`: `startCamera()` same pattern, but also sizes the overlay `<canvas>` to match the video's actual resolution once metadata loads. `captureAndRecognize()` runs on a `setInterval(1500ms)` — grabs a frame, `POST`s to `/api/recognize`, and on response calls `drawBoxes()` (draws a colored rectangle + label per detected face, color keyed to decision) and `renderResultPanel()` (updates the SVG gauge's `stroke-dashoffset` based on trust score, fills in the sub-score bars, shows rule vs. ML decision).

`dashboard.js`: `loadDashboard()` fetches `/api/dashboard-summary` and calls three render functions — `renderAttendanceTable()` (includes the presence category badge), `renderPresenceChart()` (a Plotly horizontal bar chart), `renderRecentLogs()`. Runs once on load, then every 5 seconds via `setInterval`.

`logs.js`: `loadMlStatus()` fetches `/api/ml-status` and updates the four stat cards plus enables/disables the Train button based on `ready_to_train`. `renderMetricsDetail()` builds the per-class metrics table and the feature-importance bars after a successful training run. `labelLog()` posts a label and immediately calls `loadMlStatus()` again (so the "labeled examples" counter updates live). `loadLogs()` builds the full table with three decision columns (rule/ML/final) plus the Accept/Retry/Reject buttons for unlabeled rows.

`users.js`: one function, `deleteUser()` — confirms, `DELETE`s, removes the table row on success.

## Frontend — `style.css`

Defines every color/font/spacing as a CSS custom property under `:root` (`--accent`, `--bg-panel`, `--font-mono`, etc.) — every other rule references these variables rather than hardcoding colors, which is why the whole app's theme could be swapped by editing one block. The rest is organized by component: layout shell → sidebar/nav → page headers → cards/stat cards → buttons/inputs → the webcam panel → tables → badges → the trust gauge → sub-score bars → misc utility classes, roughly matching the order components appear on the page.

---

## A note on process, not just code

Two real bugs surfaced during testing of the ML-promotion feature, worth knowing for your viva as evidence of your development process:

1. **Vocabulary mismatch**: the ML model's labels (`"accept"`) didn't match the rule engine's vocabulary (`"auto_accept"`), which silently broke presence logging the moment a model got promoted. Fixed with an explicit translation at the one place `recognition_service.py` computes the authoritative decision.
2. **Timestamp boundary edge case**: `insert_recognition_log()`'s timestamp is written a few milliseconds *before* `presence.record_presence_ping()` updates `entry_time`, so the very ping that creates a session could fall a hair outside its own bucket window. Fixed by widening `compute_presence_consistency()`'s window with `min()`/`max()` against the actual timestamps, not just the session's recorded entry/last-seen.

Both were caught by writing and running real end-to-end tests before shipping, not by inspection — a good habit to describe if asked how you validated the system.
