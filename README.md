# TrustFace AI

**Privacy-Preserving Intelligent Presence Verification System**

> Traditional face attendance systems answer *"who is this?"*. TrustFace AI asks
> the harder question: *how certain am I, and should I actually trust this
> recognition enough to act on it?*

Built as an MCA project, engineered to industry standard: explainable decision
scoring, encrypted biometric storage, and presence intelligence — running
entirely on a single laptop with open-source, open-weight tools.

---

## The Core Idea

```
Who is this?  →  How certain am I?  →  Can I trust this?  →  Accept / Retry / Reject  →  How long have they been present?
   (ArcFace)      (cosine similarity)     (Trust Engine)         (decision policy)          (Presence Intelligence)
```

Attendance is only *one* application of this pipeline — the same trust-scored
identity verification generalizes to access control, exam proctoring, or any
scenario where blind acceptance of an AI prediction is a real risk.

## Why This Is Different From a Normal Attendance System

| Typical face attendance system | TrustFace AI |
|---|---|
| Stores raw face photos | Stores only AES-256-GCM **encrypted embeddings** |
| Binary "Matched / Not Matched" | Graded **Trust Score** (0–100) with a full breakdown |
| Blindly accepts every match | **Auto Accept / Retry / Reject** decision policy |
| One-time attendance mark | **Presence Intelligence**: entry time, last seen, presence % |
| Black-box confidence | Every sub-score is a documented, explainable formula |

## Modules

1. **Face Detection & Recognition** — InsightFace (ArcFace/RetinaFace), 512-d embeddings, cosine similarity
2. **Privacy Layer** — AES-256-GCM encrypted embeddings, no raw biometric storage
3. **Trust Engine (rule-based)** — weighted, explainable scoring: similarity, blur, brightness, pose, face size, spoof, historical stability
   **3b. Trust Engine (ML)** — a Decision Tree / Random Forest trained on your own human-labeled recognition attempts, with real accuracy/precision/recall/confusion-matrix/feature-importance numbers. Falls back to the rule-based engine until enough labels exist (cold-start pattern)
4. **Presence Intelligence** — entry/last-seen/duration/**presence %** AND **presence consistency %** (timeline-bucketed: continuous vs. interrupted presence, not just raw duration)
5. **Dashboard** — live stats, presence charts, recognition audit log with inline human-labeling controls
6. **Anti-spoofing (pluggable)** — an FFT texture heuristic by default, with a documented drop-in slot for a real pretrained ONNX anti-spoof model
7. **Learning Lab** — five standalone `.py` scripts (no Jupyter needed) covering embeddings, cosine similarity, FAR/FRR/ROC evaluation, PCA/t-SNE visualization, and the full supervised ML training pipeline — see `learning_lab/README.md`

## Tech Stack

Python · FastAPI · InsightFace (ONNX Runtime) · OpenCV · SQLite · PyCryptodome (AES) · scikit-learn (Decision Tree / Random Forest) · matplotlib · Jinja2 · custom CSS · Vanilla JS · Plotly

No paid APIs. No cloud GPUs. Runs fully offline after the one-time model download.

---

## Quick Start (Local, VS Code)

### 1. Prerequisites
- Python 3.9 – 3.11 (InsightFace/ONNX Runtime wheels are most reliable on these versions)
- A webcam
- ~500MB free disk space (for the InsightFace model pack)

### 2. Clone / open the project in VS Code
Open the `TrustFace-AI/` folder in VS Code.

### 3. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```
> If `insightface` fails to build on Windows, install the "Microsoft C++ Build Tools"
> (or simply `pip install insightface --only-binary :all:`), then re-run.

### 5. Run the app
```bash
python run.py
```
The **first run** will automatically download the `buffalo_l` InsightFace model
pack (~350MB) into `~/.insightface/models/` — this needs internet access once.
After that, everything runs fully offline.

### 6. Open the app
Go to **http://127.0.0.1:8000** in your browser. Allow webcam access when prompted.

---

## How To Use It

1. **Register** (`/register`) — enter a name + ID, capture a clear frontal frame.
   The embedding is computed, AES-encrypted, and stored. The raw image is never
   saved to the database.
2. **Live Attendance** (`/attendance`) — start the camera; every 1.5s a frame is
   analyzed. Watch the live Trust Score gauge and sub-score breakdown. Only
   `auto_accept` decisions are logged as attendance.
3. **Dashboard** (`/`) — see who's present today, presence %, presence
   **consistency** %, average trust, and recent recognition events.
4. **Manage Users** (`/users`) — view/delete registered identities.
5. **Recognition Logs** (`/logs`) — full audit trail of every recognition
   attempt. **Label attempts** (Accept/Retry/Reject) to build ground-truth
   training data, then click **Train Model** once you have ≥30 labels to
   train the ML Trust Engine and see real accuracy/precision/recall/
   confusion-matrix/feature-importance numbers.
6. **Learning Lab** (`learning_lab/`) — run the numbered scripts from a
   terminal to go deeper on each AI concept with real data from your own
   database. See `learning_lab/README.md`.

### The two Trust Engines, and why both exist

- **Rule-based** (`backend/trust_engine.py`) — a weighted sum, works from
  the very first recognition with zero data. This is what powers the app
  by default.
- **ML-based** (`backend/ml_trust_engine.py`) — a Decision Tree / Random
  Forest trained on YOUR human-labeled logs. Cold-start problem: it needs
  `MIN_TRAINING_SAMPLES` (default 30) labeled examples before it can train
  at all. Once trained, every future recognition shows BOTH the rule-based
  decision and the ML model's prediction side-by-side (in the UI and in
  `recognition_logs.ml_decision`), so you can directly compare them.

---

## Project Structure

```
TrustFace-AI/
├── backend/
│   ├── main.py                 # FastAPI app & routes
│   ├── config.py                # All tunable constants, explained
│   ├── database.py               # SQLite schema + CRUD
│   ├── privacy.py                 # AES-256-GCM embedding encryption
│   ├── face_engine.py              # InsightFace wrapper + quality metrics
│   ├── trust_engine.py              # Rule-based explainable trust score
│   ├── ml_trust_engine.py            # ML Trust Engine: train/predict/evaluate
│   ├── spoof_model.py                 # Pluggable anti-spoof ONNX model + fallback
│   ├── presence.py                     # Presence intelligence + consistency
│   └── recognition_service.py           # End-to-end pipeline orchestration
├── frontend/
│   ├── templates/                # Jinja2 HTML pages
│   └── static/{css,js}            # Styling + browser logic (webcam, charts, labeling UI)
├── learning_lab/                    # Standalone .py scripts (no Jupyter) - see its README
│   ├── 01_embeddings_and_similarity.py
│   ├── 02_threshold_evaluation.py
│   ├── 03_embedding_visualization.py
│   ├── 04_train_trust_classifier.py
│   ├── 05_presence_consistency_demo.py
│   └── output/                      # Generated plots land here
├── experiments/                       # Reproducible comparisons/ablations for your paper - see its README
│   ├── 01_threshold_comparison.py
│   ├── 02_blur_brightness_ablation.py
│   ├── 03_trust_score_feature_ablation.py
│   ├── 04_decision_tree_depth_comparison.py
│   ├── 05_synthetic_data_augmentation.py
│   └── output/
├── database/                       # trustface.db + secret.key (generated, gitignored)
├── models/                          # InsightFace cache + trained ML model (gitignored)
├── assets/unknown_faces/             # optional demo image retention
├── docs/                              # ARCHITECTURE.md, THRESHOLDS.md, MODEL_CARD_ARCFACE.md
├── paper/                              # research paper scaffold
├── requirements.txt
├── run.py
└── README.md
```

## Tuning the System

All thresholds and weights live in `backend/config.py`:
- `FACE_MATCH_THRESHOLD` — cosine similarity cutoff for "same person"
- `TRUST_WEIGHTS` — how much each quality signal contributes to the trust score
- `TRUST_AUTO_ACCEPT` / `TRUST_RETRY` — decision boundaries

See `docs/THRESHOLDS.md` for the methodology to tune these against your own
collected data (relevant for your paper's Evaluation section).

## Security & Privacy Notes

- `database/secret.key` (AES key) and `database/trustface.db` are **gitignored**
  — never commit them.
- Losing `secret.key` makes all stored embeddings permanently unreadable
  (correct security behavior — this is why it's called a *secret* key).
- Set `SAVE_REGISTRATION_IMAGE = False` in `config.py` for a stricter
  no-image-retention posture.

## Honest Limitations (state these in your viva/paper)

- Anti-spoofing ships with a **lightweight frequency-domain heuristic** by
  default. `spoof_model.py` provides a documented plug-in point for a real
  pretrained ONNX anti-spoof model (see its docstring) — it will
  transparently switch to using it once the model file is placed at
  `models/anti_spoof.onnx`. Without that file, the heuristic is not
  production-grade liveness detection.
- The ML Trust Engine needs real, human-labeled usage data before it can
  train at all (cold-start problem) — until then the system correctly and
  automatically falls back to the rule-based engine. Its accuracy numbers
  are only as good as the labeled data you feed it, and with a small,
  self-collected dataset should be reported honestly, not oversold.
- The FAR/FRR/ROC evaluation in `learning_lab/02_threshold_evaluation.py`
  approximates genuine/impostor trials from logged + labeled data rather
  than a rigorous scripted verification protocol — state this
  approximation explicitly in your paper's methodology.
- Matching uses a linear scan over the gallery (fine for tens–hundreds of
  users; a larger deployment would need an ANN index like FAISS).
- Presence tracking assumes a single camera / single identity per frame focus,
  not multi-person continuous tracking.

## Research Paper Direction

*"Trust-Based Privacy-Preserving Presence Verification Using Face Embeddings"*
— we propose and evaluate a trust-based decision framework built around a
pretrained face recognizer, **not** a novel recognition algorithm. State it
exactly that way in your abstract/intro; see `paper/OUTLINE.md`.

## License

MIT — see `LICENSE`.
