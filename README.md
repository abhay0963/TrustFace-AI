# TrustFace AI

**Privacy-Preserving Intelligent Presence Verification System**

Most face attendance systems answer one question: *"who is this?"* and
act on it immediately. TrustFace AI adds a step in between: *how certain
am I, and is that certain enough to act on?* Attendance is just one use
of that trust-scored identity pipeline — the same design applies to
access control, exam proctoring, or anywhere blindly trusting an AI match
is a real risk.

## What makes it different

| Typical attendance system | TrustFace AI |
|---|---|
| Stores raw face photos | Stores only AES-256-GCM **encrypted embeddings** |
| Binary matched/not-matched | A **Trust Score** (0-100) with a full breakdown |
| Accepts every match | **Accept / Retry / Reject**, with a reason |
| One-time timestamp | Duration **and** consistency (was the person there the whole time?) |
| Black-box confidence | Every score traces to a formula, or a classifier you can inspect |

## The five pieces

1. **Face recognition** — InsightFace/ArcFace, pretrained, 512-d embeddings, cosine similarity
2. **Privacy** — AES-256-GCM encrypted embeddings, no raw images stored
3. **Trust Engine** — a rule-based formula from day one; a trained Decision Tree/Random Forest takes over once you've labeled enough real attempts
4. **Presence Intelligence** — entry/last-seen/duration, plus a consistency % and a session category (Continuous / Frequent Exits / Brief Appearance / Suspicious Intermittent)
5. **Learning Lab** — five scripts (`learning_lab/`) that teach the underlying concepts using your own data — no Jupyter needed

## Tech stack

Python · FastAPI · InsightFace (ONNX Runtime) · OpenCV · SQLite ·
PyCryptodome (AES) · scikit-learn · matplotlib · Jinja2 · vanilla JS ·
Plotly. No paid APIs, no cloud GPU — runs on one laptop.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:8000`. First run downloads the InsightFace model
(~350MB, one-time) — everything after that is offline.

1. **Register** a face at `/register`.
2. **Live Attendance** at `/attendance` — watch the trust score and decision update live.
3. **Dashboard** at `/` — today's attendance, presence %, consistency, recent activity.
4. **Logs** at `/logs` — label attempts (Accept/Retry/Reject), then train the ML Trust Engine once you have ~30 labels.

## Project structure

```
TrustFace-AI/
├── backend/            FastAPI app + all logic (see docs/ARCHITECTURE.md)
├── frontend/           Jinja2 templates + vanilla JS + CSS
├── learning_lab/       5 scripts teaching the AI concepts with real data
├── database/           trustface.db + secret.key (generated, gitignored)
├── models/             InsightFace cache + trained ML model (generated, gitignored)
├── docs/                ARCHITECTURE.md, CODE_WALKTHROUGH.md, MODEL_CARD_ARCFACE.md
├── paper/                research paper outline
├── requirements.txt
└── run.py
```

## Tuning

`backend/config.py` holds every threshold and weight, each with a short
comment on what it does. See `docs/ARCHITECTURE.md` for how to tune the
match threshold using your own collected data.

## Honest limitations

- Anti-spoofing is a texture heuristic unless you drop in a real ONNX model at `models/anti_spoof.onnx` (see `spoof_model.py`).
- The ML Trust Engine needs real labeled data before it trains at all — it falls back to the rule engine until then.
- FAR/FRR evaluation approximates genuine/impostor trials from logged data, not a scripted verification protocol.
- Gallery matching is a linear scan — fine at tens/hundreds of users, not thousands.
- Presence tracking assumes one camera, one person in frame.

## Research framing

This is a **system architecture and decision framework**, not a novel
recognition algorithm — ArcFace is used as-is. See `paper/OUTLINE.md`.

## License

MIT — see `LICENSE`.
