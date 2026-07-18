# Learning Lab

This is the "AI Learning Playground" — since the project runs in VS Code
locally (not Colab/Jupyter), each concept is a **standalone, runnable `.py`
script** instead of a notebook. Every script prints an explanation to the
terminal as it runs and saves any plots to `learning_lab/output/`. Run them
in order after you've registered a few users and used Live Attendance for a
while (some scripts need real logged data to be useful).

```bash
cd TrustFace-AI
.venv\Scripts\activate          # or source .venv/bin/activate
python learning_lab/01_embeddings_and_similarity.py
python learning_lab/02_threshold_evaluation.py
python learning_lab/03_embedding_visualization.py
python learning_lab/04_train_trust_classifier.py
python learning_lab/05_presence_consistency_demo.py
```

## What each script teaches

| Script | Concept | Needs |
|---|---|---|
| `01_embeddings_and_similarity.py` | What a 512-d embedding actually looks like; hand-computing cosine similarity between real registered users | ≥2 registered users |
| `02_threshold_evaluation.py` | FAR / FRR / ROC curve / accuracy vs. face-match threshold, computed from your own logged data | Some recognition attempts, some human-labeled |
| `03_embedding_visualization.py` | PCA and t-SNE — seeing *why* embeddings work by plotting them in 2D, colored by identity | ≥2 registered users |
| `04_train_trust_classifier.py` | The full supervised ML pipeline: features → labels → train/test split → train → evaluate → confusion matrix → feature importances | ≥30 human-labeled logs (see the Logs page) |
| `05_presence_consistency_demo.py` | Presence timeline bucketing — turning raw pings into a "% continuous" analytic | A user with attendance today |

## Why scripts instead of notebooks

Notebooks are great for exploratory, cell-by-cell iteration in Colab/Jupyter.
Since this project runs entirely in VS Code against a live local SQLite
database, plain scripts that connect to `backend/`, pull real data, print a
clear explanation, and save a plot to disk are simpler to run, version, and
re-run after every attendance session — no kernel state, no notebook diffing
headaches in git. Each script is short enough to read top-to-bottom in one
sitting, which was the actual goal notebooks were trying to serve.

Every plot is saved to `learning_lab/output/*.png` — these are exactly the
figures you'd drop into your research paper's Evaluation / Experiments
section.
