"""
01_threshold_comparison.py
=============================
QUESTION: How would accuracy, FAR, and FRR change if we picked a different
FACE_MATCH_THRESHOLD than the one currently configured?

This directly answers the review comment "threshold comparison (0.5 vs 0.6
vs 0.7)" with real computed numbers instead of guesses.

DATA SOURCE
-------------
Uses real genuine/impostor similarity scores from your labeled
recognition_logs if you have enough (see learning_lab/02 for the same
methodology + its caveats). If you don't have enough real data yet, this
script falls back to a SYNTHETIC genuine/impostor distribution so the
comparison methodology can still be demonstrated - every output from the
synthetic path is clearly labeled "SYNTHETIC DATA" and should be presented
as such, never as real evaluation results.

RUN
----
python experiments/01_threshold_comparison.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import database as db
from config import FACE_MATCH_THRESHOLD

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATE_THRESHOLDS = [0.35, 0.45, 0.55, 0.65, 0.75]


def get_real_scores():
    with db.db_cursor() as cur:
        genuine = [r["similarity"] for r in cur.execute(
            "SELECT similarity FROM recognition_logs WHERE human_label='accept' AND similarity IS NOT NULL")]
        impostor = [r["similarity"] for r in cur.execute(
            "SELECT similarity FROM recognition_logs WHERE (decision='unknown' OR human_label='reject') AND similarity IS NOT NULL")]
    return np.array(genuine), np.array(impostor)


def get_synthetic_scores(n=200):
    """Clearly-labeled synthetic stand-in, roughly shaped like real ArcFace score distributions."""
    rng = np.random.default_rng(42)
    genuine = np.clip(rng.normal(0.72, 0.09, n), 0, 1)
    impostor = np.clip(rng.normal(0.28, 0.11, n), 0, 1)
    return genuine, impostor


def evaluate_threshold(genuine, impostor, t):
    far = np.mean(impostor >= t) if len(impostor) else 0.0
    frr = np.mean(genuine < t) if len(genuine) else 0.0
    correct = np.sum(genuine >= t) + np.sum(impostor < t)
    accuracy = correct / (len(genuine) + len(impostor))
    return accuracy, far, frr


def main():
    print("=" * 70)
    print("EXPERIMENT 01: Threshold Comparison")
    print("=" * 70)

    genuine, impostor = get_real_scores()
    is_synthetic = len(genuine) < 5 or len(impostor) < 5

    if is_synthetic:
        print("\n*** Not enough real labeled data - using SYNTHETIC genuine/impostor ***")
        print("*** distributions to demonstrate the methodology. Label these clearly ***")
        print("*** as synthetic if you use this output in your paper.                ***\n")
        genuine, impostor = get_synthetic_scores()
    else:
        print(f"\nUsing REAL data: {len(genuine)} genuine, {len(impostor)} impostor scores.\n")

    thresholds_to_test = sorted(set(CANDIDATE_THRESHOLDS + [FACE_MATCH_THRESHOLD]))

    print(f"{'Threshold':<12}{'Accuracy':<12}{'FAR':<10}{'FRR':<10}{'Note':<20}")
    rows = []
    for t in thresholds_to_test:
        acc, far, frr = evaluate_threshold(genuine, impostor, t)
        note = "<- currently configured" if abs(t - FACE_MATCH_THRESHOLD) < 1e-9 else ""
        print(f"{t:<12.2f}{acc*100:<11.2f}%{far*100:<9.2f}%{frr*100:<9.2f}%{note}")
        rows.append((t, acc, far, frr))

    x = [r[0] for r in rows]
    acc_vals = [r[1] * 100 for r in rows]
    far_vals = [r[2] * 100 for r in rows]
    frr_vals = [r[3] * 100 for r in rows]

    title_suffix = " (SYNTHETIC DATA)" if is_synthetic else ""
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.02
    ax.bar([xi - width for xi in x], acc_vals, width=width, label="Accuracy", color="#2dd4bf")
    ax.bar(x, far_vals, width=width, label="FAR", color="#ef4d5f")
    ax.bar([xi + width for xi in x], frr_vals, width=width, label="FRR", color="#f5a623")
    ax.axvline(FACE_MATCH_THRESHOLD, color="#5b6b7a", linestyle="--", label=f"Configured ({FACE_MATCH_THRESHOLD})")
    ax.set_xlabel("Candidate threshold")
    ax.set_ylabel("Percent")
    ax.set_title(f"Threshold Comparison{title_suffix}")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "threshold_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}{title_suffix}")


if __name__ == "__main__":
    main()
