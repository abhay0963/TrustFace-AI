"""
02_threshold_evaluation.py
=============================
CONCEPT: False Acceptance Rate (FAR), False Rejection Rate (FRR), ROC
curves, and threshold selection - the standard way biometric systems are
evaluated.

METHODOLOGY (read this before trusting the numbers)
--------------------------------------------------------
We don't have an explicit "genuine trial / impostor trial" experiment
design (that would mean deliberately having each registered user attempt
recognition against their own profile AND against everyone else's, with
ground truth recorded for every attempt). Instead we approximate using
what's already logged plus human labels from the Logs page:

    GENUINE similarity scores  = attempts where a human confirmed
                                  human_label == "accept" (a person correctly
                                  matched, correctly accepted)
    IMPOSTOR similarity scores = attempts where the system found no
                                  confident match (decision == "unknown")
                                  OR a human explicitly marked human_label
                                  == "reject"

This is a reasonable proxy for a student project, but it IS an
approximation - state this explicitly in your paper's methodology section
rather than presenting these numbers as a rigorous FAR/FRR study. A more
rigorous version would run a scripted enrollment + verification protocol
with deliberate genuine/impostor trials.

WHAT THIS SCRIPT COMPUTES
-----------------------------
- FAR(t) = fraction of IMPOSTOR scores >= threshold t   (wrongly accepted)
- FRR(t) = fraction of GENUINE scores  < threshold t    (wrongly rejected)
- Sweeps t from 0.0 to 1.0, plots both curves, marks the Equal Error Rate
  (EER) point where FAR and FRR cross - the classic single-number summary
  of a biometric system's separability.
- Plots a proper ROC curve (True Accept Rate vs False Accept Rate).
- Prints a confusion matrix at the CURRENTLY CONFIGURED threshold
  (config.FACE_MATCH_THRESHOLD), so you can see exactly how many
  genuine/impostor attempts land on each side of your real, live setting.

RUN
----
python learning_lab/02_threshold_evaluation.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed - just save PNGs
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import database as db
from config import FACE_MATCH_THRESHOLD

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_genuine_and_impostor_scores():
    with db.db_cursor() as cur:
        cur.execute("""
            SELECT similarity FROM recognition_logs
            WHERE human_label = 'accept' AND similarity IS NOT NULL
        """)
        genuine = [row["similarity"] for row in cur.fetchall()]

        cur.execute("""
            SELECT similarity FROM recognition_logs
            WHERE (decision = 'unknown' OR human_label = 'reject') AND similarity IS NOT NULL
        """)
        impostor = [row["similarity"] for row in cur.fetchall()]

    return np.array(genuine), np.array(impostor)


def compute_far_frr(genuine, impostor, thresholds):
    far_list, frr_list = [], []
    for t in thresholds:
        far = np.mean(impostor >= t) if len(impostor) else 0.0
        frr = np.mean(genuine < t) if len(genuine) else 0.0
        far_list.append(far)
        frr_list.append(frr)
    return np.array(far_list), np.array(frr_list)


def main():
    print("=" * 70)
    print("LEARNING LAB 02: FAR / FRR / ROC Threshold Evaluation")
    print("=" * 70)

    genuine, impostor = get_genuine_and_impostor_scores()
    print(f"\nGenuine scores collected: {len(genuine)}  (human-labeled 'accept')")
    print(f"Impostor scores collected: {len(impostor)}  (decision='unknown' or human-labeled 'reject')")

    if len(genuine) < 3 or len(impostor) < 3:
        print("\nNot enough labeled data yet for a meaningful evaluation.")
        print("Use Live Attendance for a while, then label some attempts on the Logs page")
        print("(mix of Accept and Reject/Unknown cases), and re-run this script.")
        return

    thresholds = np.arange(0.0, 1.01, 0.02)
    far, frr = compute_far_frr(genuine, impostor, thresholds)

    # Equal Error Rate: threshold where FAR and FRR are closest
    eer_idx = np.argmin(np.abs(far - frr))
    eer_threshold = thresholds[eer_idx]
    eer_value = (far[eer_idx] + frr[eer_idx]) / 2

    print(f"\nEqual Error Rate (EER): {eer_value*100:.2f}% at threshold = {eer_threshold:.2f}")
    print(f"Your CURRENTLY CONFIGURED threshold (config.FACE_MATCH_THRESHOLD) = {FACE_MATCH_THRESHOLD}")

    # Confusion matrix at the currently configured threshold
    t = FACE_MATCH_THRESHOLD
    true_accepts = np.sum(genuine >= t)
    false_rejects = np.sum(genuine < t)
    false_accepts = np.sum(impostor >= t)
    true_rejects = np.sum(impostor < t)

    print("\nConfusion matrix at FACE_MATCH_THRESHOLD = {:.2f}:".format(t))
    print(f"                  Predicted Accept    Predicted Reject")
    print(f"  Actual Genuine   {true_accepts:>16}    {false_rejects:>16}")
    print(f"  Actual Impostor  {false_accepts:>16}    {true_rejects:>16}")

    accuracy = (true_accepts + true_rejects) / (len(genuine) + len(impostor))
    print(f"\nOverall accuracy at this threshold: {accuracy*100:.2f}%")

    # ---- Plot 1: FAR / FRR vs threshold ----
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far * 100, label="FAR (False Acceptance Rate)", color="#ef4d5f")
    plt.plot(thresholds, frr * 100, label="FRR (False Rejection Rate)", color="#f5a623")
    plt.axvline(eer_threshold, color="#5b6b7a", linestyle="--", label=f"EER threshold ({eer_threshold:.2f})")
    plt.axvline(FACE_MATCH_THRESHOLD, color="#2dd4bf", linestyle="--", label=f"Configured threshold ({FACE_MATCH_THRESHOLD})")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Rate (%)")
    plt.title("FAR / FRR vs. Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "far_frr_vs_threshold.png"), dpi=150)
    plt.close()

    # ---- Plot 2: ROC curve (True Accept Rate vs False Accept Rate) ----
    tar = 1 - frr  # True Accept Rate
    plt.figure(figsize=(6, 6))
    plt.plot(far, tar, color="#2dd4bf")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#5b6b7a", label="Random guess baseline")
    plt.xlabel("False Acceptance Rate")
    plt.ylabel("True Acceptance Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    print(f"\nSaved plots to {OUTPUT_DIR}/far_frr_vs_threshold.png and roc_curve.png")
    print("These are ready to drop directly into your paper's Evaluation section.")


if __name__ == "__main__":
    main()
