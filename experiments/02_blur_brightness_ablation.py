"""
02_blur_brightness_ablation.py
=================================
QUESTION: How much does the Trust Score actually move when blur or
brightness changes, with everything else held at a "good" baseline?

This is a controlled ABLATION, not a data-driven evaluation - it directly
exercises trust_engine.compute_trust_score() with synthetic sweeps, so it
always runs regardless of how much real data you've collected. Good for
sanity-checking the weights in config.TRUST_WEIGHTS: if a small blur change
swings the trust score far more than intended, that's a signal the weight
needs revisiting.

RUN
----
python experiments/02_blur_brightness_ablation.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import trust_engine

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A "good" baseline profile - everything except the swept feature stays fixed here.
BASELINE_SIMILARITY = 0.75
BASELINE_QUALITY = {"blur": 90, "brightness": 85, "pose": 90, "face_size": 85, "spoof": 90}


def sweep_feature(feature_name, values):
    scores = []
    for v in values:
        quality = dict(BASELINE_QUALITY)
        quality[feature_name] = v
        result = trust_engine.compute_trust_score(BASELINE_SIMILARITY, quality, recent_similarities=[])
        scores.append(result["trust_score"])
    return scores


def main():
    print("=" * 70)
    print("EXPERIMENT 02: Blur & Brightness Ablation")
    print("=" * 70)
    print(f"\nBaseline: similarity={BASELINE_SIMILARITY}, quality={BASELINE_QUALITY}")
    print("Sweeping ONE feature at a time from 0-100, holding everything else fixed.\n")

    values = np.arange(0, 101, 5)
    blur_scores = sweep_feature("blur", values)
    brightness_scores = sweep_feature("brightness", values)

    print(f"{'Value':<8}{'Trust (blur swept)':<22}{'Trust (brightness swept)':<25}")
    for v, b, r in zip(values, blur_scores, brightness_scores):
        print(f"{v:<8}{b:<22.2f}{r:<25.2f}")

    blur_range = max(blur_scores) - min(blur_scores)
    brightness_range = max(brightness_scores) - min(brightness_scores)
    print(f"\nTotal trust-score swing from blur (0->100):       {blur_range:.2f} points")
    print(f"Total trust-score swing from brightness (0->100): {brightness_range:.2f} points")
    print(f"(Expected ratio ~= weight ratio: "
          f"{trust_engine.TRUST_WEIGHTS['blur']} / {trust_engine.TRUST_WEIGHTS['brightness']} = "
          f"{trust_engine.TRUST_WEIGHTS['blur']/trust_engine.TRUST_WEIGHTS['brightness']:.2f})")

    plt.figure(figsize=(8, 5))
    plt.plot(values, blur_scores, marker="o", label="Blur swept", color="#2dd4bf")
    plt.plot(values, brightness_scores, marker="o", label="Brightness swept", color="#f5a623")
    plt.xlabel("Feature value (0-100)")
    plt.ylabel("Resulting Trust Score")
    plt.title("Trust Score Sensitivity: Blur vs. Brightness")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "blur_brightness_ablation.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
