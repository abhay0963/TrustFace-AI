"""
05_synthetic_data_augmentation.py
====================================
QUESTION: With very little real labeled data (realistic for a 3-day MVP),
can perturbation-based synthetic augmentation help train a better
classifier - and how do we prove it honestly, without accidentally
inflating our own accuracy numbers?

METHOD (read this carefully - the methodology IS the point of this script)
--------------------------------------------------------------------------
1. Split REAL labeled data into train/test FIRST, before any synthetic
   data is generated. The test set stays 100% REAL, always - synthetic
   examples are NEVER evaluated on, only trained on. Evaluating on
   synthetic data would let us "prove" whatever we want; that's not a
   real result.
2. For each REAL training example, generate a small number of SYNTHETIC
   siblings by adding Gaussian noise to its quality-metric features
   (blur/brightness/pose/face_size/spoof - each jittered independently,
   clipped back into a valid range) while keeping its human_label
   unchanged. This simulates "the same underlying quality situation,
   captured on a slightly different frame."
3. Train one model on REAL-ONLY training data, another on REAL+SYNTHETIC
   training data. Evaluate BOTH on the same real-only test set.
4. Compare test accuracy. If augmentation genuinely helps, the augmented
   model should do at least as well, ideally better, on the untouched real
   test set - report whichever actually happened, honestly, even if
   augmentation doesn't help (that's a valid and reportable result too).
5. Save the full augmented dataset (real + synthetic, clearly flagged in an
   `is_synthetic` column) to a CSV for inspection/reproducibility.

RUN
----
python experiments/05_synthetic_data_augmentation.py
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import database as db
from config import ML_RANDOM_STATE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_REAL_EXAMPLES = 8  # lower bar than ml_trust_engine's 30 - this script exists precisely for the scarce-data case
SYNTHETIC_PER_REAL = 4  # how many jittered siblings to generate per real example
NOISE_STD = 8.0  # standard deviation of the Gaussian jitter, on the 0-100 quality scale

FEATURE_KEYS = ["similarity_x100", "blur_score", "brightness_score", "pose_score", "face_size_score", "spoof_score"]


def build_real_rows():
    rows = db.get_labeled_training_data()
    clean = []
    for r in rows:
        if None in (r["similarity"], r["blur_score"], r["brightness_score"],
                    r["pose_score"], r["face_size_score"], r["spoof_score"]):
            continue
        clean.append({
            "similarity_x100": r["similarity"] * 100.0,
            "blur_score": r["blur_score"],
            "brightness_score": r["brightness_score"],
            "pose_score": r["pose_score"],
            "face_size_score": r["face_size_score"],
            "spoof_score": r["spoof_score"],
            "human_label": r["human_label"],
        })
    return clean


def jitter_row(row, rng):
    synthetic = dict(row)
    for key in FEATURE_KEYS:
        noisy = row[key] + rng.normal(0, NOISE_STD)
        lo, hi = (0, 100) if key != "similarity_x100" else (0, 100)
        synthetic[key] = float(np.clip(noisy, lo, hi))
    return synthetic


def to_xy(rows):
    X = np.array([[r[k] for k in FEATURE_KEYS] for r in rows], dtype=np.float32)
    y = np.array([r["human_label"] for r in rows])
    return X, y


def main():
    print("=" * 70)
    print("EXPERIMENT 05: Synthetic Data Augmentation (perturbation-based)")
    print("=" * 70)

    real_rows = build_real_rows()
    print(f"\nReal labeled examples available: {len(real_rows)} (minimum for this demo: {MIN_REAL_EXAMPLES})")
    if len(real_rows) < MIN_REAL_EXAMPLES:
        print("\nNot enough real labeled data even for this lower-bar demo.")
        print("Label a few more attempts on the Logs page (even 8-10 is enough to try this), then re-run.")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(ML_RANDOM_STATE)

    # Step 1: split REAL data into train/test BEFORE any synthetic generation.
    labels = [r["human_label"] for r in real_rows]
    try:
        train_rows, test_rows = train_test_split(
            real_rows, test_size=0.3, random_state=ML_RANDOM_STATE, stratify=labels
        )
    except ValueError:
        # Some class has too few members to stratify - fall back to a plain random split.
        print("(Too few examples of some class to stratify - using a plain random split instead.)")
        train_rows, test_rows = train_test_split(real_rows, test_size=0.3, random_state=ML_RANDOM_STATE)

    print(f"Real train: {len(train_rows)}   Real test (NEVER touched by synthetic data): {len(test_rows)}")

    # Step 2: generate synthetic siblings ONLY from the training rows.
    synthetic_rows = []
    for row in train_rows:
        for _ in range(SYNTHETIC_PER_REAL):
            synthetic_rows.append(jitter_row(row, rng))

    print(f"Generated {len(synthetic_rows)} synthetic training examples "
          f"({SYNTHETIC_PER_REAL} per real training example, Gaussian noise std={NOISE_STD}).")

    # Step 3: train real-only vs real+synthetic, evaluate both on the same real-only test set.
    X_train_real, y_train_real = to_xy(train_rows)
    X_train_aug, y_train_aug = to_xy(train_rows + synthetic_rows)
    X_test, y_test = to_xy(test_rows)

    model_real_only = DecisionTreeClassifier(max_depth=4, random_state=ML_RANDOM_STATE)
    model_real_only.fit(X_train_real, y_train_real)
    acc_real_only = accuracy_score(y_test, model_real_only.predict(X_test))

    model_augmented = DecisionTreeClassifier(max_depth=4, random_state=ML_RANDOM_STATE)
    model_augmented.fit(X_train_aug, y_train_aug)
    acc_augmented = accuracy_score(y_test, model_augmented.predict(X_test))

    print(f"\n{'Model':<30}{'Test Accuracy (on REAL test set)':<35}")
    print(f"{'Real data only':<30}{acc_real_only*100:<34.2f}%")
    print(f"{'Real + synthetic augmented':<30}{acc_augmented*100:<34.2f}%")

    delta = (acc_augmented - acc_real_only) * 100
    if delta > 0.5:
        verdict = f"Augmentation HELPED on this test set (+{delta:.2f} points)."
    elif delta < -0.5:
        verdict = f"Augmentation HURT on this test set ({delta:.2f} points) - report this honestly too."
    else:
        verdict = "No meaningful difference on this test set."
    print(f"\nVerdict: {verdict}")
    print("Note: with a tiny real test set, this single comparison has high variance -")
    print("treat it as illustrative of the methodology, not a definitive claim, in your paper.")

    # Save the full augmented dataset (clearly flagged) to CSV.
    csv_path = os.path.join(OUTPUT_DIR, "synthetic_augmented_dataset.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_KEYS + ["human_label", "is_synthetic"])
        for r in train_rows:
            writer.writerow([r[k] for k in FEATURE_KEYS] + [r["human_label"], 0])
        for r in synthetic_rows:
            writer.writerow([r[k] for k in FEATURE_KEYS] + [r["human_label"], 1])
    print(f"\nSaved full labeled dataset (real + synthetic, flagged via is_synthetic column) to {csv_path}")

    plt.figure(figsize=(6, 5))
    plt.bar(["Real only", "Real + Synthetic"], [acc_real_only * 100, acc_augmented * 100],
            color=["#5b6b7a", "#2dd4bf"])
    plt.ylabel("Test Accuracy on REAL held-out data (%)")
    plt.title("Effect of Synthetic Augmentation on Test Accuracy")
    plt.ylim(0, 100)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "synthetic_augmentation_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
