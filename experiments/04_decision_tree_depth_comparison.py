"""
04_decision_tree_depth_comparison.py
=======================================
QUESTION: How does Decision Tree depth affect training vs. test accuracy?
The classic overfitting/underfitting curve, run on YOUR real labeled data.

WHY THIS MATTERS
-------------------
ml_trust_engine.py trains a DecisionTreeClassifier with max_depth=4 as a
fixed choice. This script justifies that choice empirically instead of
leaving it as an arbitrary number: too shallow underfits (both train and
test accuracy are low), too deep overfits (train accuracy keeps climbing
toward 100% while test accuracy plateaus or drops - the tree is
memorizing noise in the small training set rather than learning general
patterns).

RUN
----
python experiments/04_decision_tree_depth_comparison.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import database as db
from config import MIN_TRAINING_SAMPLES, ML_TEST_SIZE, ML_RANDOM_STATE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEPTHS_TO_TEST = [1, 2, 3, 4, 5, 6, 8, 10, None]  # None = unlimited depth


def build_dataset():
    rows = db.get_labeled_training_data()
    X, y = [], []
    for r in rows:
        if None in (r["similarity"], r["blur_score"], r["brightness_score"],
                    r["pose_score"], r["face_size_score"], r["spoof_score"]):
            continue
        X.append([r["similarity"] * 100.0, r["blur_score"], r["brightness_score"],
                  r["pose_score"], r["face_size_score"], r["spoof_score"]])
        y.append(r["human_label"])
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    print("=" * 70)
    print("EXPERIMENT 04: Decision Tree Depth Comparison")
    print("=" * 70)

    X, y = build_dataset()
    print(f"\nLabeled examples: {len(X)} (minimum required: {MIN_TRAINING_SAMPLES})")
    if len(X) < MIN_TRAINING_SAMPLES:
        print("\nNot enough labeled data yet. Label more attempts on the Logs page and re-run.")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE, stratify=y
    )

    print(f"\n{'Max Depth':<12}{'Train Accuracy':<18}{'Test Accuracy':<18}")
    train_accs, test_accs = [], []
    for depth in DEPTHS_TO_TEST:
        model = DecisionTreeClassifier(max_depth=depth, random_state=ML_RANDOM_STATE)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        depth_label = "unlimited" if depth is None else str(depth)
        print(f"{depth_label:<12}{train_acc*100:<17.2f}%{test_acc*100:<17.2f}%")

    best_idx = int(np.argmax(test_accs))
    best_depth = DEPTHS_TO_TEST[best_idx]
    print(f"\nBest test accuracy at max_depth={best_depth} ({test_accs[best_idx]*100:.2f}%)")
    print("If train accuracy keeps rising while test accuracy plateaus/drops at higher depths,")
    print("that's overfitting - the tree is memorizing the (small) training set.")

    x_labels = ["∞" if d is None else str(d) for d in DEPTHS_TO_TEST]
    x_pos = range(len(DEPTHS_TO_TEST))

    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, [a * 100 for a in train_accs], marker="o", label="Train accuracy", color="#f5a623")
    plt.plot(x_pos, [a * 100 for a in test_accs], marker="o", label="Test accuracy", color="#2dd4bf")
    plt.xticks(list(x_pos), x_labels)
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy (%)")
    plt.title("Decision Tree Depth vs. Train/Test Accuracy")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "decision_tree_depth_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
