"""
04_train_trust_classifier.py
===============================
CONCEPT: The full supervised machine learning pipeline, end to end -
features, labels, train/test split, training, evaluation, confusion
matrix, feature importances. This is the actual "we trained a model"
story for your paper/viva (see backend/ml_trust_engine.py for the
production version this script exercises).

WHAT THIS SCRIPT DOES
-----------------------
1. Loads every recognition_logs row that has a human_label (ground truth,
   set on the Logs page) - this is your (X, y) dataset.
2. Explains and prints the feature matrix shape.
3. Trains BOTH a Decision Tree and a Random Forest (for comparison), each
   on the same train/test split, so you can directly compare a single
   interpretable tree against an ensemble.
4. Prints accuracy, per-class precision/recall/F1, and a confusion matrix
   for both, saves a confusion matrix heatmap PNG, and - for the Decision
   Tree specifically - prints the actual learned if/else rules, which you
   can literally read aloud in a viva.
5. Saves the BETTER of the two (by test accuracy) as the production model
   used by recognition_service.py at inference time (same file
   backend/main.py's /api/train-model endpoint would produce).

RUN
----
python learning_lab/04_train_trust_classifier.py
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import database as db
from config import MIN_TRAINING_SAMPLES, ML_TEST_SIZE, ML_RANDOM_STATE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = ["similarity_x100", "blur", "brightness", "pose", "face_size", "spoof"]


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


def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Greens")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center",
                      color="white" if cm[i][j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def main():
    print("=" * 70)
    print("LEARNING LAB 04: Training the Trust Engine Classifier")
    print("=" * 70)

    X, y = build_dataset()
    print(f"\nLabeled examples available: {len(X)} (minimum required: {MIN_TRAINING_SAMPLES})")

    if len(X) < MIN_TRAINING_SAMPLES:
        print("\nNot enough labeled data yet.")
        print("Go to the Logs page after some Live Attendance sessions and label")
        print(f"at least {MIN_TRAINING_SAMPLES} attempts (Accept/Retry/Reject), then re-run this.")
        return

    print(f"Feature matrix shape: {X.shape}  (rows=examples, cols=6 features)")
    print(f"Features: {FEATURE_NAMES}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import joblib
    from config import ML_MODEL_PATH, ML_METRICS_PATH

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE, stratify=y
    )
    print(f"\nTrain/test split: {len(X_train)} training examples, {len(X_test)} held-out test examples "
          f"({ML_TEST_SIZE*100:.0f}% held out, random_state={ML_RANDOM_STATE} for reproducibility)")

    labels_sorted = sorted(set(y))
    results = {}

    for name, model in [
        ("Decision Tree", DecisionTreeClassifier(max_depth=4, random_state=ML_RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=ML_RANDOM_STATE)),
    ]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

        print(f"\n{'-'*70}\n{name}\n{'-'*70}")
        print(f"Test accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0))

        plot_confusion_matrix(cm, labels_sorted, f"{name} - Confusion Matrix",
                               f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        results[name] = {"model": model, "accuracy": acc, "cm": cm.tolist()}

    print(f"\n{'-'*70}\nDecision Tree Rules (fully interpretable - read this in your viva)\n{'-'*70}")
    print(export_text(results["Decision Tree"]["model"], feature_names=FEATURE_NAMES))

    if hasattr(results["Random Forest"]["model"], "feature_importances_"):
        print(f"\n{'-'*70}\nRandom Forest Feature Importances\n{'-'*70}")
        importances = results["Random Forest"]["model"].feature_importances_
        for fname, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 50)
            print(f"  {fname:<18} {imp:.3f}  {bar}")

    # Save the better-performing model as the production model.
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n{'='*70}")
    print(f"Best model: {best_name} (test accuracy {results[best_name]['accuracy']*100:.2f}%)")
    print(f"Saving as the production model used by recognition_service.py...")

    joblib.dump(results[best_name]["model"], ML_MODEL_PATH)
    metrics = {
        "model_type": "decision_tree" if best_name == "Decision Tree" else "random_forest",
        "trained_on": len(X_train),
        "tested_on": len(X_test),
        "total_labeled_examples": len(X),
        "accuracy": round(float(results[best_name]["accuracy"]), 4),
        "confusion_matrix": results[best_name]["cm"],
        "confusion_matrix_labels": labels_sorted,
        "feature_names": FEATURE_NAMES,
    }
    with open(ML_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {ML_MODEL_PATH}")
    print(f"Saved metrics to {ML_METRICS_PATH}")
    print(f"Confusion matrix plots saved to {OUTPUT_DIR}/")
    print("\nThe live app (recognition_service.py) will now use this trained model")
    print("alongside the rule-based Trust Engine on every future recognition.")


if __name__ == "__main__":
    main()
