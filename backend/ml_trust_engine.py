"""
ml_trust_engine.py
====================
MODULE 3b: LEARNED TRUST ENGINE (this is the actual "machine learning" part)

WHY THIS EXISTS
-----------------
trust_engine.py is a rule-based weighted sum: `trust = Σ w_i * score_i`. It's
transparent and works from day one with zero data, but it is NOT a trained
model - the weights are hand-picked, not learned. If someone asks "where's
the ML in this project," the honest original answer was "InsightFace" (a
model we didn't train). This module gives you a second, genuinely honest
answer: a supervised classifier trained on YOUR OWN collected, human-labeled
data.

THE PIPELINE (classic supervised learning, small-tabular-data flavor)
-------------------------------------------------------------------------
1. COLLECT: every recognition attempt already logs 6 numeric features
   (similarity, blur, brightness, pose, face_size, spoof) - see
   recognition_service.py / database.recognition_logs.
2. LABEL: a human reviewer looks at logged attempts on the Logs page and
   clicks Accept / Retry / Reject - the TRUE correct decision, as judged by
   a person, not the rule engine's guess. This is "human-in-the-loop"
   labeling, and it is what turns raw logs into a genuine (X, y) dataset.
3. SPLIT: once enough labels exist (MIN_TRAINING_SAMPLES), we split into a
   train set and a held-out test set (never seen during training) - the
   test set is what makes the accuracy numbers below meaningful rather than
   just "memorized the training data."
4. TRAIN: fit a DecisionTreeClassifier (and, if you have `scikit-learn`'s
   RandomForestClassifier available, that too) on the training split.
5. EVALUATE: compute accuracy, precision, recall, F1, and a confusion
   matrix on the TEST split. Save these to disk so they can be reported in
   your paper/viva with an honest "trained on N examples, tested on M
   held-out examples, accuracy = X%" statement.
6. DEPLOY: save the trained model with `joblib`. recognition_service.py
   loads it (if present) and uses its prediction instead of - or alongside
   - the rule-based decision.

WHY A DECISION TREE / RANDOM FOREST INSTEAD OF A NEURAL NETWORK?
--------------------------------------------------------------------
This is an important design justification to be ready to defend:
  - We have 6 NUMERIC, already-engineered features (not raw pixels) -
    exactly the regime where classical ML (trees, forests, gradient
    boosting) matches or beats deep learning, and needs orders of magnitude
    less data to do it.
  - A decision tree is directly INTERPRETABLE - you can print the actual
    if/else rules it learned (`export_text`) and show them in your viva,
    which is impossible with a neural net of this scale.
  - A Random Forest (an ensemble of many trees, each trained on a random
    subset of data/features) usually generalizes a bit better than a single
    tree and gives you FEATURE IMPORTANCES - "which signal did the model
    end up trusting most?" - another great paper/viva talking point.
  - With only tens-to-hundreds of labeled examples (realistic for a 3-day
    MVP), a deep learning model would simply overfit; it needs thousands+
    examples to be meaningfully better than a tree here.

COLD START PROBLEM (be ready to explain this honestly)
-----------------------------------------------------------
On a brand new install, there are ZERO human labels, so there is nothing to
train on yet. That's why recognition_service.py always falls back to the
rule-based trust_engine.py until MIN_TRAINING_SAMPLES labels exist - the
system is USABLE from minute one, and gets a genuinely learned decision
layer once you've used it enough to generate training data. This
bootstrap-then-learn pattern is common in real production ML systems
(cold-start problem), not a workaround unique to this project.
"""

import os
import json
import numpy as np

from config import ML_MODEL_PATH, ML_METRICS_PATH, MIN_TRAINING_SAMPLES, ML_TEST_SIZE, ML_RANDOM_STATE
import database as db

FEATURE_NAMES = ["similarity_x100", "blur", "brightness", "pose", "face_size", "spoof"]


def _build_feature_matrix(rows: list):
    """
    Convert raw DB rows into (X, y) numpy arrays.
    NOTE: similarity is stored as a 0-1 cosine value while every other
    feature is already on a 0-100 scale - we multiply similarity by 100 so
    all features share comparable magnitude. Tree-based models don't
    strictly REQUIRE feature scaling (unlike e.g. k-NN or logistic
    regression with regularization), but keeping magnitudes comparable
    still makes feature-importance numbers easier to reason about.
    """
    X, y = [], []
    for r in rows:
        if None in (r["similarity"], r["blur_score"], r["brightness_score"],
                    r["pose_score"], r["face_size_score"], r["spoof_score"]):
            continue  # skip incomplete rows
        X.append([
            r["similarity"] * 100.0,
            r["blur_score"],
            r["brightness_score"],
            r["pose_score"],
            r["face_size_score"],
            r["spoof_score"],
        ])
        y.append(r["human_label"])
    return np.array(X, dtype=np.float32), np.array(y)


def training_status() -> dict:
    """Used by the UI/API to tell the user how close they are to a trainable model."""
    labeled = db.count_labeled_examples()
    return {
        "labeled_examples": labeled,
        "min_required": MIN_TRAINING_SAMPLES,
        "ready_to_train": labeled >= MIN_TRAINING_SAMPLES,
        "model_exists": os.path.exists(ML_MODEL_PATH),
    }


def train_model(model_type: str = "random_forest") -> dict:
    """
    Train a classifier on all currently human-labeled recognition_logs rows.
    Returns a metrics dict (also written to disk at ML_METRICS_PATH) so the
    frontend / your paper can show real accuracy/precision/recall numbers.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import joblib

    rows = db.get_labeled_training_data()
    X, y = _build_feature_matrix(rows)

    if len(X) < MIN_TRAINING_SAMPLES:
        return {
            "success": False,
            "error": f"Only {len(X)} labeled examples available, need at least {MIN_TRAINING_SAMPLES}. "
                     f"Label more recognition attempts on the Logs page first.",
        }

    # Stratify keeps the class proportions (accept/retry/reject) similar in
    # both splits - important with small, possibly imbalanced datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE, stratify=y
    )

    if model_type == "decision_tree":
        # max_depth=4 keeps the tree small enough to print and fully explain in a viva.
        model = DecisionTreeClassifier(max_depth=4, random_state=ML_RANDOM_STATE)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=ML_RANDOM_STATE)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    labels_sorted = sorted(set(y))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_sorted, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist()

    per_class = {
        labels_sorted[i]: {
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i]),
        }
        for i in range(len(labels_sorted))
    }

    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = {
            FEATURE_NAMES[i]: round(float(model.feature_importances_[i]), 4)
            for i in range(len(FEATURE_NAMES))
        }

    metrics = {
        "model_type": model_type,
        "trained_on": len(X_train),
        "tested_on": len(X_test),
        "total_labeled_examples": len(X),
        "accuracy": round(accuracy, 4),
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels_sorted,
        "feature_importances": feature_importances,
        "feature_names": FEATURE_NAMES,
    }

    joblib.dump(model, ML_MODEL_PATH)
    with open(ML_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return {"success": True, "metrics": metrics}


_cached_model = None


def model_available() -> bool:
    return os.path.exists(ML_MODEL_PATH)


def _load_model():
    global _cached_model
    if _cached_model is None and model_available():
        import joblib
        _cached_model = joblib.load(ML_MODEL_PATH)
    return _cached_model


def predict_decision(similarity: float, quality: dict):
    """
    Run the trained classifier on one recognition attempt's features.
    Returns (decision, confidence) or (None, None) if no model is trained
    yet - callers should fall back to the rule-based trust_engine in that case.
    """
    model = _load_model()
    if model is None:
        return None, None

    features = np.array([[
        similarity * 100.0,
        quality.get("blur", 0),
        quality.get("brightness", 0),
        quality.get("pose", 0),
        quality.get("face_size", 0),
        quality.get("spoof", 0),
    ]], dtype=np.float32)

    prediction = model.predict(features)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = round(float(np.max(proba)), 3)

    return str(prediction), confidence


def load_metrics():
    if os.path.exists(ML_METRICS_PATH):
        with open(ML_METRICS_PATH) as f:
            return json.load(f)
    return None


def explain_tree_rules() -> str:
    """
    Returns the trained tree's decision rules as plain text (only works if
    the current saved model is a DecisionTreeClassifier - a Random Forest
    has too many trees to print usefully, but you can still call this after
    training with model_type='decision_tree' to get a single explainable
    tree for your viva).
    """
    from sklearn.tree import export_text
    model = _load_model()
    if model is None or not hasattr(model, "tree_"):
        return "No single-tree model is currently trained (train with model_type='decision_tree' to get printable rules)."
    return export_text(model, feature_names=FEATURE_NAMES)
