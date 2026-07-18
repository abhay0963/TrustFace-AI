"""
03_trust_score_feature_ablation.py
=====================================
QUESTION: Which single feature, if degraded to its worst case, hurts the
Trust Score the most? This is the rule-based engine's equivalent of the ML
Trust Engine's feature_importances_ output (see ml_trust_engine.py) - here
computed directly from the WEIGHTED-SUM FORMULA rather than learned, so you
can directly compare "what we designed the weights to prioritize" against
"what the trained classifier ended up actually prioritizing" - a great
paper figure pairing.

METHOD
-------
Start from an ideal baseline (every feature at its best value). For each
feature in turn, degrade ONLY that feature to a poor value (holding all
others at their best) and measure how far the Trust Score drops. Rank
features by this drop.

RUN
----
python experiments/03_trust_score_feature_ablation.py
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import trust_engine

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_SIMILARITY = 0.95
BEST_QUALITY = {"blur": 95, "brightness": 90, "pose": 95, "face_size": 90, "spoof": 95}
WORST_VALUE = 10  # a "poor" reading for whichever feature is being degraded


def main():
    print("=" * 70)
    print("EXPERIMENT 03: Trust Score Feature Ablation")
    print("=" * 70)

    baseline_result = trust_engine.compute_trust_score(BEST_SIMILARITY, BEST_QUALITY, recent_similarities=[])
    baseline_score = baseline_result["trust_score"]
    print(f"\nBaseline (everything ideal): Trust Score = {baseline_score:.2f}, decision = {baseline_result['decision']}")
    print(f"Degrading ONE feature at a time to {WORST_VALUE}/100, all others held at baseline:\n")

    impacts = {}

    # Similarity is handled separately since it's on a 0-1 scale, not 0-100.
    result = trust_engine.compute_trust_score(0.10, BEST_QUALITY, recent_similarities=[])
    impacts["similarity"] = baseline_score - result["trust_score"]

    for feature in ["blur", "brightness", "pose", "face_size", "spoof"]:
        quality = dict(BEST_QUALITY)
        quality[feature] = WORST_VALUE
        result = trust_engine.compute_trust_score(BEST_SIMILARITY, quality, recent_similarities=[])
        impacts[feature] = baseline_score - result["trust_score"]

    ranked = sorted(impacts.items(), key=lambda x: -x[1])
    print(f"{'Feature':<15}{'Score Drop':<15}{'Configured Weight':<20}")
    for feature, drop in ranked:
        weight = trust_engine.TRUST_WEIGHTS.get(feature, "n/a (similarity is handled separately)")
        print(f"{feature:<15}{drop:<15.2f}{str(weight):<20}")

    print("\nInterpretation: the ranking here should roughly track config.TRUST_WEIGHTS -")
    print("if it doesn't, that usually means a non-linear scoring curve (e.g. face_size_score's")
    print("clamping) is interacting with the weight in a way worth double-checking.")

    plt.figure(figsize=(8, 5))
    features = [r[0] for r in ranked]
    drops = [r[1] for r in ranked]
    plt.barh(features, drops, color="#2dd4bf")
    plt.xlabel("Trust Score drop when this feature is degraded")
    plt.title("Feature Ablation: Impact on Trust Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "trust_score_feature_ablation.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")
    print("\nTip: after training the ML Trust Engine (learning_lab/04), compare this plot")
    print("directly against its feature_importances_ output for a rule-based vs. learned comparison.")


if __name__ == "__main__":
    main()
