"""
trust_engine.py
=================
MODULE 3: TRUST ENGINE (the project's core novelty)

DESIGN PHILOSOPHY
-------------------
A raw face-match similarity score answers "does this look like a match?"
It does NOT answer "should I actually trust and act on this match right
now?". A high similarity score computed from a blurry, half-turned, badly
lit, or possibly spoofed frame is much less trustworthy than the same
similarity score from a sharp, frontal, well-lit, clearly-live frame.

The Trust Engine combines several EXPLAINABLE sub-scores (each 0-100)
into one final Trust Score using a transparent WEIGHTED SUM - deliberately
avoiding a black-box learned model, so every number can be justified in a
viva or research paper.

    trust_score = sum(weight_i * sub_score_i)   for i in
        {similarity, blur, brightness, pose, face_size, spoof}

    (+ a small historical-stability adjustment, explained below)

DECISION LOGIC
---------------
    trust_score >= TRUST_AUTO_ACCEPT  -> "auto_accept" (mark attendance)
    TRUST_RETRY <= trust_score < TRUST_AUTO_ACCEPT -> "retry" (ask for another frame)
    trust_score < TRUST_RETRY -> "reject"
"""

import numpy as np

from config import TRUST_WEIGHTS, TRUST_AUTO_ACCEPT, TRUST_RETRY, FACE_MATCH_THRESHOLD


def similarity_to_score(similarity: float) -> float:
    """
    Map cosine similarity (roughly -1 to 1, but practically 0 to 1 for
    faces) onto a 0-100 scale, anchored around our match threshold so the
    score behaves intuitively:
      - similarity == FACE_MATCH_THRESHOLD  -> score ~= 50
      - similarity == 1.0                   -> score = 100
      - similarity <= 0                     -> score = 0
    """
    if similarity <= 0:
        return 0.0
    if similarity >= 1:
        return 100.0
    if similarity >= FACE_MATCH_THRESHOLD:
        # Linear stretch from [threshold, 1.0] -> [50, 100]
        frac = (similarity - FACE_MATCH_THRESHOLD) / (1.0 - FACE_MATCH_THRESHOLD)
        return 50.0 + frac * 50.0
    else:
        # Linear stretch from [0, threshold] -> [0, 50]
        frac = similarity / FACE_MATCH_THRESHOLD
        return frac * 50.0


def historical_stability_bonus(recent_similarities: list) -> float:
    """
    OPTIONAL small adjustment (-5 to +5 points) based on how consistent
    this user's recent recognition similarities have been. A person who
    has been recognized consistently with low variance is a slightly more
    "known-good" signal than a brand-new or highly erratic history.
    Returns a delta to ADD to the base trust score (can be negative).
    """
    if len(recent_similarities) < 3:
        return 0.0  # not enough history to judge stability yet
    std = float(np.std(recent_similarities))
    mean = float(np.mean(recent_similarities))
    if mean <= 0:
        return 0.0
    coefficient_of_variation = std / mean
    if coefficient_of_variation < 0.05:
        return 5.0    # very stable history
    elif coefficient_of_variation < 0.15:
        return 2.0    # reasonably stable
    elif coefficient_of_variation > 0.35:
        return -5.0   # erratic / suspicious history
    return 0.0


def compute_trust_score(similarity: float, quality: dict, recent_similarities=None) -> dict:
    """
    Main entry point. Returns a dict with the final score, decision,
    and every sub-score so the frontend can show a full breakdown
    (no black box).
    """
    sub_scores = {
        "similarity": similarity_to_score(similarity),
        "blur": quality.get("blur", 0),
        "brightness": quality.get("brightness", 0),
        "pose": quality.get("pose", 0),
        "face_size": quality.get("face_size", 0),
        "spoof": quality.get("spoof", 0),
    }

    base_score = sum(TRUST_WEIGHTS[k] * sub_scores[k] for k in TRUST_WEIGHTS)

    bonus = historical_stability_bonus(recent_similarities or [])
    final_score = float(np.clip(base_score + bonus, 0, 100))

    if final_score >= TRUST_AUTO_ACCEPT:
        decision = "auto_accept"
    elif final_score >= TRUST_RETRY:
        decision = "retry"
    else:
        decision = "reject"

    return {
        "trust_score": round(final_score, 2),
        "decision": decision,
        "sub_scores": {k: round(v, 2) for k, v in sub_scores.items()},
        "stability_bonus": bonus,
        "weights": TRUST_WEIGHTS,
    }
