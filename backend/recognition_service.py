"""
recognition_service.py
========================
Orchestrates a single "recognize this frame" request end-to-end:

  1. Decode the incoming image
  2. Detect faces (face_engine)
  3. For each face: decrypt the stored gallery embeddings (privacy) and
     find the best cosine-similarity match
  4. Compute quality metrics + Trust Score (trust_engine)
  5. Apply the decision (auto_accept / retry / reject)
  6. On auto_accept: update presence intelligence (presence) and log it
  7. Always: write a recognition_logs row for audit/research purposes

This keeps main.py (the API layer) thin - it just calls
`recognize_frame(image_bgr)` and returns the JSON result.
"""

import base64
import numpy as np
import cv2

import database as db
import face_engine
import privacy
import trust_engine
import ml_trust_engine
import presence
from config import FACE_MATCH_THRESHOLD, ML_ENGINE_PROMOTE_WHEN_READY


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Convert a 'data:image/jpeg;base64,...' string from the browser into an OpenCV BGR image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_bgr


def _load_gallery():
    """
    Decrypt every registered user's embedding into memory for comparison.
    For an MVP-scale gallery (tens to low hundreds of users) a simple
    linear scan is fast enough (a few milliseconds); a production system
    with thousands of users would use an approximate-nearest-neighbour
    vector index (e.g. FAISS) instead - noted as future work.
    """
    users = db.get_all_users_with_embeddings()
    gallery = []
    for u in users:
        try:
            emb = privacy.decrypt_embedding(u["embedding_enc"])
            gallery.append({"id": u["id"], "name": u["name"], "external_id": u["external_id"], "embedding": emb})
        except Exception:
            continue  # corrupted/incompatible record - skip safely
    return gallery


def find_best_match(embedding: np.ndarray, gallery: list):
    best_user, best_sim = None, -1.0
    for entry in gallery:
        sim = face_engine.cosine_similarity(embedding, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_user = entry
    return best_user, best_sim


def recognize_frame(image_bgr: np.ndarray) -> dict:
    """Main pipeline. Returns a JSON-serializable dict describing every face found."""
    faces = face_engine.detect_faces(image_bgr)
    gallery = _load_gallery()

    results = []
    for face in faces:
        quality = face_engine.compute_quality_metrics(image_bgr, face)
        best_user, similarity = find_best_match(face.embedding, gallery)

        is_match = best_user is not None and similarity >= FACE_MATCH_THRESHOLD

        if is_match:
            recent_sims = db.get_recent_similarities_for_user(best_user["id"])
            trust_result = trust_engine.compute_trust_score(similarity, quality, recent_sims)
            user_id = best_user["id"]
            name = best_user["name"]
            external_id = best_user["external_id"]
        else:
            # Unknown face: we still compute a trust breakdown for transparency,
            # but similarity_to_score will naturally pull the score down.
            trust_result = trust_engine.compute_trust_score(max(similarity, 0), quality, [])
            trust_result["decision"] = "unknown"
            user_id = None
            name = "Unknown"
            external_id = None

        # ML Trust Engine: only produces a result once a model has been
        # trained on human-labeled data (see ml_trust_engine.py). Until
        # then this is (None, None) and the rule-based decision remains
        # authoritative - the cold-start fallback in action.
        ml_decision, ml_confidence = ml_trust_engine.predict_decision(similarity, quality)

        # AUTHORITATIVE DECISION: this is what actually drives behaviour
        # (presence logging, the UI's primary badge). Bootstrap-then-replace
        # policy - see config.ML_ENGINE_PROMOTE_WHEN_READY: the ML model
        # takes over from the rule engine once trained, rather than the two
        # running forever as separate, equally-weighted systems. Both raw
        # decisions are still logged below purely for audit/comparison -
        # that's what powers the Learning Lab's evaluation scripts, not a
        # second production decision-maker.
        #
        # VOCABULARY NOTE: the ML model is trained on human_label values
        # ("accept"/"retry"/"reject" - what a reviewer clicks), while the
        # rule engine's decision vocabulary is "auto_accept"/"retry"/
        # "reject"/"unknown". They mean the same thing but use different
        # words for the accept case - normalize here so presence logging
        # (which checks specifically for "auto_accept") keeps working
        # correctly once the ML engine takes over.
        if is_match and ml_decision is not None and ML_ENGINE_PROMOTE_WHEN_READY:
            authoritative_decision = "auto_accept" if ml_decision == "accept" else ml_decision
            decision_source = "ml"
        else:
            authoritative_decision = trust_result["decision"]
            decision_source = "rule"

        log_id = db.insert_recognition_log(
            user_id=user_id,
            similarity=round(float(similarity), 4),
            trust_score=trust_result["trust_score"],
            decision=trust_result["decision"],
            quality=quality,
            ml_decision=ml_decision,
            final_decision=authoritative_decision,
        )

        if is_match and authoritative_decision == "auto_accept":
            presence.record_presence_ping(user_id, trust_result["trust_score"], similarity)

        results.append({
            "log_id": log_id,
            "bbox": [int(v) for v in face.bbox],
            "user_id": user_id,
            "name": name,
            "external_id": external_id,
            "similarity": round(float(similarity), 4),
            "trust_score": trust_result["trust_score"],
            "decision": authoritative_decision,
            "decision_source": decision_source,
            "rule_decision": trust_result["decision"],
            "sub_scores": trust_result["sub_scores"],
            "weights": trust_result["weights"],
            "spoof_source": quality.get("spoof_source", "heuristic"),
            "ml_decision": ml_decision,
            "ml_confidence": ml_confidence,
        })

    return {"faces_detected": len(faces), "results": results}


def register_face(image_bgr: np.ndarray, name: str, external_id: str) -> dict:
    """
    MODULE 1 + 2: detect exactly one face in the registration image, compute
    its embedding, encrypt it, and persist a new user row.
    """
    if db.external_id_exists(external_id):
        return {"success": False, "error": f"ID '{external_id}' is already registered."}

    faces = face_engine.detect_faces(image_bgr)
    if len(faces) == 0:
        return {"success": False, "error": "No face detected. Please try again with better lighting."}
    if len(faces) > 1:
        return {"success": False, "error": "Multiple faces detected. Please ensure only one person is in frame."}

    face = faces[0]
    quality = face_engine.compute_quality_metrics(image_bgr, face)

    # Basic registration quality gate - reuse the same explainable metrics.
    if quality["blur"] < 25:
        return {"success": False, "error": "Image is too blurry. Please hold steady and retry."}
    if quality["face_size"] < 20:
        return {"success": False, "error": "Face is too small/far from camera. Please move closer."}

    embedding_enc = privacy.encrypt_embedding(face.embedding)
    user_id = db.create_user(name=name, external_id=external_id, embedding_enc=embedding_enc)

    return {
        "success": True,
        "user_id": user_id,
        "name": name,
        "external_id": external_id,
        "quality": quality,
    }
