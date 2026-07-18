"""
face_engine.py
===============
MODULE 1: FACE DETECTION AND RECOGNITION

WHAT IS INSIGHTFACE / ARCFACE?
--------------------------------
InsightFace is an open-source computer vision library. The model pack we
use ("buffalo_l") bundles two neural networks:

1. DETECTOR (RetinaFace-style, a CNN):
   Input  -> a full image / video frame
   Output -> bounding box(es) + 5 facial landmarks (eyes, nose, mouth
             corners) for every face found. Landmarks are used to
             ALIGN the face (rotate/crop it into a standard pose)
             before recognition, which massively improves accuracy.

2. RECOGNIZER (ArcFace, a ResNet CNN trained with "Additive Angular
   Margin Loss"):
   Input  -> an aligned face crop (112x112 pixels)
   Output -> a 512-dimensional EMBEDDING vector.
   ArcFace is trained so that embeddings of the SAME person end up close
   together in this 512-d space (small angle between vectors), and
   embeddings of DIFFERENT people end up far apart (large angle). This
   is why we can compare two people using COSINE SIMILARITY instead of
   ever needing to retrain a classifier for new users - a brand new
   person can be enrolled just by computing one embedding.

COSINE SIMILARITY
-------------------
For two vectors a and b:
    cosine_similarity = (a . b) / (||a|| * ||b||)
This measures the COSINE OF THE ANGLE between the vectors, ignoring their
magnitude. InsightFace embeddings are usually near-unit-length already,
but we normalize explicitly to be safe.

QUALITY METRICS (used later by the Trust Engine)
---------------------------------------------------
We also compute simple, explainable image-quality signals here:
  - blur:       Laplacian variance (sharper images -> higher variance)
  - brightness: mean pixel intensity
  - pose:       yaw estimate from the 5-point landmarks (how frontal)
  - face_size:  face bounding-box area relative to the frame
  - spoof:      a lightweight heuristic (NOT a trained anti-spoof model -
                see docstring on estimate_spoof_risk for the honest
                limitation you should state in your paper/viva)
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import INSIGHTFACE_MODEL_NAME, INSIGHTFACE_CTX_ID, INSIGHTFACE_DET_SIZE

_face_app = None


def get_face_app() -> FaceAnalysis:
    """
    Lazily initialize the InsightFace app (singleton pattern).
    Loading the model is slow (downloads on first run, then loads weights
    into memory), so we only do it once per process.
    """
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME)
        _face_app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=INSIGHTFACE_DET_SIZE)
    return _face_app


def detect_faces(image_bgr: np.ndarray):
    """
    Run detection + recognition on a BGR image (OpenCV's default format).
    Returns a list of InsightFace `Face` objects, each with:
      .bbox        -> [x1, y1, x2, y2]
      .kps         -> 5 landmark points
      .embedding   -> 512-d numpy vector (already computed by ArcFace)
      .det_score   -> detector's own confidence for this face
    """
    app = get_face_app()
    return app.get(image_bgr)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_blur_score(face_crop_gray: np.ndarray) -> float:
    """
    Laplacian variance: convolve the image with a Laplacian kernel
    (a 2nd-derivative edge detector). Sharp images have lots of high-
    frequency edge content -> high variance. Blurry images look "smooth"
    -> low variance. We cap/normalize to a 0-100 scale for the Trust Engine.
    """
    variance = cv2.Laplacian(face_crop_gray, cv2.CV_64F).var()
    # Empirically, variance > ~150 is "sharp" for a webcam face crop.
    score = min(100.0, (variance / 150.0) * 100.0)
    return round(float(score), 2)


def compute_brightness_score(face_crop_gray: np.ndarray) -> float:
    """
    Mean pixel intensity (0-255). We want it near the middle of the range
    (not too dark, not blown-out/overexposed), so we score based on
    distance from an ideal midpoint of 130.
    """
    mean_val = float(np.mean(face_crop_gray))
    ideal = 130.0
    deviation = abs(mean_val - ideal)
    score = max(0.0, 100.0 - (deviation / ideal) * 100.0)
    return round(float(score), 2)


def compute_pose_score(kps: np.ndarray) -> float:
    """
    Rough frontal-pose estimate using the 5 landmarks:
    [left_eye, right_eye, nose, left_mouth, right_mouth].
    If the face is turned sideways, the nose point drifts away from the
    horizontal midpoint of the two eyes. We turn that horizontal offset
    (normalized by eye distance) into a 0-100 "how frontal" score.
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist == 0:
        return 0.0
    offset_ratio = abs(nose[0] - eye_mid_x) / eye_dist
    # offset_ratio near 0 -> frontal. Above ~0.5 -> strongly turned.
    score = max(0.0, 100.0 - offset_ratio * 200.0)
    return round(float(score), 2)


def compute_face_size_score(bbox, frame_shape) -> float:
    """
    Face bounding box area relative to the full frame area. Too small
    (person far from camera) hurts recognition reliability.
    """
    x1, y1, x2, y2 = bbox
    face_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    frame_area = frame_shape[0] * frame_shape[1]
    ratio = face_area / frame_area if frame_area > 0 else 0
    # A face taking up ~8-40% of the frame is considered ideal.
    if ratio < 0.02:
        score = (ratio / 0.02) * 60.0
    elif ratio > 0.6:
        score = max(0.0, 100.0 - (ratio - 0.6) * 200.0)
    else:
        score = 100.0
    return round(float(min(100.0, score)), 2)


def estimate_spoof_risk(face_crop_bgr: np.ndarray) -> float:
    """
    LIGHTWEIGHT, EXPLAINABLE anti-spoofing heuristic (NOT a trained
    liveness-detection model - be upfront about this limitation in your
    viva/paper). We use frequency-domain texture analysis:

    A real face photographed by a webcam has natural high-frequency
    texture (skin pores, micro-shadows). A face SPOOFED by holding up a
    printed photo or a phone screen tends to show:
      - moire/screen-door patterns (phone screens), or
      - flatter texture with fewer natural high frequencies (printouts)

    We approximate this by looking at the ratio of high-frequency to
    total energy in the grayscale face crop's FFT (Fast Fourier
    Transform). This returns a "liveness score" 0-100 (higher = more
    likely a real live face). It is intentionally simple and should be
    described as a heuristic proof-of-concept, not a production-grade
    anti-spoofing system (those typically require depth sensors, IR
    cameras, or dedicated deep-learning liveness models).
    """
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6

    y, x = np.ogrid[:h, :w]
    center_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    low_freq_energy = magnitude[center_mask].sum()
    total_energy = magnitude.sum() + 1e-6
    high_freq_ratio = 1.0 - (low_freq_energy / total_energy)

    # Map a "reasonable" texture ratio range to a 0-100 liveness score.
    score = np.clip((high_freq_ratio - 0.15) / (0.55 - 0.15), 0.0, 1.0) * 100.0
    return round(float(score), 2)


def compute_quality_metrics(image_bgr: np.ndarray, face) -> dict:
    """
    Bundle all quality sub-scores for one detected face into a dict.
    `spoof` is computed via spoof_model.get_spoof_score(), which
    transparently uses a real pretrained ONNX anti-spoof model if one has
    been placed at config.ANTI_SPOOF_MODEL_PATH, and otherwise falls back
    to estimate_spoof_risk() (the FFT heuristic below). `spoof_source`
    tells you which one actually ran for this frame - surfaced in the logs
    so the distinction is never hidden.
    """
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_bgr.shape[1], x2), min(image_bgr.shape[0], y2)
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return {"blur": 0, "brightness": 0, "pose": 0, "face_size": 0, "spoof": 0, "spoof_source": "heuristic"}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    import spoof_model
    spoof_result = spoof_model.get_spoof_score(crop)

    return {
        "blur": compute_blur_score(gray),
        "brightness": compute_brightness_score(gray),
        "pose": compute_pose_score(face.kps),
        "face_size": compute_face_size_score(face.bbox, image_bgr.shape),
        "spoof": spoof_result["score"],
        "spoof_source": spoof_result["source"],
    }
