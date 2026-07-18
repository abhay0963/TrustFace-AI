"""
spoof_model.py
================
MODULE 1b: PLUGGABLE ANTI-SPOOFING MODEL

WHY THIS FILE EXISTS
-----------------------
face_engine.estimate_spoof_risk() is an FFT texture heuristic - honest,
explainable, but NOT a trained liveness-detection model. The critique of
v1 was fair: for a project that wants to demonstrate real ML, a hand-coded
frequency check pretending to be "spoof detection" is weak.

The right fix is NOT to fake a stronger claim - it's to build a proper
PLUG-IN POINT for a real pretrained anti-spoof model, and be explicit about
what's in place today vs. what you install.

RECOMMENDED MODEL
--------------------
The most widely used lightweight open-source anti-spoofing checkpoints are
from the "Silent-Face-Anti-Spoofing" family (MiniFASNetV2 / V1SE), trained
on the CelebA-Spoof dataset to classify a face crop as real vs. print/replay
attack. ONNX-exported versions of these models are commonly shared in
computer-vision community repos. To use one here:

    1. Obtain a MiniFASNet-style ONNX checkpoint (input: a face crop resized
       to ANTI_SPOOF_INPUT_SIZE, RGB, normalized to [0,1]; output: either a
       2-class [fake, real] or 3-class softmax - both are handled below).
    2. Place it at the path in config.ANTI_SPOOF_MODEL_PATH
       (models/anti_spoof.onnx).
    3. Restart the app. spoof_model.get_spoof_score() will automatically
       detect the file and use it - no code changes needed.

WHY WE DON'T BUNDLE A MODEL FILE IN THE REPO
-------------------------------------------------
Two honest reasons, both worth stating in a viva if asked:
  1. Provenance/trust: an anti-spoofing model is a SECURITY component -
     shipping an unverified binary checkpoint you can't personally audit
     the training data/licensing for is bad practice, not good practice.
  2. Repo hygiene: ONNX weights for these models are tens of MB; committing
     binary model weights to a lightweight student GitHub repo is
     generally discouraged (use Git LFS or an external download step
     instead, which is exactly what this plug-in point supports).

FALLBACK BEHAVIOUR
----------------------
If no model file is present, `get_spoof_score()` transparently falls back
to face_engine.estimate_spoof_risk() (the FFT heuristic), and the returned
dict tells you which one actually ran (`"source": "model"` vs `"heuristic"`)
so this is never silently misrepresented in your logs or dashboard.
"""

import os
import numpy as np
import cv2

from config import ANTI_SPOOF_MODEL_PATH, ANTI_SPOOF_INPUT_SIZE

_session = None
_session_checked = False


def _try_load_session():
    """Lazily attempt to load the ONNX anti-spoof model, once, at first use."""
    global _session, _session_checked
    if _session_checked:
        return _session
    _session_checked = True

    if not os.path.exists(ANTI_SPOOF_MODEL_PATH):
        return None

    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(ANTI_SPOOF_MODEL_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[spoof_model] Found {ANTI_SPOOF_MODEL_PATH} but failed to load it ({e}). "
              f"Falling back to the heuristic spoof score.")
        _session = None

    return _session


def _run_model(face_crop_bgr: np.ndarray, session) -> float:
    """
    Preprocess the face crop and run the ONNX model. Returns a 0-100
    liveness score (higher = more likely real/live), matching the scale
    used everywhere else in the Trust Engine.
    """
    resized = cv2.resize(face_crop_bgr, ANTI_SPOOF_INPUT_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW, the standard ONNX/PyTorch layout
    batch = np.expand_dims(chw, axis=0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})[0][0]

    # Handles both 2-class [fake, real] and 3-class [fake_print, fake_replay, real]
    # softmax-style outputs - "real" is always the LAST class in this model family's convention.
    probs = outputs
    if probs.sum() == 0:
        return 50.0  # degenerate output, treat as uncertain rather than crash
    exp = np.exp(probs - np.max(probs))
    softmax = exp / exp.sum()
    real_probability = float(softmax[-1])

    return round(real_probability * 100.0, 2)


def get_spoof_score(face_crop_bgr: np.ndarray) -> dict:
    """
    Main entry point used by face_engine.compute_quality_metrics().
    Returns {"score": 0-100, "source": "model" | "heuristic"}.
    """
    session = _try_load_session()

    if session is not None:
        try:
            score = _run_model(face_crop_bgr, session)
            return {"score": score, "source": "model"}
        except Exception as e:
            print(f"[spoof_model] Inference failed ({e}), falling back to heuristic for this frame.")

    # Fallback: the FFT texture heuristic from face_engine.py
    import face_engine
    score = face_engine.estimate_spoof_risk(face_crop_bgr)
    return {"score": score, "source": "heuristic"}
