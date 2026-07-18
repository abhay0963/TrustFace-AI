"""
config.py
=========
Central configuration for TrustFace AI.

WHY A CONFIG FILE?
------------------
Instead of scattering "magic numbers" (like 0.55, 0.85, 100) all over the
codebase, we keep every tunable value in ONE place. This is a basic but
important software engineering practice: it makes the system easier to
tune, easier to explain in your viva ("why is the threshold 0.55?"), and
easier to reference in your research paper's "Experimental Setup" section.
"""

import os

# -----------------------------------------------------------------------
# BASE PATHS
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "trustface.db")
KEY_FILE_PATH = os.path.join(DATABASE_DIR, "secret.key")  # AES key, never commit this
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
UNKNOWN_FACES_DIR = os.path.join(ASSETS_DIR, "unknown_faces")

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# FACE RECOGNITION MODEL (InsightFace / ArcFace)
# -----------------------------------------------------------------------
# "buffalo_l" is InsightFace's standard pretrained model pack. It bundles:
#   - a face DETECTOR   (RetinaFace-style) -> finds bounding boxes + 5 landmarks
#   - a face RECOGNIZER  (ArcFace, ResNet backbone) -> turns a face crop into
#     a 512-dimensional embedding vector that represents "who" the face is.
# It downloads automatically (~350MB) the first time you run the app,
# straight from InsightFace's model zoo, and is cached locally afterwards.
INSIGHTFACE_MODEL_NAME = "buffalo_l"

# ctx_id = -1 means "run on CPU". Set to 0 if you have a CUDA GPU configured.
INSIGHTFACE_CTX_ID = -1
INSIGHTFACE_DET_SIZE = (640, 640)  # detector input resolution

# -----------------------------------------------------------------------
# FACE MATCHING THRESHOLD
# -----------------------------------------------------------------------
# Two embeddings are compared using COSINE SIMILARITY, a number between
# -1 and 1 that measures the angle between two vectors:
#   1.0  -> identical direction (very likely same person)
#   0.0  -> unrelated
#   -1.0 -> opposite direction
# For ArcFace embeddings on real-world faces, similarity scores for the
# SAME person are usually well above 0.45-0.55, and different people
# usually score below that. This threshold is a classic trade-off between
# False Acceptance Rate (FAR) and False Rejection Rate (FRR) - see
# docs/THRESHOLDS.md for how to tune it using your own collected data.
FACE_MATCH_THRESHOLD = 0.45

# -----------------------------------------------------------------------
# TRUST ENGINE WEIGHTS
# -----------------------------------------------------------------------
# The Trust Score is a WEIGHTED SUM of several sub-scores (each 0-100).
# Weights must sum to 1.0. Tune these based on what matters most to you.
TRUST_WEIGHTS = {
    "similarity": 0.45,   # how close the embedding match is (most important)
    "blur": 0.15,         # image sharpness (Laplacian variance)
    "brightness": 0.10,   # exposure quality
    "pose": 0.10,         # how frontal the face is
    "face_size": 0.10,    # how large/close the face is in frame
    "spoof": 0.10,        # simple anti-spoofing heuristic
}

# Decision thresholds applied to the FINAL trust score (0-100)
TRUST_AUTO_ACCEPT = 85   # >= this -> automatically mark attendance
TRUST_RETRY = 60         # between RETRY and AUTO_ACCEPT -> ask for another frame
# < TRUST_RETRY -> reject

# -----------------------------------------------------------------------
# PRESENCE INTELLIGENCE
# -----------------------------------------------------------------------
# A "session" is the working/class window we compute presence % against.
SESSION_START_HOUR = 9    # 09:00
SESSION_END_HOUR = 17     # 17:00
# If no recognition ping arrives within this many seconds, we consider the
# person to have "left" and a new entry (re-entry) will be logged.
PRESENCE_TIMEOUT_SECONDS = 120

# -----------------------------------------------------------------------
# WEBCAM / LIVE RECOGNITION LOOP
# -----------------------------------------------------------------------
# How often the frontend JS sends a captured frame to the backend for
# recognition, in milliseconds.
CAPTURE_INTERVAL_MS = 1500

# -----------------------------------------------------------------------
# ML TRUST ENGINE (learned decision model, on top of the rule-based one)
# -----------------------------------------------------------------------
# The rule-based Trust Engine (trust_engine.py) is a WEIGHTED SUM - simple,
# explainable, but not "machine learning" in the trained-model sense. Once
# enough HUMAN-VERIFIED labels exist (see database.human_label / the "Label"
# buttons on the Logs page), we train a small classical ML classifier
# (Decision Tree / Random Forest) to predict the decision instead.
#
# IMPORTANT DESIGN NOTE: this is a ONE-WAY EVOLUTION, not two permanent
# parallel systems. The rule engine is the bootstrap mechanism that makes
# the app usable from minute one with zero data. Once ML_ENGINE_PROMOTE_WHEN_READY
# is True (default) and a trained model exists, its prediction BECOMES the
# authoritative decision used for accept/reject/presence logging - the
# rule-based score is still computed and shown (it remains a useful,
# always-available explainability signal and a sanity check), but it no
# longer drives behaviour once a trained model has taken over. Version
# story: V1 rule engine -> V2 Decision Tree -> V3 Random Forest, each
# replacing the previous production decision-maker, not stacking forever.
ML_MODEL_PATH = os.path.join(MODELS_DIR, "trust_classifier.pkl")
ML_METRICS_PATH = os.path.join(MODELS_DIR, "trust_classifier_metrics.json")
# Minimum human-labeled examples required before we'll even attempt training.
# Below this, a train/test split is too noisy to trust - we simply keep
# using the rule-based engine and tell you how many more labels are needed.
MIN_TRAINING_SAMPLES = 30
# Fraction of labeled data held out for testing (never trained on).
ML_TEST_SIZE = 0.25
ML_RANDOM_STATE = 42  # fixed seed -> reproducible train/test splits, reproducible results for your paper
# Set False to keep the rule engine authoritative even after a model is
# trained (useful while you're still validating the ML model's behaviour
# before trusting it to drive real attendance decisions).
ML_ENGINE_PROMOTE_WHEN_READY = True

# -----------------------------------------------------------------------
# ANTI-SPOOFING MODEL (pluggable)
# -----------------------------------------------------------------------
# face_engine.estimate_spoof_risk() is an FFT texture HEURISTIC, not a
# trained model - fine for an MVP but explicitly not production liveness
# detection. This path is where a real pretrained ONNX anti-spoof model
# (e.g. a MiniFASNet-style checkpoint from the Silent-Face-Anti-Spoofing
# family) can be dropped in. If the file exists, spoof_model.py loads and
# uses it transparently; if it doesn't, the system automatically falls back
# to the heuristic - see spoof_model.py for exactly how to plug one in and
# why we ship a documented fallback instead of bundling an unverified
# binary model file.
ANTI_SPOOF_MODEL_PATH = os.path.join(MODELS_DIR, "anti_spoof.onnx")
ANTI_SPOOF_INPUT_SIZE = (80, 80)  # expected crop size for most MiniFASNet-style checkpoints

# -----------------------------------------------------------------------
# PRESENCE CONSISTENCY (Module 4 upgrade)
# -----------------------------------------------------------------------
# Instead of just accumulating presence_seconds, we also bucket the session
# into fixed-size windows and check whether the user was detected in each
# window. This produces a CONSISTENCY score - e.g. "95% continuous" vs
# "61% interrupted" - which is a much more informative signal than raw
# duration alone (someone present for 4 hours but only in the first and
# last 5 minutes should NOT look the same as someone present continuously).
PRESENCE_BUCKET_MINUTES = 5

# Session category thresholds (see presence.classify_session) - turns a
# consistency % + transition pattern into a human-readable category instead
# of a bare number.
PRESENCE_CATEGORY_BRIEF_MAX_PCT = 15         # below this presence %, regardless of consistency -> "Brief Appearance"
PRESENCE_CATEGORY_CONTINUOUS_MIN_CONSISTENCY = 85   # at/above this consistency -> "Continuous Attendance"
PRESENCE_CATEGORY_SUSPICIOUS_TRANSITION_RATE = 0.4  # fraction of bucket-to-bucket flips that counts as "erratic"

# -----------------------------------------------------------------------
# PRIVACY LAYER
# -----------------------------------------------------------------------
# We NEVER store raw face images tied to an identity in the database.
# Only the encrypted 512-d embedding vector is stored. Raw registration
# images are optionally kept in assets/ for demo purposes only, and can be
# disabled entirely by setting SAVE_REGISTRATION_IMAGE = False.
SAVE_REGISTRATION_IMAGE = True
