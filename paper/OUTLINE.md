# Research Paper Outline

**Working Title:** Trust-Based Privacy-Preserving Presence Verification Using
Face Embeddings

**Positioning:** This paper contributes a *system architecture and decision
framework*, not a novel neural network. State it explicitly: **"We propose
and evaluate a trust-based decision framework around pretrained face
recognition"** — not "we developed a novel AI algorithm." ArcFace/
InsightFace is used as a pretrained component (see `docs/MODEL_CARD_ARCFACE.md`);
the contribution is the trust-scoring, ML decision, and presence layers
built around it.

## 1. Introduction
- Motivation: blind trust in AI predictions in existing biometric attendance
  systems; privacy concerns from raw image storage; lack of confidence-aware
  decision-making.
- Contribution statement (3 bullet points, matching the 3 novel modules:
  Trust Engine, Privacy Layer, Presence Intelligence).

## 2. Related Work
- Face recognition: ArcFace / InsightFace / RetinaFace (cite original papers).
- Biometric template protection (why raw storage is a known risk).
- Existing attendance systems and their limitations (cite a few surveyed
  systems — Kaggle/GitHub projects, commercial products).

## 3. Problem Statement
- Formalize: given a video frame stream, decide `accept / retry / reject`
  for an identity claim, while (a) preserving privacy of stored biometric
  data and (b) producing an explainable confidence signal instead of a
  binary match.

## 4. Methodology / System Architecture
- Reuse `docs/ARCHITECTURE.md` diagrams.
- Module-by-module description: Detection & Recognition, Privacy Layer,
  Trust Engine (formula + weights table), Presence Intelligence.
- Explicitly state the Trust Engine formula:
  `trust = Σ w_i * score_i` and justify each `w_i`.

## 5. Experimental Setup
- Hardware: laptop spec, CPU-only inference.
- Dataset: your own small collected dataset (N registered users, M
  recognition attempts, genuine vs. impostor split). State this is a
  small-scale, self-collected evaluation — be honest about sample size.
- Metrics: FAR, FRR, average trust score, retry rate (see
  `docs/THRESHOLDS.md`).

## 6. Results
- Threshold sensitivity plot (FAR/FRR vs. threshold).
- Trust score distribution histograms (genuine vs. impostor vs. poor-quality
  frames).
- Presence accuracy vs. manually verified ground truth for a test session.

## 7. Discussion
- Why weighted-sum explainable scoring was chosen over a learned trust model
  (interpretability, small-data setting, viva-defensibility).
- Honest limitations (heuristic spoof-detection, linear-scan matching,
  single-camera assumption) — copy from README's Limitations section.

## 8. Future Work
- Trained liveness/anti-spoofing model.
- ANN-based vector search (FAISS) for large galleries.
- Multi-camera fusion for presence tracking.
- Learned (rather than hand-weighted) trust calibration, validated against
  a larger labeled dataset.

## 9. Conclusion

## References
- Deng, J. et al. "ArcFace: Additive Angular Margin Loss for Deep Face
  Recognition." CVPR 2019.
- InsightFace project documentation.
- Add any FAR/FRR/biometric-privacy papers you cite during literature review.
