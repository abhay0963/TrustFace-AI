# Model Card — InsightFace `buffalo_l` (RetinaFace + ArcFace)

Following the standard "Model Cards for Model Reporting" format (Mitchell et
al., 2019), adapted for a student project. This documents the pretrained
backbone TrustFace AI builds on — understanding it is part of learning AI,
not just calling `.get(image)` on it.

## Model Details

- **Detector**: RetinaFace-style single-shot detector. Outputs a bounding
  box + 5 facial landmarks (left eye, right eye, nose, left mouth corner,
  right mouth corner) per detected face.
- **Recognizer**: ArcFace, a ResNet-based CNN trained with **Additive
  Angular Margin Loss**. Outputs a 512-dimensional embedding per aligned
  face crop.
- **Model pack**: `buffalo_l`, InsightFace's standard pretrained bundle.
- **Format**: ONNX (Open Neural Network Exchange) — a portable, framework-
  independent format for trained model weights, executed here via ONNX
  Runtime on CPU.
- **Developer**: InsightFace / DeepInsight (open-source project); ArcFace
  itself was introduced by Deng et al., *"ArcFace: Additive Angular Margin
  Loss for Deep Face Recognition"*, CVPR 2019.
- **License**: check InsightFace's repository for the current license terms
  applicable to the specific model pack before any commercial use — this
  project uses it for educational/research purposes only.

## Intended Use

- **Primary intended use**: face detection and face verification/
  identification (1:N matching against a small, locally-stored gallery) in
  a research/educational context.
- **Primary intended users**: developers and researchers building face-
  recognition-adjacent systems who need a strong pretrained baseline
  without training their own recognizer.
- **Out-of-scope uses**: covert/non-consensual surveillance, law-
  enforcement identification without appropriate legal safeguards, any
  deployment where a false match or false non-match carries safety-critical
  consequences without a human-in-the-loop review step (this project's own
  Trust Engine + human-labeling workflow is a direct response to that risk).

## Training Data (high-level — we did not train this model)

ArcFace-family models in the `buffalo_l` pack are typically trained on
large-scale, web-scraped face datasets such as **MS1M (MS-Celeb-1M)** or
**Glint360K** — collections of celebrity/public-figure face images scraped
from the web, spanning on the order of tens of thousands to hundreds of
thousands of identities and millions of images. We did not curate, clean,
or have visibility into this training data ourselves — it comes bundled
with the pretrained weights. This is an important honest boundary: our
project's claims about accuracy, bias, and behavior should be scoped to
*our own evaluation* (see `learning_lab/02_threshold_evaluation.py`), not
assumed to inherit whatever InsightFace's original authors reported.

## Evaluation Data & Results (as reported by the model's original authors)

Standard face-verification benchmarks commonly reported for ArcFace-family
models include LFW (Labeled Faces in the Wild), CFP-FP (frontal-profile
pairs), AgeDB (age-varied pairs), and MegaFace — typically in the
99%+ accuracy range on constrained benchmarks like LFW. These are the
**original authors' reported numbers on their test sets**, not results we
reproduced. Our own empirical results — computed on our own small,
self-collected, human-labeled dataset — live in `learning_lab/output/` and
should be cited separately and explicitly as ours.

## Embedding Details

- **Dimensionality**: 512.
- **Comparison metric**: cosine similarity (see `face_engine.cosine_similarity`).
- **Why cosine similarity works without further training**: ArcFace's
  Additive Angular Margin Loss explicitly shapes the embedding space during
  training so that identity is encoded in *direction* — same-identity
  embeddings cluster at small angles, different identities separate at
  larger angles — even for identities never seen during training. This is
  what makes "register a brand-new person, immediately compare them" work
  without any fine-tuning.

## Known Strengths

- Strong accuracy on frontal, reasonably well-lit faces — the common case
  for a webcam-based attendance/access scenario.
- Fast CPU inference (no GPU required) — appropriate for this project's
  single-laptop constraint.
- Generalizes to unseen identities at inference time (metric learning, not
  closed-set classification).

## Known Weaknesses / Limitations

- **Pose and occlusion**: accuracy degrades on extreme side profiles, heavy
  occlusion (masks, hands over face), or unusual head angles — this is why
  `face_engine.compute_pose_score()` exists as a Trust Engine input, not
  something the recognizer itself corrects for.
- **Lighting extremes**: very dark or overexposed frames reduce embedding
  quality — this is why `compute_brightness_score()` exists.
- **Low resolution / distance**: a face that's too small in frame produces
  a less reliable embedding — this is why `compute_face_size_score()`
  exists.
- **No built-in liveness/anti-spoofing**: ArcFace answers "whose face is
  this," not "is this a live person in front of the camera" — a printed
  photo or screen replay of a registered face can, in principle, produce a
  high-similarity embedding. This is exactly why this project layers a
  separate spoof-detection signal (`spoof_model.py`) on top rather than
  trusting the recognizer alone.

## Bias & Fairness Considerations

Face recognition models trained on web-scraped datasets have a
well-documented history of **demographic accuracy disparities** —
published research (e.g. NIST's Face Recognition Vendor Test studies) has
repeatedly found many face recognition systems perform less reliably for
some demographic groups than others, often correlated with
under-representation of those groups in training data. We have **not**
independently audited `buffalo_l` for this ourselves (that would require a
demographically balanced, labeled evaluation set well beyond this
project's scope). Two honest, actionable takeaways to state in a viva or
paper:

1. Do not claim this system is bias-free — state plainly that no
   independent fairness audit was performed as part of this project.
2. The Trust Engine's multi-signal design (not relying on similarity
   alone) and the human-labeling / review workflow are partial mitigations
   — a low-confidence match gets a `retry` rather than a silent wrong
   accept — but they are not a substitute for a proper fairness evaluation
   before any real-world deployment beyond a course project demo.

## Expected Operating Conditions (for this project)

- Indoor, webcam-distance range (roughly 0.5–2m from camera).
- Reasonably lit indoor environments (classroom/office lighting).
- Largely frontal or near-frontal face angles.
- CPU-only inference — expect inference latency of tens to a few hundred
  milliseconds per frame on a typical laptop, adequate for the 1.5s capture
  interval used by Live Attendance but not for high-frame-rate video
  analytics.

## References

- Deng, J., Guo, J., Xue, N., Zafeiriou, S. *"ArcFace: Additive Angular
  Margin Loss for Deep Face Recognition."* CVPR 2019.
- InsightFace project: https://github.com/deepinsight/insightface
- NIST Face Recognition Vendor Test (FRVT) reports — for demographic
  performance disparity context, cite the most current NIST FRVT report at
  the time of your paper's submission.
