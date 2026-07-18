# Experiments

`learning_lab/` teaches concepts. This folder runs **reproducible
comparisons and ablations** whose output plots are meant to be dropped
directly into your research paper's Evaluation/Experiments section — each
script prints its exact configuration so results are reproducible.

```bash
python experiments/01_threshold_comparison.py
python experiments/02_blur_brightness_ablation.py
python experiments/03_trust_score_feature_ablation.py
python experiments/04_decision_tree_depth_comparison.py
python experiments/05_synthetic_data_augmentation.py
```

| Script | Question it answers | Needs real DB data? |
|---|---|---|
| `01_threshold_comparison.py` | How does accuracy/FAR/FRR change across several candidate `FACE_MATCH_THRESHOLD` values? | Uses real labeled logs if present, else clearly-marked synthetic distributions |
| `02_blur_brightness_ablation.py` | How sensitive is the Trust Score to blur and brightness specifically, holding everything else constant? | No — pure function of `trust_engine.py`, always runs |
| `03_trust_score_feature_ablation.py` | Which single feature moves the Trust Score the most if it's degraded? | No — pure function of `trust_engine.py`, always runs |
| `04_decision_tree_depth_comparison.py` | The classic overfitting/underfitting curve — test accuracy vs. tree depth | Yes — needs ≥30 human-labeled logs |
| `05_synthetic_data_augmentation.py` | Can perturbation-based synthetic augmentation help a small real dataset train a better classifier? | Best with some real labeled logs (falls back to fully synthetic otherwise) |

## On synthetic data — be explicit about it

`01` and `05` can generate or use synthetic data when real data is scarce.
**Every plot and printed table generated from synthetic data is labeled
"SYNTHETIC" in its title/output.** If you use these figures in your paper,
keep that label — presenting synthetic ablation results as if they were
collected from real users would be a serious methodological error. The
correct framing is: *"we additionally validated the Trust Engine's
sensitivity to individual features using controlled synthetic perturbation,
since our real deployment was necessarily small-scale for a 3-day MVP."*
That's a legitimate and common practice in small-data ML projects, stated
honestly.
