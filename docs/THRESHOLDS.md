# Tuning Thresholds & Weights

This is the methodology to justify your numbers in a viva or paper, instead
of leaving them as unexplained magic constants.

## 1. `FACE_MATCH_THRESHOLD` (cosine similarity cutoff)

1. Register 5–10 people.
2. Run `/attendance` for a few minutes per person, letting `recognition_logs`
   accumulate genuine-match similarity scores (`user_id` correctly matches
   the person in frame).
3. Have an unregistered person (or someone else's registered face at an angle)
   trigger a few impostor attempts.
4. Export `recognition_logs` (SQLite → CSV, or query directly) and separate:
   - **Genuine scores**: similarity when `user_id` is the correct person
   - **Impostor scores**: similarity when it's the wrong / unregistered person
5. Plot both distributions. The threshold that best separates them (minimizing
   overlap) is your empirically-justified `FACE_MATCH_THRESHOLD`.
6. This is exactly a **ROC curve / FAR-FRR trade-off** analysis — perfect
   material for your paper's Evaluation section:
   - **False Acceptance Rate (FAR)**: fraction of impostor attempts scoring
     above the threshold (wrongly accepted)
   - **False Rejection Rate (FRR)**: fraction of genuine attempts scoring
     below the threshold (wrongly rejected)
   - Lowering the threshold reduces FRR but increases FAR, and vice versa.

## 2. `TRUST_WEIGHTS`

Start from the defaults in `config.py` (similarity weighted highest at 0.45).
To tune:
- If you find good-quality frames from the wrong person are being
  auto-accepted too often → increase the `similarity` weight relative to
  the others.
- If legitimate users in poor lighting are being rejected too often →
  reduce the `brightness` / `blur` weights, or improve room lighting instead
  (the honest fix).
- Keep weights summing to 1.0.

## 3. `TRUST_AUTO_ACCEPT` / `TRUST_RETRY`

These are your organization's risk tolerance, not something to compute
purely from data:
- Higher `TRUST_AUTO_ACCEPT` (e.g. 90+) = stricter, more retries, fewer
  wrong auto-accepts. Appropriate for higher-stakes use (e.g. exam
  proctoring).
- Lower `TRUST_AUTO_ACCEPT` (e.g. 75) = more convenient, faster attendance
  marking, slightly higher risk tolerance. Appropriate for a low-stakes
  classroom attendance demo.

## Suggested Paper Experiment Table

| Metric | How to compute |
|---|---|
| FAR @ threshold T | impostor scores ≥ T / total impostor attempts |
| FRR @ threshold T | genuine scores < T / total genuine attempts |
| Avg. trust score (genuine) | mean `trust_score` where `decision == auto_accept` and match is correct |
| Retry rate | count(`decision == retry`) / total attempts |
| Presence accuracy | manually verified attendance vs. system-recorded attendance for a test session |
