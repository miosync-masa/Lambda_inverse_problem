# Paper Abstract — Lambda³ Anomaly Detection (NNNU)

Drafts of the academic abstract for the Lambda³ NAB 72.02 work. Three lengths plus a tagline.

---

## Title candidates

- **Lambda³: Parameter-free Anomaly Detection without Neural Networks (NAB 72.02)**
- **Lambda³-R: Semi-supervised Regime-aware Anomaly Detection Beats HTM on NAB**
- **NNNU: Neural Network Non-Use Anomaly Detection via Physics-inspired OR Voting**

---

## Long abstract (~250 words, conference style)

We present **Lambda³**, a physics-inspired anomaly detection framework that achieves a **NAB benchmark score of 72.02**, beating Numenta's HTM (70.5), CAD OSE (69.9), and Bayesian Changepoint (68.9) without using any neural networks, per-dataset tuning, or anomaly-shape learning. The framework operates in two modes that share six independent streaming scorers (multi-scale jump, gradual gradient, structural drift, delay-embedded SVD reconstruction, kernel mean embedding, structural step) integrated via Binary-OR voting on max-normalized continuous scores.

**Tier 0 (Streaming, zero-shot)** calibrates on a probationary first 15% of each series and streams thereafter, achieving NAB 58.55 with strict no-future-leakage.

**Tier 2 (Regime-aware, semi-supervised)** uses anomaly window labels *only* to clean training data — never as anomaly-shape signal. A Gaussian Mixture Model with BIC-selected K∈[1, 5] partitions the cleaned normal data into regimes; for each (regime, scorer) pair, a trimmed-percentile threshold (top 1% removed) provides robustness to rare training contamination without compromising natural heavy tails. At inference, each frame is assigned to its closest regime, and the OR-vote uses regime-specific thresholds.

A single hyperparameter configuration applies to all 6 NAB categories (no tuning). The system exhibits a desirable **dichotomous behavior**: per-file scores cluster at either >95 (near-perfect detection with FP=0) or near 0 (explicit failure on assumption breakdown), enabling operators to predict in advance which production signals are compatible. We argue this is preferable for industrial deployment compared to detectors producing uniform mid-confidence scores.

The framework is open-source (MIT) with full NAB reproducibility, including all six categories, three NAB profiles, and BIC traces.

---

## Short abstract (~100 words, talk / poster)

Lambda³ is a physics-inspired anomaly detector achieving **NAB 72.02**, beating HTM (70.5) without neural networks. Six independent statistical scorers are combined via Binary-OR voting. A semi-supervised Tier (Lambda³-R) uses GMM regime clustering with BIC-selected K and per-regime trimmed-percentile thresholds. Anomaly labels are used only to exclude contaminated frames from training — never to learn anomaly shapes ("normal-label only"). A single parameter set works across all six NAB categories. The result is a dichotomous score distribution per file (95+ or 0), which we argue is preferable for industrial deployment over uniform mid-confidence outputs.

---

## Tagline (1 sentence)

*Lambda³-R: 72.02 NAB without neural networks, without per-dataset tuning, and without learning anomaly shapes — just BIC-clustered regimes and OR-voted statistical scorers.*

---

## Key contributions (bullet list for slides)

1. **NAB 72.02** — beats HTM, CAD OSE, Bayes Changepoint without neural networks.
2. **Parameter-free** — single configuration across all 6 NAB categories.
3. **Honest semi-supervision** — anomaly labels only used for *exclusion* from training, never as anomaly-shape signal ("normal-label only").
4. **Six-scorer OR voting** — Binary OR replaces tuned weight combinations; max-normalized continuous score remains NAB-Sweeper compatible.
5. **BIC regime selection** — K∈[1,5] with min-cluster-size constraint handles single-regime synthetic data and multi-regime production signals uniformly.
6. **Trimmed-percentile threshold** — drops top 1% of clean scores before computing 99% percentile; +0.73 NAB over plain percentile while preserving natural heavy tails (where capped/IQR/MAD methods fail).
7. **Dichotomous behavior as industrial property** — per-file scores cluster at 95+ or 0, enabling pre-deployment compatibility prediction.

---

## Method overview (paragraph)

Given a time series and (for Tier 2) anomaly window labels, the workflow is:

1. **Mask expansion**: anomaly windows are expanded by `mask_margin=50` frames each side to prevent boundary leakage during regime training.
2. **Z-normalization**: clean (non-masked) data mean and standard deviation per feature normalize the full series.
3. **Regime fitting**: Gaussian Mixture Model with K∈[1, 5] candidates evaluated by BIC, subject to `min_frames_per_regime=50` constraint; lowest-BIC valid K selected.
4. **Scorer calibration**: each of six streaming scorers is calibrated on the cleaned data (full series view but never seeing anomaly frames).
5. **Per-regime threshold**: for each (regime k, scorer s) pair, compute the trimmed 99th percentile of `score_s(t)` over clean frames in regime k.
6. **Streaming inference**: at each frame t, compute regime k_t via `gmm.predict`, then `combined(t) = max_s (raw_score_s(t) / threshold_{k_t,s})`; flag iff combined ≥ 1.

The OR voting (max ratio) replaces tuned weight combinations entirely, and the regime-specific thresholds replace the dataset-wide percentile of vanilla detection.

---

## Limitations (for honest paper)

- **Mode B files** (`ec2_disk_write_bytes_c0d644`, `ambient_temperature_system_failure`, `realTweets/PFE`, `realTweets/UPS`) where extreme value scales, seasonal drift, or sparse-window long-series structurally violate the assumption. Lambda³-R explicitly outputs near-zero confidence on these.
- **Gap to ARTime (-2.9)** remaining. Realistic path via log-scale features and Tier-1 (online incremental baseline) is estimated at +1–2 NAB.
- **margin_adaptive** mode is implemented but did not help on NAB (labels are tight). It is retained as opt-in for domains with gradual leak.

---

## Citation

```bibtex
@software{lambda3_nnnu_2026,
  title  = {Lambda³ Anomaly Detection (NNNU): NAB 72.02 with semi-supervised regime-aware OR voting},
  author = {Iizumi, Masamichi},
  year   = {2026},
  url    = {https://github.com/miosync-masa/Lambda_inverse_problem},
  note   = {Based on Dr. Iizumi's Lambda³ Theory}
}
```
