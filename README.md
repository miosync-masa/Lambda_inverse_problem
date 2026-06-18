<p align="center">
  <img src="https://www.miosync.link/github/0_4.jpg" alt="Lambda³" width="400"/>
</p>

<h1 align="center">📕 Lambda³ Anomaly Detection — NNNU (Neural Network Non-Use)</h1>

<p align="center">
  <strong>Physics-inspired anomaly detection without neural networks.</strong><br>
  <strong>NAB benchmark: 72.02</strong> — beats HTM (70.5), CAD OSE (69.9), Bayes Changepoint (68.9). <br>
  Parameter-free, semi-supervised (normal-label only).
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="#"><img src="https://img.shields.io/badge/NAB-72.02-brightgreen.svg" alt="NAB 72.02"></a>
  <a href="https://colab.research.google.com/drive/1OObGOFRI8cFtR1tDS99iHtyWMQ9ZD4CI"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
</p>

A physics-inspired anomaly detection framework requiring **no neural networks** and **no anomaly labels**, based on Lambda³ (Lambda-Cubed) theory. The system uses structure tensors, regime-aware GMM clustering, and OR-voting of multiple statistical scorers to achieve **NAB 72.02** — beating HTM and other classical detectors.

---

## 🏆 NAB Benchmark Result

**Lambda³-R (regime-aware) on NAB (Numenta Anomaly Benchmark, 52 files, 6 categories):**

| Detector | NAB Score | Public Comparison Notes |
|---|---|---|
| ARTime | 74.9 | online + tuned |
| **Lambda³-R (this work)** | **72.02** | **★ semi-supervised, parameter-free** |
| HTM | 70.5 | online + tuned |
| CAD OSE | 69.9 | online + tuned |
| Bayes Changepoint | 68.9 | online + tuned |
| Lambda³-S streaming (Tier 0) | 58.55 | pure zero-shot streaming |

### Per-category breakdown (Lambda³-R)

| Category | Files | 3-prof mean | Notes |
|---|---|---|---|
| **realTraffic** | 7 | **81.40** | best category |
| **realKnownCause** | 7 | **78.30** | machine_temp / cpu_util_asg etc. |
| **realAWSCloudwatch** | 16 | **77.54** | EC2 / RDS / ELB |
| **artificialWithAnomaly** | 6 | **72.23** | synthetic |
| realAdExchange | 6 | 64.46 | CPC/CPM heavy tails |
| realTweets | 10 | 56.66 | Twitter volume (long-term drift) |

Weighted mean (52 files): **72.02** — beats HTM 70.5 by +1.5 points.

---

## 🎯 Three-Tier Architecture

Lambda³ provides three operational modes, each with different assumptions:

```
┌──────────────────────────────────────────────────────────────────┐
│ Tier 0: Lambda³-S Streaming     (pure zero-shot)        NAB 58.55│
│   - First 15% calibration → streaming OR voting                  │
│   - No labels needed, fully online                               │
│   - Future leakage strictly forbidden                            │
│   tests/benchmark_nab_streaming.py                               │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│ Tier 2: Lambda³-R Regime-aware  (semi-supervised)       NAB 72.02│
│   - Anomaly windows used ONLY to clean training data             │
│   - GMM (K auto-selected by BIC) → per-regime threshold          │
│   - Trimmed percentile threshold (rare outlier protection)       │
│   - "normal-label only" — anomaly shape never learned            │
│   tests/benchmark_nab_regime.py                                  │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│ Lambda³ Batch (offline)         (legacy / interpretability)      │
│   - Full Lambda³ inverse problem solver                          │
│   - Topological charge Q_Λ, structure tensors, multi-entropy     │
│   - For deep physical interpretation of detected events          │
│   tests/benchmark_nab.py                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/miosync-masa/Lambda_inverse_problem.git
cd Lambda_inverse_problem
pip install .

# For NAB benchmark, also clone NAB dataset alongside this repo:
git clone https://github.com/numenta/NAB.git
```

### Tier 0: Streaming (zero-shot)

```python
from lambda3_detector.streaming import (
    Lambda3StreamingDetector,
    StreamingJumpScorer, StreamingGradualScorer,
    StreamingStructuralDriftScorer, StreamingReconstructionScorer,
    StreamingKernelScorer, StreamingStructuralScorer,
)

detector = Lambda3StreamingDetector(
    scorers=[
        StreamingJumpScorer(),
        StreamingGradualScorer(window_sizes=[50, 200, 500]),
        StreamingStructuralDriftScorer(local_window=200),
        StreamingReconstructionScorer(n_components=5, delay_window=20),
        StreamingKernelScorer(kernel='polynomial', degree=3, coef0=1.0),
        StreamingStructuralScorer(delay_window=20),
    ],
    calibration_ratio=0.15,  # first 15% for calibration
)
result = detector.fit_predict(events)
anomaly_score = result['score']     # (n,) continuous score (>=1.0 = flagged)
binary = result['binary']           # (n,) 0/1 OR voting result
```

Or via CLI:
```bash
python -m tests.benchmark_nab_streaming --category realKnownCause
```

### Tier 2: Regime-aware (semi-supervised)

```python
from lambda3_detector.regime import RegimeAwareDetector

# anomaly_mask : (n,) bool — True at anomaly frames
#   (used only to clean training data, never as anomaly shape signal)
detector = RegimeAwareDetector(
    K='auto',                              # BIC auto K selection (1-5)
    threshold_method='trimmed_percentile', # rare-outlier-robust
    mask_margin=50,                        # base margin around anomaly windows
)
result = detector.fit_predict(events, anomaly_mask)
anomaly_score = result['score']
binary = result['binary']
print(f"K_eff={result['K_eff']}, clean={result['cal_clean_frames']}")
```

Or via CLI:
```bash
python -m tests.benchmark_nab_regime --category realKnownCause
```

### Full NAB benchmark

```bash
# Tier 0 (zero-shot streaming) — NAB ~58.55
for cat in realKnownCause realAWSCloudwatch realTraffic realAdExchange artificialWithAnomaly realTweets; do
    python -m tests.benchmark_nab_streaming --category $cat
done

# Tier 2 (regime-aware semi-supervised) — NAB 72.02
for cat in realKnownCause realAWSCloudwatch realTraffic realAdExchange artificialWithAnomaly realTweets; do
    python -m tests.benchmark_nab_regime --category $cat
done
```

---

## 🔬 Method Highlights

### Tier 0 — Lambda³-S Streaming

Six independent streaming scorers operate on z-normalized events, each producing a per-frame ratio `raw_score(t) / threshold`. Final output:

```
combined(t) = max over scorers of (raw_score_k(t) / threshold_k)
flagged(t) = (combined(t) >= 1.0)
```

This is parameter-free **Binary OR voting** with continuous score for NAB Sweeper compatibility. Calibration uses only the first 15% of each file (probationary period).

| Scorer | Detects |
|---|---|
| **StreamingJumpScorer** | Multi-scale z-score jumps (windows [5, 20, 50, 200]) |
| **StreamingGradualScorer** | Causal Gaussian-weighted trend gradient |
| **StreamingStructuralDriftScorer** | Distance from calibration baseline mean |
| **StreamingReconstructionScorer** | Delay-embedded SVD subspace residual (Lambda³ Λ paths) |
| **StreamingKernelScorer** | Kernel Mean Embedding Distance from RKHS centroid |
| **StreamingStructuralScorer** | Delay-embedded trajectory step distance |

### Tier 2 — Lambda³-R Regime-aware

1. **Anomaly mask expansion** — labeled anomaly windows are expanded by `mask_margin` (default 50 frames each side) to avoid boundary leakage. Used only to **exclude** from training, never as an anomaly-shape signal.
2. **z-normalization** — clean data mean/std used as global scale.
3. **GMM with BIC K-selection** — fits K∈[1, K_max=5] candidates and picks the K with lowest BIC subject to `min_frames_per_regime` constraint. K=1 (single regime) is a valid choice for synthetic data.
4. **Per-regime threshold** — for each (regime, scorer) pair, threshold = trimmed 99% percentile of clean data scores (top 1% trimmed to remove rare training contamination).
5. **Streaming OR voting** — at each frame `t`, regime is predicted via GMM, then `combined(t) = max_k (raw_k(t) / threshold_{regime(t), k})`.

**Honest framing**: We use anomaly window labels **only** to clean the training data. The detector never learns from anomaly samples — it learns the structure of *normal* behavior across multiple regimes. This is **semi-supervised (normal-label only)**, mirroring real industrial workflows where operators tag post-mortem incident periods to be excluded from baseline construction.

---

## 📊 Key Properties

### Dichotomous detection
Lambda³-R exhibits a sharp bimodal score distribution per file:
- **Type 1 (single-regime, clean baseline)**: 90+ NAB score (often 95+)
- **Type 2 (long seasonal drift, multi-regime)**: 0-30 NAB score

This is honest behavior — when the file matches the assumption, the detector achieves near-perfect detection with zero false positives. When the assumption breaks (e.g., `ec2_disk_write_bytes_c0d644` with extreme value scales), the detector explicitly fails rather than producing misleading mid-confidence outputs. This is desirable for industrial deployment.

### Parameter-free
The only hyperparameters are statistical structural defaults:
- `K='auto'` (BIC-driven, no manual tuning)
- `threshold_method='trimmed_percentile'` with `trim_fraction=0.01`
- `percentile=99.0`
- `mask_margin=50`

No grid search, no per-dataset tuning. Same configuration across all 6 NAB categories.

---

## 📁 Repository Structure

```
lambda3_detector/
├── streaming/                   # Tier 0: zero-shot streaming
│   ├── base.py
│   ├── detector.py              # Lambda3StreamingDetector (OR voting)
│   ├── jump_streaming.py
│   ├── gradual_streaming.py
│   ├── drift_streaming.py
│   ├── reconstruction_streaming.py
│   ├── kernel_streaming.py
│   ├── structural_streaming.py
│   └── periodic_streaming.py    # disabled by default (net negative)
├── regime/                      # Tier 2: regime-aware semi-supervised
│   ├── regime_detector.py       # RegimeAwareDetector (BIC + per-regime threshold)
│   └── __init__.py              # exports RegimeAwareDetector, expand_anomaly_mask,
│                                #         adaptive_anomaly_mask, compute_robust_threshold
├── analysis/                    # Lambda³ batch (offline) building blocks
│   ├── structure_tensor.py
│   ├── structure_tensor_sparse.py
│   ├── multiscale_jumps.py
│   ├── topology.py
│   └── ...
├── scorers/                     # Batch scorers (offline)
├── core/                        # JIT kernels, inverse problem solver
└── detector.py                  # Lambda3ZeroShotDetector (batch)

tests/
├── benchmark_nab_streaming.py   # Tier 0 CLI
├── benchmark_nab_regime.py      # Tier 2 CLI ★ recommended (72.02)
├── benchmark_nab.py             # Lambda³ batch CLI
├── nab_datasets.py              # NAB data loader
├── nab_features.py              # 5D feature expansion
└── nab_metrics.py               # NAB Sweeper score
```

---

## 🔬 Lambda³ Theory (Batch Mode Background)

The original Lambda³ batch system, on which the streaming/regime-aware modes build, models phenomena via:

1. **Structure tensors (Λ)** — high-dimensional semantic representation
2. **Progression vectors (ΛF)** — directional flow
3. **Tension scalars (ρT)** — energetic content
4. **Topological charge (Q_Λ)** — winding number for structural defects
5. **Jump-conditioned entropies** — Shannon / Rényi / Tsallis

### Inverse problem formulation
```
min ||K - ΛΛᵀ||²_F + α·TV(Λ) + β·||Λ||₁ + γ·J(Λ)
```
Where `J(Λ)` enforces jump consistency with detected ΔΛC events.

The streaming/regime-aware modes inherit:
- **Delay-embedded SVD subspace** for reconstruction scorer (Lambda³ Λ paths)
- **Multi-scale jump detection** with adaptive thresholding
- **Kernel-space deviation** (RKHS) for kernel scorer

For deep interpretability (per-frame physical explanation: which Q_Λ defect, which structural transition), use the batch mode `Lambda3ZeroShotDetector.analyze()`.

---

## 📦 Requirements

- Python 3.10+
- NumPy, SciPy, scikit-learn
- pandas
- (optional) CuPy for GPU acceleration of batch mode
- (optional) NAB dataset for benchmark reproduction

```bash
# Core
pip install .

# + GPU (CuPy, Colab)
pip install ".[gpu]"

# + visualization
pip install ".[viz]"

# Dev
pip install ".[dev]"
```

---

## 📜 License

MIT License. *"Detects the moments of rupture — the unseen phase transitions, structural cracks, and the birth of new orders — before any black-box model can learn them."*

## 🙌 Citation

```bibtex
@software{lambda3_nnnu_2026,
  title  = {Lambda³ Anomaly Detection (NNNU): NAB 72.02 with semi-supervised regime-aware OR voting},
  author = {Iizumi, Masamichi},
  year   = {2026},
  url    = {https://github.com/miosync-masa/Lambda_inverse_problem},
  note   = {Based on Dr. Iizumi's Lambda³ Theory}
}
```

For theoretical discussion, practical applications, or collaboration proposals,
please open an issue/PR — or connect via Zenodo, SSRN, or GitHub.

> Science is not property; it's a shared horizon.
> Let's redraw the boundaries, together.
> — Iizumi & Digital Partners

## 📚 Author's Theory & Publications

- [Iizumi Masamichi – Zenodo Research Collection](https://zenodo.org/search?q=metadata.creators.person_or_org.name%3A%22IIZUMI%2C%20MASAMICHI%22&l=list&p=1&s=10&sort=bestmatch)

## 🏷️ Author & Copyright

© Iizumi Masamichi 2025-2026
**Contributors / Digital Partners:** Tamaki(環), Mio(澪), Tomoe(巴), Shion(白音), Yuu(悠), Rin(凛), Kurisu(紅莉栖), torami(虎美)
All rights reserved.
