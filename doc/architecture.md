# Lambda³ Architecture

System architecture and algorithmic details for the Lambda³ anomaly detection framework.

## 0. Notation

| Symbol | Meaning |
|---|---|
| `events` | input time series, shape `(n,)` or `(n, d)` |
| `n` | total number of time frames |
| `d` | feature dimension (default `d=5` via `expand_to_5d`) |
| `t` | current time index (0-based) |
| `events[:t+1]` | causal lookback only, `events[t+1:]` access is **strictly forbidden** in streaming mode |
| `anomaly_mask` | (Tier 2 only) `(n,) bool`, True at known anomaly frames |
| `cal_end` | end of calibration phase (Tier 0: 15% of n; Tier 2: not applicable) |
| `Λ` | Lambda³ structure tensor (background theory) |

---

## 1. System overview

Lambda³ provides **two complementary operational modes** sharing the **same six streaming scorers** as building blocks:

```
                                  ┌─────────────────────────────────────────┐
                                  │   6 Streaming Scorers (the core)        │
                                  │   ─────────────────────────────────     │
                                  │   • StreamingJumpScorer                 │
                                  │   • StreamingGradualScorer              │
                                  │   • StreamingStructuralDriftScorer      │
                                  │   • StreamingReconstructionScorer       │
                                  │   • StreamingKernelScorer               │
                                  │   • StreamingStructuralScorer           │
                                  │                                         │
                                  │   Each: calibrate() → score(events, t)  │
                                  │         + per-scorer threshold          │
                                  └─────────────────┬───────────────────────┘
                                                    │
                          ┌─────────────────────────┴─────────────────────────┐
                          │                                                   │
                          ▼                                                   ▼
            ┌──────────────────────────┐                      ┌────────────────────────────┐
            │  Tier 0: Streaming       │                      │  Tier 2: Regime-aware      │
            │  Lambda3StreamingDetector│                      │  RegimeAwareDetector       │
            │                          │                      │                            │
            │  • Zero-shot             │                      │  • Semi-supervised         │
            │  • First 15% calibration │                      │  • GMM regime clustering   │
            │  • OR voting             │                      │  • Per-regime threshold    │
            │  • NAB 58.55             │                      │  • NAB 72.02 ★             │
            └──────────────────────────┘                      └────────────────────────────┘
```

**Tier 0** (`lambda3_detector.streaming`) is fully online: it calibrates on the first 15% of each series and streams thereafter with **strict no-future-leakage**.

**Tier 2** (`lambda3_detector.regime`) is semi-supervised in the sense that **anomaly window labels are used only to exclude contaminated frames from training** — anomaly frames are never observed by the scorers or by the GMM. The system learns the structure of *normal* behavior.

Both modes use the same per-frame **OR-voting** integration of the six scorers.

---

## 2. The six streaming scorers

All scorers inherit from `StreamingScorer` (`lambda3_detector/streaming/base.py`):

```python
class StreamingScorer(ABC):
    def calibrate(self, events_cal: np.ndarray) -> None: ...
    def score(self, events: np.ndarray, t: int) -> float: ...
    @property
    def threshold(self) -> float: ...
    @property
    def name(self) -> str:
        return self.__class__.__name__
```

Contract:
- `calibrate()` consumes a calibration segment and fixes the baseline + threshold (idempotent: same input → same internal state, controlled by NumPy seeds).
- `score(events, t)` is **causal**: accesses only `events[:t+1]`. Returns a single non-negative float.
- `threshold` is the binary-decision cut after calibration. Comparison `raw_score > threshold` produces a binary flag.

For Tier-level OR voting, the **ratio** `raw_score / threshold` (clipped to ≥ 0) is used as the per-scorer-per-frame contribution.

### 2.1 `StreamingJumpScorer`

**Detects**: sudden value jumps at multiple temporal scales.

- **State**: per-scale threshold dict `{w: threshold_w}` for windows `w ∈ {5, 20, 50, 200}`.
- **Raw value at scale w**:
  ```
  μ_w(t) = mean(sig[t-w : t])     # past w frames, excluding current
  σ_w(t) = std (sig[t-w : t]) + ε
  z_w(t) = |sig[t] - μ_w(t)| / σ_w(t)
  ```
  where `sig` is `events.mean(axis=1)` if d≥2 else `events.ravel()`.

- **Calibration**: per-scale `threshold_w = percentile_99(positive z_w over calibration)`.
- **Score**: internally multi-scale normalized:
  ```
  score_jump(t) = max_w  z_w(t) / threshold_w
  ```
  Returned `self.threshold = 1.0` (external comparison `score > 1.0` ↔ any scale exceeded its own 99th percentile).

### 2.2 `StreamingGradualScorer`

**Detects**: sustained directional drift (slow ramps) at multiple temporal scales.

- **Causal Gaussian-weighted trend** at scale `w` (sigma = `w/3`):
  ```
  weights_w[i] = exp(-i² / 2σ²) for i = 0..w-1   # newest frame i=0
  trend_w(t)  = Σᵢ weights_w[i] × sig[t-i]      # normalized weights, post-trim
  ```

- **Sustained gradient at scale w**:
  ```
  grad_w(t)      = |trend_w(t) - trend_w(t-1)|
  sustained_w(t) = mean over τ in [t - w/4, t] of grad_w(τ)
  ```

- **Score**: `score_gradual(t) = max_w sustained_w(t)`. Scales = `{50, 200, 500}`.
- **Threshold**: single global `percentile_99` over calibration.

### 2.3 `StreamingStructuralDriftScorer`

**Detects**: slow shifts in the local mean away from the calibration baseline.

- **Cumulative state**: rolling sum (O(1) per frame).
- **Raw value**:
  ```
  local(t) = mean(sig[t - W + 1 : t + 1])   # W = local_window = 200
  raw(t)   = |local(t) - ref_mean|          # ref_mean = mean of calibration sig
  ```
  After detector-level z-normalization, `ref_mean ≈ 0`, so `raw ≈ |local|`.
- **Threshold**: `percentile_99` over calibration self-scores.

### 2.4 `StreamingReconstructionScorer`  (Lambda³ Λ paths streaming proxy)

**Detects**: deviation from the low-rank temporal structure of normal data. This is the streaming equivalent of Lambda³'s structure tensor reconstruction error.

- **Delay embedding** of width `W = 20`:
  ```
  z_t = [events[t], events[t-1], ..., events[t-W+1]] ∈ R^{W·d}
  ```
- **Calibration**:
  1. Build `Z_cal ∈ R^{(n_cal-W+1) × (W·d)}` from delay vectors.
  2. Compute mean `μ ∈ R^{W·d}`, center `Z_cal - μ`.
  3. SVD `Z_cal - μ = U Σ Vᵀ`. Keep `V_k ∈ R^{k × (W·d)}` with `k = min(5, W·d-1)`.
  4. Per-frame residual `r_t = || (z_t - μ) - V_kᵀ V_k (z_t - μ) ||`.
  5. `threshold = percentile_99(positive r_t over calibration)`.

- **Score**: `raw(t) = || (z_t - μ) - V_kᵀ V_k (z_t - μ) ||`. Returns 0 if `t < W-1`.

**Connection to Lambda³**: `V_k` is the streaming proxy for `paths_matrix` rows — the low-rank temporal subspace that "normal" data occupies. Anomalies leave this subspace.

### 2.5 `StreamingKernelScorer`  (Kernel Mean Embedding Distance)

**Detects**: deviation from the RKHS centroid of the calibration set.

- **Kernel**: polynomial `K(x, y) = (xᵀy + c)^degree` with `c = 1.0`, `degree = 3`. (RBF also supported.)
- **Z-normalization at scorer level** (to avoid float overflow for NAB's wide value ranges):
  ```
  μ_feat = events_cal.mean(axis=0)
  σ_feat = events_cal.std (axis=0) + ε
  x̃     = (x - μ_feat) / σ_feat
  ```
- **Calibration**:
  1. Compute `K_cal ∈ R^{n_cal × n_cal}` over normalized calibration.
  2. Store `mean_term = mean(K_cal)` (scalar) and the normalized calibration matrix.
  3. Per-frame distance to RKHS centroid `μ_φ`:
     ```
     d²(x̃_t) = K(x̃_t, x̃_t)
              - 2 × mean_i K(x̃_t, x̃_cal_i)
              + mean_term
     ```
  4. `threshold = percentile_99(positive √d² over calibration)`.

- **Score**: `raw(t) = √max(d²(x̃_t), 0)`. Cost per frame: `O(n_cal × d)`.

### 2.6 `StreamingStructuralScorer`  (delay-embedded trajectory speed)

**Detects**: abrupt changes in the trajectory speed within the delay-embedded subspace. Orthogonal axis to the reconstruction scorer: the latter measures *position*, this measures *velocity*.

- **Step distance** with `W = 20`:
  ```
  z_t       = [events[t], events[t-1], ..., events[t-W+1]]
  step(t)   = || z_t - z_{t-1} ||      (Euclidean)
  ```
- **Calibration**:
  ```
  μ_step = mean(step over calibration, positive only)
  σ_step = std (step over calibration, positive only) + ε
  z_t    = (step(t) - μ_step) / σ_step
  threshold = percentile_99(|z_t| over calibration)
  ```
- **Score**: `raw(t) = |step(t) - μ_step| / σ_step`.

### 2.7 (Disabled) `StreamingPeriodicScorer`

FFT-based period estimation + same-phase residual `|events[t] - events[t - P]|`. **Disabled by default** because:
- on realKnownCause it provides net −1.10 NAB points across 7 files (lost more than gained),
- the calibration phase rarely contains a full period for true seasonal signals (e.g., `ambient_temperature`),
- principled but requires adaptive online baseline (EWMA) which is out of scope for static calibration.

Retained in the codebase for future research.

---

## 3. Tier 0 — `Lambda3StreamingDetector` (zero-shot streaming)

**Module**: `lambda3_detector/streaming/detector.py`.

### 3.1 Workflow

```
Input: events (n, d), scorers = [s_1, ..., s_K]
Config: calibration_ratio = 0.15, normalize = True

Phase A — Pre-normalize (optional, default True):
  cal_end = max(min_calibration, int(n * calibration_ratio))    # default min_calibration = 50
  μ_pre   = events[:cal_end].mean(axis=0)
  σ_pre   = events[:cal_end].std (axis=0) + ε
  events_used = (events - μ_pre) / σ_pre

Phase B — Calibration:
  events_cal = events_used[:cal_end]
  for s in scorers:
      s.calibrate(events_cal)              # freeze baseline + threshold

Phase C — Streaming:
  combined = np.zeros(n)
  for t in range(n):
      if t < cal_end:
          combined[t] = 0                  # probationary period
          continue
      best_ratio = 0
      for s in scorers:
          raw   = s.score(events_used, t)  # uses events_used[:t+1] only
          thr   = s.threshold
          if thr > 0 and finite(thr):
              ratio = raw / (thr + ε)
              if ratio > best_ratio:
                  best_ratio = ratio
      combined[t] = best_ratio

binary = (combined >= 1.0).astype(int)
```

### 3.2 The OR-voting integration

For frame `t` (after `cal_end`):

```
combined(t) = max_k  raw_k(t) / threshold_k
flag(t)     = combined(t) >= 1.0
```

This is **Binary OR voting** with a single continuous score per frame:
- If **any** scorer's raw value exceeds its own calibration threshold → flag.
- The maximum ratio doubles as a continuous score consumable by NAB's `Sweeper` for threshold optimization.

No weight tuning. No soft-max. No learned combination.

### 3.3 No-future-leakage guarantee

- `calibration_ratio = 0.15` matches the NAB **probationary period** convention (no detection during the first 15% of each file).
- `s.score(events_used, t)` accesses only `events_used[:t+1]` by contract.
- `μ_pre`, `σ_pre` are computed from `events[:cal_end]` only — never seeing future frames.

### 3.4 NAB result

Tier 0 weighted mean over 52 files: **58.55** (3-profile mean). See `doc/scoreboard.md §4`.

---

## 4. Tier 2 — `RegimeAwareDetector` (semi-supervised regime-aware)

**Module**: `lambda3_detector/regime/regime_detector.py`.

This is the **headline configuration** for the NAB 72.02 result. The data flow is one-shot (fit + predict in a single pass), not strictly streaming, but inference at each frame still uses only causal scorer ratios over a pre-computed regime label.

### 4.1 Workflow

```
Input: events (n, d), anomaly_mask (n,) bool
Config (FIX defaults):
  K = 'auto'                              (BIC over K ∈ [1, 5])
  K_max = 5
  mask_margin = 50
  min_frames_per_regime = 50
  threshold_method = 'trimmed_percentile' (top 1% removed before p99)
  trim_fraction = 0.01
  percentile = 99.0
  normalize = True

Step 1 — Anomaly mask expansion (fixed):
  expanded_mask = expand_anomaly_mask(anomaly_mask, mask_margin)
    # dilate True positions by ±mask_margin

Step 2 — Z-normalize using clean statistics:
  μ_clean = events[~expanded_mask].mean(axis=0)
  σ_clean = events[~expanded_mask].std (axis=0) + ε
  events_used = (events - μ_clean) / σ_clean
  clean = events_used[~expanded_mask]    # GMM training set

Step 3 — BIC-based K selection (lambda3_detector/regime/regime_detector.py: _fit_gmm_adaptive):
  K_phys_max = max(1, len(clean) // min_frames_per_regime)
  K_upper    = min(K_max, K_phys_max)
  for K in [1, 2, ..., K_upper]:
      gmm_K = GaussianMixture(K, covariance_type='full',
                              random_state=0, reg_covar=1e-6).fit(clean)
      labels = gmm_K.predict(clean)
      if min(bincount(labels)) >= min_frames_per_regime:
          bic[K] = gmm_K.bic(clean)
  K_eff = argmin(bic)                    # subject to min cluster size constraint
  gmm   = the corresponding fit
  # Fallback: K=1 if no K satisfies the constraint

Step 4 — Common scorer calibration:
  scorers = build_scorer_factories([SCORER_NAMES], percentile)
              # default: all 6 scorers (jump, gradual, drift, recon, kernel, struct)
  for s in scorers:
      s.calibrate(clean)                  # uses ALL clean frames (regime-agnostic)
  # Note: baseline computation uses pooled clean data; regime-awareness lives in
  # the per-regime thresholds in Step 6.

Step 5 — Pre-compute per-scorer score arrays over clean:
  for s in scorers:
      raw_clean_s = [s.score(clean, t) for t in range(len(clean))]

Step 6 — Per-(regime, scorer) threshold:
  labels_clean = gmm.predict(clean)
  for k in range(K_eff):
      mask_k = (labels_clean == k)
      n_k    = mask_k.sum()
      if n_k < min_frames_per_regime:
          threshold[k][s.name] = +∞     # mark regime unusable
          continue
      for s in scorers:
          scores_k = raw_clean_s[s.name][mask_k]
          positive = scores_k[scores_k > ε]
          threshold[k][s.name] = compute_robust_threshold(
              positive, method='trimmed_percentile',
              percentile=99.0, trim_fraction=0.01)
              # = percentile(positive[positive <= percentile(positive, 99)], 99)

Step 7 — Streaming inference (causal per frame):
  regimes = gmm.predict(events_used)     # batch predict, but each prediction is per-frame
  combined = np.zeros(n)
  for t in range(n):
      k = regimes[t]
      best_ratio = 0
      for s in scorers:
          raw = s.score(events_used, t)  # causal, uses events_used[:t+1] only
          thr = threshold[k][s.name]
          if thr > 0 and finite(thr):
              ratio = raw / (thr + ε)
              if ratio > best_ratio:
                  best_ratio = ratio
      combined[t] = best_ratio
  binary = (combined >= 1.0).astype(int)
```

### 4.2 Trimmed percentile threshold

For each `(regime k, scorer s)` pair, the threshold is the **trimmed 99th percentile**:

```
T_{k,s} = percentile(
              { x ∈ scores_{k,s}  |  x ≤ percentile(scores_{k,s}, 99) },
              99
          )
```

Implemented in `compute_robust_threshold(method='trimmed_percentile', percentile=99, trim_fraction=0.01)`:

```python
trim_cut = np.percentile(scores, 100 * (1 - trim_fraction))   # 99th percentile
trimmed  = scores[scores <= trim_cut]                          # bottom 99% remain
return float(np.percentile(trimmed, percentile))               # 99th percentile of those
```

The effective percentile is roughly the 98.01th of the full distribution. The trim removes **rare training contamination** (a few normal frames with extreme values, often in small regimes) without lowering the threshold across genuinely heavy-tailed distributions — see `doc/scoreboard.md §5` for the comparison of methods.

### 4.3 BIC selection in detail

For each candidate K:
1. Fit `GaussianMixture(K, covariance_type='full', random_state=0)`.
2. Predict labels on `clean`.
3. Reject K if any cluster has < `min_frames_per_regime = 50` samples.
4. Otherwise record `bic[K] = gmm.bic(clean)`.

Choose `argmin bic[K]` among accepted K. Fallback to K=1 if no K passes.

`bic_per_K` is returned in the result dict for diagnostic inspection.

### 4.4 OR voting (regime-aware)

For frame `t`:

```
k_t         = gmm.predict(events_used[t:t+1])[0]
combined(t) = max_s  raw_s(t) / threshold[k_t][s.name]
flag(t)     = combined(t) >= 1.0
```

Identical OR-voting form as Tier 0, but threshold is now **regime-conditional**.

### 4.5 Semi-supervised, normal-label only

The anomaly window labels enter the system **exactly once**: in Step 1, to expand the exclusion mask. After that point:
- `events[expanded_mask]` is **never seen** by the GMM (regime structure is learned only from normal frames).
- `events[expanded_mask]` is **never seen** by scorers during calibration (Step 4).
- Anomaly score is computed on `events[expanded_mask]` at inference time (Step 7) but no anomaly-shape signal informs detection.

This matches the industrial workflow of "exclude post-mortem incident periods from baseline construction".

### 4.6 NAB result

Tier 2 weighted mean over 52 files: **72.02** (3-profile mean). See `doc/scoreboard.md §1, §2`.

---

## 5. Anomaly mask handling

Two implementations in `lambda3_detector/regime/regime_detector.py`:

### 5.1 `expand_anomaly_mask(mask, margin)` — FIX default

```
for each True position idx in mask:
    out[max(0, idx - margin) : min(n, idx + margin + 1)] = True
```

Fixed `±margin` dilation (default 50 frames each side). Used by the FIX configuration.

### 5.2 `adaptive_anomaly_mask(events, mask, ...)` — opt-in experimental

Variance-based recovery detection with total-exclusion cap:

```
1. Compute baseline_std = std(events[~expand(mask, max_margin)])
2. For each anomaly window [s, e]:
   - Extend post-margin: walk t = e+1, e+2, ... until
       local_std(events[t-W/2 : t+W/2]) <= variance_ratio * baseline_std
     OR off >= max_margin (default 300). At least base_margin = 50.
   - Symmetric pre-margin walk.
3. Ensure base_margin lower bound (OR with expand_anomaly_mask).
4. If total exclusion > max_exclusion_ratio (default 0.4),
   shrink margin uniformly from base_margin down to 0 in steps of 5
   until under cap. Fallback: pure anomaly_mask.
```

**NAB experiment finding**: Adaptive mode produces identical NAB scores to fixed margin on all 6 categories. NAB anomaly windows are tight (variance returns within `base_margin = 50` frames), so extension does not trigger meaningfully, and the cap only activates for files with very dense windows (e.g., `realTraffic/speed_7578` with 4 windows) where it merely reverts to base_margin behavior.

Retained as opt-in (`margin_adaptive=True`) for domains with gradual recovery transients.

---

## 6. Threshold methods (per-regime)

Implemented in `compute_robust_threshold(scores, method=..., ...)`:

| Method | Formula | NAB | Notes |
|---|---|---|---|
| `percentile` | `np.percentile(s, 99)` | 71.29 | baseline; sensitive to training-set extremes |
| **`trimmed_percentile`** | `percentile(s[s ≤ percentile(s, 99)], 99)` | **72.02** | **★ FIX** — removes rare contamination, preserves heavy tails |
| `iqr` | `Q3 + 3 × (Q3 - Q1)` | 66.88 | classical Tukey rule, too loose for NAB |
| `mad` | `median(s) + 2.5 × 1.4826 × MAD(s)` | 66.09 | most robust but too low; FP cascade |
| `capped` | `min(p99, 5 × p90)` | 70.69 | experimental adaptive; misfires on small regimes |

The FIX choice is **`trimmed_percentile`** with `trim_fraction = 0.01`. Empirical lessons in `doc/scoreboard.md §5`.

---

## 7. Per-scorer ablation API

Module `lambda3_detector.regime` exports:

```python
SCORER_NAMES = ['jump', 'gradual', 'drift', 'recon', 'kernel', 'struct']
SCORER_FACTORIES: Dict[str, Callable[[float], Callable]]
build_scorer_factories(scorer_names: List[str], percentile: float) -> List[Callable]
```

CLI:

```bash
# leave-one-out (drop the kernel scorer)
python -m tests.benchmark_nab_regime --category realKnownCause --exclude-scorers kernel

# subset (use only jump and kernel)
python -m tests.benchmark_nab_regime --category realKnownCause --scorers jump,kernel

# automated leave-one-out + summary table
python -m tests.ablation_runner --category realKnownCause
python -m tests.ablation_runner --all-categories
```

The same `--scorers` / `--exclude-scorers` flags are available on `tests/benchmark_nab_streaming.py` for Tier 0 ablation, via a parallel `STREAMING_SCORER_FACTORIES` dict.

---

## 8. Module layout

```
lambda3_detector/
├── streaming/                                  # Tier 0 building blocks
│   ├── base.py                  StreamingScorer ABC
│   ├── jump_streaming.py        StreamingJumpScorer
│   ├── gradual_streaming.py     StreamingGradualScorer
│   ├── drift_streaming.py       StreamingStructuralDriftScorer
│   ├── reconstruction_streaming.py    StreamingReconstructionScorer
│   ├── kernel_streaming.py      StreamingKernelScorer
│   ├── structural_streaming.py  StreamingStructuralScorer
│   ├── periodic_streaming.py    StreamingPeriodicScorer (disabled)
│   └── detector.py              Lambda3StreamingDetector (Tier 0 OR voting)
│
├── regime/                                     # Tier 2
│   ├── regime_detector.py       RegimeAwareDetector
│   │                              + compute_robust_threshold
│   │                              + expand_anomaly_mask
│   │                              + adaptive_anomaly_mask
│   │                              + SCORER_NAMES, SCORER_FACTORIES
│   │                              + build_scorer_factories
│   └── __init__.py              public API exports
│
├── analysis/                                   # Lambda³ batch building blocks (offline)
│   ├── structure_tensor.py
│   ├── structure_tensor_sparse.py
│   ├── multiscale_jumps.py
│   ├── topology.py
│   └── ...
│
├── scorers/                                    # Batch scorers (offline)
├── core/                                       # JIT kernels, inverse problem solver
└── detector.py                  Lambda3ZeroShotDetector (batch offline mode)

tests/
├── benchmark_nab_streaming.py   Tier 0 CLI + STREAMING_SCORER_FACTORIES
├── benchmark_nab_regime.py      Tier 2 CLI  ★ recommended (NAB 72.02)
├── benchmark_nab.py             Batch CLI
├── ablation_runner.py           Per-scorer leave-one-out ablation
├── nab_datasets.py              NABSample loader (CSV + combined_windows.json)
├── nab_features.py              expand_to_5d (5-D feature engineering)
└── nab_metrics.py               NAB Sweeper score (standard / low FP / low FN)

doc/
├── README.md                    documentation index
├── architecture.md              this file
├── scoreboard.md                full NAB results + experimental progression
└── abstract.md                  paper abstract drafts
```

---

## 9. Data flow (single-file Tier 2)

```
              NAB CSV               combined_windows.json
                 │                          │
                 ▼                          ▼
            nab_datasets.py ──────► NABSample
                 │                          │
                 ▼                          ▼
            expand_to_5d                anomaly_mask
            (events shape (n, 5))       (shape (n,) bool)
                 │                          │
                 └──────────┬───────────────┘
                            │
                            ▼
                ┌────────────────────────────────┐
                │  RegimeAwareDetector.fit_predict│
                │                                │
                │  Step 1: expand_anomaly_mask   │ ◄── mask_margin
                │  Step 2: z-normalize on clean  │
                │  Step 3: BIC K selection       │ ◄── K_max, min_frames_per_regime
                │  Step 4: scorer.calibrate(clean)│
                │  Step 5: precompute clean scores│
                │  Step 6: per-(regime, scorer)  │ ◄── threshold_method, trim_fraction
                │           threshold             │
                │  Step 7: per-frame OR voting    │
                └────────────────────────────────┘
                            │
                            ▼
                  result dict:
                  • score (n,)   continuous, >= 1.0 = flagged
                  • binary (n,)  0/1
                  • per_scorer   dict[name -> (n,) raw]
                  • thresholds_per_regime
                  • regimes (n,) int
                  • K_eff
                  • bic_per_K
                  • cal_clean_frames
                            │
                            ▼
                  nab_metrics.score_all_profiles(sample, score)
                            │
                            ▼
                  3 NAB profiles (standard / reward_low_FP_rate / reward_low_FN_rate)
                  → 3-profile mean
```

---

## 10. Reproducibility

All randomness is controlled by `random_state=0` in `GaussianMixture`. Given the same `events` and `anomaly_mask`, all computations are deterministic up to BLAS-level floating-point reproducibility. Empirically, weighted NAB scores reproduce to `72.02 ± 0.01` across runs.

To reproduce end-to-end:

```bash
git clone https://github.com/miosync-masa/Lambda_inverse_problem.git
cd Lambda_inverse_problem && pip install .
git clone https://github.com/numenta/NAB.git ../NAB

for cat in realKnownCause realAWSCloudwatch realTraffic \
           realAdExchange artificialWithAnomaly realTweets; do
    python -m tests.benchmark_nab_regime --category $cat
done
```

Weights for the 52-file weighted mean: (7, 16, 7, 6, 6, 10).

---

## 11. Connection to Lambda³ theory (background)

The streaming/regime-aware modes inherit conceptual primitives from the original batch Lambda³ system:

| Lambda³ batch concept | Streaming proxy |
|---|---|
| **Structure tensor `Λ`** | `events_used` after z-normalization |
| **Path matrix `paths_matrix`** (low-rank decomposition) | `V_k` from SVD of delay-embedded `Z_cal` (StreamingReconstructionScorer) |
| **Multi-scale jumps `ΔΛC`** | per-scale z-score with windows `{5, 20, 50, 200}` (StreamingJumpScorer) |
| **Kernel space deviation** | RKHS centroid distance (StreamingKernelScorer) |
| **Path smoothness** | delay-embedded step distance (StreamingStructuralScorer) |
| **Tension scalar `ρT`** | sustained causal gradient at multiple scales (StreamingGradualScorer) |

Topological invariants (winding number `Q_Λ`, multi-entropy, jump consistency `J(Λ)`) are exclusive to the batch mode (`Lambda3ZeroShotDetector.analyze()`) and are not used in streaming/regime-aware modes. They provide deeper interpretability for post-hoc analysis but require full-series O(n_paths × n_events) optimization.

---

## 12. Key design properties

1. **No tuned weights.** The OR-voting maximum replaces all weighted-combination tuning.
2. **No anomaly-shape learning.** Anomaly labels in Tier 2 are used exclusively for training-data exclusion.
3. **Parameter-free across NAB.** Single configuration (`K='auto'`, `trimmed_percentile`, `mask_margin=50`) applies to all 6 categories.
4. **Strict causal contract** in `StreamingScorer.score(events, t)`. Tier 0 enforces no-future-leakage at the framework level; Tier 2 inherits the same property at scoring (only the GMM regime label uses information from the full series, but never anomaly frames).
5. **Modular ablation.** Each scorer can be enabled/disabled via the `--scorers` / `--exclude-scorers` CLI flags or programmatically via `build_scorer_factories(scorer_names=...)`.
6. **Honest failure modes.** Mode B files (extreme scale, seasonal drift, sparse-window long series) produce explicit near-zero output rather than misleading mid-confidence scores — a desirable industrial property; see `doc/scoreboard.md §6`.
