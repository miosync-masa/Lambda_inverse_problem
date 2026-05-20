# `lambda3_detector` — Modular Lambda³ Anomaly Detection

This package is the split form of the old monolith `lambda3_detector_v2.py` (4108 lines, kept as a backup).
It preserves the public surface (`from lambda3_detector import Lambda3ZeroShotDetector, L3Config`) and adds
**independently usable scorers** so ablation experiments are no longer blocked by tight coupling.

## Quick start

```python
from lambda3_detector import Lambda3ZeroShotDetector, L3Config

detector = Lambda3ZeroShotDetector(L3Config())  # use_sparse_solver=True がデフォルト
result   = detector.analyze(events)             # structure-tensor + jumps + physics
scores   = detector.detect_anomalies(result, events)
```

> **Default solver = sparse** (since v0.2). Per-scenario benchmark で sparse は full に平均
> +0.05 AUC 勝つ。旧挙動が必要な場合は `L3Config(use_sparse_solver=False)` で legacy
> モードに戻せる（`tests/test_split_equivalence.py` で numerical equivalence 検証済み）。

## Use a single scorer

Each scorer satisfies the `AnomalyScorer` ABC and can be invoked standalone:

```python
from lambda3_detector import Lambda3ZeroShotDetector
from lambda3_detector.scorers import (
    JumpScorer, HybridScorer, KernelScorer, StructuralScorer, ScoreIntegrator,
)

result = Lambda3ZeroShotDetector().analyze(events)

jump   = JumpScorer().score(events, result)
hybrid = HybridScorer(alpha=0.3, w_topo=0.5).score(events, result)
kernel = KernelScorer(kernel_type=1, degree=7).score(events, result)   # Polynomial
struct = StructuralScorer().score(events, result)

# Combine them with explicit weights
combined = ScoreIntegrator(default_weights={
    'jump': 0.2, 'hybrid': 0.35, 'kernel': 0.3, 'structural': 0.15,
}).combine({'jump': jump, 'hybrid': hybrid, 'kernel': kernel, 'structural': struct})
```

## Module layout

```
lambda3_detector/
├── __init__.py              # public facade (Lambda3ZeroShotDetector, L3Config, …)
├── config.py                # dataclasses + mutable globals + percentile setters
├── core/                    # pure numerical kernels (JIT)
│   ├── adaptive_params.py   #   adaptive window-size inference
│   ├── jumps_jit.py         #   diff / threshold / local-std / sync profile
│   ├── topology_jit.py      #   topological charge Q_Λ
│   ├── entropy_jit.py       #   Shannon / Rényi / Tsallis + 6-vector
│   ├── kernels_jit.py       #   RBF / Poly / Sigmoid / Laplacian / Periodic + Gram
│   ├── pulsation_jit.py     #   pulsation energy (jumps & path)
│   └── inverse_problem_jit.py  # objectives + hybrid Tikhonov scores
├── analysis/                # pipelines built on core kernels
│   ├── multiscale_jumps.py     #  detect_multiscale_jumps[_with_params]
│   ├── structure_tensor.py     #  solve_inverse_problem[_jump_constrained]
│   └── physical_quantities.py  #  compute_topology / energies / entropies / classify
├── scorers/                 # ★ablation-ready★ anomaly scorers
│   ├── base.py              #   AnomalyScorer ABC
│   ├── jump_scorer.py
│   ├── hybrid_scorer.py
│   ├── kernel_scorer.py
│   ├── structural_scorer.py
│   └── score_integrator.py  #   weighted blend + aggressive DE optimizer
├── features/
│   ├── extractor.py         # Lambda3FeatureExtractor (basic + advanced + FFT)
│   └── optimizer.py         # Lambda3FeatureOptimizer (L1 + correlation)
├── detector.py              # Lambda3ZeroShotDetector — orchestration only
├── visualization.py         # visualize_results
└── io_utils.py              # save_results
```

## Dependency direction (no cycles)

```
config ──▶ core ──▶ analysis ──▶ scorers ──┐
                                            ├──▶ detector ──▶ visualization / io_utils
                              features ────┘
```

Every arrow points one way. `config` has no internal deps; `core` only imports `config`; and so on.
The `tests/` package consumes the public surface — it is never imported by production modules.

## How globals mutate

`update_global_constants` / `update_detection_percentiles` (both in `config.py`) live-mutate module
attributes. Analysis modules read them via attribute access (`from lambda3_detector import config; config.LOCAL_WINDOW_SIZE`)
so updates flow through. `@njit` functions that took the original values as Python *defaults* freeze
those defaults at compile time — that's intentional and was already the behavior of the monolith.

## Verifying the split

```bash
cd Lambda_inverse_problem
python -m pytest tests/test_split_equivalence.py -v
python multi_channel_main.py                       # existing call site still works
python -m tests.ablation                           # single + combined scorer AUCs
```

The equivalence test asserts identical paths, topological quantities, and `detect_anomalies` scores
between the new package and the original `lambda3_detector_v2.py` (kept as backup).

## Tests / evaluation (`../tests/`)

| File | Purpose |
| --- | --- |
| `anomaly_generators.py` | 11 anomaly-pattern generators (pulse, phase_jump, periodic, decay, bifurcation, multi_path, topological_jump, cascade, partial_periodic, superposition, resonance). Use `init_anomaly_patterns(detector)` to attach them. |
| `datasets.py`           | `create_complex_natural_dataset` — synthetic multi-cluster data with mixed anomaly scenarios. |
| `benchmark.py`          | `evaluate_performance` (AUC + Top-10). |
| `ablation.py`           | Single-scorer and all-combination AUC sweep. |
| `test_split_equivalence.py` | Numerical equivalence between monolith and package. |
