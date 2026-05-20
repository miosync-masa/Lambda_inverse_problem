"""
Phase 2: 既存4 scorer (sparse default 後) に DriftScorer を 5番目として加えた効果を測る。

ベース重み (production):
    jump:0.20  hybrid:0.35  kernel:0.30  structural:0.15

drift_weight ∈ {0.00, 0.05, 0.10, 0.15, 0.20} を sweep し、他の4 scorer は
合計が 1 - drift_weight になるよう比例配分。シナリオごとに AUC を測る。

Usage::
    python -m tests.benchmark_drift_addition
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from lambda3_detector import L3Config, Lambda3ZeroShotDetector
from lambda3_detector.scorers import (
    DriftScorer,
    HybridScorer,
    JumpScorer,
    KernelScorer,
    ScoreIntegrator,
    StructuralScorer,
)

from tests.datasets import create_complex_natural_dataset


SCENARIOS = [
    'progressive_degradation',
    'periodic_burst',
    'chaotic_bifurcation',
    'partial_anomaly',
]

DRIFT_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.20]
BASE = {'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15}


def make_weights(drift_weight: float) -> Dict[str, float]:
    """drift_weight を取り、残りを既存比率で配分"""
    scale = 1.0 - drift_weight
    w = {k: v * scale for k, v in BASE.items()}
    w['drift'] = drift_weight
    return w


def compute_component_scores(events, result):
    np.random.seed(0); jump   = JumpScorer().score(events, result)
    np.random.seed(0); hybrid = HybridScorer().score(events, result)
    np.random.seed(0); kernel = KernelScorer(kernel_type=1, degree=7, coef0=1.0).score(events, result)
    np.random.seed(0); struct = StructuralScorer().score(events, result)
    np.random.seed(0); drift  = DriftScorer().score(events, result)
    return {'jump': jump, 'hybrid': hybrid, 'kernel': kernel,
            'structural': struct, 'drift': drift}


def benchmark(scenario: str, seed: int) -> Dict:
    np.random.seed(seed)
    events, labels, _ = create_complex_natural_dataset(
        n_events=300, n_features=10, anomaly_ratio=0.15, scenario_filter=scenario,
    )
    detector = Lambda3ZeroShotDetector(L3Config())  # sparse default
    np.random.seed(0); result = detector.analyze(events)
    components = compute_component_scores(events, result)

    aucs = {}
    for dw in DRIFT_WEIGHTS:
        w = make_weights(dw)
        integrator = ScoreIntegrator(default_weights=w)
        scores = integrator.combine(components)
        aucs[dw] = float(roc_auc_score(labels, scores))

    # DriftScorer単独AUCも参考に
    drift_auc = float(roc_auc_score(labels, components['drift']))

    return {'scenario': scenario, 'seed': seed, 'aucs': aucs, 'drift_alone': drift_auc}


def main():
    print("=" * 100)
    print("Phase 2: DriftScorer addition  (sparse default, 5 scorers, sweep drift_weight)")
    print("=" * 100)

    seeds = [42, 123]
    results: List[Dict] = []

    for scenario in SCENARIOS:
        print(f"\n■ scenario = {scenario}")
        for seed in seeds:
            r = benchmark(scenario, seed)
            results.append(r)
            line = '  '.join(f"dw={dw:.2f}:{r['aucs'][dw]:.4f}" for dw in DRIFT_WEIGHTS)
            print(f"  seed={seed:<4}  {line}  (drift alone={r['drift_alone']:.4f})")

    # 集計
    print("\n" + "=" * 100)
    print("シナリオ別 drift_weight sweep  (mean across 2 seeds)")
    print("=" * 100)
    header = f"  {'scenario':<26}"
    for dw in DRIFT_WEIGHTS:
        header += f"  dw={dw:.2f}".rjust(10)
    header += "  drift_alone".rjust(13)
    header += "  best_dw".rjust(10)
    print(header)
    print("-" * len(header))

    for scenario in SCENARIOS:
        subset = [r for r in results if r['scenario'] == scenario]
        if not subset:
            continue
        line = f"  {scenario:<26}"
        means = {}
        for dw in DRIFT_WEIGHTS:
            m = float(np.mean([r['aucs'][dw] for r in subset]))
            means[dw] = m
            line += f"  {m:.4f}".rjust(10)
        drift_alone_mean = float(np.mean([r['drift_alone'] for r in subset]))
        line += f"  {drift_alone_mean:.4f}".rjust(13)
        best_dw = max(means.items(), key=lambda x: x[1])[0]
        line += f"  {best_dw:.2f}".rjust(10)
        print(line)

    # 全シナリオ平均
    print("-" * len(header))
    line = f"  {'OVERALL (mean)':<26}"
    overall = {}
    for dw in DRIFT_WEIGHTS:
        m = float(np.mean([r['aucs'][dw] for r in results]))
        overall[dw] = m
        line += f"  {m:.4f}".rjust(10)
    drift_alone_overall = float(np.mean([r['drift_alone'] for r in results]))
    line += f"  {drift_alone_overall:.4f}".rjust(13)
    best_dw_overall = max(overall.items(), key=lambda x: x[1])[0]
    line += f"  {best_dw_overall:.2f}".rjust(10)
    print(line)


if __name__ == "__main__":
    main()
