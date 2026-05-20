"""
V2 demo regime (n≈1000, n_features=20, anomaly=1.5%, 混合シナリオ) での再現比較。

V2 オリジナル demo (Basic Mode):
    Data shape: (999, 20)  Anomaly ratio: 1.50%
    AUC=0.9625  Top-10=0.90  Detection time≈30s  Analyze time≈155s

ここでは以下の3構成を比較:
  (A) full + production weights         ← V2 baseline 再現
  (B) sparse + production weights        ← Phase 1 default
  (C) sparse + Acute-alone (0.4·z_j + 0.6·z_h) ← Dual benchmark の Acute部分

Usage::
    python -m tests.benchmark_v2_regime
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from lambda3_detector import L3Config, Lambda3ZeroShotDetector
from lambda3_detector.scorers import (
    HybridScorer,
    JumpScorer,
    KernelScorer,
    ScoreIntegrator,
    StructuralScorer,
)

from tests.datasets import create_complex_natural_dataset


def _z(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    if s < 1e-10:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / s


def _topk(scores, labels, k=10):
    return float(np.mean(labels[np.argsort(scores)[-k:]]))


def compute_components(detector, result, events):
    np.random.seed(0); jump   = JumpScorer().score(events, result)
    np.random.seed(0); hybrid = HybridScorer().score(events, result)
    np.random.seed(0); kernel = KernelScorer(kernel_type=1, degree=7, coef0=1.0).score(events, result)
    np.random.seed(0); struct = StructuralScorer().score(events, result)
    return {'jump': jump, 'hybrid': hybrid, 'kernel': kernel, 'structural': struct}


def evaluate_config(config_name: str, paths_mode: str, seed: int) -> Dict:
    """1 (config × seed) を回す"""
    np.random.seed(seed)
    events, labels, anomaly_details = create_complex_natural_dataset(
        n_events=1000, n_features=20, anomaly_ratio=0.015,  # V2 demo regime
        # scenario_filter=None  → 4種混合（V2 オリジナル挙動）
    )

    use_sparse = (paths_mode == 'sparse')

    t0 = time.perf_counter()
    detector = Lambda3ZeroShotDetector(L3Config(use_sparse_solver=use_sparse))
    np.random.seed(0)
    result = detector.analyze(events)
    t_analyze = time.perf_counter() - t0

    t0 = time.perf_counter()
    components = compute_components(detector, result, events)
    t_score = time.perf_counter() - t0

    # 構成A: production 4 scorers
    prod_w = {'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15}
    prod_scores = ScoreIntegrator(default_weights=prod_w).combine(components)

    # 構成C: acute-alone (sparse pathsに最適化された z-mix)
    acute_scores = 0.4 * _z(components['jump']) + 0.6 * _z(components['hybrid'])

    return {
        'config': config_name,
        'paths_mode': paths_mode,
        'seed': seed,
        'n_events': events.shape[0],
        'n_anom': int(labels.sum()),
        't_analyze': t_analyze,
        't_score': t_score,
        'auc_prod': float(roc_auc_score(labels, prod_scores)),
        'top10_prod': _topk(prod_scores, labels),
        'auc_acute': float(roc_auc_score(labels, acute_scores)),
        'top10_acute': _topk(acute_scores, labels),
        'auc_jump_only':  float(roc_auc_score(labels, components['jump'])),
        'auc_hybrid_only': float(roc_auc_score(labels, components['hybrid'])),
        'auc_kernel_only': float(roc_auc_score(labels, components['kernel'])),
        'auc_struct_only': float(roc_auc_score(labels, components['structural'])),
        'anomaly_types_present': sorted(set(anomaly_details)),
    }


def main():
    print("=" * 100)
    print("V2 regime: n≈1000 × 20 feats × anomaly=1.5%  ×  混合シナリオ (4種ランダム)")
    print("V2 reference: Basic Mode AUC=0.9625  Top-10=0.90  analyze≈155s + detect≈30s")
    print("=" * 100)

    seeds = [42, 123]
    rows: List[Dict] = []

    for seed in seeds:
        print(f"\n■ seed={seed}")

        for label, mode in [('FULL (V2 baseline)', 'full'),
                             ('SPARSE (Phase 1)',   'sparse')]:
            r = evaluate_config(label, mode, seed)
            rows.append(r)
            print(f"  -- {label}: paths={mode} --")
            print(f"     analyze={r['t_analyze']:7.2f}s  score={r['t_score']:6.2f}s")
            print(f"     #anom={r['n_anom']}  anomaly_types={r['anomaly_types_present']}")
            print(f"     [prod  weights] AUC={r['auc_prod']:.4f}  Top10={r['top10_prod']:.2f}")
            print(f"     [acute (j+h)  ] AUC={r['auc_acute']:.4f}  Top10={r['top10_acute']:.2f}")
            print(f"     [individual AUCs] "
                  f"j={r['auc_jump_only']:.3f}  h={r['auc_hybrid_only']:.3f}  "
                  f"k={r['auc_kernel_only']:.3f}  s={r['auc_struct_only']:.3f}")

    # ===== summary =====
    print("\n" + "=" * 100)
    print("集計 (mean across seeds)")
    print("=" * 100)
    print(f"  {'config':<30}  {'auc_prod':>10}  {'top10':>6}  "
          f"{'auc_acute':>10}  {'top10':>6}  "
          f"{'analyze':>9}  {'score':>7}")
    for label, mode in [('FULL (V2 baseline)', 'full'),
                         ('SPARSE (Phase 1)',   'sparse')]:
        subset = [r for r in rows if r['paths_mode'] == mode]
        if not subset:
            continue
        m = lambda k: float(np.mean([r[k] for r in subset]))
        print(f"  {label:<30}  "
              f"{m('auc_prod'):>10.4f}  {m('top10_prod'):>6.2f}  "
              f"{m('auc_acute'):>10.4f}  {m('top10_acute'):>6.2f}  "
              f"{m('t_analyze'):>8.2f}s  {m('t_score'):>6.2f}s")

    print("\n  scorer別 AUC (mean across seeds × modes)")
    print(f"  {'paths_mode':<10}  {'jump':>7}  {'hybrid':>7}  {'kernel':>7}  {'struct':>7}")
    for mode in ['full', 'sparse']:
        subset = [r for r in rows if r['paths_mode'] == mode]
        if not subset:
            continue
        m = lambda k: float(np.mean([r[k] for r in subset]))
        print(f"  {mode:<10}  "
              f"{m('auc_jump_only'):>7.4f}  {m('auc_hybrid_only'):>7.4f}  "
              f"{m('auc_kernel_only'):>7.4f}  {m('auc_struct_only'):>7.4f}")


if __name__ == "__main__":
    main()
