"""
Single (production) vs Dual-track detector を 4 シナリオで比較。

各シナリオ × 2 seeds で:
  - Single: Lambda3ZeroShotDetector + production weights (j:0.20 h:0.35 k:0.30 s:0.15)
  - Dual:   Lambda3DualDetector with max(z_acute, z_chronic) fusion

Usage::
    python -m tests.benchmark_dual_per_scenario
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from lambda3_detector import L3Config, Lambda3DualDetector, Lambda3ZeroShotDetector
from lambda3_detector.scorers import ScoreIntegrator

from tests.datasets import create_complex_natural_dataset


SCENARIOS = [
    'progressive_degradation',
    'periodic_burst',
    'chaotic_bifurcation',
    'partial_anomaly',
]


def _single_run(events: np.ndarray, labels: np.ndarray) -> Dict:
    t0 = time.perf_counter()
    det = Lambda3ZeroShotDetector(L3Config())
    np.random.seed(0)
    result = det.analyze(events)
    np.random.seed(0)
    scores = det.detect_anomalies(result, events, use_adaptive_weights=False)
    t = time.perf_counter() - t0
    return {
        'auc': float(roc_auc_score(labels, scores)),
        'top10': float(np.mean(labels[np.argsort(scores)[-10:]])),
        'time': t,
    }


def _dual_run(events: np.ndarray, labels: np.ndarray) -> Dict:
    t0 = time.perf_counter()
    det = Lambda3DualDetector(L3Config())
    np.random.seed(0)
    det.analyze(events)
    np.random.seed(0)
    fused = det.detect_anomalies(events)
    t = time.perf_counter() - t0
    return {
        'auc': float(roc_auc_score(labels, fused)),
        'auc_acute': float(roc_auc_score(labels, det.acute_scores)),
        'auc_chronic': float(roc_auc_score(labels, det.chronic_scores)),
        'top10': float(np.mean(labels[np.argsort(fused)[-10:]])),
        'time': t,
    }


def benchmark_scenario(scenario: str, seed: int) -> Dict:
    np.random.seed(seed)
    events, labels, _ = create_complex_natural_dataset(
        n_events=300, n_features=10, anomaly_ratio=0.15,
        scenario_filter=scenario,
    )

    single = _single_run(events, labels)
    dual = _dual_run(events, labels)

    return {
        'scenario': scenario, 'seed': seed,
        'n_anom': int(labels.sum()),
        'single': single,
        'dual': dual,
    }


def main():
    print("=" * 100)
    print("Single (production weights) vs Dual-track (max-z fusion)  — per scenario")
    print("300 events × 10 features × anomaly_ratio=0.15  ×  2 seeds")
    print("=" * 100)

    seeds = [42, 123]
    all_results: List[Dict] = []

    for scenario in SCENARIOS:
        print(f"\n■ scenario = {scenario}")
        for seed in seeds:
            r = benchmark_scenario(scenario, seed)
            all_results.append(r)
            s = r['single']; d = r['dual']
            print(f"  seed={seed:<4}  #anom={r['n_anom']:>2}")
            print(f"           [single] AUC={s['auc']:.4f}  Top10={s['top10']:.2f}  "
                  f"time={s['time']:6.2f}s")
            print(f"           [dual  ] AUC={d['auc']:.4f}  Top10={d['top10']:.2f}  "
                  f"time={d['time']:6.2f}s  "
                  f"(acute={d['auc_acute']:.4f}, chronic={d['auc_chronic']:.4f})")
            print(f"           [Δ     ] ΔAUC={d['auc'] - s['auc']:+.4f}  "
                  f"speedup={s['time']/max(d['time'],1e-9):.2f}x")

    print("\n" + "=" * 100)
    print("シナリオ別集計 (mean across seeds)")
    print("=" * 100)
    print(f"  {'scenario':<26}  {'single AUC':>11}  {'dual AUC':>11}  "
          f"{'ΔAUC':>8}  {'acute':>8}  {'chronic':>8}  "
          f"{'single t':>9}  {'dual t':>9}")
    for scenario in SCENARIOS:
        subset = [r for r in all_results if r['scenario'] == scenario]
        if not subset:
            continue
        single_auc = np.mean([r['single']['auc'] for r in subset])
        dual_auc   = np.mean([r['dual']['auc']   for r in subset])
        acute_auc  = np.mean([r['dual']['auc_acute']   for r in subset])
        chronic_auc= np.mean([r['dual']['auc_chronic'] for r in subset])
        single_t   = np.mean([r['single']['time'] for r in subset])
        dual_t     = np.mean([r['dual']['time']   for r in subset])
        print(f"  {scenario:<26}  {single_auc:>11.4f}  {dual_auc:>11.4f}  "
              f"{dual_auc-single_auc:>+8.4f}  {acute_auc:>8.4f}  {chronic_auc:>8.4f}  "
              f"{single_t:>8.2f}s  {dual_t:>8.2f}s")

    # 全体集計
    all_single = np.mean([r['single']['auc'] for r in all_results])
    all_dual   = np.mean([r['dual']['auc']   for r in all_results])
    print(f"\n  {'OVERALL (across all scenarios)':<26}  "
          f"{all_single:>11.4f}  {all_dual:>11.4f}  "
          f"{all_dual-all_single:>+8.4f}")


if __name__ == "__main__":
    main()
