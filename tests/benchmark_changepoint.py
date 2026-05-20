"""
Change-point detection ベンチマーク。

産業利用 regime (時間順序保持) で5 scorer (jump/hybrid/kernel/structural/drift) と
production combined を評価。

Usage::
    cd Lambda_inverse_problem
    python -m tests.benchmark_changepoint
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from lambda3_detector import L3Config, Lambda3ZeroShotDetector
from lambda3_detector.scorers import (
    DriftScorer,
    HybridScorer,
    JumpScorer,
    KernelScorer,
    ScoreIntegrator,
    StructuralScorer,
    polarity_symmetric_score,
)

from tests.changepoint_datasets import SCENARIOS, create_changepoint_dataset
from tests.changepoint_metrics import evaluate_changepoint, format_metrics


def compute_scorer_outputs(events, result):
    np.random.seed(0); jump   = JumpScorer().score(events, result)
    np.random.seed(0); hybrid = HybridScorer().score(events, result)
    np.random.seed(0); kernel = KernelScorer(kernel_type=1, degree=7, coef0=1.0).score(events, result)
    np.random.seed(0); struct = StructuralScorer().score(events, result)
    np.random.seed(0); drift  = DriftScorer().score(events, result)
    return {
        'jump': jump, 'hybrid': hybrid, 'kernel': kernel,
        'structural': struct, 'drift': drift,
    }


def run_one(scenario: str, seed: int,
            n_normal_pre: int = 300, n_anomaly: int = 100, n_normal_post: int = 300,
            n_features: int = 20, intensity: float = 2.0) -> Dict:
    np.random.seed(seed)
    events, labels, info = create_changepoint_dataset(
        n_normal_pre=n_normal_pre,
        n_anomaly=n_anomaly,
        n_normal_post=n_normal_post,
        n_features=n_features,
        scenario=scenario,
        intensity=intensity,
    )

    t0 = time.perf_counter()
    detector = Lambda3ZeroShotDetector(L3Config())  # sparse default
    np.random.seed(0)
    result = detector.analyze(events)
    t_analyze = time.perf_counter() - t0

    components = compute_scorer_outputs(events, result)

    # production combined (raw)
    prod_w = {'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15}
    prod_scores = ScoreIntegrator(default_weights=prod_w).combine(
        {k: v for k, v in components.items() if k in prod_w}
    )
    components['production'] = prod_scores

    # === (E) mixed: hybrid/kernel のみ symmetric, 他は raw ===
    # calibration: n_normal_pre の半分（必ず正常区間内）
    cal_frames = max(10, info.n_normal_pre // 2)
    SYM_COMPONENTS = ['hybrid', 'kernel']

    # 個別 scorer は polarity_symmetric 適用 or raw のいずれか
    components_mixed: Dict[str, np.ndarray] = {}
    for name, scores in components.items():
        if name == 'production':
            continue
        if name in SYM_COMPONENTS:
            components_mixed[name] = polarity_symmetric_score(
                scores, calibration_frames=cal_frames
            )
        else:
            components_mixed[name] = scores

    # production combined を symmetric_components 経由で再構成
    prod_scores_mixed = ScoreIntegrator(
        default_weights=prod_w,
        symmetric_components=SYM_COMPONENTS,
    ).combine(
        {k: v for k, v in components.items() if k in prod_w},
        calibration_frames=cal_frames,
    )
    components_mixed['production'] = prod_scores_mixed

    # 評価 (raw / mixed それぞれ)
    metrics_raw: Dict[str, dict] = {}
    metrics_mixed: Dict[str, dict] = {}
    for name, scores in components.items():
        metrics_raw[name] = evaluate_changepoint(scores, labels, info)
    for name, scores in components_mixed.items():
        metrics_mixed[name] = evaluate_changepoint(scores, labels, info)

    return {
        'scenario': scenario, 'seed': seed, 'info': info,
        't_analyze': t_analyze,
        'metrics': metrics_raw,
        'metrics_mixed': metrics_mixed,
    }


def main():
    print("=" * 110)
    print("Change-point benchmark (time-ordered  [N_pre | A | N_post])")
    print("n_normal_pre=300, n_anomaly=100, n_normal_post=300, n_features=20, intensity=2.0")
    print("=" * 110)

    seeds = [42, 123]
    scorer_order = ['jump', 'hybrid', 'kernel', 'structural', 'drift', 'production']
    rows: List[Dict] = []

    for scenario in SCENARIOS:
        print(f"\n■ scenario = {scenario}")
        for seed in seeds:
            r = run_one(scenario, seed)
            rows.append(r)
            info = r['info']
            print(f"  seed={seed:<4}  true_window=[{info.true_start}, {info.true_end})  "
                  f"analyze={r['t_analyze']:.1f}s")
            print("  -- raw --")
            for name in scorer_order:
                m = r['metrics'][name]
                print(format_metrics(m, name))
            print("  -- mixed (hybrid+kernel symmetric, others raw) --")
            for name in scorer_order:
                m = r['metrics_mixed'][name]
                print(format_metrics(m, name))

    # ===== 集計 =====
    print("\n" + "=" * 120)
    print("シナリオ別 scorer 性能 (raw vs mixed [hybrid+kernel sym], mean across seeds)")
    print("=" * 120)
    print(f"  {'scenario':<24}  {'scorer':<12}  "
          f"{'raw_AUC':>8}  {'mix_AUC':>8}  {'delta':>7}  "
          f"{'raw_det%':>8}  {'mix_det%':>8}  "
          f"{'raw_recall':>10}  {'mix_recall':>10}")
    print("-" * 120)

    for scenario in SCENARIOS:
        subset = [r for r in rows if r['scenario'] == scenario]
        if not subset:
            continue
        for name in scorer_order:
            ms_r = [r['metrics'][name] for r in subset]
            ms_m = [r['metrics_mixed'][name] for r in subset]
            auc_r = float(np.mean([m.auc for m in ms_r]))
            auc_m = float(np.mean([m.auc for m in ms_m]))
            det_r = float(np.mean([1.0 if m.detected else 0.0 for m in ms_r])) * 100
            det_m = float(np.mean([1.0 if m.detected else 0.0 for m in ms_m])) * 100
            rec_r = float(np.mean([m.recall_in_window for m in ms_r]))
            rec_m = float(np.mean([m.recall_in_window for m in ms_m]))
            print(f"  {scenario:<24}  {name:<12}  "
                  f"{auc_r:>8.4f}  {auc_m:>8.4f}  {auc_m - auc_r:>+7.3f}  "
                  f"{det_r:>7.0f}%  {det_m:>7.0f}%  "
                  f"{rec_r:>10.3f}  {rec_m:>10.3f}")
        print("-" * 120)

    # ===== production combined overall =====
    print("\n  Production combined — overall (4 scenarios × seeds)")
    raws   = [r['metrics']['production']        for r in rows]
    mixes  = [r['metrics_mixed']['production']  for r in rows]
    auc_r = float(np.mean([m.auc for m in raws]))
    auc_m = float(np.mean([m.auc for m in mixes]))
    det_r = float(np.mean([1.0 if m.detected else 0.0 for m in raws])) * 100
    det_m = float(np.mean([1.0 if m.detected else 0.0 for m in mixes])) * 100
    print(f"    raw   : AUC={auc_r:.4f}  detection={det_r:.0f}%")
    print(f"    mixed : AUC={auc_m:.4f}  detection={det_m:.0f}%   Δ={auc_m - auc_r:+.4f}")


if __name__ == "__main__":
    main()
