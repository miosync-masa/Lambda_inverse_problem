"""
NAB realKnownCause ベンチマーク。

各 CSV を Lambda3 detector に (n, 1) で投入、5 scorer から production combined を
raw / mixed (hybrid+kernel symmetric) の 2 系統で計算し、
    - NAB 公式 Sweeper の 3 profile スコア
    - 既存 changepoint_metrics (最初の anomaly window を true window として)
を出力する。

Usage::
    cd Lambda_inverse_problem
    python -m tests.benchmark_nab
    python -m tests.benchmark_nab --category realKnownCause
    python -m tests.benchmark_nab --windows-file combined_windows_tiny.json  # 軽量
"""

from __future__ import annotations

import argparse
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
)

from tests.changepoint_datasets import ChangePointInfo
from tests.changepoint_metrics import evaluate_changepoint
from tests.nab_datasets import iter_category
from tests.nab_features import expand_to_5d, expand_to_6d
from tests.nab_metrics import NABScore, format_nab_score, score_all_profiles


PROD_WEIGHTS = {'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15}
SYM_COMPONENTS = ['hybrid', 'kernel']


def compute_scorer_outputs(events: np.ndarray, result, use_gpu: bool = False,
                            kernel_mode: str = 'poly') -> Dict[str, np.ndarray]:
    """
    kernel_mode:
        'poly' (default): polynomial kernel 固定 (degree=7, coef0=1.0)、GPU 可
        'auto'         : kernel_type=-1 で 90+ candidate を sweep（periodic 含む）。
                          周期データ向けだが重い (CPU only)。GPU フラグは無視。
    """
    np.random.seed(0); jump   = JumpScorer().score(events, result)
    np.random.seed(0); hybrid = HybridScorer().score(events, result)
    np.random.seed(0)
    if kernel_mode == 'auto':
        # 自動カーネル選択 (RBF / Polynomial / Sigmoid / Laplacian / Periodic を網羅 sweep)
        kernel = KernelScorer(kernel_type=-1).score(events, result)
    else:
        kernel = KernelScorer(
            kernel_type=1, degree=7, coef0=1.0, use_gpu=use_gpu,
        ).score(events, result)
    np.random.seed(0); struct = StructuralScorer().score(events, result)
    np.random.seed(0); drift  = DriftScorer().score(events, result)
    return {
        'jump': jump, 'hybrid': hybrid, 'kernel': kernel,
        'structural': struct, 'drift': drift,
    }


def _resolve_calibration_frames(n: int, first_window_start: int) -> int:
    # NAB probation = 先頭15%。calibration はその範囲に収める。
    cal = max(50, int(0.15 * n))
    cal = min(cal, max(5, first_window_start - 1))
    return cal


def _build_cp_info(sample) -> ChangePointInfo:
    si, ei = sample.window_indices[0]
    return ChangePointInfo(
        true_start=si,
        true_end=ei + 1,
        n_normal_pre=si,
        n_anomaly=ei + 1 - si,
        n_normal_post=sample.n - (ei + 1),
        scenario=sample.name,
    )


def run_one(sample, n_features: int = 5, feature_window: int = 30,
            use_gpu: bool = False, kernel_mode: str = 'poly') -> Dict:
    print(f"\n■ {sample.name}  n={sample.n}  #windows={len(sample.windows_ts)}  "
          f"features={n_features}  gpu={use_gpu}  kernel={kernel_mode}")

    if n_features == 1:
        events = sample.values  # (n, 1)
    elif n_features == 5:
        events = expand_to_5d(sample.values, window=feature_window)
    elif n_features == 6:
        events = expand_to_6d(sample.values, window=feature_window)
    else:
        raise ValueError(f"unsupported n_features={n_features}")

    t0 = time.perf_counter()
    detector = Lambda3ZeroShotDetector(L3Config(use_gpu=use_gpu))
    np.random.seed(0)
    result = detector.analyze(events)
    t_analyze = time.perf_counter() - t0

    components = compute_scorer_outputs(events, result, use_gpu=use_gpu,
                                         kernel_mode=kernel_mode)

    cal_frames = _resolve_calibration_frames(
        sample.n, sample.window_indices[0][0]
    )

    # raw
    prod_raw = ScoreIntegrator(default_weights=PROD_WEIGHTS).combine(
        {k: v for k, v in components.items() if k in PROD_WEIGHTS}
    )

    # mixed: hybrid + kernel symmetric
    prod_mixed = ScoreIntegrator(
        default_weights=PROD_WEIGHTS,
        symmetric_components=SYM_COMPONENTS,
    ).combine(
        {k: v for k, v in components.items() if k in PROD_WEIGHTS},
        calibration_frames=cal_frames,
    )

    # === NAB Sweeper (3 profile) ===
    nab_raw   = score_all_profiles(sample, prod_raw)
    nab_mixed = score_all_profiles(sample, prod_mixed)

    print(f"  analyze={t_analyze:.1f}s  cal_frames={cal_frames}  probation≈{int(0.15 * sample.n)}")
    print("  -- raw --")
    for prof, s in nab_raw.items():
        print(format_nab_score(s, 'production'))
    print("  -- mixed (hybrid+kernel sym) --")
    for prof, s in nab_mixed.items():
        print(format_nab_score(s, 'production'))

    # === changepoint metrics (first window only) ===
    info = _build_cp_info(sample)
    labels = sample.labels
    try:
        cp_raw   = evaluate_changepoint(prod_raw,   labels, info, calibration_frames=cal_frames)
        cp_mixed = evaluate_changepoint(prod_mixed, labels, info, calibration_frames=cal_frames)
        print(f"  cp_raw   AUC={cp_raw.auc:.4f}  det={cp_raw.detected}  "
              f"ttd={cp_raw.ttd}  recall={cp_raw.recall_in_window:.3f}")
        print(f"  cp_mixed AUC={cp_mixed.auc:.4f}  det={cp_mixed.detected}  "
              f"ttd={cp_mixed.ttd}  recall={cp_mixed.recall_in_window:.3f}")
    except Exception as e:
        cp_raw = cp_mixed = None
        print(f"  cp_metrics skipped: {e}")

    return {
        'name': sample.name,
        'n': sample.n,
        'n_windows': len(sample.windows_ts),
        't_analyze': t_analyze,
        'cal_frames': cal_frames,
        'nab_raw': nab_raw,
        'nab_mixed': nab_mixed,
        'cp_raw': cp_raw,
        'cp_mixed': cp_mixed,
    }


def _agg_nab(rows: List[Dict], key: str) -> Dict[str, Dict[str, float]]:
    """profile → {'mean_normalized':..., 'sum_normalized':...}"""
    out: Dict[str, Dict[str, float]] = {}
    if not rows:
        return out
    profiles = list(rows[0][key].keys())
    for prof in profiles:
        norms = [r[key][prof].normalized for r in rows]
        out[prof] = {
            'mean': float(np.mean(norms)),
            'min': float(np.min(norms)),
            'max': float(np.max(norms)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', default='realKnownCause')
    ap.add_argument('--windows-file', default='combined_windows.json')
    ap.add_argument('--features', type=int, default=5, choices=[1, 5, 6],
                    help='1=raw univariate, '
                         '5=[raw,rmean,rstd,diff2,lag1ac], '
                         '6=5次元+stddev_MAD偏差 (signal-absence catch)')
    ap.add_argument('--feature-window', type=int, default=30)
    ap.add_argument('--use-gpu', action='store_true',
                    help='enable CuPy GPU path (Colab 等)')
    ap.add_argument('--kernel', choices=['poly', 'auto'], default='poly',
                    help='poly: polynomial degree=7 (GPU可)、'
                         'auto: 90+ kernels (periodic 含む) sweep (CPU, 重い)')
    args = ap.parse_args()

    print("=" * 110)
    print(f"NAB benchmark  category={args.category}  windows={args.windows_file}  "
          f"features={args.features}  feat_w={args.feature_window}  "
          f"gpu={args.use_gpu}  kernel={args.kernel}")
    print("=" * 110)

    rows: List[Dict] = []
    for sample in iter_category(args.category, windows_file=args.windows_file):
        rows.append(run_one(sample, n_features=args.features,
                            feature_window=args.feature_window,
                            use_gpu=args.use_gpu,
                            kernel_mode=args.kernel))

    if not rows:
        print("\n(no samples matched)")
        return

    print("\n" + "=" * 110)
    print(f"Aggregated NAB scores across {len(rows)} files ({args.category})")
    print("=" * 110)
    agg_raw   = _agg_nab(rows, 'nab_raw')
    agg_mixed = _agg_nab(rows, 'nab_mixed')

    print(f"  {'profile':<22}  {'raw_mean':>9}  {'mix_mean':>9}  {'Δ':>7}  "
          f"{'raw_min':>8}  {'mix_min':>8}  {'raw_max':>8}  {'mix_max':>8}")
    print("-" * 110)
    for prof in agg_raw.keys():
        r = agg_raw[prof]; m = agg_mixed[prof]
        delta = m['mean'] - r['mean']
        print(f"  {prof:<22}  {r['mean']:>9.2f}  {m['mean']:>9.2f}  {delta:>+7.2f}  "
              f"{r['min']:>8.2f}  {m['min']:>8.2f}  {r['max']:>8.2f}  {m['max']:>8.2f}")

    # changepoint summary (first window)
    cp_rows = [r for r in rows if r['cp_raw'] is not None]
    if cp_rows:
        auc_r = float(np.mean([r['cp_raw'].auc   for r in cp_rows]))
        auc_m = float(np.mean([r['cp_mixed'].auc for r in cp_rows]))
        det_r = float(np.mean([1.0 if r['cp_raw'].detected   else 0.0 for r in cp_rows])) * 100
        det_m = float(np.mean([1.0 if r['cp_mixed'].detected else 0.0 for r in cp_rows])) * 100
        print(f"\n  changepoint (first window, {len(cp_rows)} files)")
        print(f"    raw   : AUC={auc_r:.4f}  detection={det_r:.0f}%")
        print(f"    mixed : AUC={auc_m:.4f}  detection={det_m:.0f}%   Δ={auc_m - auc_r:+.4f}")


if __name__ == "__main__":
    main()
