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
                          GPU 可 (kernel_anomaly_scores_auto_gpu)。
        'both'         : poly と auto の両方を返す。non-kernel scorer は共有計算。
                          dict に 'kernel' (poly) と 'kernel_auto' の両方が入る。

    注: scorer 呼び出し順は jump → hybrid → kernel(s) → struct → drift。
    過去 commit でこの順序の方が良いスコアを出した実績があり、また経験的に
    順序を変えると kernel 出力が縮退するファイルがある (struct/drift 自体は
    paths を変更しないが、GPU L-BFGS-B の非凸最適化が float32 で非決定的に
    なるため、run-to-run の数値順序差が paths に伝播する可能性がある)。
    """
    np.random.seed(0); jump   = JumpScorer().score(events, result)
    np.random.seed(0); hybrid = HybridScorer().score(events, result)

    out: Dict[str, np.ndarray] = {'jump': jump, 'hybrid': hybrid}

    # === Kernel(s) — struct/drift より前に置く ===
    if kernel_mode == 'both':
        np.random.seed(0)
        out['kernel'] = KernelScorer(
            kernel_type=1, degree=7, coef0=1.0, use_gpu=use_gpu,
        ).score(events, result)
        np.random.seed(0)
        out['kernel_auto'] = KernelScorer(
            kernel_type=-1, use_gpu=use_gpu,
        ).score(events, result)
    elif kernel_mode == 'auto':
        np.random.seed(0)
        out['kernel'] = KernelScorer(
            kernel_type=-1, use_gpu=use_gpu,
        ).score(events, result)
    else:  # 'poly'
        np.random.seed(0)
        out['kernel'] = KernelScorer(
            kernel_type=1, degree=7, coef0=1.0, use_gpu=use_gpu,
        ).score(events, result)

    np.random.seed(0); out['structural'] = StructuralScorer().score(events, result)
    np.random.seed(0); out['drift']      = DriftScorer().score(events, result)
    return out


def _zn_cal(scores: np.ndarray, cal_frames: int, eps: float = 1e-12) -> np.ndarray:
    """calibration 窓内の平均・標準偏差で z-norm。複数 production score の
    ensemble 合成前にスケールを揃える。"""
    cal = scores[:cal_frames]
    mu = float(np.mean(cal))
    sd = float(np.std(cal)) + eps
    return (scores - mu) / sd


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
            use_gpu: bool = False, kernel_mode: str = 'poly',
            ensemble: bool = False) -> Dict:
    eff_kernel = 'both' if ensemble else kernel_mode
    print(f"\n■ {sample.name}  n={sample.n}  #windows={len(sample.windows_ts)}  "
          f"features={n_features}  gpu={use_gpu}  kernel={eff_kernel}"
          f"{'  [ENSEMBLE]' if ensemble else ''}")

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
                                         kernel_mode=eff_kernel)

    cal_frames = _resolve_calibration_frames(
        sample.n, sample.window_indices[0][0]
    )

    # === production score 構築 ===
    def _combine(kernel_arr, symmetric: bool):
        comp_subset = {
            'jump': components['jump'],
            'hybrid': components['hybrid'],
            'kernel': kernel_arr,
            'structural': components['structural'],
        }
        if symmetric:
            return ScoreIntegrator(
                default_weights=PROD_WEIGHTS,
                symmetric_components=SYM_COMPONENTS,
            ).combine(comp_subset, calibration_frames=cal_frames)
        return ScoreIntegrator(default_weights=PROD_WEIGHTS).combine(comp_subset)

    prod_raw = _combine(components['kernel'], symmetric=False)
    prod_mixed = _combine(components['kernel'], symmetric=True)

    if ensemble:
        prod_auto_raw = _combine(components['kernel_auto'], symmetric=False)
        prod_auto_mixed = _combine(components['kernel_auto'], symmetric=True)

        # === DEBUG: ensemble 配列が本当に別物か診断 ===
        k_poly = components['kernel']
        k_auto = components['kernel_auto']
        print(f"  [DEBUG] kernel arrays:")
        print(f"    id(poly)={id(k_poly)}  id(auto)={id(k_auto)}  "
              f"same_obj={k_poly is k_auto}")
        print(f"    poly: shape={k_poly.shape} dtype={k_poly.dtype} "
              f"mean={float(np.mean(k_poly)):.4f} std={float(np.std(k_poly)):.4f} "
              f"min={float(np.min(k_poly)):.4f} max={float(np.max(k_poly)):.4f}")
        print(f"    auto: shape={k_auto.shape} dtype={k_auto.dtype} "
              f"mean={float(np.mean(k_auto)):.4f} std={float(np.std(k_auto)):.4f} "
              f"min={float(np.min(k_auto)):.4f} max={float(np.max(k_auto)):.4f}")
        print(f"    array_equal={np.array_equal(k_poly, k_auto)}  "
              f"max_abs_diff={float(np.max(np.abs(k_poly - k_auto))):.6e}")
        print(f"  [DEBUG] production arrays:")
        print(f"    prod_raw: mean={float(np.mean(prod_raw)):.4f} "
              f"std={float(np.std(prod_raw)):.4f} "
              f"min={float(np.min(prod_raw)):.4f} max={float(np.max(prod_raw)):.4f}")
        print(f"    prod_auto_raw: mean={float(np.mean(prod_auto_raw)):.4f} "
              f"std={float(np.std(prod_auto_raw)):.4f} "
              f"min={float(np.min(prod_auto_raw)):.4f} max={float(np.max(prod_auto_raw)):.4f}")
        print(f"    prod_raw == prod_auto_raw: {np.array_equal(prod_raw, prod_auto_raw)}")

        # max of z-normalized (calibration window 基準)
        prod_ens_raw = np.maximum(
            _zn_cal(prod_raw, cal_frames),
            _zn_cal(prod_auto_raw, cal_frames),
        )
        prod_ens_mixed = np.maximum(
            _zn_cal(prod_mixed, cal_frames),
            _zn_cal(prod_auto_mixed, cal_frames),
        )
    else:
        prod_auto_raw = prod_auto_mixed = None
        prod_ens_raw = prod_ens_mixed = None

    # === NAB Sweeper ===
    nab_raw   = score_all_profiles(sample, prod_raw)
    nab_mixed = score_all_profiles(sample, prod_mixed)
    if ensemble:
        nab_auto_raw   = score_all_profiles(sample, prod_auto_raw)
        nab_auto_mixed = score_all_profiles(sample, prod_auto_mixed)
        nab_ens_raw    = score_all_profiles(sample, prod_ens_raw)
        nab_ens_mixed  = score_all_profiles(sample, prod_ens_mixed)
    else:
        nab_auto_raw = nab_auto_mixed = None
        nab_ens_raw = nab_ens_mixed = None

    print(f"  analyze={t_analyze:.1f}s  cal_frames={cal_frames}  probation≈{int(0.15 * sample.n)}")
    if ensemble:
        # ensemble 経路: poly / auto / ensemble を横並びで表示
        print("  -- raw --")
        for prof in nab_raw.keys():
            sp, sa, se = nab_raw[prof], nab_auto_raw[prof], nab_ens_raw[prof]
            print(f"    {prof:<22}  poly_norm={sp.normalized:+7.2f}  "
                  f"auto_norm={sa.normalized:+7.2f}  ens_norm={se.normalized:+7.2f}")
        print("  -- mixed (hybrid+kernel sym) --")
        for prof in nab_mixed.keys():
            sp, sa, se = nab_mixed[prof], nab_auto_mixed[prof], nab_ens_mixed[prof]
            print(f"    {prof:<22}  poly_norm={sp.normalized:+7.2f}  "
                  f"auto_norm={sa.normalized:+7.2f}  ens_norm={se.normalized:+7.2f}")
    else:
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
        'nab_auto_raw': nab_auto_raw,
        'nab_auto_mixed': nab_auto_mixed,
        'nab_ens_raw': nab_ens_raw,
        'nab_ens_mixed': nab_ens_mixed,
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
                         'auto: 90+ kernels (periodic 含む) sweep (GPU可)')
    ap.add_argument('--ensemble', action='store_true',
                    help='poly + auto を両方計算し、cal-window z-norm の '
                         '要素ごと max を ensemble production score とする')
    args = ap.parse_args()

    print("=" * 110)
    print(f"NAB benchmark  category={args.category}  windows={args.windows_file}  "
          f"features={args.features}  feat_w={args.feature_window}  "
          f"gpu={args.use_gpu}  kernel={args.kernel}"
          f"{'  [ENSEMBLE on]' if args.ensemble else ''}")
    print("=" * 110)

    rows: List[Dict] = []
    for sample in iter_category(args.category, windows_file=args.windows_file):
        rows.append(run_one(sample, n_features=args.features,
                            feature_window=args.feature_window,
                            use_gpu=args.use_gpu,
                            kernel_mode=args.kernel,
                            ensemble=args.ensemble))

    if not rows:
        print("\n(no samples matched)")
        return

    print("\n" + "=" * 110)
    print(f"Aggregated NAB scores across {len(rows)} files ({args.category})")
    print("=" * 110)

    if args.ensemble:
        # poly / auto / ensemble × raw / mixed の 6 系統
        agg_p_r = _agg_nab(rows, 'nab_raw')
        agg_a_r = _agg_nab(rows, 'nab_auto_raw')
        agg_e_r = _agg_nab(rows, 'nab_ens_raw')
        agg_p_m = _agg_nab(rows, 'nab_mixed')
        agg_a_m = _agg_nab(rows, 'nab_auto_mixed')
        agg_e_m = _agg_nab(rows, 'nab_ens_mixed')

        print(f"  {'profile':<22}  "
              f"{'poly_raw':>8}  {'auto_raw':>8}  {'ens_raw':>8}  "
              f"{'poly_mix':>8}  {'auto_mix':>8}  {'ens_mix':>8}")
        print("-" * 110)
        for prof in agg_p_r.keys():
            print(f"  {prof:<22}  "
                  f"{agg_p_r[prof]['mean']:>8.2f}  {agg_a_r[prof]['mean']:>8.2f}  "
                  f"{agg_e_r[prof]['mean']:>8.2f}  "
                  f"{agg_p_m[prof]['mean']:>8.2f}  {agg_a_m[prof]['mean']:>8.2f}  "
                  f"{agg_e_m[prof]['mean']:>8.2f}")

        print("\n  3-profile mean (column):")
        col_means = {
            'poly_raw': np.mean([agg_p_r[p]['mean'] for p in agg_p_r]),
            'auto_raw': np.mean([agg_a_r[p]['mean'] for p in agg_a_r]),
            'ens_raw':  np.mean([agg_e_r[p]['mean'] for p in agg_e_r]),
            'poly_mix': np.mean([agg_p_m[p]['mean'] for p in agg_p_m]),
            'auto_mix': np.mean([agg_a_m[p]['mean'] for p in agg_a_m]),
            'ens_mix':  np.mean([agg_e_m[p]['mean'] for p in agg_e_m]),
        }
        for k, v in col_means.items():
            print(f"    {k:<10} = {v:6.2f}")
    else:
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
