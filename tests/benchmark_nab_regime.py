"""
NAB regime-aware benchmark (semi-supervised, normal-label only).

streaming Tier 0 (benchmark_nab_streaming.py) との違い:
  - 先頭 15% calibration ではなく、全期間から anomaly window を除外した
    clean data で GMM(K) regime cluster を fit
  - 各 regime ごとに per-scorer threshold を fit
  - streaming 時に gmm.predict で frame の regime を判定、regime 別 threshold で OR voting

学術的分類: **semi-supervised (normal-label only)**。
anomaly の "shape" は学習しない (NAB の combined_windows.json は frame 除外のみに使用)。
industrial 文脈: operator-tagged post-mortem 期間を除いた歴史データで baseline 作成。

Usage::
    python -m tests.benchmark_nab_regime
    python -m tests.benchmark_nab_regime --category realKnownCause
    python -m tests.benchmark_nab_regime --K 3 --mask-margin 50
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from lambda3_detector.regime import RegimeAwareDetector

from tests.nab_datasets import iter_category
from tests.nab_features import expand_to_5d
from tests.nab_metrics import format_nab_score, score_all_profiles


def make_anomaly_mask(sample) -> np.ndarray:
    """combined_windows.json の anomaly window を frame-wise mask に変換。"""
    mask = np.zeros(sample.n, dtype=bool)
    for si, ei in sample.window_indices:
        mask[si:ei + 1] = True
    return mask


def run_one(sample,
            K=3,
            K_max: int = 5,
            mask_margin: int = 50,
            margin_adaptive: bool = False,
            margin_max: int = 300,
            margin_max_exclusion_ratio: float = 0.4,
            margin_recovery_window: int = 30,
            margin_variance_ratio: float = 2.0,
            n_features: int = 5,
            feature_window: int = 30,
            percentile: float = 99.0,
            threshold_method: str = 'percentile',
            iqr_k: float = 3.0,
            mad_k: float = 2.5,
            trim_fraction: float = 0.01,
            cap_ratio: float = 5.0,
            cap_quantile: float = 90.0,
            cap_min_regime_size: int = 300,
            min_frames_per_regime: int = 50) -> Dict:
    n_windows = len(sample.windows_ts)
    K_disp = K if isinstance(K, str) else int(K)
    print(f"\n■ {sample.name}  n={sample.n}  #windows={n_windows}  "
          f"K={K_disp}  margin={mask_margin}  features={n_features}  "
          f"percentile={percentile}  thr_method={threshold_method}  [REGIME]")

    if n_features == 1:
        events = sample.values
    elif n_features == 5:
        events = expand_to_5d(sample.values, window=feature_window)
    else:
        raise ValueError(f"unsupported n_features={n_features}")

    anomaly_mask = make_anomaly_mask(sample)
    if not anomaly_mask.any():
        print("  (no anomaly windows, skipping)")
        return None

    detector = RegimeAwareDetector(
        K=K, K_max=K_max, mask_margin=mask_margin, percentile=percentile,
        margin_adaptive=margin_adaptive,
        margin_max=margin_max,
        margin_max_exclusion_ratio=margin_max_exclusion_ratio,
        margin_recovery_window=margin_recovery_window,
        margin_variance_ratio=margin_variance_ratio,
        threshold_method=threshold_method,
        iqr_k=iqr_k, mad_k=mad_k, trim_fraction=trim_fraction,
        cap_ratio=cap_ratio, cap_quantile=cap_quantile,
        cap_min_regime_size=cap_min_regime_size,
        min_frames_per_regime=min_frames_per_regime,
    )

    t0 = time.perf_counter()
    try:
        result = detector.fit_predict(events, anomaly_mask)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return None
    t_run = time.perf_counter() - t0

    score = result['score']
    binary = result['binary']
    K_eff = result['K_eff']
    clean_n = result['cal_clean_frames']

    # regime ごとのサンプル分布
    regimes = result['regimes']
    regime_dist = " ".join(
        f"k{k}={int((regimes == k).sum())}" for k in range(K_eff)
    )
    bic_str = ""
    if result.get('bic_per_K'):
        bic_str = "  bic={" + ", ".join(
            f"{k}:{v:.0f}" for k, v in sorted(result['bic_per_K'].items())
        ) + "}"

    margin_str = ""
    minfo = getattr(detector, 'margin_info', None)
    if minfo is not None:
        margin_str = (
            f"  margin_adaptive: pre={minfo['avg_margin_pre']:.0f} "
            f"post={minfo['avg_margin_post']:.0f} "
            f"excl_pre={minfo['exclusion_ratio_pre_cap']:.1%} "
            f"excl_final={minfo['exclusion_ratio_final']:.1%} "
            f"eff_cap={minfo['effective_margin_after_cap']}"
        )

    print(f"  K_eff={K_eff}  clean_frames={clean_n}  "
          f"total_run={t_run:.1f}s  #flagged={int(binary.sum())}/{sample.n}  "
          f"regime_dist=[{regime_dist}]{bic_str}{margin_str}")

    # threshold print: regime 別の dict を見やすく
    for k in range(K_eff):
        thr_str = "  ".join(
            f"{name}={v:.3g}" for name, v in
            result['thresholds_per_regime'][k].items()
        )
        print(f"    regime{k}: {thr_str}")

    nab_scores = score_all_profiles(sample, score)
    for prof, s in nab_scores.items():
        print(format_nab_score(s, 'regime'))

    return {
        'name': sample.name,
        'n': sample.n,
        'n_windows': n_windows,
        'K_eff': K_eff,
        'clean_frames': clean_n,
        't_run': t_run,
        'nab_scores': nab_scores,
        'flag_count': int(binary.sum()),
    }


def _agg(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    rows = [r for r in rows if r is not None]
    if not rows:
        return out
    profiles = list(rows[0]['nab_scores'].keys())
    for prof in profiles:
        norms = [r['nab_scores'][prof].normalized for r in rows]
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
    ap.add_argument('--features', type=int, default=5, choices=[1, 5])
    ap.add_argument('--feature-window', type=int, default=30)
    ap.add_argument('--percentile', type=float, default=99.0)
    ap.add_argument('--K', default='auto',
                    help='GMM 成分数 (regime 数)。int (1-K_max) または "auto" '
                         '("auto" は BIC 自動選択、default)')
    ap.add_argument('--K-max', type=int, default=5,
                    help='K="auto" のときの最大候補 (default 5)')
    ap.add_argument('--mask-margin', type=int, default=50,
                    help='anomaly window 前後の除外マージン (frame、固定 or adaptive 時の base)')
    ap.add_argument('--margin-adaptive', action='store_true',
                    help='adaptive_anomaly_mask を有効化 (gradual leak 検出 + 総除外率 cap)')
    ap.add_argument('--margin-max', type=int, default=300,
                    help='adaptive 延長の上限 frame (default 300)')
    ap.add_argument('--margin-max-exclusion-ratio', type=float, default=0.4,
                    help='総除外率の cap (default 0.4 = 40%)')
    ap.add_argument('--margin-recovery-window', type=int, default=30,
                    help='local variance を測る window size (default 30)')
    ap.add_argument('--margin-variance-ratio', type=float, default=2.0,
                    help='baseline の何倍までを recovered と見るか (default 2.0)')
    ap.add_argument('--min-frames-per-regime', type=int, default=50,
                    help='各 regime に必要な最小サンプル数 (BIC 採用条件)')
    ap.add_argument('--threshold-method', default='percentile',
                    choices=['percentile', 'trimmed_percentile', 'iqr', 'mad', 'capped'],
                    help='regime ごと threshold 計算手法 (default percentile=baseline)')
    ap.add_argument('--iqr-k', type=float, default=3.0,
                    help='iqr method の係数 (default 3.0)')
    ap.add_argument('--mad-k', type=float, default=2.5,
                    help='mad method の係数 (default 2.5)')
    ap.add_argument('--trim-fraction', type=float, default=0.01,
                    help='trimmed_percentile method の上位除外割合 (default 0.01)')
    ap.add_argument('--cap-ratio', type=float, default=5.0,
                    help='capped method: cap = cap_ratio * percentile(cap_quantile) (default 5.0)')
    ap.add_argument('--cap-quantile', type=float, default=90.0,
                    help='capped method の cap base quantile (default 90.0)')
    ap.add_argument('--cap-min-regime-size', type=int, default=300,
                    help='capped を有効化する最小 regime サイズ (default 300、未満は percentile fallback)')
    args = ap.parse_args()

    # K parse: int or 'auto'
    K_raw = args.K
    if isinstance(K_raw, str) and K_raw.lower() == 'auto':
        K_param = 'auto'
    else:
        K_param = int(K_raw)

    print("=" * 110)
    K_disp = K_param if isinstance(K_param, str) else int(K_param)
    print(f"NAB REGIME-AWARE benchmark  category={args.category}  "
          f"K={K_disp}  K_max={args.K_max}  mask_margin={args.mask_margin}  "
          f"min_frames_per_regime={args.min_frames_per_regime}  "
          f"thr_method={args.threshold_method}  "
          f"windows={args.windows_file}  features={args.features}  "
          f"percentile={args.percentile}")
    print("=" * 110)

    rows: List[Dict] = []
    for sample in iter_category(args.category, windows_file=args.windows_file):
        r = run_one(
            sample,
            K=K_param, K_max=args.K_max,
            mask_margin=args.mask_margin,
            margin_adaptive=args.margin_adaptive,
            margin_max=args.margin_max,
            margin_max_exclusion_ratio=args.margin_max_exclusion_ratio,
            margin_recovery_window=args.margin_recovery_window,
            margin_variance_ratio=args.margin_variance_ratio,
            n_features=args.features, feature_window=args.feature_window,
            percentile=args.percentile,
            threshold_method=args.threshold_method,
            iqr_k=args.iqr_k, mad_k=args.mad_k, trim_fraction=args.trim_fraction,
            cap_ratio=args.cap_ratio, cap_quantile=args.cap_quantile,
            cap_min_regime_size=args.cap_min_regime_size,
            min_frames_per_regime=args.min_frames_per_regime,
        )
        if r is not None:
            rows.append(r)

    if not rows:
        print("\n(no samples matched)")
        return

    print("\n" + "=" * 110)
    print(f"Aggregated regime-aware NAB scores "
          f"({len(rows)} files, category={args.category})")
    print("=" * 110)
    agg = _agg(rows)
    print(f"  {'profile':<22}  {'mean':>9}  {'min':>9}  {'max':>9}")
    print("-" * 110)
    for prof, stats in agg.items():
        print(f"  {prof:<22}  {stats['mean']:>9.2f}  "
              f"{stats['min']:>9.2f}  {stats['max']:>9.2f}")

    three_prof_mean = np.mean([agg[p]['mean'] for p in agg])
    print(f"\n  3-profile mean = {three_prof_mean:6.2f}")


if __name__ == "__main__":
    main()
