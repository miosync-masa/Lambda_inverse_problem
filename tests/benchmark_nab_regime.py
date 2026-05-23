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
            K: int = 3,
            mask_margin: int = 50,
            n_features: int = 5,
            feature_window: int = 30,
            percentile: float = 99.0) -> Dict:
    n_windows = len(sample.windows_ts)
    print(f"\n■ {sample.name}  n={sample.n}  #windows={n_windows}  "
          f"K={K}  margin={mask_margin}  features={n_features}  "
          f"percentile={percentile}  [REGIME]")

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
        K=K, mask_margin=mask_margin, percentile=percentile,
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

    print(f"  K_eff={K_eff}  clean_frames={clean_n}  "
          f"total_run={t_run:.1f}s  #flagged={int(binary.sum())}/{sample.n}  "
          f"regime_dist=[{regime_dist}]")

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
    ap.add_argument('--K', type=int, default=3,
                    help='GMM 成分数 (regime 数)')
    ap.add_argument('--mask-margin', type=int, default=50,
                    help='anomaly window 前後の除外マージン (frame)')
    args = ap.parse_args()

    print("=" * 110)
    print(f"NAB REGIME-AWARE benchmark  category={args.category}  "
          f"K={args.K}  mask_margin={args.mask_margin}  "
          f"windows={args.windows_file}  features={args.features}  "
          f"percentile={args.percentile}")
    print("=" * 110)

    rows: List[Dict] = []
    for sample in iter_category(args.category, windows_file=args.windows_file):
        r = run_one(
            sample,
            K=args.K, mask_margin=args.mask_margin,
            n_features=args.features, feature_window=args.feature_window,
            percentile=args.percentile,
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
