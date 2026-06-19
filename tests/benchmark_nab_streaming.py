"""
NAB streaming benchmark — Lambda³ streaming prototype.

batch 版 benchmark_nab.py との違い:
  - detector.analyze (全期間 batch) を使わない
  - Lambda3StreamingDetector で先頭 15% で calibrate → 残り 85% を streaming
  - 各 scorer が独立に "正常 baseline" を確立し、binary OR で flag 決定
  - 出力は per-frame max-normalized score (NAB Sweeper にそのまま渡せる)

Future leakage が厳密に排除されるため、NAB scoreboard との fair comparison。
batch 版より低スコアになる見込み (上限値) だが、HTM/ARTime と同条件で評価可能。

Usage::
    python -m tests.benchmark_nab_streaming
    python -m tests.benchmark_nab_streaming --category realKnownCause
    python -m tests.benchmark_nab_streaming --windows-file combined_windows_tiny.json
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from lambda3_detector.streaming import (
    Lambda3StreamingDetector,
    StreamingGradualScorer,
    StreamingJumpScorer,
    StreamingKernelScorer,
    StreamingPeriodicScorer,
    StreamingReconstructionScorer,
    StreamingStructuralDriftScorer,
    StreamingStructuralScorer,
)

from tests.nab_datasets import iter_category
from tests.nab_features import expand_to_5d
from tests.nab_metrics import format_nab_score, score_all_profiles


#: Mapping from short scorer name → factory builder for streaming.
#: Mirrors lambda3_detector.regime.SCORER_FACTORIES but for the streaming
#: scorer classes (per-scorer ablation via --scorers / --exclude-scorers).
STREAMING_SCORER_FACTORIES = {
    'jump':    lambda p: StreamingJumpScorer(percentile=p),
    'gradual': lambda p: StreamingGradualScorer(
        window_sizes=[50, 200, 500], percentile=p),
    'drift':   lambda p: StreamingStructuralDriftScorer(
        local_window=200, percentile=p),
    'recon':   lambda p: StreamingReconstructionScorer(
        n_components=5, delay_window=20, percentile=p),
    'kernel':  lambda p: StreamingKernelScorer(
        kernel='polynomial', degree=3, coef0=1.0, percentile=p),
    'struct':  lambda p: StreamingStructuralScorer(
        delay_window=20, percentile=p),
}

STREAMING_SCORER_NAMES = ['jump', 'gradual', 'drift', 'recon', 'kernel', 'struct']

# NOTE: StreamingPeriodicScorer は default 構成から外している。
#   理由 (realKnownCause 7-file 実測):
#     - ambient_temp (季節 drift で calibration 不整合) で救済不能
#     - nyc_taxi で noise floor 上昇により -10.96 ポイント副作用
#     - 6-scorer 比 net mean は -1.10
#   思想は正しいが static calibration の streaming 設計と相性悪く、
#   adaptive baseline (EWMA 等) が必須。研究課題として温存。


def make_detector(percentile: float = 99.0,
                  scorer_names: list = None) -> Lambda3StreamingDetector:
    """Streaming detector configuration with optional scorer subset。"""
    names = scorer_names if scorer_names is not None else STREAMING_SCORER_NAMES
    scorers = [STREAMING_SCORER_FACTORIES[n](percentile) for n in names]
    return Lambda3StreamingDetector(
        scorers=scorers,
        calibration_ratio=0.15,
        min_calibration=50,
    )


def run_one(sample, n_features: int = 5, feature_window: int = 30,
            percentile: float = 99.0,
            scorer_names: list = None) -> Dict:
    sc_disp = ",".join(scorer_names) if scorer_names else "all"
    print(f"\n■ {sample.name}  n={sample.n}  #windows={len(sample.windows_ts)}  "
          f"features={n_features}  percentile={percentile}  "
          f"scorers=[{sc_disp}]  [STREAMING]")

    if n_features == 1:
        events = sample.values
    elif n_features == 5:
        events = expand_to_5d(sample.values, window=feature_window)
    else:
        raise ValueError(f"unsupported n_features={n_features}")

    detector = make_detector(percentile=percentile, scorer_names=scorer_names)
    t0 = time.perf_counter()
    result = detector.fit_predict(events)
    t_run = time.perf_counter() - t0

    score = result['score']
    binary = result['binary']
    cal_end = result['cal_end']
    print(f"  cal_end={cal_end}  total_run={t_run:.1f}s  "
          f"#flagged_frames={int(binary.sum())}/{sample.n - cal_end}")
    print(f"  thresholds: " + "  ".join(
        f"{k}={v:.3g}" for k, v in result['thresholds'].items()
    ))

    # NAB scoring with max-normalized continuous score
    nab_scores = score_all_profiles(sample, score)
    for prof, s in nab_scores.items():
        print(format_nab_score(s, 'streaming'))

    return {
        'name': sample.name,
        'n': sample.n,
        'n_windows': len(sample.windows_ts),
        'cal_end': cal_end,
        't_run': t_run,
        'nab_scores': nab_scores,
        'thresholds': result['thresholds'],
        'flag_count': int(binary.sum()),
    }


def _agg(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
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
    ap.add_argument('--percentile', type=float, default=99.0,
                    help='calibration threshold percentile (default 99.0)')
    ap.add_argument('--scorers', default='all',
                    help=f'使用する scorer のカンマ区切り (default all = {",".join(STREAMING_SCORER_NAMES)})。'
                         f'例: "jump,kernel"')
    ap.add_argument('--exclude-scorers', default='',
                    help='除外する scorer のカンマ区切り (leave-one-out ablation 用)')
    args = ap.parse_args()

    # Scorer selection
    if args.scorers.strip().lower() == 'all':
        included = list(STREAMING_SCORER_NAMES)
    else:
        included = [s.strip() for s in args.scorers.split(',') if s.strip()]
        for s in included:
            if s not in STREAMING_SCORER_NAMES:
                ap.error(f"unknown scorer {s!r}; valid: {STREAMING_SCORER_NAMES}")
    if args.exclude_scorers.strip():
        excluded = {s.strip() for s in args.exclude_scorers.split(',') if s.strip()}
        for s in excluded:
            if s not in STREAMING_SCORER_NAMES:
                ap.error(f"unknown exclude scorer {s!r}; valid: {STREAMING_SCORER_NAMES}")
        included = [s for s in included if s not in excluded]
    if not included:
        ap.error("no scorers selected after include/exclude")
    scorer_names_active = included

    print("=" * 110)
    print(f"NAB STREAMING benchmark  category={args.category}  "
          f"scorers=[{','.join(scorer_names_active)}]  "
          f"windows={args.windows_file}  features={args.features}  "
          f"percentile={args.percentile}")
    print("=" * 110)

    rows: List[Dict] = []
    for sample in iter_category(args.category, windows_file=args.windows_file):
        rows.append(run_one(sample, n_features=args.features,
                            feature_window=args.feature_window,
                            percentile=args.percentile,
                            scorer_names=scorer_names_active))

    if not rows:
        print("\n(no samples matched)")
        return

    print("\n" + "=" * 110)
    print(f"Aggregated streaming NAB scores ({len(rows)} files, category={args.category})")
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
