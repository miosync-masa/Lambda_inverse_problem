"""
Per-scorer ablation runner — Lambda³-R (regime-aware) leave-one-out.

各 scorer を 1 つずつ除外し、NAB score への寄与を測定する。
論文の ablation table を生成する用途。

Usage:
    # 単一カテゴリで全 6 scorer leave-one-out + baseline (= 7 runs)
    python -m tests.ablation_runner --category realKnownCause

    # 全カテゴリで実行 (52 file × 7 = 重い、Colab GPU 推奨)
    python -m tests.ablation_runner --all-categories

各実行で benchmark_nab_regime.py を呼び、最後に summary table を表示。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Dict, List, Tuple

SCORERS = ['jump', 'gradual', 'drift', 'recon', 'kernel', 'struct']
CATEGORIES_DEFAULT = [
    'realKnownCause',
    'realAWSCloudwatch',
    'realTraffic',
    'realAdExchange',
    'artificialWithAnomaly',
    'realTweets',
]


def parse_3prof_mean(output: str) -> float:
    """benchmark output から '3-profile mean = XX.XX' を抜き出す。"""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('3-profile mean'):
            # "3-profile mean =  72.02"
            parts = line.split('=')
            if len(parts) == 2:
                try:
                    return float(parts[1].strip())
                except ValueError:
                    pass
    return float('nan')


def run_single(category: str, excluded: str) -> Tuple[float, float]:
    """1 ablation run: NAB 3-prof mean と所要時間を返す。

    excluded='' で baseline (全 6 scorer)、他は leave-one-out。
    """
    cmd = [
        sys.executable, '-m', 'tests.benchmark_nab_regime',
        '--category', category,
        '--threshold-method', 'trimmed_percentile',
    ]
    if excluded:
        cmd += ['--exclude-scorers', excluded]

    label = excluded if excluded else 'baseline (all 6)'
    print(f"\n▶▶ {category}  [{label}]")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  ERROR rc={proc.returncode}")
        print(proc.stderr[-2000:])
        return float('nan'), elapsed

    score = parse_3prof_mean(proc.stdout)
    print(f"  → 3-prof mean = {score:.2f}  ({elapsed:.0f}s)")
    return score, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', default='realKnownCause',
                    help='ablation を実行するカテゴリ (default realKnownCause)')
    ap.add_argument('--all-categories', action='store_true',
                    help='全 6 カテゴリで実行 (~重い)')
    args = ap.parse_args()

    cats = CATEGORIES_DEFAULT if args.all_categories else [args.category]

    results: Dict[str, Dict[str, float]] = {}  # category → {excluded → score}
    for cat in cats:
        results[cat] = {}
        # baseline
        baseline, _ = run_single(cat, '')
        results[cat]['baseline'] = baseline
        # leave-one-out
        for scorer in SCORERS:
            sc, _ = run_single(cat, scorer)
            results[cat][scorer] = sc

    # Summary table
    print("\n" + "=" * 110)
    print("Ablation Summary  (Lambda³-R, threshold_method=trimmed_percentile)")
    print("=" * 110)
    headers = ['Category', 'baseline'] + [f'-{s}' for s in SCORERS]
    print("  " + "  ".join(f"{h:>14}" for h in headers))
    print("  " + "  ".join("-" * 14 for _ in headers))
    for cat in cats:
        row = [cat[:14]]
        b = results[cat]['baseline']
        row.append(f"{b:>14.2f}")
        for s in SCORERS:
            v = results[cat][s]
            delta = v - b
            row.append(f"{v:>7.2f}({delta:+.2f})")
        print("  " + "  ".join(row))

    print("\n注: delta は baseline からの差分。大きく下がる scorer = 重要寄与。")
    print("    例: -jump=70.50 (-1.52) → jump scorer 除外で 1.52 ポイント下落 → 重要")


if __name__ == '__main__':
    main()
