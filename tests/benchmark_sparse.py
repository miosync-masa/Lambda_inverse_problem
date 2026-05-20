"""
Full逆問題ソルバ vs Sparse逆問題ソルバの head-to-head ベンチマーク。

各 (seed, n_events) で:
  1. ジャンプ構造を共通検出
  2. full / sparse 両方で paths を解く（wall time計測）
  3. それぞれの paths から物理量＋scorerをまわして AUC を比較
  4. paths同士の相関 / 異常スコア相関も出す

Usage::

    python -m tests.benchmark_sparse
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from lambda3_detector import L3Config, Lambda3Result
from lambda3_detector.analysis.multiscale_jumps import detect_multiscale_jumps
from lambda3_detector.analysis.physical_quantities import (
    classify_structures,
    compute_jump_aware_topology,
    compute_jump_conditional_entropies,
    compute_pulsation_energies,
)
from lambda3_detector.analysis.structure_tensor import inverse_problem_jump_constrained
from lambda3_detector.analysis.structure_tensor_sparse import solve_inverse_problem_sparse
from lambda3_detector.scorers import (
    HybridScorer,
    JumpScorer,
    KernelScorer,
    ScoreIntegrator,
    StructuralScorer,
)
from lambda3_detector import detector as _detector_mod
from lambda3_detector.core.adaptive_params import compute_adaptive_window_size

from tests.datasets import create_complex_natural_dataset


def _build_result(events: np.ndarray, paths: Dict[int, np.ndarray],
                  jump_structures: Dict) -> Lambda3Result:
    """既存detectorと同じ手順で paths から Lambda3Result を組み立てる"""
    charges, stabilities = compute_jump_aware_topology(paths, jump_structures)
    energies = compute_pulsation_energies(paths, jump_structures)
    entropies = compute_jump_conditional_entropies(paths, jump_structures)
    classifications = classify_structures(paths, charges, stabilities, jump_structures)
    return Lambda3Result(
        paths=paths,
        topological_charges=charges,
        stabilities=stabilities,
        energies=energies,
        entropies=entropies,
        classifications=classifications,
        jump_structures=jump_structures,
    )


def _score_all(events: np.ndarray, result: Lambda3Result,
               labels: np.ndarray) -> Dict[str, float]:
    """4 scorerを叩いて、本番デフォルト重みで統合した AUC を返す"""
    np.random.seed(0); jump = JumpScorer().score(events, result)
    np.random.seed(0); hybrid = HybridScorer().score(events, result)
    np.random.seed(0); kernel = KernelScorer(kernel_type=1, degree=7, coef0=1.0).score(events, result)
    np.random.seed(0); structural = StructuralScorer().score(events, result)

    integrator = ScoreIntegrator(default_weights={
        'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15,
    })
    combined = integrator.combine({
        'jump': jump, 'hybrid': hybrid, 'kernel': kernel, 'structural': structural,
    })
    auc = roc_auc_score(labels, combined)
    top10 = float(np.mean(labels[np.argsort(combined)[-10:]]))

    # hybridだけのAUCも参考に
    auc_hybrid_only = roc_auc_score(labels, hybrid)
    return {
        'auc_full': float(auc), 'top10': top10,
        'auc_hybrid_only': float(auc_hybrid_only),
        'jump': jump, 'hybrid': hybrid, 'kernel': kernel, 'structural': structural,
        'combined': combined,
    }


def benchmark_one(events: np.ndarray, labels: np.ndarray, config: L3Config,
                   expand_window: int = 5, n_static_samples: int = 50,
                   verbose: bool = True) -> Dict:
    """1データセットでの full vs sparse 比較"""

    # adaptive window（detector.analyzeと同じ前処理）
    aw = compute_adaptive_window_size(events)
    _detector_mod._config.update_global_constants(aw)

    # 共通: ジャンプ検出
    jump_structures = detect_multiscale_jumps(events)

    n_paths = config.n_paths
    n_events = events.shape[0]

    # ====== full solver ======
    t0 = time.perf_counter()
    paths_full = inverse_problem_jump_constrained(
        events, jump_structures, n_paths, config.alpha, config.beta,
    )
    t_full = time.perf_counter() - t0
    res_full = _build_result(events, paths_full, jump_structures)
    metrics_full = _score_all(events, res_full, labels)

    # ====== sparse solver ======
    t0 = time.perf_counter()
    paths_sparse, stats = solve_inverse_problem_sparse(
        events, jump_structures, n_paths, config.alpha, config.beta,
        expand_window=expand_window, n_static_samples=n_static_samples,
        verbose=verbose,
    )
    t_sparse = time.perf_counter() - t0
    res_sparse = _build_result(events, paths_sparse, jump_structures)
    metrics_sparse = _score_all(events, res_sparse, labels)

    # paths 同士の相関 (path index ごとに correlation を取って平均)
    path_corrs = []
    for i in range(n_paths):
        # 符号の不定性を吸収（|·|で比較）
        a = np.abs(paths_full[i])
        b = np.abs(paths_sparse[i])
        c = np.corrcoef(a, b)[0, 1]
        if np.isnan(c):
            c = 0.0
        path_corrs.append(c)

    # 最終異常スコアの相関
    combined_corr = float(np.corrcoef(metrics_full['combined'], metrics_sparse['combined'])[0, 1])

    return {
        'n_events': n_events,
        'sparse_stats': stats,
        'time_full': t_full,
        'time_sparse': t_sparse,
        'speedup': t_full / max(t_sparse, 1e-9),
        'auc_full': metrics_full['auc_full'],
        'auc_sparse': metrics_sparse['auc_full'],
        'auc_delta': metrics_sparse['auc_full'] - metrics_full['auc_full'],
        'top10_full': metrics_full['top10'],
        'top10_sparse': metrics_sparse['top10'],
        'auc_hybrid_full': metrics_full['auc_hybrid_only'],
        'auc_hybrid_sparse': metrics_sparse['auc_hybrid_only'],
        'mean_path_corr': float(np.mean(path_corrs)),
        'min_path_corr': float(np.min(path_corrs)),
        'combined_score_corr': combined_corr,
    }


def main():
    print("=" * 84)
    print("Sparse vs Full inverse-problem solver head-to-head")
    print("=" * 84)

    seeds = [42, 123, 456]
    sizes = [200, 300, 500]
    config = L3Config()

    all_results = []
    for seed in seeds:
        for n_events in sizes:
            np.random.seed(seed)
            events, labels, _ = create_complex_natural_dataset(
                n_events=n_events, n_features=10, anomaly_ratio=0.15,
            )
            print(f"\n[seed={seed}  n_events={n_events}]")
            r = benchmark_one(events, labels, config)
            r['seed'] = seed
            all_results.append(r)
            print(f"  full   : t={r['time_full']:7.2f}s  AUC={r['auc_full']:.4f}  "
                  f"hybridAUC={r['auc_hybrid_full']:.4f}  Top10={r['top10_full']:.2f}")
            print(f"  sparse : t={r['time_sparse']:7.2f}s  AUC={r['auc_sparse']:.4f}  "
                  f"hybridAUC={r['auc_hybrid_sparse']:.4f}  Top10={r['top10_sparse']:.2f}")
            print(f"           speedup={r['speedup']:.2f}x  ΔAUC={r['auc_delta']:+.4f}  "
                  f"path_corr(mean/min)={r['mean_path_corr']:.3f}/{r['min_path_corr']:.3f}  "
                  f"score_corr={r['combined_score_corr']:.3f}")

    print("\n" + "=" * 84)
    print("集計 (mean across seeds, per n_events)")
    print("=" * 84)
    for n in sizes:
        subset = [r for r in all_results if r['n_events'] == n]
        if not subset:
            continue
        print(f"\n  n_events={n}:")
        print(f"    time:    full={np.mean([r['time_full'] for r in subset]):7.2f}s  "
              f"sparse={np.mean([r['time_sparse'] for r in subset]):7.2f}s  "
              f"speedup={np.mean([r['speedup'] for r in subset]):.2f}x")
        print(f"    AUC:     full={np.mean([r['auc_full'] for r in subset]):.4f}  "
              f"sparse={np.mean([r['auc_sparse'] for r in subset]):.4f}  "
              f"ΔAUC={np.mean([r['auc_delta'] for r in subset]):+.4f}")
        print(f"    Top10:   full={np.mean([r['top10_full'] for r in subset]):.2f}  "
              f"sparse={np.mean([r['top10_sparse'] for r in subset]):.2f}")
        print(f"    pair-reduction: "
              f"{np.mean([r['sparse_stats']['reduction_ratio'] for r in subset])*100:.1f}%")
        print(f"    path corr (mean): "
              f"{np.mean([r['mean_path_corr'] for r in subset]):.3f}")
        print(f"    combined score corr: "
              f"{np.mean([r['combined_score_corr'] for r in subset]):.3f}")


if __name__ == "__main__":
    main()
