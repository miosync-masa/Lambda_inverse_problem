"""
スパース版構造テンソル推定。

ジャンプ近傍 × ジャンプ近傍を密、ジャンプ近傍 × 静的領域をサンプリング、
静的領域同士もサンプリング、対角は全フレーム必須、というペア選択で
データfit項の計算量を O(n²) → O(|Ω|) に削減する。

不活性フレームの Λ は dense TV 正則化と隣接活性フレームを介して
最適化中に決まるため、後付けの interpolation は不要。
"""

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from ..core.inverse_problem_sparse_jit import (
    inverse_problem_objective_sparse_jit,
    inverse_problem_topo_objective_sparse_jit,
)
from ..core.jumps_jit import compute_jump_consistency_term
from .structure_tensor import _initialize_with_jump_structure


def compute_significant_pairs(events: np.ndarray,
                              jump_structures: Dict,
                              expand_window: int = 5,
                              n_static_samples: int = 50,
                              seed: int = 0) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    構造的に重要な (i,j) ペアを抽出する。

    含めるもの:
      1. ジャンプ ±expand_window のフレーム同士（dense, 全方向）
      2. ジャンプ近傍 × 静的領域サンプル（両方向）
      3. 静的領域サンプル同士（dense）
      4. 対角 (i,i) は全フレーム強制（Λの各列のノルムをアンカー）

    Returns:
        pairs:  (n_pairs, 2) int64 array
        stats:  カウント情報（debug/report用）
    """
    n_events = len(events)
    rng = np.random.default_rng(seed)

    pairs = set()

    # 1. expanded jump vicinity
    unified = jump_structures['integrated']['unified_jumps']
    jump_indices = np.where(unified)[0]
    expanded = set()
    for ji in jump_indices:
        for offset in range(-expand_window, expand_window + 1):
            idx = int(ji) + offset
            if 0 <= idx < n_events:
                expanded.add(idx)
    expanded_list = sorted(expanded)

    # jump-vicinity × jump-vicinity (dense, both directions)
    for i in expanded_list:
        for j in expanded_list:
            pairs.add((i, j))

    # 2. 静的領域サンプリング (jump近傍を除外)
    static_pool = sorted(set(range(n_events)) - expanded)
    if len(static_pool) > 0:
        n_samples = min(n_static_samples, len(static_pool))
        static_samples = sorted(rng.choice(static_pool, size=n_samples, replace=False).tolist())
    else:
        static_samples = []

    # 3. jump-vicinity × static (両方向)
    for ji in expanded_list:
        for sp in static_samples:
            pairs.add((ji, sp))
            pairs.add((sp, ji))

    # 4. static × static
    for s1 in static_samples:
        for s2 in static_samples:
            pairs.add((s1, s2))

    # 5. 対角 (anchor) — 全フレーム必須
    for i in range(n_events):
        pairs.add((i, i))

    pair_array = np.array(sorted(pairs), dtype=np.int64)

    stats = {
        'n_events': n_events,
        'n_full_pairs': n_events * n_events,
        'n_jump_indices': len(jump_indices),
        'n_expanded': len(expanded),
        'n_static_samples': len(static_samples),
        'n_selected_pairs': len(pair_array),
        'reduction_ratio': 1.0 - len(pair_array) / (n_events * n_events),
    }
    return pair_array, stats


def solve_inverse_problem_sparse(events: np.ndarray,
                                  jump_structures: Dict,
                                  n_paths: int,
                                  alpha: float,
                                  beta: float,
                                  topo_weight: float = 0.1,
                                  expand_window: int = 5,
                                  n_static_samples: int = 50,
                                  use_jump_init: bool = True,
                                  verbose: bool = False) -> Tuple[Dict[int, np.ndarray], Dict[str, int]]:
    """
    スパース版逆問題ソルバ（ジャンプ構造制約付き、no_topo / with_topo の MAX 合成）。

    Returns:
        paths:  {i: Λ[i]} 正規化済み
        stats:  ペア統計
    """
    n_events = events.shape[0]
    pair_array, stats = compute_significant_pairs(
        events, jump_structures,
        expand_window=expand_window, n_static_samples=n_static_samples,
    )

    if verbose:
        print(f"  sparse pairs: {stats['n_selected_pairs']:,} / "
              f"{stats['n_full_pairs']:,} "
              f"({stats['reduction_ratio']*100:.1f}% reduction)")

    # 初期値: 既存の jump-aware initialization を流用（cheap, O(n³) で一度きり）
    if use_jump_init:
        jump_mask = jump_structures['integrated']['unified_jumps']
        jump_weights = jump_structures['integrated']['jump_importance']
        Lambda_init = _initialize_with_jump_structure(events, jump_mask, jump_weights, n_paths)
    else:
        events_gram = np.ascontiguousarray(events @ events.T)
        _, V = np.linalg.eigh(events_gram)
        Lambda_init = V[:, -n_paths:].T

    # ====== no-topo ======
    jump_mask = jump_structures['integrated']['unified_jumps']
    jump_weights = jump_structures['integrated']['jump_importance']

    def objective_no_topo(Lambda_flat):
        Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, n_events))
        base = inverse_problem_objective_sparse_jit(
            Lambda_matrix, events, pair_array, alpha, beta, jump_weight=0.5)
        jump_term = compute_jump_consistency_term(Lambda_matrix, jump_mask, jump_weights)
        return base + jump_term

    res_no = minimize(
        objective_no_topo, Lambda_init.flatten(),
        method='L-BFGS-B', options={'maxiter': 1000},
    )
    L_no = res_no.x.reshape(n_paths, n_events)

    # ====== with-topo ======
    def objective_with_topo(Lambda_flat):
        Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, n_events))
        base = inverse_problem_topo_objective_sparse_jit(
            Lambda_matrix, events, pair_array, alpha, beta, jump_weight=0.5, topo_weight=topo_weight)
        jump_term = compute_jump_consistency_term(Lambda_matrix, jump_mask, jump_weights)
        return base + jump_term

    res_to = minimize(
        objective_with_topo, Lambda_init.flatten(),
        method='L-BFGS-B', options={'maxiter': 1000},
    )
    L_to = res_to.x.reshape(n_paths, n_events)

    # MAX 合成
    L_max = np.maximum(np.abs(L_no), np.abs(L_to))
    paths = {i: path / (np.linalg.norm(path) + 1e-8) for i, path in enumerate(L_max)}
    return paths, stats
