"""
構造テンソル推定（逆問題ソルバ）。

通常版と「ジャンプ構造制約付き」版の両方を提供し、それぞれ
トポロジカル保存則あり／なしを別々に解いて要素ごとに |·|-max 合成する。
"""

from typing import Dict

import numpy as np
from scipy.optimize import minimize

from ..core.inverse_problem_jit import (
    inverse_problem_objective_jit,
    inverse_problem_topo_objective_jit,
)
from ..core.jumps_jit import compute_jump_consistency_term


def solve_inverse_problem(events: np.ndarray,
                          n_paths: int,
                          alpha: float,
                          beta: float,
                          topo_weight: float = 0.1) -> Dict[int, np.ndarray]:
    """
    構造テンソル推定（通常＆トポロジカル保存破れ両対応, MAX合成）
    """
    events_gram = np.ascontiguousarray(events @ events.T)
    _, V = np.linalg.eigh(events_gram)
    Lambda_init = V[:, -n_paths:].T.flatten()

    # 1. 通常逆問題
    def objective_no_topo(Lambda_flat):
        Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
        return inverse_problem_objective_jit(
            Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5
        )

    result_no_topo = minimize(
        objective_no_topo,
        Lambda_init,
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    Lambda_no_topo = result_no_topo.x.reshape(n_paths, events.shape[0])

    # 2. トポロジカルペナルティあり
    def objective_with_topo(Lambda_flat):
        Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
        return inverse_problem_topo_objective_jit(
            Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5, topo_weight=topo_weight
        )

    result_with_topo = minimize(
        objective_with_topo,
        Lambda_init,
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    Lambda_with_topo = result_with_topo.x.reshape(n_paths, events.shape[0])

    # 3. パスごと/イベントごとにMAX合成（絶対値ベースなど）
    Lambda_max = np.maximum(np.abs(Lambda_no_topo), np.abs(Lambda_with_topo))

    # 4. 返却（正規化）
    return {i: path / (np.linalg.norm(path) + 1e-8) for i, path in enumerate(Lambda_max)}


def inverse_problem_jump_constrained(events: np.ndarray,
                                     jump_structures: Dict,
                                     n_paths: int,
                                     alpha: float,
                                     beta: float,
                                     topo_weight: float = 0.1) -> Dict[int, np.ndarray]:
    """
    ジャンプ構造を活用した逆問題
    - トポロジカル保存律「あり」と「なし」両方計算し、要素ごとにMAX合成して返す
    """
    jump_mask = jump_structures['integrated']['unified_jumps']
    jump_weights = jump_structures['integrated']['jump_importance']
    events_gram = np.ascontiguousarray(events @ events.T)
    Lambda_init = _initialize_with_jump_structure(events, jump_mask, jump_weights, n_paths)

    # 1. 保存律なしで最適化
    def objective_no_topo(Lambda_flat):
        Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
        base_obj = inverse_problem_objective_jit(
            Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5)
        jump_term = compute_jump_consistency_term(Lambda_matrix, jump_mask, jump_weights)
        return base_obj + jump_term

    result_no_topo = minimize(
        objective_no_topo,
        Lambda_init.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    Lambda_no_topo = result_no_topo.x.reshape(n_paths, events.shape[0])

    # 2. 保存律ありで最適化
    def objective_with_topo(Lambda_flat):
        Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
        base_obj = inverse_problem_topo_objective_jit(
            Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5, topo_weight=topo_weight)
        jump_term = compute_jump_consistency_term(Lambda_matrix, jump_mask, jump_weights)
        return base_obj + jump_term

    result_with_topo = minimize(
        objective_with_topo,
        Lambda_init.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    Lambda_with_topo = result_with_topo.x.reshape(n_paths, events.shape[0])

    # 3. MAX合成
    Lambda_max = np.maximum(np.abs(Lambda_no_topo), np.abs(Lambda_with_topo))

    # 4. 各パス正規化して辞書で返却
    return {i: path / (np.linalg.norm(path) + 1e-8) for i, path in enumerate(Lambda_max)}


def _initialize_with_jump_structure(events: np.ndarray,
                                    jump_mask: np.ndarray,
                                    jump_weights: np.ndarray,
                                    n_paths: int) -> np.ndarray:
    """ジャンプ構造を反映した初期値生成"""
    n_events = events.shape[0]
    Lambda_init = np.zeros((n_paths, n_events))

    # 固有値分解ベース
    _, V = np.linalg.eigh(events @ events.T)
    base_paths = V[:, -n_paths:].T

    # ジャンプ位置で不連続性を導入
    for p in range(n_paths):
        Lambda_init[p] = base_paths[p]

        # ジャンプ位置での値を強調
        for i in range(n_events):
            if jump_mask[i]:
                if p % 2 == 0:
                    Lambda_init[p, i] *= (1 + jump_weights[i])
                else:
                    Lambda_init[p, i] *= -(1 + jump_weights[i])

    return Lambda_init
