"""
スパース版逆問題JIT目的関数。

データfit項を「構造的に重要な (i,j) ペアのみ」で評価する。
TV／L1／jump／topo 正則化は密のまま（O(n_paths · n_events) で安いため）、
不活性フレームのΛも隣接活性フレームから自然に決まる。

数学的な等価性:
  full:   ||G - Λ^TΛ||²_F = Σ_{i,j ∈ [n]²} (G[i,j] - Σ_k Λ[k,i]Λ[k,j])²
  sparse: Σ_{(i,j) ∈ Ω}    (G[i,j] - Σ_k Λ[k,i]Λ[k,j])²

|Ω| << n² でも、Ω に対角 (i,i) を含めることで各フレームの Λ ノルムが
データから直接拘束され、識別可能性が保たれる。
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def inverse_problem_objective_sparse_jit(
    Lambda_matrix: np.ndarray,
    events: np.ndarray,
    pair_indices: np.ndarray,   # (n_pairs, 2) int64
    alpha: float,
    beta: float,
    jump_weight: float = 0.5,
):
    """スパース版データfit + 密なTV/L1/jump正則化"""
    n_paths, n_events = Lambda_matrix.shape
    n_pairs = pair_indices.shape[0]
    n_dim = events.shape[1]

    # ====== sparse data fit ======
    data_fit = 0.0
    for k in prange(n_pairs):
        i = pair_indices[k, 0]
        j = pair_indices[k, 1]

        # G[i,j] = <events[i], events[j]>  をオンザフライ計算
        gram_ij = 0.0
        for d in range(n_dim):
            gram_ij += events[i, d] * events[j, d]

        # reconstruction[i,j] = Σ_k Λ[k,i] Λ[k,j]
        recon_ij = 0.0
        for p in range(n_paths):
            recon_ij += Lambda_matrix[p, i] * Lambda_matrix[p, j]

        diff = gram_ij - recon_ij
        data_fit += diff * diff

    # ====== TV reg (dense - cheap, smooths Λ across all frames) ======
    tv_reg = 0.0
    for i in range(n_paths - 1):
        for j in range(n_events):
            tv_reg += np.abs(Lambda_matrix[i+1, j] - Lambda_matrix[i, j])
    for i in range(n_paths):
        for j in range(n_events - 1):
            tv_reg += np.abs(Lambda_matrix[i, j+1] - Lambda_matrix[i, j])

    # ====== L1 reg (dense) ======
    l1_reg = np.sum(np.abs(Lambda_matrix))

    # ====== jump reg (dense) ======
    jump_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        if n_events < 2:
            continue
        deltas = np.abs(path[1:] - path[:-1])
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + 2.5 * std_delta
        for delta in deltas:
            if delta > threshold:
                jump_reg += jump_weight * delta

    return data_fit + alpha * tv_reg + beta * l1_reg + jump_reg


@njit(parallel=True)
def inverse_problem_topo_objective_sparse_jit(
    Lambda_matrix: np.ndarray,
    events: np.ndarray,
    pair_indices: np.ndarray,
    alpha: float,
    beta: float,
    jump_weight: float = 0.5,
    topo_weight: float = 0.1,
):
    """スパース版（トポロジカル保存則ペナルティ込み）"""
    n_paths, n_events = Lambda_matrix.shape
    n_pairs = pair_indices.shape[0]
    n_dim = events.shape[1]

    # sparse data fit
    data_fit = 0.0
    for k in prange(n_pairs):
        i = pair_indices[k, 0]
        j = pair_indices[k, 1]
        gram_ij = 0.0
        for d in range(n_dim):
            gram_ij += events[i, d] * events[j, d]
        recon_ij = 0.0
        for p in range(n_paths):
            recon_ij += Lambda_matrix[p, i] * Lambda_matrix[p, j]
        diff = gram_ij - recon_ij
        data_fit += diff * diff

    # TV reg (dense)
    tv_reg = 0.0
    for i in range(n_paths - 1):
        for j in range(n_events):
            tv_reg += np.abs(Lambda_matrix[i+1, j] - Lambda_matrix[i, j])
    for i in range(n_paths):
        for j in range(n_events - 1):
            tv_reg += np.abs(Lambda_matrix[i, j+1] - Lambda_matrix[i, j])

    # L1 reg (dense)
    l1_reg = np.sum(np.abs(Lambda_matrix))

    # jump reg (dense)
    jump_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        if n_events < 2:
            continue
        deltas = np.abs(path[1:] - path[:-1])
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + 2.5 * std_delta
        for delta in deltas:
            if delta > threshold:
                jump_reg += jump_weight * delta

    # topo reg (dense)
    topo_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        if n_events < 3:
            continue
        phase = np.arctan2(path[1:], path[:-1])
        q_diff = phase[1:] - phase[:-1]
        for j in range(len(q_diff)):
            if q_diff[j] > np.pi:
                q_diff[j] -= 2 * np.pi
            elif q_diff[j] < -np.pi:
                q_diff[j] += 2 * np.pi
        topo_reg += np.sum(q_diff ** 2)

    return data_fit + alpha * tv_reg + beta * l1_reg + jump_reg + topo_weight * topo_reg
