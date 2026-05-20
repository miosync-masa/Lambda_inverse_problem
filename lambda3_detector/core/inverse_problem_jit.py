"""
逆問題関連のJITカーネル: 構造テンソル推定の目的関数と
Tikhonov式の再構成誤差／ハイブリッド異常スコア。
"""

import numpy as np
from numba import njit, prange

from .pulsation_jit import compute_pulsation_energy_from_path, find_jump_indices


@njit(parallel=True)
def inverse_problem_objective_jit(Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5):
    """逆問題の目的関数（JIT最適化版）"""
    n_paths, n_events = Lambda_matrix.shape
    reconstruction = np.zeros((n_events, n_events))
    for i in prange(n_events):
        for j in range(n_events):
            for k in range(n_paths):
                reconstruction[i, j] += Lambda_matrix[k, i] * Lambda_matrix[k, j]
    data_fit = np.sum((events_gram - reconstruction)**2)

    tv_reg = 0.0
    for i in range(n_paths - 1):
        for j in range(n_events):
            tv_reg += np.abs(Lambda_matrix[i+1, j] - Lambda_matrix[i, j])
    for i in range(n_paths):
        for j in range(n_events - 1):
            tv_reg += np.abs(Lambda_matrix[i, j+1] - Lambda_matrix[i, j])

    l1_reg = np.sum(np.abs(Lambda_matrix))

    # ジャンプ正則化
    jump_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        n_delta = n_events - 1
        if n_delta == 0:
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
def inverse_problem_topo_objective_jit(
    Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5, topo_weight=0.1
):
    """
    逆問題の目的関数（JIT最適化版）
    - data_fit: 再構成誤差
    - tv_reg: Total Variation正則化
    - l1_reg: L1正則化
    - jump_reg: ジャンプ正則化
    - topo_reg: トポロジカル保存則ペナルティ（QΛ）
    """
    n_paths, n_events = Lambda_matrix.shape
    reconstruction = np.zeros((n_events, n_events))
    for i in prange(n_events):
        for j in range(n_events):
            for k in range(n_paths):
                reconstruction[i, j] += Lambda_matrix[k, i] * Lambda_matrix[k, j]
    data_fit = np.sum((events_gram - reconstruction)**2)

    # Total Variation正則化
    tv_reg = 0.0
    for i in range(n_paths - 1):
        for j in range(n_events):
            tv_reg += np.abs(Lambda_matrix[i+1, j] - Lambda_matrix[i, j])
    for i in range(n_paths):
        for j in range(n_events - 1):
            tv_reg += np.abs(Lambda_matrix[i, j+1] - Lambda_matrix[i, j])

    # L1正則化
    l1_reg = np.sum(np.abs(Lambda_matrix))

    # ジャンプ正則化
    jump_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        n_delta = n_events - 1
        if n_delta == 0:
            continue
        deltas = np.abs(path[1:] - path[:-1])
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + 2.5 * std_delta
        for delta in deltas:
            if delta > threshold:
                jump_reg += jump_weight * delta

    # === QΛトポロジカル保存ペナルティ追加 ===
    topo_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        if n_events < 3:
            continue
        # 位相差列
        phase = np.arctan2(path[1:], path[:-1])
        q_diff = phase[1:] - phase[:-1]
        # 2π補正
        for j in range(len(q_diff)):
            if q_diff[j] > np.pi:
                q_diff[j] -= 2 * np.pi
            elif q_diff[j] < -np.pi:
                q_diff[j] += 2 * np.pi
        # 連続性ペナルティ（2乗和）
        topo_reg += np.sum(q_diff ** 2)

    # === 全体損失 ===
    return data_fit + alpha * tv_reg + beta * l1_reg + jump_reg + topo_weight * topo_reg


@njit
def compute_lambda3_reconstruction_error(paths_matrix: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Lambda³再構成誤差の計算（Tikhonov精神の継承）"""
    n_paths, n_events = paths_matrix.shape
    n_features = events.shape[1]

    # 1. 観測データのGram行列（正規化済み）
    events_gram = np.zeros((n_events, n_events))
    for i in range(n_events):
        for j in range(n_events):
            events_gram[i, j] = np.dot(events[i], events[j])

    # Gram行列の正規化（スケール不変性）
    gram_norm = np.sqrt(np.trace(events_gram @ events_gram))
    if gram_norm > 0:
        events_gram /= gram_norm

    # 2. Lambda³構造による再構成
    recon_gram = np.zeros((n_events, n_events))
    for k in range(n_paths):
        for i in range(n_events):
            for j in range(n_events):
                recon_gram[i, j] += paths_matrix[k, i] * paths_matrix[k, j]

    # 再構成の正規化
    recon_norm = np.sqrt(np.trace(recon_gram @ recon_gram))
    if recon_norm > 0:
        recon_gram /= recon_norm

    # 3. イベントごとの再構成誤差
    event_errors = np.zeros(n_events)
    for i in range(n_events):
        row_error = 0.0
        for j in range(n_events):
            diff = events_gram[i, j] - recon_gram[i, j]
            row_error += diff * diff
        event_errors[i] = np.sqrt(row_error)

    return event_errors


@njit(parallel=True)
def compute_lambda3_hybrid_tikhonov_scores(
    paths_matrix: np.ndarray,
    events: np.ndarray,
    charges: np.ndarray,
    stabilities: np.ndarray,
    alpha: float = 0.5,
    jump_scale: float = 2.0,
    use_union: bool = True,
    w_topo: float = 0.2,
    w_pulse: float = 0.3,
) -> np.ndarray:
    """Lambda³ハイブリッドTikhonov融合異常スコア"""
    n_paths, n_events = paths_matrix.shape

    # 全体誤差
    errors_all = compute_lambda3_reconstruction_error(paths_matrix, events)

    # ジャンプindex
    if use_union:
        idx_set = set()
        for k in range(n_paths):
            idxs = find_jump_indices(paths_matrix[k], jump_scale)
            for idx in idxs:
                idx_set.add(idx+1)
        jump_idx = np.array(list(idx_set), dtype=np.int64)
    else:
        Qarr = np.array([np.sum(np.diff(paths_matrix[k])) for k in range(n_paths)])
        main_idx = np.argmax(np.abs(Qarr))
        idxs = find_jump_indices(paths_matrix[main_idx], jump_scale)
        jump_idx = idxs + 1

    # ジャンプ誤差ベクトル
    jump_error = np.zeros_like(errors_all)
    for idx in jump_idx:
        if idx < len(jump_error):
            jump_error[idx] = errors_all[idx]

    # パスごとの異常度
    path_anomaly_scores = np.zeros(n_paths)
    for p in prange(n_paths):
        path = paths_matrix[p]
        topo_score = np.abs(charges[p]) + 0.5 * stabilities[p]
        # パスから拍動エネルギーを計算（構造テンソル用）
        jump_int, asymm, pulse_pow = compute_pulsation_energy_from_path(path)
        pulse_score = 0.4 * jump_int + 0.3 * np.abs(asymm) + 0.3 * pulse_pow
        path_anomaly_scores[p] = w_topo * topo_score + w_pulse * pulse_score

    # イベントごと加重
    structural_component = np.zeros(n_events)
    for i in prange(n_events):
        for p in range(n_paths):
            contribution = np.abs(paths_matrix[p, i])
            structural_component[i] += contribution * path_anomaly_scores[p]

    # ハイブリッド合成
    hybrid_score = alpha * errors_all + (1 - alpha) * jump_error
    event_scores = hybrid_score + structural_component

    # 標準化
    mean_score = np.mean(event_scores)
    std_score = np.std(event_scores)
    if std_score > 0:
        event_scores = (event_scores - mean_score) / std_score

    return event_scores
