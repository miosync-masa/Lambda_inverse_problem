"""
GPU 版 scorers（Step 1+2 では KernelScorer のみ）。

KernelScorer は CPU 版で O(n²) の Gram 行列 + O(n²) の再構成行列を構築する。
1.8k 行で 7.5 秒、10k 行では数百秒に達する次のボトルネック。

GPU では:
  - Gram 行列を一度の matmul (n×d × d×n) or 距離計算で構築
  - 再構成 K_recon = K * (paths.T @ paths) も element-wise / matmul
  - Frobenius 正規化と行ノルムは cp.sum / cp.sqrt で 1 行
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .backend import cp, DEFAULT_DTYPE, ensure_gpu, to_cpu


# =============================================================================
# Pairwise primitives (GPU)
# =============================================================================

def _pairwise_sq_dist_gpu(X: "cp.ndarray") -> "cp.ndarray":
    """||x_i - x_j||² の n×n 行列。X: (n, d) float32 → (n, n) float32。"""
    sq = cp.sum(X * X, axis=1)  # (n,)
    G = X @ X.T                  # (n, n)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    return cp.maximum(D2, 0.0)   # 数値誤差で負になり得るので clamp


def _pairwise_l1_dist_gpu(X: "cp.ndarray", chunk: int = 512) -> "cp.ndarray":
    """Σ_d |x_{id} - x_{jd}| の n×n 行列。

    (n, n, d) の中間配列を避けるため、行ブロック単位で計算。
    """
    n = X.shape[0]
    out = cp.empty((n, n), dtype=X.dtype)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # X[s:e, None, :] - X[None, :, :]  → (chunk, n, d)
        diff = cp.abs(X[s:e, None, :] - X[None, :, :])
        out[s:e] = cp.sum(diff, axis=-1)
    return out


# =============================================================================
# Kernel Gram matrix (GPU)
# =============================================================================

def compute_kernel_gram_matrix_gpu(
    data: np.ndarray,
    kernel_type: int = 0,
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
    alpha: float = 0.01,
    period: float = 10.0,
    length_scale: float = 1.0,
) -> "cp.ndarray":
    """CPU 版 compute_kernel_gram_matrix と数値的に等価な GPU 実装。"""
    X = ensure_gpu(data)  # (n, d) float32
    n = X.shape[0]

    if kernel_type == 0:  # RBF
        D2 = _pairwise_sq_dist_gpu(X)
        K = cp.exp(-gamma * D2)
    elif kernel_type == 1:  # Polynomial
        G = X @ X.T
        K = (G + coef0) ** degree
    elif kernel_type == 2:  # Sigmoid
        G = X @ X.T
        K = cp.tanh(alpha * G + coef0)
    elif kernel_type == 3:  # Laplacian
        D1 = _pairwise_l1_dist_gpu(X)
        K = cp.exp(-gamma * D1)
    elif kernel_type == 4:  # Periodic
        # periodic: exp(-2 Σ_d sin²(π|x_id-x_jd|/period) / length_scale²)
        n_total = X.shape[0]
        K = cp.empty((n_total, n_total), dtype=X.dtype)
        chunk = 512
        inv_period = float(np.pi / period)
        inv_ls2 = float(1.0 / (length_scale ** 2))
        for s in range(0, n_total, chunk):
            e = min(s + chunk, n_total)
            diff = cp.abs(X[s:e, None, :] - X[None, :, :])
            sin_term = cp.sin(inv_period * diff)
            K[s:e] = cp.exp(-2.0 * inv_ls2 * cp.sum(sin_term * sin_term, axis=-1))
    else:  # default: Laplacian
        D1 = _pairwise_l1_dist_gpu(X)
        K = cp.exp(-gamma * D1)

    # 対称化（数値誤差吸収）
    K = 0.5 * (K + K.T)
    return K.astype(DEFAULT_DTYPE)


# =============================================================================
# Kernel anomaly score (GPU)
# =============================================================================

def kernel_anomaly_scores_gpu(
    events: np.ndarray,
    paths_dict: dict,            # {i: ndarray(n_events,)} from Lambda3Result.paths
    kernel_type: int = 0,
    **kernel_params,
) -> np.ndarray:
    """CPU 版 compute_kernel_anomaly_scores_with_params と等価な GPU 実装。

    Returns:
        kernel_scores: (n_events,) float64 numpy
    """
    X = ensure_gpu(events)
    n_events = X.shape[0]

    K = compute_kernel_gram_matrix_gpu(events, kernel_type=kernel_type, **kernel_params)

    # paths_matrix (n_paths, n_events) on GPU
    paths_matrix = cp.asarray(
        np.stack(list(paths_dict.values())), dtype=DEFAULT_DTYPE,
    )

    # K_recon[i, j] = K[i, j] * Σ_k paths[k, i] * paths[k, j]
    P = paths_matrix.T @ paths_matrix   # (n_events, n_events)
    K_recon = K * P

    # Frobenius 正規化 (K は対称なので trace(K@K) = sum(K²))
    K_norm = cp.sqrt(cp.sum(K * K))
    if float(K_norm) > 0.0:
        K = K / K_norm
    recon_norm = cp.sqrt(cp.sum(K_recon * K_recon))
    if float(recon_norm) > 0.0:
        K_recon = K_recon / recon_norm

    diff = K - K_recon
    row_err = cp.sqrt(cp.sum(diff * diff, axis=1))   # (n_events,)
    return to_cpu(row_err).astype(np.float64)
