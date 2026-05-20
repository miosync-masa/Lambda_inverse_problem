"""
Shannon / Rényi / Tsallis ほか6指標の高速エントロピー計算。
"""

import numpy as np
from numba import njit


@njit
def compute_entropy_shannon_jit(path: np.ndarray, eps: float = 1e-10) -> float:
    """Shannonエントロピーの高速計算"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    entropy = 0.0
    for p in norm_path:
        if p > 0:
            entropy -= p * np.log(p)

    return entropy


@njit
def compute_entropy_renyi_jit(path: np.ndarray, alpha: float = 2.0, eps: float = 1e-10) -> float:
    """Renyiエントロピーの高速計算"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    if alpha == 1.0:
        return compute_entropy_shannon_jit(path, eps)

    sum_p_alpha = 0.0
    for p in norm_path:
        sum_p_alpha += p ** alpha

    return (1.0 / (1.0 - alpha)) * np.log(sum_p_alpha)


@njit
def compute_entropy_tsallis_jit(path: np.ndarray, q: float = 1.5, eps: float = 1e-10) -> float:
    """Tsallisエントロピーの高速計算"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    if q == 1.0:
        return compute_entropy_shannon_jit(path, eps)

    sum_p_q = 0.0
    for p in norm_path:
        sum_p_q += p ** q

    return (1.0 - sum_p_q) / (q - 1.0)


@njit
def compute_all_entropies_jit(path: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """全エントロピー指標の高速計算（配列で返す）"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    # 6つの指標を計算
    entropies = np.zeros(6)

    # Shannon
    shannon = 0.0
    for p in norm_path:
        if p > 0:
            shannon -= p * np.log(p)
    entropies[0] = shannon

    # Renyi (α=2)
    sum_p2 = 0.0
    for p in norm_path:
        sum_p2 += p ** 2
    entropies[1] = -np.log(sum_p2)

    # Tsallis (q=1.5)
    sum_p15 = 0.0
    for p in norm_path:
        sum_p15 += p ** 1.5
    entropies[2] = (1.0 - sum_p15) / 0.5

    # Max
    entropies[3] = np.max(norm_path)

    # Min
    entropies[4] = np.min(norm_path)

    # Variance
    mean_p = np.mean(norm_path)
    var = 0.0
    for p in norm_path:
        var += (p - mean_p) ** 2
    entropies[5] = var / len(norm_path)

    return entropies
