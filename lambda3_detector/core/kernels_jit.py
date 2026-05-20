"""
JITカーネル関数群（RBF / Polynomial / Sigmoid / Laplacian / Periodic）と
共通のGram行列計算。
"""

import numpy as np
from numba import njit, prange

from ..config import (
    DEFAULT_KERNEL_TYPE,
    DEFAULT_GAMMA,
    DEFAULT_DEGREE,
    DEFAULT_COEF0,
    DEFAULT_ALPHA,
)


@njit
def periodic_kernel(x: np.ndarray, y: np.ndarray,
                   period: float = 1.0,
                   length_scale: float = 1.0) -> float:
    """周期カーネル（周期的パターン検出用）"""
    diff = x - y
    # 周期的距離
    periodic_dist = np.sin(np.pi * np.abs(diff) / period)
    return np.exp(-2 * np.sum(periodic_dist ** 2) / (length_scale ** 2))


@njit
def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """RBFカーネル（ガウシアンカーネル）"""
    diff = x - y
    return np.exp(-gamma * np.dot(diff, diff))


@njit
def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, coef0: float = 1.0) -> float:
    """多項式カーネル"""
    return (np.dot(x, y) + coef0) ** degree


@njit
def sigmoid_kernel(x: np.ndarray, y: np.ndarray, alpha: float = 0.01, coef0: float = 0.0) -> float:
    """シグモイドカーネル"""
    return np.tanh(alpha * np.dot(x, y) + coef0)


@njit
def laplacian_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """ラプラシアンカーネル"""
    diff = np.abs(x - y)
    return np.exp(-gamma * np.sum(diff))


@njit(parallel=True)
def compute_kernel_gram_matrix(data: np.ndarray,
                               kernel_type: int = DEFAULT_KERNEL_TYPE,
                               gamma: float = DEFAULT_GAMMA,
                               degree: int = DEFAULT_DEGREE,
                               coef0: float = DEFAULT_COEF0,
                               alpha: float = DEFAULT_ALPHA,
                               period: float = 10.0,      # 周期カーネル用
                               length_scale: float = 1.0  # 周期カーネル用
                               ) -> np.ndarray:
    """カーネルGram行列の計算（拡張版）"""
    n = data.shape[0]
    K = np.zeros((n, n))

    for i in prange(n):
        for j in range(i, n):
            if kernel_type == 0:  # RBF
                K[i, j] = rbf_kernel(data[i], data[j], gamma)
            elif kernel_type == 1:  # Polynomial
                K[i, j] = polynomial_kernel(data[i], data[j], degree, coef0)
            elif kernel_type == 2:  # Sigmoid
                K[i, j] = sigmoid_kernel(data[i], data[j], alpha, coef0)
            elif kernel_type == 3:  # Laplacian
                K[i, j] = laplacian_kernel(data[i], data[j], gamma)
            elif kernel_type == 4:  # Periodic
                K[i, j] = periodic_kernel(data[i], data[j], period, length_scale)
            else:  # デフォルト: Laplacian
                K[i, j] = laplacian_kernel(data[i], data[j], gamma)

            K[j, i] = K[i, j]  # 対称性

    return K
