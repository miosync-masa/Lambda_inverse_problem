"""
トポロジカルチャージ Q_Λ とセグメント安定性の高速計算。
"""

from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_topological_charge_jit(path: np.ndarray, n_segments: int = 10) -> Tuple[float, float]:
    """トポロジカルチャージの高速計算"""
    n = len(path)
    closed_path = np.empty(n + 1)
    closed_path[:-1] = path
    closed_path[-1] = path[0]

    # 位相計算
    theta = np.empty(n)
    for i in prange(n):
        theta[i] = np.arctan2(closed_path[i+1], closed_path[i])

    # チャージ計算
    Q_Lambda = 0.0
    for i in range(n-1):
        diff = theta[i+1] - theta[i]
        # 位相のジャンプを処理
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        Q_Lambda += diff
    Q_Lambda /= (2 * np.pi)

    # セグメント安定性
    Q_segments = np.zeros(n_segments)
    for seg in range(n_segments):
        start = seg * n // n_segments
        end = (seg + 1) * n // n_segments
        if end > start + 1:
            seg_sum = 0.0
            for i in range(start, end-1):
                diff = theta[i+1] - theta[i]
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                seg_sum += diff
            Q_segments[seg] = seg_sum

    stability = np.std(Q_segments)
    return Q_Lambda, stability
