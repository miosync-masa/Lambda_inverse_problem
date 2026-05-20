"""
拍動エネルギー計算（ジャンプベース／パスベース）と
ジャンプindex抽出ユーティリティ。
"""

from typing import Tuple

import numpy as np
from numba import njit


@njit
def compute_pulsation_energy_from_jumps(
    pos_jumps: np.ndarray,
    neg_jumps: np.ndarray,
    diff: np.ndarray,
    rho_t: np.ndarray
) -> Tuple[float, float, float]:
    """検出済みジャンプから拍動エネルギーを計算"""
    # ジャンプ強度（検出済みジャンプの差分値の総和）
    pos_intensity = 0.0
    neg_intensity = 0.0

    for i in range(len(diff)):
        if pos_jumps[i] == 1:
            pos_intensity += diff[i]
        if neg_jumps[i] == 1:
            neg_intensity += np.abs(diff[i])

    jump_intensity = pos_intensity + neg_intensity

    # 非対称性（-1 to +1）
    asymmetry = (pos_intensity - neg_intensity) / (pos_intensity + neg_intensity + 1e-10)

    # 拍動パワー（ジャンプ数×強度×平均テンション）
    n_jumps = np.sum(pos_jumps) + np.sum(neg_jumps)

    # ジャンプ位置での平均テンション
    avg_tension = 0.0
    if n_jumps > 0:
        tension_sum = 0.0
        count = 0
        for i in range(len(rho_t)):
            if pos_jumps[i] == 1 or neg_jumps[i] == 1:
                tension_sum += rho_t[i]
                count += 1
        avg_tension = tension_sum / count if count > 0 else 0.0

    pulsation_power = jump_intensity * n_jumps * (1 + avg_tension) / len(diff)

    return jump_intensity, asymmetry, pulsation_power


@njit
def compute_pulsation_energy_from_path(path: np.ndarray) -> Tuple[float, float, float]:
    """パスデータから拍動エネルギーを計算（構造テンソル解析用）"""
    if len(path) < 2:
        return 0.0, 0.0, 0.0

    # 差分とジャンプ検出
    diff = np.diff(path)
    abs_diff = np.abs(diff)
    threshold = np.mean(abs_diff) + 2.0 * np.std(abs_diff)

    # ジャンプ検出
    pos_mask = diff > threshold
    neg_mask = diff < -threshold

    # ジャンプ強度
    pos_intensity = np.sum(diff[pos_mask]) if np.any(pos_mask) else 0.0
    neg_intensity = np.sum(np.abs(diff[neg_mask])) if np.any(neg_mask) else 0.0
    jump_intensity = pos_intensity + neg_intensity

    # 非対称性
    asymmetry = (pos_intensity - neg_intensity) / (pos_intensity + neg_intensity + 1e-10)

    # 拍動パワー
    n_jumps = np.sum(pos_mask) + np.sum(neg_mask)
    pulsation_power = jump_intensity * n_jumps / len(path)

    return jump_intensity, asymmetry, pulsation_power


@njit
def find_jump_indices(path: np.ndarray, jump_scale: float = 2.0):
    """パス内のジャンプindex配列を返す（ΔΛCイベント）"""
    delta = np.abs(np.diff(path))
    th = np.mean(delta) + jump_scale * np.std(delta)
    return np.where(delta > th)[0]
