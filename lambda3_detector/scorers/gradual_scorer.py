"""
GradualTransitionScorer: 多時間スケールの緩やかな遷移検出。

Lambda³ の jump/hybrid/structural scorer は「急峻な構造変化」を catch するのが
得意だが、「**緩やかに蓄積する状態異常**」を見落とす（NAB の
machine_temperature_system_failure の 2nd anomaly が catastrophic failure を
引き起こすパターン、ambient_temperature_system_failure の遷延的空調異常など）。

getter-one (BANKAI) の `extended_detection_gpu.detect_gradual_transitions`
からロジックを移植: window=[500, 1000, 2000] の multi-scale で
Gaussian-smooth → gradient → sustained gradient (再 Gaussian-smooth) を取り、
各 scale で z-norm して平均する。

n_events < smallest window のファイルでは active window が 0 になり、
score は zero array を返す（小さいデータには寄与しない）。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..config import Lambda3Result
from .base import AnomalyScorer


def compute_gradual_transition_scores(
    events: np.ndarray,
    window_sizes: tuple = (500, 1000, 2000),
) -> np.ndarray:
    """events の column-mean signal に対して multi-scale gradual transition 検出。

    Args:
        events: (n, d) or (n,)
        window_sizes: 解析する時間スケールのリスト

    Returns:
        scores: (n,) per-frame gradual anomaly score (z-norm 済、active scale 数で平均)
    """
    signal = events.mean(axis=1) if events.ndim > 1 else events.ravel()
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    scores = np.zeros(n, dtype=np.float64)

    active = 0
    for w in window_sizes:
        if w >= n:
            continue
        # 1. 長期トレンド抽出 (Gaussian smoothing with sigma = w/3)
        trend = gaussian_filter1d(signal, sigma=w / 3.0)
        # 2. 勾配
        grad = np.gradient(trend)
        # 3. 持続的な勾配 (絶対値を w/6 sigma で smoothing)
        sustained = gaussian_filter1d(np.abs(grad), sigma=w / 6.0)
        # 4. z-norm して加算
        s_std = float(np.std(sustained))
        if s_std > 1e-10:
            scores += (sustained - float(np.mean(sustained))) / s_std
            active += 1

    if active > 0:
        scores /= active
    return scores


class GradualTransitionScorer(AnomalyScorer):
    """multi-scale gradual transition detector (no Lambda3Result deps; works on events alone)."""

    def __init__(self, window_sizes: tuple = (500, 1000, 2000)):
        self.window_sizes = tuple(window_sizes)

    def score(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        return compute_gradual_transition_scores(events, self.window_sizes)
