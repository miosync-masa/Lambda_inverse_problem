"""
StructuralDriftScorer: 先頭 reference window 平均との距離を smooth したスコア。

getter-one (BANKAI) `detect_structural_drift` から移植。
GradualTransitionScorer が「変化の速度」(gradient magnitude) を見るのに対し、
こちらは「baseline からの距離」(state distance) を見る。

両者の使い分け:
    gradual : 遷移の **瞬間** だけスパイク。状態間の移動中にピーク。
    drift   : 遷移**後**ずっと高い値を維持。居座り型異常で持続反応。

NAB の遷延的状態異常 (ambient_temperature_system_failure の 15-22 日続く
空調異常、machine_temperature の catastrophic failure 後の状態) を catch する
ことが狙い。

baseline は先頭 reference_window フレームの平均。NAB の probationary period
(先頭 15%) と同期する設計が筋だが、ファイル長によって固定窓と probationary が
食い違うので、本実装では reference_window と n//4 の小さい方を採用 (短い
ファイルで baseline が anomaly に被らないため)。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..config import Lambda3Result
from .base import AnomalyScorer


def compute_structural_drift_scores(
    events: np.ndarray,
    reference_window: int = 1000,
    smooth_sigma: float = 100.0,
) -> np.ndarray:
    """先頭 reference_window のローカル平均を baseline とし、
    各フレームの local mean からの相対距離を Gaussian smoothing。

    Args:
        events: (n, d) or (n,)
        reference_window: baseline 計算窓
        smooth_sigma: 最終 smoothing の sigma

    Returns:
        scores: (n,) float64
    """
    signal = events.mean(axis=1) if events.ndim > 1 else events.ravel()
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 20:
        return np.zeros(n, dtype=np.float64)

    # baseline window: reference_window と n//4 の小さい方、最低 10
    w_ref = max(10, min(reference_window, max(10, n // 4)))
    ref_mean = float(np.mean(signal[:w_ref]))

    # 各フレームの周辺平均 (window 半径 = w_ref / 2)
    # cumsum O(n) で高速計算
    cs = np.concatenate([[0.0], np.cumsum(signal)])
    half = max(1, w_ref // 2)
    local = np.empty(n, dtype=np.float64)
    for t in range(n):
        s = max(0, t - half)
        e = min(n, t + half)
        local[t] = (cs[e] - cs[s]) / (e - s)

    # baseline からの相対距離 (|ref| が 0 近傍だと相対化が不安定なので絶対値+eps)
    drift = np.abs(local - ref_mean) / (abs(ref_mean) + 1e-10)

    # smooth
    if smooth_sigma > 0 and smooth_sigma * 3 < n:
        drift = gaussian_filter1d(drift, sigma=smooth_sigma)
    return drift


class StructuralDriftScorer(AnomalyScorer):
    """先頭 reference window 平均からの distance を smooth した anomaly スコア。

    既存 ``DriftScorer`` (analysis/drift_detection.py 経由の jump-conditional
    drift) とは別物。あちらは jump 直前後の局所トレンド、こちらはグローバル
    baseline からの偏差距離。
    """

    def __init__(self, reference_window: int = 1000, smooth_sigma: float = 100.0):
        self.reference_window = reference_window
        self.smooth_sigma = smooth_sigma

    def score(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        return compute_structural_drift_scores(
            events,
            reference_window=self.reference_window,
            smooth_sigma=self.smooth_sigma,
        )
