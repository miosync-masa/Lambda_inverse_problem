"""Streaming structural scorer (path increment in delay-embedded subspace)。

batch StructuralScorer は paths_matrix の局所変化分散、effective charge 変化、
エネルギー集中度を組み合わせる。streaming で同じ paths の online 更新は重い
ため、代わりに **delay-embedded trajectory の "smoothness" baseline からの逸脱**
を測る。

具体的に: calibration の delay-embedded vectors の連続フレーム間 distance の
分布を baseline とし、streaming で同じ計算をして比較する。Lambda³ Λ paths の
構造的滑らかさ (paths_matrix[k] の隣接フレーム値が連続的に変化する性質) の
streaming 等価物。

Reconstruction scorer が「subspace 距離」を、本 scorer は「subspace 内軌道の
速度」を測る ― 直交した structural axis。
"""

from __future__ import annotations

import numpy as np

from .base import StreamingScorer


def _ensure_2d(events: np.ndarray) -> np.ndarray:
    if events.ndim == 1:
        return events.reshape(-1, 1)
    return events


class StreamingStructuralScorer(StreamingScorer):
    """Delay-embedded trajectory の連続フレーム間 distance を baseline と比較。"""

    def __init__(self, delay_window: int = 20, percentile: float = 99.0):
        self._W = int(delay_window)
        self._percentile = float(percentile)
        self._cal_mean: float = 0.0       # baseline distance mean
        self._cal_std: float = 1.0        # baseline distance std
        self._threshold: float = float('inf')
        self._cal_done = False

    def _delay_vec(self, X: np.ndarray, t: int) -> np.ndarray | None:
        if t < self._W - 1:
            return None
        seg = X[t - self._W + 1: t + 1]
        return seg.reshape(-1)

    def _step_distance(self, X: np.ndarray, t: int) -> float:
        """frame t と t-1 の delay-embedded vector の Euclidean distance。"""
        if t < self._W:
            return 0.0
        z_t = self._delay_vec(X, t)
        z_prev = self._delay_vec(X, t - 1)
        if z_t is None or z_prev is None:
            return 0.0
        return float(np.linalg.norm(z_t - z_prev))

    def calibrate(self, events_cal: np.ndarray) -> None:
        X = _ensure_2d(np.asarray(events_cal, dtype=np.float64))
        n_cal, d = X.shape
        if n_cal < self._W + 2:
            self._threshold = float('inf')
            self._cal_done = True
            return

        # calibration 内の連続フレーム間 distance
        dists = []
        for t in range(self._W, n_cal):
            dists.append(self._step_distance(X, t))
        dists = np.array(dists, dtype=np.float64)
        positive = dists[dists > 1e-12]
        if len(positive) < 5:
            self._threshold = float('inf')
            self._cal_done = True
            return

        self._cal_mean = float(np.mean(positive))
        self._cal_std = float(np.std(positive)) + 1e-10
        # z-score 化した distance の percentile threshold
        z_scores = (dists - self._cal_mean) / self._cal_std
        self._threshold = float(np.percentile(np.abs(z_scores), self._percentile))
        self._cal_done = True

    def score(self, events: np.ndarray, t: int) -> float:
        if not self._cal_done:
            return 0.0
        X = _ensure_2d(events)
        d = self._step_distance(X, t)
        # z-score 化 (baseline からどれだけ離れた "速度" か)
        return float(abs((d - self._cal_mean) / self._cal_std))

    @property
    def threshold(self) -> float:
        if not self._cal_done:
            raise RuntimeError("StreamingStructuralScorer: calibrate() を先に呼ぶこと")
        return self._threshold
