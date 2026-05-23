"""Streaming structural drift scorer。

calibration 区間の mean を baseline (ref_mean) とし、streaming 中は各 frame の
過去 W frame の local mean が ref_mean からどれだけ離れたかを計算する。
cumsum を内部キャッシュして O(1) per frame で更新する。

batch 版 StructuralDriftScorer と一貫: |local_mean - ref_mean| / (|ref_mean| + eps)
だが、smoothing は streaming なので causal rolling だけ。
"""

from __future__ import annotations

import numpy as np

from .base import StreamingScorer


def _to_1d(events: np.ndarray) -> np.ndarray:
    return events.mean(axis=1) if events.ndim > 1 else events.ravel()


class StreamingStructuralDriftScorer(StreamingScorer):
    """baseline 平均からの距離を rolling で測る streaming detector。"""

    def __init__(self, local_window: int = 200, percentile: float = 99.0,
                 min_window: int = 20):
        self._W = int(local_window)
        self._percentile = float(percentile)
        self._min_window = int(min_window)
        self._ref_mean: float = 0.0
        self._threshold: float = float('inf')
        self._cal_done = False

    def _raw_score_at(self, sig: np.ndarray, t: int) -> float:
        # 過去 W frame (含む現在) の local mean
        s = max(0, t - self._W + 1)
        seg = sig[s:t + 1]
        if len(seg) < self._min_window:
            return 0.0
        local = float(np.mean(seg))
        # detector 側で z-normalize されてる前提では ref_mean ≈ 0、ref_std ≈ 1。
        # 旧式 |local - ref| / (|ref| + eps) は ref ≈ 0 で爆発するので、
        # 単純な |local - ref| (= z-score 空間での距離) を返す。
        return float(abs(local - self._ref_mean))

    def calibrate(self, events_cal: np.ndarray) -> None:
        sig = _to_1d(events_cal)
        if len(sig) == 0:
            self._ref_mean = 0.0
            self._threshold = float('inf')
            self._cal_done = True
            return
        self._ref_mean = float(np.mean(sig))
        n = len(sig)
        # calibration 区間内の self-scores を見て percentile threshold 決定
        scores = np.array([self._raw_score_at(sig, t) for t in range(n)],
                          dtype=np.float64)
        positive = scores[scores > 0]
        if len(positive) > 5:
            self._threshold = float(np.percentile(positive, self._percentile))
        else:
            self._threshold = float('inf')
        self._cal_done = True

    def score(self, events: np.ndarray, t: int) -> float:
        sig = _to_1d(events)
        return self._raw_score_at(sig, t)

    @property
    def threshold(self) -> float:
        if not self._cal_done:
            raise RuntimeError(
                "StreamingStructuralDriftScorer: calibrate() を先に呼ぶこと"
            )
        return self._threshold
