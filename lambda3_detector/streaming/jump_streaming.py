"""Streaming-friendly jump scorer。

calibration 区間で frame-to-frame の |Δsignal| 分布を見て、percentile threshold
(99%) を固定。streaming 中は各 frame で diff を計算して threshold と比較する
だけ。lookback = 1 frame のみ、最も軽量。

Lambda3 既存 JumpScorer は multi-scale jump 解析を行うが、こちらは streaming
prototype として最も簡素な「1-step finite difference 異常」検出。今後 multi-scale
化したければ複数の lookback 窓を持たせれば良い。
"""

from __future__ import annotations

import numpy as np

from .base import StreamingScorer


def _to_1d(events: np.ndarray) -> np.ndarray:
    return events.mean(axis=1) if events.ndim > 1 else events.ravel()


class StreamingJumpScorer(StreamingScorer):
    """frame-to-frame の |Δsignal| anomaly。"""

    def __init__(self, percentile: float = 99.0):
        self._percentile = float(percentile)
        self._threshold: float = float('inf')
        self._cal_done = False

    def calibrate(self, events_cal: np.ndarray) -> None:
        sig = _to_1d(events_cal)
        if len(sig) < 2:
            self._threshold = float('inf')
        else:
            diffs = np.abs(np.diff(sig))
            self._threshold = float(np.percentile(diffs, self._percentile))
        self._cal_done = True

    def score(self, events: np.ndarray, t: int) -> float:
        if t < 1:
            return 0.0
        sig = _to_1d(events)
        return float(abs(sig[t] - sig[t - 1]))

    @property
    def threshold(self) -> float:
        if not self._cal_done:
            raise RuntimeError("StreamingJumpScorer: calibrate() を先に呼ぶこと")
        return self._threshold
