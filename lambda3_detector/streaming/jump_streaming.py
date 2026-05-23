"""Streaming-friendly multi-scale jump scorer。

各 lookback window w で「現在 frame の値が過去 w frame の rolling mean±std から
どれだけ z-score 的に逸脱したか」を計算する。calibration 区間で per-scale の
percentile threshold (default 99%) を固定し、streaming 中は各 scale の
raw / threshold の max を返す。

scorer 内部で multi-scale を normalize するため、外側 Lambda3StreamingDetector
からはこの scorer の threshold = 1.0 として扱う (どれかの scale が自身の閾値を
超えたら出力 > 1.0)。

Lambda3 既存 (batch) JumpScorer の multi-scale 集約思想を踏襲。短い window
(e.g. 5) は急峻 spike を、長い window (e.g. 200) は trend 内の局所突出を catch。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import StreamingScorer


def _to_1d(events: np.ndarray) -> np.ndarray:
    return events.mean(axis=1) if events.ndim > 1 else events.ravel()


class StreamingJumpScorer(StreamingScorer):
    """Multi-scale streaming jump detection (内部 z-score, per-scale percentile threshold)."""

    def __init__(self,
                 window_sizes: List[int] = (5, 20, 50, 200),
                 percentile: float = 99.0):
        self._window_sizes = list(window_sizes)
        self._percentile = float(percentile)
        self._thresholds: dict = {}    # window -> float
        self._cal_done = False

    def _raw_at(self, sig: np.ndarray, t: int, w: int) -> float:
        """frame t での z-score 風 jump 強度 (window w)。"""
        if t < 2:
            return 0.0
        s = max(0, t - w)
        seg = sig[s:t]    # 過去 w frame (現在は含まない)
        if len(seg) < 2:
            return 0.0
        mu = float(np.mean(seg))
        sd = float(np.std(seg)) + 1e-12
        return abs(float(sig[t]) - mu) / sd

    def calibrate(self, events_cal: np.ndarray) -> None:
        sig = _to_1d(events_cal)
        n = len(sig)
        for w in self._window_sizes:
            if w >= n:
                self._thresholds[w] = float('inf')
                continue
            scores = np.array([self._raw_at(sig, t, w) for t in range(n)],
                              dtype=np.float64)
            positive = scores[scores > 0]
            if len(positive) > 5:
                self._thresholds[w] = float(np.percentile(positive, self._percentile))
            else:
                self._thresholds[w] = float('inf')
        self._cal_done = True

    def score(self, events: np.ndarray, t: int) -> float:
        """各 scale の raw / threshold の max を返す。
        内部で per-scale normalize 済なので threshold property は 1.0 固定。"""
        sig = _to_1d(events)
        best_ratio = 0.0
        for w in self._window_sizes:
            thr = self._thresholds.get(w, float('inf'))
            if not np.isfinite(thr) or thr <= 0:
                continue
            raw = self._raw_at(sig, t, w)
            r = raw / (thr + 1e-12)
            if r > best_ratio:
                best_ratio = r
        return best_ratio

    @property
    def threshold(self) -> float:
        if not self._cal_done:
            raise RuntimeError("StreamingJumpScorer: calibrate() を先に呼ぶこと")
        return 1.0   # score() は既に per-scale normalize 済
