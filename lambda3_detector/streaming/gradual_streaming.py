"""Streaming gradual transition scorer。

batch 版 GradualTransitionScorer (scipy.ndimage.gaussian_filter1d を全期間に
適用) は future leakage する。streaming では各 frame t で
過去 W frame だけを参照する片側 (causal) Gaussian smoothing を使う。

Algorithm (per scale w):
  1. tail = events[max(0, t-w+1) : t+1]    # 最近 w フレーム
  2. trend_t = causal Gaussian weighted mean of tail (sigma = w/3)
  3. 速度 (gradient) は trend_t と trend_{t-1} の差で近似:
       grad_t = trend_t - trend_{t-1}
  4. sustained_t = 過去 w/2 フレーム分の |grad| を平均

calibration では同じ計算を全 calibration 区間で行い percentile threshold 取る。
複数 scale (default [50, 200, 500] — 短め設定、prototype) の sustained_t の
最大を最終 raw score。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import StreamingScorer


def _to_1d(events: np.ndarray) -> np.ndarray:
    return events.mean(axis=1) if events.ndim > 1 else events.ravel()


def _causal_gaussian_weights(window: int, sigma: float) -> np.ndarray:
    """過去 window frame ぶんの Gaussian 重み (新しい frame ほど重い)。"""
    # x = 0 が「現在」、x = -(window-1) が「最古」。
    # 重み = exp(-x^2 / (2 sigma^2)) を x=-(w-1)..0 で評価、正規化。
    x = np.arange(window, dtype=np.float64)  # 0..w-1 (現在から見て過去への距離)
    w = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return w / w.sum()


class StreamingGradualScorer(StreamingScorer):
    """multi-scale causal trend gradient (batch 版 GradualTransitionScorer の streaming 版)。"""

    def __init__(self, window_sizes: List[int] = (50, 200, 500),
                 percentile: float = 99.0):
        self._window_sizes = list(window_sizes)
        self._percentile = float(percentile)
        self._weights = {w: _causal_gaussian_weights(w, sigma=w / 3.0)
                         for w in self._window_sizes}
        self._threshold: float = float('inf')
        self._cal_done = False

    def _trend_at(self, sig: np.ndarray, t: int, w: int) -> float:
        """frame t における過去 w frame の causal Gaussian-weighted mean。"""
        if t < 0 or t >= len(sig):
            return 0.0
        s = max(0, t - w + 1)
        seg = sig[s:t + 1]
        wts = self._weights[w][-len(seg):]   # 後ろ寄せ (現在に近いほど重い)
        wts = wts / wts.sum()
        return float(np.dot(seg, wts))

    def _raw_score(self, sig: np.ndarray, t: int) -> float:
        if t < 1:
            return 0.0
        n = len(sig)
        best = 0.0
        for w in self._window_sizes:
            if w >= n or t < w // 2:
                continue
            # 速度 (1-step gradient of causal trend)
            tr_now = self._trend_at(sig, t, w)
            tr_prev = self._trend_at(sig, t - 1, w)
            grad = abs(tr_now - tr_prev)
            # 持続性: 過去 w/4 frames の |grad| を平均
            lookback = max(2, w // 4)
            s = max(1, t - lookback + 1)
            grads = []
            for tt in range(s, t + 1):
                if tt < 1:
                    continue
                a = self._trend_at(sig, tt, w)
                b = self._trend_at(sig, tt - 1, w)
                grads.append(abs(a - b))
            sustained = float(np.mean(grads)) if grads else grad
            if sustained > best:
                best = sustained
        return best

    def calibrate(self, events_cal: np.ndarray) -> None:
        sig = _to_1d(events_cal)
        n = len(sig)
        scores = np.zeros(n, dtype=np.float64)
        for t in range(n):
            scores[t] = self._raw_score(sig, t)
        positive = scores[scores > 0]
        if len(positive) > 5:
            self._threshold = float(np.percentile(positive, self._percentile))
        else:
            self._threshold = float('inf')
        self._cal_done = True

    def score(self, events: np.ndarray, t: int) -> float:
        sig = _to_1d(events)
        return self._raw_score(sig, t)

    @property
    def threshold(self) -> float:
        if not self._cal_done:
            raise RuntimeError("StreamingGradualScorer: calibrate() を先に呼ぶこと")
        return self._threshold
