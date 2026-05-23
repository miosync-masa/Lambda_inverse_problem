"""Streaming periodic-deviation scorer。

周期パターンを持つ signal (例: ambient_temperature の 24h 空調サイクル、
realTraffic の朝夕ラッシュ、realTweets の曜日効果) では、「**正常周期からの逸脱**」
そのものが anomaly 信号になる。値そのもの・速度・mean・subspace 距離などは
正常周期内で広く分布するため、これらの軸では catch できない。

本 scorer は calibration 区間で dominant period P を推定し、streaming で
``|events[t] - events[t - P]|`` を anomaly score とする。
"同曜日同時刻との比較"。正常周期維持なら residual ≈ 0、周期崩壊で大。

Period 推定:
  - FFT power spectrum で dominant frequency を取り、period = 1/freq
  - min_period / max_period でレンジを制限
  - 推定失敗時 (周期性弱) は threshold を inf にして scorer 自体を OR vote から外す

Lambda³ 既存 batch にあった `periodic_kernel` + `_estimate_periods` の
streaming-friendly 版とも言える設計。
"""

from __future__ import annotations

import numpy as np

from .base import StreamingScorer


def _to_1d(events: np.ndarray) -> np.ndarray:
    return events.mean(axis=1) if events.ndim > 1 else events.ravel()


def _estimate_period_fft(sig: np.ndarray,
                         min_period: int = 12,
                         max_period: int | None = None) -> int | None:
    """FFT power spectrum で dominant period を推定。"""
    n = len(sig)
    if max_period is None:
        max_period = n // 3
    if min_period >= max_period or n < 2 * min_period:
        return None

    # DC 除去 + zero-mean
    s = sig - float(np.mean(sig))
    if float(np.std(s)) < 1e-10:
        return None

    yf = np.fft.rfft(s)
    xf = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(yf) ** 2

    # 周波数範囲: 1/max_period 〜 1/min_period
    freq_min = 1.0 / max_period
    freq_max = 1.0 / min_period
    mask = (xf > freq_min) & (xf < freq_max)
    if not mask.any():
        return None

    valid_power = power.copy()
    valid_power[~mask] = -np.inf
    peak_idx = int(np.argmax(valid_power))
    if peak_idx == 0 or not np.isfinite(valid_power[peak_idx]):
        return None
    freq = float(xf[peak_idx])
    if freq <= 0:
        return None
    return max(min_period, int(round(1.0 / freq)))


class StreamingPeriodicScorer(StreamingScorer):
    """周期残差 |events[t] - events[t-P]| ベースの anomaly score。"""

    def __init__(self, period_hint: int | None = None,
                 min_period: int = 12, max_period: int | None = None,
                 percentile: float = 99.0):
        self._period_hint = period_hint
        self._min_period = int(min_period)
        self._max_period = max_period
        self._percentile = float(percentile)
        self._period: int | None = None
        self._threshold: float = float('inf')
        self._cal_done = False

    def calibrate(self, events_cal: np.ndarray) -> None:
        sig = _to_1d(events_cal)
        n_cal = len(sig)
        if self._period_hint is not None:
            self._period = int(self._period_hint)
        else:
            self._period = _estimate_period_fft(
                sig,
                min_period=self._min_period,
                max_period=self._max_period,
            )

        if (self._period is None
                or self._period < self._min_period
                or self._period >= n_cal):
            self._threshold = float('inf')   # 周期推定失敗 → scorer OFF
            self._cal_done = True
            return

        residuals = np.array([
            abs(float(sig[t] - sig[t - self._period]))
            for t in range(self._period, n_cal)
        ], dtype=np.float64)
        positive = residuals[residuals > 1e-12]
        if len(positive) > 5:
            self._threshold = float(np.percentile(positive, self._percentile))
        else:
            self._threshold = float('inf')
        self._cal_done = True

    def score(self, events: np.ndarray, t: int) -> float:
        if not self._cal_done or self._period is None:
            return 0.0
        sig = _to_1d(events)
        if t < self._period:
            return 0.0
        return float(abs(float(sig[t]) - float(sig[t - self._period])))

    @property
    def threshold(self) -> float:
        if not self._cal_done:
            raise RuntimeError("StreamingPeriodicScorer: calibrate() を先に呼ぶこと")
        return self._threshold

    @property
    def period(self) -> int | None:
        return self._period
