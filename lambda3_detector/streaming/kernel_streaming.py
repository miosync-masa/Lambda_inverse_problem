"""Streaming kernel anomaly scorer (Kernel Mean Embedding Distance)。

batch KernelScorer は K (events 全期間 Gram) と K_recon (paths^T paths * K)
の Frobenius 距離をもとに anomaly を測るが、streaming で paths を更新する
コストが大きい。代わりに **Kernel Mean Embedding Distance** で:

    score(x_t) = || φ(x_t) - μ_cal ||_H
              = sqrt( K(x_t, x_t)
                      - 2 mean_i K(x_t, x_cal_i)
                      + mean_{i,j} K(x_cal_i, x_cal_j) )

を採用する。これは「正常状態の RKHS 重心からの距離」で、Lambda³ KernelScorer
の「kernel-space deviation」軸の streaming-friendly proxy。

Polynomial kernel (degree=3, coef0=1.0) を default。input feature を
z-normalize してから kernel 計算するため、NAB の大きい値 (taxi 数千、
disk_write 数百万) でも overflow しない。

計算量: per frame O(n_cal * d)。n_cal=2000 / d=5 なら ~10k FLOPS、軽量。
"""

from __future__ import annotations

import numpy as np

from .base import StreamingScorer


def _ensure_2d(events: np.ndarray) -> np.ndarray:
    if events.ndim == 1:
        return events.reshape(-1, 1)
    return events


class StreamingKernelScorer(StreamingScorer):
    """RKHS 重心 (kernel mean embedding) からの distance を streaming anomaly score とする。"""

    def __init__(self, kernel: str = 'polynomial', degree: int = 3,
                 coef0: float = 1.0, gamma: float = 1.0,
                 percentile: float = 99.0):
        self._kernel = kernel
        self._degree = int(degree)
        self._coef0 = float(coef0)
        self._gamma = float(gamma)
        self._percentile = float(percentile)

        self._events_cal: np.ndarray | None = None    # (n_cal, d) z-normalized
        self._mean_feat: np.ndarray | None = None     # (d,) feature-wise mean
        self._std_feat: np.ndarray | None = None      # (d,) feature-wise std
        self._mean_term: float = 0.0                  # mean_{i,j} K(x_i, x_j)
        self._threshold: float = float('inf')
        self._cal_done = False

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mean_feat) / self._std_feat

    def _kernel_vec(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """K(x, Y[i]) for all i. x: (d,), Y: (n, d) → (n,)。"""
        if self._kernel == 'polynomial':
            inner = Y @ x   # (n,)
            return (inner + self._coef0) ** self._degree
        elif self._kernel == 'rbf':
            d2 = np.sum((Y - x) ** 2, axis=1)
            return np.exp(-self._gamma * d2)
        else:
            raise ValueError(f"unknown kernel: {self._kernel}")

    def _self_kernel(self, x: np.ndarray) -> float:
        if self._kernel == 'polynomial':
            return float((float(np.dot(x, x)) + self._coef0) ** self._degree)
        elif self._kernel == 'rbf':
            return 1.0   # exp(0) = 1
        else:
            raise ValueError(f"unknown kernel: {self._kernel}")

    def _distance(self, x_norm: np.ndarray) -> float:
        """|| φ(x) - μ_cal ||_H における distance (z-normalized 入力前提)。"""
        kxx = self._self_kernel(x_norm)
        k_xc = self._kernel_vec(x_norm, self._events_cal)
        cross_mean = float(np.mean(k_xc))
        d2 = kxx - 2.0 * cross_mean + self._mean_term
        return float(np.sqrt(max(d2, 0.0)))

    def calibrate(self, events_cal: np.ndarray) -> None:
        X = _ensure_2d(np.asarray(events_cal, dtype=np.float64))
        n_cal, d = X.shape
        if n_cal < 5 or d < 1:
            self._threshold = float('inf')
            self._cal_done = True
            return

        # z-normalize features (cal で固定したパラメータを streaming で使い回す)
        self._mean_feat = X.mean(axis=0)
        self._std_feat = X.std(axis=0) + 1e-10
        Xn = (X - self._mean_feat) / self._std_feat
        self._events_cal = Xn

        # K_cal を計算 (polynomial の場合 O(n_cal^2 * d))
        if self._kernel == 'polynomial':
            inner = Xn @ Xn.T   # (n_cal, n_cal)
            K = (inner + self._coef0) ** self._degree
        elif self._kernel == 'rbf':
            sq = np.sum(Xn * Xn, axis=1)
            D2 = sq[:, None] + sq[None, :] - 2.0 * (Xn @ Xn.T)
            K = np.exp(-self._gamma * np.maximum(D2, 0.0))
        else:
            raise ValueError(f"unknown kernel: {self._kernel}")

        self._mean_term = float(np.mean(K))

        # per-cal-frame distance → threshold percentile
        distances = np.array([self._distance(Xn[i]) for i in range(n_cal)],
                             dtype=np.float64)
        positive = distances[distances > 1e-12]
        if len(positive) > 5:
            self._threshold = float(np.percentile(positive, self._percentile))
        else:
            self._threshold = float('inf')
        self._cal_done = True

    def score(self, events: np.ndarray, t: int) -> float:
        if not self._cal_done or self._events_cal is None:
            return 0.0
        X = _ensure_2d(events)
        if t < 0 or t >= X.shape[0]:
            return 0.0
        x_norm = self._normalize(X[t])
        return self._distance(x_norm)

    @property
    def threshold(self) -> float:
        if not self._cal_done:
            raise RuntimeError("StreamingKernelScorer: calibrate() を先に呼ぶこと")
        return self._threshold
