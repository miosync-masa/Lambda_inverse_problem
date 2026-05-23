"""Streaming reconstruction (hybrid simplification) scorer。

batch HybridScorer は Lambda³ の paths_matrix を使った再構成誤差 + jump 誤差 +
topo/pulse 加重 構造異常度の複合スコアだが、streaming prototype として最も
核心の「正常状態の低ランク構造からの逸脱」だけ取り出す。

具体的には:
  Calibration: events_cal の SVD で top-k 主成分 V_cal (k = n_components)
               これは Λ_cal paths の simplification — events_cal の Gram の
               low-rank approximation の factor を直接取る形。
  Streaming, frame t:
               events[t] を V_cal subspace に projection、residual norm を score。
               「正常データで張られた部分空間に居なければ異常」という直接表現。

「正常状態の clean reference 部分空間からの距離」という streaming 設計の
中核理念をそのまま実装している。
"""

from __future__ import annotations

import numpy as np

from .base import StreamingScorer


def _ensure_2d(events: np.ndarray) -> np.ndarray:
    if events.ndim == 1:
        return events.reshape(-1, 1)
    return events


class StreamingReconstructionScorer(StreamingScorer):
    """Time-delay embedding + SVD top-k 部分空間からの residual norm。

    Lambda³ の Λ paths は時間方向の構造を捉える設計。これを streaming で
    模倣するため、**delay-embedded window** を作って SVD する:

        z_t = [ events[t], events[t-1], ..., events[t-W+1] ]  ∈ R^{W*d}

    calibration 区間の delay-embedded vectors を行列 Z_cal にして SVD、
    top-k 部分空間 V_cal ⊂ R^{W*d} を baseline とする。streaming で
    各 frame t について z_t を構築し V_cal subspace への residual を score。

    Lambda³ Λ paths の "時間方向に展開した正常構造" の streaming 等価物。
    delay W は短すぎると spatial SVD と同じ、長すぎると n_cal よりサンプル
    不足。default W=20 は短期 trajectory pattern を捉える程度。
    """

    def __init__(self, n_components: int = 5,
                 delay_window: int = 20,
                 percentile: float = 99.0):
        self._n_components = int(n_components)
        self._W = int(delay_window)
        self._percentile = float(percentile)
        self._V: np.ndarray | None = None     # (k, W*d) 部分空間基底
        self._mean: np.ndarray | None = None  # (W*d,) delay-embedded vector の mean
        self._threshold: float = float('inf')
        self._cal_done = False

    def _delay_vec(self, X: np.ndarray, t: int) -> np.ndarray | None:
        """frame t の delay-embedded vector を返す。t < W-1 では None。"""
        if t < self._W - 1:
            return None
        seg = X[t - self._W + 1: t + 1]   # (W, d)
        return seg.reshape(-1)            # (W*d,)

    def _residual(self, z: np.ndarray) -> float:
        zc = z - self._mean
        proj = zc @ self._V.T              # (k,)
        recon = proj @ self._V             # (W*d,)
        return float(np.linalg.norm(zc - recon))

    def calibrate(self, events_cal: np.ndarray) -> None:
        X = _ensure_2d(np.asarray(events_cal, dtype=np.float64))
        n_cal, d = X.shape

        if n_cal < self._W + 2:
            self._V = None
            self._threshold = float('inf')
            self._cal_done = True
            return

        # Delay-embedded matrix: rows = delay vectors at each valid t
        rows = []
        for t in range(self._W - 1, n_cal):
            z = self._delay_vec(X, t)
            if z is not None:
                rows.append(z)
        Z = np.asarray(rows, dtype=np.float64)   # (n_cal - W + 1, W*d)
        if Z.shape[0] < 2:
            self._V = None
            self._threshold = float('inf')
            self._cal_done = True
            return

        self._mean = Z.mean(axis=0)
        Zc = Z - self._mean

        # SVD top-k (k < W*d で残差が意味を持つ)
        try:
            _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        except np.linalg.LinAlgError:
            self._V = None
            self._threshold = float('inf')
            self._cal_done = True
            return

        k = max(1, min(self._n_components, Vt.shape[0] - 1))
        self._V = Vt[:k]   # (k, W*d)

        # calibration residual の percentile で threshold 決定
        residuals = np.array([self._residual(z) for z in Z], dtype=np.float64)
        positive = residuals[residuals > 1e-12]
        if len(positive) > 5:
            self._threshold = float(np.percentile(positive, self._percentile))
        else:
            self._threshold = float('inf')
        self._cal_done = True

    def score(self, events: np.ndarray, t: int) -> float:
        if not self._cal_done or self._V is None:
            return 0.0
        X = _ensure_2d(events)
        z = self._delay_vec(X, t)
        if z is None:
            return 0.0
        return self._residual(z)

    @property
    def threshold(self) -> float:
        if not self._cal_done:
            raise RuntimeError("StreamingReconstructionScorer: calibrate() を先に呼ぶこと")
        return self._threshold
