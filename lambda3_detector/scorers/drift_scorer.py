"""
DriftScorer: rolling-window CUSUM (default) ベースのドリフトスコアラ。

structure tensor を必要としない（生イベントのみで計算可能）ため、
Lambda3Result を要求するが内容は使わない。AnomalyScorer ABC を満たす
ために形式上の引数として受け取る。
"""

import numpy as np

from ..analysis.drift_detection import compute_drift_scores
from ..config import Lambda3Result
from .base import AnomalyScorer


class DriftScorer(AnomalyScorer):
    """CUSUM ベースのドリフト／change-point スコアラ。

    Args:
        mode: 'rolling' (default, change-point検出向け) or 'static' (legacy)

        rolling 用 hyperparams:
          ref_window: rolling baseline 窓幅 (default 50)
          k: |z|閾値、|z_t| - k を累積 (default 1.0 ≈ 1σ)
          decay: 累積の忘却率 (default 0.95 → halflife ~ 14フレーム)

        static 用:
          k_factor: CUSUM slack の MAD 倍数 (default 0.5)
    """

    def __init__(self, *,
                 mode: str = 'rolling',
                 ref_window: int = 50,
                 k: float = 1.0,
                 decay: float = 0.95,
                 k_factor: float = 0.5):
        self.mode = mode
        self.ref_window = ref_window
        self.k = k
        self.decay = decay
        self.k_factor = k_factor

    def score(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        # lambda3_resultはABC契約上必須だが、CUSUMは生eventsだけで計算可能
        return compute_drift_scores(
            events,
            mode=self.mode,
            ref_window=self.ref_window,
            k=self.k,
            decay=self.decay,
            k_factor=self.k_factor,
        )
