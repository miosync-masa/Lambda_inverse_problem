"""
HybridScorer: Tikhonov再構成誤差 + ジャンプ誤差 + 構造異常度の融合スコア。
"""

import numpy as np

from ..config import Lambda3Result
from ..core.inverse_problem_jit import compute_lambda3_hybrid_tikhonov_scores
from .base import AnomalyScorer


class HybridScorer(AnomalyScorer):
    """Lambda³ハイブリッドTikhonov融合スコア。

    再構成誤差と構造ジャンプを ``alpha`` で混合し、トポロジカル／拍動
    重みを乗せたパス異常度を加算する。
    """

    def __init__(self,
                 alpha: float = 0.3,
                 jump_scale: float = 1.2,
                 use_union: bool = True,
                 w_topo: float = 0.5,
                 w_pulse: float = 0.3):
        self.alpha = alpha
        self.jump_scale = jump_scale
        self.use_union = use_union
        self.w_topo = w_topo
        self.w_pulse = w_pulse

    def score(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        paths_matrix = np.stack(list(lambda3_result.paths.values()))
        charges = np.array(list(lambda3_result.topological_charges.values()))
        stabilities = np.array(list(lambda3_result.stabilities.values()))

        return compute_lambda3_hybrid_tikhonov_scores(
            paths_matrix, events, charges, stabilities,
            alpha=self.alpha,
            jump_scale=self.jump_scale,
            use_union=self.use_union,
            w_topo=self.w_topo,
            w_pulse=self.w_pulse,
        )
