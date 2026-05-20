"""
StructuralScorer: Lambda³構造（パス間相関、トポロジカルチャージ、エネルギー集中度）
の歪みを直接評価する。
"""

import numpy as np

from ..config import Lambda3Result
from .base import AnomalyScorer


def compute_structural_anomaly_scores(events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
    """Lambda³構造の歪みを直接評価"""
    n_events = events.shape[0]
    scores = np.zeros(n_events)

    paths_matrix = np.stack(list(lambda3_result.paths.values()))

    # 1. パス間の相関破壊
    for i in range(n_events):
        if i > 0:
            # 各パスの局所的変化
            local_changes = np.abs(paths_matrix[:, i] - paths_matrix[:, i-1])
            # 変化の不均一性（一部のパスだけ大きく変化）
            scores[i] += np.std(local_changes) * np.max(local_changes)

    # 2. トポロジカルチャージの急変
    charges = np.array(list(lambda3_result.topological_charges.values()))
    for i in range(1, n_events):
        # 各イベントでの実効的チャージ
        eff_charge_curr = np.sum(np.abs(paths_matrix[:, i]) * np.abs(charges))
        eff_charge_prev = np.sum(np.abs(paths_matrix[:, i-1]) * np.abs(charges))
        scores[i] += np.abs(eff_charge_curr - eff_charge_prev)

    # 3. エネルギー集中度
    for i in range(n_events):
        path_energies = paths_matrix[:, i] ** 2
        # エネルギーが特定のパスに集中している場合
        concentration = np.max(path_energies) / (np.sum(path_energies) + 1e-10)
        scores[i] += concentration ** 2

    return scores


class StructuralScorer(AnomalyScorer):
    """Lambda³構造の歪みを直接評価するスコアラ。"""

    def score(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        return compute_structural_anomaly_scores(events, lambda3_result)
