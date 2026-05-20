"""
JumpScorer: ジャンプ構造から直接イベント単位の異常スコアを計算。
"""

import numpy as np

from ..analysis.multiscale_jumps import detect_multiscale_jumps
from ..config import Lambda3Result
from .base import AnomalyScorer


def compute_jump_anomaly_scores(jump_structures: dict, events: np.ndarray) -> np.ndarray:
    """ジャンプ構造から直接異常スコアを計算"""
    n_events = events.shape[0]
    scores = np.zeros(n_events)

    # 統合ジャンプスコア
    integrated = jump_structures['integrated']

    # ジャンプの重要度に基づくスコア
    jump_mask = integrated['unified_jumps'].astype(float)
    importance = integrated['jump_importance']

    # 配列サイズの確認と調整
    if len(jump_mask) != n_events:
        # ジャンプ構造とイベント数が一致しない場合は、小さい方に合わせる
        min_length = min(len(jump_mask), n_events)
        jump_mask = jump_mask[:min_length]
        importance = importance[:min_length]
        scores = scores[:min_length]

    # 重要度が高いジャンプのみを考慮
    importance_threshold = np.percentile(importance[importance > 0], 75) if np.any(importance > 0) else 0.5
    significant_jumps = jump_mask * (importance >= importance_threshold)

    scores += significant_jumps * importance

    # 各特徴のジャンプ寄与
    feature_scores = []
    for f, data in jump_structures['features'].items():
        if data['jump_intensity'] > 0:
            feature_score = np.zeros(n_events)

            # 強いジャンプのみを考慮
            strong_jumps = (data['pos_jumps'] + data['neg_jumps']) * (
                np.abs(data['diff']) > np.percentile(np.abs(data['diff']), 98)
            )

            feature_score = strong_jumps * data['jump_intensity']

            # 非対称性が高い場合はペナルティ
            if np.abs(data['asymmetry']) > 0.8:
                feature_score *= (1 + np.abs(data['asymmetry']))

            feature_scores.append(feature_score)

    if feature_scores:
        # 特徴間の最大値を取る
        # 配列サイズを統一
        min_length = min(n_events, min(len(fs) for fs in feature_scores))
        feature_scores_aligned = [fs[:min_length] for fs in feature_scores]
        feature_contribution = np.max(feature_scores_aligned, axis=0)

        # scoresの長さも調整
        if len(scores) > min_length:
            scores = scores[:min_length]
        elif len(scores) < min_length:
            new_scores = np.zeros(min_length)
            new_scores[:len(scores)] = scores
            scores = new_scores

        scores += feature_contribution * 0.5

    # 最終的な長さをn_eventsに合わせる
    if len(scores) != n_events:
        final_scores = np.zeros(n_events)
        final_scores[:min(len(scores), n_events)] = scores[:min(len(scores), n_events)]
        return final_scores

    return scores


class JumpScorer(AnomalyScorer):
    """Score events purely by integrated multiscale jump structure."""

    def score(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        jump_structures = lambda3_result.jump_structures
        if jump_structures is None:
            jump_structures = detect_multiscale_jumps(events)
        return compute_jump_anomaly_scores(jump_structures, events)
