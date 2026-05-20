"""
Lambda³特徴量の最適化（L1正則化／相関ベース）。
"""

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..config import OptimizationResult


class Lambda3FeatureOptimizer:
    """
    Lambda³特徴量の最適化モジュール
    明確なサンプルから最適な特徴量の組み合わせを学習
    """

    def __init__(self,
                 max_features: int = 20,
                 regularization: float = 0.1):
        self.max_features = max_features
        self.regularization = regularization

    def optimize_features(self,
                         features: Dict[str, np.ndarray],
                         labels: np.ndarray,
                         paths_matrix: np.ndarray,
                         mode: str = "robust") -> OptimizationResult:
        """
        特徴量の最適化（パス特徴量用）

        Args:
            features: 特徴量辞書
            labels: ラベル（0: 正常, 1: 異常）
            paths_matrix: パス行列（射影用）
            mode: "fast" or "robust"
        """
        # 特徴量を配列に変換
        feature_names = list(features.keys())
        feature_arrays = []
        for name in feature_names:
            feat = features[name]
            if feat.ndim == 1:
                feature_arrays.append(feat)
            else:
                # 多次元の場合は最初の要素のみ使用
                feature_arrays.append(feat.flatten()[:1])

        # 全ての特徴量を同じ長さに揃える
        n_samples = len(labels)
        feature_matrix = np.zeros((n_samples, len(feature_names)))
        for i, feat in enumerate(feature_arrays):
            if len(feat) == n_samples:
                feature_matrix[:, i] = feat
            elif len(feat) == 1:
                # スカラー特徴量の場合は全サンプルで同じ値
                feature_matrix[:, i] = feat[0]
            else:
                # サンプル数と合わない場合はスキップ
                feature_matrix[:, i] = 0

        if mode == "fast":
            # 単純な相関ベースの選択
            correlations = {}
            for i, name in enumerate(feature_names):
                corr = np.abs(np.corrcoef(feature_matrix[:, i], labels)[0, 1])
                correlations[name] = corr

            # 上位特徴を選択
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:self.max_features]]

            # 重みは相関値
            weights = {name: correlations[name] for name in selected_features}

            # 簡易AUC計算
            selected_indices = [feature_names.index(name) for name in selected_features]
            selected_matrix = feature_matrix[:, selected_indices]
            scores = np.sum(selected_matrix, axis=1)
            auc = roc_auc_score(labels, scores)

        else:  # robust
            # ロジスティック回帰による特徴選択
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(feature_matrix)

            # L1正則化で特徴選択
            model = LogisticRegression(
                penalty='l1',
                C=1.0/self.regularization,
                solver='liblinear',
                max_iter=1000
            )
            model.fit(X_scaled, labels)

            # 非ゼロ係数の特徴を選択
            non_zero_idx = np.where(np.abs(model.coef_[0]) > 1e-5)[0]

            if len(non_zero_idx) == 0:
                # フォールバック：相関が最も高い特徴を使用
                correlations = [np.abs(np.corrcoef(X_scaled[:, i], labels)[0, 1])
                               for i in range(X_scaled.shape[1])]
                non_zero_idx = [np.argmax(correlations)]

            selected_features = [feature_names[i] for i in non_zero_idx[:self.max_features]]

            # 重みは係数の絶対値
            weights = {}
            for i, name in enumerate(feature_names):
                if name in selected_features:
                    weights[name] = np.abs(model.coef_[0][i])

            # 重みの正規化
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}

            # AUC計算
            auc = roc_auc_score(labels, model.decision_function(X_scaled))

            # 相関も記録
            correlations = {}
            for i, name in enumerate(feature_names):
                corr = np.abs(np.corrcoef(feature_matrix[:, i], labels)[0, 1])
                correlations[name] = corr

        return OptimizationResult(
            selected_features=selected_features,
            weights=weights,
            auc=auc,
            feature_correlations=correlations
        )

    def optimize_features_for_events(self,
                                   event_features: Dict[str, np.ndarray],
                                   labels: np.ndarray,
                                   mode: str = "robust") -> OptimizationResult:
        """
        イベント特徴量の最適化（既に射影済みの特徴量用）
        """
        # 基本的に同じロジックだが、射影は不要
        return self.optimize_features(event_features, labels, None, mode)
