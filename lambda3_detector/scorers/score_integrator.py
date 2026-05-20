"""
ScoreIntegrator: 各スコアラの出力を重み付け統合し、必要なら差分進化で
重みを最適化する。

統合戦略は ``detect_anomalies`` のデフォルト挙動と一致させてあり、
detectorからもablation experimentからも同じロジックで再利用できる。
"""

from typing import Dict, Iterable, Optional

import numpy as np

from .polarity import polarity_symmetric_score


def integrate_scores(component_scores: Dict[str, np.ndarray],
                     weights: Dict[str, float]) -> np.ndarray:
    """複数のスコアコンポーネントを統合"""
    # 重みの正規化
    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()}

    # 各コンポーネントを標準化してから統合
    integrated_scores = np.zeros_like(list(component_scores.values())[0])

    for component, scores in component_scores.items():
        if component in norm_weights:
            # 標準化
            if np.std(scores) > 1e-10:
                scores_norm = (scores - np.mean(scores)) / np.std(scores)
            else:
                scores_norm = scores

            integrated_scores += norm_weights[component] * scores_norm

    return integrated_scores


def optimize_component_weights_aggressive(component_scores: Dict[str, np.ndarray],
                                           labels: np.ndarray,
                                           event_indices: np.ndarray = None,
                                           events: np.ndarray = None,
                                           force_all_components: bool = True,
                                           remove_collinearity: bool = False,
                                           collinearity_threshold: float = 0.95,
                                           verbose: bool = False) -> Dict[str, float]:
    """
    全コンポーネントを強制的に使用する積極的最適化
    """
    from scipy.optimize import differential_evolution, minimize
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    feature_names = list(component_scores.keys())
    n_features = len(feature_names)
    n_samples = len(labels)

    # 1. 特徴量行列の構築
    feature_matrix = np.column_stack([component_scores[name] for name in feature_names])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

    # 2. 多重共線性の除去（強制モードではスキップ）
    if not force_all_components and remove_collinearity and n_features > 2:
        # 既存のコードと同じ処理
        corr_matrix = np.corrcoef(feature_matrix.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # ... (既存の共線性除去コード)

    # 3. スケーリング（ロバスト版）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    # 4. サンプル重みの計算（よりアグレッシブに）
    sample_weights = np.ones(n_samples)

    # クラスバランスを強く考慮
    try:
        from sklearn.utils.class_weight import compute_sample_weight
        class_weights = compute_sample_weight('balanced', labels)
        # より極端なクラス重みを適用
        class_weights = np.power(class_weights, 1.5)  # 不均衡をより強調
        sample_weights *= class_weights
    except Exception:
        pass

    sample_weights = np.clip(sample_weights, 0.1, 10.0)

    # 5. 差分進化を最初から使用（L1正則化をスキップ）
    if verbose:
        print(f"Aggressive optimization with {n_features} components")

    def objective(weights):
        # 重み付きスコア
        scores = X_scaled @ weights

        # より急峻なシグモイド変換（異常をより明確に分離）
        scores_normalized = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
        probs = 1 / (1 + np.exp(-2 * scores_normalized))  # 係数2で急峻化

        try:
            # AUC最大化
            auc = roc_auc_score(labels, probs, sample_weight=sample_weights)

            # ペナルティ項
            penalty = 0

            if force_all_components:
                # 全コンポーネント使用を強制
                min_weight = np.min(weights)
                if min_weight < 0.05:  # 最小5%の重み
                    penalty += 0.2 * (0.05 - min_weight) ** 2

                # 重みの分散を促進（一つの特徴に偏らないように）
                weight_std = np.std(weights)
                if weight_std > 0.4:  # 分散が大きすぎる場合
                    penalty += 0.1 * (weight_std - 0.4)

            # エントロピー正則化（重みの多様性を促進）
            weights_norm = weights / (np.sum(weights) + 1e-8)
            entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
            max_entropy = np.log(n_features)
            entropy_bonus = 0.05 * (entropy / max_entropy)  # 0-0.05のボーナス

            return -(auc + entropy_bonus - penalty)

        except Exception as e:
            if verbose:
                print(f"Objective function error: {e}")
            return 1.0

    # 境界設定：強制モードでは最小値を高く設定
    if force_all_components:
        bounds = [(0.05, 1.0) for _ in range(n_features)]  # 最小5%
    else:
        bounds = [(0.0, 1.0) for _ in range(n_features)]

    # 差分進化の実行（より多くの反復）
    result_de = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=100 if force_all_components else 50,  # 強制モードではより多く
        popsize=20,
        mutation=(0.5, 1.5),  # より広い探索
        recombination=0.7,
        seed=42,
        polish=True,  # 最終的な局所最適化
        disp=verbose
    )

    # 最適重みの取得
    best_weights = result_de.x

    # 6. 追加の局所最適化（Nelder-Mead）
    if force_all_components:
        # 初期値は差分進化の結果
        local_result = minimize(
            objective,
            best_weights,
            method='Nelder-Mead',
            options={'maxiter': 200}
        )

        if local_result.fun < result_de.fun:
            best_weights = local_result.x
            if verbose:
                print("Local optimization improved the solution")

    # 7. 正規化と最小重みの保証
    if force_all_components:
        # 最小重みを保証
        best_weights = np.maximum(best_weights, 0.05)

    # 合計が1になるように正規化
    best_weights = best_weights / np.sum(best_weights)

    # 最終的なAUCを計算
    final_scores = X_scaled @ best_weights
    final_probs = 1 / (1 + np.exp(-2 * (final_scores - np.mean(final_scores)) / (np.std(final_scores) + 1e-8)))
    final_auc = roc_auc_score(labels, final_probs, sample_weight=sample_weights)

    # 辞書形式に変換
    optimal_weights = {feature_names[i]: best_weights[i] for i in range(n_features)}

    if verbose:
        print(f"\nAggressive optimization completed:")
        print(f"  Final AUC: {final_auc:.4f}")
        print(f"  All weights > 0.05: {all(w >= 0.05 for w in best_weights)}")
        print(f"  Weight std: {np.std(best_weights):.4f}")

        sorted_weights = sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)
        print("\nComponent weights:")
        for feat, weight in sorted_weights:
            print(f"  {feat}: {weight:.4f}")

    return optimal_weights


class ScoreIntegrator:
    """軽量な統合インターフェース。

    Example::

        integrator = ScoreIntegrator(
            default_weights={
                'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15,
            },
            symmetric_components=['hybrid', 'kernel'],
        )
        combined = integrator.combine(
            {'jump': js, 'hybrid': hs, 'kernel': ks, 'structural': ss},
            calibration_frames=cal_frames,
        )

    ``symmetric_components`` に指定した scorer 出力だけ ``|z|`` 変換を施し
    （baseline は先頭 ``calibration_frames`` フレーム）、残りは raw のまま統合する。
    polarity 感受性の強い hybrid / kernel のみ symmetric 化して
    他 scorer の peak 形状を保つ用途を想定。
    """

    def __init__(self,
                 default_weights: Dict[str, float] = None,
                 symmetric_components: Optional[Iterable[str]] = None):
        self.default_weights = default_weights or {
            'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15,
        }
        self.symmetric_components = list(symmetric_components or [])

    def combine(self,
                component_scores: Dict[str, np.ndarray],
                weights: Dict[str, float] = None,
                calibration_frames: Optional[int] = None) -> np.ndarray:
        weights = weights or self.default_weights

        if self.symmetric_components:
            if calibration_frames is None:
                raise ValueError(
                    "symmetric_components が指定されている場合は "
                    "calibration_frames を渡してください"
                )
            transformed = {}
            for name, scores in component_scores.items():
                if name in self.symmetric_components:
                    transformed[name] = polarity_symmetric_score(
                        scores, calibration_frames
                    )
                else:
                    transformed[name] = scores
            return integrate_scores(transformed, weights)

        return integrate_scores(component_scores, weights)

    def optimize_weights(self,
                         component_scores: Dict[str, np.ndarray],
                         labels: np.ndarray,
                         **kwargs) -> Dict[str, float]:
        return optimize_component_weights_aggressive(
            component_scores, labels, **kwargs
        )
