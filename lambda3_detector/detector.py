"""
Lambda³ Zero-Shot Detector - orchestration layer.

The detector ties together:
    1. ``analysis.*`` pipelines (jump detection, inverse problem, physics)
    2. ``scorers.*`` (jump / hybrid / kernel / structural)
    3. ``scorers.score_integrator`` for adaptive weight tuning

It exposes the same public surface (``analyze``, ``detect_anomalies``,
``explain_anomaly``, ``save_results``, ``visualize_results``) as the
pre-split monolith for drop-in compatibility.
"""

from typing import Dict, Tuple

import numpy as np

from .analysis.multiscale_jumps import (
    detect_multiscale_jumps,
    detect_multiscale_jumps_with_params,
)
from .analysis.physical_quantities import (
    classify_structures,
    compute_energies,
    compute_entropies,
    compute_jump_aware_topology,
    compute_jump_conditional_entropies,
    compute_pulsation_energies,
    compute_topology,
)
from .analysis.structure_tensor import (
    inverse_problem_jump_constrained,
    solve_inverse_problem,
)
from .analysis.structure_tensor_sparse import solve_inverse_problem_sparse
from .config import L3Config, Lambda3Result, OptimizationResult
from . import config as _config
from .core.adaptive_params import compute_adaptive_window_size
from .core.inverse_problem_jit import compute_lambda3_hybrid_tikhonov_scores
from .features.extractor import Lambda3FeatureExtractor
from .features.optimizer import Lambda3FeatureOptimizer
from .scorers.jump_scorer import compute_jump_anomaly_scores
from .scorers.kernel_scorer import (
    compute_kernel_anomaly_scores,
    compute_kernel_anomaly_scores_optimized,
    compute_kernel_anomaly_scores_with_params,
    estimate_periods,
)
from .scorers.score_integrator import (
    integrate_scores,
    optimize_component_weights_aggressive,
)
from .scorers.structural_scorer import compute_structural_anomaly_scores


class Lambda3ZeroShotDetector:
    """
    リファクタリング版Lambda³ゼロショット異常検知システム
    基本：構造テンソル解析 + 特徴量最適化
    オプション：ジャンプ解析、カーネル空間、アンサンブル
    """

    def __init__(self, config: L3Config = None):
        self.config = config or L3Config()
        self.feature_extractor = Lambda3FeatureExtractor()
        self.feature_optimizer = Lambda3FeatureOptimizer()
        self.jump_analyzer = None
        self._analysis_cache = {}
        # 異常パターン生成関数の初期化（テストからattachされる; 未attach時はNone）
        self.anomaly_patterns: Dict[str, callable] = {}

    def analyze(self, events: np.ndarray, n_paths: int = None) -> Lambda3Result:
        """Lambda³解析の実行"""
        if n_paths is None:
            n_paths = self.config.n_paths

        # 動的パラメータ調整
        adaptive_params = compute_adaptive_window_size(events)
        _config.update_global_constants(adaptive_params)

        # キャッシュチェック
        cache_key = f"{events.shape}_{n_paths}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        # ジャンプ構造の検出
        jump_structures = detect_multiscale_jumps(events)
        self.jump_analyzer = jump_structures

        # 構造テンソル推定（逆問題）
        # default: sparse ソルバ (ジャンプ近傍重点・静的削減) — シナリオ平均 +0.05 AUC
        # legacy : full  ソルバ (use_sparse_solver=False で旧挙動)
        if self.config.use_sparse_solver and jump_structures:
            if getattr(self.config, 'use_gpu', False):
                from .gpu import solve_inverse_problem_sparse_gpu
                paths, _stats = solve_inverse_problem_sparse_gpu(
                    events, jump_structures, n_paths,
                    self.config.alpha, self.config.beta,
                    expand_window=self.config.sparse_expand_window,
                    n_static_samples=self.config.sparse_n_static_samples,
                    verbose=False,
                )
            else:
                paths, _stats = solve_inverse_problem_sparse(
                    events, jump_structures, n_paths,
                    self.config.alpha, self.config.beta,
                    expand_window=self.config.sparse_expand_window,
                    n_static_samples=self.config.sparse_n_static_samples,
                    verbose=False,
                )
        elif jump_structures:
            paths = inverse_problem_jump_constrained(
                events, jump_structures, n_paths,
                self.config.alpha, self.config.beta,
            )
        else:
            paths = solve_inverse_problem(
                events, n_paths, self.config.alpha, self.config.beta,
            )

        # 物理量計算
        if jump_structures:
            charges, stabilities = compute_jump_aware_topology(paths, jump_structures)
            energies = compute_pulsation_energies(paths, jump_structures)
            entropies = compute_jump_conditional_entropies(paths, jump_structures)
        else:
            charges, stabilities = compute_topology(paths)
            energies = compute_energies(paths)
            entropies = compute_entropies(paths)

        classifications = classify_structures(paths, charges, stabilities, jump_structures)

        result = Lambda3Result(
            paths=paths,
            topological_charges=charges,
            stabilities=stabilities,
            energies=energies,
            entropies=entropies,
            classifications=classifications,
            jump_structures=jump_structures
        )

        # キャッシュ保存
        self._analysis_cache[cache_key] = result

        return result

    def detect_anomalies(self, result: Lambda3Result, events: np.ndarray,
                    use_adaptive_weights: bool = False) -> np.ndarray:
        """
        圧倒的な性能を目指す革命的異常検知
        """
        n_events = events.shape[0]
        paths_matrix = np.stack(list(result.paths.values()))
        charges = np.array(list(result.topological_charges.values()))
        stabilities = np.array(list(result.stabilities.values()))

        # 1. マルチスケールジャンプ検出（複数の解像度で異常を捕捉）
        multi_jump_scores = []
        for window, percentile in zip(_config.MULTI_SCALE_WINDOWS, _config.MULTI_SCALE_PERCENTILES):
            jump_analyzer = detect_multiscale_jumps_with_params(
                events, window_size=window, percentile=percentile
            )
            jump_scores = compute_jump_anomaly_scores(jump_analyzer, events)
            multi_jump_scores.append(self._ensure_length(jump_scores, n_events))

        # 最大値を取る（どのスケールでも異常なら異常）
        jump_anomaly_scores = np.max(multi_jump_scores, axis=0)

        # 2. 強化版ハイブリッドスコア
        hybrid_scores = compute_lambda3_hybrid_tikhonov_scores(
            paths_matrix, events, charges, stabilities,
            alpha=0.3,      # アグレッシブに
            jump_scale=1.2, # より多くのジャンプを捕捉
            use_union=True,
            w_topo=0.5,     # トポロジーを重視
            w_pulse=0.3     # 拍動も考慮
        )
        hybrid_scores = self._ensure_length(hybrid_scores, n_events)

        # 3. アンサンブルカーネル戦略（複数カーネルの強みを統合）
        kernel_scores_list = []
        kernel_types = [
            (0, {'gamma': 1.0}),      # RBF
            (1, {'degree': 7, 'coef0': 1.0}),  # Polynomial (現在最良)
            (3, {'gamma': 0.5})       # Laplacian
        ]
        for k_type, k_params in kernel_types:
            k_scores = compute_kernel_anomaly_scores_with_params(
                events, result, kernel_type=k_type, **k_params
            )
            kernel_scores_list.append(self._ensure_length(k_scores, n_events))

        # 各カーネルの最良スコアを採用
        kernel_scores = np.max(kernel_scores_list, axis=0)

        # 4. 新規：構造的異常スコア
        structural_scores = compute_structural_anomaly_scores(events, result)
        structural_scores = self._ensure_length(structural_scores, n_events)

        # 5. 革新的な統合戦略
        if use_adaptive_weights:
            # より積極的な適応
            base_scores = (
                0.20 * jump_anomaly_scores +
                0.35 * hybrid_scores +
                0.30 * kernel_scores +
                0.15 * structural_scores
            )

            # 明確なサンプルの選択（より積極的に）
            score_percentiles = np.percentile(base_scores, [10, 90])
            clear_normal = base_scores < score_percentiles[0]
            clear_anomaly = base_scores > score_percentiles[1]

            if np.sum(clear_normal) > 5 and np.sum(clear_anomaly) > 5:
                clear_mask = clear_normal | clear_anomaly
                clear_labels = clear_anomaly[clear_mask].astype(int)
                clear_indices = np.where(clear_mask)[0]

                component_scores = {
                    'jump': jump_anomaly_scores[clear_mask],
                    'hybrid': hybrid_scores[clear_mask],
                    'kernel': kernel_scores[clear_mask],
                    'structural': structural_scores[clear_mask]
                }

                self._last_result = result

                # 強制的に全コンポーネントを使用する最適化
                optimal_weights = optimize_component_weights_aggressive(
                    component_scores,
                    clear_labels,
                    event_indices=clear_indices,
                    events=events,
                    force_all_components=True,
                    verbose=True
                )

                print(f"Adaptive weights learned from {len(clear_labels)} clear samples:")
                for name, weight in optimal_weights.items():
                    print(f"  {name}: {weight:.3f}")

                # 最適化された重みで最終スコアを計算
                final_scores = (
                    optimal_weights.get('jump', 0.20) * jump_anomaly_scores +
                    optimal_weights.get('hybrid', 0.35) * hybrid_scores +
                    optimal_weights.get('kernel', 0.30) * kernel_scores +
                    optimal_weights.get('structural', 0.15) * structural_scores
                )
            else:
                print("Not enough clear samples, using enhanced default weights")
                final_scores = base_scores
        else:
            # デフォルトでも全要素を活用
            final_scores = (
                0.20 * jump_anomaly_scores +
                0.35 * hybrid_scores +
                0.30 * kernel_scores +
                0.15 * structural_scores
            )

        # 6. 非線形変換で異常を強調
        final_scores = np.sign(final_scores) * np.power(np.abs(final_scores), 0.8)

        # 7. Jump情報による適応的再スコアリング
        if result.jump_structures:
            final_scores = self._apply_jump_based_rescoring(
                final_scores,
                result.jump_structures,
                events.shape[1]
            )

        # 8. 革新的な適応的標準化
        return self._adaptive_standardize(final_scores)

    # ===============================
    # スコア後処理ヘルパー
    # ===============================

    def _ensure_length(self, scores: np.ndarray, target_length: int) -> np.ndarray:
        """スコア配列の長さを安全に統一"""
        if len(scores) != target_length:
            if len(scores) < target_length:
                # 短い場合はゼロパディング
                padded_scores = np.zeros(target_length)
                padded_scores[:len(scores)] = scores
                return padded_scores
            else:
                # 長い場合は切り詰め
                return scores[:target_length]
        return scores

    def _adaptive_standardize(self, scores: np.ndarray) -> np.ndarray:
        """外れ値に対してより敏感な標準化"""
        # 1. 基本統計量
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))  # Median Absolute Deviation

        # 2. 外れ値の識別
        if mad > 0:
            z_scores = 0.6745 * (scores - median) / mad  # MADベースのzスコア
        else:
            z_scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-10)

        # 3. 外れ値を強調する変換
        # 正常範囲（|z| < 2）はそのまま、異常値は指数的に強調
        emphasized = np.where(
            np.abs(z_scores) < 2,
            z_scores,
            np.sign(z_scores) * (2 + np.log1p(np.abs(z_scores) - 2) * 3)
        )

        return emphasized

    def _apply_jump_based_rescoring(self,
                                base_scores: np.ndarray,
                                jump_structures: Dict,
                                n_features: int) -> np.ndarray:
        """Jump情報を使った適応的再スコアリング"""
        rescored = base_scores.copy()
        integrated = jump_structures['integrated']

        for i in range(len(base_scores)):
            if integrated['unified_jumps'][i]:
                # このイベントのジャンプ情報を取得
                importance = integrated['jump_importance'][i]

                # 同期している特徴数をカウント
                sync_features = 0
                for f, data in jump_structures['features'].items():
                    if i < len(data['pos_jumps']) and (data['pos_jumps'][i] or data['neg_jumps'][i]):
                        sync_features += 1

                sync_ratio = sync_features / n_features

                # クラスターに属しているかチェック
                in_cluster = any(i in range(c['start'], c['end'])
                              for c in integrated['jump_clusters'])

                # 再スコアリング
                importance_factor = 1 + importance

                # 非線形同期性係数
                if sync_ratio > 0.8:
                    sync_factor = sync_ratio ** 0.5
                else:
                    sync_factor = sync_ratio ** 2

                cluster_factor = 1.2 if in_cluster else 1.0

                rescored[i] = base_scores[i] * importance_factor * sync_factor * cluster_factor

        return rescored

    def _compute_sync_anomaly_scores(self, jump_structures: Dict) -> np.ndarray:
        """Calculate synchronization anomaly scores"""
        # unified_jumpsから実際のイベント数を決定
        if 'integrated' in jump_structures and 'unified_jumps' in jump_structures['integrated']:
            n_events = len(jump_structures['integrated']['unified_jumps'])
        else:
            # フォールバック：最初の特徴のジャンプ配列から推定
            first_feature = list(jump_structures['features'].values())[0]
            n_events = len(first_feature['pos_jumps'])

        scores = np.zeros(n_events)

        # Anomalies in high synchronization clusters
        sync_threshold = 0.7
        sync_matrix = jump_structures['integrated']['sync_matrix']
        n_features = len(sync_matrix)

        # Synchronization anomaly degree for each feature
        for f_idx in range(n_features):
            if f_idx in jump_structures['features']:
                feature_data = jump_structures['features'][f_idx]
                feature_sync = np.mean([sync_matrix[f_idx, j] for j in range(n_features) if j != f_idx])

                if feature_sync > sync_threshold:
                    pos_jumps = feature_data['pos_jumps']
                    neg_jumps = feature_data['neg_jumps']

                    # Ensure jumps arrays have correct length
                    jumps_len = min(len(pos_jumps), len(neg_jumps), n_events)
                    jumps = np.zeros(n_events, dtype=bool)
                    jumps[:jumps_len] = (pos_jumps[:jumps_len] | neg_jumps[:jumps_len]).astype(bool)

                    scores += jumps * feature_sync

        # Normalize by number of features
        if n_features > 0:
            scores = scores / n_features

        return scores

    def _select_clear_samples(self,
                        base_scores: np.ndarray,
                        percentiles: Tuple[float, float],
                        events: np.ndarray = None,
                        result: Lambda3Result = None) -> Tuple[np.ndarray, np.ndarray]:
        """明確な正常/異常サンプルの選択（弱い異常パターン考慮版）"""
        low_threshold = np.percentile(base_scores, percentiles[0])
        high_threshold = np.percentile(base_scores, percentiles[1])

        clear_normal = base_scores < low_threshold
        clear_anomaly = base_scores > high_threshold

        # === 弱い異常パターンの追加検出 ===
        if events is not None and result is not None:
            middle_scores = base_scores[(~clear_normal) & (~clear_anomaly)]
            if len(middle_scores) > 0:
                middle_indices = np.where((~clear_normal) & (~clear_anomaly))[0]

                # 1. 部分的異常の検出（特徴量の一部だけ異常）
                partial_anomalies = []
                for idx in middle_indices:
                    feature_deviations = np.abs(events[idx] - np.median(events, axis=0)) / (np.std(events, axis=0) + 1e-10)
                    # 30%以上の特徴が異常値を示す
                    if np.sum(feature_deviations > 3.0) >= events.shape[1] * 0.3:
                        partial_anomalies.append(idx)

                # 2. 緩やかな劣化パターン（前後との相関が低い）
                degradation_anomalies = []
                for idx in middle_indices:
                    if 1 < idx < len(events) - 1:
                        # 前後のイベントとの相関
                        corr_prev = np.corrcoef(events[idx], events[idx-1])[0,1]
                        corr_next = np.corrcoef(events[idx], events[idx+1])[0,1]
                        # 相関が急激に低下
                        if corr_prev < 0.3 or corr_next < 0.3:
                            degradation_anomalies.append(idx)

                # 3. 微小な周期的異常（FFTで検出）
                periodic_anomalies = []
                if len(middle_indices) > 10:
                    for idx in middle_indices:
                        # 局所的なFFT（前後5イベント）
                        local_start = max(0, idx - 5)
                        local_end = min(len(events), idx + 6)
                        local_fft = np.abs(np.fft.fft(events[local_start:local_end], axis=0))
                        # 特定周波数にピーク
                        if np.max(local_fft[1:len(local_fft)//2]) > np.mean(local_fft) * 5:
                            periodic_anomalies.append(idx)

                # 4. ジャンプ構造からの弱い異常
                weak_jump_anomalies = []
                if result.jump_structures:
                    jump_importance = result.jump_structures['integrated']['jump_importance']
                    for idx in middle_indices:
                        # 弱いジャンプ（0.2-0.4の重要度）
                        if 0.2 < jump_importance[idx] < 0.4:
                            weak_jump_anomalies.append(idx)

                # 弱い異常を統合
                weak_anomaly_set = set(partial_anomalies + degradation_anomalies +
                                      periodic_anomalies + weak_jump_anomalies)

                # スコアに基づいて上位を異常として追加
                weak_anomaly_indices = np.array(sorted(weak_anomaly_set))
                if len(weak_anomaly_indices) > 0:
                    weak_scores = base_scores[weak_anomaly_indices]
                    # 中間領域の上位30%を異常として追加
                    weak_threshold = np.percentile(weak_scores, 70)
                    additional_anomalies = weak_anomaly_indices[weak_scores >= weak_threshold]

                    # 既存の明確な異常に追加
                    clear_anomaly[additional_anomalies] = True

        clear_mask = clear_normal | clear_anomaly
        clear_indices = np.where(clear_mask)[0]
        clear_labels = clear_anomaly[clear_mask].astype(int)

        # デバッグ情報
        if events is not None:
            print(f"  Clear samples enhanced:")
            print(f"    Original clear: {np.sum(clear_normal) + np.sum(clear_anomaly)}")
            print(f"    After weak pattern detection: {len(clear_indices)}")

        return clear_indices, clear_labels

    def _compute_with_optimized_features(self,
                                       features: Dict[str, np.ndarray],
                                       optimization_result: OptimizationResult,
                                       paths_matrix: np.ndarray) -> np.ndarray:
        """最適化された特徴量でスコア計算"""
        # 選択された特徴のみを使用
        selected_features = {
            k: features[k] for k in optimization_result.selected_features
        }

        # イベント空間に射影
        event_features = self.feature_extractor.project_to_event_space(
            selected_features, paths_matrix
        )

        # 重み付き合成
        scores = np.zeros(paths_matrix.shape[1])
        for feat_name, weight in optimization_result.weights.items():
            if feat_name in event_features:
                feat_scores = event_features[feat_name]
                # 標準化
                if np.std(feat_scores) > 1e-10:
                    feat_scores = (feat_scores - np.mean(feat_scores)) / np.std(feat_scores)
                scores += weight * feat_scores

        return scores

    def _integrate_scores(self,
                        component_scores: Dict[str, np.ndarray],
                        weights: Dict[str, float]) -> np.ndarray:
        """複数のスコアコンポーネントを統合"""
        return integrate_scores(component_scores, weights)

    def _standardize_scores(self, scores: np.ndarray) -> np.ndarray:
        """スコアの頑健な標準化"""
        median_score = np.median(scores)
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25

        if iqr > 0:
            standardized = (scores - median_score) / (1.5 * iqr)
        else:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            if std_score > 0:
                standardized = (scores - mean_score) / std_score
            else:
                standardized = scores

        return standardized

    # ===============================
    # 説明・I/O・可視化（薄いラッパー）
    # ===============================

    def explain_anomaly(self, event_idx: int, result: Lambda3Result, events: np.ndarray) -> Dict:
        """異常の物理的説明を生成"""
        explanation = {
            'event_index': event_idx,
            'anomaly_score': 0.0,
            'jump_based': {},
            'topological': {},
            'energetic': {},
            'entropic': {},
            'kernel_space': {},
            'recommendation': ""
        }

        # 異常スコア計算
        anomaly_scores = self.detect_anomalies(result, events)
        explanation['anomaly_score'] = float(anomaly_scores[event_idx])

        # ジャンプベースの説明
        if self.jump_analyzer and result.jump_structures:
            integrated = result.jump_structures['integrated']
            if integrated['unified_jumps'][event_idx]:
                sync_features = []
                for f, data in result.jump_structures['features'].items():
                    if (data['pos_jumps'][event_idx] or data['neg_jumps'][event_idx]):
                        sync_features.append(f)

                explanation['jump_based'] = {
                    'is_jump': True,
                    'importance': float(integrated['jump_importance'][event_idx]),
                    'synchronized_features': sync_features,
                    'n_sync_features': len(sync_features),
                    'in_cluster': any(event_idx in c['indices'] for c in integrated['jump_clusters'])
                }

        # トポロジカル説明
        topo_info = {}
        for p, path in result.paths.items():
            if event_idx > 0:
                delta = np.abs(path[event_idx] - path[event_idx-1])
                path_std = np.std(np.diff(path))
                if delta > path_std * 2:
                    topo_info[f'path_{p}'] = {
                        'charge': float(result.topological_charges[p]),
                        'stability': float(result.stabilities[p]),
                        'classification': result.classifications[p],
                        'delta_lambda': float(delta),
                        'relative_jump': float(delta / path_std)
                    }
        explanation['topological'] = topo_info

        # エネルギー説明
        energy_info = {}
        for p in result.paths.keys():
            energy_info[f'path_{p}'] = {
                'total_energy': float(result.energies[p]),
                'local_energy': float(result.paths[p][event_idx]**2)
            }
        explanation['energetic'] = energy_info

        # エントロピー説明
        entropy_info = {}
        for p, ent_dict in result.entropies.items():
            main_entropy = ent_dict.get('shannon', 0)
            jump_entropy = ent_dict.get('shannon_jump', None)

            entropy_info[f'path_{p}'] = {
                'shannon': float(main_entropy),
                'jump_conditional': float(jump_entropy) if jump_entropy else None
            }
        explanation['entropic'] = entropy_info

        # 推奨アクション
        if explanation['anomaly_score'] > 2.0:
            if explanation['jump_based'].get('is_jump') and explanation['jump_based']['importance'] > 0.7:
                explanation['recommendation'] = "Critical structural transition detected. " \
                                              "Immediate investigation required. " \
                                              "Multiple synchronized features show simultaneous jumps."
            else:
                explanation['recommendation'] = "High anomaly score detected. Investigation recommended."
        elif explanation['anomaly_score'] > 1.0:
            explanation['recommendation'] = "Moderate anomaly detected. Monitor adjacent events for cascading effects."
        else:
            explanation['recommendation'] = "Low anomaly level. Continue normal monitoring."

        return explanation

    def save_results(self, *args, **kwargs):
        """Delegate to :mod:`lambda3_detector.io_utils`."""
        from .io_utils import save_results
        return save_results(self, *args, **kwargs)

    def visualize_results(self, *args, **kwargs):
        """Delegate to :mod:`lambda3_detector.visualization`."""
        from .visualization import visualize_results
        return visualize_results(self, *args, **kwargs)
