"""
Multi-Channel Lambda³ Detector - Integration Module
マルチチャンネル統合異常検知モジュール
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings
import os
import json

# 独立したsync_analyzerモジュールからインポート
from sync_analyzer import SyncAnalyzer, SyncResult, extract_jump_events, prepare_sync_data

warnings.filterwarnings('ignore')

@dataclass
class IntegrationResult:
    """統合検知結果を格納するデータクラス"""
    integrated_scores: np.ndarray
    channel_contributions: Dict[str, np.ndarray]
    sync_result: SyncResult
    sync_network: nx.Graph
    anomaly_regions: List[Tuple[int, int]]
    defect_locations: Optional[Dict] = None
    confidence: float = 0.0

@dataclass
class ChannelConfig:
    """チャンネル設定（センサー位置など）"""
    name: str
    position: Optional[np.ndarray] = None  # 3D position for localization
    sensitivity: float = 1.0
    noise_level: float = 0.1

class MultiChannelDetector:
    """
    マルチチャンネルLambda³統合検知器
    複数センサーからの情報を統合して高精度な異常検知を実現
    """
    
    def __init__(self, 
                 sync_threshold: float = 0.3,
                 lag_window: int = 10,
                 integration_method: str = "weighted_sync"):
        """
        Args:
            sync_threshold: 同期判定の閾値
            lag_window: 同期解析のラグウィンドウ
            integration_method: 統合方法 ("weighted_sync", "max_pool", "neural_vote")
        """
        self.sync_analyzer = SyncAnalyzer(lag_window=lag_window, sync_threshold=sync_threshold)
        self.integration_method = integration_method
        self.channel_configs = {}
    
    def set_channel_configs(self, configs: Dict[str, ChannelConfig]):
        """チャンネル設定を登録"""
        self.channel_configs = configs
    
    def integrate_channels(self, 
                         lambda3_results: Dict[str, Dict],
                         use_jump_sync: bool = True,
                         visualize: bool = False) -> IntegrationResult:
        """
        複数チャンネルのLambda³結果を統合
        
        Args:
            lambda3_results: 各チャンネルのLambda³解析結果
                            {'ch1': {'scores': np.ndarray, 'result': Lambda3Result}, ...}
            use_jump_sync: ジャンプイベントの同期も考慮するか
            visualize: 可視化するか
            
        Returns:
            IntegrationResult: 統合結果
        """
        print(f"Integrating {len(lambda3_results)} channels using {self.integration_method}")
        
        # 入力データの確認
        print("\nInput data check:")
        for ch, data in lambda3_results.items():
            if 'scores' in data:
                scores = data['scores']
                print(f"  {ch}: shape={scores.shape}, mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
            else:
                print(f"  {ch}: No 'scores' found in data!")
        
        # 1. データ準備
        anomaly_scores_dict, jump_events_dict = prepare_sync_data(lambda3_results)
        
        if not anomaly_scores_dict:
            raise ValueError("No anomaly scores found in lambda3_results!")
        
        # 2. 同期解析
        print("\nPhase 2: Pairwise synchronization analysis")
        
        # 異常スコアの同期解析
        sync_result = self.sync_analyzer.create_sync_matrix(anomaly_scores_dict)
        
        # ジャンプイベントの同期解析（オプション）
        jump_sync_result = None
        if use_jump_sync and jump_events_dict:
            print("Analyzing jump event synchronization...")
            jump_sync_result = self.sync_analyzer.create_sync_matrix(jump_events_dict)
        
        # 3. 同期ネットワーク構築
        print("\nPhase 3: Building synchronization network")
        sync_network = self._build_enhanced_network(
            anomaly_scores_dict, sync_result, jump_sync_result
        )
        
        # 4. 統合スコア計算
        print("\nPhase 3: Computing integrated anomaly scores")
        integrated_scores, channel_contributions = self._compute_integrated_scores(
            anomaly_scores_dict, sync_result, sync_network
        )
        
        # 5. 異常領域の検出
        anomaly_regions = self._detect_anomaly_regions(integrated_scores)
        
        # 6. 欠陥位置推定（センサー位置情報がある場合）
        defect_locations = None
        if self.channel_configs and any(c.position is not None for c in self.channel_configs.values()):
            print("\nPhase 3: Estimating defect locations")
            defect_locations = self._estimate_defect_locations(
                anomaly_regions, sync_result, sync_network
            )
        
        # 7. 信頼度計算
        confidence = self._calculate_detection_confidence(
            sync_result, integrated_scores, channel_contributions
        )
        
        # 8. 結果の可視化（オプション）
        if visualize:
            self._visualize_integration_results(
                integrated_scores, channel_contributions, 
                sync_result, sync_network, anomaly_regions
            )
        
        result = IntegrationResult(
            integrated_scores=integrated_scores,
            channel_contributions=channel_contributions,
            sync_result=sync_result,
            sync_network=sync_network,
            anomaly_regions=anomaly_regions,
            defect_locations=defect_locations,
            confidence=confidence
        )
        
        print(f"\nIntegration completed. Confidence: {confidence:.2%}")
        print(f"Detected {len(anomaly_regions)} anomaly regions")
        
        return result
    
    def _build_enhanced_network(self,
                              anomaly_scores_dict: Dict[str, np.ndarray],
                              sync_result: SyncResult,
                              jump_sync_result: Optional[SyncResult] = None) -> nx.Graph:
        """強化された同期ネットワークを構築"""
        # 基本ネットワーク
        G = self.sync_analyzer.build_sync_network(anomaly_scores_dict, directed=False)
        
        # ジャンプ同期情報を追加
        if jump_sync_result:
            for i, ch1 in enumerate(jump_sync_result.series_names):
                for j, ch2 in enumerate(jump_sync_result.series_names):
                    if i < j and G.has_edge(ch1, ch2):
                        jump_sync = jump_sync_result.sync_matrix[i, j]
                        G[ch1][ch2]['jump_sync'] = jump_sync
        
        # ノード属性を追加
        for node in G.nodes():
            if node in self.channel_configs:
                config = self.channel_configs[node]
                G.nodes[node]['position'] = config.position
                G.nodes[node]['sensitivity'] = config.sensitivity
                G.nodes[node]['noise_level'] = config.noise_level
        
        return G
    
    def _compute_integrated_scores(self,
                                 anomaly_scores_dict: Dict[str, np.ndarray],
                                 sync_result: SyncResult,
                                 sync_network: nx.Graph) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """統合異常スコアを計算"""
        
        if self.integration_method == "weighted_sync":
            return self._weighted_sync_integration(anomaly_scores_dict, sync_result, sync_network)
        elif self.integration_method == "max_pool":
            return self._max_pool_integration(anomaly_scores_dict)
        elif self.integration_method == "neural_vote":
            return self._neural_vote_integration(anomaly_scores_dict, sync_result)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
    
    def _weighted_sync_integration(self,
                                 anomaly_scores_dict: Dict[str, np.ndarray],
                                 sync_result: SyncResult,
                                 sync_network: nx.Graph) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """同期重み付き統合"""
        channels = list(anomaly_scores_dict.keys())
        n_samples = len(next(iter(anomaly_scores_dict.values())))
        
        print(f"\nWeighted sync integration:")
        print(f"  Channels: {channels}")
        print(f"  Samples: {n_samples}")
        
        # ネットワーク中心性に基づく重み
        centrality = nx.degree_centrality(sync_network)
        print(f"  Centrality: {centrality}")
        
        # 各チャンネルの寄与度
        channel_contributions = {}
        integrated_scores = np.zeros(n_samples)
        
        total_weight_sum = 0.0
        
        for ch_idx, channel in enumerate(channels):
            # 基本重み：中心性
            base_weight = centrality.get(channel, 1.0 / len(channels))
            
            # 同期強度による調整
            sync_strengths = sync_result.sync_matrix[ch_idx]
            # 自己相関（対角成分）を除外
            non_self_sync = np.concatenate([sync_strengths[:ch_idx], sync_strengths[ch_idx+1:]])
            
            if len(non_self_sync) > 0:
                sync_weight = np.mean(non_self_sync)
            else:
                sync_weight = 0.5  # デフォルト値
            
            # 最終重み
            weight = base_weight * (1 + sync_weight)
            total_weight_sum += weight
            
            # 寄与度計算
            scores = anomaly_scores_dict[channel]
            print(f"  {channel}: base_weight={base_weight:.3f}, sync_weight={sync_weight:.3f}, final_weight={weight:.3f}")
            print(f"    Score stats: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
            
            contribution = scores * weight
            channel_contributions[channel] = contribution
            integrated_scores += contribution
        
        # 正規化
        if total_weight_sum > 0:
            integrated_scores /= total_weight_sum
            print(f"  Total weight sum: {total_weight_sum:.3f}")
        else:
            print(f"  WARNING: Total weight sum is zero!")
            # フォールバック：単純平均
            integrated_scores = np.mean(list(anomaly_scores_dict.values()), axis=0)
        
        print(f"  Integrated score stats: mean={np.mean(integrated_scores):.3f}, std={np.std(integrated_scores):.3f}")
        
        return integrated_scores, channel_contributions
    
    def _max_pool_integration(self,
                            anomaly_scores_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """最大値プーリング統合"""
        # 各時点で最大のスコアを採用
        scores_array = np.array(list(anomaly_scores_dict.values()))
        integrated_scores = np.max(scores_array, axis=0)
        
        # 各チャンネルの寄与度（最大値を取った回数）
        channel_contributions = {}
        max_indices = np.argmax(scores_array, axis=0)
        channels = list(anomaly_scores_dict.keys())
        
        for ch_idx, channel in enumerate(channels):
            contribution = np.zeros_like(integrated_scores)
            contribution[max_indices == ch_idx] = integrated_scores[max_indices == ch_idx]
            channel_contributions[channel] = contribution
        
        return integrated_scores, channel_contributions
    
    def _neural_vote_integration(self,
                               anomaly_scores_dict: Dict[str, np.ndarray],
                               sync_result: SyncResult) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """ニューラル投票による統合（同期クラスタベース）"""
        # クラスタリング
        clusters = self.sync_analyzer.cluster_by_sync(sync_result, n_clusters=min(3, len(anomaly_scores_dict)))
        
        # クラスタごとに代表スコアを計算
        cluster_scores = {}
        for cluster_id in set(clusters.values()):
            cluster_channels = [ch for ch, cid in clusters.items() if cid == cluster_id]
            if cluster_channels:
                cluster_array = np.array([anomaly_scores_dict[ch] for ch in cluster_channels])
                # クラスタ内での中央値を代表値とする
                cluster_scores[cluster_id] = np.median(cluster_array, axis=0)
        
        # クラスタ間での投票
        all_cluster_scores = np.array(list(cluster_scores.values()))
        
        # 異常判定の閾値（各クラスタの平均+2σ）
        thresholds = []
        for scores in all_cluster_scores:
            threshold = np.mean(scores) + 2 * np.std(scores)
            thresholds.append(threshold)
        
        # 投票：閾値を超えたクラスタ数
        votes = np.zeros(len(all_cluster_scores[0]))
        for i, (scores, threshold) in enumerate(zip(all_cluster_scores, thresholds)):
            votes += (scores > threshold).astype(float)
        
        # 統合スコア：投票数に基づく
        integrated_scores = votes / len(cluster_scores)
        
        # チャンネル寄与度
        channel_contributions = {}
        for channel, scores in anomaly_scores_dict.items():
            cluster_id = clusters[channel]
            cluster_weight = 1.0 / sum(1 for cid in clusters.values() if cid == cluster_id)
            channel_contributions[channel] = scores * cluster_weight * integrated_scores
        
        return integrated_scores, channel_contributions
    
    def _detect_anomaly_regions(self, 
                              integrated_scores: np.ndarray,
                              threshold_sigma: float = 2.0,
                              min_duration: int = 5) -> List[Tuple[int, int]]:
        """連続的な異常領域を検出"""
        # デバッグ情報
        print(f"\nDetecting anomaly regions...")
        print(f"  Score stats: mean={np.mean(integrated_scores):.3f}, std={np.std(integrated_scores):.3f}")
        print(f"  Score range: [{np.min(integrated_scores):.3f}, {np.max(integrated_scores):.3f}]")
        
        # 閾値設定
        threshold = np.mean(integrated_scores) + threshold_sigma * np.std(integrated_scores)
        print(f"  Threshold (mean + {threshold_sigma}*std): {threshold:.3f}")
        
        # 異常フラグ
        anomaly_flags = integrated_scores > threshold
        n_anomaly_points = np.sum(anomaly_flags)
        print(f"  Points above threshold: {n_anomaly_points}/{len(integrated_scores)}")
        
        # 連続領域の検出
        regions = []
        in_region = False
        start = 0
        
        for i, flag in enumerate(anomaly_flags):
            if flag and not in_region:
                in_region = True
                start = i
            elif not flag and in_region:
                if i - start >= min_duration:
                    regions.append((start, i))
                in_region = False
        
        # 最後の領域
        if in_region and len(anomaly_flags) - start >= min_duration:
            regions.append((start, len(anomaly_flags)))
        
        print(f"  Detected {len(regions)} regions (min_duration={min_duration})")
        
        return regions
    
    def _estimate_defect_locations(self,
                                 anomaly_regions: List[Tuple[int, int]],
                                 sync_result: SyncResult,
                                 sync_network: nx.Graph) -> Dict:
        """欠陥位置を推定（センサー位置情報を使用）"""
        defect_locations = {}
        
        for region_idx, (start, end) in enumerate(anomaly_regions):
            # この領域での各ペアのラグ情報を収集
            lag_data = []
            
            for edge in sync_network.edges(data=True):
                ch1, ch2 = edge[0], edge[1]
                lag = edge[2].get('lag', 0)
                
                # センサー位置
                pos1 = sync_network.nodes[ch1].get('position')
                pos2 = sync_network.nodes[ch2].get('position')
                
                if pos1 is not None and pos2 is not None:
                    lag_data.append({
                        'channels': (ch1, ch2),
                        'positions': (pos1, pos2),
                        'lag': lag,
                        'sync_strength': edge[2]['weight']
                    })
            
            if lag_data:
                # 三角測量による位置推定
                estimated_position = self._triangulate_position(lag_data)
                
                defect_locations[f'region_{region_idx}'] = {
                    'time_range': (start, end),
                    'estimated_position': estimated_position,
                    'confidence': self._calculate_localization_confidence(lag_data)
                }
        
        return defect_locations
    
    def _triangulate_position(self, lag_data: List[Dict]) -> np.ndarray:
        """ラグ情報から欠陥位置を三角測量"""
        # 簡易実装：重み付き平均
        # TODO: より高度な三角測量アルゴリズムの実装
        
        positions = []
        weights = []
        
        for data in lag_data:
            pos1, pos2 = data['positions']
            lag = data['lag']
            sync_strength = data['sync_strength']
            
            # ラグから推定される相対位置
            if lag != 0:
                direction = (pos2 - pos1) / np.linalg.norm(pos2 - pos1)
                estimated_pos = pos1 + direction * lag * 0.1  # 仮の係数
            else:
                estimated_pos = (pos1 + pos2) / 2
            
            positions.append(estimated_pos)
            weights.append(sync_strength)
        
        # 重み付き平均
        positions = np.array(positions)
        weights = np.array(weights) / np.sum(weights)
        
        return np.sum(positions * weights[:, np.newaxis], axis=0)
    
    def _calculate_localization_confidence(self, lag_data: List[Dict]) -> float:
        """位置推定の信頼度を計算"""
        if len(lag_data) < 3:
            return 0.3  # 最小限の信頼度
        
        # ラグの一貫性
        lags = [abs(d['lag']) for d in lag_data]
        lag_consistency = 1.0 / (1.0 + np.std(lags))
        
        # 同期強度
        sync_strengths = [d['sync_strength'] for d in lag_data]
        avg_sync = np.mean(sync_strengths)
        
        # 総合信頼度
        confidence = 0.5 * lag_consistency + 0.5 * avg_sync
        
        return min(confidence, 0.95)  # 上限を設定
    
    def _calculate_detection_confidence(self,
                                      sync_result: SyncResult,
                                      integrated_scores: np.ndarray,
                                      channel_contributions: Dict[str, np.ndarray]) -> float:
        """検出結果の総合信頼度を計算"""
        # 1. チャンネル間の同期度
        sync_matrix = sync_result.sync_matrix
        avg_sync = np.mean(sync_matrix[sync_matrix < 1.0])  # Exclude diagonal
        
        # 2. 寄与度の一貫性
        contributions = np.array(list(channel_contributions.values()))
        contribution_correlation = np.mean(np.corrcoef(contributions))
        
        # 3. スコアの明確性（SNR的な指標）
        score_snr = np.mean(integrated_scores) / (np.std(integrated_scores) + 1e-6)
        score_clarity = 1.0 / (1.0 + np.exp(-score_snr))
        
        # 総合信頼度
        confidence = 0.4 * avg_sync + 0.3 * contribution_correlation + 0.3 * score_clarity
        
        return confidence
    
    def _visualize_integration_results(self,
                                     integrated_scores: np.ndarray,
                                     channel_contributions: Dict[str, np.ndarray],
                                     sync_result: SyncResult,
                                     sync_network: nx.Graph,
                                     anomaly_regions: List[Tuple[int, int]]):
        """統合結果の可視化"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 統合スコアと各チャンネルの寄与
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(integrated_scores, 'k-', linewidth=2, label='Integrated')
        for channel, contrib in channel_contributions.items():
            ax1.plot(contrib, '--', alpha=0.5, label=channel)
        
        # 異常領域をハイライト
        for start, end in anomaly_regions:
            ax1.axvspan(start, end, alpha=0.3, color='red')
        
        ax1.set_title('Integrated Anomaly Score')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Score')
        ax1.legend()
        
        # 2. 同期マトリックス
        ax2 = plt.subplot(3, 3, 2)
        im = ax2.imshow(sync_result.sync_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(sync_result.series_names)))
        ax2.set_yticks(range(len(sync_result.series_names)))
        ax2.set_xticklabels(sync_result.series_names, rotation=45)
        ax2.set_yticklabels(sync_result.series_names)
        ax2.set_title('Synchronization Matrix')
        plt.colorbar(im, ax=ax2)
        
        # 3. 同期ネットワーク
        ax3 = plt.subplot(3, 3, 3)
        pos = nx.spring_layout(sync_network)
        
        # エッジの重みで太さを調整
        edges = sync_network.edges()
        weights = [sync_network[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_nodes(sync_network, pos, ax=ax3, node_size=500)
        nx.draw_networkx_labels(sync_network, pos, ax=ax3)
        nx.draw_networkx_edges(sync_network, pos, ax=ax3, width=np.array(weights)*5)
        
        ax3.set_title('Synchronization Network')
        ax3.axis('off')
        
        # 4. チャンネル別寄与度の時系列
        n_channels = len(channel_contributions)
        for idx, (channel, contrib) in enumerate(channel_contributions.items()):
            ax = plt.subplot(3, 3, 4 + idx)
            ax.plot(contrib, label=channel)
            ax.fill_between(range(len(contrib)), 0, contrib, alpha=0.3)
            
            # 異常領域
            for start, end in anomaly_regions:
                ax.axvspan(start, end, alpha=0.2, color='red')
            
            ax.set_title(f'{channel} Contribution')
            ax.set_xlabel('Time')
            ax.set_ylabel('Contribution')
        
        plt.tight_layout()
        plt.show()

# ===============================
# Utility Functions
# ===============================
def load_lambda3_results(channel_name: str,
                        save_dir: str = "./lambda3_results") -> Dict[str, Any]:
    """
    保存されたLambda³解析結果を読み込み
    
    Args:
        channel_name: チャンネル名
        save_dir: 保存ディレクトリ
        
    Returns:
        読み込んだデータの辞書
    """
    import json
    
    channel_dir = os.path.join(save_dir, channel_name)
    if not os.path.exists(channel_dir):
        raise FileNotFoundError(f"No saved results found for channel {channel_name}")
    
    loaded_data = {}
    
    # 1. 異常スコア（必須）
    scores_path = os.path.join(channel_dir, "anomaly_scores.npy")
    if os.path.exists(scores_path):
        loaded_data['scores'] = np.load(scores_path)
    else:
        raise FileNotFoundError(f"Anomaly scores not found for {channel_name}")
    
    # 2. ジャンプイベント（オプション）
    jump_path = os.path.join(channel_dir, "jump_events.npy")
    if os.path.exists(jump_path):
        loaded_data['jump_events'] = np.load(jump_path)
    
    # 3. ジャンプ重要度（オプション）
    importance_path = os.path.join(channel_dir, "jump_importance.npy")
    if os.path.exists(importance_path):
        loaded_data['jump_importance'] = np.load(importance_path)
    
    # 4. メタデータ（オプション）
    metadata_path = os.path.join(channel_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            loaded_data['metadata'] = json.load(f)
    
    print(f"Loaded Lambda³ results for {channel_name}")
    return loaded_data

def prepare_for_integration(save_dir: str = "./lambda3_results",
                          channels: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    保存されたLambda³結果を統合解析用に準備
    
    Args:
        save_dir: Lambda³結果の保存ディレクトリ
        channels: 読み込むチャンネルのリスト（Noneの場合は全て）
        
    Returns:
        multi_channel_detector.integrate_channels()に渡せる形式のデータ
    """
    import json
    
    integration_data = {}
    
    # チャンネルリストの決定
    if channels is None:
        # サマリーファイルから取得を試みる
        summary_path = os.path.join(save_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            channels = summary.get('channels', [])
        else:
            # サマリーがない場合はディレクトリから推定
            channels = [d for d in os.listdir(save_dir) 
                       if os.path.isdir(os.path.join(save_dir, d)) and not d.startswith('.')]
    
    # 各チャンネルのデータを読み込み
    for channel in channels:
        try:
            channel_data = load_lambda3_results(channel, save_dir)
            integration_data[channel] = channel_data
        except FileNotFoundError as e:
            print(f"Warning: Skipping {channel} - {e}")
            continue
    
    if not integration_data:
        raise ValueError("No valid channel data found for integration")
    
    print(f"Prepared {len(integration_data)} channels for integration:")
    for channel in integration_data.keys():
        print(f"  - {channel}")
    
    return integration_data

def save_integration_results(result: IntegrationResult, 
                           base_path: str = "./results"):
    """統合結果を保存"""
    import os
    os.makedirs(base_path, exist_ok=True)
    
    # スコアの保存
    np.save(f"{base_path}/integrated_scores.npy", result.integrated_scores)
    
    # チャンネル寄与度の保存
    for channel, contrib in result.channel_contributions.items():
        np.save(f"{base_path}/contribution_{channel}.npy", contrib)
    
    # 同期マトリックスの保存
    np.save(f"{base_path}/sync_matrix.npy", result.sync_result.sync_matrix)
    
    # ネットワークの保存（numpy配列属性を除外）
    # GEXFはnumpy配列を保存できないので、別形式で保存
    try:
        # ノード属性からnumpy配列を一時的に除去
        G_copy = result.sync_network.copy()
        for node in G_copy.nodes():
            node_data = G_copy.nodes[node]
            for key, value in list(node_data.items()):
                if isinstance(value, np.ndarray):
                    # numpy配列は文字列に変換
                    node_data[key + '_shape'] = str(value.shape)
                    del node_data[key]
        
        # エッジ属性も同様に処理
        for edge in G_copy.edges():
            edge_data = G_copy.edges[edge]
            for key, value in list(edge_data.items()):
                if isinstance(value, (np.ndarray, dict)):
                    if isinstance(value, dict):
                        # sync_profileなどの辞書も文字列化
                        edge_data[key + '_keys'] = str(list(value.keys()))
                    del edge_data[key]
        
        nx.write_gexf(G_copy, f"{base_path}/sync_network.gexf")
    except Exception as e:
        print(f"Warning: Could not save network in GEXF format: {e}")
        # 代替としてpickleで保存
        import pickle
        with open(f"{base_path}/sync_network.pkl", 'wb') as f:
            pickle.dump(result.sync_network, f)
    
    # メタデータ
    import json
    metadata = {
        'anomaly_regions': result.anomaly_regions,
        'confidence': float(result.confidence) if not np.isnan(result.confidence) else 0.0,
        'channels': result.sync_result.series_names
    }
    
    if result.defect_locations:
        metadata['defect_locations'] = {
            k: {
                'time_range': v['time_range'],
                'position': v['estimated_position'].tolist() if isinstance(v['estimated_position'], np.ndarray) else v['estimated_position'],
                'confidence': float(v['confidence']) if not np.isnan(v['confidence']) else 0.0
            }
            for k, v in result.defect_locations.items()
        }
    
    with open(f"{base_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to {base_path}/")

def demo_multi_channel_detection():
    """デモ: 3チャンネル統合検知"""
    print("=== Multi-Channel Lambda³ Detection Demo ===")
    
    # ダミーデータの生成（実際はLambda³の結果を使用）
    n_samples = 500
    
    # 正常部分
    normal_base = np.random.randn(n_samples) * 0.5
    
    # 異常を注入（位置をずらして）
    anomaly_positions = [(100, 120), (300, 330)]
    
    # 3チャンネルのデータ
    ch1_scores = normal_base.copy()
    ch2_scores = normal_base.copy() 
    ch3_scores = normal_base.copy()
    
    # 異常を各チャンネルに異なるタイミングで注入
    for start, end in anomaly_positions:
        ch1_scores[start:end] += np.random.randn(end-start) * 3
        ch2_scores[start+2:end+2] += np.random.randn(end-start) * 2.5  # 2サンプル遅延
        ch3_scores[start+5:end+5] += np.random.randn(end-start) * 2    # 5サンプル遅延
    
    # Lambda³結果の模擬
    lambda3_results = {
        'ch1': {'scores': ch1_scores},
        'ch2': {'scores': ch2_scores},
        'ch3': {'scores': ch3_scores}
    }
    
    # チャンネル設定
    configs = {
        'ch1': ChannelConfig('ch1', position=np.array([0, 0, 0])),
        'ch2': ChannelConfig('ch2', position=np.array([1, 0, 0])),
        'ch3': ChannelConfig('ch3', position=np.array([0.5, 1, 0]))
    }
    
    # 統合検知
    detector = MultiChannelDetector()
    detector.set_channel_configs(configs)
    
    result = detector.integrate_channels(lambda3_results, visualize=True)
    
    # 結果の保存
    save_integration_results(result)
    
    return result

if __name__ == "__main__":
    result = demo_multi_channel_detection()
