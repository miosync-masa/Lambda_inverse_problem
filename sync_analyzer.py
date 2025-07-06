
"""
Synchronization Analysis Module for Multi-Channel Lambda³ Detection
ペアワイズ同期解析モジュール
"""
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from numba import njit, prange
from sklearn.cluster import AgglomerativeClustering

# Constants
LAG_WINDOW_DEFAULT = 10
SYNC_THRESHOLD_DEFAULT = 0.3
WINDOW_SIZE_DEFAULT = 20

@dataclass
class SyncResult:
    """同期解析結果を格納するデータクラス"""
    sync_matrix: np.ndarray
    series_names: List[str]
    sync_network: Optional[nx.DiGraph] = None
    clusters: Optional[Dict[str, int]] = None
    max_sync_pairs: Optional[List[Tuple[str, str, float, int]]] = None

# ===============================
# JIT-optimized core functions
# ===============================
@njit
def sync_rate_at_lag(series_a: np.ndarray, series_b: np.ndarray, lag: int) -> float:
    """JIT-compiled synchronization rate calculation for a specific lag."""
    if lag < 0:
        if -lag < len(series_a):
            return np.mean(series_a[-lag:] * series_b[:lag])
        else:
            return 0.0
    elif lag > 0:
        if lag < len(series_b):
            return np.mean(series_a[:-lag] * series_b[lag:])
        else:
            return 0.0
    else:
        return np.mean(series_a * series_b)

@njit(parallel=True)
def calculate_sync_profile_jit(series_a: np.ndarray, series_b: np.ndarray,
                               lag_window: int) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """JIT-compiled synchronization profile calculation with parallelization."""
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1)
    sync_values = np.empty(n_lags)

    for i in prange(n_lags):
        lag = lags[i]
        sync_values[i] = sync_rate_at_lag(series_a, series_b, lag)

    max_sync = 0.0
    optimal_lag = 0
    for i in range(n_lags):
        if sync_values[i] > max_sync:
            max_sync = sync_values[i]
            optimal_lag = lags[i]

    return lags, sync_values, max_sync, optimal_lag

@njit
def calculate_windowed_sync(series_a: np.ndarray, series_b: np.ndarray, 
                           window_size: int, lag_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate time-varying synchronization with sliding window."""
    T = len(series_a)
    n_windows = T - window_size + 1
    sync_rates = np.zeros(n_windows)
    optimal_lags = np.zeros(n_windows, dtype=np.int32)
    
    for t in range(n_windows):
        window_a = series_a[t:t+window_size]
        window_b = series_b[t:t+window_size]
        
        _, _, max_sync, opt_lag = calculate_sync_profile_jit(window_a, window_b, lag_window)
        sync_rates[t] = max_sync
        optimal_lags[t] = opt_lag
    
    return sync_rates, optimal_lags

# ===============================
# Main Synchronization Analyzer
# ===============================
class SyncAnalyzer:
    """
    同期解析を実行するメインクラス
    Lambda³の異常スコアやジャンプイベントの同期性を解析
    """
    
    def __init__(self, lag_window: int = LAG_WINDOW_DEFAULT, 
                 sync_threshold: float = SYNC_THRESHOLD_DEFAULT):
        """
        Args:
            lag_window: 同期解析で考慮する最大ラグ
            sync_threshold: 同期とみなす閾値
        """
        self.lag_window = lag_window
        self.sync_threshold = sync_threshold
    
    def calculate_sync_profile(self, series_a: np.ndarray, series_b: np.ndarray) -> Tuple[Dict[int, float], float, int]:
        """
        2つの系列間の同期プロファイルを計算
        
        Args:
            series_a: 系列A（異常スコアまたはジャンプイベント）
            series_b: 系列B（異常スコアまたはジャンプイベント）
            
        Returns:
            sync_profile: ラグごとの同期率
            max_sync: 最大同期率
            optimal_lag: 最適ラグ
        """
        # Ensure float64 for JIT function
        series_a = series_a.astype(np.float64)
        series_b = series_b.astype(np.float64)
        
        lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
            series_a, series_b, self.lag_window
        )
        
        # Convert to dictionary
        sync_profile = {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}
        
        return sync_profile, float(max_sync), int(optimal_lag)
    
    def create_sync_matrix(self, series_dict: Dict[str, np.ndarray]) -> SyncResult:
        """
        全系列ペアの同期行列を作成
        
        Args:
            series_dict: チャンネル名をキーとする系列の辞書
            
        Returns:
            SyncResult: 同期解析結果
        """
        series_names = list(series_dict.keys())
        n = len(series_names)
        sync_matrix = np.zeros((n, n))
        max_sync_pairs = []
        
        print(f"Creating sync matrix for {n} channels...")
        
        for i, name_a in enumerate(series_names):
            for j, name_b in enumerate(series_names):
                if i == j:
                    sync_matrix[i, j] = 1.0  # Self-sync is perfect
                    continue
                
                series_a = series_dict[name_a].astype(np.float64)
                series_b = series_dict[name_b].astype(np.float64)
                
                _, _, max_sync, optimal_lag = calculate_sync_profile_jit(
                    series_a, series_b, self.lag_window
                )
                
                sync_matrix[i, j] = max_sync
                
                # Record significant synchronizations
                if max_sync >= self.sync_threshold and i < j:
                    max_sync_pairs.append((name_a, name_b, max_sync, optimal_lag))
        
        # Sort by sync strength
        max_sync_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return SyncResult(
            sync_matrix=sync_matrix,
            series_names=series_names,
            max_sync_pairs=max_sync_pairs
        )
    
    def build_sync_network(self, series_dict: Dict[str, np.ndarray], 
                          directed: bool = True) -> nx.Graph:
        """
        同期ネットワークを構築
        
        Args:
            series_dict: チャンネル名をキーとする系列の辞書
            directed: 有向グラフにするか
            
        Returns:
            同期ネットワーク（NetworkXグラフ）
        """
        G = nx.DiGraph() if directed else nx.Graph()
        series_names = list(series_dict.keys())
        
        # Add nodes
        for name in series_names:
            G.add_node(name)
        
        # Add edges based on synchronization
        edge_count = 0
        for name_a in series_names:
            for name_b in series_names:
                if name_a == name_b:
                    continue
                
                if not directed and name_a > name_b:  # Skip duplicate edges for undirected
                    continue
                
                sync_profile, max_sync, optimal_lag = self.calculate_sync_profile(
                    series_dict[name_a], series_dict[name_b]
                )
                
                if max_sync >= self.sync_threshold:
                    G.add_edge(name_a, name_b,
                              weight=max_sync,
                              lag=optimal_lag,
                              sync_profile=sync_profile)
                    edge_count += 1
        
        print(f"Sync network created: {G.number_of_nodes()} nodes, {edge_count} edges")
        return G
    
    def cluster_by_sync(self, sync_result: SyncResult, n_clusters: int = 2) -> Dict[str, int]:
        """
        同期パターンに基づいてチャンネルをクラスタリング
        
        Args:
            sync_result: 同期解析結果
            n_clusters: クラスタ数
            
        Returns:
            チャンネル名とクラスタIDの辞書
        """
        # Use 1-sync as distance metric
        distance_matrix = 1 - sync_result.sync_matrix
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='precomputed', 
            linkage='average'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        clusters = {name: int(label) 
                   for name, label in zip(sync_result.series_names, labels)}
        
        return clusters
    
    def analyze_dynamic_sync(self, series_dict: Dict[str, np.ndarray], 
                           window_size: int = WINDOW_SIZE_DEFAULT) -> Dict[str, np.ndarray]:
        """
        時間変化する同期率を解析
        
        Args:
            series_dict: チャンネル名をキーとする系列の辞書
            window_size: スライディングウィンドウのサイズ
            
        Returns:
            各ペアの時間変化する同期率
        """
        dynamic_sync = {}
        series_names = list(series_dict.keys())
        
        for i, name_a in enumerate(series_names):
            for j, name_b in enumerate(series_names[i+1:], i+1):
                series_a = series_dict[name_a].astype(np.float64)
                series_b = series_dict[name_b].astype(np.float64)
                
                sync_rates, optimal_lags = calculate_windowed_sync(
                    series_a, series_b, window_size, self.lag_window
                )
                
                pair_key = f"{name_a}-{name_b}"
                dynamic_sync[pair_key] = {
                    'sync_rates': sync_rates,
                    'optimal_lags': optimal_lags,
                    'time_points': np.arange(window_size//2, 
                                           len(series_a) - window_size//2 + 1)
                }
        
        return dynamic_sync
    
    def detect_sync_anomalies(self, sync_result: SyncResult, 
                            baseline_sync: Optional[np.ndarray] = None) -> np.ndarray:
        """
        同期パターンの異常を検出
        
        Args:
            sync_result: 現在の同期解析結果
            baseline_sync: ベースラインの同期行列（正常時）
            
        Returns:
            各チャンネルの同期異常スコア
        """
        sync_matrix = sync_result.sync_matrix
        n_channels = len(sync_matrix)
        
        if baseline_sync is None:
            # Use median sync as baseline
            baseline_sync = np.median(sync_matrix[sync_matrix < 1.0])
        
        # Calculate deviation from baseline for each channel
        anomaly_scores = np.zeros(n_channels)
        
        for i in range(n_channels):
            # Exclude self-sync (diagonal)
            channel_syncs = np.concatenate([sync_matrix[i, :i], sync_matrix[i, i+1:]])
            
            if isinstance(baseline_sync, np.ndarray):
                baseline_channel = np.concatenate([baseline_sync[i, :i], baseline_sync[i, i+1:]])
                deviation = np.mean(np.abs(channel_syncs - baseline_channel))
            else:
                deviation = np.std(channel_syncs)
            
            anomaly_scores[i] = deviation
        
        # Normalize scores
        if np.std(anomaly_scores) > 0:
            anomaly_scores = (anomaly_scores - np.mean(anomaly_scores)) / np.std(anomaly_scores)
        
        return anomaly_scores

# ===============================
# Utility Functions
# ===============================
def extract_jump_events(anomaly_scores: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """
    異常スコアからジャンプイベントを抽出
    
    Args:
        anomaly_scores: Lambda³異常スコア
        threshold: ジャンプとみなす閾値（標準偏差の倍数）
        
    Returns:
        バイナリジャンプイベント系列
    """
    mean_score = np.mean(anomaly_scores)
    std_score = np.std(anomaly_scores)
    jump_threshold = mean_score + threshold * std_score
    
    return (anomaly_scores > jump_threshold).astype(np.float64)

def prepare_sync_data(lambda3_results: Dict[str, Dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Lambda³結果から同期解析用のデータを準備
    
    Args:
        lambda3_results: 各チャンネルのLambda³解析結果
        
    Returns:
        anomaly_scores_dict: 異常スコアの辞書
        jump_events_dict: ジャンプイベントの辞書
    """
    anomaly_scores_dict = {}
    jump_events_dict = {}
    
    for channel, result in lambda3_results.items():
        # Extract anomaly scores
        if 'scores' in result:
            anomaly_scores_dict[channel] = result['scores']
            
            # Extract jump events
            jump_events = extract_jump_events(result['scores'])
            jump_events_dict[channel] = jump_events
        
        # Also check for pre-computed jump events
        if 'jump_events' in result:
            jump_events_dict[channel] = result['jump_events']
    
    return anomaly_scores_dict, jump_events_dict
