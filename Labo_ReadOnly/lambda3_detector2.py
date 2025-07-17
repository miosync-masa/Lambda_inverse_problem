"""
Lambda³ Zero-Shot Dual Anomaly Detection System - QΛBreak_Version
Author: Based on Iizumi's Lambda³ Theory
"""

import json
import os
import pickle
import time
import pywt
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numba
from numba import cuda, jit, njit, prange
from numba.typed import Dict as NumbaDict
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import hilbert
from scipy.stats import entropy as scipy_entropy
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ===============================
# Global Constants (for JIT Optimization)
# ===============================

DELTA_PERCENTILE = 94.0          # Percentile threshold for global jump detection (large jumps)
LOCAL_WINDOW_SIZE = 15           # Window size for local statistics (used in adaptive detection)
LOCAL_JUMP_PERCENTILE = 91.0     # Percentile threshold for local jumps
WINDOW_SIZE = 30                 # General-purpose window size (e.g., for rolling std/mean)

# Multi-scale jump detection parameters
MULTI_SCALE_WINDOWS = [3, 5, 10, 20, 40]      # Detect jumps at multiple temporal resolutions
MULTI_SCALE_PERCENTILES = [75.0, 80.0, 85.0, 90.0, 93.0, 95.0]  # Adaptive thresholds for each scale

# ===============================
# Global Constants (for KERNEL optimization) - Tuned Version
# ===============================

DEFAULT_KERNEL_TYPE = 3      # 3 = Laplacian kernel as default (RBF=1, Poly=2, Laplace=3)
DEFAULT_GAMMA = 1.0          # Gamma parameter for kernel functions
DEFAULT_DEGREE = 3           # Degree for polynomial kernel
DEFAULT_COEF0 = 1.0          # Coefficient for polynomial kernel
DEFAULT_ALPHA = 0.01         # Regularization weight (Tikhonov or ridge-like penalties)

# ===============================
# Data Class Definitions
# ===============================

@dataclass
class Lambda3Result:
    """
    Data class for storing results of Lambda³ structural analysis.
    """
    paths: Dict[int, np.ndarray]                   # Structure tensor paths for each component
    topological_charges: Dict[int, float]          # Topological charge Q_Λ for each path
    stabilities: Dict[int, float]                  # Topological stability σ_Q for each path
    energies: Dict[int, float]                     # Pulsation/energy metrics for each path
    entropies: Dict[int, Dict[str, float]]         # Multi-type entropies (Shannon, Rényi, Tsallis, etc.)
    classifications: Dict[int, str]                # Path-level classification labels (if any)
    jump_structures: Optional[Dict] = None         # Additional jump/transition structure info (optional)

@dataclass
class L3Config:
    """
    Configuration parameters for Lambda³ analysis.
    """
    alpha: float = 0.05         # L2 regularization (reduced from 0.1)
    beta: float = 0.005         # L1 regularization (reduced from 0.01)
    n_paths: int = 7            # Number of structure tensor paths (increased from 5)
    jump_scale: float = 1.5     # Sensitivity of jump detection (decreased from 2.0)
    use_union: bool = True      # Whether to use union of jumps across scales
    w_topo: float = 0.3         # Weight for topological anomaly score (increased from 0.2)
    w_pulse: float = 0.2        # Weight for pulsation score (decreased from 0.3)

@dataclass
class OptimizationResult:
    """
    Data class for storing optimization results (e.g., feature selection, weight tuning).
    """
    selected_features: List[str]                # Selected features (after optimization)
    weights: Dict[str, float]                   # Component/feature weights
    auc: float                                 # Final AUC score
    feature_correlations: Dict[str, float]      # Correlation scores for selected features
    feature_groups: Optional[Dict[str, List[str]]] = None  # Optional grouping (e.g., for ensemble/group detection)

@dataclass
class DetectionStrategy:
    """
    Data class defining detection strategy.
    """
    method: str                                # "single", "group", "ensemble", etc.
    features: List[str]                        # Features/components used for detection
    weights: Dict[str, float]                  # Weights for each feature/component
    confidence: float                          # Confidence score of current detection logic

# ===============================
# adaptive　Parameter
# ===============================
def compute_adaptive_window_size(
    events: np.ndarray,
    base_window: int = 30,
    min_window: int = 10,
    max_window: int = None
) -> Dict[str, int]:
    n_events, n_features = events.shape

    # データ長依存のmax_window: 例として最大で「n_events // 10」, ただし2000超えない
    if max_window is None:
        max_window = max(100, min(n_events // 10, 2000))

    # データサイズに基づく基準調整
    if n_events > 300:
        size_adjusted_base = base_window
    elif n_events > 100:
        size_adjusted_base = int(base_window * 0.8)
    else:
        size_adjusted_base = int(base_window * 0.6)

    # 最小でもn_events/20は確保
    size_adjusted_base = max(size_adjusted_base, n_events // 20)

    # 1. グローバルボラティリティの計算
    global_std = np.std(events)
    global_mean = np.mean(np.abs(events))
    volatility_ratio = global_std / (global_mean + 1e-10)

    # 2. 時系列的な変動性（隣接イベント間の変化率）
    temporal_changes = np.diff(events, axis=0)
    temporal_volatility = np.mean(np.std(temporal_changes, axis=0))

    # 3. 特徴量間の相関構造の複雑さ
    correlation_matrix = np.corrcoef(events.T)
    correlation_complexity = 1.0 - np.mean(np.abs(correlation_matrix[np.triu_indices(n_features, k=1)]))

    # 4. 局所的な変動パターンの検出
    local_volatilities = []
    for i in range(0, n_events - base_window, base_window // 2):
        window_data = events[i:i + base_window]
        local_volatilities.append(np.std(window_data))

    volatility_variation = np.std(local_volatilities) / (np.mean(local_volatilities) + 1e-10)

    # 5. スペクトル解析による支配的周期の推定
    fft_magnitudes = np.abs(np.fft.fft(events, axis=0))
    # 低周波成分の割合
    low_freq_ratio = np.sum(fft_magnitudes[:n_events//10]) / np.sum(fft_magnitudes[:n_events//2])

    # === ウィンドウサイズの計算 ===

    # 基本スケーリング係数
    scale_factor = 1.0

    # ボラティリティが高い場合は小さいウィンドウ
    if volatility_ratio > 2.0:  # 1.5から2.0に緩和
        scale_factor *= 0.8     # 0.7から0.8に緩和
    elif volatility_ratio < 0.3:  # 0.5から0.3に変更
        scale_factor *= 1.5     # 1.3から1.5に増加

    # 時間的変動が大きい場合は小さいウィンドウ
    if temporal_volatility > global_std * 2.0:  # 1.5から2.0に緩和
        scale_factor *= 0.9     # 0.8から0.9に緩和
    elif temporal_volatility < global_std * 0.3:  # 0.5から0.3に変更
        scale_factor *= 1.4     # 1.2から1.4に増加

    # 相関構造が複雑な場合は大きいウィンドウ
    if correlation_complexity > 0.7:
        scale_factor *= 1.2
    elif correlation_complexity < 0.3:
        scale_factor *= 0.9

    # 局所的変動が大きい場合は適応的に
    if volatility_variation > 1.0:
        scale_factor *= 0.85

    # 低周波成分が支配的な場合は大きいウィンドウ
    if low_freq_ratio > 0.8:
        scale_factor *= 1.4
    elif low_freq_ratio < 0.3:
        scale_factor *= 0.8

    # === 用途別のウィンドウサイズ ===

    # 局所統計量用（標準偏差など）
    local_window = int(size_adjusted_base * scale_factor)
    local_window = np.clip(local_window, min_window, max_window)

    # ジャンプ検出用（より敏感に、小さめ）
    jump_window = int(local_window * 0.5)  # 0.7から0.5に変更
    jump_window = np.clip(jump_window, min_window // 2, max_window // 3)  # 上限も調整

    # エントロピー計算用（より安定に）
    entropy_window = int(local_window * 1.3)
    entropy_window = np.clip(entropy_window, min_window * 2, max_window)

    # マルチスケール解析用（より広いレンジ）
    multiscale_windows = []
    for scale in [0.5, 1.0, 2.0, 4.0, 8.0]:  # 8.0を追加
        window = int(local_window * scale)
        window = np.clip(window, min_window, max_window)
        multiscale_windows.append(window)

    # テンション計算用（ρT）- より大きなウィンドウで安定的に
    tension_window = int(local_window * 1.5)  # 1.5倍で大きく
    tension_window = np.clip(tension_window, min_window, max_window)

    return {
        'local': local_window,
        'jump': jump_window,
        'entropy': entropy_window,
        'tension': tension_window,
        'multiscale': multiscale_windows,
        'volatility_metrics': {
            'global_volatility': volatility_ratio,
            'temporal_volatility': temporal_volatility,
            'correlation_complexity': correlation_complexity,
            'local_variation': volatility_variation,
            'low_freq_ratio': low_freq_ratio,
            'scale_factor': scale_factor
        }
    }

def update_global_constants(window_sizes: Dict[str, int]):
    """グローバル定数を動的に更新"""
    global LOCAL_WINDOW_SIZE, WINDOW_SIZE, MULTI_SCALE_WINDOWS

    LOCAL_WINDOW_SIZE = window_sizes['local']
    WINDOW_SIZE = window_sizes['tension']
    MULTI_SCALE_WINDOWS = window_sizes['multiscale']

    print(f"Window sizes updated:")
    print(f"  LOCAL_WINDOW_SIZE: {LOCAL_WINDOW_SIZE}")
    print(f"  WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"  MULTI_SCALE_WINDOWS: {MULTI_SCALE_WINDOWS}")

# 拡張版：Lambda³構造を考慮した動的調整
def compute_lambda3_adaptive_parameters(events: np.ndarray,
                                      result: Optional['Lambda3Result'] = None) -> Dict[str, any]:
    """
    Lambda³解析結果も考慮した完全な適応的パラメータ設定
    """
    # 基本的なウィンドウサイズ計算
    window_sizes = compute_adaptive_window_size(events)

    # Lambda³構造による調整
    if result is not None:
        # トポロジカルチャージの分布
        charges = np.array(list(result.topological_charges.values()))
        charge_volatility = np.std(charges) / (np.mean(np.abs(charges)) + 1e-10)

        # 安定性の分布
        stabilities = np.array(list(result.stabilities.values()))
        mean_stability = np.mean(stabilities)

        # 構造が不安定な場合は小さいウィンドウ
        if mean_stability > 2.0:
            window_sizes['local'] = int(window_sizes['local'] * 0.8)
            window_sizes['jump'] = int(window_sizes['jump'] * 0.7)

        # ジャンプ頻度による調整
        if result.jump_structures:
            jump_density = result.jump_structures['integrated']['n_total_jumps'] / len(events)
            if jump_density > 0.2:  # ジャンプが多い
                window_sizes['jump'] = max(5, int(window_sizes['jump'] * 0.6))
                # マルチスケールも調整
                window_sizes['multiscale'] = [max(5, int(w * 0.7)) for w in window_sizes['multiscale']]

    # パーセンタイルの動的調整
    volatility_metrics = window_sizes['volatility_metrics']

    # デルタパーセンタイル（ジャンプ検出閾値）
    if volatility_metrics['global_volatility'] > 1.5:
        delta_percentile = 92.0  # より厳しく
    elif volatility_metrics['global_volatility'] < 0.5:
        delta_percentile = 96.0  # より緩く
    else:
        delta_percentile = 94.0

    # 局所ジャンプパーセンタイル
    if volatility_metrics['temporal_volatility'] > volatility_metrics['global_volatility']:
        local_jump_percentile = 89.0  # より敏感に
    else:
        local_jump_percentile = 91.0

    # マルチスケール用のパーセンタイル
    multiscale_percentiles = []
    for i, window in enumerate(window_sizes['multiscale']):
        # 小さいウィンドウほど低いパーセンタイル（敏感）
        base_percentile = 85.0 + (i * 3.0)
        # ボラティリティで調整
        adjusted = base_percentile - (volatility_metrics['global_volatility'] - 1.0) * 5.0
        multiscale_percentiles.append(np.clip(adjusted, 80.0, 98.0))

    return {
        'window_sizes': window_sizes,
        'delta_percentile': delta_percentile,
        'local_jump_percentile': local_jump_percentile,
        'multiscale_percentiles': multiscale_percentiles,
        'adaptive_config': {
            'jump_scale': 1.5 / volatility_metrics['scale_factor'],  # 逆相関
            'alpha': 0.05 * volatility_metrics['scale_factor'],      # 正則化も調整
            'beta': 0.005 * volatility_metrics['scale_factor'],
            'use_union': volatility_metrics['local_variation'] > 0.8,  # 変動が大きければunion
            'w_topo': 0.3 + 0.2 * volatility_metrics['correlation_complexity'],  # 相関が複雑ならトポロジー重視
            'w_pulse': 0.2 + 0.1 * volatility_metrics['temporal_volatility']     # 時間変動が大きければ拍動重視
        }
    }

def apply_adaptive_parameters(detector: 'Lambda3ZeroShotDetector',
                            events: np.ndarray,
                            result: Optional['Lambda3Result'] = None):
    """検出器に適応的パラメータを適用"""

    # パラメータ計算
    params = compute_lambda3_adaptive_parameters(events, result)

    # グローバル定数の更新
    update_global_constants(params['window_sizes'])

    # 検出器の設定更新
    detector.config.jump_scale = params['adaptive_config']['jump_scale']
    detector.config.alpha = params['adaptive_config']['alpha']
    detector.config.beta = params['adaptive_config']['beta']
    detector.config.use_union = params['adaptive_config']['use_union']
    detector.config.w_topo = params['adaptive_config']['w_topo']
    detector.config.w_pulse = params['adaptive_config']['w_pulse']

    # マルチスケールパラメータの更新
    global DELTA_PERCENTILE, LOCAL_JUMP_PERCENTILE, MULTI_SCALE_PERCENTILES
    DELTA_PERCENTILE = params['delta_percentile']
    LOCAL_JUMP_PERCENTILE = params['local_jump_percentile']
    MULTI_SCALE_PERCENTILES = params['multiscale_percentiles']

    print(f"\nAdaptive parameters applied:")
    print(f"  Jump scale: {detector.config.jump_scale:.3f}")
    print(f"  Delta percentile: {DELTA_PERCENTILE:.1f}")
    print(f"  Multiscale percentiles: {MULTI_SCALE_PERCENTILES}")

    return params

# ===============================
# JIT最適化コア関数（ジャンプ検出）
# ===============================
@njit
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """JIT-compiled difference calculation and threshold computation."""
    diff = np.empty(len(data))
    diff[0] = 0
    for i in range(1, len(data)):
        diff[i] = data[i] - data[i-1]

    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, percentile)
    return diff, threshold

@njit
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """JIT-compiled jump detection based on threshold."""
    n = len(diff)
    pos_jumps = np.zeros(n, dtype=np.int32)
    neg_jumps = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if diff[i] > threshold:
            pos_jumps[i] = 1
        elif diff[i] < -threshold:
            neg_jumps[i] = 1

    return pos_jumps, neg_jumps

@njit
def compute_jump_consistency_term(Lambda_matrix, jump_mask, jump_weights):
    n_paths, n_events = Lambda_matrix.shape
    consistency = 0.0

    for p in range(n_paths):
        for i in range(1, n_events):
            delta = np.abs(Lambda_matrix[p, i] - Lambda_matrix[p, i-1])

            if jump_mask[i] == 1:  # boolではなく整数比較
                consistency -= jump_weights[i] * delta
            else:
                consistency += 0.1 * delta

    return consistency

@njit
def calculate_local_std(data: np.ndarray, window: int) -> np.ndarray:
    """JIT-compiled local standard deviation calculation."""
    n = len(data)
    local_std = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)

        subset = data[start:end]
        mean = np.mean(subset)
        variance = np.sum((subset - mean) ** 2) / len(subset)
        local_std[i] = np.sqrt(variance)

    return local_std

@njit
def calculate_rho_t(data: np.ndarray, window: int) -> np.ndarray:
    """JIT-compiled tension scalar (ρT) calculation."""
    n = len(data)
    rho_t = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = i + 1

        subset = data[start:end]
        if len(subset) > 1:
            mean = np.mean(subset)
            variance = np.sum((subset - mean) ** 2) / len(subset)
            rho_t[i] = np.sqrt(variance)
        else:
            rho_t[i] = 0.0

    return rho_t

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

# ===============================
# カーネル関数
# ===============================
@njit
def periodic_kernel(x: np.ndarray, y: np.ndarray,
                   period: float = 1.0,
                   length_scale: float = 1.0) -> float:
    """周期カーネル（周期的パターン検出用）"""
    diff = x - y
    # 周期的距離
    periodic_dist = np.sin(np.pi * np.abs(diff) / period)
    return np.exp(-2 * np.sum(periodic_dist ** 2) / (length_scale ** 2))

@njit
def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """RBFカーネル（ガウシアンカーネル）"""
    diff = x - y
    return np.exp(-gamma * np.dot(diff, diff))

@njit
def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, coef0: float = 1.0) -> float:
    """多項式カーネル"""
    return (np.dot(x, y) + coef0) ** degree

@njit
def sigmoid_kernel(x: np.ndarray, y: np.ndarray, alpha: float = 0.01, coef0: float = 0.0) -> float:
    """シグモイドカーネル"""
    return np.tanh(alpha * np.dot(x, y) + coef0)

@njit
def laplacian_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """ラプラシアンカーネル"""
    diff = np.abs(x - y)
    return np.exp(-gamma * np.sum(diff))

@njit(parallel=True)
def compute_kernel_gram_matrix(data: np.ndarray,
                               kernel_type: int = DEFAULT_KERNEL_TYPE,
                               gamma: float = DEFAULT_GAMMA,
                               degree: int = DEFAULT_DEGREE,
                               coef0: float = DEFAULT_COEF0,
                               alpha: float = DEFAULT_ALPHA,
                               period: float = 10.0,      # 周期カーネル用
                               length_scale: float = 1.0  # 周期カーネル用
                               ) -> np.ndarray:
    """カーネルGram行列の計算（拡張版）"""
    n = data.shape[0]
    K = np.zeros((n, n))

    for i in prange(n):
        for j in range(i, n):
            if kernel_type == 0:  # RBF
                K[i, j] = rbf_kernel(data[i], data[j], gamma)
            elif kernel_type == 1:  # Polynomial
                K[i, j] = polynomial_kernel(data[i], data[j], degree, coef0)
            elif kernel_type == 2:  # Sigmoid
                K[i, j] = sigmoid_kernel(data[i], data[j], alpha, coef0)
            elif kernel_type == 3:  # Laplacian
                K[i, j] = laplacian_kernel(data[i], data[j], gamma)
            elif kernel_type == 4:  # Periodic
                K[i, j] = periodic_kernel(data[i], data[j], period, length_scale)
            else:  # デフォルト: Laplacian
                K[i, j] = laplacian_kernel(data[i], data[j], gamma)

            K[j, i] = K[i, j]  # 対称性

    return K

# ===============================
# 拍動エネルギー計算
# ===============================
@njit
def compute_pulsation_energy_from_jumps(
    pos_jumps: np.ndarray,
    neg_jumps: np.ndarray,
    diff: np.ndarray,
    rho_t: np.ndarray
) -> Tuple[float, float, float]:
    """検出済みジャンプから拍動エネルギーを計算"""
    # ジャンプ強度（検出済みジャンプの差分値の総和）
    pos_intensity = 0.0
    neg_intensity = 0.0

    for i in range(len(diff)):
        if pos_jumps[i] == 1:
            pos_intensity += diff[i]
        if neg_jumps[i] == 1:
            neg_intensity += np.abs(diff[i])

    jump_intensity = pos_intensity + neg_intensity

    # 非対称性（-1 to +1）
    asymmetry = (pos_intensity - neg_intensity) / (pos_intensity + neg_intensity + 1e-10)

    # 拍動パワー（ジャンプ数×強度×平均テンション）
    n_jumps = np.sum(pos_jumps) + np.sum(neg_jumps)

    # ジャンプ位置での平均テンション
    avg_tension = 0.0
    if n_jumps > 0:
        tension_sum = 0.0
        count = 0
        for i in range(len(rho_t)):
            if pos_jumps[i] == 1 or neg_jumps[i] == 1:
                tension_sum += rho_t[i]
                count += 1
        avg_tension = tension_sum / count if count > 0 else 0.0

    pulsation_power = jump_intensity * n_jumps * (1 + avg_tension) / len(diff)

    return jump_intensity, asymmetry, pulsation_power

@njit
def compute_pulsation_energy_from_path(path: np.ndarray) -> Tuple[float, float, float]:
    """パスデータから拍動エネルギーを計算（構造テンソル解析用）"""
    if len(path) < 2:
        return 0.0, 0.0, 0.0

    # 差分とジャンプ検出
    diff = np.diff(path)
    abs_diff = np.abs(diff)
    threshold = np.mean(abs_diff) + 2.0 * np.std(abs_diff)

    # ジャンプ検出
    pos_mask = diff > threshold
    neg_mask = diff < -threshold

    # ジャンプ強度
    pos_intensity = np.sum(diff[pos_mask]) if np.any(pos_mask) else 0.0
    neg_intensity = np.sum(np.abs(diff[neg_mask])) if np.any(neg_mask) else 0.0
    jump_intensity = pos_intensity + neg_intensity

    # 非対称性
    asymmetry = (pos_intensity - neg_intensity) / (pos_intensity + neg_intensity + 1e-10)

    # 拍動パワー
    n_jumps = np.sum(pos_mask) + np.sum(neg_mask)
    pulsation_power = jump_intensity * n_jumps / len(path)

    return jump_intensity, asymmetry, pulsation_power

@njit
def find_jump_indices(path: np.ndarray, jump_scale: float = 2.0):
    """パス内のジャンプindex配列を返す（ΔΛCイベント）"""
    delta = np.abs(np.diff(path))
    th = np.mean(delta) + jump_scale * np.std(delta)
    return np.where(delta > th)[0]

# ===============================
# トポロジカルチャージ計算
# ===============================
@njit(parallel=True)
def compute_topological_charge_jit(path: np.ndarray, n_segments: int = 10) -> Tuple[float, float]:
    """トポロジカルチャージの高速計算"""
    n = len(path)
    closed_path = np.empty(n + 1)
    closed_path[:-1] = path
    closed_path[-1] = path[0]

    # 位相計算
    theta = np.empty(n)
    for i in prange(n):
        theta[i] = np.arctan2(closed_path[i+1], closed_path[i])

    # チャージ計算
    Q_Lambda = 0.0
    for i in range(n-1):
        diff = theta[i+1] - theta[i]
        # 位相のジャンプを処理
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        Q_Lambda += diff
    Q_Lambda /= (2 * np.pi)

    # セグメント安定性
    Q_segments = np.zeros(n_segments)
    for seg in range(n_segments):
        start = seg * n // n_segments
        end = (seg + 1) * n // n_segments
        if end > start + 1:
            seg_sum = 0.0
            for i in range(start, end-1):
                diff = theta[i+1] - theta[i]
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                seg_sum += diff
            Q_segments[seg] = seg_sum

    stability = np.std(Q_segments)
    return Q_Lambda, stability

# ===============================
# エントロピー計算
# ===============================
@njit
def compute_entropy_shannon_jit(path: np.ndarray, eps: float = 1e-10) -> float:
    """Shannonエントロピーの高速計算"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    entropy = 0.0
    for p in norm_path:
        if p > 0:
            entropy -= p * np.log(p)

    return entropy

@njit
def compute_entropy_renyi_jit(path: np.ndarray, alpha: float = 2.0, eps: float = 1e-10) -> float:
    """Renyiエントロピーの高速計算"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    if alpha == 1.0:
        return compute_entropy_shannon_jit(path, eps)

    sum_p_alpha = 0.0
    for p in norm_path:
        sum_p_alpha += p ** alpha

    return (1.0 / (1.0 - alpha)) * np.log(sum_p_alpha)

@njit
def compute_entropy_tsallis_jit(path: np.ndarray, q: float = 1.5, eps: float = 1e-10) -> float:
    """Tsallisエントロピーの高速計算"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    if q == 1.0:
        return compute_entropy_shannon_jit(path, eps)

    sum_p_q = 0.0
    for p in norm_path:
        sum_p_q += p ** q

    return (1.0 - sum_p_q) / (q - 1.0)

@njit
def compute_all_entropies_jit(path: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """全エントロピー指標の高速計算（配列で返す）"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    # 6つの指標を計算
    entropies = np.zeros(6)

    # Shannon
    shannon = 0.0
    for p in norm_path:
        if p > 0:
            shannon -= p * np.log(p)
    entropies[0] = shannon

    # Renyi (α=2)
    sum_p2 = 0.0
    for p in norm_path:
        sum_p2 += p ** 2
    entropies[1] = -np.log(sum_p2)

    # Tsallis (q=1.5)
    sum_p15 = 0.0
    for p in norm_path:
        sum_p15 += p ** 1.5
    entropies[2] = (1.0 - sum_p15) / 0.5

    # Max
    entropies[3] = np.max(norm_path)

    # Min
    entropies[4] = np.min(norm_path)

    # Variance
    mean_p = np.mean(norm_path)
    var = 0.0
    for p in norm_path:
        var += (p - mean_p) ** 2
    entropies[5] = var / len(norm_path)

    return entropies

# ===============================
# 逆問題関連
# ===============================
@njit(parallel=True)
def inverse_problem_objective_jit(Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5):
    """逆問題の目的関数（JIT最適化版）"""
    n_paths, n_events = Lambda_matrix.shape
    reconstruction = np.zeros((n_events, n_events))
    for i in prange(n_events):
        for j in range(n_events):
            for k in range(n_paths):
                reconstruction[i, j] += Lambda_matrix[k, i] * Lambda_matrix[k, j]
    data_fit = np.sum((events_gram - reconstruction)**2)

    tv_reg = 0.0
    for i in range(n_paths - 1):
        for j in range(n_events):
            tv_reg += np.abs(Lambda_matrix[i+1, j] - Lambda_matrix[i, j])
    for i in range(n_paths):
        for j in range(n_events - 1):
            tv_reg += np.abs(Lambda_matrix[i, j+1] - Lambda_matrix[i, j])

    l1_reg = np.sum(np.abs(Lambda_matrix))

    # ジャンプ正則化
    jump_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        n_delta = n_events - 1
        if n_delta == 0:
            continue
        deltas = np.abs(path[1:] - path[:-1])
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + 2.5 * std_delta
        for delta in deltas:
            if delta > threshold:
                jump_reg += jump_weight * delta

    return data_fit + alpha * tv_reg + beta * l1_reg + jump_reg

@njit(parallel=True)
def inverse_problem_topo_objective_jit(
    Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5, topo_weight=0.1
):
    """
    逆問題の目的関数（JIT最適化版）
    - data_fit: 再構成誤差
    - tv_reg: Total Variation正則化
    - l1_reg: L1正則化
    - jump_reg: ジャンプ正則化
    - topo_reg: トポロジカル保存則ペナルティ（QΛ）
    """
    n_paths, n_events = Lambda_matrix.shape
    reconstruction = np.zeros((n_events, n_events))
    for i in prange(n_events):
        for j in range(n_events):
            for k in range(n_paths):
                reconstruction[i, j] += Lambda_matrix[k, i] * Lambda_matrix[k, j]
    data_fit = np.sum((events_gram - reconstruction)**2)

    # Total Variation正則化
    tv_reg = 0.0
    for i in range(n_paths - 1):
        for j in range(n_events):
            tv_reg += np.abs(Lambda_matrix[i+1, j] - Lambda_matrix[i, j])
    for i in range(n_paths):
        for j in range(n_events - 1):
            tv_reg += np.abs(Lambda_matrix[i, j+1] - Lambda_matrix[i, j])

    # L1正則化
    l1_reg = np.sum(np.abs(Lambda_matrix))

    # ジャンプ正則化
    jump_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        n_delta = n_events - 1
        if n_delta == 0:
            continue
        deltas = np.abs(path[1:] - path[:-1])
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + 2.5 * std_delta
        for delta in deltas:
            if delta > threshold:
                jump_reg += jump_weight * delta

    # === QΛトポロジカル保存ペナルティ追加 ===
    topo_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        if n_events < 3:
            continue
        # 位相差列
        phase = np.arctan2(path[1:], path[:-1])
        q_diff = phase[1:] - phase[:-1]
        # 2π補正
        for j in range(len(q_diff)):
            if q_diff[j] > np.pi:
                q_diff[j] -= 2 * np.pi
            elif q_diff[j] < -np.pi:
                q_diff[j] += 2 * np.pi
        # 連続性ペナルティ（2乗和）
        topo_reg += np.sum(q_diff ** 2)

    # === 全体損失 ===
    return data_fit + alpha * tv_reg + beta * l1_reg + jump_reg + topo_weight * topo_reg

@njit
def compute_lambda3_reconstruction_error(paths_matrix: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Lambda³再構成誤差の計算（Tikhonov精神の継承）"""
    n_paths, n_events = paths_matrix.shape
    n_features = events.shape[1]

    # 1. 観測データのGram行列（正規化済み）
    events_gram = np.zeros((n_events, n_events))
    for i in range(n_events):
        for j in range(n_events):
            events_gram[i, j] = np.dot(events[i], events[j])

    # Gram行列の正規化（スケール不変性）
    gram_norm = np.sqrt(np.trace(events_gram @ events_gram))
    if gram_norm > 0:
        events_gram /= gram_norm

    # 2. Lambda³構造による再構成
    recon_gram = np.zeros((n_events, n_events))
    for k in range(n_paths):
        for i in range(n_events):
            for j in range(n_events):
                recon_gram[i, j] += paths_matrix[k, i] * paths_matrix[k, j]

    # 再構成の正規化
    recon_norm = np.sqrt(np.trace(recon_gram @ recon_gram))
    if recon_norm > 0:
        recon_gram /= recon_norm

    # 3. イベントごとの再構成誤差
    event_errors = np.zeros(n_events)
    for i in range(n_events):
        row_error = 0.0
        for j in range(n_events):
            diff = events_gram[i, j] - recon_gram[i, j]
            row_error += diff * diff
        event_errors[i] = np.sqrt(row_error)

    return event_errors

@njit(parallel=True)
def compute_lambda3_hybrid_tikhonov_scores(
    paths_matrix: np.ndarray,
    events: np.ndarray,
    charges: np.ndarray,
    stabilities: np.ndarray,
    alpha: float = 0.5,
    jump_scale: float = 2.0,
    use_union: bool = True,
    w_topo: float = 0.2,
    w_pulse: float = 0.3,
) -> np.ndarray:
    """Lambda³ハイブリッドTikhonov融合異常スコア"""
    n_paths, n_events = paths_matrix.shape

    # 全体誤差
    errors_all = compute_lambda3_reconstruction_error(paths_matrix, events)

    # ジャンプindex
    if use_union:
        idx_set = set()
        for k in range(n_paths):
            idxs = find_jump_indices(paths_matrix[k], jump_scale)
            for idx in idxs:
                idx_set.add(idx+1)
        jump_idx = np.array(list(idx_set), dtype=np.int64)
    else:
        Qarr = np.array([np.sum(np.diff(paths_matrix[k])) for k in range(n_paths)])
        main_idx = np.argmax(np.abs(Qarr))
        idxs = find_jump_indices(paths_matrix[main_idx], jump_scale)
        jump_idx = idxs + 1

    # ジャンプ誤差ベクトル
    jump_error = np.zeros_like(errors_all)
    for idx in jump_idx:
        if idx < len(jump_error):
            jump_error[idx] = errors_all[idx]

    # パスごとの異常度
    path_anomaly_scores = np.zeros(n_paths)
    for p in prange(n_paths):
        path = paths_matrix[p]
        topo_score = np.abs(charges[p]) + 0.5 * stabilities[p]
        # パスから拍動エネルギーを計算（構造テンソル用）
        jump_int, asymm, pulse_pow = compute_pulsation_energy_from_path(path)
        pulse_score = 0.4 * jump_int + 0.3 * np.abs(asymm) + 0.3 * pulse_pow
        path_anomaly_scores[p] = w_topo * topo_score + w_pulse * pulse_score

    # イベントごと加重
    structural_component = np.zeros(n_events)
    for i in prange(n_events):
        for p in range(n_paths):
            contribution = np.abs(paths_matrix[p, i])
            structural_component[i] += contribution * path_anomaly_scores[p]

    # ハイブリッド合成
    hybrid_score = alpha * errors_all + (1 - alpha) * jump_error
    event_scores = hybrid_score + structural_component

    # 標準化
    mean_score = np.mean(event_scores)
    std_score = np.std(event_scores)
    if std_score > 0:
        event_scores = (event_scores - mean_score) / std_score

    return event_scores

# ===============================
# 特徴量抽出モジュール
# ===============================
class Lambda3FeatureExtractor:
    """Lambda³特徴量抽出の統一インターフェース（周波数特徴強化版）"""

    def extract_basic_features(self, result: Lambda3Result, events: np.ndarray = None) -> Dict[str, np.ndarray]:
        """基本特徴量の抽出（更新版）"""
        n_paths = len(result.paths)
        paths_matrix = np.stack(list(result.paths.values()))

        # Lambda³コア物理量
        basic_features = {
            'Q_Λ': np.array([result.topological_charges[i] for i in range(n_paths)]),
            'E': np.array([result.energies[i] for i in range(n_paths)]),
            'σ_Q': np.array([result.stabilities[i] for i in range(n_paths)])
        }

        # エントロピー特徴
        for i in range(n_paths):
            ent = result.entropies[i]
            if isinstance(ent, dict):
                basic_features[f'S_shannon_{i}'] = np.array([ent.get('shannon', 0)])
                basic_features[f'S_renyi_{i}'] = np.array([ent.get('renyi_2', 0)])
                basic_features[f'S_tsallis_{i}'] = np.array([ent.get('tsallis_1.5', 0)])

        # 拍動特徴
        for i in range(n_paths):
            path = paths_matrix[i]
            jump_int, asymm, pulse_pow = compute_pulsation_energy_from_path(path)
            basic_features[f'jump_int_{i}'] = np.array([jump_int])
            basic_features[f'asymm_{i}'] = np.array([asymm])
            basic_features[f'pulse_pow_{i}'] = np.array([pulse_pow])

        # === 新規：パス空間での周波数特徴 ===
        for i in range(n_paths):
            path = paths_matrix[i]
            freq_features = self._extract_frequency_features_from_path(path)
            for fname, fval in freq_features.items():
                basic_features[f'{fname}_{i}'] = np.array([fval])

        return basic_features

    def extract_advanced_features(self, result: Lambda3Result, events: np.ndarray) -> Dict[str, np.ndarray]:
        """高度な特徴量の生成（周波数特徴強化版）"""
        # resultが辞書の場合はそのまま使用、Lambda3Resultの場合は属性にアクセス
        if isinstance(result, dict):
            features = result.copy()
            n_paths = max(int(k.split('_')[-1]) for k in features.keys()
                        if any(k.startswith(prefix) for prefix in ['S_shannon_', 'S_renyi_', 'S_tsallis_'])) + 1
            n_events = events.shape[0]

            if 'Q_Λ' in features:
                Qs = features['Q_Λ']
                Es = features.get('E', np.zeros(n_paths))
                Sigmas = features.get('σ_Q', np.ones(n_paths))
            else:
                Qs = np.zeros(n_paths)
                Es = np.zeros(n_paths)
                Sigmas = np.ones(n_paths)
        else:
            n_paths = len(result.paths)
            n_events = events.shape[0]
            paths_matrix = np.stack(list(result.paths.values()))
            Qs = np.array([result.topological_charges[i] for i in range(n_paths)])
            Es = np.array([result.energies[i] for i in range(n_paths)])
            Sigmas = np.array([result.stabilities[i] for i in range(n_paths)])

        # 2. 物理的に意味のある組み合わせ特徴
        features = {
            "Q_Λ": Qs,
            "σ_Q": Sigmas,
            "E": Es,
            "Q_Λ/σ_Q": Qs / (Sigmas + 1e-8),
            "Q_Λ×E": Qs * Es,
            "sq_Q_Λ": Qs ** 2,
            "E×σ_Q": Es * Sigmas,
            "sqrt_σ_Q": np.sqrt(Sigmas + 1e-8),
            "log_σ_Q": np.log(Sigmas + 1e-8),
            "sqrt_Q_Λ": np.sqrt(np.abs(Qs) + 1e-8),
            "log_Q_Λ": np.log(np.abs(Qs) + 1e-8),
        }

        # === 新規：イベント空間での周波数特徴 ===
        # 各イベント特徴量のFFT解析
        event_fft_features = self._extract_event_frequency_features(events)
        for fname, fvals in event_fft_features.items():
            features[f'event_{fname}'] = fvals

        # 3. エントロピー、拍動、統計特徴（パスごと）
        if hasattr(result, 'paths'):
            paths_matrix = np.stack(list(result.paths.values()))

            # === 新規：パスごとの周波数特徴 ===
            path_freq_features = {}
            for i in range(n_paths):
                path = paths_matrix[i]
                freq_feats = self._extract_frequency_features_from_path(path)
                for fname, fval in freq_feats.items():
                    if fname not in path_freq_features:
                        path_freq_features[fname] = []
                    path_freq_features[fname].append(fval)

            # パス周波数特徴の統計量
            for fname, fvals in path_freq_features.items():
                features[f'path_{fname}_mean'] = np.array([np.mean(fvals)])
                features[f'path_{fname}_std'] = np.array([np.std(fvals)])
                features[f'path_{fname}_max'] = np.array([np.max(fvals)])

            for i in range(n_paths):
                # 既存のエントロピー特徴
                ent = result.entropies[i]
                if isinstance(ent, dict):
                    features[f'S_shannon_{i}'] = np.array([ent.get('shannon', 0)])
                    features[f'S_renyi_{i}'] = np.array([ent.get('renyi_2', 0)])
                    features[f'S_tsallis_{i}'] = np.array([ent.get('tsallis_1.5', 0)])

                # 既存の拍動エネルギー
                path = paths_matrix[i]
                jump_int, asymm, pulse_pow = compute_pulsation_energy_from_path(path)
                features[f'jump_int_{i}'] = np.array([jump_int])
                features[f'asymm_{i}'] = np.array([asymm])
                features[f'pulse_pow_{i}'] = np.array([pulse_pow])

                # 歪度と尖度
                mean_path = np.mean(path)
                std_path = np.std(path)
                if std_path > 1e-10:
                    skew = np.sum((path - mean_path)**3) / (len(path) * std_path**3)
                    kurt = np.sum((path - mean_path)**4) / (len(path) * std_path**4) - 3
                else:
                    skew = 0.0
                    kurt = 0.0
                features[f'skew_{i}'] = np.array([np.nan_to_num(skew)])
                features[f'kurt_{i}'] = np.array([np.nan_to_num(kurt)])

                # 自己相関
                if len(path) > 1:
                    ac = np.correlate(path - mean_path, path - mean_path, mode='full')[len(path)-1:]
                    if np.var(path) > 1e-10:
                        ac = ac / (np.var(path) * np.arange(len(path), 0, -1))
                        features[f'autocorr_{i}'] = np.array([np.mean(ac[:5])])
                    else:
                        features[f'autocorr_{i}'] = np.array([0.0])

        elif isinstance(result, dict):
            for k, v in result.items():
                if k not in features:
                    features[k] = v

        # 4. 主要特徴のペアワイズ組み合わせ
        main_keys = ["Q_Λ", "σ_Q", "E"]
        for i, f1 in enumerate(main_keys):
            for j, f2 in enumerate(main_keys):
                if i < j and len(features[f1]) == len(features[f2]):
                    features[f'{f1}×{f2}'] = features[f1] * features[f2]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        r = features[f1] / (features[f2] + 1e-10)
                        features[f'{f1}/{f2}'] = np.nan_to_num(r, nan=0.0, posinf=10.0, neginf=-10.0)

        # === 新規：Lambda³物理量と周波数特徴の相互作用 ===
        if 'event_freq_peak_amp' in features:
            # Q_Λと周波数振幅の相互作用
            features['Q_Λ×freq_amplitude'] = Qs.mean() * features['event_freq_peak_amp']
            features['σ_Q×freq_amplitude'] = Sigmas.mean() * features['event_freq_peak_amp']
            features['E×freq_amplitude'] = Es.mean() * features['event_freq_peak_amp']

            # 周波数エネルギーとの相互作用
            if 'event_freq_energy' in features:
                features['Q_Λ×freq_energy'] = Qs.mean() * features['event_freq_energy']
                features['Q_Λ/freq_energy'] = Qs.mean() / (features['event_freq_energy'] + 1e-10)

        # 5. 非線形変換（全特徴に適用）
        for k, v in list(features.items()):
            # 周波数関連特徴は既に非線形なのでスキップ
            if 'freq' not in k and 'fft' not in k:
                features[f'log_{k}'] = np.log1p(np.abs(v))
                features[f'sqrt_{k}'] = np.sqrt(np.abs(v))
                features[f'sq_{k}'] = v ** 2
                features[f'sig_{k}'] = 1 / (1 + np.exp(-v))

        # 6. イベントレベルの統計量
        features['event_mean'] = np.mean(events, axis=1)
        features['event_std'] = np.std(events, axis=1)

        return features

    def _extract_frequency_features_from_path(self, path: np.ndarray) -> Dict[str, float]:
        """パスの周波数特徴を抽出"""
        # FFT計算
        fft = np.fft.fft(path)
        fft_abs = np.abs(fft)
        fft_freqs = np.fft.fftfreq(len(path))

        # 正の周波数のみ
        pos_mask = fft_freqs > 0
        pos_freqs = fft_freqs[pos_mask]
        pos_fft = fft_abs[pos_mask]

        features = {}

        if len(pos_fft) > 0:
            # ピーク周波数とその振幅
            peak_idx = np.argmax(pos_fft)
            features['freq_peak'] = pos_freqs[peak_idx]
            features['freq_peak_amp'] = pos_fft[peak_idx]

            # 周波数エネルギー
            features['freq_energy'] = np.sum(pos_fft ** 2)

            # スペクトル重心
            if np.sum(pos_fft) > 0:
                features['freq_centroid'] = np.sum(pos_freqs * pos_fft) / np.sum(pos_fft)
            else:
                features['freq_centroid'] = 0.0

            # スペクトルエントロピー
            if np.sum(pos_fft) > 0:
                norm_fft = pos_fft / np.sum(pos_fft)
                features['freq_entropy'] = -np.sum(norm_fft * np.log(norm_fft + 1e-10))
            else:
                features['freq_entropy'] = 0.0

            # 高周波/低周波比率
            mid_point = len(pos_fft) // 2
            if mid_point > 0:
                low_energy = np.sum(pos_fft[:mid_point] ** 2)
                high_energy = np.sum(pos_fft[mid_point:] ** 2)
                features['freq_hf_lf_ratio'] = high_energy / (low_energy + 1e-10)
            else:
                features['freq_hf_lf_ratio'] = 0.0
        else:
            # デフォルト値
            features = {
                'freq_peak': 0.0,
                'freq_peak_amp': 0.0,
                'freq_energy': 0.0,
                'freq_centroid': 0.0,
                'freq_entropy': 0.0,
                'freq_hf_lf_ratio': 0.0
            }

        return features

    def _extract_event_frequency_features(self, events: np.ndarray) -> Dict[str, np.ndarray]:
        """イベント空間での周波数特徴を抽出"""
        n_events, n_features = events.shape

        # 各特徴次元でFFT
        all_ffts = np.fft.fft(events, axis=0)
        all_fft_abs = np.abs(all_ffts)

        features = {}

        # 全体的な周波数特徴
        # ピーク振幅（各特徴の最大値の平均）
        features['freq_peak_amp'] = np.mean(np.max(all_fft_abs[1:n_events//2], axis=0))

        # 周波数エネルギー（全特徴の平均）
        features['freq_energy'] = np.mean(np.sum(all_fft_abs ** 2, axis=0))

        # 支配的周波数の分散（特徴間の周波数パターンの違い）
        peak_freqs = []
        for f in range(n_features):
            fft_f = all_fft_abs[1:n_events//2, f]
            if len(fft_f) > 0 and np.max(fft_f) > 0:
                peak_idx = np.argmax(fft_f)
                peak_freqs.append(peak_idx / len(fft_f))

        if peak_freqs:
            features['freq_dispersion'] = np.std(peak_freqs)
        else:
            features['freq_dispersion'] = 0.0

        # 低周波成分の割合
        low_freq_cutoff = n_events // 10
        if low_freq_cutoff > 1:
            low_freq_energy = np.sum(all_fft_abs[1:low_freq_cutoff] ** 2)
            total_energy = np.sum(all_fft_abs[1:n_events//2] ** 2)
            features['low_freq_ratio'] = low_freq_energy / (total_energy + 1e-10)
        else:
            features['low_freq_ratio'] = 0.0

        # 特徴間の周波数相関
        if n_features > 1:
            freq_corrs = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    corr = np.corrcoef(all_fft_abs[:, i], all_fft_abs[:, j])[0, 1]
                    freq_corrs.append(corr)
            features['freq_correlation'] = np.mean(freq_corrs)
        else:
            features['freq_correlation'] = 0.0

        # これらはスカラー値なので、イベント数に合わせて拡張
        for fname in list(features.keys()):
            features[fname] = np.full(n_events, features[fname])

        return features

    def project_to_event_space(self,
                              features: Dict[str, np.ndarray],
                              paths_matrix: np.ndarray,
                              event_indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """パス特徴量をイベント空間に射影"""
        n_paths, n_events = paths_matrix.shape

        if event_indices is None:
            event_indices = np.arange(n_events)

        event_features = {}

        for name, vals in features.items():
            if vals.shape[0] == n_paths:  # パス特徴量
                event_scores = np.zeros(len(event_indices))
                for i, evt_idx in enumerate(event_indices):
                    event_scores[i] = np.sum(np.abs(paths_matrix[:, evt_idx]) * vals)
                event_features[name] = event_scores
            elif len(vals) == len(event_indices):  # 既にイベント特徴量
                event_features[name] = vals

        return event_features

# ===============================
# 特徴量最適化モジュール
# ===============================
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

# ===============================
# 統一異常検知システム
# ===============================
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
        # 異常パターン生成関数の初期化
        self.anomaly_patterns = {
            'pulse': self._generate_pulse_anomaly,
            'phase_jump': self._generate_phase_jump_anomaly,
            'periodic': self._generate_periodic_anomaly,
            'structural_decay': self._generate_decay_anomaly,
            'bifurcation': self._generate_bifurcation_anomaly,
            'multi_path': self._generate_multi_path_anomaly,
            'topological_jump': self._generate_topological_jump_anomaly,
            'cascade': self._generate_cascade_anomaly,
            'partial_periodic': self._generate_partial_periodic_anomaly,
            'superposition': self._generate_superposition_anomaly,
            'resonance': self._generate_resonance_anomaly
        }

    def analyze(self, events: np.ndarray, n_paths: int = None) -> Lambda3Result:
        """Lambda³解析の実行"""
        if n_paths is None:
            n_paths = self.config.n_paths

        # 動的パラメータ調整（新規追加）
        adaptive_params = compute_adaptive_window_size(events)
        update_global_constants(adaptive_params)

        # キャッシュチェック
        cache_key = f"{events.shape}_{n_paths}"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        # ジャンプ構造の検出
        jump_structures = self._detect_multiscale_jumps(events)
        self.jump_analyzer = jump_structures

        # 構造テンソル推定（逆問題）
        if jump_structures:
            paths = self._inverse_problem_jump_constrained(events, jump_structures, n_paths)
        else:
            paths = self._solve_inverse_problem(events, n_paths)

        # 物理量計算
        if jump_structures:
            charges, stabilities = self._compute_jump_aware_topology(paths, jump_structures)
            energies = self._compute_pulsation_energies(paths, jump_structures)
            entropies = self._compute_jump_conditional_entropies(paths, jump_structures)
        else:
            charges, stabilities = self._compute_topology(paths)
            energies = self._compute_energies(paths)
            entropies = self._compute_entropies(paths)

        classifications = self._classify_structures(paths, charges, stabilities, jump_structures)

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
        for window, percentile in zip(MULTI_SCALE_WINDOWS, MULTI_SCALE_PERCENTILES):
            jump_analyzer = self._detect_multiscale_jumps_with_params(
                events, window_size=window, percentile=percentile
            )
            jump_scores = self._compute_jump_anomaly_scores(jump_analyzer, events)
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
            k_scores = self._compute_kernel_anomaly_scores_with_params(
                events, result, kernel_type=k_type, **k_params
            )
            kernel_scores_list.append(self._ensure_length(k_scores, n_events))

        # 各カーネルの最良スコアを採用
        kernel_scores = np.max(kernel_scores_list, axis=0)

        # 4. 新規：構造的異常スコア
        structural_scores = self._compute_structural_anomaly_scores(result, events)
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
                optimal_weights = self._optimize_component_weights_aggressive(
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

    def _compute_structural_anomaly_scores(self, result: Lambda3Result, events: np.ndarray) -> np.ndarray:
        """Lambda³構造の歪みを直接評価"""
        n_events = events.shape[0]
        scores = np.zeros(n_events)

        paths_matrix = np.stack(list(result.paths.values()))

        # 1. パス間の相関破壊
        for i in range(n_events):
            if i > 0:
                # 各パスの局所的変化
                local_changes = np.abs(paths_matrix[:, i] - paths_matrix[:, i-1])
                # 変化の不均一性（一部のパスだけ大きく変化）
                scores[i] += np.std(local_changes) * np.max(local_changes)

        # 2. トポロジカルチャージの急変
        charges = np.array(list(result.topological_charges.values()))
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

    def _detect_multiscale_jumps_with_params(self, events: np.ndarray,
                                        window_size: int = None,
                                        percentile: float = None) -> Dict:
        """パラメータ化されたジャンプ検出（動的調整版）"""
        n_events, n_features = events.shape
        jump_data = {'features': {}, 'integrated': {}}

        # 動的パラメータ計算（未指定の場合）
        if window_size is None or percentile is None:
            adaptive_params = compute_adaptive_window_size(events)
            if window_size is None:
                window_size = adaptive_params['jump']
            if percentile is None:
                # ボラティリティに基づく動的パーセンタイル
                volatility = adaptive_params['volatility_metrics']['global_volatility']
                percentile = 94.0 - (volatility - 1.0) * 2.0  # 高ボラティリティでより敏感に
                percentile = np.clip(percentile, 85.0, 98.0)

        # 各特徴次元でのジャンプ検出（カスタムパラメータ使用）
        for f in range(n_features):
            data = events[:, f]

            # 特徴量ごとの局所ボラティリティ
            feature_volatility = np.std(data) / (np.mean(np.abs(data)) + 1e-10)

            # 特徴量別の動的調整
            feature_window = int(window_size * (1.0 / (1.0 + feature_volatility)))
            feature_window = max(5, min(feature_window, 50))

            feature_percentile = percentile - feature_volatility * 1.5
            feature_percentile = np.clip(feature_percentile, 80.0, 98.0)

            # カスタムパーセンタイルでジャンプ検出
            diff, threshold = calculate_diff_and_threshold(data, feature_percentile)
            pos_jumps, neg_jumps = detect_jumps(diff, threshold)

            # カスタムウィンドウサイズで局所適応的ジャンプ
            local_std = calculate_local_std(data, feature_window)
            score = np.abs(diff) / (local_std + 1e-8)

            # 動的な局所閾値（特徴量の特性に応じて）
            local_percentile = feature_percentile - 2.0  # 局所的にはより敏感に
            local_threshold = np.percentile(score[score > 0], local_percentile)
            local_jumps = (score > local_threshold).astype(int)

            # テンションスカラー（動的ウィンドウ）
            tension_window = int(feature_window * 0.8)  # テンション用は少し小さく
            rho_t = calculate_rho_t(data, tension_window)

            # 拍動エネルギー
            jump_intensity, asymmetry, pulse_power = compute_pulsation_energy_from_jumps(
                pos_jumps, neg_jumps, diff, rho_t
            )

            jump_data['features'][f] = {
                'pos_jumps': pos_jumps,
                'neg_jumps': neg_jumps,
                'local_jumps': local_jumps,
                'rho_t': rho_t,
                'diff': diff,
                'threshold': threshold,
                'jump_intensity': jump_intensity,
                'asymmetry': asymmetry,
                'pulse_power': pulse_power,
                'window_size': feature_window,  # 実際に使用したウィンドウサイズ
                'percentile': feature_percentile,  # 実際に使用したパーセンタイル
                'feature_volatility': feature_volatility,  # デバッグ用
                'tension_window': tension_window  # テンション計算用ウィンドウ
            }

        # 統合ジャンプパターン
        jump_data['integrated'] = self._integrate_cross_feature_jumps(jump_data['features'])

        # 適応的パラメータの記録
        jump_data['adaptive_params'] = {
            'base_window': window_size,
            'base_percentile': percentile,
            'n_features_with_jumps': sum(1 for f in jump_data['features'].values()
                                        if np.any(f['pos_jumps']) or np.any(f['neg_jumps'])),
            'avg_feature_volatility': np.mean([f['feature_volatility']
                                              for f in jump_data['features'].values()])
        }

        return jump_data

    def _compute_kernel_anomaly_scores_with_params(self,
                                              events: np.ndarray,
                                              result: Lambda3Result,
                                              kernel_type: int,
                                              **kernel_params) -> np.ndarray:
        """パラメータを指定してカーネル異常スコアを計算"""
        # カーネルGram行列の計算
        K = compute_kernel_gram_matrix(
            events,
            kernel_type=kernel_type,
            gamma=kernel_params.get('gamma', 1.0),
            degree=kernel_params.get('degree', 3),
            coef0=kernel_params.get('coef0', 1.0),
            alpha=kernel_params.get('alpha', 0.01)
        )

        paths_matrix = np.stack(list(result.paths.values()))
        n_events = events.shape[0]

        # カーネル空間での再構成
        K_recon = np.zeros((n_events, n_events))
        for i in range(n_events):
            for j in range(n_events):
                for k in range(len(paths_matrix)):
                    K_recon[i, j] += paths_matrix[k, i] * K[i, j] * paths_matrix[k, j]

        # 正規化
        K_norm = np.sqrt(np.trace(K @ K))
        if K_norm > 0:
            K /= K_norm

        recon_norm = np.sqrt(np.trace(K_recon @ K_recon))
        if recon_norm > 0:
            K_recon /= recon_norm

        # イベントごとの再構成誤差
        kernel_scores = np.zeros(n_events)
        for i in range(n_events):
            row_error = 0.0
            for j in range(n_events):
                diff = K[i, j] - K_recon[i, j]
                row_error += diff * diff
            kernel_scores[i] = np.sqrt(row_error)

        return kernel_scores

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

    def _optimize_component_weights_aggressive(self,
                                         component_scores: Dict[str, np.ndarray],
                                         labels: np.ndarray,
                                         event_indices: np.ndarray = None,
                                         events: np.ndarray = None,
                                         force_all_components: bool = True,
                                         remove_collinearity: bool = False,  # 強制モードではOFF
                                         collinearity_threshold: float = 0.95,
                                         verbose: bool = False) -> Dict[str, float]:
        """
        全コンポーネントを強制的に使用する積極的最適化
        """
        from scipy.optimize import differential_evolution
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
        except:
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
            from scipy.optimize import minimize

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


    def save_results(self,
                    result: Lambda3Result,
                    anomaly_scores: np.ndarray,
                    events: np.ndarray,
                    channel_name: str,
                    save_dir: str = "./lambda3_results",
                    save_full_result: bool = False,
                    compress: bool = True) -> Dict[str, str]:
        """
        Lambda³解析結果をファイルに保存（jump_structures詳細版）

        Args:
            result: Lambda3Result オブジェクト
            anomaly_scores: 異常スコア配列
            events: 元のイベントデータ
            channel_name: チャンネル名
            save_dir: 保存ディレクトリ
            save_full_result: 完全な結果を保存するか（大きくなる可能性）
            compress: npzで圧縮保存するか

        Returns:
            保存したファイルパスの辞書
        """
        # ディレクトリ作成
        channel_dir = os.path.join(save_dir, channel_name)
        os.makedirs(channel_dir, exist_ok=True)

        saved_files = {}

        # 1. 異常スコアの保存（必須）
        scores_path = os.path.join(channel_dir, "anomaly_scores.npy")
        np.save(scores_path, anomaly_scores)
        saved_files['anomaly_scores'] = scores_path

        # 2. ジャンプ構造の詳細保存（拡張版）
        if result.jump_structures:
            # 統合ジャンプ情報
            jump_events = result.jump_structures['integrated']['unified_jumps']
            jump_path = os.path.join(channel_dir, "jump_events.npy")
            np.save(jump_path, jump_events)
            saved_files['jump_events'] = jump_path

            # ジャンプ重要度
            jump_importance = result.jump_structures['integrated']['jump_importance']
            importance_path = os.path.join(channel_dir, "jump_importance.npy")
            np.save(importance_path, jump_importance)
            saved_files['jump_importance'] = importance_path

            # 同期マトリックス
            sync_matrix = result.jump_structures['integrated']['sync_matrix']
            sync_path = os.path.join(channel_dir, "jump_sync_matrix.npy")
            np.save(sync_path, sync_matrix)
            saved_files['jump_sync_matrix'] = sync_path

            # ジャンプクラスター情報
            jump_clusters = result.jump_structures['integrated']['jump_clusters']
            clusters_path = os.path.join(channel_dir, "jump_clusters.json")
            with open(clusters_path, 'w') as f:
                json.dump(jump_clusters, f, indent=2)
            saved_files['jump_clusters'] = clusters_path

            # 各特徴のジャンプ詳細（圧縮保存）
            feature_jumps = {}
            for f_idx, f_data in result.jump_structures['features'].items():
                # NumPy配列として保存
                feature_jumps[f'feature_{f_idx}_pos_jumps'] = f_data['pos_jumps']
                feature_jumps[f'feature_{f_idx}_neg_jumps'] = f_data['neg_jumps']
                feature_jumps[f'feature_{f_idx}_rho_t'] = f_data['rho_t']
                feature_jumps[f'feature_{f_idx}_diff'] = f_data['diff']

                # スカラー値はメタデータとして配列に
                feature_jumps[f'feature_{f_idx}_metadata'] = np.array([
                    f_data['threshold'],
                    f_data['jump_intensity'],
                    f_data['asymmetry'],
                    f_data['pulse_power']
                ])

            features_path = os.path.join(channel_dir, "jump_features.npz")
            if compress:
                np.savez_compressed(features_path, **feature_jumps)
            else:
                np.savez(features_path, **feature_jumps)
            saved_files['jump_features'] = features_path

            # 統合情報のサマリー
            jump_summary = {
                'n_total_jumps': int(result.jump_structures['integrated']['n_total_jumps']),
                'max_sync': float(result.jump_structures['integrated']['max_sync']),
                'n_clusters': len(jump_clusters),
                'n_features': len(result.jump_structures['features'])
            }
            summary_path = os.path.join(channel_dir, "jump_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(jump_summary, f, indent=2)
            saved_files['jump_summary'] = summary_path

        # 3. 主要な物理量のみ保存（軽量化）
        physics_data = {
            'topological_charges': list(result.topological_charges.values()),
            'stabilities': list(result.stabilities.values()),
            'energies': list(result.energies.values()),
            'classifications': result.classifications
        }
        physics_path = os.path.join(channel_dir, "physics_quantities.json")
        with open(physics_path, 'w') as f:
            json.dump(physics_data, f, indent=2)
        saved_files['physics_quantities'] = physics_path

        # 4. メタデータ
        metadata = {
            'channel_name': channel_name,
            'n_events': events.shape[0],
            'n_features': events.shape[1],
            'n_paths': len(result.paths),
            'anomaly_score_stats': {
                'mean': float(np.mean(anomaly_scores)),
                'std': float(np.std(anomaly_scores)),
                'max': float(np.max(anomaly_scores)),
                'min': float(np.min(anomaly_scores)),
                'percentile_95': float(np.percentile(anomaly_scores, 95))
            },
            'n_jumps': int(np.sum(result.jump_structures['integrated']['unified_jumps']))
                       if result.jump_structures else 0,
            'config': asdict(self.config),
            'saved_at': str(np.datetime64('now'))
        }
        metadata_path = os.path.join(channel_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_path

        # 5. 完全な結果オブジェクト（オプション、デバッグ用）
        if save_full_result:
            # 構造テンソル（パス）も含める
            paths_dict = {}
            for i, path in result.paths.items():
                paths_dict[f'path_{i}'] = path
            paths_path = os.path.join(channel_dir, "lambda_paths.npz")
            if compress:
                np.savez_compressed(paths_path, **paths_dict)
            else:
                np.savez(paths_path, **paths_dict)
            saved_files['lambda_paths'] = paths_path

            # エントロピー情報
            entropy_data = {}
            for i, ent_dict in result.entropies.items():
                if isinstance(ent_dict, dict):
                    entropy_data[str(i)] = ent_dict
            entropy_path = os.path.join(channel_dir, "entropies.json")
            with open(entropy_path, 'w') as f:
                json.dump(entropy_data, f, indent=2)
            saved_files['entropies'] = entropy_path

            # Pickleで完全保存
            full_result_path = os.path.join(channel_dir, "full_result.pkl")
            with open(full_result_path, 'wb') as f:
                pickle.dump(result, f)
            saved_files['full_result'] = full_result_path

        print(f"Saved Lambda³ results for {channel_name} to {channel_dir}")
        print(f"  - Anomaly scores: shape={anomaly_scores.shape}, mean={metadata['anomaly_score_stats']['mean']:.3f}")
        print(f"  - Detected jumps: {metadata['n_jumps']}")
        if result.jump_structures:
            print(f"  - Jump clusters: {len(jump_clusters)}")
            print(f"  - Max feature sync: {result.jump_structures['integrated']['max_sync']:.3f}")

        return saved_files

    def visualize_results(self,
                         events: np.ndarray,
                         result: Lambda3Result,
                         anomaly_scores: np.ndarray = None) -> plt.Figure:
        """統合的な可視化（ジャンプ構造を中心に）"""
        if anomaly_scores is None:
            anomaly_scores = self.detect_anomalies(result, events)

        fig = plt.figure(figsize=(20, 15))

        # 1. ジャンプ構造の可視化
        ax1 = plt.subplot(3, 4, 1)
        if result.jump_structures:
            integrated = result.jump_structures['integrated']
            ax1.plot(integrated['jump_importance'], 'b-', label='Jump Importance')
            ax1.scatter(np.where(integrated['unified_jumps'])[0],
                       integrated['jump_importance'][integrated['unified_jumps'] == 1],
                       color='red', s=50, label='Jump Events')

            # クラスターをハイライト
            for cluster in integrated['jump_clusters']:
                ax1.axvspan(cluster['start'], cluster['end'], alpha=0.3, color='yellow')
        ax1.set_title('Jump Structure Analysis')
        ax1.set_xlabel('Event Index')
        ax1.set_ylabel('Importance')
        ax1.legend()

        # 2. 同期マトリックス
        ax2 = plt.subplot(3, 4, 2)
        if result.jump_structures:
            sync_matrix = result.jump_structures['integrated']['sync_matrix']
            im = ax2.imshow(sync_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax2)
        ax2.set_title('Feature Synchronization Matrix')

        # 3. 異常スコアの時系列
        ax3 = plt.subplot(3, 4, 3)
        ax3.plot(anomaly_scores, 'g-', linewidth=2)
        ax3.axhline(y=2.0, color='r', linestyle='--', label='Critical Threshold')
        ax3.axhline(y=1.0, color='orange', linestyle='--', label='Warning Threshold')
        ax3.set_title('Anomaly Scores (Zero-Shot)')
        ax3.set_xlabel('Event Index')
        ax3.set_ylabel('Score')
        ax3.legend()

        # 4. トポロジカル異常マップ
        ax4 = plt.subplot(3, 4, 4)
        for i in result.paths:
            ax4.scatter(result.topological_charges[i],
                       result.stabilities[i],
                       s=100, label=f'Path {i}')
        ax4.set_xlabel('Topological Charge Q_Lambda')
        ax4.set_ylabel('Stability Sigma_Q')
        ax4.set_title('Topological Anomaly Map')
        ax4.legend()

        # 5-8. 各パスの詳細（ジャンプ強調）
        for idx, (i, path) in enumerate(result.paths.items()):
            if idx >= 4:
                break

            ax = plt.subplot(3, 4, 5 + idx)
            ax.plot(path, 'b-', alpha=0.7, label='Lambda Structure')

            # ジャンプイベントをマーク
            if result.jump_structures:
                jump_mask = result.jump_structures['integrated']['unified_jumps']
                jump_indices = np.where(jump_mask)[0]
                if len(jump_indices) > 0:
                    ax.scatter(jump_indices, path[jump_indices],
                              color='red', s=50, label='Jumps', zorder=5)

            ax.set_title(f'Path {i}: {result.classifications[i]}')
            ax.set_xlabel('Event Index')
            ax.set_ylabel('Lambda Amplitude')
            ax.legend()

            # 物理量表示
            textstr = f'$Q_\\Lambda$={result.topological_charges[i]:.3f}\n' \
                      f'$\\sigma_Q$={result.stabilities[i]:.3f}\n' \
                      f'$E$={result.energies[i]:.3f}'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 9. エントロピー比較
        ax9 = plt.subplot(3, 4, 9)
        entropy_types = ['shannon', 'renyi_2', 'tsallis_1.5']
        for i, ent_type in enumerate(entropy_types):
            values = []
            for p in result.paths:
                ent_dict = result.entropies[p]
                if isinstance(ent_dict, dict):
                    values.append(ent_dict.get(ent_type, 0))
                else:
                    values.append(0)
            ax9.bar(np.arange(len(values)) + i*0.3, values, 0.3, label=ent_type)
        ax9.set_title('Multi-Entropy Comparison')
        ax9.set_xlabel('Path Index')
        ax9.set_ylabel('Entropy')
        ax9.legend()

        # 10. 拍動エネルギー分布
        ax10 = plt.subplot(3, 4, 10)
        if result.jump_structures:
            pulse_energies = []
            feature_names = []
            for f_idx, f_data in result.jump_structures['features'].items():
                pulse_energies.append(f_data['pulse_power'])
                feature_names.append(f'F{f_idx}')
            ax10.bar(range(len(pulse_energies)), pulse_energies)
            ax10.set_title('Pulsation Energy Distribution (Features)')
            ax10.set_xlabel('Feature Index')
            ax10.set_ylabel('Pulse Power')
            if len(feature_names) <= 10:
                ax10.set_xticks(range(len(feature_names)))
                ax10.set_xticklabels(feature_names)
        else:
            # フォールバック：パスから計算
            pulse_energies = []
            for p, path in result.paths.items():
                _, _, pulse_power = compute_pulsation_energy_from_path(path)
                pulse_energies.append(pulse_power)
            ax10.bar(range(len(pulse_energies)), pulse_energies)
            ax10.set_title('Pulsation Energy Distribution (Paths)')
            ax10.set_xlabel('Path Index')
            ax10.set_ylabel('Pulse Power')

        # 11. PCA投影
        ax11 = plt.subplot(3, 4, 11)
        if events.shape[1] > 2:
            pca = PCA(n_components=2)
            events_2d = pca.fit_transform(events)
            scatter = ax11.scatter(events_2d[:, 0], events_2d[:, 1],
                                  c=anomaly_scores, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax11)
        ax11.set_title('Event Space (PCA) - Anomaly Colored')

        # 12. カーネル空間投影
        ax12 = plt.subplot(3, 4, 12)
        # 簡易的なカーネルPCA可視化
        K = compute_kernel_gram_matrix(events[:50], kernel_type=3, gamma=1.0)  # サンプリング
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        idx = np.argsort(eigenvalues)[::-1][:2]
        kernel_proj = eigenvectors[:, idx]
        ax12.scatter(kernel_proj[:, 0], kernel_proj[:, 1],
                    c=anomaly_scores[:50], cmap='plasma', alpha=0.7)
        ax12.set_title('Kernel Space Projection (Laplacian)')

        plt.tight_layout()
        return fig

    # ===============================
    # 内部メソッド（完全実装）
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

    def _standardize_scores(self, scores: np.ndarray) -> np.ndarray:
        """スコアの頑健な標準化"""
        # 中央値と四分位範囲を使用（外れ値に頑健）
        median_score = np.median(scores)
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25

        if iqr > 0:
            standardized = (scores - median_score) / (1.5 * iqr)
        else:
            # フォールバック：通常の標準化
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            if std_score > 0:
                standardized = (scores - mean_score) / std_score
            else:
                standardized = scores

        return standardized

    def _solve_inverse_problem(self, events: np.ndarray, n_paths: int, topo_weight: float = 0.1) -> Dict[int, np.ndarray]:
        """
        構造テンソル推定（通常＆トポロジカル保存破れ両対応, MAX合成）
        """
        events_gram = np.ascontiguousarray(events @ events.T)
        _, V = np.linalg.eigh(events_gram)
        Lambda_init = V[:, -n_paths:].T.flatten()

        # 1. 通常逆問題
        def objective_no_topo(Lambda_flat):
            Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
            return inverse_problem_objective_jit(
                Lambda_matrix, events_gram, self.config.alpha, self.config.beta, jump_weight=0.5
            )

        result_no_topo = minimize(
            objective_no_topo,
            Lambda_init,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        Lambda_no_topo = result_no_topo.x.reshape(n_paths, events.shape[0])

        # 2. トポロジカルペナルティあり
        def objective_with_topo(Lambda_flat):
            Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
            return inverse_problem_topo_objective_jit(
                Lambda_matrix, events_gram, self.config.alpha, self.config.beta, jump_weight=0.5, topo_weight=topo_weight
            )

        result_with_topo = minimize(
            objective_with_topo,
            Lambda_init,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        Lambda_with_topo = result_with_topo.x.reshape(n_paths, events.shape[0])

        # 3. パスごと/イベントごとにMAX合成（絶対値ベースなど）
        Lambda_max = np.maximum(np.abs(Lambda_no_topo), np.abs(Lambda_with_topo))

        # 4. 返却（正規化）
        return {i: path / (np.linalg.norm(path) + 1e-8) for i, path in enumerate(Lambda_max)}

    def _compute_topology(self, paths: Dict[int, np.ndarray]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """トポロジカル量の計算"""
        charges = {}
        stabilities = {}

        for i, path in paths.items():
            Q, sigma = compute_topological_charge_jit(path)
            charges[i] = Q
            stabilities[i] = sigma

        return charges, stabilities

    def _compute_energies(self, paths: Dict[int, np.ndarray]) -> Dict[int, float]:
        """エネルギー計算"""
        energies = {}
        for i, path in paths.items():
            basic_energy = np.sum(path**2)
            jump_int, _, pulse_pow = compute_pulsation_energy_from_path(path)
            energies[i] = basic_energy + 0.3 * pulse_pow

        return energies

    def _compute_entropies(self, paths: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
        """エントロピー計算"""
        entropies = {}
        entropy_keys = ["shannon", "renyi_2", "tsallis_1.5", "max", "min", "var"]

        for i, path in paths.items():
            all_entropies = compute_all_entropies_jit(path)
            entropy_dict = {}
            for j, key in enumerate(entropy_keys):
                entropy_dict[key] = all_entropies[j]
            entropies[i] = entropy_dict

        return entropies

    def _classify_structures(self,
                           paths: Dict[int, np.ndarray],
                           charges: Dict[int, float],
                           stabilities: Dict[int, float],
                           jump_structures: Optional[Dict] = None) -> Dict[int, str]:
        """構造分類"""
        classifications = {}

        for i in paths.keys():
            Q = charges[i]
            sigma = stabilities[i]

            # 基本分類
            if Q < -0.5:
                base = "反物質的構造（吸収系）"
            elif Q > 0.5:
                base = "物質的構造（放出系）"
            else:
                base = "中性構造（平衡）"

            # 修飾
            tags = []

            if sigma > 2.5:
                tags.append("不安定/カオス的")
            elif sigma < 0.5:
                tags.append("超安定")

            # ジャンプ特性があれば追加
            if jump_structures and i < len(jump_structures['features']):
                feature_data = jump_structures['features'].get(i, list(jump_structures['features'].values())[0])
                pulse_power = feature_data['pulse_power']
                asymmetry = feature_data['asymmetry']

                if pulse_power > 5:
                    tags.append("高頻度拍動")
                elif pulse_power < 0.1:
                    tags.append("静的構造")

                if abs(asymmetry) > 0.7:
                    if asymmetry > 0:
                        tags.append("正方向優位")
                    else:
                        tags.append("負方向優位")

            # 分類完成
            if tags:
                classifications[i] = base + "・" + "／".join(tags)
            else:
                classifications[i] = base

        return classifications

    # ===============================
    # ジャンプ解析メソッド
    # ===============================

    def _detect_multiscale_jumps(self, events: np.ndarray) -> Dict:
        """多次元・多スケールジャンプ検出"""
        n_events, n_features = events.shape
        jump_data = {'features': {}, 'integrated': {}}

        # 各特徴次元でのジャンプ検出
        for f in range(n_features):
            data = events[:, f]

            # ここで配列サイズを確認
            assert len(data) == n_events, f"Feature {f} has wrong size: {len(data)} vs {n_events}"

            # 基本ジャンプ検出
            diff, threshold = calculate_diff_and_threshold(data, DELTA_PERCENTILE)
            pos_jumps, neg_jumps = detect_jumps(diff, threshold)

            # 局所適応的ジャンプ
            local_std = calculate_local_std(data, LOCAL_WINDOW_SIZE)
            score = np.abs(diff) / (local_std + 1e-8)
            local_threshold = np.percentile(score, LOCAL_JUMP_PERCENTILE)
            local_jumps = (score > local_threshold).astype(int)

            # テンションスカラー
            rho_t = calculate_rho_t(data, WINDOW_SIZE)

            # 拍動エネルギー
            jump_intensity, asymmetry, pulse_power = compute_pulsation_energy_from_jumps(
                pos_jumps, neg_jumps, diff, rho_t
            )

            jump_data['features'][f] = {
                'pos_jumps': pos_jumps,
                'neg_jumps': neg_jumps,
                'local_jumps': local_jumps,
                'rho_t': rho_t,
                'diff': diff,
                'threshold': threshold,
                'jump_intensity': jump_intensity,
                'asymmetry': asymmetry,
                'pulse_power': pulse_power
            }

        # 統合ジャンプパターン
        jump_data['integrated'] = self._integrate_cross_feature_jumps(jump_data['features'])

        return jump_data

    def _integrate_cross_feature_jumps(self, feature_jumps: Dict) -> Dict:
        """特徴間のジャンプ同期性を解析"""
        n_features = len(feature_jumps)
        features_list = list(feature_jumps.keys())

        # 統合ジャンプマスク
        first_key = features_list[0]
        n_events = len(feature_jumps[first_key]['pos_jumps'])
        unified_jumps = np.zeros(n_events, dtype=np.int64)

        # ジャンプ重要度
        jump_importance = np.zeros(n_events)

        for f in features_list:
            jumps = feature_jumps[f]['pos_jumps'] | feature_jumps[f]['neg_jumps']
            unified_jumps |= jumps
            jump_importance += jumps.astype(float)

        # ジャンプ同期率の計算
        sync_matrix = np.zeros((n_features, n_features))
        for i, f1 in enumerate(features_list):
            for j, f2 in enumerate(features_list):
                if i < j:
                    jumps1 = feature_jumps[f1]['pos_jumps'] | feature_jumps[f1]['neg_jumps']
                    jumps2 = feature_jumps[f2]['pos_jumps'] | feature_jumps[f2]['neg_jumps']

                    # 同期プロファイル計算
                    _, _, max_sync, optimal_lag = calculate_sync_profile_jit(
                        jumps1.astype(np.float64),
                        jumps2.astype(np.float64),
                        lag_window=5
                    )
                    sync_matrix[i, j] = max_sync
                    sync_matrix[j, i] = max_sync

        # ジャンプクラスター検出
        jump_clusters = self._detect_jump_clusters(unified_jumps, jump_importance)

        return {
            'unified_jumps': unified_jumps,
            'jump_importance': jump_importance / n_features,  # 正規化
            'sync_matrix': sync_matrix,
            'jump_clusters': jump_clusters,
            'n_total_jumps': np.sum(unified_jumps),
            'max_sync': np.max(sync_matrix[np.triu_indices(n_features, k=1)])
        }

    def _detect_jump_clusters(self,
                            unified_jumps: np.ndarray,
                            jump_importance: np.ndarray,
                            min_cluster_size: int = 3) -> List[Dict]:
        """ジャンプのクラスター（連続的な構造変化）を検出"""
        clusters = []
        in_cluster = False
        cluster_start = 0

        for i in range(len(unified_jumps)):
            if unified_jumps[i] and not in_cluster:
                in_cluster = True
                cluster_start = i
            elif not unified_jumps[i] and in_cluster:
                # クラスター終了
                cluster_size = i - cluster_start
                if cluster_size >= min_cluster_size:
                    clusters.append({
                        'start': cluster_start,
                        'end': i,
                        'size': cluster_size,
                        'indices': list(range(cluster_start, i)),
                        'density': np.mean(jump_importance[cluster_start:i]),
                        'total_importance': np.sum(jump_importance[cluster_start:i])
                    })
                in_cluster = False

        # 最後のクラスター処理
        if in_cluster:
            cluster_size = len(unified_jumps) - cluster_start
            if cluster_size >= min_cluster_size:
                clusters.append({
                    'start': cluster_start,
                    'end': len(unified_jumps),
                    'size': cluster_size,
                    'indices': list(range(cluster_start, len(unified_jumps))),
                    'density': np.mean(jump_importance[cluster_start:]),
                    'total_importance': np.sum(jump_importance[cluster_start:])
                })

        return clusters

    def _inverse_problem_jump_constrained(self,
                                      events: np.ndarray,
                                      jump_structures: Dict,
                                      n_paths: int,
                                      topo_weight: float = 0.1) -> Dict[int, np.ndarray]:
        """
        ジャンプ構造を活用した逆問題
        - トポロジカル保存律「あり」と「なし」両方計算し、要素ごとにMAX合成して返す
        """
        jump_mask = jump_structures['integrated']['unified_jumps']
        jump_weights = jump_structures['integrated']['jump_importance']
        events_gram = np.ascontiguousarray(events @ events.T)
        Lambda_init = self._initialize_with_jump_structure(events, jump_mask, jump_weights, n_paths)

        # 1. 保存律なしで最適化
        def objective_no_topo(Lambda_flat):
            Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
            base_obj = inverse_problem_objective_jit(
                Lambda_matrix, events_gram, self.config.alpha, self.config.beta, jump_weight=0.5)
            jump_term = self._compute_jump_consistency_term(Lambda_matrix, jump_mask, jump_weights)
            return base_obj + jump_term

        result_no_topo = minimize(
            objective_no_topo,
            Lambda_init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        Lambda_no_topo = result_no_topo.x.reshape(n_paths, events.shape[0])

        # 2. 保存律ありで最適化
        def objective_with_topo(Lambda_flat):
            Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
            base_obj = inverse_problem_topo_objective_jit(
                Lambda_matrix, events_gram, self.config.alpha, self.config.beta, jump_weight=0.5, topo_weight=topo_weight)
            jump_term = self._compute_jump_consistency_term(Lambda_matrix, jump_mask, jump_weights)
            return base_obj + jump_term

        result_with_topo = minimize(
            objective_with_topo,
            Lambda_init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        Lambda_with_topo = result_with_topo.x.reshape(n_paths, events.shape[0])

        # 3. MAX合成
        Lambda_max = np.maximum(np.abs(Lambda_no_topo), np.abs(Lambda_with_topo))

        # 4. 各パス正規化して辞書で返却
        return {i: path / (np.linalg.norm(path) + 1e-8) for i, path in enumerate(Lambda_max)}

    def _initialize_with_jump_structure(self,
                                      events: np.ndarray,
                                      jump_mask: np.ndarray,
                                      jump_weights: np.ndarray,
                                      n_paths: int) -> np.ndarray:
        """ジャンプ構造を反映した初期値生成"""
        n_events = events.shape[0]
        Lambda_init = np.zeros((n_paths, n_events))

        # 固有値分解ベース
        _, V = np.linalg.eigh(events @ events.T)
        base_paths = V[:, -n_paths:].T

        # ジャンプ位置で不連続性を導入
        for p in range(n_paths):
            Lambda_init[p] = base_paths[p]

            # ジャンプ位置での値を強調
            for i in range(n_events):
                if jump_mask[i]:
                    if p % 2 == 0:
                        Lambda_init[p, i] *= (1 + jump_weights[i])
                    else:
                        Lambda_init[p, i] *= -(1 + jump_weights[i])

        return Lambda_init

    def _compute_jump_consistency_term(self,
                                     Lambda_matrix: np.ndarray,
                                     jump_mask: np.ndarray,
                                     jump_weights: np.ndarray) -> float:
        """ジャンプ整合性の評価"""
        return compute_jump_consistency_term(Lambda_matrix, jump_mask, jump_weights)

    def _compute_jump_aware_topology(self,
                                   paths: Dict[int, np.ndarray],
                                   jump_structures: Dict) -> Tuple[Dict[int, float], Dict[int, float]]:
        """ジャンプ構造を考慮したトポロジカル量計算"""
        charges = {}
        stabilities = {}

        for i, path in paths.items():
            # 基本的なトポロジカルチャージ
            Q, sigma = compute_topological_charge_jit(path)

            # ジャンプ位置での位相変化を考慮
            jump_mask = jump_structures['integrated']['unified_jumps']
            jump_phase_shift = 0.0

            for j in range(1, len(path)):
                if jump_mask[j]:
                    # ジャンプ位置での位相変化
                    phase_diff = np.arctan2(path[j], path[j-1]) - \
                               np.arctan2(path[j-1], path[j-2] if j > 1 else path[0])
                    jump_phase_shift += phase_diff

            # ジャンプ補正したチャージ
            charges[i] = Q + jump_phase_shift / (2 * np.pi)
            stabilities[i] = sigma

        return charges, stabilities

    def _compute_pulsation_energies(self,
                                  paths: Dict[int, np.ndarray],
                                  jump_structures: Dict) -> Dict[int, float]:
        """拍動エネルギーの計算（ジャンプ構造を優先使用）"""
        energies = {}

        for i, path in paths.items():
            # 基本エネルギー
            basic_energy = np.sum(path**2)

            # 対応する特徴のジャンプエネルギーを統合
            total_pulse_power = 0.0
            n_features = len(jump_structures['features'])

            for f_idx, f_data in jump_structures['features'].items():
                pulse_power = f_data['pulse_power']
                total_pulse_power += pulse_power

            avg_pulse_power = total_pulse_power / n_features if n_features > 0 else 0.0

            # 統合エネルギー
            energies[i] = basic_energy + 0.3 * avg_pulse_power

        return energies

    def _compute_jump_conditional_entropies(self,
                                      paths: Dict[int, np.ndarray],
                                      jump_structures: Dict) -> Dict[int, Dict[str, float]]:
        """ジャンプイベントでの条件付きエントロピー（局所構造解析強化版）"""
        entropies = {}
        jump_mask = jump_structures['integrated']['unified_jumps']

        # ジャンプ周辺の窓サイズ
        JUMP_WINDOW = 5  # ジャンプ前後5イベント

        for i, path in paths.items():
            # 全体のエントロピー
            all_entropies = compute_all_entropies_jit(path)
            entropy_keys = ["shannon", "renyi_2", "tsallis_1.5", "max", "min", "var"]

            # ジャンプ位置と非ジャンプ位置で分離
            jump_indices = np.where(jump_mask)[0]
            non_jump_indices = np.where(~jump_mask)[0]

            entropy_dict = {}

            # 全体エントロピー
            for j, key in enumerate(entropy_keys):
                entropy_dict[key] = all_entropies[j]

            # ジャンプ条件付きエントロピー
            if len(jump_indices) > 0:
                jump_path = path[jump_indices]
                jump_entropies = compute_all_entropies_jit(jump_path)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_jump"] = jump_entropies[j]

            # 非ジャンプ条件付きエントロピー
            if len(non_jump_indices) > 0:
                non_jump_path = path[non_jump_indices]
                non_jump_entropies = compute_all_entropies_jit(non_jump_path)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_non_jump"] = non_jump_entropies[j]

            # === 新規追加：ジャンプ周辺のエントロピー解析 ===

            # 1. ジャンプ前後の窓内エントロピー
            jump_vicinity_indices = set()
            for jump_idx in jump_indices:
                for offset in range(-JUMP_WINDOW, JUMP_WINDOW + 1):
                    vicinity_idx = jump_idx + offset
                    if 0 <= vicinity_idx < len(path):
                        jump_vicinity_indices.add(vicinity_idx)

            if jump_vicinity_indices:
                vicinity_indices = np.array(sorted(jump_vicinity_indices))
                vicinity_path = path[vicinity_indices]
                vicinity_entropies = compute_all_entropies_jit(vicinity_path)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_vicinity"] = vicinity_entropies[j]

            # 2. ジャンプ前のエントロピー（構造崩壊の前兆）
            pre_jump_indices = []
            for jump_idx in jump_indices:
                for offset in range(1, JUMP_WINDOW + 1):
                    pre_idx = jump_idx - offset
                    if pre_idx >= 0:
                        pre_jump_indices.append(pre_idx)

            if pre_jump_indices:
                pre_jump_path = path[pre_jump_indices]
                pre_jump_entropies = compute_all_entropies_jit(pre_jump_path)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_pre_jump"] = pre_jump_entropies[j]

            # 3. ジャンプ後のエントロピー（構造再編成）
            post_jump_indices = []
            for jump_idx in jump_indices:
                for offset in range(1, JUMP_WINDOW + 1):
                    post_idx = jump_idx + offset
                    if post_idx < len(path):
                        post_jump_indices.append(post_idx)

            if post_jump_indices:
                post_jump_path = path[post_jump_indices]
                post_jump_entropies = compute_all_entropies_jit(post_jump_path)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_post_jump"] = post_jump_entropies[j]

            # 4. エントロピー勾配（ジャンプによる構造変化の急峻さ）
            if len(jump_indices) > 0:
                entropy_gradients = []
                for jump_idx in jump_indices:
                    # ジャンプ前後のローカルエントロピーを計算
                    pre_start = max(0, jump_idx - JUMP_WINDOW)
                    pre_end = jump_idx
                    post_start = jump_idx + 1
                    post_end = min(len(path), jump_idx + JUMP_WINDOW + 1)

                    if pre_end > pre_start and post_end > post_start:
                        pre_local = compute_all_entropies_jit(path[pre_start:pre_end])
                        post_local = compute_all_entropies_jit(path[post_start:post_end])

                        # 各エントロピータイプの勾配
                        gradient = post_local - pre_local
                        entropy_gradients.append(gradient)

                if entropy_gradients:
                    mean_gradients = np.mean(entropy_gradients, axis=0)
                    for j, key in enumerate(entropy_keys):
                        entropy_dict[f"{key}_gradient"] = mean_gradients[j]

            # 5. 局所エントロピー変動性（ジャンプ周辺の不安定性）
            if len(jump_indices) > 0:
                local_variations = []
                for jump_idx in jump_indices:
                    # 各ジャンプ周辺での小窓エントロピー計算
                    for window_start in range(max(0, jump_idx - JUMP_WINDOW),
                                            min(len(path) - 2, jump_idx + JUMP_WINDOW)):
                        if window_start + 3 <= len(path):  # 最小3点で計算
                            local_window = path[window_start:window_start + 3]
                            local_ent = compute_all_entropies_jit(local_window)
                            local_variations.append(local_ent)

                if local_variations:
                    # 変動性を標準偏差で評価
                    variations_std = np.std(local_variations, axis=0)
                    for j, key in enumerate(entropy_keys):
                        entropy_dict[f"{key}_local_variation"] = variations_std[j]

            # 6. 遠隔エントロピー（ジャンプから離れた領域）
            if len(jump_vicinity_indices) > 0:
                all_indices = set(range(len(path)))
                remote_indices = sorted(all_indices - jump_vicinity_indices)

                if remote_indices:
                    remote_path = path[np.array(remote_indices)]
                    remote_entropies = compute_all_entropies_jit(remote_path)
                    for j, key in enumerate(entropy_keys):
                        entropy_dict[f"{key}_remote"] = remote_entropies[j]

                    # ジャンプ近傍と遠隔の比率（構造的な局所性の指標）
                    for j, key in enumerate(entropy_keys):
                        if f"{key}_vicinity" in entropy_dict and entropy_dict[f"{key}_remote"] != 0:
                            ratio = entropy_dict[f"{key}_vicinity"] / (entropy_dict[f"{key}_remote"] + 1e-10)
                            entropy_dict[f"{key}_locality_ratio"] = ratio

            entropies[i] = entropy_dict

        return entropies

    def _compute_jump_anomaly_scores(self,
                               jump_structures: Dict,
                               events: np.ndarray) -> np.ndarray:
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

    def _compute_kernel_anomaly_scores_optimized(self,
                                            events: np.ndarray,
                                            result: Lambda3Result) -> np.ndarray:
        """最適なカーネルを自動選択してカーネル空間での異常スコアを計算（周期カーネル追加版）"""

        # 周期推定（データから自動検出）
        estimated_periods = self._estimate_periods(events)

        # カーネルタイプとパラメータの候補
        kernel_configs = [
            {'type': 0, 'name': 'RBF', 'params': {'gamma': gamma}}
            for gamma in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
        ] + [
            {'type': 1, 'name': 'Polynomial', 'params': {'degree': d, 'coef0': c}}
            for d in [2, 3, 4, 5, 7] for c in [0.0, 0.5, 1.0, 2.0]
        ] + [
            {'type': 2, 'name': 'Sigmoid', 'params': {'alpha': a, 'coef0': 0.0}}
            for a in [0.001, 0.01, 0.1, 1.0]
        ] + [
            {'type': 3, 'name': 'Laplacian', 'params': {'gamma': gamma}}
            for gamma in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
        ] + [
            # 新規：周期カーネル（検出された周期に基づく）
            {'type': 4, 'name': 'Periodic', 'params': {'period': p, 'length_scale': ls}}
            for p in estimated_periods for ls in [0.5, 1.0, 2.0]
        ]

        paths_matrix = np.stack(list(result.paths.values()))
        n_events = events.shape[0]

        # サンプリングして計算量を削減（大規模データの場合）
        if n_events > 300:
            sample_idx = np.random.choice(n_events, 300, replace=False)
            events_sample = events[sample_idx]
            paths_sample = paths_matrix[:, sample_idx]
        else:
            events_sample = events
            paths_sample = paths_matrix
            sample_idx = np.arange(n_events)

        best_score = -np.inf
        best_config = None
        best_scores = None

        # 各カーネルで評価
        for config in kernel_configs:
            # カーネルGram行列の計算
            kernel_params = {
                'kernel_type': config['type'],
                'gamma': config['params'].get('gamma', 1.0),
                'degree': config['params'].get('degree', 3),
                'coef0': config['params'].get('coef0', 1.0),
                'alpha': config['params'].get('alpha', 0.01),
                'period': config['params'].get('period', 10.0),
                'length_scale': config['params'].get('length_scale', 1.0)
            }

            K = compute_kernel_gram_matrix(events_sample, **kernel_params)

            # カーネル空間での再構成
            n_sample = len(events_sample)
            K_recon = np.zeros((n_sample, n_sample))
            for i in range(n_sample):
                for j in range(n_sample):
                    for k in range(len(paths_sample)):
                        K_recon[i, j] += paths_sample[k, i] * K[i, j] * paths_sample[k, j]

            # 正規化
            K_norm = np.sqrt(np.trace(K @ K))
            if K_norm > 0:
                K /= K_norm

            recon_norm = np.sqrt(np.trace(K_recon @ K_recon))
            if recon_norm > 0:
                K_recon /= recon_norm

            # 再構成誤差の計算
            reconstruction_error = np.linalg.norm(K - K_recon, 'fro')

            # Lambda³理論の観点：再構成誤差が大きいほど、構造テンソルが
            # そのカーネル空間で異常を捉えやすい
            score = -reconstruction_error  # 負の誤差をスコアとする

            if score > best_score:
                best_score = score
                best_config = config

                # このカーネルでの異常スコアを計算
                kernel_scores = np.zeros(n_sample)
                for i in range(n_sample):
                    row_error = 0.0
                    for j in range(n_sample):
                        diff = K[i, j] - K_recon[i, j]
                        row_error += diff * diff
                    kernel_scores[i] = np.sqrt(row_error)

                # サンプリングした場合は全データに拡張
                if n_events > 300:
                    full_scores = np.zeros(n_events)
                    full_scores[sample_idx] = kernel_scores
                    # 残りは最近傍で補間
                    for i in range(n_events):
                        if i not in sample_idx:
                            # 最近傍のサンプル点を見つける
                            distances = np.sum((events_sample - events[i])**2, axis=1)
                            nearest_idx = np.argmin(distances)
                            full_scores[i] = kernel_scores[nearest_idx]
                    best_scores = full_scores
                else:
                    best_scores = kernel_scores

        print(f"Optimal kernel: {best_config['name']} with params {best_config['params']}")

        return best_scores

    def _estimate_periods(self, events: np.ndarray) -> List[float]:
        """データから周期を自動推定"""
        n_events = events.shape[0]
        periods = []

        # 各特徴量でFFT解析
        for i in range(events.shape[1]):
            fft = np.fft.fft(events[:, i])
            fft_abs = np.abs(fft[1:n_events//2])

            # 上位3つのピーク周波数を検出
            if len(fft_abs) > 3:
                peak_indices = np.argsort(fft_abs)[-3:]
                for idx in peak_indices:
                    if fft_abs[idx] > np.mean(fft_abs) * 2:
                        # 周波数から周期に変換
                        period = n_events / (idx + 1)
                        if 5 <= period <= n_events / 2:  # 妥当な周期範囲
                            periods.append(period)

        # 重複を除去して代表的な周期を選択
        if periods:
            unique_periods = []
            sorted_periods = sorted(set(periods))
            for p in sorted_periods:
                # 近い周期はグループ化
                if not any(abs(p - up) < 2 for up in unique_periods):
                    unique_periods.append(p)
            return unique_periods[:5]  # 最大5つの周期
        else:
            # デフォルト周期
            return [10.0, 20.0, 50.0]

    def _compute_kernel_anomaly_scores(self,
                                    events: np.ndarray,
                                    result: Lambda3Result,
                                    kernel_type: int = -1) -> np.ndarray:
        """カーネル空間での異常スコア計算（自動選択オプション付き）"""

        # kernel_type = -1 の場合は自動選択
        if kernel_type == -1:
            return self._compute_kernel_anomaly_scores_optimized(events, result)

        # 既存の実装（特定のカーネルを使用）
        K = compute_kernel_gram_matrix(events, kernel_type, gamma=1.0)

        # 以下、既存のコードと同じ...
        paths_matrix = np.stack(list(result.paths.values()))
        n_events = events.shape[0]

        K_recon = np.zeros((n_events, n_events))
        for i in range(n_events):
            for j in range(n_events):
                for k in range(len(paths_matrix)):
                    K_recon[i, j] += paths_matrix[k, i] * K[i, j] * paths_matrix[k, j]

        K_norm = np.sqrt(np.trace(K @ K))
        if K_norm > 0:
            K /= K_norm

        recon_norm = np.sqrt(np.trace(K_recon @ K_recon))
        if recon_norm > 0:
            K_recon /= recon_norm

        kernel_scores = np.zeros(n_events)
        for i in range(n_events):
            row_error = 0.0
            for j in range(n_events):
                diff = K[i, j] - K_recon[i, j]
                row_error += diff * diff
            kernel_scores[i] = np.sqrt(row_error)

        return kernel_scores

    #===============================
    # 異常パターン生成メソッド（地獄モード）
    # ===============================
    def _generate_pulse_anomaly(self, events: np.ndarray, intensity: float = 3, decay_rate: float = 0.5, n_pulses: int = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events, n_features = events.shape
        n_pulses_safe = min(n_pulses, n_events)
        pulse_indices = np.random.choice(n_events, size=n_pulses_safe, replace=False)
        for idx in pulse_indices:
            pulse = np.zeros(n_features)
            affected_dims = np.random.choice(n_features, size=np.random.randint(1, n_features//2 + 1), replace=False)
            pulse[affected_dims] = np.random.randn(len(affected_dims)) * intensity * np.random.uniform(0.8, 1.2)
            if np.random.rand() < 0.5:
                pulse *= -1
            events_copy[idx] += pulse
            for offset in range(1, 4):
                decay = np.exp(-decay_rate * offset)
                if idx - offset >= 0:
                    events_copy[idx - offset] += pulse * decay * np.random.uniform(0.5, 1.0)
                if idx + offset < n_events:
                    events_copy[idx + offset] += pulse * decay * np.random.uniform(0.5, 1.0)
        noise_mask = np.random.rand(*events_copy.shape) < 0.01
        events_copy[noise_mask] += np.random.normal(0, intensity/6, np.sum(noise_mask))
        return events_copy

    def _generate_phase_jump_anomaly(self, events: np.ndarray, intensity: float = 3, spread: int = 4) -> np.ndarray:
        events_copy = events.copy()
        n_events, n_features = events.shape
        idx = np.random.randint(n_events)
        events_copy[idx] = -np.sign(events_copy[idx]) * (np.abs(events_copy[idx]) ** np.random.uniform(1.2, 2.0)) * intensity
        events_copy[idx] += np.random.randn(n_features) * intensity * 0.3
        for offset in range(1, spread + 1):
            decay_factor = np.exp(-0.6 * offset)
            random_phase_shift = np.random.uniform(-np.pi, np.pi, n_features)
            modulation_factor = intensity * decay_factor * np.random.uniform(0.6, 1.2)
            if idx - offset >= 0:
                events_copy[idx - offset] += (
                    np.sin(events_copy[idx]) * modulation_factor
                    + np.cos(random_phase_shift) * modulation_factor * 0.5
                )
            if idx + offset < n_events:
                events_copy[idx + offset] += (
                    np.sin(events_copy[idx]) * modulation_factor
                    + np.cos(random_phase_shift) * modulation_factor * 0.5
                )
        distant_offset = spread + np.random.randint(1, 3)
        distant_decay = np.exp(-1.2 * distant_offset)
        distant_idx = idx + distant_offset if (idx + distant_offset < n_events) else idx - distant_offset
        if 0 <= distant_idx < n_events:
            events_copy[distant_idx] += np.random.randn(n_features) * intensity * distant_decay
        return events_copy

    def _generate_periodic_anomaly(self, events: np.ndarray, intensity: float = 2, disruption_prob=0.2) -> np.ndarray:
        events_copy = events.copy()
        n_events, n_features = events.shape
        t = np.arange(n_events)
        period = np.random.randint(max(3, n_events // 10), max(4, n_events // 4))
        phase = np.random.uniform(0, 2*np.pi)
        base_signal = intensity * np.sin(2 * np.pi * t / period + phase)
        for f in range(n_features):
            feat_phase = phase + np.random.uniform(-np.pi/5, np.pi/5)
            feat_signal = base_signal * np.random.uniform(0.7, 1.3)
            if np.random.rand() < disruption_prob:
                idx = np.random.randint(2, max(3, n_events-2)) if n_events > 3 else 0
                feat_signal[idx-1:idx+2] += np.random.uniform(2, 4) * intensity * np.random.choice([-1, 1])
            if np.random.rand() < disruption_prob and n_events >= 4:
                low = n_events // 4
                high = 3 * n_events // 4
                if high > low:
                    jump_idx = np.random.randint(low, high)
                    feat_signal[jump_idx:] *= -1
            if np.random.rand() < disruption_prob:
                del_start = np.random.randint(0, max(1, n_events - n_events // 8))
                del_end = del_start + np.random.randint(2, max(3, n_events // 8))
                del_end = min(n_events, del_end)
                feat_signal[del_start:del_end] = 0
            feat_signal += np.random.normal(0, 0.15 * intensity, n_events)
            events_copy[:, f] += feat_signal
        return events_copy

    def _generate_decay_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        decay_start = n_events // 2
        decay_length = n_events - decay_start
        decay = np.exp(-intensity * np.arange(decay_length) / decay_length)
        oscillation = np.sin(np.arange(decay_length) * 0.5) * 0.3
        decay_with_osc = decay * (1 + oscillation)
        n_features = events.shape[1]
        feature_decay_rates = np.random.uniform(0.5, 1.5, n_features)
        for i in range(decay_length):
            events_copy[decay_start + i] *= decay_with_osc[i]
            events_copy[decay_start + i] *= feature_decay_rates
            noise_level = (1 - decay[i]) * intensity * 0.5
            events_copy[decay_start + i] += np.random.normal(0, noise_level, n_features)
        n_spikes = max(1, decay_length // 10)
        spike_positions = np.random.choice(range(decay_start, n_events), n_spikes, replace=False)
        for pos in spike_positions:
            spike_features = np.random.choice(n_features, np.random.randint(1, max(2, n_features//3)), replace=False)
            events_copy[pos, spike_features] *= np.random.uniform(2, 4)
        if decay_length > 5:
            for i in range(decay_start + 2, n_events):
                correlation_loss = 1 - decay[i - decay_start]
                events_copy[i] = (1 - correlation_loss) * events_copy[i] + \
                                correlation_loss * np.random.randn(n_features) * np.std(events_copy[:decay_start])
        return events_copy

    def _generate_bifurcation_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        n_features = events.shape[1]
        split_point = n_events // 2
        post_split_length = n_events - split_point
        mode1 = np.random.randn(n_features) * intensity
        mode2 = -mode1 + np.random.randn(n_features) * intensity * 0.5
        for i in range(post_split_length):
            t = i / post_split_length
            bifurcation_strength = np.sqrt(t) * intensity
            if (split_point + i) % 2 == 0:
                events_copy[split_point + i] += mode1 * bifurcation_strength
                rotation_angle = t * np.pi / 4
                events_copy[split_point + i] = self._rotate_features(events_copy[split_point + i], rotation_angle)
            else:
                events_copy[split_point + i] += mode2 * bifurcation_strength
                rotation_angle = -t * np.pi / 4
                events_copy[split_point + i] = self._rotate_features(events_copy[split_point + i], rotation_angle)
        if split_point > 0 and split_point < n_events:
            events_copy[split_point] *= np.random.uniform(0.1, 0.5)
            events_copy[split_point] += np.random.randn(n_features) * intensity * 2
            chaos_range = min(5, split_point // 10)
            for j in range(max(0, split_point - chaos_range), min(n_events, split_point + chaos_range)):
                distance_from_split = abs(j - split_point)
                chaos_intensity = intensity * np.exp(-distance_from_split / chaos_range)
                events_copy[j] += np.random.randn(n_features) * chaos_intensity
        if post_split_length > 10:
            high_freq = np.random.uniform(0.3, 0.5)
            for i in range(split_point, n_events):
                phase = (i - split_point) * high_freq * 2 * np.pi
                amplitude = intensity * 0.3 * ((i - split_point) / post_split_length)
                events_copy[i] += np.sin(phase) * amplitude * np.random.randn(n_features)
        if n_features > 3:
            correlation_matrix = np.random.randn(n_features, n_features)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            correlation_matrix = np.exp(-np.abs(correlation_matrix))
            for i in range(split_point, n_events):
                events_copy[i] = correlation_matrix @ events_copy[i]
        return events_copy

    def _rotate_features(self, features: np.ndarray, angle: float) -> np.ndarray:
        if len(features) < 2:
            return features
        rotated = features.copy()
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        temp0 = rotated[0] * cos_a - rotated[1] * sin_a
        temp1 = rotated[0] * sin_a + rotated[1] * cos_a
        rotated[0], rotated[1] = temp0, temp1
        return rotated

    def _generate_multi_path_anomaly(self, events: np.ndarray, intensity: float = 2, interaction: float = 0.3) -> np.ndarray:
        events_copy = events.copy()
        n_events, n_features = events.shape
        n_available = n_events
        n_paths = min(np.random.randint(2, 5), n_available)
        path_indices = np.random.choice(n_available, size=n_paths, replace=False)
        paths_directions = np.random.randn(n_paths, n_features)
        paths_directions /= np.linalg.norm(paths_directions, axis=1, keepdims=True)
        base_intensities = intensity * np.random.uniform(0.8, 1.8, size=n_paths)
        polarities = np.random.choice([-1, 1], size=n_paths)
        for i, idx in enumerate(path_indices):
            pulse = paths_directions[i] * base_intensities[i] * polarities[i]
            events_copy[idx] += pulse * np.random.uniform(1.5, 2.5)
            for offset in range(1, 3):
                decay = np.exp(-0.5 * offset)
                spike_factor = (base_intensities[i] ** 2) * decay
                if idx - offset >= 0:
                    events_copy[idx - offset] += pulse * spike_factor * np.random.uniform(0.5, 1.2)
                if idx + offset < n_events:
                    events_copy[idx + offset] += pulse * spike_factor * np.random.uniform(0.5, 1.2)
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                midpoint = (path_indices[i] + path_indices[j]) // 2
                interaction_vector = (paths_directions[i] * paths_directions[j])
                interaction_strength = intensity * interaction * np.random.uniform(1.0, 2.0)
                if midpoint < n_events:
                    events_copy[midpoint] += interaction_vector * interaction_strength * np.random.choice([-1, 1])
                    for offset in range(1, 3):
                        interaction_decay = np.exp(-0.4 * offset)
                        if midpoint - offset >= 0:
                            events_copy[midpoint - offset] += interaction_vector * interaction_strength * interaction_decay
                        if midpoint + offset < n_events:
                            events_copy[midpoint + offset] += interaction_vector * interaction_strength * interaction_decay
        if np.random.rand() < 0.3:
            spike_idx = np.random.randint(0, n_events)
            spike_magnitude = intensity * np.random.uniform(3, 5)
            events_copy[spike_idx] += np.random.randn(n_features) * spike_magnitude
        return events_copy

    def _generate_partial_periodic_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        if n_events < 4:
            t = np.arange(n_events)
            period = max(2, n_events // 2)
            modulation = intensity * np.sin(2 * np.pi * t / period)
            events_copy += modulation[:, np.newaxis]
            return events_copy
        min_width = max(2, n_events // 6)
        max_width = max(min_width + 1, n_events // 2)
        width = np.random.randint(min_width, max_width)
        if width >= n_events:
            width = n_events - 1
        start = np.random.randint(0, n_events - width)
        end = start + width
        period = max(2, width // 3)
        t = np.arange(width)
        modulation = intensity * np.sin(2 * np.pi * t / period)
        events_copy[start:end] += modulation[:, np.newaxis]
        return events_copy

    def _generate_topological_jump_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        if n_events < 3:
            events_copy *= -intensity
            return events_copy
        min_point = max(1, n_events // 3)
        max_point = max(min_point + 1, 2 * n_events // 3)
        if min_point >= max_point:
            jump_point = n_events // 2
        else:
            jump_point = np.random.randint(min_point, max_point)
        if jump_point > 0:
            events_copy[:jump_point] *= np.exp(-0.1 * np.arange(jump_point))[:, np.newaxis]
        if jump_point < n_events:
            events_copy[jump_point:] = -events_copy[jump_point:] * intensity
        if jump_point < n_events:
            events_copy[jump_point] = np.random.randn(events.shape[1]) * intensity * 2
        return events_copy

    def _generate_superposition_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_patterns = np.random.randint(2, 4)
        base_patterns = ['pulse', 'periodic', 'phase_jump']
        patterns = np.random.choice(base_patterns, min(n_patterns, len(base_patterns)), replace=False)
        for pattern in patterns:
            weight = np.random.uniform(0.3, 0.7)
            if pattern in self.anomaly_patterns:
                events_copy = weight * events_copy + (1 - weight) * self.anomaly_patterns[pattern](events_copy, intensity)
        return events_copy

    def _generate_cascade_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        if n_events < 2:
            events_copy *= intensity
            return events_copy
        start_idx = np.random.randint(0, max(1, n_events // 2))
        events_copy[start_idx] += np.random.randn(events.shape[1]) * intensity
        for i in range(start_idx + 1, min(start_idx + 10, n_events)):
            decay = np.exp(-0.3 * (i - start_idx))
            events_copy[i] += events_copy[i-1] * 0.5 * decay
        return events_copy

    def _generate_resonance_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        if n_events < 4:
            events_copy *= intensity
            return events_copy
        fft = np.fft.fft(events_copy, axis=0)
        max_freq = max(2, len(fft) // 4)
        resonance_freq = np.random.randint(1, max_freq)
        if resonance_freq < len(fft):
            fft[resonance_freq] *= intensity
            if len(fft) - resonance_freq > 0:
                fft[-resonance_freq] *= intensity
        events_copy = np.real(np.fft.ifft(fft, axis=0))
        return events_copy

    def generate_anomalies(self, events: np.ndarray, pattern: str = 'pulse',
                          intensity: float = 3) -> np.ndarray:

        if pattern in self.anomaly_patterns:
            return self.anomaly_patterns[pattern](events, intensity)
        else:
            raise ValueError(f"Unknown anomaly pattern: {pattern}")

# ===============================
# 異常パターン生成メソッド（統合用）
# ===============================
def _init_anomaly_patterns(detector):
    """異常パターン生成関数の初期化"""
    return {
        'pulse': detector._generate_pulse_anomaly,
        'phase_jump': detector._generate_phase_jump_anomaly,
        'periodic': detector._generate_periodic_anomaly,
        'structural_decay': detector._generate_decay_anomaly,
        'bifurcation': detector._generate_bifurcation_anomaly,
        'multi_path': detector._generate_multi_path_anomaly,
        'topological_jump': detector._generate_topological_jump_anomaly,
        'cascade': detector._generate_cascade_anomaly,
        'partial_periodic': detector._generate_partial_periodic_anomaly,
        'superposition': detector._generate_superposition_anomaly,
        'resonance': detector._generate_resonance_anomaly
    }

# データセット生成関数
def create_complex_natural_dataset(n_events=200, n_features=20, anomaly_ratio=0.15):
    """より自然で複雑な異常を含むデータセットを生成"""

    # 異常パターン生成用の一時的な検出器
    temp_detector = Lambda3ZeroShotDetector()

    # 1. 基底構造の多様化
    normal_events = []

    # 複数の正常クラスターを生成（現実的な多様性）
    n_clusters = 3
    for i in range(n_clusters):
        cluster_size = (n_events - int(n_events * anomaly_ratio)) // n_clusters
        cov_matrix = np.eye(n_features)
        for j in range(n_features):
            for k in range(j+1, min(j+3, n_features)):
                corr = np.random.uniform(-0.7, 0.7)
                cov_matrix[j, k] = corr
                cov_matrix[k, j] = corr
        variances = np.random.uniform(0.5, 2.0, n_features)
        cov_matrix = cov_matrix * np.outer(np.sqrt(variances), np.sqrt(variances))
        cluster_mean = np.random.randn(n_features) * 2
        cluster_events = np.random.multivariate_normal(cluster_mean, cov_matrix, cluster_size)
        normal_events.append(cluster_events)

    normal_events = np.vstack(normal_events)

    # 2. 複雑な異常パターンの生成
    n_anomalies = int(n_events * anomaly_ratio)
    anomaly_events = []
    anomaly_labels_detailed = []

    # 異常パターンの組み合わせと時間発展
    anomaly_scenarios = [
        {
            'name': 'progressive_degradation',
            'patterns': ['structural_decay', 'cascade', 'topological_jump'],
            'progression': 'sequential',
            'intensity_profile': lambda t: 1 + 3 * t
        },
        {
            'name': 'periodic_burst',
            'patterns': ['periodic', 'pulse', 'resonance'],
            'progression': 'mixed',
            'intensity_profile': lambda t: 3 * (1 + np.sin(2 * np.pi * t))
        },
        {
            'name': 'chaotic_bifurcation',
            'patterns': ['bifurcation', 'multi_path', 'phase_jump'],
            'progression': 'simultaneous',
            'intensity_profile': lambda t: 2 * np.exp(t)
        },
        {
            'name': 'partial_anomaly',
            'patterns': ['partial_periodic', 'superposition'],
            'progression': 'feature_specific',
            'intensity_profile': lambda t: 2 + t
        }
    ]

    for i in range(n_anomalies):
        scenario = np.random.choice(anomaly_scenarios)
        base_idx = np.random.randint(len(normal_events))
        base_event = normal_events[base_idx].copy()
        temporal_position = i / n_anomalies

        if scenario['progression'] == 'sequential':
            anomaly = base_event.reshape(1, -1)
            for pattern in scenario['patterns']:
                intensity = scenario['intensity_profile'](temporal_position)
                if pattern in temp_detector.anomaly_patterns:
                    anomaly = temp_detector.anomaly_patterns[pattern](
                        anomaly, intensity * np.random.uniform(0.8, 1.2)
                    )

        elif scenario['progression'] == 'mixed':
            n_patterns = np.random.randint(1, len(scenario['patterns']) + 1)
            selected_patterns = np.random.choice(scenario['patterns'], n_patterns, replace=False)
            anomaly = base_event.reshape(1, -1)
            for pattern in selected_patterns:
                intensity = scenario['intensity_profile'](temporal_position)
                if pattern in temp_detector.anomaly_patterns:
                    anomaly = temp_detector.anomaly_patterns[pattern](
                        anomaly, intensity * np.random.uniform(0.5, 1.5)
                    )

        elif scenario['progression'] == 'simultaneous':
            anomalies = []
            for pattern in scenario['patterns']:
                intensity = scenario['intensity_profile'](temporal_position)
                if pattern in temp_detector.anomaly_patterns:
                    temp_anomaly = temp_detector.anomaly_patterns[pattern](
                        base_event.reshape(1, -1),
                        intensity * np.random.uniform(0.7, 1.3)
                    )
                    anomalies.append(temp_anomaly[0])
            if anomalies:
                weights = np.random.dirichlet(np.ones(len(anomalies)))
                anomaly = np.average(anomalies, axis=0, weights=weights).reshape(1, -1)
            else:
                anomaly = base_event.reshape(1, -1)

        else:  # feature_specific
            anomaly = base_event.reshape(1, -1)
            affected_features = np.random.choice(n_features,
                                               size=np.random.randint(1, n_features//2),
                                               replace=False)
            for pattern in scenario['patterns']:
                intensity = scenario['intensity_profile'](temporal_position)
                if pattern in temp_detector.anomaly_patterns:
                    temp_anomaly = temp_detector.anomaly_patterns[pattern](
                        anomaly, intensity
                    )
                    anomaly[0, affected_features] = temp_anomaly[0, affected_features]

        anomaly_events.append(anomaly[0])
        anomaly_labels_detailed.append(scenario['name'])

    # 3. ノイズと外れ値の追加
    anomaly_events = np.array(anomaly_events)
    noise_mask = np.random.random(anomaly_events.shape) < 0.1
    anomaly_events[noise_mask] += np.random.normal(0, 0.5, np.sum(noise_mask))

    outlier_positions = np.random.choice(len(anomaly_events),
                                       size=max(1, len(anomaly_events)//20),
                                       replace=False)
    for pos in outlier_positions:
        outlier_features = np.random.choice(n_features,
                                          size=np.random.randint(1, 3),
                                          replace=False)
        anomaly_events[pos, outlier_features] *= np.random.choice([-1, 1]) * np.random.uniform(5, 10)

    # 4. 最終的なデータセット構築
    events = np.vstack([normal_events, anomaly_events])
    labels = np.array([0]*len(normal_events) + [1]*len(anomaly_events))

    # 時系列的な相関を追加
    for i in range(1, len(events)):
        if np.random.random() < 0.3:
            events[i] = 0.7 * events[i] + 0.3 * events[i-1]

    # シャッフル
    block_size = 10
    n_blocks = len(events) // block_size
    block_indices = np.arange(n_blocks)
    np.random.shuffle(block_indices)

    shuffled_events = []
    shuffled_labels = []
    for block_idx in block_indices:
        start = block_idx * block_size
        end = min(start + block_size, len(events))
        shuffled_events.append(events[start:end])
        shuffled_labels.append(labels[start:end])

    if len(events) % block_size != 0:
        shuffled_events.append(events[n_blocks * block_size:])
        shuffled_labels.append(labels[n_blocks * block_size:])

    events = np.vstack(shuffled_events)
    labels = np.hstack(shuffled_labels)

    return events, labels, anomaly_labels_detailed

# ===============================
# ユーティリティ関数
# ===============================
def evaluate_performance(detector: Lambda3ZeroShotDetector,
                       events: np.ndarray,
                       labels: np.ndarray,
                       config: Dict[str, bool] = None) -> Dict[str, float]:
    """性能評価ユーティリティ"""

    if config is None:
        config = {
            "use_feature_optimization": True,
            "use_jump_analysis": False,
            "use_kernel_space": False,
            "use_ensemble": False
        }

    # Lambda³解析
    result = detector.analyze(events)

    # 異常検知
    scores = detector.detect_anomalies(result, events, **config)

    # AUC計算
    auc = roc_auc_score(labels, scores)

    # トップ10の精度
    top_10_indices = np.argsort(scores)[-10:]
    top_10_accuracy = np.mean(labels[top_10_indices])

    return {
        "auc": auc,
        "top_10_accuracy": top_10_accuracy,
        "config": config
    }

# ===============================
# MainDemo code
# ===============================
def demo_refactored_system():
    """リファクタリング版システムのデモ（適応的重み最適化を含む）"""
    np.random.seed(66)

    print("=== Lambda³ Zero-Shot Anomaly Detection System (Enhanced) ===")
    print("Unified Architecture with Adaptive Weight Optimization")
    print("=" * 60)

    # 1. データセット生成
    print("\n1. Generating complex dataset...")
    events, labels, anomaly_details = create_complex_natural_dataset(
        n_events=7500,
        n_features=20,
        anomaly_ratio=0.0003
    )
    print(f"Data shape: {events.shape}")
    print(f"Anomaly ratio: {np.mean(labels):.2%}")
    print(f"Anomaly types: {set(anomaly_details)}")

    # 2. 検出器の初期化
    print("\n2. Initializing detector...")
    config = L3Config()
    detector = Lambda3ZeroShotDetector(config)

    # 3. Lambda³解析
    print("\n3. Running Lambda³ analysis...")
    start_time = time.time()
    result = detector.analyze(events)
    analysis_time = time.time() - start_time
    print(f"Analysis completed in {analysis_time:.3f}s")

    # 4. 異なるモードでの異常検知
    print("\n4. Evaluating detection modes...")

    # 基本モード（デフォルト重み）
    print("\n--- Basic Mode (Default Weights) ---")
    start_time = time.time()
    basic_scores = detector.detect_anomalies(result, events, use_adaptive_weights=False)
    basic_time = time.time() - start_time
    basic_auc = roc_auc_score(labels, basic_scores)
    top_10_indices = np.argsort(basic_scores)[-10:]
    basic_top10 = np.mean(labels[top_10_indices])

    print(f"  AUC: {basic_auc:.4f}")
    print(f"  Top-10 Accuracy: {basic_top10:.2f}")
    print(f"  Detection Time: {basic_time:.3f}s")

    # 適応的重み最適化モード
    print("\n--- Adaptive Mode (Optimized Weights) ---")
    start_time = time.time()
    adaptive_scores = detector.detect_anomalies(result, events, use_adaptive_weights=True)
    adaptive_time = time.time() - start_time
    adaptive_auc = roc_auc_score(labels, adaptive_scores)
    top_10_indices = np.argsort(adaptive_scores)[-10:]
    adaptive_top10 = np.mean(labels[top_10_indices])

    print(f"  AUC: {adaptive_auc:.4f}")
    print(f"  Top-10 Accuracy: {adaptive_top10:.2f}")
    print(f"  Detection Time: {adaptive_time:.3f}s")

    # 6. 最良スコアでの詳細分析
    best_mode = "Basic"
    best_scores = basic_scores
    best_auc = basic_auc

    if adaptive_auc > best_auc:
        best_mode = "Adaptive"
        best_scores = adaptive_scores
        best_auc = adaptive_auc

    print(f"\n5. Best mode: {best_mode} (AUC: {best_auc:.4f})")

    # トップ異常の説明
    print("\nTop 5 anomalies with explanations:")
    top_5_indices = np.argsort(best_scores)[-5:]
    for i, idx in enumerate(top_5_indices[::-1]):
        explanation = detector.explain_anomaly(idx, result, events)
        print(f"\n{i+1}. Event {idx}:")
        print(f"   Score: {best_scores[idx]:.3f}")
        print(f"   True Label: {'Anomaly' if labels[idx] else 'Normal'}")
        print(f"   Recommendation: {explanation['recommendation']}")
        if explanation['jump_based']:
            print(f"   Jump Info: {explanation['jump_based']}")

    # 7. 可視化
    print("\n6. Generating visualizations...")
    fig = detector.visualize_results(events, result, best_scores)
    plt.suptitle(f'Lambda³ Zero-Shot Detection Results - {best_mode} Mode', fontsize=16)

    # 8. 最終サマリー
    print("\n" + "=" * 60)
    print("=== Final Summary ===")
    print(f"Dataset: {events.shape[0]} events, {events.shape[1]} features")
    print(f"Anomaly ratio: {np.mean(labels):.2%}")

    print("\nPerformance Summary:")
    print(f"  Basic Mode:    AUC={basic_auc:.4f}, Top-10={basic_top10:.2f}")
    print(f"  Adaptive Mode: AUC={adaptive_auc:.4f}, Top-10={adaptive_top10:.2f}")

    print(f"\nBest Performance: {best_mode} with {best_auc:.4f} AUC")

    if best_auc > 0.9:
        print(f"\n🚀 REVOLUTIONARY: {best_auc:.1%} AUC with ZERO training!")
        print("Lambda³ theory has achieved superhuman anomaly detection!")
    elif best_auc > 0.8:
        print(f"\n🎉 BREAKTHROUGH: {best_auc:.1%} AUC with ZERO training!")
        print("The unified architecture demonstrates exceptional performance!")
    elif best_auc > 0.7:
        print(f"\n✨ EXCELLENT: {best_auc:.1%} AUC with ZERO training!")
        print("Lambda³ theory shows remarkable zero-shot capabilities!")

    # ジャンプ統計の表示
    if hasattr(detector, 'jump_analyzer') and detector.jump_analyzer:
        print(f"\nJump Analysis Statistics:")
        print(f"  Total jumps detected: {detector.jump_analyzer['integrated']['n_total_jumps']}")
        print(f"  Jump clusters: {len(detector.jump_analyzer['integrated']['jump_clusters'])}")
        print(f"  Max synchronization: {detector.jump_analyzer['integrated']['max_sync']:.3f}")

    plt.show()

    return detector, result, {
        'basic': {'auc': basic_auc, 'top10': basic_top10, 'time': basic_time},
        'adaptive': {'auc': adaptive_auc, 'top10': adaptive_top10, 'time': adaptive_time}
    }

if __name__ == "__main__":
    detector, result, metrics = demo_refactored_system()
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print(f"Lambda³ Zero-Shot Detection System - Ready for deployment")
