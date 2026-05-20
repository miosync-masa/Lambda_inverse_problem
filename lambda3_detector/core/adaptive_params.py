"""
Adaptive window-size and parameter inference for Lambda³ analysis.

These functions read data and the optional :class:`Lambda3Result` to
recommend window sizes, percentiles, and regularizers. The "apply"
variant mutates the module-level constants in :mod:`lambda3_detector.config`.
"""

from typing import Dict, Optional

import numpy as np

from .. import config
from ..config import Lambda3Result, update_global_constants, update_detection_percentiles


def compute_adaptive_window_size(
    events: np.ndarray,
    base_window: int = 30,
    min_window: int = 10,
    max_window: int = None,
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
    # n_features=1 のとき corrcoef はスカラー (=1.0) になり triu_indices できない
    # → 単変量は「特徴量間相関」自体が未定義なので 0.0 (中立) を返す
    if n_features >= 2:
        correlation_matrix = np.corrcoef(events.T)
        correlation_complexity = 1.0 - np.mean(
            np.abs(correlation_matrix[np.triu_indices(n_features, k=1)])
        )
    else:
        correlation_complexity = 0.0

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


# 拡張版：Lambda³構造を考慮した動的調整
def compute_lambda3_adaptive_parameters(events: np.ndarray,
                                      result: Optional[Lambda3Result] = None) -> Dict[str, any]:
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


def apply_adaptive_parameters(detector,
                              events: np.ndarray,
                              result: Optional[Lambda3Result] = None):
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
    update_detection_percentiles(
        params['delta_percentile'],
        params['local_jump_percentile'],
        params['multiscale_percentiles'],
    )

    print(f"\nAdaptive parameters applied:")
    print(f"  Jump scale: {detector.config.jump_scale:.3f}")
    print(f"  Delta percentile: {config.DELTA_PERCENTILE:.1f}")
    print(f"  Multiscale percentiles: {config.MULTI_SCALE_PERCENTILES}")

    return params
