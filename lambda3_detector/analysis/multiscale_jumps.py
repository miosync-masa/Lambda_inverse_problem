"""
マルチスケール・多次元ジャンプ検出パイプライン。

特徴次元ごとにジャンプ検出 → 同期マトリックス計算 → クラスター抽出までを
スタンドアロン関数として提供（detectorクラスから委譲される）。
"""

from typing import Dict, List

import numpy as np

from .. import config
from ..core.adaptive_params import compute_adaptive_window_size
from ..core.jumps_jit import (
    calculate_diff_and_threshold,
    calculate_local_std,
    calculate_rho_t,
    calculate_sync_profile_jit,
    detect_jumps,
)
from ..core.pulsation_jit import compute_pulsation_energy_from_jumps


def detect_multiscale_jumps(events: np.ndarray) -> Dict:
    """多次元・多スケールジャンプ検出（デフォルトのglobal定数を使用）"""
    n_events, n_features = events.shape
    jump_data = {'features': {}, 'integrated': {}}

    # 各特徴次元でのジャンプ検出
    for f in range(n_features):
        data = events[:, f]

        # ここで配列サイズを確認
        assert len(data) == n_events, f"Feature {f} has wrong size: {len(data)} vs {n_events}"

        # 基本ジャンプ検出
        diff, threshold = calculate_diff_and_threshold(data, config.DELTA_PERCENTILE)
        pos_jumps, neg_jumps = detect_jumps(diff, threshold)

        # 局所適応的ジャンプ
        local_std = calculate_local_std(data, config.LOCAL_WINDOW_SIZE)
        score = np.abs(diff) / (local_std + 1e-8)
        local_threshold = np.percentile(score, config.LOCAL_JUMP_PERCENTILE)
        local_jumps = (score > local_threshold).astype(int)

        # テンションスカラー
        rho_t = calculate_rho_t(data, config.WINDOW_SIZE)

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
    jump_data['integrated'] = integrate_cross_feature_jumps(jump_data['features'])

    return jump_data


def detect_multiscale_jumps_with_params(events: np.ndarray,
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
    jump_data['integrated'] = integrate_cross_feature_jumps(jump_data['features'])

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


def integrate_cross_feature_jumps(feature_jumps: Dict) -> Dict:
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
    jump_clusters = detect_jump_clusters(unified_jumps, jump_importance)

    # n_features=1 だと triu_indices(1, k=1) が空配列 → max が空に失敗
    if n_features >= 2:
        max_sync_val = float(np.max(sync_matrix[np.triu_indices(n_features, k=1)]))
    else:
        max_sync_val = 0.0  # 単変量では「特徴量間 sync」は定義されない

    return {
        'unified_jumps': unified_jumps,
        'jump_importance': jump_importance / n_features,  # 正規化
        'sync_matrix': sync_matrix,
        'jump_clusters': jump_clusters,
        'n_total_jumps': np.sum(unified_jumps),
        'max_sync': max_sync_val,
    }


def detect_jump_clusters(unified_jumps: np.ndarray,
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
