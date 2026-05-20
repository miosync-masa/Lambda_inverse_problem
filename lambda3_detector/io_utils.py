"""
Lambda³結果の保存ユーティリティ。
"""

import json
import os
import pickle
from dataclasses import asdict
from typing import Dict

import numpy as np

from .config import Lambda3Result


def save_results(detector,
                 result: Lambda3Result,
                 anomaly_scores: np.ndarray,
                 events: np.ndarray,
                 channel_name: str,
                 save_dir: str = "./lambda3_results",
                 save_full_result: bool = False,
                 compress: bool = True) -> Dict[str, str]:
    """
    Lambda³解析結果をファイルに保存（jump_structures詳細版）
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
        'config': asdict(detector.config),
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
