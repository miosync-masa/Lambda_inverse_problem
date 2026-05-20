"""
複雑な合成データセット生成（``create_complex_natural_dataset``）。

本番detectorからは切り離されており、ベンチマーク／ablation／同値性テスト
からのみ使用する。``Lambda3ZeroShotDetector`` への一時的なpatterns
attachmentは :mod:`tests.anomaly_generators` 経由で行う。
"""

import numpy as np

from .anomaly_generators import build_anomaly_patterns


def create_complex_natural_dataset(n_events=200, n_features=20, anomaly_ratio=0.15,
                                    scenario_filter: str = None):
    """より自然で複雑な異常を含むデータセットを生成

    Args:
        scenario_filter: シナリオ名を指定すると、その1種類だけで全異常を生成する。
            None なら従来通り4種からランダム選択。
            有効値: 'progressive_degradation', 'periodic_burst',
                    'chaotic_bifurcation', 'partial_anomaly'
    """

    # 異常パターン辞書（detectorに依存しないスタンドアロン版）
    patterns = build_anomaly_patterns()

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

    # scenario_filter で1種類に絞る
    if scenario_filter is not None:
        filtered = [s for s in anomaly_scenarios if s['name'] == scenario_filter]
        if not filtered:
            raise ValueError(
                f"Unknown scenario_filter={scenario_filter!r}. "
                f"Valid: {[s['name'] for s in anomaly_scenarios]}"
            )
        anomaly_scenarios = filtered

    for i in range(n_anomalies):
        scenario = np.random.choice(anomaly_scenarios)
        base_idx = np.random.randint(len(normal_events))
        base_event = normal_events[base_idx].copy()
        temporal_position = i / n_anomalies

        if scenario['progression'] == 'sequential':
            anomaly = base_event.reshape(1, -1)
            for pattern in scenario['patterns']:
                intensity = scenario['intensity_profile'](temporal_position)
                if pattern in patterns:
                    anomaly = patterns[pattern](
                        anomaly, intensity * np.random.uniform(0.8, 1.2)
                    )

        elif scenario['progression'] == 'mixed':
            n_patterns = np.random.randint(1, len(scenario['patterns']) + 1)
            selected_patterns = np.random.choice(scenario['patterns'], n_patterns, replace=False)
            anomaly = base_event.reshape(1, -1)
            for pattern in selected_patterns:
                intensity = scenario['intensity_profile'](temporal_position)
                if pattern in patterns:
                    anomaly = patterns[pattern](
                        anomaly, intensity * np.random.uniform(0.5, 1.5)
                    )

        elif scenario['progression'] == 'simultaneous':
            anomalies = []
            for pattern in scenario['patterns']:
                intensity = scenario['intensity_profile'](temporal_position)
                if pattern in patterns:
                    temp_anomaly = patterns[pattern](
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
                if pattern in patterns:
                    temp_anomaly = patterns[pattern](
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
