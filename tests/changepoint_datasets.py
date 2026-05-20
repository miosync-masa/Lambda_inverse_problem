"""
Change-point detection 用データセット生成器。

産業利用シナリオ:
    [N_pre | A_region | N_post]
    └ Normal ┘└ Anomaly ┘└ Normal (recovery) ┘

時間順序は **絶対に保存**。block shuffle はしない。

サポートする異常シナリオ:
    - progressive_degradation : 指数減衰 + 振動 + 特徴ごとの異なる減衰率
    - periodic_burst          : 周期的modulation（振幅大）
    - chaotic_bifurcation     : 分岐後の双方向drift + 高周波modulation
    - partial_anomaly         : 一部特徴のみへの周期modulation

Returns:
    events  : (n_total, n_features)
    labels  : (n_total,)  — anomaly_region のみ 1
    info    : {'true_start', 'true_end', 'scenario', 'change_point'}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


SCENARIOS = (
    'progressive_degradation',
    'periodic_burst',
    'chaotic_bifurcation',
    'partial_anomaly',
)


@dataclass
class ChangePointInfo:
    true_start: int           # anomaly_region 開始 index (= n_normal_pre)
    true_end: int             # anomaly_region 終了 index (= n_normal_pre + n_anomaly, exclusive)
    n_normal_pre: int
    n_anomaly: int
    n_normal_post: int
    scenario: str

    @property
    def change_point(self) -> int:
        return self.true_start


# ===============================
# 正常区間生成
# ===============================

def _generate_normal_region(n: int, n_features: int, n_clusters: int = 3) -> np.ndarray:
    """複数クラスタを持つ正常イベント（既存 dataset と同様の生成則）"""
    chunks = []
    base = n // n_clusters
    remainder = n - base * n_clusters

    for i in range(n_clusters):
        size = base + (1 if i < remainder else 0)
        cov_matrix = np.eye(n_features)
        for j in range(n_features):
            for k in range(j + 1, min(j + 3, n_features)):
                corr = np.random.uniform(-0.7, 0.7)
                cov_matrix[j, k] = corr
                cov_matrix[k, j] = corr
        variances = np.random.uniform(0.5, 2.0, n_features)
        cov_matrix = cov_matrix * np.outer(np.sqrt(variances), np.sqrt(variances))
        cluster_mean = np.random.randn(n_features) * 2
        chunks.append(np.random.multivariate_normal(cluster_mean, cov_matrix, size))

    return np.vstack(chunks)


# ===============================
# 区間ベース異常パターン（時間順序保存）
# ===============================

def _apply_progressive_degradation(events: np.ndarray, start: int, end: int,
                                    intensity: float = 2.0) -> None:
    """[start, end) に対して累積的な構造崩壊を印加（in-place）。

    特徴:
      - 指数減衰の amplitude scaling
      - 高周波振動の重畳
      - 特徴ごとの異なる減衰率
      - 末尾でランダムスパイク
      - 正常との相関が時間とともに失われる（detail）
    """
    region = events[start:end]
    n_region = end - start
    n_features = events.shape[1]

    decay = np.exp(-intensity * np.arange(n_region) / max(n_region, 1))
    oscillation = np.sin(np.arange(n_region) * 0.5) * 0.3
    decay_with_osc = decay * (1 + oscillation)
    feature_decay_rates = np.random.uniform(0.5, 1.5, n_features)

    for i in range(n_region):
        region[i] *= decay_with_osc[i]
        region[i] *= feature_decay_rates
        noise_level = (1 - decay[i]) * intensity * 0.5
        region[i] += np.random.normal(0, noise_level, n_features)

    # スパイク
    n_spikes = max(1, n_region // 10)
    spike_positions = np.random.choice(n_region, n_spikes, replace=False)
    for pos in spike_positions:
        spike_features = np.random.choice(
            n_features, np.random.randint(1, max(2, n_features // 3)), replace=False,
        )
        region[pos, spike_features] *= np.random.uniform(2, 4)


def _apply_periodic_burst(events: np.ndarray, start: int, end: int,
                          intensity: float = 2.0,
                          disruption_prob: float = 0.2) -> None:
    """[start, end) で持続的な周期 burst（振幅増大・周期破綻あり）。"""
    region = events[start:end]
    n_region = end - start
    n_features = events.shape[1]

    t = np.arange(n_region)
    period = np.random.randint(max(3, n_region // 10), max(4, n_region // 4))
    phase = np.random.uniform(0, 2 * np.pi)
    base_signal = intensity * np.sin(2 * np.pi * t / period + phase)

    for f in range(n_features):
        feat_signal = base_signal * np.random.uniform(0.7, 1.3)
        # disrupt: 振幅スパイク
        if np.random.rand() < disruption_prob and n_region > 3:
            idx = np.random.randint(2, max(3, n_region - 2))
            feat_signal[idx-1:idx+2] += (
                np.random.uniform(2, 4) * intensity * np.random.choice([-1, 1])
            )
        # disrupt: 位相反転
        if np.random.rand() < disruption_prob and n_region >= 4:
            low, high = n_region // 4, 3 * n_region // 4
            if high > low:
                jump_idx = np.random.randint(low, high)
                feat_signal[jump_idx:] *= -1
        feat_signal += np.random.normal(0, 0.15 * intensity, n_region)
        region[:, f] += feat_signal


def _apply_chaotic_bifurcation(events: np.ndarray, start: int, end: int,
                                intensity: float = 2.0) -> None:
    """[start, end) で分岐後の双方向drift + 高周波noise。"""
    region = events[start:end]
    n_region = end - start
    n_features = events.shape[1]

    split_point = n_region // 2
    post_split_length = n_region - split_point
    mode1 = np.random.randn(n_features) * intensity
    mode2 = -mode1 + np.random.randn(n_features) * intensity * 0.5

    for i in range(post_split_length):
        t = i / max(post_split_length, 1)
        bifurcation_strength = np.sqrt(t) * intensity
        idx_in_region = split_point + i
        if idx_in_region % 2 == 0:
            region[idx_in_region] += mode1 * bifurcation_strength
        else:
            region[idx_in_region] += mode2 * bifurcation_strength

    # 分岐点で大きく擾乱
    if 0 < split_point < n_region:
        region[split_point] *= np.random.uniform(0.1, 0.5)
        region[split_point] += np.random.randn(n_features) * intensity * 2

    # 高周波 modulation
    if post_split_length > 10:
        high_freq = np.random.uniform(0.3, 0.5)
        for i in range(split_point, n_region):
            phase = (i - split_point) * high_freq * 2 * np.pi
            amplitude = intensity * 0.3 * ((i - split_point) / post_split_length)
            region[i] += np.sin(phase) * amplitude * np.random.randn(n_features)


def _apply_partial_anomaly(events: np.ndarray, start: int, end: int,
                            intensity: float = 2.0,
                            affected_fraction: float = 0.3) -> None:
    """[start, end) の **一部特徴のみ** に周期modulationを印加。"""
    region = events[start:end]
    n_region = end - start
    n_features = events.shape[1]

    period = max(2, n_region // 4)
    t = np.arange(n_region)
    modulation = intensity * np.sin(2 * np.pi * t / period)

    n_affected = max(1, int(n_features * affected_fraction))
    affected_features = np.random.choice(n_features, n_affected, replace=False)
    for f in affected_features:
        region[:, f] += modulation * np.random.uniform(0.8, 1.2)


_PATTERNS = {
    'progressive_degradation': _apply_progressive_degradation,
    'periodic_burst':          _apply_periodic_burst,
    'chaotic_bifurcation':     _apply_chaotic_bifurcation,
    'partial_anomaly':         _apply_partial_anomaly,
}


# ===============================
# 公開API
# ===============================

def create_changepoint_dataset(
    n_normal_pre: int = 400,
    n_anomaly: int = 100,
    n_normal_post: int = 400,
    n_features: int = 20,
    scenario: str = 'progressive_degradation',
    intensity: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, ChangePointInfo]:
    """[N_pre | A_region | N_post] の change-point dataset を生成。

    Args:
        n_normal_pre   : 異常前の正常区間長
        n_anomaly      : 異常区間長
        n_normal_post  : 異常後の正常区間長
        n_features     : 特徴次元
        scenario       : 4種から選択
        intensity      : 異常強度

    Returns:
        events  : (n_total, n_features)
        labels  : (n_total,)
        info    : ChangePointInfo
    """
    if scenario not in _PATTERNS:
        raise ValueError(
            f"Unknown scenario={scenario!r}. Valid: {list(_PATTERNS.keys())}"
        )

    n_total = n_normal_pre + n_anomaly + n_normal_post

    # 1. 全部正常イベントとして生成（時系列順序保持のため一括）
    events = _generate_normal_region(n_total, n_features)

    # 2. 弱い時系列相関を付与（産業データらしさ）
    for i in range(1, n_total):
        if np.random.rand() < 0.3:
            events[i] = 0.7 * events[i] + 0.3 * events[i - 1]

    # 3. anomaly_region に scenario を印加
    true_start = n_normal_pre
    true_end = n_normal_pre + n_anomaly
    _PATTERNS[scenario](events, true_start, true_end, intensity=intensity)

    # 4. labels
    labels = np.zeros(n_total, dtype=np.int64)
    labels[true_start:true_end] = 1

    info = ChangePointInfo(
        true_start=true_start,
        true_end=true_end,
        n_normal_pre=n_normal_pre,
        n_anomaly=n_anomaly,
        n_normal_post=n_normal_post,
        scenario=scenario,
    )
    return events, labels, info
