"""
KernelScorer: カーネル空間（RBF / Polynomial / Sigmoid / Laplacian / Periodic）
での再構成誤差ベース異常スコア。
"""

from typing import List

import numpy as np

from ..config import Lambda3Result
from ..core.kernels_jit import compute_kernel_gram_matrix
from .base import AnomalyScorer


def compute_kernel_anomaly_scores_with_params(events: np.ndarray,
                                              lambda3_result: Lambda3Result,
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

    paths_matrix = np.stack(list(lambda3_result.paths.values()))
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


def estimate_periods(events: np.ndarray) -> List[float]:
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


def compute_kernel_anomaly_scores_optimized(events: np.ndarray,
                                            lambda3_result: Lambda3Result) -> np.ndarray:
    """最適なカーネルを自動選択してカーネル空間での異常スコアを計算（周期カーネル追加版）"""

    # 周期推定（データから自動検出）
    estimated_periods = estimate_periods(events)

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

    paths_matrix = np.stack(list(lambda3_result.paths.values()))
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
    for cfg in kernel_configs:
        # カーネルGram行列の計算
        kernel_params = {
            'kernel_type': cfg['type'],
            'gamma': cfg['params'].get('gamma', 1.0),
            'degree': cfg['params'].get('degree', 3),
            'coef0': cfg['params'].get('coef0', 1.0),
            'alpha': cfg['params'].get('alpha', 0.01),
            'period': cfg['params'].get('period', 10.0),
            'length_scale': cfg['params'].get('length_scale', 1.0)
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
            best_config = cfg

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


def compute_kernel_anomaly_scores(events: np.ndarray,
                                  lambda3_result: Lambda3Result,
                                  kernel_type: int = -1) -> np.ndarray:
    """カーネル空間での異常スコア計算（自動選択オプション付き）"""

    # kernel_type = -1 の場合は自動選択
    if kernel_type == -1:
        return compute_kernel_anomaly_scores_optimized(events, lambda3_result)

    # 既存の実装（特定のカーネルを使用）
    K = compute_kernel_gram_matrix(events, kernel_type, gamma=1.0)

    # 以下、既存のコードと同じ...
    paths_matrix = np.stack(list(lambda3_result.paths.values()))
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


class KernelScorer(AnomalyScorer):
    """カーネル空間での再構成誤差スコア。

    ``kernel_type=-1`` で全カーネル探索（自動選択）、それ以外は固定の
    カーネル種別＋パラメータを使う。

    ``use_gpu=True`` で固定カーネル経路（kernel_type != -1）も自動選択経路
    （kernel_type == -1）も CuPy 版にディスパッチする。
    """

    def __init__(self, kernel_type: int = -1, use_gpu: bool = False, **kernel_params):
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.use_gpu = use_gpu

    def score(self, events: np.ndarray, lambda3_result: Lambda3Result) -> np.ndarray:
        if self.kernel_type == -1:
            if self.use_gpu:
                from ..gpu import kernel_anomaly_scores_auto_gpu
                return kernel_anomaly_scores_auto_gpu(
                    events, lambda3_result.paths,
                )
            return compute_kernel_anomaly_scores_optimized(events, lambda3_result)
        if self.use_gpu:
            from ..gpu import kernel_anomaly_scores_gpu
            return kernel_anomaly_scores_gpu(
                events, lambda3_result.paths,
                kernel_type=self.kernel_type, **self.kernel_params,
            )
        if self.kernel_params:
            return compute_kernel_anomaly_scores_with_params(
                events, lambda3_result, self.kernel_type, **self.kernel_params,
            )
        return compute_kernel_anomaly_scores(events, lambda3_result, self.kernel_type)
