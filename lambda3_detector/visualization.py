"""
Lambda³解析結果の可視化（ジャンプ構造を中心とした統合プロット）。
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .config import Lambda3Result
from .core.kernels_jit import compute_kernel_gram_matrix
from .core.pulsation_jit import compute_pulsation_energy_from_path


def visualize_results(detector,
                      events: np.ndarray,
                      result: Lambda3Result,
                      anomaly_scores: np.ndarray = None) -> plt.Figure:
    """統合的な可視化（ジャンプ構造を中心に）"""
    if anomaly_scores is None:
        anomaly_scores = detector.detect_anomalies(result, events)

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
