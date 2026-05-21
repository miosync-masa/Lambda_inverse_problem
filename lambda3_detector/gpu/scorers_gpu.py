"""
GPU 版 scorers（Step 1+2 では KernelScorer のみ）。

KernelScorer は CPU 版で O(n²) の Gram 行列 + O(n²) の再構成行列を構築する。
1.8k 行で 7.5 秒、10k 行では数百秒に達する次のボトルネック。

GPU では:
  - Gram 行列を一度の matmul (n×d × d×n) or 距離計算で構築
  - 再構成 K_recon = K * (paths.T @ paths) も element-wise / matmul
  - Frobenius 正規化と行ノルムは cp.sum / cp.sqrt で 1 行
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .backend import cp, DEFAULT_DTYPE, ensure_gpu, to_cpu


# =============================================================================
# Pairwise primitives (GPU)
# =============================================================================

def _pairwise_sq_dist_gpu(X: "cp.ndarray") -> "cp.ndarray":
    """||x_i - x_j||² の n×n 行列。X: (n, d) float32 → (n, n) float32。"""
    sq = cp.sum(X * X, axis=1)  # (n,)
    G = X @ X.T                  # (n, n)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    return cp.maximum(D2, 0.0)   # 数値誤差で負になり得るので clamp


def _pairwise_l1_dist_gpu(X: "cp.ndarray", chunk: int = 512) -> "cp.ndarray":
    """Σ_d |x_{id} - x_{jd}| の n×n 行列。

    (n, n, d) の中間配列を避けるため、行ブロック単位で計算。
    """
    n = X.shape[0]
    out = cp.empty((n, n), dtype=X.dtype)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # X[s:e, None, :] - X[None, :, :]  → (chunk, n, d)
        diff = cp.abs(X[s:e, None, :] - X[None, :, :])
        out[s:e] = cp.sum(diff, axis=-1)
    return out


# =============================================================================
# Kernel Gram matrix (GPU)
# =============================================================================

def compute_kernel_gram_matrix_gpu(
    data: np.ndarray,
    kernel_type: int = 0,
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
    alpha: float = 0.01,
    period: float = 10.0,
    length_scale: float = 1.0,
) -> "cp.ndarray":
    """CPU 版 compute_kernel_gram_matrix と数値的に等価な GPU 実装。

    注: Polynomial (kernel_type=1) は (G + coef0) ** degree を計算するが、
    events が大きい値 (NAB nyc_taxi の 1000 台、ec2_disk_write の 数百万) で
    degree=7 にすると float32 では確実に overflow → NaN。CPU 版 @njit は
    float64 デフォルトなのでこの問題は起きない。GPU でも float64 で計算し、
    最後に DEFAULT_DTYPE に落とす。
    """
    X32 = ensure_gpu(data)         # float32, 距離計算系で使う
    X64 = X32.astype(cp.float64)   # 多項式・Sigmoid 等の積系で overflow 防止
    n = X32.shape[0]

    if kernel_type == 0:  # RBF
        D2 = _pairwise_sq_dist_gpu(X32)
        K = cp.exp(-gamma * D2)
    elif kernel_type == 1:  # Polynomial — float64 で計算 (overflow 回避)
        G = X64 @ X64.T
        K = (G + coef0) ** degree
        # 入力 features の magnitude (NAB nyc_taxi で 30k 級) で K が
        # float32 max を簡単に超えるので、max-abs で正規化して [-1, 1] に。
        # 下流 (kernel_anomaly_scores_gpu) で Frobenius 正規化するため、
        # K の絶対スケールはスコアに影響しない (shape のみ重要)。
        K_absmax = float(cp.max(cp.abs(K)))
        if K_absmax > 1e-12:
            K = K / K_absmax
    elif kernel_type == 2:  # Sigmoid — alpha * <x,y> が大きいと tanh 飽和、念のため float64
        G = X64 @ X64.T
        K = cp.tanh(alpha * G + coef0)
    elif kernel_type == 3:  # Laplacian
        D1 = _pairwise_l1_dist_gpu(X32)
        K = cp.exp(-gamma * D1)
    elif kernel_type == 4:  # Periodic
        # periodic: exp(-2 Σ_d sin²(π|x_id-x_jd|/period) / length_scale²)
        n_total = X32.shape[0]
        K = cp.empty((n_total, n_total), dtype=X32.dtype)
        chunk = 512
        inv_period = float(np.pi / period)
        inv_ls2 = float(1.0 / (length_scale ** 2))
        for s in range(0, n_total, chunk):
            e = min(s + chunk, n_total)
            diff = cp.abs(X32[s:e, None, :] - X32[None, :, :])
            sin_term = cp.sin(inv_period * diff)
            K[s:e] = cp.exp(-2.0 * inv_ls2 * cp.sum(sin_term * sin_term, axis=-1))
    else:  # default: Laplacian
        D1 = _pairwise_l1_dist_gpu(X32)
        K = cp.exp(-gamma * D1)

    # 対称化（数値誤差吸収）
    K = 0.5 * (K + K.T)
    # float64 段階での非有限を 0 化
    K = cp.where(cp.isfinite(K), K, 0.0)
    # float32 へキャスト
    K = K.astype(DEFAULT_DTYPE)
    # float64→float32 キャストで巨大値が inf 化することがあるので再度クリーンアップ
    # (NAB nyc_taxi で polynomial degree=7 が float32 max 3.4e38 を超える)
    K = cp.where(cp.isfinite(K), K, 0.0)
    return K


# =============================================================================
# Kernel anomaly score (GPU)
# =============================================================================

def kernel_anomaly_scores_gpu(
    events: np.ndarray,
    paths_dict: dict,            # {i: ndarray(n_events,)} from Lambda3Result.paths
    kernel_type: int = 0,
    **kernel_params,
) -> np.ndarray:
    """CPU 版 compute_kernel_anomaly_scores_with_params と等価な GPU 実装。

    Returns:
        kernel_scores: (n_events,) float64 numpy
    """
    X = ensure_gpu(events)
    n_events = X.shape[0]

    K = compute_kernel_gram_matrix_gpu(events, kernel_type=kernel_type, **kernel_params)

    # paths_matrix (n_paths, n_events) on GPU
    paths_matrix = cp.asarray(
        np.stack(list(paths_dict.values())), dtype=DEFAULT_DTYPE,
    )

    # K_recon[i, j] = K[i, j] * Σ_k paths[k, i] * paths[k, j]
    P = paths_matrix.T @ paths_matrix   # (n_events, n_events)
    K_recon = K * P

    # Frobenius 正規化 (K は対称なので trace(K@K) = sum(K²))
    K_norm = cp.sqrt(cp.sum(K * K))
    if float(K_norm) > 0.0:
        K = K / K_norm
    recon_norm = cp.sqrt(cp.sum(K_recon * K_recon))
    if float(recon_norm) > 0.0:
        K_recon = K_recon / recon_norm

    diff = K - K_recon
    row_err = cp.sqrt(cp.sum(diff * diff, axis=1))   # (n_events,)
    return to_cpu(row_err).astype(np.float64)


# =============================================================================
# Auto-select kernel (CPU compute_kernel_anomaly_scores_optimized の GPU 版)
# =============================================================================

def _estimate_periods_gpu(events: np.ndarray, n_top: int = 3) -> list:
    """FFT で各特徴の dominant period を推定。元の estimate_periods と等価。

    n=22k 程度なら CPU FFT で十分速い (10ms 級)、GPU 化のメリット薄いのでそのまま numpy。
    """
    n_events = events.shape[0]
    periods = []
    for i in range(events.shape[1]):
        fft = np.fft.fft(events[:, i])
        fft_abs = np.abs(fft[1:n_events // 2])
        if len(fft_abs) > n_top:
            peak_indices = np.argsort(fft_abs)[-n_top:]
            for idx in peak_indices:
                if fft_abs[idx] > np.mean(fft_abs) * 2:
                    period = n_events / (idx + 1)
                    if 5 <= period <= n_events / 2:
                        periods.append(float(period))
    if not periods:
        return [10.0, 20.0, 50.0]
    # 近い周期をグループ化
    unique = []
    for p in sorted(set(periods)):
        if not any(abs(p - up) < 2 for up in unique):
            unique.append(p)
    return unique[:5]


def kernel_anomaly_scores_auto_gpu(
    events: np.ndarray,
    paths_dict: dict,
    n_sample: int = 300,
    seed: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """compute_kernel_anomaly_scores_optimized の GPU 等価実装。

    90+ kernel candidate (RBF / Polynomial / Sigmoid / Laplacian / Periodic) を
    sweep し、reconstruction error が最大のカーネルを採用 → そのカーネルでの
    per-event 行誤差を返す。auto-select の重さ (CPU で律速) を GPU に逃がす。

    n_events > n_sample のときはランダムサンプリング (CPU 版と同じ random 流儀)
    し、残りは最近傍補間で full-length に拡張。
    """
    n_events, n_dim = events.shape
    paths_matrix_full = np.stack(list(paths_dict.values()))  # (n_paths, n_events)

    rng = np.random.default_rng(seed)
    if n_events > n_sample:
        sample_idx = np.sort(rng.choice(n_events, n_sample, replace=False))
    else:
        sample_idx = np.arange(n_events)
    events_sample = events[sample_idx]
    paths_sample = paths_matrix_full[:, sample_idx]
    n_s = len(sample_idx)

    # 周期推定 (CPU FFT)
    estimated_periods = _estimate_periods_gpu(events_sample)

    # kernel configs (CPU 版と同じ)
    kernel_configs = (
        [{'type': 0, 'params': {'gamma': g}}
         for g in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]]
        + [{'type': 1, 'params': {'degree': d, 'coef0': c}}
           for d in [2, 3, 4, 5, 7] for c in [0.0, 0.5, 1.0, 2.0]]
        + [{'type': 2, 'params': {'alpha': a, 'coef0': 0.0}}
           for a in [0.001, 0.01, 0.1, 1.0]]
        + [{'type': 3, 'params': {'gamma': g}}
           for g in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]]
        + [{'type': 4, 'params': {'period': p, 'length_scale': ls}}
           for p in estimated_periods for ls in [0.5, 1.0, 2.0]]
    )

    # GPU 常駐: events_sample と paths_sample から P を事前計算 (loop 内で共有)
    paths_sample_gpu = cp.asarray(paths_sample, dtype=DEFAULT_DTYPE)
    P = paths_sample_gpu.T @ paths_sample_gpu  # (n_s, n_s)

    best_score = -float('inf')
    best_K = None
    best_K_recon = None
    best_cfg = None

    for cfg in kernel_configs:
        K = compute_kernel_gram_matrix_gpu(
            events_sample, kernel_type=cfg['type'], **cfg['params'],
        )
        K_recon = K * P

        K_norm = cp.sqrt(cp.sum(K * K))
        K_n = K / K_norm if float(K_norm) > 0.0 else K
        recon_norm = cp.sqrt(cp.sum(K_recon * K_recon))
        K_r = K_recon / recon_norm if float(recon_norm) > 0.0 else K_recon

        # Frobenius 距離 = ||K - K_recon||_F (正規化済み)
        recon_error = float(cp.sqrt(cp.sum((K_n - K_r) * (K_n - K_r))))
        score = -recon_error  # CPU 版と同じ符号慣例 (大きい = 構造が遠い = カーネル選好)

        if score > best_score:
            best_score = score
            best_K = K_n
            best_K_recon = K_r
            best_cfg = cfg

    if verbose:
        print(f"  [GPU auto-kernel] best: type={best_cfg['type']} "
              f"params={best_cfg['params']}  recon_err={-best_score:.4f}")

    # 採用カーネルでの per-sample 行誤差
    diff = best_K - best_K_recon
    row_err_sample = cp.sqrt(cp.sum(diff * diff, axis=1))  # (n_s,)
    sample_scores = to_cpu(row_err_sample).astype(np.float64)

    # フル長へ拡張 (subsampled 時のみ最近傍補間)
    if n_events > n_sample:
        # 各 i について sample_idx 内の最近傍 (event 空間 L2)
        X_full = events
        X_s = events_sample
        # 大きい n_events だと O(n * n_s * d) になるが、n_s=300 なので n=22k でも 22k*300*5 = 33M flops, GPU で 1ms 級
        Xf_g = cp.asarray(X_full, dtype=DEFAULT_DTYPE)
        Xs_g = cp.asarray(X_s, dtype=DEFAULT_DTYPE)
        # dist^2 = ||a||^2 + ||b||^2 - 2 a·b
        af = cp.sum(Xf_g * Xf_g, axis=1)[:, None]  # (n_events, 1)
        bs = cp.sum(Xs_g * Xs_g, axis=1)[None, :]  # (1, n_s)
        cross = Xf_g @ Xs_g.T
        d2 = af + bs - 2.0 * cross  # (n_events, n_s)
        nearest = to_cpu(cp.argmin(d2, axis=1))
        full_scores = sample_scores[nearest]
    else:
        full_scores = sample_scores

    return full_scores
