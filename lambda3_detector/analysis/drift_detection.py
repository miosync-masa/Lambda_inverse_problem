"""
CUSUM ベースのドリフト検出。

2 モード:
    'rolling' (default — change-point 検出向け):
        - ベースライン μ_t を直前 ref_window フレームから動的に再推定
        - 検出指標 = |z_t| = |(x_t - μ_t) / σ_t|
        - 累積 g_t = max(0, decay * g_{t-1} + |z_t| - k)
        - decay < 1 が old anomaly を忘却 → post-anomaly で減衰、stuck 回避
        - boundary でスパイク + 異常期間中も適度に高スコア

    'static' (legacy — 静的ベースライン想定の streaming 向け):
        - 全期間 median + MAD で baseline 固定
        - signed-deviation CUSUM + spread CUSUM (旧実装)
        - 異常区間が長いと accumulator が stuck

特徴間の集約は両モードとも単純平均。
"""

import numpy as np
from numba import njit, prange


# ===============================
# Static CUSUM (legacy)
# ===============================

@njit
def _cusum_two_sided_static(signal: np.ndarray, mu: float, k: float) -> np.ndarray:
    """Two-sided CUSUM, max(g+, g-) per t. State resets implicit via max(0, .)."""
    n = len(signal)
    out = np.zeros(n)
    g_pos = 0.0
    g_neg = 0.0
    for t in range(n):
        dev = signal[t] - mu
        g_pos = max(0.0, g_pos + dev - k)
        g_neg = max(0.0, g_neg - dev - k)
        out[t] = max(g_pos, g_neg)
    return out


@njit(parallel=True)
def _compute_drift_scores_static(events: np.ndarray, k_factor: float = 0.5) -> np.ndarray:
    """Legacy: 各特徴量で CUSUM (値) + CUSUM (絶対偏差) を取って平均。"""
    n_events, n_features = events.shape
    out = np.zeros(n_events)

    for f in prange(n_features):
        col = events[:, f]
        mu = np.median(col)
        mad = np.median(np.abs(col - mu)) + 1e-8
        sigma_est = 1.4826 * mad
        k = k_factor * sigma_est
        val_cusum = _cusum_two_sided_static(col, mu, k)

        spread = np.abs(col - mu)
        mu_s = np.median(spread)
        mad_s = np.median(np.abs(spread - mu_s)) + 1e-8
        k_s = k_factor * 1.4826 * mad_s
        spread_cusum = _cusum_two_sided_static(spread, mu_s, k_s)

        for t in range(n_events):
            out[t] += val_cusum[t] + spread_cusum[t]

    for t in range(n_events):
        out[t] /= n_features
    return out


# ===============================
# Rolling CUSUM (default — change-point向け)
# ===============================

@njit
def _rolling_z_cusum(signal: np.ndarray,
                     ref_window: int,
                     k: float,
                     decay: float) -> np.ndarray:
    """Rolling z-score CUSUM with decay.

    Each step:
        μ_t, σ_t = rolling mean / std over previous ref_window frames
        z_t = (signal[t] - μ_t) / σ_t
        g_t = max(0, decay * g_{t-1} + |z_t| - k)

    Returns: (n,) per-frame score
    """
    n = len(signal)
    out = np.zeros(n)
    if n <= ref_window:
        return out

    g = 0.0
    for t in range(ref_window, n):
        # rolling mean
        s = 0.0
        for i in range(t - ref_window, t):
            s += signal[i]
        ref_mean = s / ref_window

        # rolling std (population)
        v = 0.0
        for i in range(t - ref_window, t):
            d = signal[i] - ref_mean
            v += d * d
        ref_std = np.sqrt(v / ref_window) + 1e-8

        abs_z = np.abs(signal[t] - ref_mean) / ref_std
        g = max(0.0, decay * g + abs_z - k)
        out[t] = g

    return out


@njit(parallel=True)
def _compute_drift_scores_rolling(events: np.ndarray,
                                   ref_window: int = 50,
                                   k: float = 1.0,
                                   decay: float = 0.95) -> np.ndarray:
    """各特徴量で rolling-z CUSUM を取り、特徴間で平均。"""
    n_events, n_features = events.shape
    out = np.zeros(n_events)

    for f in prange(n_features):
        col = events[:, f]
        scores_f = _rolling_z_cusum(col, ref_window, k, decay)
        for t in range(n_events):
            out[t] += scores_f[t]

    for t in range(n_events):
        out[t] /= n_features
    return out


# ===============================
# 公開API
# ===============================

def compute_drift_scores(events: np.ndarray,
                          *,
                          mode: str = 'rolling',
                          ref_window: int = 50,
                          k: float = 1.0,
                          decay: float = 0.95,
                          k_factor: float = 0.5) -> np.ndarray:
    """ドリフトスコアを計算（モード切替可）。

    Args:
        events: (n_events, n_features)
        mode: 'rolling' (default) or 'static'

        Rolling params:
          ref_window: rolling baseline 窓幅（既定50）
          k: |z|閾値 — |z_t| - k が累積される（既定1.0、~1σ）
          decay: 累積の忘却率（既定0.95 → halflife≈14フレーム）

        Static params:
          k_factor: CUSUM slack の MAD 倍数（既定0.5）

    Returns:
        score: (n_events,) — 高いほど異常
    """
    ev = np.ascontiguousarray(events, dtype=np.float64)
    if mode == 'rolling':
        return _compute_drift_scores_rolling(ev, ref_window, k, decay)
    elif mode == 'static':
        return _compute_drift_scores_static(ev, k_factor)
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'rolling' or 'static'.")


def detect_drift(events: np.ndarray, **kwargs) -> dict:
    """Convenience wrapper. kwargs は compute_drift_scores へ委譲。

    Returns:
        {'cusum_score': (n_events,), 'mode': str, ...}
    """
    score = compute_drift_scores(events, **kwargs)
    return {'cusum_score': score, **kwargs}
