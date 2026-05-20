"""
NAB の単変量シリーズを Lambda3 detector 用に多変量化する特徴量エンジニアリング。

5 次元構成 (window=30 デフォルト):
    f0: raw            x[t]
    f1: rolling_mean   過去 w 点の平均
    f2: rolling_std    過去 w 点の標準偏差
    f3: 2nd-order diff x[t] - 2 x[t-1] + x[t-2]
    f4: lag1 autocorr  過去 w 点での x[t] と x[t-1] の Pearson 相関

境界処理:
    - rolling_mean: min_periods=1 (常に値あり)
    - rolling_std / lag1 autocorr: min_periods=2、それ以下は 0
    - diff2: 先頭 2 点は 0
    - 非有限 (NaN/Inf) は 0 にフォールバック

multi-channel detector の sync / corr 系 scorer に「raw と派生量の構造的関係」を
見せることで、単変量 NaN 退化を避けつつ scorer 全体を活用するのが目的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expand_to_5d(values: np.ndarray, window: int = 30) -> np.ndarray:
    """単変量 (n,1) or (n,) → 多変量 (n, 5)。

    Returns:
        (n, 5) float64 配列。列順は [raw, rmean, rstd, diff2, lag1_autocorr]。
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 2:
        raise ValueError(f"need n>=2, got n={n}")
    if window < 2:
        raise ValueError(f"window must be >=2, got {window}")

    s = pd.Series(x)
    s_lag = s.shift(1)

    f0 = x
    f1 = s.rolling(window, min_periods=1).mean().to_numpy()
    f2 = s.rolling(window, min_periods=2).std().to_numpy()

    f3 = np.zeros(n, dtype=np.float64)
    if n >= 3:
        f3[2:] = x[2:] - 2.0 * x[1:-1] + x[:-2]

    f4 = s.rolling(window, min_periods=2).corr(s_lag).to_numpy()

    # 非有限を 0 で埋めて clip
    for arr in (f1, f2, f3, f4):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(f4, -1.0, 1.0, out=f4)

    return np.column_stack([f0, f1, f2, f3, f4])
