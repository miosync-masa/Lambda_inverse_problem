"""
CuPy backend 初期化と共通ユーティリティ。

CPU フォールバックは無い (Colab/GPU 専用前提)。CuPy が無い環境では import 時に例外。
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except ImportError as _err:  # noqa: F841
    raise ImportError(
        "lambda3_detector.gpu requires CuPy (GPU only, no CPU fallback).\n"
        "Install via: pip install cupy-cuda12x  (Colab default)"
    )


DEFAULT_DTYPE = cp.float32


def ensure_gpu(x: Any, dtype=None) -> "cp.ndarray":
    """ndarray を GPU に乗せ、dtype を統一する。既に cp.ndarray なら dtype のみ確認。"""
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if isinstance(x, cp.ndarray):
        if x.dtype == dtype:
            return x
        return x.astype(dtype)
    return cp.asarray(x, dtype=dtype)


def to_cpu(x: "cp.ndarray") -> np.ndarray:
    """GPU array → numpy float64 (host)。scipy.optimize 等への受け渡し用。"""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def device_info() -> dict:
    """現在使っている GPU の概要を返す（ログ用）。"""
    dev = cp.cuda.Device()
    free, total = dev.mem_info
    return {
        'device_id': dev.id,
        'name': cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode(),
        'compute_capability': f"{dev.compute_capability[0]}.{dev.compute_capability[1]}"
                              if isinstance(dev.compute_capability, tuple)
                              else str(dev.compute_capability),
        'mem_free_mb': free // (1024 * 1024),
        'mem_total_mb': total // (1024 * 1024),
    }
