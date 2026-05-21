"""
Lambda3 detector の GPU (CuPy) 実装。

Colab / 単機 GPU 環境前提。CPU フォールバックは持たない。
CPU 実装は `lambda3_detector.analysis.structure_tensor_sparse` 等にそのまま残置。

L3Config.use_gpu=True (デフォルト) で `detector.analyze()` が gpu.* を経由する。
"""

from .backend import (
    cp,
    DEFAULT_DTYPE,
    ensure_gpu,
    to_cpu,
    device_info,
)
from .inverse_sparse_gpu import solve_inverse_problem_sparse_gpu
from .scorers_gpu import (
    compute_kernel_gram_matrix_gpu,
    kernel_anomaly_scores_gpu,
    kernel_anomaly_scores_auto_gpu,
)

__all__ = [
    'cp',
    'DEFAULT_DTYPE',
    'ensure_gpu',
    'to_cpu',
    'device_info',
    'solve_inverse_problem_sparse_gpu',
    'compute_kernel_gram_matrix_gpu',
    'kernel_anomaly_scores_gpu',
    'kernel_anomaly_scores_auto_gpu',
]
