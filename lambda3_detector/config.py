"""
Lambda³ configuration: dataclasses and mutable global constants.

All JIT-relevant tunables live here so that other modules can read them
via attribute access (``from . import config; config.LOCAL_WINDOW_SIZE``)
and see live updates performed by :func:`update_global_constants` /
:func:`apply_adaptive_parameters`.

Numba caveat: ``@njit`` functions that reference these names as defaults
freeze the value at compile time. The runtime mutation here is intentional
for the *Python-level* call sites (e.g. analysis pipelines that read the
constant inside the function body each call).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# ===============================
# Global Constants (for JIT Optimization)
# ===============================

DELTA_PERCENTILE = 94.0          # Percentile threshold for global jump detection (large jumps)
LOCAL_WINDOW_SIZE = 15           # Window size for local statistics (used in adaptive detection)
LOCAL_JUMP_PERCENTILE = 91.0     # Percentile threshold for local jumps
WINDOW_SIZE = 30                 # General-purpose window size (e.g., for rolling std/mean)

# Multi-scale jump detection parameters
MULTI_SCALE_WINDOWS = [3, 5, 10, 20, 40]      # Detect jumps at multiple temporal resolutions
MULTI_SCALE_PERCENTILES = [75.0, 80.0, 85.0, 90.0, 93.0, 95.0]  # Adaptive thresholds for each scale

# ===============================
# Global Constants (for KERNEL optimization) - Tuned Version
# ===============================

DEFAULT_KERNEL_TYPE = 3      # 3 = Laplacian kernel as default (RBF=1, Poly=2, Laplace=3)
DEFAULT_GAMMA = 1.0          # Gamma parameter for kernel functions
DEFAULT_DEGREE = 3           # Degree for polynomial kernel
DEFAULT_COEF0 = 1.0          # Coefficient for polynomial kernel
DEFAULT_ALPHA = 0.01         # Regularization weight (Tikhonov or ridge-like penalties)

# ===============================
# Data Class Definitions
# ===============================

@dataclass
class Lambda3Result:
    """
    Data class for storing results of Lambda³ structural analysis.
    """
    paths: Dict[int, np.ndarray]                   # Structure tensor paths for each component
    topological_charges: Dict[int, float]          # Topological charge Q_Λ for each path
    stabilities: Dict[int, float]                  # Topological stability σ_Q for each path
    energies: Dict[int, float]                     # Pulsation/energy metrics for each path
    entropies: Dict[int, Dict[str, float]]         # Multi-type entropies (Shannon, Rényi, Tsallis, etc.)
    classifications: Dict[int, str]                # Path-level classification labels (if any)
    jump_structures: Optional[Dict] = None         # Additional jump/transition structure info (optional)

@dataclass
class L3Config:
    """
    Configuration parameters for Lambda³ analysis.
    """
    alpha: float = 0.05         # L2 regularization (reduced from 0.1)
    beta: float = 0.005         # L1 regularization (reduced from 0.01)
    n_paths: int = 7            # Number of structure tensor paths (increased from 5)
    jump_scale: float = 1.5     # Sensitivity of jump detection (decreased from 2.0)
    use_union: bool = True      # Whether to use union of jumps across scales
    w_topo: float = 0.3         # Weight for topological anomaly score (increased from 0.2)
    w_pulse: float = 0.2        # Weight for pulsation score (decreased from 0.3)
    use_sparse_solver: bool = True  # 既定でsparseソルバを使う (+0.05 AUC mean across scenarios)
    sparse_expand_window: int = 5   # sparseソルバ: jump近傍の拡張幅
    sparse_n_static_samples: int = 50  # sparseソルバ: 静的領域サンプル数

    # GPU 実行（Colab 等、CuPy インストール前提）
    # True で gpu/* を経由する。CPU フォールバックは無いので CuPy 必須。
    # ローカル Mac での既存 CPU ベンチ動作を壊さないため default は False。
    # Colab では `L3Config(use_gpu=True)` を明示する。
    use_gpu: bool = False
    gpu_dtype: str = 'float32'

@dataclass
class OptimizationResult:
    """
    Data class for storing optimization results (e.g., feature selection, weight tuning).
    """
    selected_features: List[str]                # Selected features (after optimization)
    weights: Dict[str, float]                   # Component/feature weights
    auc: float                                 # Final AUC score
    feature_correlations: Dict[str, float]      # Correlation scores for selected features
    feature_groups: Optional[Dict[str, List[str]]] = None  # Optional grouping (e.g., for ensemble/group detection)

@dataclass
class DetectionStrategy:
    """
    Data class defining detection strategy.
    """
    method: str                                # "single", "group", "ensemble", etc.
    features: List[str]                        # Features/components used for detection
    weights: Dict[str, float]                  # Weights for each feature/component
    confidence: float                          # Confidence score of current detection logic


# ===============================
# Mutators for global constants
# ===============================

def update_global_constants(window_sizes: Dict[str, int]):
    """グローバル定数を動的に更新"""
    global LOCAL_WINDOW_SIZE, WINDOW_SIZE, MULTI_SCALE_WINDOWS

    LOCAL_WINDOW_SIZE = window_sizes['local']
    WINDOW_SIZE = window_sizes['tension']
    MULTI_SCALE_WINDOWS = window_sizes['multiscale']

    print(f"Window sizes updated:")
    print(f"  LOCAL_WINDOW_SIZE: {LOCAL_WINDOW_SIZE}")
    print(f"  WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"  MULTI_SCALE_WINDOWS: {MULTI_SCALE_WINDOWS}")


def update_detection_percentiles(delta: float, local_jump: float, multiscale: List[float]):
    """Detection percentilesの動的更新（adaptive_paramsから呼ばれる）"""
    global DELTA_PERCENTILE, LOCAL_JUMP_PERCENTILE, MULTI_SCALE_PERCENTILES
    DELTA_PERCENTILE = delta
    LOCAL_JUMP_PERCENTILE = local_jump
    MULTI_SCALE_PERCENTILES = multiscale
