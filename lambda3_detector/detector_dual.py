"""
Lambda3DualDetector — 急性／慢性 dual-track 異常検知。

設計:
  Acute track (急性 — jump-localized anomalies)
    paths      : solve_inverse_problem_sparse  (jump近傍重点)
    scorers    : JumpScorer + HybridScorer(alpha=0.3)
    aggregation: 0.4·z(jump) + 0.6·z(hybrid)

  Chronic track (慢性 — drift / extended anomalies)
    paths      : inverse_problem_jump_constrained  (full)
    scorers    : StructuralScorer + KernelScorer + DriftScorer
    aggregation: mean of z-standardized scores

  Fusion: max(z(acute), z(chronic))

ジャンプ構造の検出と物理量計算は両 track 共通だが、
sparse path / full path の **解の違い** が急性／慢性の役割分担を生む。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import config as _config
from .analysis.multiscale_jumps import detect_multiscale_jumps
from .analysis.physical_quantities import (
    classify_structures,
    compute_jump_aware_topology,
    compute_jump_conditional_entropies,
    compute_pulsation_energies,
)
from .analysis.structure_tensor import inverse_problem_jump_constrained
from .analysis.structure_tensor_sparse import solve_inverse_problem_sparse
from .config import L3Config, Lambda3Result
from .core.adaptive_params import compute_adaptive_window_size
from .scorers import (
    DriftScorer,
    HybridScorer,
    JumpScorer,
    KernelScorer,
    StructuralScorer,
)


def _z(x: np.ndarray) -> np.ndarray:
    """Standardize to zero mean, unit variance (safe on flat input)."""
    s = float(np.std(x))
    if s < 1e-10:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / s


def _build_result(events: np.ndarray,
                  paths: Dict[int, np.ndarray],
                  jump_structures: Dict) -> Lambda3Result:
    charges, stabilities = compute_jump_aware_topology(paths, jump_structures)
    energies = compute_pulsation_energies(paths, jump_structures)
    entropies = compute_jump_conditional_entropies(paths, jump_structures)
    classifications = classify_structures(paths, charges, stabilities, jump_structures)
    return Lambda3Result(
        paths=paths,
        topological_charges=charges,
        stabilities=stabilities,
        energies=energies,
        entropies=entropies,
        classifications=classifications,
        jump_structures=jump_structures,
    )


class Lambda3DualDetector:
    """Dual-track Lambda³ detector with late fusion.

    Example::

        detector = Lambda3DualDetector(L3Config())
        detector.analyze(events)
        scores = detector.detect_anomalies(events)
        # 必要なら個別の track score にもアクセス
        acute   = detector.acute_scores
        chronic = detector.chronic_scores
    """

    def __init__(self, config: L3Config = None):
        self.config = config or L3Config()
        self.acute_result: Lambda3Result | None = None
        self.chronic_result: Lambda3Result | None = None
        self.jump_structures: Dict | None = None
        self.acute_scores: np.ndarray | None = None
        self.chronic_scores: np.ndarray | None = None

    def analyze(self, events: np.ndarray) -> "Lambda3DualDetector":
        """両 track の構造テンソルを解いて Lambda3Result を構築。"""
        # adaptive window前処理（既存 detector と同じ手順）
        aw = compute_adaptive_window_size(events)
        _config.update_global_constants(aw)

        # ジャンプ構造は共通
        self.jump_structures = detect_multiscale_jumps(events)
        n_paths = self.config.n_paths

        # ----- Acute: sparse solver -----
        paths_a, _stats = solve_inverse_problem_sparse(
            events, self.jump_structures, n_paths,
            self.config.alpha, self.config.beta,
            verbose=False,
        )
        self.acute_result = _build_result(events, paths_a, self.jump_structures)

        # ----- Chronic: full solver -----
        paths_c = inverse_problem_jump_constrained(
            events, self.jump_structures, n_paths,
            self.config.alpha, self.config.beta,
        )
        self.chronic_result = _build_result(events, paths_c, self.jump_structures)

        return self

    # ===============================
    # scoring
    # ===============================

    def _acute_score(self, events: np.ndarray) -> np.ndarray:
        if self.acute_result is None:
            raise RuntimeError("analyze() を先に呼んで下さい")
        np.random.seed(0); jump   = JumpScorer().score(events, self.acute_result)
        np.random.seed(0); hybrid = HybridScorer(alpha=0.3).score(events, self.acute_result)
        return 0.4 * _z(jump) + 0.6 * _z(hybrid)

    def _chronic_score(self, events: np.ndarray) -> np.ndarray:
        if self.chronic_result is None:
            raise RuntimeError("analyze() を先に呼んで下さい")
        np.random.seed(0); struct = StructuralScorer().score(events, self.chronic_result)
        np.random.seed(0); kernel = KernelScorer(kernel_type=1, degree=7, coef0=1.0).score(events, self.chronic_result)
        np.random.seed(0); drift  = DriftScorer().score(events, self.chronic_result)
        return (_z(struct) + _z(kernel) + _z(drift)) / 3.0

    def detect_anomalies(self, events: np.ndarray) -> np.ndarray:
        """両 track のスコアを計算し、max(z, z) で late fusion した最終スコアを返す。"""
        acute = self._acute_score(events)
        chronic = self._chronic_score(events)

        self.acute_scores = acute
        self.chronic_scores = chronic

        return np.maximum(_z(acute), _z(chronic))
