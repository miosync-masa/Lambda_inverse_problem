"""Analysis pipelines: multiscale jump detection, inverse-problem structure tensor, physical quantities."""

from .multiscale_jumps import (
    detect_multiscale_jumps,
    detect_multiscale_jumps_with_params,
    integrate_cross_feature_jumps,
    detect_jump_clusters,
)
from .structure_tensor import (
    solve_inverse_problem,
    inverse_problem_jump_constrained,
)
from .structure_tensor_sparse import (
    compute_significant_pairs,
    solve_inverse_problem_sparse,
)
from .drift_detection import compute_drift_scores, detect_drift
from .physical_quantities import (
    compute_topology,
    compute_energies,
    compute_entropies,
    classify_structures,
    compute_jump_aware_topology,
    compute_pulsation_energies,
    compute_jump_conditional_entropies,
)

__all__ = [
    'detect_multiscale_jumps',
    'detect_multiscale_jumps_with_params',
    'integrate_cross_feature_jumps',
    'detect_jump_clusters',
    'solve_inverse_problem',
    'inverse_problem_jump_constrained',
    'compute_significant_pairs',
    'solve_inverse_problem_sparse',
    'compute_drift_scores',
    'detect_drift',
    'compute_topology',
    'compute_energies',
    'compute_entropies',
    'classify_structures',
    'compute_jump_aware_topology',
    'compute_pulsation_energies',
    'compute_jump_conditional_entropies',
]
