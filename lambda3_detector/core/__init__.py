"""Pure numerical core: JIT-compiled kernels, jumps, topology, entropy, and inverse-problem objectives."""

from .adaptive_params import (
    compute_adaptive_window_size,
    compute_lambda3_adaptive_parameters,
    apply_adaptive_parameters,
)
from .jumps_jit import (
    calculate_diff_and_threshold,
    detect_jumps,
    calculate_local_std,
    calculate_rho_t,
    compute_jump_consistency_term,
    sync_rate_at_lag,
    calculate_sync_profile_jit,
)
from .topology_jit import compute_topological_charge_jit
from .entropy_jit import (
    compute_entropy_shannon_jit,
    compute_entropy_renyi_jit,
    compute_entropy_tsallis_jit,
    compute_all_entropies_jit,
)
from .kernels_jit import (
    periodic_kernel,
    rbf_kernel,
    polynomial_kernel,
    sigmoid_kernel,
    laplacian_kernel,
    compute_kernel_gram_matrix,
)
from .pulsation_jit import (
    compute_pulsation_energy_from_jumps,
    compute_pulsation_energy_from_path,
    find_jump_indices,
)
from .inverse_problem_jit import (
    inverse_problem_objective_jit,
    inverse_problem_topo_objective_jit,
    compute_lambda3_reconstruction_error,
    compute_lambda3_hybrid_tikhonov_scores,
)

__all__ = [
    'compute_adaptive_window_size',
    'compute_lambda3_adaptive_parameters',
    'apply_adaptive_parameters',
    'calculate_diff_and_threshold',
    'detect_jumps',
    'calculate_local_std',
    'calculate_rho_t',
    'compute_jump_consistency_term',
    'sync_rate_at_lag',
    'calculate_sync_profile_jit',
    'compute_topological_charge_jit',
    'compute_entropy_shannon_jit',
    'compute_entropy_renyi_jit',
    'compute_entropy_tsallis_jit',
    'compute_all_entropies_jit',
    'periodic_kernel',
    'rbf_kernel',
    'polynomial_kernel',
    'sigmoid_kernel',
    'laplacian_kernel',
    'compute_kernel_gram_matrix',
    'compute_pulsation_energy_from_jumps',
    'compute_pulsation_energy_from_path',
    'find_jump_indices',
    'inverse_problem_objective_jit',
    'inverse_problem_topo_objective_jit',
    'compute_lambda3_reconstruction_error',
    'compute_lambda3_hybrid_tikhonov_scores',
]
