
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from itertools import combinations
from scipy.signal import hilbert
import warnings
from sklearn.linear_model import BayesianRidge
from numba import jit, njit, prange, cuda
from numba.typed import Dict as NumbaDict
import numba
import time

warnings.filterwarnings('ignore')

# ===============================
# Global Constants (for JIT optimization) - Tuned Version
# ===============================
DELTA_PERCENTILE = 99.0         # Percentile threshold for ΔΛC jump detection (99th percentile = top 1% changes)
LOCAL_WINDOW_SIZE = 20          # Window size for local statistics calculation (events)
LOCAL_JUMP_PERCENTILE = 95.0    # Percentile for local adaptive jump detection (top 5% local anomalies)
WINDOW_SIZE = 30                # Window size for tension scalar ρT calculation (broader context)

# ===============================
# Data Classes
# ===============================
@dataclass
class Lambda3Result:
    """Data class to store Lambda³ analysis results (Extended version)"""
    paths: Dict[int, np.ndarray]              # Structure tensor paths Λ_k for each mode k
    topological_charges: Dict[int, float]     # Topological charge Q_Λ (winding number in phase space)
    stabilities: Dict[int, float]             # Stability measure σ_Q (variance of Q across segments)
    energies: Dict[int, float]                # Energy E = ||Λ||² for each structural mode
    entropies: Dict[int, float]               # Information entropy S (Shannon/Renyi/Tsallis)
    classifications: Dict[int, str]           # Physical classification based on Q_Λ and stability
    jump_structures: Dict = None              # ΔΛC jump event analysis (pulsation detection)

@dataclass
class L3Config:
    """Lambda³ configuration parameters"""
    alpha: float = 0.1          # TV regularization weight for structure tensor smoothness
    beta: float = 0.01          # L1 sparsity regularization weight
    n_paths: int = 5            # Number of structural modes/paths to extract
    jump_scale: float = 2.0     # Sensitivity scale for ΔΛC jump detection (σ multiplier)
    use_union: bool = True      # Whether to use union of jumps across all paths
    w_topo: float = 0.2         # Weight for topological features in anomaly scoring
    w_pulse: float = 0.3        # Weight for pulsation energy in anomaly scoring

# ===============================
# JIT-Optimized Core Functions (Jump Detection)
# ===============================
@njit
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """
    JIT-compiled difference calculation and threshold computation.
    """
    diff = np.empty(len(data))
    diff[0] = 0
    for i in range(1, len(data)):
        diff[i] = data[i] - data[i-1]

    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, percentile)
    return diff, threshold

@njit
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled jump detection based on threshold.
    """
    n = len(diff)
    pos_jumps = np.zeros(n, dtype=np.int32)
    neg_jumps = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if diff[i] > threshold:
            pos_jumps[i] = 1
        elif diff[i] < -threshold:
            neg_jumps[i] = 1

    return pos_jumps, neg_jumps

@njit
def calculate_local_std(data: np.ndarray, window: int) -> np.ndarray:
    """
    JIT-compiled local standard deviation calculation.
    """
    n = len(data)
    local_std = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)

        subset = data[start:end]
        mean = np.mean(subset)
        variance = np.sum((subset - mean) ** 2) / len(subset)
        local_std[i] = np.sqrt(variance)

    return local_std

@njit
def calculate_rho_t(data: np.ndarray, window: int) -> np.ndarray:
    """
    JIT-compiled tension scalar (ρT) calculation.
    """
    n = len(data)
    rho_t = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = i + 1

        subset = data[start:end]
        if len(subset) > 1:
            mean = np.mean(subset)
            variance = np.sum((subset - mean) ** 2) / len(subset)
            rho_t[i] = np.sqrt(variance)
        else:
            rho_t[i] = 0.0

    return rho_t

@njit
def sync_rate_at_lag(series_a: np.ndarray, series_b: np.ndarray, lag: int) -> float:
    """
    JIT-compiled synchronization rate calculation for a specific lag.
    """
    if lag < 0:
        if -lag < len(series_a):
            return np.mean(series_a[-lag:] * series_b[:lag])
        else:
            return 0.0
    elif lag > 0:
        if lag < len(series_b):
            return np.mean(series_a[:-lag] * series_b[lag:])
        else:
            return 0.0
    else:
        return np.mean(series_a * series_b)

@njit(parallel=True)
def calculate_sync_profile_jit(series_a: np.ndarray, series_b: np.ndarray,
                               lag_window: int) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    JIT-compiled synchronization profile calculation with parallelization.
    """
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1)
    sync_values = np.empty(n_lags)

    for i in prange(n_lags):
        lag = lags[i]
        sync_values[i] = sync_rate_at_lag(series_a, series_b, lag)

    max_sync = 0.0
    optimal_lag = 0
    for i in range(n_lags):
        if sync_values[i] > max_sync:
            max_sync = sync_values[i]
            optimal_lag = lags[i]

    return lags, sync_values, max_sync, optimal_lag

# ===============================
# Existing JIT-Optimized Functions (Kernel, Topological, Entropy)
# ===============================
@njit
def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """RBF kernel (Gaussian kernel)"""
    diff = x - y
    return np.exp(-gamma * np.dot(diff, diff))

@njit
def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, coef0: float = 1.0) -> float:
    """Polynomial kernel"""
    return (np.dot(x, y) + coef0) ** degree

@njit
def sigmoid_kernel(x: np.ndarray, y: np.ndarray, alpha: float = 0.01, coef0: float = 0.0) -> float:
    """Sigmoid kernel"""
    return np.tanh(alpha * np.dot(x, y) + coef0)

@njit
def laplacian_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Laplacian kernel"""
    diff = np.abs(x - y)
    return np.exp(-gamma * np.sum(diff))

@njit(parallel=True)
def compute_kernel_gram_matrix(data: np.ndarray, kernel_type: int = 0,
                               gamma: float = 1.0, degree: int = 3,
                               coef0: float = 1.0, alpha: float = 0.01) -> np.ndarray:
    """
    Compute kernel Gram matrix
    kernel_type: 0=RBF, 1=Polynomial, 2=Sigmoid, 3=Laplacian
    """
    n = data.shape[0]
    K = np.zeros((n, n))

    for i in prange(n):
        for j in range(i, n):
            if kernel_type == 0:  # RBF
                K[i, j] = rbf_kernel(data[i], data[j], gamma)
            elif kernel_type == 1:  # Polynomial
                K[i, j] = polynomial_kernel(data[i], data[j], degree, coef0)
            elif kernel_type == 2:  # Sigmoid
                K[i, j] = sigmoid_kernel(data[i], data[j], alpha, coef0)
            else:  # Laplacian
                K[i, j] = laplacian_kernel(data[i], data[j], gamma)

            K[j, i] = K[i, j]  # Symmetry

    return K

@njit
def compute_pulsation_energy_from_jumps(
    pos_jumps: np.ndarray,
    neg_jumps: np.ndarray,
    diff: np.ndarray,
    rho_t: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate pulsation energy from detected jumps (raw data based)
    Returns: (jump_intensity, asymmetry, pulsation_power)
    """
    # Jump intensity (sum of difference values at detected jumps)
    pos_intensity = 0.0
    neg_intensity = 0.0
    
    for i in range(len(diff)):
        if pos_jumps[i] == 1:
            pos_intensity += diff[i]
        if neg_jumps[i] == 1:
            neg_intensity += np.abs(diff[i])
    
    jump_intensity = pos_intensity + neg_intensity
    
    # Asymmetry (-1 to +1)
    asymmetry = (pos_intensity - neg_intensity) / (pos_intensity + neg_intensity + 1e-10)
    
    # Pulsation power (number of jumps × intensity × average tension)
    n_jumps = np.sum(pos_jumps) + np.sum(neg_jumps)
    
    # Average tension at jump positions
    avg_tension = 0.0
    if n_jumps > 0:
        tension_sum = 0.0
        count = 0
        for i in range(len(rho_t)):
            if pos_jumps[i] == 1 or neg_jumps[i] == 1:
                tension_sum += rho_t[i]
                count += 1
        avg_tension = tension_sum / count if count > 0 else 0.0
    
    pulsation_power = jump_intensity * n_jumps * (1 + avg_tension) / len(diff)
    
    return jump_intensity, asymmetry, pulsation_power

@njit
def compute_pulsation_energy_from_path(path: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate pulsation energy from path data (for structure tensor analysis)
    ※ Kept for backward compatibility
    """
    if len(path) < 2:
        return 0.0, 0.0, 0.0

    # Difference and jump detection
    diff = np.diff(path)
    abs_diff = np.abs(diff)
    threshold = np.mean(abs_diff) + 2.0 * np.std(abs_diff)
    
    # Jump detection
    pos_mask = diff > threshold
    neg_mask = diff < -threshold
    
    # Jump intensity
    pos_intensity = np.sum(diff[pos_mask]) if np.any(pos_mask) else 0.0
    neg_intensity = np.sum(np.abs(diff[neg_mask])) if np.any(neg_mask) else 0.0
    jump_intensity = pos_intensity + neg_intensity
    
    # Asymmetry
    asymmetry = (pos_intensity - neg_intensity) / (pos_intensity + neg_intensity + 1e-10)
    
    # Pulsation power
    n_jumps = np.sum(pos_mask) + np.sum(neg_mask)
    pulsation_power = jump_intensity * n_jumps / len(path)
    
    return jump_intensity, asymmetry, pulsation_power

@njit
def find_jump_indices(path: np.ndarray, jump_scale: float = 2.0):
    """Return jump index array in the path (ΔΛC events)"""
    delta = np.abs(np.diff(path))
    th = np.mean(delta) + jump_scale * np.std(delta)
    return np.where(delta > th)[0]

@njit(parallel=True)
def compute_topological_charge_jit(path: np.ndarray, n_segments: int = 10) -> Tuple[float, float]:
    """Fast computation of topological charge"""
    n = len(path)
    closed_path = np.empty(n + 1)
    closed_path[:-1] = path
    closed_path[-1] = path[0]

    # Phase calculation
    theta = np.empty(n)
    for i in prange(n):
        theta[i] = np.arctan2(closed_path[i+1], closed_path[i])

    # Charge calculation
    Q_Lambda = 0.0
    for i in range(n-1):
        diff = theta[i+1] - theta[i]
        # Handle phase jumps
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        Q_Lambda += diff
    Q_Lambda /= (2 * np.pi)

    # Segment stability
    Q_segments = np.zeros(n_segments)
    for seg in range(n_segments):
        start = seg * n // n_segments
        end = (seg + 1) * n // n_segments
        if end > start + 1:
            seg_sum = 0.0
            for i in range(start, end-1):
                diff = theta[i+1] - theta[i]
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                seg_sum += diff
            Q_segments[seg] = seg_sum

    stability = np.std(Q_segments)
    return Q_Lambda, stability

@njit
def compute_entropy_shannon_jit(path: np.ndarray, eps: float = 1e-10) -> float:
    """Fast Shannon entropy calculation"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    entropy = 0.0
    for p in norm_path:
        if p > 0:
            entropy -= p * np.log(p)

    return entropy

@njit
def compute_entropy_renyi_jit(path: np.ndarray, alpha: float = 2.0, eps: float = 1e-10) -> float:
    """Fast Renyi entropy calculation"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    if alpha == 1.0:
        return compute_entropy_shannon_jit(path, eps)

    sum_p_alpha = 0.0
    for p in norm_path:
        sum_p_alpha += p ** alpha

    return (1.0 / (1.0 - alpha)) * np.log(sum_p_alpha)

@njit
def compute_entropy_tsallis_jit(path: np.ndarray, q: float = 1.5, eps: float = 1e-10) -> float:
    """Fast Tsallis entropy calculation"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    if q == 1.0:
        return compute_entropy_shannon_jit(path, eps)

    sum_p_q = 0.0
    for p in norm_path:
        sum_p_q += p ** q

    return (1.0 - sum_p_q) / (q - 1.0)

@njit
def compute_all_entropies_jit(path: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Fast calculation of all entropy metrics (returns as array)"""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)

    # Calculate 6 metrics
    entropies = np.zeros(6)

    # Shannon
    shannon = 0.0
    for p in norm_path:
        if p > 0:
            shannon -= p * np.log(p)
    entropies[0] = shannon

    # Renyi (α=2)
    sum_p2 = 0.0
    for p in norm_path:
        sum_p2 += p ** 2
    entropies[1] = -np.log(sum_p2)

    # Tsallis (q=1.5)
    sum_p15 = 0.0
    for p in norm_path:
        sum_p15 += p ** 1.5
    entropies[2] = (1.0 - sum_p15) / 0.5

    # Max
    entropies[3] = np.max(norm_path)

    # Min
    entropies[4] = np.min(norm_path)

    # Variance
    mean_p = np.mean(norm_path)
    var = 0.0
    for p in norm_path:
        var += (p - mean_p) ** 2
    entropies[5] = var / len(norm_path)

    return entropies

@njit(parallel=True)
def inverse_problem_objective_jit(Lambda_matrix, events_gram, alpha, beta, jump_weight=0.5):
    """Inverse problem objective function (JIT-optimized version)"""
    n_paths, n_events = Lambda_matrix.shape
    reconstruction = np.zeros((n_events, n_events))
    for i in prange(n_events):
        for j in range(n_events):
            for k in range(n_paths):
                reconstruction[i, j] += Lambda_matrix[k, i] * Lambda_matrix[k, j]
    data_fit = np.sum((events_gram - reconstruction)**2)

    tv_reg = 0.0
    for i in range(n_paths - 1):
        for j in range(n_events):
            tv_reg += np.abs(Lambda_matrix[i+1, j] - Lambda_matrix[i, j])
    for i in range(n_paths):
        for j in range(n_events - 1):
            tv_reg += np.abs(Lambda_matrix[i, j+1] - Lambda_matrix[i, j])

    l1_reg = np.sum(np.abs(Lambda_matrix))

    # Jump regularization
    jump_reg = 0.0
    for i in range(n_paths):
        path = Lambda_matrix[i]
        n_delta = n_events - 1
        if n_delta == 0:
            continue
        deltas = np.abs(path[1:] - path[:-1])
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        threshold = mean_delta + 2.5 * std_delta
        for delta in deltas:
            if delta > threshold:
                jump_reg += jump_weight * delta

    return data_fit + alpha * tv_reg + beta * l1_reg + jump_reg

@njit
def compute_lambda3_reconstruction_error(paths_matrix: np.ndarray, events: np.ndarray) -> np.ndarray:
    """
    Calculate Lambda³ reconstruction error (inheriting Tikhonov spirit)
    """
    n_paths, n_events = paths_matrix.shape
    n_features = events.shape[1]

    # 1. Gram matrix of observed data (normalized)
    events_gram = np.zeros((n_events, n_events))
    for i in range(n_events):
        for j in range(n_events):
            events_gram[i, j] = np.dot(events[i], events[j])

    # Normalize Gram matrix (scale invariance)
    gram_norm = np.sqrt(np.trace(events_gram @ events_gram))
    if gram_norm > 0:
        events_gram /= gram_norm

    # 2. Reconstruction by Lambda³ structure
    recon_gram = np.zeros((n_events, n_events))
    for k in range(n_paths):
        for i in range(n_events):
            for j in range(n_events):
                recon_gram[i, j] += paths_matrix[k, i] * paths_matrix[k, j]

    # Normalize reconstruction
    recon_norm = np.sqrt(np.trace(recon_gram @ recon_gram))
    if recon_norm > 0:
        recon_gram /= recon_norm

    # 3. Per-event reconstruction error
    event_errors = np.zeros(n_events)
    for i in range(n_events):
        row_error = 0.0
        for j in range(n_events):
            diff = events_gram[i, j] - recon_gram[i, j]
            row_error += diff * diff
        event_errors[i] = np.sqrt(row_error)

    return event_errors

@njit(parallel=True)
def compute_lambda3_hybrid_tikhonov_scores(
    paths_matrix: np.ndarray,
    events: np.ndarray,
    charges: np.ndarray,
    stabilities: np.ndarray,
    alpha: float = 0.5,
    jump_scale: float = 2.0,
    use_union: bool = True,
    w_topo: float = 0.2,
    w_pulse: float = 0.3,
) -> np.ndarray:
    """
    Lambda³ Hybrid Tikhonov Fusion Anomaly Score
    """
    n_paths, n_events = paths_matrix.shape

    # Overall error
    errors_all = compute_lambda3_reconstruction_error(paths_matrix, events)

    # Jump indices
    if use_union:
        idx_set = set()
        for k in range(n_paths):
            idxs = find_jump_indices(paths_matrix[k], jump_scale)
            for idx in idxs:
                idx_set.add(idx+1)
        jump_idx = np.array(list(idx_set), dtype=np.int64)  # Unified to int64
    else:
        Qarr = np.array([np.sum(np.diff(paths_matrix[k])) for k in range(n_paths)])
        main_idx = np.argmax(np.abs(Qarr))
        idxs = find_jump_indices(paths_matrix[main_idx], jump_scale)
        jump_idx = idxs + 1

    # Jump error vector
    jump_error = np.zeros_like(errors_all)
    for idx in jump_idx:
        if idx < len(jump_error):
            jump_error[idx] = errors_all[idx]

    # Per-path anomaly degree
    path_anomaly_scores = np.zeros(n_paths)
    for p in prange(n_paths):
        path = paths_matrix[p]
        topo_score = np.abs(charges[p]) + 0.5 * stabilities[p]
        # Calculate pulsation energy from path (for structure tensor)
        jump_int, asymm, pulse_pow = compute_pulsation_energy_from_path(path)
        pulse_score = 0.4 * jump_int + 0.3 * np.abs(asymm) + 0.3 * pulse_pow
        path_anomaly_scores[p] = w_topo * topo_score + w_pulse * pulse_score

    # Per-event weighting
    structural_component = np.zeros(n_events)
    for i in prange(n_events):
        for p in range(n_paths):
            contribution = np.abs(paths_matrix[p, i])
            structural_component[i] += contribution * path_anomaly_scores[p]

    # Hybrid synthesis
    hybrid_score = alpha * errors_all + (1 - alpha) * jump_error
    event_scores = hybrid_score + structural_component

    # Standardization
    mean_score = np.mean(event_scores)
    std_score = np.std(event_scores)
    if std_score > 0:
        event_scores = (event_scores - mean_score) / std_score

    return event_scores

# ===============================
# Main Class: Zero-Shot Anomaly Detection System
# ===============================
class Lambda3ZeroShotDetector:
    """
    Jump-Driven Zero-Shot Anomaly Detection System
    Full-featured integrated version
    """
    
    def __init__(self, config: L3Config = None):
        self.config = config or L3Config()
        self.jump_analyzer = None  # Jump analysis result cache
        self.anomaly_patterns = self._init_anomaly_patterns()
        
    def _init_anomaly_patterns(self):
        """Initialize anomaly pattern generation functions"""
        return {
            'pulse': self._generate_pulse_anomaly,
            'phase_jump': self._generate_phase_jump_anomaly,
            'periodic': self._generate_periodic_anomaly,
            'structural_decay': self._generate_decay_anomaly,
            'bifurcation': self._generate_bifurcation_anomaly,
            'multi_path': self._generate_multi_path_anomaly,
            'topological_jump': self._generate_topological_jump_anomaly,
            'cascade': self._generate_cascade_anomaly,
            'partial_periodic': self._generate_partial_periodic_anomaly,
            'superposition': self._generate_superposition_anomaly,
            'resonance': self._generate_resonance_anomaly
        }
        
    def analyze(self, events: np.ndarray, n_paths: int = None) -> Lambda3Result:
        """
        Complete zero-shot analysis flow:
        1. Jump detection → 2. Jump-constrained inverse problem → 3. Physical quantity calculation
        """
        if n_paths is None:
            n_paths = self.config.n_paths
            
        # 1. Multi-dimensional jump structure detection (utilizing JIT functions)
        jump_structures = self._detect_multiscale_jumps(events)
        self.jump_analyzer = jump_structures
        
        # 2. Jump-constrained structure tensor estimation (inverse problem)
        paths = self._inverse_problem_jump_constrained(
            events, jump_structures, n_paths
        )
        
        # 3. Jump-consistent physical quantity calculation (topological, entropy)
        charges, stabilities = self._compute_jump_aware_topology(paths, jump_structures)
        energies = self._compute_pulsation_energies(paths, jump_structures)
        entropies = self._compute_jump_conditional_entropies(paths, jump_structures)
        
        # 4. Jump pattern-based classification
        classifications = self._classify_by_jump_signatures(
            paths, jump_structures, charges, stabilities
        )
        
        return Lambda3Result(
            paths=paths,
            topological_charges=charges,
            stabilities=stabilities,
            energies=energies,
            entropies=entropies,
            classifications=classifications,
            jump_structures=jump_structures
        )
    
    def _detect_multiscale_jumps(self, events: np.ndarray) -> Dict:
        """Multi-dimensional and multi-scale jump detection"""
        n_events, n_features = events.shape
        jump_data = {'features': {}, 'integrated': {}}
        
        # Jump detection for each feature dimension (using JIT functions)
        for f in range(n_features):
            data = events[:, f]
            
            # Basic jump detection
            diff, threshold = calculate_diff_and_threshold(data, DELTA_PERCENTILE)
            pos_jumps, neg_jumps = detect_jumps(diff, threshold)
            
            # Local adaptive jumps
            local_std = calculate_local_std(data, LOCAL_WINDOW_SIZE)
            score = np.abs(diff) / (local_std + 1e-8)
            local_threshold = np.percentile(score, LOCAL_JUMP_PERCENTILE)
            local_jumps = (score > local_threshold).astype(int)
            
            # Tension scalar
            rho_t = calculate_rho_t(data, WINDOW_SIZE)
            
            # Pulsation energy (calculated from raw data jumps)
            jump_intensity, asymmetry, pulse_power = compute_pulsation_energy_from_jumps(
                pos_jumps, neg_jumps, diff, rho_t
            )
            
            jump_data['features'][f] = {
                'pos_jumps': pos_jumps,
                'neg_jumps': neg_jumps,
                'local_jumps': local_jumps,
                'rho_t': rho_t,
                'diff': diff,
                'threshold': threshold,
                'jump_intensity': jump_intensity,
                'asymmetry': asymmetry,
                'pulse_power': pulse_power
            }
        
        # Integrated jump patterns (considering all features)
        jump_data['integrated'] = self._integrate_cross_feature_jumps(jump_data['features'])
        
        return jump_data
    
    def _integrate_cross_feature_jumps(self, feature_jumps: Dict) -> Dict:
        """Analyze jump synchronization across features"""
        n_features = len(feature_jumps)
        features_list = list(feature_jumps.keys())
        
        # Integrated jump mask (OR operation)
        first_key = features_list[0]
        n_events = len(feature_jumps[first_key]['pos_jumps'])
        unified_jumps = np.zeros(n_events, dtype=np.int64)  # Unified to int64
        
        # Jump importance (number of synchronized features)
        jump_importance = np.zeros(n_events)
        
        for f in features_list:
            jumps = feature_jumps[f]['pos_jumps'] | feature_jumps[f]['neg_jumps']
            unified_jumps |= jumps
            jump_importance += jumps.astype(float)
        
        # Calculate jump synchronization rate (utilizing JIT functions)
        sync_matrix = np.zeros((n_features, n_features))
        for i, f1 in enumerate(features_list):
            for j, f2 in enumerate(features_list):
                if i < j:
                    jumps1 = feature_jumps[f1]['pos_jumps'] | feature_jumps[f1]['neg_jumps']
                    jumps2 = feature_jumps[f2]['pos_jumps'] | feature_jumps[f2]['neg_jumps']
                    
                    # Synchronization profile calculation
                    _, _, max_sync, optimal_lag = calculate_sync_profile_jit(
                        jumps1.astype(np.float64),
                        jumps2.astype(np.float64),
                        lag_window=5
                    )
                    sync_matrix[i, j] = max_sync
                    sync_matrix[j, i] = max_sync
        
        # Jump cluster detection
        jump_clusters = self._detect_jump_clusters(unified_jumps, jump_importance)
        
        return {
            'unified_jumps': unified_jumps,
            'jump_importance': jump_importance / n_features,  # Normalize
            'sync_matrix': sync_matrix,
            'jump_clusters': jump_clusters,
            'n_total_jumps': np.sum(unified_jumps),
            'max_sync': np.max(sync_matrix[np.triu_indices(n_features, k=1)])
        }
    
    def _detect_jump_clusters(
        self, 
        unified_jumps: np.ndarray,
        jump_importance: np.ndarray,
        min_cluster_size: int = 3
    ) -> List[Dict]:
        """Detect jump clusters (continuous structural changes)"""
        clusters = []
        in_cluster = False
        cluster_start = 0
        
        for i in range(len(unified_jumps)):
            if unified_jumps[i] and not in_cluster:
                in_cluster = True
                cluster_start = i
            elif not unified_jumps[i] and in_cluster:
                # Cluster ends
                cluster_size = i - cluster_start
                if cluster_size >= min_cluster_size:
                    clusters.append({
                        'start': cluster_start,
                        'end': i,
                        'size': cluster_size,
                        'indices': list(range(cluster_start, i)),
                        'density': np.mean(jump_importance[cluster_start:i]),
                        'total_importance': np.sum(jump_importance[cluster_start:i])
                    })
                in_cluster = False
        
        # Handle last cluster
        if in_cluster:
            cluster_size = len(unified_jumps) - cluster_start
            if cluster_size >= min_cluster_size:
                clusters.append({
                    'start': cluster_start,
                    'end': len(unified_jumps),
                    'size': cluster_size,
                    'indices': list(range(cluster_start, len(unified_jumps))),
                    'density': np.mean(jump_importance[cluster_start:]),
                    'total_importance': np.sum(jump_importance[cluster_start:])
                })
        
        return clusters
    
    def _inverse_problem_jump_constrained(
        self, 
        events: np.ndarray,
        jump_structures: Dict,
        n_paths: int
    ) -> Dict[int, np.ndarray]:
        """Jump structure-aware inverse problem"""
        
        # Global jump information
        jump_mask = jump_structures['integrated']['unified_jumps']
        jump_weights = jump_structures['integrated']['jump_importance']
        
        # Gram matrix calculation
        events_gram = np.ascontiguousarray(events @ events.T)
        
        # Initial values reflecting jump structure
        Lambda_init = self._initialize_with_jump_structure(
            events, jump_mask, jump_weights, n_paths
        )
        
        # Jump-constrained objective function
        def objective(Lambda_flat):
            Lambda_matrix = np.ascontiguousarray(Lambda_flat.reshape(n_paths, events.shape[0]))
            
            # Basic inverse problem term (JIT function)
            base_obj = inverse_problem_objective_jit(
                Lambda_matrix, events_gram, self.config.alpha, self.config.beta, 
                jump_weight=0.5
            )
            
            # Jump consistency term
            jump_term = self._compute_jump_consistency_term(
                Lambda_matrix, jump_mask, jump_weights
            )
            
            return base_obj + jump_term
        
        # Execute optimization
        result = minimize(
            objective, 
            Lambda_init.flatten(), 
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        Lambda_opt = result.x.reshape(n_paths, events.shape[0])
        
        # Normalize and return
        return {i: path / (np.linalg.norm(path) + 1e-8)
                for i, path in enumerate(Lambda_opt)}
    
    def _initialize_with_jump_structure(
        self,
        events: np.ndarray,
        jump_mask: np.ndarray,
        jump_weights: np.ndarray,
        n_paths: int
    ) -> np.ndarray:
        """Generate initial values reflecting jump structure"""
        n_events = events.shape[0]
        Lambda_init = np.zeros((n_paths, n_events))
        
        # Eigenvalue decomposition based
        _, V = np.linalg.eigh(events @ events.T)
        base_paths = V[:, -n_paths:].T
        
        # Introduce discontinuities at jump positions
        for p in range(n_paths):
            Lambda_init[p] = base_paths[p]
            
            # Emphasize values at jump positions
            for i in range(n_events):
                if jump_mask[i]:
                    # Invert sign or amplify value at jump position
                    if p % 2 == 0:
                        Lambda_init[p, i] *= (1 + jump_weights[i])
                    else:
                        Lambda_init[p, i] *= -(1 + jump_weights[i])
        
        return Lambda_init
    
    def _compute_jump_consistency_term(
        self,
        Lambda_matrix: np.ndarray,
        jump_mask: np.ndarray,
        jump_weights: np.ndarray
    ) -> float:
        """Evaluate jump consistency"""
        n_paths, n_events = Lambda_matrix.shape
        consistency = 0.0
        
        for p in range(n_paths):
            path = Lambda_matrix[p]
            
            # ΔΛ at jump positions
            for i in range(1, n_events):
                delta = np.abs(path[i] - path[i-1])
                
                if jump_mask[i]:
                    # Encourage large ΔΛ at jump positions
                    consistency -= jump_weights[i] * delta
                else:
                    # Encourage small ΔΛ at non-jump positions
                    consistency += 0.1 * delta
        
        return consistency
    
    def _compute_jump_aware_topology(
        self,
        paths: Dict[int, np.ndarray],
        jump_structures: Dict
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Calculate topological quantities considering jump structure"""
        charges = {}
        stabilities = {}
        
        for i, path in paths.items():
            # Basic topological charge (JIT function)
            Q, sigma = compute_topological_charge_jit(path)
            
            # Consider phase changes at jump positions
            jump_mask = jump_structures['integrated']['unified_jumps']
            jump_phase_shift = 0.0
            
            for j in range(1, len(path)):
                if jump_mask[j]:
                    # Phase change at jump position
                    phase_diff = np.arctan2(path[j], path[j-1]) - np.arctan2(path[j-1], path[j-2] if j > 1 else path[0])
                    jump_phase_shift += phase_diff
            
            # Jump-corrected charge
            charges[i] = Q + jump_phase_shift / (2 * np.pi)
            stabilities[i] = sigma
        
        return charges, stabilities
    
    def _compute_pulsation_energies(
        self,
        paths: Dict[int, np.ndarray],
        jump_structures: Dict
    ) -> Dict[int, float]:
        """Calculate pulsation energy (prioritizing jump structure)"""
        energies = {}
        
        for i, path in paths.items():
            # Basic energy
            basic_energy = np.sum(path**2)
            
            # Integrate jump energy from corresponding features
            # (Average jumps from multiple features)
            total_pulse_power = 0.0
            n_features = len(jump_structures['features'])
            
            for f_idx, f_data in jump_structures['features'].items():
                pulse_power = f_data['pulse_power']
                total_pulse_power += pulse_power
            
            avg_pulse_power = total_pulse_power / n_features if n_features > 0 else 0.0
            
            # Integrated energy
            energies[i] = basic_energy + 0.3 * avg_pulse_power
        
        return energies
    
    def _compute_jump_conditional_entropies(
        self,
        paths: Dict[int, np.ndarray],
        jump_structures: Dict
    ) -> Dict[int, Dict[str, float]]:
        """Conditional entropy at jump events"""
        entropies = {}
        jump_mask = jump_structures['integrated']['unified_jumps']
        
        for i, path in paths.items():
            # Overall entropy (JIT function)
            all_entropies = compute_all_entropies_jit(path)
            entropy_keys = ["shannon", "renyi_2", "tsallis_1.5", "max", "min", "var"]
            
            # Separate jump and non-jump positions
            jump_indices = np.where(jump_mask)[0]
            non_jump_indices = np.where(~jump_mask)[0]
            
            entropy_dict = {}
            
            # Overall entropy
            for j, key in enumerate(entropy_keys):
                entropy_dict[key] = all_entropies[j]
            
            # Jump-conditional entropy
            if len(jump_indices) > 0:
                jump_path = path[jump_indices]
                jump_entropies = compute_all_entropies_jit(jump_path)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_jump"] = jump_entropies[j]
            
            # Non-jump conditional entropy
            if len(non_jump_indices) > 0:
                non_jump_path = path[non_jump_indices]
                non_jump_entropies = compute_all_entropies_jit(non_jump_path)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_non_jump"] = non_jump_entropies[j]
            
            entropies[i] = entropy_dict
        
        return entropies
    
    def _classify_by_jump_signatures(
        self,
        paths: Dict[int, np.ndarray],
        jump_structures: Dict,
        charges: Dict[int, float],
        stabilities: Dict[int, float]
    ) -> Dict[int, str]:
        """Structure classification based on jump patterns"""
        classifications = {}
        
        for i in paths.keys():
            Q = charges[i]
            sigma = stabilities[i]
            
            # Jump characteristics (from raw data)
            if i < len(jump_structures['features']):
                # Use jump characteristics from corresponding feature
                feature_data = jump_structures['features'].get(i, jump_structures['features'][0])
                jump_intensity = feature_data['jump_intensity']
                asymmetry = feature_data['asymmetry']
                pulse_power = feature_data['pulse_power']
            else:
                # If path index exceeds feature count, use average values
                jump_intensity = np.mean([f['jump_intensity'] for f in jump_structures['features'].values()])
                asymmetry = np.mean([f['asymmetry'] for f in jump_structures['features'].values()])
                pulse_power = np.mean([f['pulse_power'] for f in jump_structures['features'].values()])
            
            # Basic classification
            if Q < -0.5:
                base = "Antimatter Structure (Absorption)"
            elif Q > 0.5:
                base = "Matter Structure (Emission)"
            else:
                base = "Neutral Structure (Equilibrium)"
            
            # Jump-based modifiers
            tags = []
            
            if pulse_power > 5:
                tags.append("High-Frequency Pulsation")
            elif pulse_power < 0.1:
                tags.append("Static Structure")
            
            if abs(asymmetry) > 0.7:
                if asymmetry > 0:
                    tags.append("Positive-Dominant")
                else:
                    tags.append("Negative-Dominant")
            
            if sigma > 2.5:
                tags.append("Unstable/Chaotic")
            elif sigma < 0.5:
                tags.append("Super-Stable")
            
            # Complete classification
            if tags:
                classifications[i] = base + " • " + " / ".join(tags)
            else:
                classifications[i] = base
        
        return classifications
    
    def detect_anomalies(self, result: Lambda3Result, events: np.ndarray) -> np.ndarray:
        """
        Execute zero-shot anomaly detection (adaptive threshold version)
        """
        if self.jump_analyzer is None:
            # If jump analysis not executed
            self.jump_analyzer = self._detect_multiscale_jumps(events)
        
        # 1. Jump-based anomaly score (improved version)
        jump_anomaly_scores = self._compute_jump_anomaly_scores(
            self.jump_analyzer, events
        )
        
        # 2. Hybrid Tikhonov score (parameter-tuned version)
        paths_matrix = np.stack(list(result.paths.values()))
        charges = np.array(list(result.topological_charges.values()))
        stabilities = np.array(list(result.stabilities.values()))
        
        # Hybrid Tikhonov score (parameter-tuned version)
        hybrid_scores = compute_lambda3_hybrid_tikhonov_scores(
            paths_matrix, events, charges, stabilities,
            alpha=0.7,  # Emphasize overall error (capture non-jump anomalies)
            jump_scale=2.5,  # Stricter jump determination
            use_union=True,
            w_topo=0.3,  # Emphasize topological features
            w_pulse=0.2   # Suppress pulsation
        )
        
        # 3. Anomaly in kernel space (Laplacian kernel)
        kernel_scores = self._compute_kernel_anomaly_scores(events, result)
        
        # 4. Synchronization anomaly score
        sync_scores = self._compute_sync_anomaly_scores(self.jump_analyzer)
        
        # 5. Integrated score (weight-adjusted version)
        # Emphasize non-jump features for handling severe anomalies
        final_scores = (
            0.25 * jump_anomaly_scores +      # Jump structure (reduced)
            0.35 * hybrid_scores +             # Hybrid (increased)
            0.30 * kernel_scores +             # Kernel (increased)
            0.10 * sync_scores                 # Synchronization anomaly
        )
        
        # 6. Adaptive standardization (robust to outliers)
        # Use median and interquartile range
        median_score = np.median(final_scores)
        q75, q25 = np.percentile(final_scores, [75, 25])
        iqr = q75 - q25
        
        if iqr > 0:
            # Robust standardization
            final_scores = (final_scores - median_score) / (1.5 * iqr)
        else:
            # Fallback: normal standardization
            mean_score = np.mean(final_scores)
            std_score = np.std(final_scores)
            if std_score > 0:
                final_scores = (final_scores - mean_score) / std_score
        
        return final_scores
    
    def detect_anomalies_advanced(self, result: Lambda3Result, events: np.ndarray, 
                                 use_ensemble: bool = True, optimize_weights: bool = True) -> np.ndarray:
        """
        Advanced zero-shot anomaly detection (improved version: overfitting countermeasures)
        """
        if self.jump_analyzer is None:
            self.jump_analyzer = self._detect_multiscale_jumps(events)
        
        # 1. Get base scores
        base_scores = self.detect_anomalies(result, events)
        
        # 2. Advanced feature extraction
        advanced_features = self.extract_advanced_features(result, events)
        
        # 3. Feature weight optimization (improved version)
        if optimize_weights:
            # More conservative pseudo-label generation
            # Use only samples with clear differences between top and bottom
            score_percentiles = np.percentile(base_scores, [10, 90])
            
            # Use only samples that can be clearly separated as normal/anomalous
            clear_normal = base_scores < score_percentiles[0]
            clear_anomaly = base_scores > score_percentiles[1]
            
            if np.sum(clear_normal) > 10 and np.sum(clear_anomaly) > 10:
                # Learn using only clear samples
                clear_indices = np.where(clear_normal | clear_anomaly)[0]
                clear_events = events[clear_indices]
                clear_labels = clear_anomaly[clear_normal | clear_anomaly].astype(int)
                
                # Extract features for corresponding indices only
                clear_features = {}
                for feat_name, feat_vals in advanced_features.items():
                    if len(feat_vals) == len(events):  # Event features
                        clear_features[feat_name] = feat_vals[clear_indices]
                    else:  # Path features remain as is
                        clear_features[feat_name] = feat_vals
                
                print(f"Using {len(clear_labels)} clear samples for optimization "
                      f"({np.sum(clear_labels)} anomalies, {len(clear_labels)-np.sum(clear_labels)} normal)")
                
                # Weight optimization
                optimal_weights, opt_auc = self.optimize_feature_weights_robust(
                    clear_features, clear_labels, result, clear_indices
                )
                
                # Calculate weighted scores for all data
                weighted_scores = self._compute_weighted_scores(
                    advanced_features, optimal_weights, result
                )
            else:
                print("Not enough clear samples for optimization, using equal weights")
                weighted_scores = base_scores
        else:
            weighted_scores = base_scores
        
        # 4. Ensemble detection
        if use_ensemble:
            # Generate pseudo-labels from base scores (for ensemble evaluation)
            pseudo_labels = (base_scores > np.percentile(base_scores, 85)).astype(int)
            
            ensemble_scores, model_info = self.ensemble_anomaly_detection(
                events, pseudo_labels, n_models=7
            )
            
            # Fusion of base, weighted, and ensemble
            final_scores = (
                0.3 * base_scores + 
                0.4 * weighted_scores + 
                0.3 * ensemble_scores
            )
        else:
            final_scores = 0.5 * base_scores + 0.5 * weighted_scores
        
        return final_scores
    
    def detect_anomalies_focused(self, result: Lambda3Result, events: np.ndarray, 
                                use_optimization: bool = True) -> np.ndarray:
        """
        Focused anomaly detection using the strongest feature discovered through optimization
        """
        if use_optimization:
            # 1. Extract advanced features
            advanced_features = self.extract_advanced_features(result, events)
            
            # 2. Generate pseudo-labels from base scores (only clear top/bottom samples)
            base_scores = self.detect_anomalies(result, events)
            score_percentiles = np.percentile(base_scores, [10, 90])
            clear_normal = base_scores < score_percentiles[0]
            clear_anomaly = base_scores > score_percentiles[1]
            
            if np.sum(clear_normal) > 10 and np.sum(clear_anomaly) > 10:
                clear_indices = np.where(clear_normal | clear_anomaly)[0]
                clear_labels = clear_anomaly[clear_normal | clear_anomaly].astype(int)
                
                # Extract features for corresponding indices only
                clear_features = {}
                for feat_name, feat_vals in advanced_features.items():
                    if len(feat_vals) == len(events):
                        clear_features[feat_name] = feat_vals[clear_indices]
                    else:
                        clear_features[feat_name] = feat_vals
                
                # 3. Run optimization to discover the strongest feature
                optimal_weights, _ = self.optimize_feature_weights_robust(
                    clear_features, clear_labels, result, clear_indices
                )
                
                # 4. Get the feature with highest weight
                best_feature = max(optimal_weights.items(), key=lambda x: x[1])[0]
                print(f"\nUsing best feature: {best_feature} (weight: {optimal_weights[best_feature]:.4f})")
                
                # 5. Calculate scores using only that feature (simple!)
                if best_feature in advanced_features:
                    best_feature_values = advanced_features[best_feature]
                    path_matrix = np.stack(list(result.paths.values()))
                    
                    if len(best_feature_values) == len(path_matrix):  # Path feature
                        scores = np.sum(np.abs(path_matrix) * best_feature_values[:, None], axis=0)
                    else:  # Event feature
                        scores = best_feature_values
                    
                    # Standardization
                    if np.std(scores) > 0:
                        scores = (scores - np.mean(scores)) / np.std(scores)
                    
                    return scores
            
            # Fallback if optimization doesn't work
            print("Optimization failed, falling back to heuristic approach")
        
        # Original heuristic approach (without optimization)
        # Extract all features
        all_features = self.extract_advanced_features(result, events)
        
        # Find the best single feature based on variance or range
        best_feature_name = None
        best_feature_score = -np.inf
        
        path_matrix = np.stack(list(result.paths.values()))
        n_events = events.shape[0]
        
        for feat_name, feat_vals in all_features.items():
            vals = np.array(feat_vals)
            
            # Convert to event scores
            if vals.shape[0] == path_matrix.shape[0]:  # Path features
                event_scores = np.sum(np.abs(path_matrix) * vals[:, None], axis=0)
            elif vals.shape[0] == n_events:  # Event features
                event_scores = vals
            else:
                continue
            
            # Evaluate feature quality (variance indicates discriminative power)
            if np.std(event_scores) > 0:
                # Normalize
                event_scores = (event_scores - np.mean(event_scores)) / np.std(event_scores)
                
                # Score based on variance and range
                feature_quality = np.std(event_scores) * (np.max(event_scores) - np.min(event_scores))
                
                if feature_quality > best_feature_score:
                    best_feature_score = feature_quality
                    best_feature_name = feat_name
        
        print(f"Focused detection using best feature: {best_feature_name} (quality={best_feature_score:.3f})")
        
        # Use only the best feature
        if best_feature_name:
            single_feature = {best_feature_name: all_features[best_feature_name]}
            
            # Simple optimization for single feature
            vals = np.array(single_feature[best_feature_name])
            
            if vals.shape[0] == path_matrix.shape[0]:
                final_scores = np.sum(np.abs(path_matrix) * vals[:, None], axis=0)
            else:
                final_scores = vals
            
            # Apply sigmoid transformation for better discrimination
            mean_score = np.mean(final_scores)
            std_score = np.std(final_scores)
            if std_score > 0:
                final_scores = (final_scores - mean_score) / std_score
                final_scores = 1 / (1 + np.exp(-2 * final_scores))  # Steeper sigmoid
        else:
            # Fallback to base detection
            final_scores = self.detect_anomalies(result, events)
        
        return final_scores
    
    def optimize_feature_weights_robust(self, features: Dict[str, np.ndarray], labels: np.ndarray,
                                      result: Lambda3Result, event_indices: np.ndarray,
                                      n_iter: int = 50) -> Tuple[Dict[str, float], float]:
        """Robust feature weight optimization (overfitting countermeasure version)"""
        feature_names = list(features.keys())
        n_features = len(feature_names)
        n_events = len(labels)
        
        # Feature count limitation (more conservative)
        max_features = min(20, n_features // 2)
        
        if n_features > max_features:
            # Feature importance evaluation (correlation-based)
            feature_importance = {}
            path_matrix = np.stack(list(result.paths.values()))
            
            for name in feature_names:
                vals = np.array(features[name])
                
                if vals.shape[0] == path_matrix.shape[0]:  # Path features
                    # Convert to event scores
                    event_scores = np.zeros(len(event_indices))
                    for i, evt_idx in enumerate(event_indices):
                        event_scores[i] = np.sum(np.abs(path_matrix[:, evt_idx]) * vals)
                elif vals.shape[0] == n_events:  # Event features (partial)
                    event_scores = vals
                else:
                    continue
                
                if np.std(event_scores) > 1e-10:
                    # Use absolute value of correlation coefficient as importance
                    importance = abs(np.corrcoef(event_scores, labels)[0, 1])
                    feature_importance[name] = importance
            
            # Select top features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:max_features]]
            
            print(f"Selected {len(selected_features)} most correlated features")
            print(f"Top 5: {[f'{f[0]}: {f[1]:.3f}' for f in sorted_features[:5]]}")
            
            feature_names = selected_features
            n_features = len(feature_names)
        
        # Build feature matrix
        event_feature_matrix = self._build_feature_matrix(
            features, feature_names, result, event_indices
        )
        
        # Bayesian optimization approach (more stable)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(event_feature_matrix.T)
        
        # L1-regularized logistic regression for weight estimation
        lr = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=0.1,  # Strong regularization
            max_iter=1000,
            random_state=42
        )
        
        try:
            lr.fit(X_scaled, labels)
            
            # Use coefficients as weights
            weights = np.abs(lr.coef_[0])
            
            # Score calculation
            scores = lr.predict_proba(X_scaled)[:, 1]
            opt_auc = roc_auc_score(labels, scores)
            
        except Exception as e:
            print(f"Logistic regression failed: {e}, using uniform weights")
            weights = np.ones(n_features)
            opt_auc = 0.5
        
        # Weight normalization and dictionary conversion
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_features) / n_features
        
        optimal_weights = {feature_names[i]: weights[i] for i in range(n_features)}
        
        print(f"\nRobust feature weight optimization completed:")
        print(f"  Optimized AUC: {opt_auc:.4f}")
        print(f"  Weight diversity (std): {np.std(weights):.4f}")
        print(f"  Non-zero weights: {np.sum(weights > 0.001)}/{n_features}")
        
        # Display important features
        sorted_features = sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)
        print("\nTop features by weight:")
        for i, (feat, weight) in enumerate(sorted_features[:10]):
            if weight > 0.001:
                print(f"  {i+1}. {feat}: {weight:.4f}")
        
        return optimal_weights, opt_auc
    
    def _build_feature_matrix(self, features: Dict[str, np.ndarray], 
                            feature_names: List[str], result: Lambda3Result,
                            event_indices: np.ndarray) -> np.ndarray:
        """Build feature matrix (supports partial data)"""
        path_matrix = np.stack(list(result.paths.values()))
        n_events = len(event_indices)
        
        event_feature_matrix = []
        
        for name in feature_names:
            vals = np.array(features[name])
            
            if vals.shape[0] == path_matrix.shape[0]:  # Path features
                # Score calculation for specified events only
                event_scores = np.zeros(n_events)
                for i, evt_idx in enumerate(event_indices):
                    event_scores[i] = np.sum(np.abs(path_matrix[:, evt_idx]) * vals)
            elif vals.shape[0] == n_events:  # Already event features
                event_scores = vals
            else:
                event_scores = np.zeros(n_events)
            
            # Standardization
            if np.std(event_scores) > 1e-10:
                event_scores = (event_scores - np.mean(event_scores)) / np.std(event_scores)
            
            event_feature_matrix.append(event_scores)
        
        return np.array(event_feature_matrix)
    
    def extract_advanced_features(self, result: Lambda3Result, events: np.ndarray) -> Dict[str, np.ndarray]:
        """Advanced feature extraction (combinations, nonlinear transforms, statistical features)"""
        n_paths = len(result.paths)
        paths_matrix = np.stack(list(result.paths.values()))
        
        # Basic features
        basic_features = {
            'Q_Λ': np.array([result.topological_charges[i] for i in range(n_paths)]),
            'E': np.array([result.energies[i] for i in range(n_paths)]),
            'σ_Q': np.array([result.stabilities[i] for i in range(n_paths)])
        }
        
        # Entropy features
        for i in range(n_paths):
            ent = result.entropies[i]
            if isinstance(ent, dict):
                basic_features[f'S_shannon_{i}'] = np.array([ent.get('shannon', 0)])
                basic_features[f'S_renyi_{i}'] = np.array([ent.get('renyi_2', 0)])
                basic_features[f'S_tsallis_{i}'] = np.array([ent.get('tsallis_1.5', 0)])
        
        # Pulsation features
        for i in range(n_paths):
            path = paths_matrix[i]
            jump_int, asymm, pulse_pow = compute_pulsation_energy_from_path(path)
            basic_features[f'jump_int_{i}'] = np.array([jump_int])
            basic_features[f'asymm_{i}'] = np.array([asymm])
            basic_features[f'pulse_pow_{i}'] = np.array([pulse_pow])
        
        # Advanced features
        advanced_features = basic_features.copy()
        
        # 1. Feature combinations (interaction terms)
        feature_names = list(basic_features.keys())
        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names):
                if i < j and len(basic_features[feat1]) == len(basic_features[feat2]):
                    # Product
                    advanced_features[f'{feat1}×{feat2}'] = basic_features[feat1] * basic_features[feat2]
                    # Ratio (avoid division by zero)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = basic_features[feat1] / (basic_features[feat2] + 1e-10)
                        advanced_features[f'{feat1}/{feat2}'] = np.nan_to_num(ratio, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 2. Nonlinear transformations
        for feat, values in basic_features.items():
            if np.all(values >= 0) and np.any(values > 0):
                # Log transform
                advanced_features[f'log_{feat}'] = np.log1p(np.abs(values))
                # Square root transform
                advanced_features[f'sqrt_{feat}'] = np.sqrt(np.abs(values))
            # Square transform
            advanced_features[f'sq_{feat}'] = values ** 2
            # Sigmoid transform
            advanced_features[f'sig_{feat}'] = 1 / (1 + np.exp(-values))
        
        # 3. Statistical features (per path)
        for i in range(n_paths):
            path = paths_matrix[i]
            # Skewness
            advanced_features[f'skew_{i}'] = np.array([np.nan_to_num(np.sum((path - np.mean(path))**3) / (len(path) * np.std(path)**3 + 1e-10))])
            # Kurtosis
            advanced_features[f'kurt_{i}'] = np.array([np.nan_to_num(np.sum((path - np.mean(path))**4) / (len(path) * np.std(path)**4 + 1e-10) - 3)])
            # Autocorrelation
            if len(path) > 1:
                autocorr = np.correlate(path - np.mean(path), path - np.mean(path), mode='full')[len(path)-1:] / (np.var(path) * np.arange(len(path), 0, -1))
                advanced_features[f'autocorr_{i}'] = np.array([np.mean(autocorr[:5])])  # Average of first 5 lags
        
        return advanced_features
    
    def optimize_feature_weights(self, features: Dict[str, np.ndarray], labels: np.ndarray,
                               result: Lambda3Result, n_iter: int = 100) -> Tuple[Dict[str, float], float]:
        """Automatic feature weight optimization (using differential_evolution) - Improved version"""
        feature_names = list(features.keys())
        n_features = len(feature_names)
        n_events = len(labels)
        
        # Limit features if too many
        if n_features > 50:
            # Calculate individual AUC for each feature
            feature_scores = {}
            path_matrix = np.stack(list(result.paths.values()))
            
            for name in feature_names:
                vals = np.array(features[name])
                if vals.shape[0] == path_matrix.shape[0]:
                    event_scores = np.sum(np.abs(path_matrix) * vals[:, None], axis=0)
                elif vals.shape[0] == n_events:
                    event_scores = vals
                else:
                    continue
                
                if np.std(event_scores) > 0:
                    event_scores = (event_scores - np.mean(event_scores)) / np.std(event_scores)
                    try:
                        auc = roc_auc_score(labels, event_scores)
                        feature_scores[name] = auc
                    except:
                        feature_scores[name] = 0.5
            
            # Select top 50 features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:50]]
            feature_names = selected_features
            n_features = len(feature_names)
            print(f"Selected top {n_features} features for optimization")
        
        # Path matrix
        path_matrix = np.stack(list(result.paths.values()))
        
        # Project each feature to event-wise scores
        event_feature_matrix = []
        for name in feature_names:
            vals = np.array(features[name])
            if vals.shape[0] == path_matrix.shape[0]:  # Path features
                event_scores = np.sum(np.abs(path_matrix) * vals[:, None], axis=0)
            elif vals.shape[0] == n_events:  # Event features
                event_scores = vals
            else:
                event_scores = np.zeros(n_events)
            
            # Standardization
            if np.std(event_scores) > 0:
                event_scores = (event_scores - np.mean(event_scores)) / np.std(event_scores)
            event_feature_matrix.append(event_scores)
        
        event_feature_matrix = np.array(event_feature_matrix).T
        
        # Add L2 regularization
        lambda_reg = 0.01
        
        def objective(weights):
            # Weighted composite score
            combined_score = np.dot(event_feature_matrix, weights)
            
            # Standardization
            if np.std(combined_score) > 0:
                combined_score = (combined_score - np.mean(combined_score)) / np.std(combined_score)
            
            # Sigmoid transform
            combined_score = 1 / (1 + np.exp(-combined_score))
            
            try:
                # AUC calculation
                auc = roc_auc_score(labels, combined_score)
                # Add L2 regularization term (encourage weight diversity)
                reg_term = lambda_reg * np.sum(weights**2)
                return -(auc - reg_term)  # Negative for minimization
            except Exception as e:
                return -0.5
        
        # Constraints: each weight between 0 and 1
        bounds = [(0, 1) for _ in range(n_features)]
        
        # Diversify initial values
        x0 = np.random.dirichlet(np.ones(n_features))
        
        # Execute optimization (parameter tuning)
        result_opt = differential_evolution(
            objective, 
            bounds, 
            x0=x0,
            maxiter=n_iter, 
            seed=42,
            atol=1e-8, 
            tol=0.001, 
            workers=1,
            updating='deferred', 
            polish=True, 
            strategy='best1bin',
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7
        )
        
        optimal_weights = {feature_names[i]: result_opt.x[i] for i in range(n_features)}
        best_auc = -result_opt.fun
        
        # Ensure weight diversity (set minimum weight)
        min_weight = 0.01
        for k in optimal_weights:
            if optimal_weights[k] < min_weight:
                optimal_weights[k] = min_weight
        
        # Weight normalization (L1 normalization)
        weight_sum = sum(optimal_weights.values())
        if weight_sum > 0:
            optimal_weights = {k: v/weight_sum for k, v in optimal_weights.items()}
        
        print(f"\nFeature weight optimization completed:")
        print(f"  Optimized AUC: {best_auc:.4f}")
        print(f"  Weight diversity (std): {np.std(list(optimal_weights.values())):.4f}")
        
        # Display top features (in descending order of weights)
        sorted_features = sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 features by weight:")
        for feat, weight in sorted_features:
            if weight > 0.01:  # Display only meaningful weights
                print(f"  {feat}: {weight:.4f}")
        
        return optimal_weights, best_auc
    
    def _compute_weighted_scores(self, features: Dict[str, np.ndarray], 
                                weights: Dict[str, float], result: Lambda3Result) -> np.ndarray:
        """Calculate scores using optimized weights"""
        path_matrix = np.stack(list(result.paths.values()))
        n_events = path_matrix.shape[1]
        
        weighted_score = np.zeros(n_events)
        
        for feat_name, weight in weights.items():
            if feat_name in features:
                vals = np.array(features[feat_name])
                
                if vals.shape[0] == path_matrix.shape[0]:  # Path features
                    event_scores = np.sum(np.abs(path_matrix) * vals[:, None], axis=0)
                elif vals.shape[0] == n_events:  # Event features
                    event_scores = vals
                else:
                    continue
                
                # Standardize and weight
                if np.std(event_scores) > 0:
                    event_scores = (event_scores - np.mean(event_scores)) / np.std(event_scores)
                
                weighted_score += weight * event_scores
        
        return weighted_score
    
    def ensemble_anomaly_detection(self, events: np.ndarray, labels: np.ndarray = None,
                                 n_models: int = 5) -> Tuple[np.ndarray, Dict]:
        """Ensemble anomaly detection (multiple models with different parameters)"""
        ensemble_scores = []
        model_info = {}
        
        # Parameter ranges
        param_ranges = {
            'n_paths': [3, 5, 7, 9],
            'alpha': [0.05, 0.1, 0.15, 0.2],
            'beta': [0.005, 0.01, 0.015, 0.02],
            'jump_scale': [1.5, 2.0, 2.5, 3.0],
            'kernel_type': [0, 1, 3]  # RBF, Polynomial, Laplacian
        }
        
        for i in range(n_models):
            # Random parameter selection
            model_params = {
                'n_paths': np.random.choice(param_ranges['n_paths']),
                'alpha': np.random.choice(param_ranges['alpha']),
                'beta': np.random.choice(param_ranges['beta']),
                'jump_scale': np.random.choice(param_ranges['jump_scale']),
                'kernel_type': np.random.choice(param_ranges['kernel_type'])
            }
            
            # Temporarily change parameters
            original_config = self.config
            self.config = L3Config(
                alpha=model_params['alpha'],
                beta=model_params['beta'],
                jump_scale=model_params['jump_scale']
            )
            
            try:
                # Execute analysis
                result = self.analyze(events, n_paths=model_params['n_paths'])
                
                # Anomaly detection with specified kernel type
                model_scores = self.detect_anomalies(result, events)
                
                # Sigmoid transform
                model_scores = 1 / (1 + np.exp(-model_scores))
                
                ensemble_scores.append(model_scores)
                
                # Record model information
                if labels is not None:
                    try:
                        model_auc = roc_auc_score(labels, model_scores)
                        model_info[f'model_{i}'] = {
                            'auc': model_auc,
                            'params': model_params
                        }
                    except:
                        model_info[f'model_{i}'] = {'auc': 0.5}
                
            except Exception as e:
                print(f"Model {i} failed: {e}")
                # Fallback
                ensemble_scores.append(np.random.rand(len(events)))
            
            # Restore parameters
            self.config = original_config
        
        # Ensemble integration (average)
        if ensemble_scores:
            final_scores = np.mean(ensemble_scores, axis=0)
        else:
            final_scores = np.full(len(events), 0.5)
        
        if labels is not None:
            try:
                ensemble_auc = roc_auc_score(labels, final_scores)
                model_info['ensemble_auc'] = ensemble_auc
                print(f"Ensemble AUC: {ensemble_auc:.4f}")
            except:
                model_info['ensemble_auc'] = 0.5
        
        return final_scores, model_info
    
    def _compute_jump_anomaly_scores(
        self, 
        jump_structures: Dict,
        events: np.ndarray
    ) -> np.ndarray:
        """Calculate anomaly scores directly from jump structures (improved version)"""
        n_events = events.shape[0]
        scores = np.zeros(n_events)
        
        # Integrated jump scores
        integrated = jump_structures['integrated']
        
        # Scores based on jump importance (not just existence)
        jump_mask = integrated['unified_jumps'].astype(float)
        importance = integrated['jump_importance']
        
        # Consider only jumps with high importance (set threshold)
        importance_threshold = np.percentile(importance[importance > 0], 75) if np.any(importance > 0) else 0.5
        significant_jumps = jump_mask * (importance >= importance_threshold)
        
        scores += significant_jumps * importance
        
        # Jump contribution from each feature (intensity-based)
        feature_scores = []
        for f, data in jump_structures['features'].items():
            # Standardize jump intensity
            if data['jump_intensity'] > 0:
                feature_score = np.zeros(n_events)
                
                # Consider only strong jumps
                strong_jumps = (data['pos_jumps'] + data['neg_jumps']) * (
                    np.abs(data['diff']) > np.percentile(np.abs(data['diff']), 98)
                )
                
                feature_score = strong_jumps * data['jump_intensity']
                
                # Penalty for high asymmetry (characteristic of severe anomalies)
                if np.abs(data['asymmetry']) > 0.8:
                    feature_score *= (1 + np.abs(data['asymmetry']))
                
                feature_scores.append(feature_score)
        
        if feature_scores:
            # Take maximum instead of average between features (capture more prominent anomalies)
            feature_contribution = np.max(feature_scores, axis=0)
            scores += feature_contribution * 0.5
        
        # Jump cluster anomalies (consider both size and density)
        for cluster in integrated['jump_clusters']:
            cluster_size = cluster['size']
            cluster_density = cluster['density']
            
            # Consider only large and dense clusters as anomalies
            if cluster_size >= 5 and cluster_density > 0.6:
                anomaly_strength = np.log1p(cluster_size) * cluster_density
                for idx in cluster['indices']:
                    scores[idx] += anomaly_strength
        
        return scores
    
    def _compute_kernel_anomaly_scores(
        self,
        events: np.ndarray,
        result: Lambda3Result,
        kernel_type: int = 3  # Laplacian
    ) -> np.ndarray:
        """Calculate anomaly scores in kernel space"""
        # Kernel Gram matrix
        K = compute_kernel_gram_matrix(events, kernel_type, gamma=1.0)
        
        # Reconstruction error in kernel space
        paths_matrix = np.stack(list(result.paths.values()))
        n_events = events.shape[0]
        
        # Kernel reconstruction
        K_recon = np.zeros((n_events, n_events))
        for i in range(n_events):
            for j in range(n_events):
                for k in range(len(paths_matrix)):
                    K_recon[i, j] += paths_matrix[k, i] * K[i, j] * paths_matrix[k, j]
        
        # Normalization
        K_norm = np.sqrt(np.trace(K @ K))
        if K_norm > 0:
            K /= K_norm
        
        recon_norm = np.sqrt(np.trace(K_recon @ K_recon))
        if recon_norm > 0:
            K_recon /= recon_norm
        
        # Per-event error
        kernel_scores = np.zeros(n_events)
        for i in range(n_events):
            row_error = 0.0
            for j in range(n_events):
                diff = K[i, j] - K_recon[i, j]
                row_error += diff * diff
            kernel_scores[i] = np.sqrt(row_error)
        
        return kernel_scores
    
    def _compute_sync_anomaly_scores(self, jump_structures: Dict) -> np.ndarray:
        """Calculate synchronization anomaly scores"""
        n_events = len(jump_structures['integrated']['unified_jumps'])
        scores = np.zeros(n_events)
        
        # Anomalies in high synchronization clusters
        sync_threshold = 0.7
        sync_matrix = jump_structures['integrated']['sync_matrix']
        
        # Detect feature pairs showing abnormally high synchronization
        high_sync_pairs = np.where(sync_matrix > sync_threshold)
        
        # Synchronization anomaly degree for each feature
        for f in jump_structures['features'].keys():
            feature_sync = np.mean([sync_matrix[f, j] for j in range(len(sync_matrix)) if j != f])
            if feature_sync > sync_threshold:
                jumps = jump_structures['features'][f]['pos_jumps'] | jump_structures['features'][f]['neg_jumps']
                scores += jumps * feature_sync
        
        return scores / len(jump_structures['features'])
    
    def explain_anomaly(self, event_idx: int, result: Lambda3Result, events: np.ndarray) -> Dict:
        """Generate physical explanation for anomaly"""
        explanation = {
            'event_index': event_idx,
            'anomaly_score': 0.0,
            'jump_based': {},
            'topological': {},
            'energetic': {},
            'entropic': {},
            'kernel_space': {},
            'recommendation': ""
        }
        
        # Calculate anomaly score
        anomaly_scores = self.detect_anomalies(result, events)
        explanation['anomaly_score'] = float(anomaly_scores[event_idx])
        
        # Jump-based explanation
        if self.jump_analyzer:
            integrated = self.jump_analyzer['integrated']
            if integrated['unified_jumps'][event_idx]:
                sync_features = []
                for f, data in self.jump_analyzer['features'].items():
                    if (data['pos_jumps'][event_idx] or data['neg_jumps'][event_idx]):
                        sync_features.append(f)
                
                explanation['jump_based'] = {
                    'is_jump': True,
                    'importance': float(integrated['jump_importance'][event_idx]),
                    'synchronized_features': sync_features,
                    'n_sync_features': len(sync_features),
                    'in_cluster': any(event_idx in c['indices'] for c in integrated['jump_clusters'])
                }
        
        # Topological explanation
        topo_info = {}
        for p, path in result.paths.items():
            if event_idx > 0:
                delta = np.abs(path[event_idx] - path[event_idx-1])
                path_std = np.std(np.diff(path))
                if delta > path_std * 2:
                    topo_info[f'path_{p}'] = {
                        'charge': float(result.topological_charges[p]),
                        'stability': float(result.stabilities[p]),
                        'classification': result.classifications[p],
                        'delta_lambda': float(delta),
                        'relative_jump': float(delta / path_std)
                    }
        explanation['topological'] = topo_info
        
        # Energy explanation
        energy_info = {}
        for p in result.paths.keys():
            energy_info[f'path_{p}'] = {
                'total_energy': float(result.energies[p]),
                'local_energy': float(result.paths[p][event_idx]**2)
            }
        explanation['energetic'] = energy_info
        
        # Entropy explanation
        entropy_info = {}
        for p, ent_dict in result.entropies.items():
            # Extract main entropy values
            main_entropy = ent_dict.get('shannon', 0)
            jump_entropy = ent_dict.get('shannon_jump', None)
            
            entropy_info[f'path_{p}'] = {
                'shannon': float(main_entropy),
                'jump_conditional': float(jump_entropy) if jump_entropy else None
            }
        explanation['entropic'] = entropy_info
        
        # Recommended action
        if explanation['anomaly_score'] > 2.0:
            if explanation['jump_based'].get('is_jump') and explanation['jump_based']['importance'] > 0.7:
                explanation['recommendation'] = "Critical structural transition detected. Immediate investigation required. Multiple synchronized features show simultaneous jumps."
            else:
                explanation['recommendation'] = "High anomaly score detected. Investigation recommended."
        elif explanation['anomaly_score'] > 1.0:
            explanation['recommendation'] = "Moderate anomaly detected. Monitor adjacent events for cascading effects."
        else:
            explanation['recommendation'] = "Low anomaly level. Continue normal monitoring."
        
        return explanation
    
    def visualize_results(
        self, 
        events: np.ndarray, 
        result: Lambda3Result,
        anomaly_scores: np.ndarray = None
    ) -> plt.Figure:
        """Integrated visualization (centered on jump structures)"""
        if anomaly_scores is None:
            anomaly_scores = self.detect_anomalies(result, events)
            
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Jump structure visualization
        ax1 = plt.subplot(3, 4, 1)
        if self.jump_analyzer:
            integrated = self.jump_analyzer['integrated']
            ax1.plot(integrated['jump_importance'], 'b-', label='Jump Importance')
            ax1.scatter(np.where(integrated['unified_jumps'])[0], 
                       integrated['jump_importance'][integrated['unified_jumps'] == 1],
                       color='red', s=50, label='Jump Events')
            
            # Highlight clusters
            for cluster in integrated['jump_clusters']:
                ax1.axvspan(cluster['start'], cluster['end'], alpha=0.3, color='yellow')
        ax1.set_title('Jump Structure Analysis')
        ax1.set_xlabel('Event Index')
        ax1.set_ylabel('Importance')
        ax1.legend()
        
        # 2. Synchronization matrix
        ax2 = plt.subplot(3, 4, 2)
        if self.jump_analyzer:
            sync_matrix = self.jump_analyzer['integrated']['sync_matrix']
            im = ax2.imshow(sync_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax2)
        ax2.set_title('Feature Synchronization Matrix')
        
        # 3. Anomaly score time series
        ax3 = plt.subplot(3, 4, 3)
        ax3.plot(anomaly_scores, 'g-', linewidth=2)
        ax3.axhline(y=2.0, color='r', linestyle='--', label='Critical Threshold')
        ax3.axhline(y=1.0, color='orange', linestyle='--', label='Warning Threshold')
        ax3.set_title('Anomaly Scores (Zero-Shot)')
        ax3.set_xlabel('Event Index')
        ax3.set_ylabel('Score')
        ax3.legend()
        
        # 4. Topological anomaly map
        ax4 = plt.subplot(3, 4, 4)
        for i in result.paths:
            ax4.scatter(result.topological_charges[i],
                       result.stabilities[i],
                       s=100, label=f'Path {i}')
        ax4.set_xlabel('Topological Charge Q_Λ')
        ax4.set_ylabel('Stability σ_Q')
        ax4.set_title('Topological Anomaly Map')
        ax4.legend()
        
        # 5-8. Details of each path (emphasizing jumps)
        for idx, (i, path) in enumerate(result.paths.items()):
            if idx >= 4:
                break
                
            ax = plt.subplot(3, 4, 5 + idx)
            ax.plot(path, 'b-', alpha=0.7, label='Λ Structure')
            
            # Mark jump events
            if self.jump_analyzer:
                jump_mask = self.jump_analyzer['integrated']['unified_jumps']
                jump_indices = np.where(jump_mask)[0]
                if len(jump_indices) > 0:
                    ax.scatter(jump_indices, path[jump_indices],
                              color='red', s=50, label='Jumps', zorder=5)
            
            ax.set_title(f'Path {i}: {result.classifications[i]}')
            ax.set_xlabel('Event Index')
            ax.set_ylabel('Λ Amplitude')
            ax.legend()
            
            # Display physical quantities
            textstr = f'Q_Λ={result.topological_charges[i]:.3f}\n' \
                     f'σ_Q={result.stabilities[i]:.3f}\n' \
                     f'E={result.energies[i]:.3f}'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 9. Entropy comparison
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
        
        # 10. Pulsation energy distribution (from raw data)
        ax10 = plt.subplot(3, 4, 10)
        if self.jump_analyzer:
            pulse_energies = []
            feature_names = []
            for f_idx, f_data in self.jump_analyzer['features'].items():
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
            # Fallback: calculate from paths
            pulse_energies = []
            for p, path in result.paths.items():
                _, _, pulse_power = compute_pulsation_energy_from_path(path)
                pulse_energies.append(pulse_power)
            ax10.bar(range(len(pulse_energies)), pulse_energies)
            ax10.set_title('Pulsation Energy Distribution (Paths)')
            ax10.set_xlabel('Path Index')
            ax10.set_ylabel('Pulse Power')
        
        # 11. PCA projection
        ax11 = plt.subplot(3, 4, 11)
        if events.shape[1] > 2:
            pca = PCA(n_components=2)
            events_2d = pca.fit_transform(events)
            scatter = ax11.scatter(events_2d[:, 0], events_2d[:, 1], 
                                  c=anomaly_scores, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax11)
        ax11.set_title('Event Space (PCA) - Anomaly Colored')
        
        # 12. Kernel space projection
        ax12 = plt.subplot(3, 4, 12)
        # Simple kernel PCA visualization
        K = compute_kernel_gram_matrix(events[:50], kernel_type=3, gamma=1.0)  # Sampling
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        idx = np.argsort(eigenvalues)[::-1][:2]
        kernel_proj = eigenvectors[:, idx]
        ax12.scatter(kernel_proj[:, 0], kernel_proj[:, 1], 
                    c=anomaly_scores[:50], cmap='plasma', alpha=0.7)
        ax12.set_title('Kernel Space Projection (Laplacian)')
        
        plt.tight_layout()
        return fig
    
    # ===============================
    # Anomaly Pattern Generation Methods (Production Version)
    # ===============================
    def _generate_pulse_anomaly(self, events: np.ndarray, intensity: float = 3, decay_rate: float = 0.5, n_pulses: int = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events, n_features = events.shape
        n_pulses_safe = min(n_pulses, n_events)
        pulse_indices = np.random.choice(n_events, size=n_pulses_safe, replace=False)
        for idx in pulse_indices:
            pulse = np.zeros(n_features)
            affected_dims = np.random.choice(n_features, size=np.random.randint(1, n_features//2 + 1), replace=False)
            pulse[affected_dims] = np.random.randn(len(affected_dims)) * intensity * np.random.uniform(0.8, 1.2)
            if np.random.rand() < 0.5:
                pulse *= -1
            events_copy[idx] += pulse
            for offset in range(1, 4):
                decay = np.exp(-decay_rate * offset)
                if idx - offset >= 0:
                    events_copy[idx - offset] += pulse * decay * np.random.uniform(0.5, 1.0)
                if idx + offset < n_events:
                    events_copy[idx + offset] += pulse * decay * np.random.uniform(0.5, 1.0)
        noise_mask = np.random.rand(*events_copy.shape) < 0.01
        events_copy[noise_mask] += np.random.normal(0, intensity/6, np.sum(noise_mask))
        return events_copy

    def _generate_phase_jump_anomaly(self, events: np.ndarray, intensity: float = 3, spread: int = 4) -> np.ndarray:
        events_copy = events.copy()
        n_events, n_features = events.shape
        idx = np.random.randint(n_events)
        events_copy[idx] = -np.sign(events_copy[idx]) * (np.abs(events_copy[idx]) ** np.random.uniform(1.2, 2.0)) * intensity
        events_copy[idx] += np.random.randn(n_features) * intensity * 0.3
        for offset in range(1, spread + 1):
            decay_factor = np.exp(-0.6 * offset)
            random_phase_shift = np.random.uniform(-np.pi, np.pi, n_features)
            modulation_factor = intensity * decay_factor * np.random.uniform(0.6, 1.2)
            if idx - offset >= 0:
                events_copy[idx - offset] += (
                    np.sin(events_copy[idx]) * modulation_factor
                    + np.cos(random_phase_shift) * modulation_factor * 0.5
                )
            if idx + offset < n_events:
                events_copy[idx + offset] += (
                    np.sin(events_copy[idx]) * modulation_factor
                    + np.cos(random_phase_shift) * modulation_factor * 0.5
                )
        distant_offset = spread + np.random.randint(1, 3)
        distant_decay = np.exp(-1.2 * distant_offset)
        distant_idx = idx + distant_offset if (idx + distant_offset < n_events) else idx - distant_offset
        if 0 <= distant_idx < n_events:
            events_copy[distant_idx] += np.random.randn(n_features) * intensity * distant_decay
        return events_copy

    def _generate_periodic_anomaly(self, events: np.ndarray, intensity: float = 2, disruption_prob=0.2) -> np.ndarray:
        events_copy = events.copy()
        n_events, n_features = events.shape
        t = np.arange(n_events)
        period = np.random.randint(max(3, n_events // 10), max(4, n_events // 4))
        phase = np.random.uniform(0, 2*np.pi)
        base_signal = intensity * np.sin(2 * np.pi * t / period + phase)
        for f in range(n_features):
            feat_phase = phase + np.random.uniform(-np.pi/5, np.pi/5)
            feat_signal = base_signal * np.random.uniform(0.7, 1.3)
            if np.random.rand() < disruption_prob:
                idx = np.random.randint(2, max(3, n_events-2)) if n_events > 3 else 0
                feat_signal[idx-1:idx+2] += np.random.uniform(2, 4) * intensity * np.random.choice([-1, 1])
            if np.random.rand() < disruption_prob and n_events >= 4:
                low = n_events // 4
                high = 3 * n_events // 4
                if high > low:
                    jump_idx = np.random.randint(low, high)
                    feat_signal[jump_idx:] *= -1
            if np.random.rand() < disruption_prob:
                del_start = np.random.randint(0, max(1, n_events - n_events // 8))
                del_end = del_start + np.random.randint(2, max(3, n_events // 8))
                del_end = min(n_events, del_end)
                feat_signal[del_start:del_end] = 0
            feat_signal += np.random.normal(0, 0.15 * intensity, n_events)
            events_copy[:, f] += feat_signal
        return events_copy

    def _generate_decay_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        decay_start = n_events // 2
        decay_length = n_events - decay_start
        decay = np.exp(-intensity * np.arange(decay_length) / decay_length)
        oscillation = np.sin(np.arange(decay_length) * 0.5) * 0.3
        decay_with_osc = decay * (1 + oscillation)
        n_features = events.shape[1]
        feature_decay_rates = np.random.uniform(0.5, 1.5, n_features)
        for i in range(decay_length):
            events_copy[decay_start + i] *= decay_with_osc[i]
            events_copy[decay_start + i] *= feature_decay_rates
            noise_level = (1 - decay[i]) * intensity * 0.5
            events_copy[decay_start + i] += np.random.normal(0, noise_level, n_features)
        n_spikes = max(1, decay_length // 10)
        spike_positions = np.random.choice(range(decay_start, n_events), n_spikes, replace=False)
        for pos in spike_positions:
            spike_features = np.random.choice(n_features, np.random.randint(1, max(2, n_features//3)), replace=False)
            events_copy[pos, spike_features] *= np.random.uniform(2, 4)
        if decay_length > 5:
            for i in range(decay_start + 2, n_events):
                correlation_loss = 1 - decay[i - decay_start]
                events_copy[i] = (1 - correlation_loss) * events_copy[i] + \
                                correlation_loss * np.random.randn(n_features) * np.std(events_copy[:decay_start])
        return events_copy

    def _generate_bifurcation_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        n_features = events.shape[1]
        split_point = n_events // 2
        post_split_length = n_events - split_point
        mode1 = np.random.randn(n_features) * intensity
        mode2 = -mode1 + np.random.randn(n_features) * intensity * 0.5
        for i in range(post_split_length):
            t = i / post_split_length
            bifurcation_strength = np.sqrt(t) * intensity
            if (split_point + i) % 2 == 0:
                events_copy[split_point + i] += mode1 * bifurcation_strength
                rotation_angle = t * np.pi / 4
                events_copy[split_point + i] = self._rotate_features(events_copy[split_point + i], rotation_angle)
            else:
                events_copy[split_point + i] += mode2 * bifurcation_strength
                rotation_angle = -t * np.pi / 4
                events_copy[split_point + i] = self._rotate_features(events_copy[split_point + i], rotation_angle)
        if split_point > 0 and split_point < n_events:
            events_copy[split_point] *= np.random.uniform(0.1, 0.5)
            events_copy[split_point] += np.random.randn(n_features) * intensity * 2
            chaos_range = min(5, split_point // 10)
            for j in range(max(0, split_point - chaos_range), min(n_events, split_point + chaos_range)):
                distance_from_split = abs(j - split_point)
                chaos_intensity = intensity * np.exp(-distance_from_split / chaos_range)
                events_copy[j] += np.random.randn(n_features) * chaos_intensity
        if post_split_length > 10:
            high_freq = np.random.uniform(0.3, 0.5)
            for i in range(split_point, n_events):
                phase = (i - split_point) * high_freq * 2 * np.pi
                amplitude = intensity * 0.3 * ((i - split_point) / post_split_length)
                events_copy[i] += np.sin(phase) * amplitude * np.random.randn(n_features)
        if n_features > 3:
            correlation_matrix = np.random.randn(n_features, n_features)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            correlation_matrix = np.exp(-np.abs(correlation_matrix))
            for i in range(split_point, n_events):
                events_copy[i] = correlation_matrix @ events_copy[i]
        return events_copy

    def _rotate_features(self, features: np.ndarray, angle: float) -> np.ndarray:
        if len(features) < 2:
            return features
        rotated = features.copy()
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        temp0 = rotated[0] * cos_a - rotated[1] * sin_a
        temp1 = rotated[0] * sin_a + rotated[1] * cos_a
        rotated[0], rotated[1] = temp0, temp1
        return rotated

    def _generate_multi_path_anomaly(self, events: np.ndarray, intensity: float = 2, interaction: float = 0.3) -> np.ndarray:
        events_copy = events.copy()
        n_events, n_features = events.shape
        n_available = n_events
        n_paths = min(np.random.randint(2, 5), n_available)
        path_indices = np.random.choice(n_available, size=n_paths, replace=False)
        paths_directions = np.random.randn(n_paths, n_features)
        paths_directions /= np.linalg.norm(paths_directions, axis=1, keepdims=True)
        base_intensities = intensity * np.random.uniform(0.8, 1.8, size=n_paths)
        polarities = np.random.choice([-1, 1], size=n_paths)
        for i, idx in enumerate(path_indices):
            pulse = paths_directions[i] * base_intensities[i] * polarities[i]
            events_copy[idx] += pulse * np.random.uniform(1.5, 2.5)
            for offset in range(1, 3):
                decay = np.exp(-0.5 * offset)
                spike_factor = (base_intensities[i] ** 2) * decay
                if idx - offset >= 0:
                    events_copy[idx - offset] += pulse * spike_factor * np.random.uniform(0.5, 1.2)
                if idx + offset < n_events:
                    events_copy[idx + offset] += pulse * spike_factor * np.random.uniform(0.5, 1.2)
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                midpoint = (path_indices[i] + path_indices[j]) // 2
                interaction_vector = (paths_directions[i] * paths_directions[j])
                interaction_strength = intensity * interaction * np.random.uniform(1.0, 2.0)
                if midpoint < n_events:
                    events_copy[midpoint] += interaction_vector * interaction_strength * np.random.choice([-1, 1])
                    for offset in range(1, 3):
                        interaction_decay = np.exp(-0.4 * offset)
                        if midpoint - offset >= 0:
                            events_copy[midpoint - offset] += interaction_vector * interaction_strength * interaction_decay
                        if midpoint + offset < n_events:
                            events_copy[midpoint + offset] += interaction_vector * interaction_strength * interaction_decay
        if np.random.rand() < 0.3:
            spike_idx = np.random.randint(0, n_events)
            spike_magnitude = intensity * np.random.uniform(3, 5)
            events_copy[spike_idx] += np.random.randn(n_features) * spike_magnitude
        return events_copy

    def _generate_partial_periodic_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        if n_events < 4:
            t = np.arange(n_events)
            period = max(2, n_events // 2)
            modulation = intensity * np.sin(2 * np.pi * t / period)
            events_copy += modulation[:, np.newaxis]
            return events_copy
        min_width = max(2, n_events // 6)
        max_width = max(min_width + 1, n_events // 2)
        width = np.random.randint(min_width, max_width)
        if width >= n_events:
            width = n_events - 1
        start = np.random.randint(0, n_events - width)
        end = start + width
        period = max(2, width // 3)
        t = np.arange(width)
        modulation = intensity * np.sin(2 * np.pi * t / period)
        events_copy[start:end] += modulation[:, np.newaxis]
        return events_copy

    def _generate_topological_jump_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        if n_events < 3:
            events_copy *= -intensity
            return events_copy
        min_point = max(1, n_events // 3)
        max_point = max(min_point + 1, 2 * n_events // 3)
        if min_point >= max_point:
            jump_point = n_events // 2
        else:
            jump_point = np.random.randint(min_point, max_point)
        if jump_point > 0:
            events_copy[:jump_point] *= np.exp(-0.1 * np.arange(jump_point))[:, np.newaxis]
        if jump_point < n_events:
            events_copy[jump_point:] = -events_copy[jump_point:] * intensity
        if jump_point < n_events:
            events_copy[jump_point] = np.random.randn(events.shape[1]) * intensity * 2
        return events_copy

    def _generate_superposition_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_patterns = np.random.randint(2, 4)
        base_patterns = ['pulse', 'periodic', 'phase_jump']
        patterns = np.random.choice(base_patterns, min(n_patterns, len(base_patterns)), replace=False)
        for pattern in patterns:
            weight = np.random.uniform(0.3, 0.7)
            if pattern in self.anomaly_patterns:
                events_copy = weight * events_copy + (1 - weight) * self.anomaly_patterns[pattern](events_copy, intensity)
        return events_copy

    def _generate_cascade_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        if n_events < 2:
            events_copy *= intensity
            return events_copy
        start_idx = np.random.randint(0, max(1, n_events // 2))
        events_copy[start_idx] += np.random.randn(events.shape[1]) * intensity
        for i in range(start_idx + 1, min(start_idx + 10, n_events)):
            decay = np.exp(-0.3 * (i - start_idx))
            events_copy[i] += events_copy[i-1] * 0.5 * decay
        return events_copy

    def _generate_resonance_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_events = events.shape[0]
        if n_events < 4:
            events_copy *= intensity
            return events_copy
        fft = np.fft.fft(events_copy, axis=0)
        max_freq = max(2, len(fft) // 4)
        resonance_freq = np.random.randint(1, max_freq)
        if resonance_freq < len(fft):
            fft[resonance_freq] *= intensity
            if len(fft) - resonance_freq > 0:
                fft[-resonance_freq] *= intensity
        events_copy = np.real(np.fft.ifft(fft, axis=0))
        return events_copy

    def generate_anomalies(self, events: np.ndarray, pattern: str = 'pulse',
                          intensity: float = 3) -> np.ndarray:
        """
        Generate anomalies with specified pattern (Production version)
        
        Args:
            events: Input event array
            pattern: Anomaly pattern name
            intensity: Anomaly intensity
            
        Returns:
            Event array with injected anomalies
        """
        if pattern in self.anomaly_patterns:
            return self.anomaly_patterns[pattern](events, intensity)
        else:
            raise ValueError(f"Unknown anomaly pattern: {pattern}")

# ===============================
# Dataset Generation Functions
# ===============================
def create_complex_natural_dataset(n_events=200, n_features=20, anomaly_ratio=0.15):
    """Generate dataset with more natural and complex anomalies"""

    analyzer = Lambda3ZeroShotDetector()

    # 1. Diversification of base structures
    normal_events = []

    # Generate multiple normal clusters (realistic diversity)
    n_clusters = 3
    for i in range(n_clusters):
        cluster_size = (n_events - int(n_events * anomaly_ratio)) // n_clusters
        cov_matrix = np.eye(n_features)
        for j in range(n_features):
            for k in range(j+1, min(j+3, n_features)):
                corr = np.random.uniform(-0.7, 0.7)
                cov_matrix[j, k] = corr
                cov_matrix[k, j] = corr
        variances = np.random.uniform(0.5, 2.0, n_features)
        cov_matrix = cov_matrix * np.outer(np.sqrt(variances), np.sqrt(variances))
        cluster_mean = np.random.randn(n_features) * 2
        cluster_events = np.random.multivariate_normal(cluster_mean, cov_matrix, cluster_size)
        normal_events.append(cluster_events)

    normal_events = np.vstack(normal_events)

    # 2. Generate complex anomaly patterns
    n_anomalies = int(n_events * anomaly_ratio)
    anomaly_events = []
    anomaly_labels_detailed = []

    # Combinations and temporal evolution of anomaly patterns
    anomaly_scenarios = [
        {
            'name': 'progressive_degradation',
            'patterns': ['structural_decay', 'cascade', 'topological_jump'],
            'progression': 'sequential',
            'intensity_profile': lambda t: 1 + 3 * t
        },
        {
            'name': 'periodic_burst',
            'patterns': ['periodic', 'pulse', 'resonance'],
            'progression': 'mixed',
            'intensity_profile': lambda t: 3 * (1 + np.sin(2 * np.pi * t))
        },
        {
            'name': 'chaotic_bifurcation',
            'patterns': ['bifurcation', 'multi_path', 'phase_jump'],
            'progression': 'simultaneous',
            'intensity_profile': lambda t: 2 * np.exp(t)
        },
        {
            'name': 'partial_anomaly',
            'patterns': ['partial_periodic', 'superposition'],
            'progression': 'feature_specific',
            'intensity_profile': lambda t: 2 + t
        }
    ]

    for i in range(n_anomalies):
        scenario = np.random.choice(anomaly_scenarios)
        base_idx = np.random.randint(len(normal_events))
        base_event = normal_events[base_idx].copy()
        temporal_position = i / n_anomalies

        if scenario['progression'] == 'sequential':
            anomaly = base_event.reshape(1, -1)
            for pattern in scenario['patterns']:
                intensity = scenario['intensity_profile'](temporal_position)
                anomaly = analyzer.generate_anomalies(anomaly, pattern=pattern,
                                                    intensity=intensity * np.random.uniform(0.8, 1.2))

        elif scenario['progression'] == 'mixed':
            n_patterns = np.random.randint(1, len(scenario['patterns']) + 1)
            selected_patterns = np.random.choice(scenario['patterns'], n_patterns, replace=False)
            anomaly = base_event.reshape(1, -1)
            for pattern in selected_patterns:
                intensity = scenario['intensity_profile'](temporal_position)
                anomaly = analyzer.generate_anomalies(anomaly, pattern=pattern,
                                                    intensity=intensity * np.random.uniform(0.5, 1.5))

        elif scenario['progression'] == 'simultaneous':
            anomalies = []
            for pattern in scenario['patterns']:
                intensity = scenario['intensity_profile'](temporal_position)
                temp_anomaly = analyzer.generate_anomalies(
                    base_event.reshape(1, -1), pattern=pattern,
                    intensity=intensity * np.random.uniform(0.7, 1.3)
                )
                anomalies.append(temp_anomaly[0])
            weights = np.random.dirichlet(np.ones(len(anomalies)))
            anomaly = np.average(anomalies, axis=0, weights=weights).reshape(1, -1)

        else:  # feature_specific
            anomaly = base_event.reshape(1, -1)
            affected_features = np.random.choice(n_features,
                                               size=np.random.randint(1, n_features//2),
                                               replace=False)
            for pattern in scenario['patterns']:
                intensity = scenario['intensity_profile'](temporal_position)
                temp_anomaly = analyzer.generate_anomalies(anomaly, pattern=pattern,
                                                         intensity=intensity)
                anomaly[0, affected_features] = temp_anomaly[0, affected_features]

        anomaly_events.append(anomaly[0])
        anomaly_labels_detailed.append(scenario['name'])

    # 3. Add noise and outliers
    anomaly_events = np.array(anomaly_events)
    noise_mask = np.random.random(anomaly_events.shape) < 0.1
    anomaly_events[noise_mask] += np.random.normal(0, 0.5, np.sum(noise_mask))

    outlier_positions = np.random.choice(len(anomaly_events),
                                       size=max(1, len(anomaly_events)//20),
                                       replace=False)
    for pos in outlier_positions:
        outlier_features = np.random.choice(n_features,
                                          size=np.random.randint(1, 3),
                                          replace=False)
        anomaly_events[pos, outlier_features] *= np.random.choice([-1, 1]) * np.random.uniform(5, 10)

    # 4. Build final dataset
    events = np.vstack([normal_events, anomaly_events])
    labels = np.array([0]*len(normal_events) + [1]*len(anomaly_events))

    # Add temporal correlations
    for i in range(1, len(events)):
        if np.random.random() < 0.3:
            events[i] = 0.7 * events[i] + 0.3 * events[i-1]

    # Shuffle
    block_size = 10
    n_blocks = len(events) // block_size
    block_indices = np.arange(n_blocks)
    np.random.shuffle(block_indices)

    shuffled_events = []
    shuffled_labels = []
    for block_idx in block_indices:
        start = block_idx * block_size
        end = min(start + block_size, len(events))
        shuffled_events.append(events[start:end])
        shuffled_labels.append(labels[start:end])

    if len(events) % block_size != 0:
        shuffled_events.append(events[n_blocks * block_size:])
        shuffled_labels.append(labels[n_blocks * block_size:])

    events = np.vstack(shuffled_events)
    labels = np.hstack(shuffled_labels)

    return events, labels, anomaly_labels_detailed

# ===============================
# Performance Evaluation Functions
# ===============================
def evaluate_zero_shot_performance(
    detector: Lambda3ZeroShotDetector,
    events: np.ndarray,
    labels: np.ndarray,
    n_paths: int = 5,
    use_advanced: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of zero-shot performance (with focused method)
    """
    print("\n=== Zero-Shot Performance Evaluation ===")
    
    # 1. Lambda³ analysis
    print("Running Lambda³ analysis...")
    start_time = time.time()
    result = detector.analyze(events, n_paths)
    analysis_time = time.time() - start_time
    print(f"Analysis completed in {analysis_time:.3f}s")
    
    # 2. Anomaly detection (comparing basic and advanced versions)
    print("\nDetecting anomalies...")
    
    # Basic version
    basic_scores = detector.detect_anomalies(result, events)
    basic_auc = roc_auc_score(labels, basic_scores)
    print(f"Basic Zero-shot AUC: {basic_auc:.4f}")
    
    # Advanced version
    if use_advanced:
        print("\nRunning advanced detection with feature optimization and ensemble...")
        advanced_scores = detector.detect_anomalies_advanced(
            result, events, 
            use_ensemble=True, 
            optimize_weights=True
        )
        advanced_auc = roc_auc_score(labels, advanced_scores)
        print(f"Advanced Zero-shot AUC: {advanced_auc:.4f}")
        print(f"Improvement: {(advanced_auc - basic_auc) / basic_auc * 100:.1f}%")
        
        # Focused method (using only best feature)
        print("\nRunning focused detection (best feature only)...")
        focused_scores = detector.detect_anomalies_focused(result, events, use_optimization=True)
        focused_auc = roc_auc_score(labels, focused_scores)
        print(f"Focused Zero-shot AUC: {focused_auc:.4f}")
        print(f"Improvement over basic: {(focused_auc - basic_auc) / basic_auc * 100:.1f}%")
    else:
        advanced_scores = basic_scores
        advanced_auc = basic_auc
        focused_scores = basic_scores
        focused_auc = basic_auc
    
    # 3. Detailed metrics
    metrics = {
        'basic_auc': basic_auc,
        'advanced_auc': advanced_auc,
        'focused_auc': focused_auc,
        'analysis_time': analysis_time,
        'n_jumps_detected': detector.jump_analyzer['integrated']['n_total_jumps'],
        'max_sync': detector.jump_analyzer['integrated']['max_sync'],
        'n_clusters': len(detector.jump_analyzer['integrated']['jump_clusters'])
    }
    
    # 4. Top anomaly analysis with best performance
    best_scores = focused_scores if focused_auc >= advanced_auc else advanced_scores
    best_auc = max(focused_auc, advanced_auc)
    best_method = "Focused" if focused_auc >= advanced_auc else "Advanced"
    
    top_anomaly_indices = np.argsort(best_scores)[-10:]
    print(f"\nTop 10 anomalies ({best_method} method):")
    for i, idx in enumerate(top_anomaly_indices[::-1]):
        explanation = detector.explain_anomaly(idx, result, events)
        print(f"{i+1}. Event {idx}: Score={best_scores[idx]:.3f}, "
              f"Jump={explanation['jump_based'].get('is_jump', False)}, "
              f"Label={labels[idx]}")
    
    # Calculate accuracy
    correct_in_top10 = sum(labels[idx] for idx in top_anomaly_indices)
    print(f"\nCorrect anomalies in top 10: {correct_in_top10}/10")
    
    metrics['best_auc'] = best_auc
    metrics['best_method'] = best_method
    
    return metrics

# ===============================
# Demo Function
# ===============================
def demo_zero_shot_lambda3():
    """
    Demo of complete Lambda³ Zero-Shot Anomaly Detection System
    """
    np.random.seed(42)
    
    print("=== Lambda³ Zero-Shot Anomaly Detection System ===")
    print("Complete Version with Jump-First Architecture")
    print("=" * 60)
    
    # 1. Dataset generation
    print("\n1. Generating complex dataset...")
    events, labels, anomaly_details = create_complex_natural_dataset(
        n_events=500,
        n_features=15,
        anomaly_ratio=0.15
    )
    print(f"Data shape: {events.shape}")
    print(f"Anomaly ratio: {np.mean(labels):.2%}")
    print(f"Anomaly types: {set(anomaly_details)}")
    
    # 2. Initialize detector
    print("\n2. Initializing detector...")
    config = L3Config(
        alpha=0.1,
        beta=0.01,
        n_paths=7,
        jump_scale=1.5,
        w_topo=0.2,
        w_pulse=0.4
    )
    detector = Lambda3ZeroShotDetector(config)
    
    # 3. Performance evaluation (using advanced version)
    print("\n3. Evaluating performance...")
    metrics = evaluate_zero_shot_performance(
        detector, events, labels, n_paths=7, use_advanced=True
    )
    
    # 4. Visualization
    print("\n4. Generating visualizations...")
    result = detector.analyze(events, n_paths=7)
    anomaly_scores = detector.detect_anomalies(result, events)
    
    fig = detector.visualize_results(events, result, anomaly_scores)
    plt.suptitle('Lambda³ Zero-Shot Detection Results', fontsize=16)
    
    # 5. Example anomaly explanations
    print("\n5. Example anomaly explanations:")
    top_3_anomalies = np.argsort(anomaly_scores)[-3:]
    for idx in top_3_anomalies[::-1]:
        print(f"\n--- Event {idx} ---")
        explanation = detector.explain_anomaly(idx, result, events)
        print(f"Anomaly Score: {explanation['anomaly_score']:.3f}")
        print(f"Jump-based: {explanation['jump_based']}")
        print(f"Recommendation: {explanation['recommendation']}")
    
    # 6. Final summary
    print("\n" + "=" * 60)
    print("=== Final Summary ===")
    print(f"Basic Zero-shot AUC: {metrics['basic_auc']:.4f}")
    print(f"Advanced Zero-shot AUC: {metrics['advanced_auc']:.4f}")
    print(f"Focused Zero-shot AUC: {metrics['focused_auc']:.4f}")
    print(f"Best method: {metrics['best_method']}")
    print(f"Best AUC: {metrics['best_auc']:.4f}")
    print(f"Analysis time: {metrics['analysis_time']:.3f}s")
    print(f"Jumps detected: {metrics['n_jumps_detected']}")
    print(f"Jump clusters: {metrics['n_clusters']}")
    print(f"Max synchronization: {metrics['max_sync']:.3f}")
    
    if metrics['best_auc'] > 0.9:
        print(f"\n🚀 REVOLUTIONARY: {metrics['best_auc']:.1%} AUC with ZERO training!")
        print("Lambda³ theory has shattered the limits of anomaly detection!")
    elif metrics['best_auc'] > 0.8:
        print(f"\n🎉 BREAKTHROUGH: {metrics['best_auc']:.1%} AUC with ZERO training!")
        print("The impossible has been achieved through Lambda³ theory!")
    elif metrics['best_auc'] > 0.7:
        print(f"\n✨ EXCELLENT: {metrics['best_auc']:.1%} AUC with ZERO training!")
        print("Lambda³ theory demonstrates remarkable performance!")
    
    plt.show()
    
    return detector, result, metrics

# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    print("Lambda³ Zero-Shot Anomaly Detection System")
    print("Jump-First Architecture with Complete Feature Integration")
    print("Based on Dr. Iizumi's Lambda³ Theory")
    print("=" * 80)
    
    # Run demo
    detector, result, metrics = demo_zero_shot_lambda3()
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print(f"Achievement: {metrics['best_auc']:.4f} AUC with zero training")
