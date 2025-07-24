"""
Lambda³ MD Anomaly Detection System - Direct Structure Tensor Version
For Molecular Dynamics trajectory analysis (e.g., Lysozyme)
Author: Modified for MD analysis with direct Λ construction
"""

import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy as scipy_entropy
from scipy import stats
from numba import jit, njit, prange
import numba

warnings.filterwarnings('ignore')

# ===============================
# MD-specific Global Constants  
# ===============================

DELTA_PERCENTILE = 94.0          # Percentile threshold for structural jumps
LOCAL_WINDOW_SIZE = 15           # Window size for local statistics
LOCAL_JUMP_PERCENTILE = 91.0     # Percentile threshold for local jumps
WINDOW_SIZE = 30                 # General-purpose window size

# Multi-scale jump detection parameters for MD
MULTI_SCALE_WINDOWS = [5, 10, 20, 50, 100]  # Different time scales in MD
MULTI_SCALE_PERCENTILES = [75.0, 80.0, 85.0, 90.0, 93.0, 95.0]

# MD-specific parameters
CONTACT_CUTOFF = 8.0  # Angstroms for contact map
RMSD_JUMP_THRESHOLD = 2.0  # Angstroms for significant structural change
DIHEDRAL_JUMP_THRESHOLD = 30.0  # Degrees for significant dihedral change

# ===============================
# Adaptive Parameter Functions for MD
# ===============================

def compute_md_adaptive_window_size(
    features: Dict[str, np.ndarray],
    trajectory_length: int,
    base_window: int = 30,
    min_window: int = 10,
    max_window: int = None
) -> Dict[str, any]:
    """
    MDトラジェクトリ用の適応的ウィンドウサイズ計算
    構造変化の時間スケールに応じて動的に調整
    """
    
    if max_window is None:
        max_window = max(100, min(trajectory_length // 10, 500))
    
    # 1. RMSD変動の解析
    rmsd_volatility = 0.0
    if 'rmsd' in features:
        rmsd = features['rmsd']
        rmsd_std = np.std(rmsd)
        rmsd_mean = np.mean(rmsd)
        rmsd_volatility = rmsd_std / (rmsd_mean + 1e-10)
    
    # 2. 接触変化の頻度
    contact_change_rate = 0.0
    if 'contact_changes' in features:
        changes = features['contact_changes']
        contact_change_rate = np.mean(changes > np.percentile(changes, 75))
    
    # 3. 構造の周期性（PC空間での解析）
    periodicity = 0.0
    if 'pc_projections' in features:
        pc1 = features['pc_projections'][:, 0]
        fft = np.fft.fft(pc1)
        fft_abs = np.abs(fft[1:len(fft)//2])
        if len(fft_abs) > 0:
            periodicity = np.max(fft_abs) / (np.mean(fft_abs) + 1e-10)
    
    # 4. 局所的な構造安定性（Rgの変動）
    rg_stability = 1.0
    if 'radius_of_gyration' in features:
        rg = features['radius_of_gyration']
        local_windows = []
        window_size = min(50, trajectory_length // 20)
        for i in range(0, len(rg) - window_size, window_size // 2):
            local_windows.append(np.std(rg[i:i+window_size]))
        if local_windows:
            rg_stability = np.std(local_windows) / (np.mean(local_windows) + 1e-10)
    
    # === ウィンドウサイズの計算 ===
    scale_factor = 1.0
    
    # RMSD変動が大きい → 小さいウィンドウ（速い変化を捉える）
    if rmsd_volatility > 0.5:
        scale_factor *= 0.7
    elif rmsd_volatility < 0.1:
        scale_factor *= 1.5
    
    # 接触変化が頻繁 → 小さいウィンドウ
    if contact_change_rate > 0.3:
        scale_factor *= 0.8
    elif contact_change_rate < 0.05:
        scale_factor *= 1.3
    
    # 周期性が強い → 周期に合わせたウィンドウ
    if periodicity > 5.0:
        # 周期を推定してウィンドウサイズに反映
        scale_factor *= 1.2
    
    # 構造が不安定 → 適応的なウィンドウ
    if rg_stability > 1.0:
        scale_factor *= 0.9
    
    # === 用途別のウィンドウサイズ ===
    
    # 局所統計量用
    local_window = int(base_window * scale_factor)
    local_window = np.clip(local_window, min_window, max_window)
    
    # ジャンプ検出用（より敏感に）
    jump_window = int(local_window * 0.6)
    jump_window = np.clip(jump_window, min_window // 2, max_window // 3)
    
    # エントロピー計算用（より安定に）
    entropy_window = int(local_window * 1.3)
    entropy_window = np.clip(entropy_window, min_window * 2, max_window)
    
    # マルチスケール解析用（MD時間スケールに特化）
    multiscale_windows = []
    # 短時間（原子振動）から長時間（ドメイン運動）まで
    for scale in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        window = int(local_window * scale)
        window = np.clip(window, min_window, max_window)
        multiscale_windows.append(window)
    
    # MD特有：構造転移検出用（中間的なウィンドウ）
    transition_window = int(local_window * 0.8)
    transition_window = np.clip(transition_window, min_window, max_window)
    
    return {
        'local': local_window,
        'jump': jump_window,
        'entropy': entropy_window,
        'transition': transition_window,
        'multiscale': multiscale_windows,
        'md_metrics': {
            'rmsd_volatility': rmsd_volatility,
            'contact_change_rate': contact_change_rate,
            'periodicity': periodicity,
            'rg_stability': rg_stability,
            'scale_factor': scale_factor
        }
    }

def update_md_global_constants(window_sizes: Dict[str, any]):
    """MD用グローバル定数を動的に更新"""
    global LOCAL_WINDOW_SIZE, WINDOW_SIZE, MULTI_SCALE_WINDOWS
    
    LOCAL_WINDOW_SIZE = window_sizes['local']
    WINDOW_SIZE = window_sizes['transition']
    MULTI_SCALE_WINDOWS = window_sizes['multiscale']
    
    print(f"MD-Adaptive window sizes updated:")
    print(f"  LOCAL_WINDOW_SIZE: {LOCAL_WINDOW_SIZE}")
    print(f"  WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"  MULTI_SCALE_WINDOWS: {MULTI_SCALE_WINDOWS}")
    print(f"  Scale factor: {window_sizes['md_metrics']['scale_factor']:.3f}")

# ===============================
# Data Class Definitions
# ===============================

@dataclass
class MDLambda3Result:
    """
    Data class for storing results of MD Lambda³ structural analysis.
    """
    paths: Dict[int, np.ndarray]                   # Structure tensor paths from MD features
    topological_charges: Dict[int, float]          # Topological charge Q_Λ for each path
    stabilities: Dict[int, float]                  # Topological stability σ_Q for each path
    energies: Dict[int, float]                     # Pulsation/energy metrics for each path
    entropies: Dict[int, Dict[str, float]]         # Multi-type entropies
    classifications: Dict[int, str]                # Path-level classification labels
    jump_structures: Optional[Dict] = None         # Structural transition info
    md_features: Optional[Dict] = None             # Original MD features used

@dataclass  
class MDConfig:
    """
    Configuration parameters for MD Lambda³ analysis.
    """
    n_paths: int = 7            # Number of structure tensor paths
    jump_scale: float = 1.5     # Sensitivity of jump detection
    use_union: bool = True      # Whether to use union of jumps across paths
    w_topo: float = 0.3         # Weight for topological anomaly score
    w_pulse: float = 0.2        # Weight for pulsation score
    w_structure: float = 0.3    # Weight for structural features
    w_dynamics: float = 0.2     # Weight for dynamic features
    
    # MD-specific parameters
    use_dihedrals: bool = True  # Use dihedral angles as features
    use_contacts: bool = False   # Use contact maps
    use_rmsd: bool = True       # Use RMSD-based features
    use_rg: bool = True         # Use radius of gyration
    use_sasa: bool = False      # Use solvent accessible surface area (optional)

# ===============================
# MD Feature Extraction
# ===============================
@njit
def compute_phase_angle_path(path: np.ndarray) -> np.ndarray:
    """パスの位相角を計算（-π to π）"""
    n = len(path)
    phases = np.zeros(n-1)
    
    for i in range(n-1):
        phases[i] = np.arctan2(path[i+1], path[i])
    
    return phases

@njit
def unwrap_phase_numba(phases: np.ndarray) -> np.ndarray:
    """位相のアンラップ（2π jumpsを除去）"""
    unwrapped = np.zeros_like(phases)
    unwrapped[0] = phases[0]
    
    for i in range(1, len(phases)):
        diff = phases[i] - phases[i-1]
        
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
            
        unwrapped[i] = unwrapped[i-1] + diff
    
    return unwrapped

@njit
def compute_local_fractal_dimension_1d(series: np.ndarray, window: int = 30) -> np.ndarray:
    """1次元時系列のフラクタル次元を計算（Box-counting法）"""
    n = len(series)
    dims = np.ones(n)
    
    for i in range(window, n - window):
        local = series[i-window:i+window]
        
        # Box-counting
        scales = np.array([2, 4, 8, 16])
        counts = np.zeros(len(scales))
        
        for j in range(len(scales)):
            scale = scales[j]
            boxes = 0
            for k in range(0, len(local)-scale, scale):
                segment = local[k:k+scale]
                if np.max(segment) - np.min(segment) > 1e-10:
                    boxes += 1
            counts[j] = max(boxes, 1)
        
        # Log-log fit for fractal dimension
        if np.max(counts) > np.min(counts):
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            # Simple linear fit
            slope = 0.0
            for j in range(len(log_scales)):
                slope += log_scales[j] * log_counts[j]
            slope = -slope / len(log_scales)
            
            # Manual clip instead of np.clip
            if slope < 0.5:
                dims[i] = 0.5
            elif slope > 2.0:
                dims[i] = 2.0
            else:
                dims[i] = slope
    
    return dims

@njit
def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calculate pairwise distance matrix for a set of coordinates."""
    n_atoms = coords.shape[0]
    dist_matrix = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            # 型を統一（float64に）
            dx = float(coords[i, 0] - coords[j, 0])
            dy = float(coords[i, 1] - coords[j, 1])
            dz = float(coords[i, 2] - coords[j, 2])
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

@njit
def calculate_contact_map(coords: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    """Calculate binary contact map from coordinates."""
    dist_matrix = calculate_distance_matrix(coords)
    contact_map = (dist_matrix < cutoff).astype(np.float64)
    # Zero out diagonal
    for i in range(contact_map.shape[0]):
        contact_map[i, i] = 0
    return contact_map

@njit
def calculate_radius_of_gyration(coords: np.ndarray) -> float:
    """Calculate radius of gyration."""
    # Manual mean calculation for numba compatibility
    n_atoms = coords.shape[0]
    center_x = 0.0
    center_y = 0.0
    center_z = 0.0
    
    for i in range(n_atoms):
        center_x += coords[i, 0]
        center_y += coords[i, 1]
        center_z += coords[i, 2]
    
    center_x /= n_atoms
    center_y /= n_atoms
    center_z /= n_atoms
    
    rg_squared = 0.0
    for i in range(n_atoms):
        dx = coords[i, 0] - center_x
        dy = coords[i, 1] - center_y
        dz = coords[i, 2] - center_z
        rg_squared += dx*dx + dy*dy + dz*dz
    return np.sqrt(rg_squared / n_atoms)

@njit
def calculate_dihedral_angle(p1: np.ndarray, p2: np.ndarray, 
                           p3: np.ndarray, p4: np.ndarray) -> float:
    """Calculate dihedral angle between four points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Normal vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    # Normalize
    n1 = n1 / (np.linalg.norm(n1) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2) + 1e-10)
    
    # Calculate angle
    cos_angle = np.dot(n1, n2)
    # Manual clip for numba compatibility
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    
    angle = np.arccos(cos_angle)
    
    # Determine sign
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
        
    return angle

def extract_md_features(trajectory: np.ndarray, 
                       config: MDConfig,
                       backbone_indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Extract MD-specific features from trajectory.
    
    Args:
        trajectory: Shape (n_frames, n_atoms, 3) - atomic coordinates
        config: MDConfig object
        backbone_indices: Indices of backbone atoms for dihedral calculation
        
    Returns:
        Dictionary of features
    """
    n_frames, n_atoms, _ = trajectory.shape
    features = {}
    
    # 1. Contact-based features
    if config.use_contacts:
        print("  - Calculating contact-based features (memory-optimized)...")
        
        # 巨大な配列は事前に確保しない！
        # 代わりに、軽量な特徴量配列を準備
        # per_atom_contacts: (フレーム数, 原子数) の形状。uint16で十分 (原子数<65535)
        per_atom_contacts = np.zeros((n_frames, n_atoms), dtype=np.uint16)
        contact_changes = np.zeros(n_frames)

        # --- オンザフライ処理 ---
        
        # まず0フレーム目の情報を計算
        # 閾値による二値化でメモリ効率化
        prev_contact_map = (calculate_contact_map(trajectory[0], CONTACT_CUTOFF) > 0.5)
        per_atom_contacts[0] = np.sum(prev_contact_map, axis=1)

        # 1フレーム目からループ開始
        for i in range(1, n_frames):
            if i % 10000 == 0:
                print(f"    Processing frame {i}/{n_frames}...")

            # 現在のフレームのコンタクトマップを計算 (使い終わったらすぐ破棄される)
            current_contact_map = (calculate_contact_map(trajectory[i], CONTACT_CUTOFF) > 0.5)
            
            # 1. 新しい特徴量: 原子ごとの接触数を計算して保存
            per_atom_contacts[i] = np.sum(current_contact_map, axis=1)
            
            # 2. 接触変化: bool配列同士の比較は高速かつ正確
            contact_changes[i] = np.sum(current_contact_map != prev_contact_map)
            
            # 3. 参照マップを更新
            prev_contact_map = current_contact_map

        # メモリを圧迫する contact_vector の代わりに、新しい特徴量を格納
        features['per_atom_contacts'] = per_atom_contacts
        features['contact_changes'] = contact_changes
        
        # メモリ使用量を計算して表示（推定）
        mem_usage_gb = (per_atom_contacts.nbytes + contact_changes.nbytes) / 1e9
        print(f"  - Contact features memory usage: ~{mem_usage_gb:.3f} GB")
    
    # 2. RMSD-based features
    if config.use_rmsd:
        rmsd_values = np.zeros(n_frames)
        local_rmsd = np.zeros(n_frames)
        
        ref_coords = trajectory[0]  # First frame as reference
        
        for i in range(n_frames):
            # Global RMSD
            diff = trajectory[i] - ref_coords
            rmsd_values[i] = np.sqrt(np.mean(diff**2))
            
            # Local RMSD (compared to previous frame)
            if i > 0:
                diff_local = trajectory[i] - trajectory[i-1]
                local_rmsd[i] = np.sqrt(np.mean(diff_local**2))
        
        features['rmsd'] = rmsd_values
        features['local_rmsd'] = local_rmsd
    
    # 3. Radius of gyration
    if config.use_rg:
        rg_values = np.zeros(n_frames)
        for i in range(n_frames):
            rg_values[i] = calculate_radius_of_gyration(trajectory[i])
        features['radius_of_gyration'] = rg_values
    
    # 4. Dihedral angles (if backbone indices provided)
    if config.use_dihedrals and backbone_indices is not None:
        n_dihedrals = len(backbone_indices) - 3
        if n_dihedrals > 0:
            dihedrals = np.zeros((n_frames, n_dihedrals))
            
            for i in range(n_frames):
                for j in range(n_dihedrals):
                    idx1, idx2, idx3, idx4 = backbone_indices[j:j+4]
                    dihedrals[i, j] = calculate_dihedral_angle(
                        trajectory[i, idx1],
                        trajectory[i, idx2],
                        trajectory[i, idx3],
                        trajectory[i, idx4]
                    )
            
            features['dihedrals'] = dihedrals
    
    # 5. Center of mass motion
    com_trajectory = np.mean(trajectory, axis=1)
    com_velocity = np.zeros_like(com_trajectory)
    com_velocity[1:] = com_trajectory[1:] - com_trajectory[:-1]
    features['com_velocity'] = com_velocity
    
    # 6. Principal component projections (simple PCA on coordinates)
    flattened_coords = trajectory.reshape(n_frames, -1)
    mean_coords = np.mean(flattened_coords, axis=0)
    centered_coords = flattened_coords - mean_coords
    
    # Simple PCA: use SVD
    U, S, Vt = np.linalg.svd(centered_coords.T, full_matrices=False)
    pc_projections = centered_coords @ U[:, :10]  # First 10 PCs
    features['pc_projections'] = pc_projections
    
    return features

# ===============================
# Direct Structure Tensor Construction
# ===============================

def construct_lambda_from_md_features(features: Dict[str, np.ndarray], 
                                    n_paths: int = 7) -> np.ndarray:
    """
    Directly construct Lambda structure tensor from MD features.
    
    This is the key difference from the original: we don't solve an inverse problem,
    but directly map physical features to the structure tensor space.
    """
    # Collect all features into a matrix
    feature_list = []
    feature_names = []
    
    for name, feat in features.items():
        if feat.ndim == 1:
            feature_list.append(feat[:, np.newaxis])
            feature_names.append(name)
        elif feat.ndim == 2:
            # For multi-dimensional features, take first few components
            n_components = min(3, feat.shape[1])
            for i in range(n_components):
                feature_list.append(feat[:, i:i+1])
                feature_names.append(f"{name}_{i}")
    
    # Combine all features
    all_features = np.hstack(feature_list)
    n_frames, n_features = all_features.shape
    
    print(f"Constructing Λ from {n_features} features over {n_frames} frames")
    
    # Normalize features
    feature_means = np.mean(all_features, axis=0)
    feature_stds = np.std(all_features, axis=0)
    feature_stds[feature_stds < 1e-10] = 1.0
    normalized_features = (all_features - feature_means) / feature_stds
    
    # Method 1: Direct mapping via nonlinear transformation
    # Each path represents a different "view" or transformation of the features
    Lambda_matrix = np.zeros((n_paths, n_frames))
    
    for p in range(n_paths):
        if p == 0:
            # Path 0: Principal structural mode (weighted sum of all features)
            weights = np.random.randn(n_features)
            weights = weights / np.linalg.norm(weights)
            Lambda_matrix[p] = normalized_features @ weights
            
        elif p == 1:
            # Path 1: Dynamic mode (emphasizes changes)
            if n_frames > 1:
                changes = np.diff(normalized_features, axis=0)
                changes = np.vstack([changes[0:1], changes])
                weights = np.random.randn(n_features)
                weights = weights / np.linalg.norm(weights)
                Lambda_matrix[p] = changes @ weights
            else:
                Lambda_matrix[p] = np.random.randn(n_frames) * 0.1
                
        elif p == 2:
            # Path 2: Nonlinear mode (squared features)
            weights = np.random.randn(n_features)
            weights = weights / np.linalg.norm(weights)
            Lambda_matrix[p] = (normalized_features ** 2) @ weights
            
        elif p == 3:
            # Path 3: Oscillatory mode (sin transformation)
            weights = np.random.randn(n_features)
            weights = weights / np.linalg.norm(weights)
            Lambda_matrix[p] = np.sin(normalized_features @ weights * np.pi)
            
        elif p == 4:
            # Path 4: Exponential mode (emphasizes extremes)
            weights = np.random.randn(n_features)
            weights = weights / np.linalg.norm(weights)
            linear_comb = normalized_features @ weights
            Lambda_matrix[p] = np.sign(linear_comb) * (np.exp(np.abs(linear_comb)) - 1)
            
        else:
            # Remaining paths: Random projections with different nonlinearities
            weights = np.random.randn(n_features)
            weights = weights / np.linalg.norm(weights)
            linear_comb = normalized_features @ weights
            
            if p % 2 == 0:
                Lambda_matrix[p] = np.tanh(linear_comb * 2)
            else:
                Lambda_matrix[p] = linear_comb ** 3
    
    # Normalize each path
    for p in range(n_paths):
        path_norm = np.linalg.norm(Lambda_matrix[p])
        if path_norm > 0:
            Lambda_matrix[p] = Lambda_matrix[p] / path_norm
    
    return Lambda_matrix

# ===============================
# JIT-optimized functions (inherited from original)
# ===============================

@njit
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """JIT-compiled difference calculation and threshold computation."""
    diff = np.empty(len(data))
    diff[0] = 0
    for i in range(1, len(data)):
        diff[i] = data[i] - data[i-1]
    
    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, percentile)
    return diff, threshold

@njit
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """JIT-compiled jump detection based on threshold."""
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
    """JIT-compiled local standard deviation calculation."""
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
    """JIT-compiled tension scalar (ρT) calculation."""
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
def compute_pulsation_energy_from_jumps(
    pos_jumps: np.ndarray,
    neg_jumps: np.ndarray,
    diff: np.ndarray,
    rho_t: np.ndarray
) -> Tuple[float, float, float]:
    """Calculate pulsation energy from detected jumps."""
    # Jump intensity
    pos_intensity = 0.0
    neg_intensity = 0.0
    
    for i in range(len(diff)):
        if pos_jumps[i] == 1:
            pos_intensity += diff[i]
        if neg_jumps[i] == 1:
            neg_intensity += np.abs(diff[i])
    
    jump_intensity = pos_intensity + neg_intensity
    
    # Asymmetry
    asymmetry = (pos_intensity - neg_intensity) / (pos_intensity + neg_intensity + 1e-10)
    
    # Pulsation power
    n_jumps = np.sum(pos_jumps) + np.sum(neg_jumps)
    
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

@njit(parallel=True)
def compute_topological_charge_jit(path: np.ndarray, n_segments: int = 10) -> Tuple[float, float]:
    """Topological charge calculation."""
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
def compute_all_entropies_jit(path: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Calculate all entropy measures."""
    abs_path = np.abs(path) + eps
    norm_path = abs_path / np.sum(abs_path)
    
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

# ===============================
# MD Lambda³ Detector
# ===============================
class MDLambda3Detector:
    """
    Lambda³ detector specialized for MD trajectory analysis.
    Uses direct structure tensor construction instead of inverse problem.
    """
    
    def __init__(self, config: MDConfig = None):
        self.config = config or MDConfig()
        self._analysis_cache = {}
        
    def analyze(self, trajectory: np.ndarray, 
                backbone_indices: Optional[np.ndarray] = None) -> MDLambda3Result:
        """
        Analyze MD trajectory using Lambda³ framework.
        
        Args:
            trajectory: (n_frames, n_atoms, 3) array of coordinates
            backbone_indices: Indices of backbone atoms for dihedral calculation
            
        Returns:
            MDLambda3Result object
        """
        n_frames = trajectory.shape[0]
        
        # Extract MD features
        print("Extracting MD features...")
        md_features = extract_md_features(trajectory, self.config, backbone_indices)
        
        # === NEW: Adaptive parameter calculation ===
        print("Computing adaptive parameters...")
        adaptive_params = compute_md_adaptive_window_size(
            md_features, 
            n_frames,
            base_window=30,
            min_window=10
        )
        update_md_global_constants(adaptive_params)
        
        # Adjust config based on MD characteristics
        md_metrics = adaptive_params['md_metrics']
        if md_metrics['rmsd_volatility'] > 0.5:
            self.config.jump_scale = 1.2  # More sensitive for high volatility
        elif md_metrics['rmsd_volatility'] < 0.1:
            self.config.jump_scale = 2.0  # Less sensitive for stable structures
            
        # Construct Lambda structure tensor directly
        print("Constructing Λ structure tensor...")
        Lambda_matrix = construct_lambda_from_md_features(md_features, self.config.n_paths)
        
        # Convert to dictionary format
        paths = {i: Lambda_matrix[i] for i in range(self.config.n_paths)}
        
        # Detect structural jumps
        print("Detecting structural transitions...")
        jump_structures = self._detect_structural_transitions(paths, md_features)
        
        # Compute topological quantities
        print("Computing topological charges...")
        charges, stabilities = self._compute_topology(paths)
        
        # Compute energies
        print("Computing pulsation energies...")
        energies = self._compute_energies(paths, jump_structures)
        
        # Compute entropies
        print("Computing entropies...")
        entropies = self._compute_entropies(paths)
        
        # Classify structures
        classifications = self._classify_md_structures(paths, charges, stabilities, 
                                                     jump_structures, md_features)
        
        result = MDLambda3Result(
            paths=paths,
            topological_charges=charges,
            stabilities=stabilities,
            energies=energies,
            entropies=entropies,
            classifications=classifications,
            jump_structures=jump_structures,
            md_features=md_features
        )
        
        return result
    
    # ===== NEW: Cumulative Drift Methods =====
    
    def analyze_with_cumulative_drift(self, trajectory: np.ndarray, 
                                     backbone_indices: Optional[np.ndarray] = None,
                                     initial_frames: int = 100) -> MDLambda3Result:
        """
        累積ドリフト解析を含む拡張版analyze
        """
        # 通常の解析を実行
        result = self.analyze(trajectory, backbone_indices)
        
        # 初期構造の特徴抽出とΛ構築
        print("Computing initial structure Lambda...")
        initial_features = extract_md_features(
            trajectory[:initial_frames], 
            self.config, 
            backbone_indices
        )
        initial_lambda = construct_lambda_from_md_features(
            initial_features, 
            self.config.n_paths
        )
        
        # 累積ドリフトの計算
        print("Computing cumulative topological drift...")
        drift_metrics = self._compute_cumulative_drift_metrics(
            result.paths, 
            {i: initial_lambda[i] for i in range(self.config.n_paths)},
            trajectory.shape[0]
        )
        
        # 結果に追加
        result.drift_metrics = drift_metrics
        
        return result
    
    def _compute_cumulative_drift_metrics(self, 
                                        current_paths: Dict[int, np.ndarray],
                                        initial_paths: Dict[int, np.ndarray],
                                        n_frames: int) -> Dict:
        """累積ドリフトメトリクスの計算"""
        
        drift_metrics = {
            'cumulative_drift': {},
            'instantaneous_drift': {},
            'fractal_dimensions': {},
            'helical_breaking_score': np.zeros(n_frames),
            'structural_boundaries': {'locations': [], 'strengths': []},
            'integrated_drift': np.zeros(n_frames)
        }
        
        # 各パスのドリフト計算
        for p in range(self.config.n_paths):
            # 累積ドリフト
            cumulative, instantaneous = self._compute_path_drift(
                current_paths[p], 
                initial_paths[p]
            )
            
            drift_metrics['cumulative_drift'][p] = cumulative
            drift_metrics['instantaneous_drift'][p] = instantaneous
            
            # フラクタル次元
            if len(cumulative) > 60:
                fractal_dims = compute_local_fractal_dimension_1d(cumulative, window=30)
            else:
                fractal_dims = np.ones_like(cumulative)
            drift_metrics['fractal_dimensions'][p] = fractal_dims
            
            # ヘリカル破壊検出（パス0,1で重点的に）
            if p < 2:
                helix_score = self._detect_helical_breaking(
                    current_paths[p], 
                    initial_paths[p]
                )
                drift_metrics['helical_breaking_score'] += helix_score * 0.5
        
        # 統合ドリフト（全パスの平均）
        all_drifts = [drift_metrics['cumulative_drift'][p] 
                     for p in range(self.config.n_paths)]
        drift_metrics['integrated_drift'] = np.mean(all_drifts, axis=0)
        
        # 構造境界の検出
        boundaries = self._detect_structural_boundaries(
            drift_metrics['integrated_drift']
        )
        drift_metrics['structural_boundaries'] = boundaries
        
        return drift_metrics
    
    @staticmethod
    @njit
    def _compute_path_drift(current_path: np.ndarray, 
                          initial_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """パスのドリフト計算（JIT最適化）"""
        n = len(current_path)
        cumulative = np.zeros(n)
        instantaneous = np.zeros(n)
        
        # 瞬間ドリフト
        for i in range(n):
            instantaneous[i] = np.abs(current_path[i] - initial_path[i])
        
        # 累積ドリフト
        cumulative[0] = instantaneous[0]
        for i in range(1, n):
            cumulative[i] = cumulative[i-1] + instantaneous[i]
        
        # 正規化
        if cumulative[-1] > 0:
            cumulative = cumulative / cumulative[-1]
            
        return cumulative, instantaneous
    
    def _detect_helical_breaking(self, current_path: np.ndarray, 
                            initial_path: np.ndarray) -> np.ndarray:
        """ヘリカル構造の破壊を検出"""
        n = len(current_path)
        helix_score = np.zeros(n)
        
        # ヘリックスの周期性（3.6残基/回転）
        helical_period = 2 * np.pi / 3.6
        
        # 位相角の計算
        current_phase = compute_phase_angle_path(current_path)
        initial_phase = compute_phase_angle_path(initial_path)
        
        # 初期構造の位相角の長さ
        n_initial = len(initial_phase)
        
        # 周期性の破壊を検出
        for i in range(min(len(current_phase), n_initial)):  # ← 短い方に合わせる
            expected_phase = (i * helical_period) % (2 * np.pi)
            
            current_deviation = np.abs(current_phase[i] - expected_phase)
            initial_deviation = np.abs(initial_phase[i] - expected_phase)
            
            # 初期構造からの逸脱
            phase_drift = np.abs(current_deviation - initial_deviation)
            
            # 累積スコア
            if i + 1 < n:
                helix_score[i + 1] = helix_score[i] * 0.95 + phase_drift
        
        # 初期構造の範囲を超えた部分は、最後の値を使って継続
        if n_initial < len(current_phase):
            last_initial_phase = initial_phase[-1]
            for i in range(n_initial, len(current_phase)):
                expected_phase = (i * helical_period) % (2 * np.pi)
                
                current_deviation = np.abs(current_phase[i] - expected_phase)
                initial_deviation = np.abs(last_initial_phase - expected_phase)  # 最後の値を使用
                
                phase_drift = np.abs(current_deviation - initial_deviation)
                
                if i + 1 < n:
                    helix_score[i + 1] = helix_score[i] * 0.95 + phase_drift
                
        return helix_score
          
    def _detect_structural_boundaries(self, integrated_drift: np.ndarray) -> Dict:
        """構造遷移の境界を検出"""
        # 一階・二階微分
        first_deriv = np.gradient(integrated_drift)
        second_deriv = np.gradient(first_deriv)
        
        # 変化率の閾値
        threshold = np.mean(np.abs(first_deriv)) + 2 * np.std(np.abs(first_deriv))
        
        # ピーク検出
        boundaries = {
            'locations': [],
            'strengths': [],
            'types': []
        }
        
        for i in range(1, len(first_deriv)-1):
            if np.abs(first_deriv[i]) > threshold:
                # 極大値チェック
                if (np.abs(first_deriv[i]) > np.abs(first_deriv[i-1]) and 
                    np.abs(first_deriv[i]) > np.abs(first_deriv[i+1])):
                    
                    boundaries['locations'].append(i)
                    boundaries['strengths'].append(np.abs(first_deriv[i]))
                    
                    # 境界のタイプ
                    if second_deriv[i] > 0:
                        boundaries['types'].append('unfolding')
                    else:
                        boundaries['types'].append('folding')
                        
        return boundaries
    
    def detect_anomalies_with_drift(self, result: MDLambda3Result) -> np.ndarray:
        """累積ドリフトを含む異常検知"""
        # 通常の異常スコア
        base_scores = self.detect_anomalies(result)
        
        if hasattr(result, 'drift_metrics'):
            drift_metrics = result.drift_metrics
            
            # 累積ドリフトベースのスコア
            drift_scores = np.zeros_like(base_scores)
            
            # 1. ヘリカル破壊スコア
            drift_scores += drift_metrics['helical_breaking_score'] * 2.0
            
            # 2. 構造境界でのスコア増幅
            for loc, strength in zip(drift_metrics['structural_boundaries']['locations'],
                                   drift_metrics['structural_boundaries']['strengths']):
                if loc < len(drift_scores):
                    # 境界周辺を強調
                    start = max(0, loc - 50)
                    end = min(len(drift_scores), loc + 50)
                    drift_scores[start:end] += strength * 0.5
            
            # 3. フラクタル次元の異常
            for p, fractal_dims in drift_metrics['fractal_dimensions'].items():
                # 高いフラクタル次元 = 複雑な構造変化
                complexity_score = (fractal_dims - 1.0) * 2.0
                drift_scores += complexity_score * 0.3
            
            # 統合スコア
            combined_scores = base_scores + drift_scores
            
            # 標準化
            combined_scores = (combined_scores - np.mean(combined_scores)) / (np.std(combined_scores) + 1e-10)
            
            return combined_scores
        else:
            return base_scores
    
    def detect_anomalies(self, result: MDLambda3Result) -> np.ndarray:
        """
        Detect anomalies in MD trajectory.
        Automatically includes cumulative drift if available.
        
        Returns:
            Array of anomaly scores for each frame.
        """
        n_frames = len(result.paths[0])
        scores = np.zeros(n_frames)
        
        # 1. Jump-based anomalies (structural transitions)
        if result.jump_structures:
            jump_scores = self._compute_jump_anomaly_scores(result.jump_structures)
            scores += self.config.w_structure * jump_scores
        
        # 2. Topological anomalies
        topo_scores = self._compute_topological_anomaly_scores(result)
        scores += self.config.w_topo * topo_scores
        
        # 3. Energetic anomalies
        energy_scores = self._compute_energy_anomaly_scores(result)
        scores += self.config.w_pulse * energy_scores
        
        # 4. MD-specific anomalies
        md_scores = self._compute_md_specific_anomalies(result)
        scores += self.config.w_dynamics * md_scores
        
        # ===== NEW: Cumulative Drift Anomalies (if available) =====
        if hasattr(result, 'drift_metrics'):
            drift_metrics = result.drift_metrics
            
            # 5. ヘリカル破壊スコア
            if 'helical_breaking_score' in drift_metrics:
                scores += drift_metrics['helical_breaking_score'] * 2.0 * self.config.w_structure
            
            # 6. 構造境界でのスコア増幅
            if 'structural_boundaries' in drift_metrics:
                for loc, strength in zip(drift_metrics['structural_boundaries']['locations'],
                                      drift_metrics['structural_boundaries']['strengths']):
                    if loc < len(scores):
                        # 境界周辺を強調
                        start = max(0, loc - 50)
                        end = min(len(scores), loc + 50)
                        scores[start:end] += strength * 0.5 * self.config.w_topo
            
            # 7. フラクタル次元の異常
            if 'fractal_dimensions' in drift_metrics:
                for p, fractal_dims in drift_metrics['fractal_dimensions'].items():
                    # 高いフラクタル次元 = 複雑な構造変化
                    complexity_score = (fractal_dims - 1.0) * 2.0
                    scores += complexity_score * 0.3 * self.config.w_dynamics
        
        # Standardize
        scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-10)
        
        return scores

    # ===== NEW: マルチスケール異常検知 =====
    def detect_anomalies_multiscale(self, result: MDLambda3Result) -> Dict[str, np.ndarray]:
        """
        マルチスケール異常検知（改良版：MADベース標準化）
        
        Returns:
            Dict with 'global' and 'local' anomaly scores
        """
        n_frames = len(result.paths[0])
        
        # === グローバル異常（全体的な構造変化）===
        global_components = {}
        
        # 1. Jump-based anomalies
        if result.jump_structures:
            global_components['jump'] = self._compute_jump_anomaly_scores(result.jump_structures)
        
        # 2. Topological anomalies
        global_components['topo'] = self._compute_topological_anomaly_scores(result)
        
        # 3. Energetic anomalies
        global_components['energy'] = self._compute_energy_anomaly_scores(result)
        
        # 4. MD-specific anomalies (RMSD, Rg)
        global_components['md'] = self._compute_md_specific_anomalies(result)
        
        # グローバルスコアの統合（コンポーネント別標準化）
        global_scores = self._integrate_scores_with_mad(global_components, {
            'jump': self.config.w_structure,
            'topo': self.config.w_topo,
            'energy': self.config.w_pulse,
            'md': self.config.w_dynamics
        })
        
        # === ローカル異常（累積ドリフトベース）===
        local_components = {}
        
        if hasattr(result, 'drift_metrics'):
            drift_metrics = result.drift_metrics
            
            # ヘリカル破壊
            if 'helical_breaking_score' in drift_metrics:
                # 初期部分を除外してスコアリング（baselineが低くなる問題を回避）
                helix_score = drift_metrics['helical_breaking_score'].copy()
                # 初期1000フレームの平均を基準にする
                if len(helix_score) > 1000:
                    baseline_mean = np.mean(helix_score[:1000])
                    helix_score = helix_score - baseline_mean  # ベースライン補正
                local_components['helical'] = helix_score
            
            # 構造境界
            if 'structural_boundaries' in drift_metrics:
                boundary_scores = np.zeros(n_frames)
                for loc, strength in zip(drift_metrics['structural_boundaries']['locations'],
                                      drift_metrics['structural_boundaries']['strengths']):
                    if loc < n_frames:
                        start = max(0, loc - 50)
                        end = min(n_frames, loc + 50)
                        window = np.exp(-0.5 * ((np.arange(start, end) - loc) / 20) ** 2)
                        boundary_scores[start:end] += strength * window
                local_components['boundary'] = boundary_scores
            
            # フラクタル次元
            if 'fractal_dimensions' in drift_metrics:
                fractal_score = np.zeros(n_frames)
                for p, fractal_dims in drift_metrics['fractal_dimensions'].items():
                    # 1.0からの偏差（1.0が正常）
                    complexity = np.abs(fractal_dims - 1.0)
                    fractal_score += complexity
                local_components['fractal'] = fractal_score / len(drift_metrics['fractal_dimensions'])
        
        # ローカルスコアの統合（コンポーネント別標準化）
        if local_components:
            local_scores = self._integrate_scores_with_mad(local_components, {
                'helical': 2.0,
                'boundary': 1.5,
                'fractal': 0.5
            })
        else:
            local_scores = np.zeros(n_frames)
        
        return {
            'global': global_scores,
            'local': local_scores
        }

    def _integrate_scores_with_mad(self, component_scores: Dict[str, np.ndarray], 
                                  weights: Dict[str, float]) -> np.ndarray:
        """MADベースの標準化を使ったスコア統合"""
        if not component_scores:
            return np.zeros(len(list(component_scores.values())[0]))
        
        n_frames = len(list(component_scores.values())[0])
        integrated = np.zeros(n_frames)
        
        # 各コンポーネントをMADで標準化してから統合
        for name, scores in component_scores.items():
            if name in weights and len(scores) > 0:
                # MAD標準化
                standardized = self._mad_standardize(scores)
                # 重み付けして統合
                integrated += weights[name] * standardized
        
        # 最終的な統合スコアもMAD標準化
        final_scores = self._mad_standardize(integrated)
        
        return final_scores

    def _mad_standardize(self, scores: np.ndarray) -> np.ndarray:
        """Median Absolute Deviationを使った頑健な標準化"""
        # 基本統計量
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        
        # MADベースの標準化
        if mad > 1e-10:
            # 0.6745は正規分布でMADを標準偏差に変換する係数
            standardized = 0.6745 * (scores - median) / mad
        else:
            # フォールバック：四分位範囲を使用
            q75, q25 = np.percentile(scores, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-10:
                standardized = (scores - median) / (1.5 * iqr)
            else:
                # 最終フォールバック：通常の標準化
                mean = np.mean(scores)
                std = np.std(scores)
                if std > 1e-10:
                    standardized = (scores - mean) / std
                else:
                    standardized = scores - median
        
        return standardized
        
    def _detect_structural_transitions(self, paths: Dict[int, np.ndarray], 
                                     md_features: Dict[str, np.ndarray]) -> Dict:
        """Detect structural transitions in MD trajectory."""
        jump_data = {'path_jumps': {}, 'feature_jumps': {}, 'integrated': {}}
        
        # === NEW: Dynamic percentile based on MD metrics ===
        # Adjust detection sensitivity based on structural volatility
        rmsd_std = np.std(md_features.get('rmsd', [0]))
        base_percentile = DELTA_PERCENTILE
        if rmsd_std > 2.0:  # High structural variability
            base_percentile = 90.0  # More sensitive
        elif rmsd_std < 0.5:  # Very stable
            base_percentile = 96.0  # Less sensitive
        
        # Detect jumps in each path
        for p, path in paths.items():
            # Use adaptive percentile
            diff, threshold = calculate_diff_and_threshold(path, base_percentile)
            pos_jumps, neg_jumps = detect_jumps(diff, threshold)
            
            # Local window from adaptive parameters (already updated globally)
            local_std = calculate_local_std(path, LOCAL_WINDOW_SIZE)
            score = np.abs(diff) / (local_std + 1e-8)
            
            # Adaptive local percentile
            local_percentile = LOCAL_JUMP_PERCENTILE - (rmsd_std - 1.0) * 2.0
            local_percentile = np.clip(local_percentile, 85.0, 95.0)
            local_threshold = np.percentile(score, local_percentile)
            local_jumps = (score > local_threshold).astype(int)
            
            rho_t = calculate_rho_t(path, WINDOW_SIZE)
            
            jump_intensity, asymmetry, pulse_power = compute_pulsation_energy_from_jumps(
                pos_jumps, neg_jumps, diff, rho_t
            )
            
            jump_data['path_jumps'][p] = {
                'pos_jumps': pos_jumps,
                'neg_jumps': neg_jumps,
                'local_jumps': local_jumps,
                'rho_t': rho_t,
                'jump_intensity': jump_intensity,
                'asymmetry': asymmetry,
                'pulse_power': pulse_power,
                'percentile_used': base_percentile,  # Store for debugging
                'local_percentile_used': local_percentile
            }
        
        # Detect jumps in MD features with adaptive thresholds
        for feat_name in ['rmsd', 'radius_of_gyration', 'contact_changes']:
            if feat_name in md_features:
                feat_data = md_features[feat_name]
                if feat_data.ndim == 1:
                    # Feature-specific percentile
                    feat_percentile = base_percentile
                    if feat_name == 'contact_changes':
                        # Contact changes are more discrete, use lower percentile
                        feat_percentile = max(85.0, base_percentile - 5.0)
                    
                    diff, threshold = calculate_diff_and_threshold(feat_data, feat_percentile)
                    pos_jumps, neg_jumps = detect_jumps(diff, threshold)
                    
                    jump_data['feature_jumps'][feat_name] = {
                        'pos_jumps': pos_jumps,
                        'neg_jumps': neg_jumps,
                        'percentile_used': feat_percentile
                    }
        
        # Integrate jumps
        n_frames = len(paths[0])
        unified_jumps = np.zeros(n_frames, dtype=bool)
        
        for p_data in jump_data['path_jumps'].values():
            unified_jumps |= (p_data['pos_jumps'] | p_data['neg_jumps']).astype(bool)
        
        for f_data in jump_data['feature_jumps'].values():
            unified_jumps |= (f_data['pos_jumps'] | f_data['neg_jumps']).astype(bool)
        
        jump_data['integrated'] = {
            'unified_jumps': unified_jumps,
            'n_jumps': np.sum(unified_jumps),
            'base_percentile': base_percentile,
            'rmsd_volatility': rmsd_std
        }
        
        return jump_data
    
    def _compute_topology(self, paths: Dict[int, np.ndarray]) -> Tuple[Dict, Dict]:
        """Compute topological charges and stabilities."""
        charges = {}
        stabilities = {}
        
        for i, path in paths.items():
            Q, sigma = compute_topological_charge_jit(path)
            charges[i] = Q
            stabilities[i] = sigma
            
        return charges, stabilities
    
    def _compute_energies(self, paths: Dict[int, np.ndarray], 
                         jump_structures: Dict) -> Dict[int, float]:
        """Compute pulsation energies."""
        energies = {}
        
        for i, path in paths.items():
            basic_energy = np.sum(path**2)
            
            if i in jump_structures['path_jumps']:
                pulse_power = jump_structures['path_jumps'][i]['pulse_power']
                energies[i] = basic_energy + 0.3 * pulse_power
            else:
                energies[i] = basic_energy
                
        return energies
    
    def _compute_entropies(self, paths: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
        """Compute various entropy measures."""
        entropies = {}
        entropy_keys = ["shannon", "renyi_2", "tsallis_1.5", "max", "min", "var"]
        
        for i, path in paths.items():
            all_entropies = compute_all_entropies_jit(path)
            entropy_dict = {}
            for j, key in enumerate(entropy_keys):
                entropy_dict[key] = all_entropies[j]
            entropies[i] = entropy_dict
            
        return entropies
    
    def _classify_md_structures(self, paths: Dict, charges: Dict, stabilities: Dict,
                               jump_structures: Dict, md_features: Dict) -> Dict[int, str]:
        """Classify structural states based on Lambda³ analysis."""
        classifications = {}
        
        for i in paths.keys():
            Q = charges[i]
            sigma = stabilities[i]
            
            # Base classification
            if Q < -0.5:
                base = "Collapsed/Compact"
            elif Q > 0.5:
                base = "Extended/Unfolded"
            else:
                base = "Native/Equilibrium"
            
            # Modifiers
            tags = []
            
            if sigma > 2.5:
                tags.append("Highly Flexible")
            elif sigma < 0.5:
                tags.append("Rigid")
            
            # Check for transitions
            if i in jump_structures['path_jumps']:
                n_jumps = np.sum(jump_structures['path_jumps'][i]['pos_jumps'] + 
                               jump_structures['path_jumps'][i]['neg_jumps'])
                if n_jumps > 5:
                    tags.append("Transitioning")
            
            # MD-specific tags
            if 'rmsd' in md_features:
                avg_rmsd = np.mean(md_features['rmsd'])
                if avg_rmsd > 5.0:
                    tags.append("Large Deviation")
            
            if tags:
                classifications[i] = f"{base} ({', '.join(tags)})"
            else:
                classifications[i] = base
                
        return classifications
    
    def _compute_jump_anomaly_scores(self, jump_structures: Dict) -> np.ndarray:
        """Compute anomaly scores based on structural jumps."""
        unified_jumps = jump_structures['integrated']['unified_jumps']
        scores = unified_jumps.astype(float)
        
        # Weight by jump intensity
        for p, p_data in jump_structures['path_jumps'].items():
            jump_mask = (p_data['pos_jumps'] | p_data['neg_jumps']).astype(bool)
            scores[jump_mask] *= (1 + p_data['jump_intensity'] / 10)
            
        return scores
    
    def _compute_topological_anomaly_scores(self, result: MDLambda3Result) -> np.ndarray:
        """Compute anomaly scores based on topological features."""
        n_frames = len(result.paths[0])
        scores = np.zeros(n_frames)
        
        # Use paths with extreme topological charges
        charges = np.array(list(result.topological_charges.values()))
        extreme_paths = np.where(np.abs(charges) > np.percentile(np.abs(charges), 75))[0]
        
        for p in extreme_paths:
            path = result.paths[p]
            # Local variations in the path
            if n_frames > 1:
                local_var = np.abs(np.diff(path))
                local_var = np.concatenate([[0], local_var])
                scores += local_var * np.abs(charges[p])
                
        return scores
    
    def _compute_energy_anomaly_scores(self, result: MDLambda3Result) -> np.ndarray:
        """Compute anomaly scores based on energetic features."""
        n_frames = len(result.paths[0])
        scores = np.zeros(n_frames)
        
        # Paths with high energy
        energies = np.array(list(result.energies.values()))
        high_energy_paths = np.where(energies > np.percentile(energies, 75))[0]
        
        for p in high_energy_paths:
            path = result.paths[p]
            # Energy concentration
            path_energy = path ** 2
            scores += path_energy * (energies[p] / np.max(energies))
            
        return scores
    
    def _compute_md_specific_anomalies(self, result: MDLambda3Result) -> np.ndarray:
        """Compute MD-specific anomaly scores."""
        n_frames = len(result.paths[0])
        scores = np.zeros(n_frames)
        
        md_features = result.md_features
        
        # RMSD-based anomalies
        if 'rmsd' in md_features:
            rmsd = md_features['rmsd']
            rmsd_threshold = np.mean(rmsd) + 2 * np.std(rmsd)
            scores[rmsd > rmsd_threshold] += 1.0
        
        # Contact change anomalies
        if 'contact_changes' in md_features:
            changes = md_features['contact_changes']
            change_threshold = np.percentile(changes, 95)
            scores[changes > change_threshold] += 1.0
        
        # Radius of gyration anomalies
        if 'radius_of_gyration' in md_features:
            rg = md_features['radius_of_gyration']
            rg_mean = np.mean(rg)
            rg_std = np.std(rg)
            scores[np.abs(rg - rg_mean) > 2 * rg_std] += 0.5
            
        return scores
    
    def visualize_results(self, result: MDLambda3Result, 
                         anomaly_scores: Optional[np.ndarray] = None) -> plt.Figure:
        """Visualize MD Lambda³ analysis results."""
        if anomaly_scores is None:
            anomaly_scores = self.detect_anomalies(result)
            
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Anomaly scores
        ax1 = plt.subplot(4, 3, 1)
        ax1.plot(anomaly_scores, 'g-', linewidth=2)
        ax1.axhline(y=2.0, color='r', linestyle='--', label='Critical')
        ax1.axhline(y=1.0, color='orange', linestyle='--', label='Warning')
        ax1.set_title('Anomaly Scores')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Score')
        ax1.legend()
        
        # 2. RMSD evolution
        ax2 = plt.subplot(4, 3, 2)
        if 'rmsd' in result.md_features:
            ax2.plot(result.md_features['rmsd'], 'b-')
            ax2.set_title('RMSD from Initial Structure')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('RMSD (Å)')
        
        # 3. Radius of gyration
        ax3 = plt.subplot(4, 3, 3)
        if 'radius_of_gyration' in result.md_features:
            ax3.plot(result.md_features['radius_of_gyration'], 'r-')
            ax3.set_title('Radius of Gyration')
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Rg (Å)')
        
        # 4-6. First 3 Lambda paths
        for i in range(3):
            ax = plt.subplot(4, 3, 4 + i)
            if i in result.paths:
                ax.plot(result.paths[i], alpha=0.7)
                ax.set_title(f'Λ Path {i}: {result.classifications[i]}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Amplitude')
                
                # Mark jumps
                if result.jump_structures and i in result.jump_structures['path_jumps']:
                    jumps = result.jump_structures['path_jumps'][i]
                    jump_frames = np.where(jumps['pos_jumps'] | jumps['neg_jumps'])[0]
                    if len(jump_frames) > 0:
                        ax.scatter(jump_frames, result.paths[i][jump_frames], 
                                 color='red', s=50, zorder=5)
        
        # 7. Topological charges
        ax7 = plt.subplot(4, 3, 7)
        charges = list(result.topological_charges.values())
        ax7.bar(range(len(charges)), charges)
        ax7.set_title('Topological Charges (Q_Λ)')
        ax7.set_xlabel('Path')
        ax7.set_ylabel('Charge')
        
        # 8. Stabilities
        ax8 = plt.subplot(4, 3, 8)
        stabilities = list(result.stabilities.values())
        ax8.bar(range(len(stabilities)), stabilities)
        ax8.set_title('Path Stabilities (σ_Q)')
        ax8.set_xlabel('Path')
        ax8.set_ylabel('Stability')
        
        # 9. Contact changes
        ax9 = plt.subplot(4, 3, 9)
        if 'contact_changes' in result.md_features:
            ax9.plot(result.md_features['contact_changes'], 'g-')
            ax9.set_title('Contact Map Changes')
            ax9.set_xlabel('Frame')
            ax9.set_ylabel('Change')
        
        # 10. PC projections
        ax10 = plt.subplot(4, 3, 10)
        if 'pc_projections' in result.md_features:
            pc = result.md_features['pc_projections']
            if pc.shape[1] >= 2:
                scatter = ax10.scatter(pc[:, 0], pc[:, 1], c=anomaly_scores, 
                                     cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=ax10)
                ax10.set_title('PC1 vs PC2 (colored by anomaly)')
                ax10.set_xlabel('PC1')
                ax10.set_ylabel('PC2')
        
        # 11. Energy distribution
        ax11 = plt.subplot(4, 3, 11)
        energies = list(result.energies.values())
        ax11.bar(range(len(energies)), energies)
        ax11.set_title('Path Energies')
        ax11.set_xlabel('Path')
        ax11.set_ylabel('Energy')
        
        # 12. Entropy comparison
        ax12 = plt.subplot(4, 3, 12)
        entropy_types = ['shannon', 'renyi_2', 'tsallis_1.5']
        for i, ent_type in enumerate(entropy_types):
            values = [result.entropies[p][ent_type] for p in result.paths]
            ax12.bar(np.arange(len(values)) + i*0.3, values, 0.3, label=ent_type)
        ax12.set_title('Multi-Entropy Comparison')
        ax12.set_xlabel('Path')
        ax12.set_ylabel('Entropy')
        ax12.legend()
        
        plt.tight_layout()
        return fig


# ===============================
# Demo function
# ===============================
def demo_md_analysis():
    """Demo of MD Lambda³ analysis on Lysozyme trajectory."""
    print("=== Lambda³ MD Anomaly Detection Demo ===")
    print("Direct Structure Tensor Construction")
    print("=" * 50)
    
    # Load Lysozyme trajectory
    print("\n1. Loading Lysozyme MD trajectory...")
    try:
        trajectory = np.load('lysozyme_50k_trajectory.npy').astype(np.float64)
        backbone_indices = np.load('lysozyme_50k_backbone_indices.npy')
        print(f"Loaded trajectory: {trajectory.shape}")
        print(f"Trajectory dtype: {trajectory.dtype}") 
        print(f"Loaded backbone indices: {len(backbone_indices)} atoms")
    except FileNotFoundError:
        print("ERROR: Trajectory files not found!")
        print("Please run the create_lysozyme_demo_trajectory() function first.")
        return None, None, None
    
    n_frames, n_atoms, _ = trajectory.shape
    print(f"Trajectory info: {n_frames} frames, {n_atoms} atoms")
    
    # Initialize detector
    print("\n2. Initializing MD Lambda³ detector...")
    config = MDConfig()
    detector = MDLambda3Detector(config)
    
    # Analyze trajectory
    print("\n3. Analyzing trajectory...")
    result = detector.analyze_with_cumulative_drift(trajectory, backbone_indices)
    
    print(f"\nAnalysis complete:")
    print(f"  - {len(result.paths)} structure tensor paths constructed")
    print(f"  - {result.jump_structures['integrated']['n_jumps']} structural transitions detected")
    print(f"  - Adaptive percentile used: {result.jump_structures['integrated']['base_percentile']:.1f}")
    print(f"  - RMSD volatility: {result.jump_structures['integrated']['rmsd_volatility']:.3f}")
    
    # Detect anomalies (Multi-scale version)
    print("\n4. Detecting anomalies (Multi-scale)...")
    multi_scores = detector.detect_anomalies_multiscale(result)
    
    # Show top anomalies for both scales
    print(f"\n  - Global anomaly analysis:")
    global_top10 = np.argsort(multi_scores['global'])[-10:][::-1]
    print(f"    Top 10 global anomalies: {global_top10}")
    
    print(f"\n  - Local anomaly analysis:")
    local_top10 = np.argsort(multi_scores['local'])[-10:][::-1]
    print(f"    Top 10 local anomalies: {local_top10}")
    
    if n_frames >= 50000:
        print(f"\nExpected major events:")
        print(f"  - 5000-7500 (partial_unfold)")
        print(f"  - 15000-17500 (helix_break)")
        print(f"  - 25000-30000 (major_unfold)")
        print(f"  - 35000-37500 (misfold)")
        print(f"  - 42500-45000 (aggregation_prone)")
    else:
        print(f"Expected anomalies around frames 400-450 (unfolding) and 600-650 (misfolding)")
    
    # Additional analysis for Lysozyme
    print("\n5. Lysozyme-specific analysis:")
    if 'rmsd' in result.md_features:
        rmsd = result.md_features['rmsd']
        print(f"  - RMSD range: {np.min(rmsd):.2f} - {np.max(rmsd):.2f} Å")
        print(f"  - Average RMSD: {np.mean(rmsd):.2f} ± {np.std(rmsd):.2f} Å")
    
    if 'radius_of_gyration' in result.md_features:
        rg = result.md_features['radius_of_gyration']
        print(f"  - Rg range: {np.min(rg):.2f} - {np.max(rg):.2f} Å")
        print(f"  - Average Rg: {np.mean(rg):.2f} ± {np.std(rg):.2f} Å")
    
    # Check specific regions (Multi-scale)
    print("\n  - Anomaly scores in key regions (Multi-scale):")
    
    # 5万フレーム版のイベント領域をチェック
    if n_frames >= 50000:
        regions = [
            (0, 1000, 'baseline'),
            (5000, 7500, 'partial_unfold'),
            (15000, 17500, 'helix_break'),
            (25000, 30000, 'major_unfold'),
            (35000, 37500, 'misfold'),
            (42500, 45000, 'aggregation_prone')
        ]
        
        for start, end, name in regions:
            print(f"    Frames {start}-{end} ({name}):")
            print(f"      Global: mean = {np.mean(multi_scores['global'][start:end]):.3f}")
            print(f"      Local:  mean = {np.mean(multi_scores['local'][start:end]):.3f}")
        
        # イベント領域の詳細解析
        print("\n  - Event detection analysis (Multi-scale):")
        events = [
            (5000, 7500, 'partial_unfold'),
            (15000, 17500, 'helix_break'),
            (25000, 30000, 'major_unfold'),
            (35000, 37500, 'misfold'),
            (42500, 45000, 'aggregation_prone')
        ]
        
        for start, end, name in events:
            # グローバルとローカル両方を解析
            global_event = multi_scores['global'][start:end]
            local_event = multi_scores['local'][start:end]
            
            # 最大値の位置
            global_max_idx = start + np.argmax(global_event)
            local_max_idx = start + np.argmax(local_event)
            
            print(f"    {name}:")
            print(f"      Global: max={np.max(global_event):.3f} at frame {global_max_idx}")
            print(f"      Local:  max={np.max(local_event):.3f} at frame {local_max_idx}")
    else:
        # 1000フレーム版用
        print(f"    Frames 400-450 (unfolding):")
        print(f"      Global: mean = {np.mean(multi_scores['global'][400:450]):.3f}")
        print(f"      Local:  mean = {np.mean(multi_scores['local'][400:450]):.3f}")
        
        print(f"    Frames 600-650 (misfolding):")
        print(f"      Global: mean = {np.mean(multi_scores['global'][600:650]):.3f}")
        print(f"      Local:  mean = {np.mean(multi_scores['local'][600:650]):.3f}")
        
        print(f"    Frames 0-100 (baseline):")
        print(f"      Global: mean = {np.mean(multi_scores['global'][0:100]):.3f}")
        print(f"      Local:  mean = {np.mean(multi_scores['local'][0:100]):.3f}")
    
    # Visualize (using global scores for compatibility)
    print("\n6. Generating visualizations...")
    fig = detector.visualize_results(result, multi_scores['global'])  # または 'local'
    if n_frames >= 10000:
        plt.suptitle(f'Lysozyme MD Lambda³ Analysis Results ({n_frames} frames) - Global View', fontsize=16)
    else:
        plt.suptitle('Lysozyme MD Lambda³ Analysis Results - Global View', fontsize=16)
    
    plt.show()
    
    # Return multi-scale results
    return detector, result, multi_scores

if __name__ == "__main__":
    detector, result, scores = demo_md_analysis()
    if detector is not None:
        print("\nDemo completed!")
        print("\nTo examine specific anomalies:")
        print("  - Use scores['global'] for large-scale structural changes")
        print("  - Use scores['local'] for local structural changes (helix breaks, etc.)")
        print("  - Use result.md_features to access MD-specific features")
        print("  - Use result.paths to examine Lambda structure tensors")
        print("  - Use result.jump_structures to analyze transitions")
