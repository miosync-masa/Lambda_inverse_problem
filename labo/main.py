"""
Hierarchical MD Lambda³ Anomaly Detection System
Refactored version with multi-scale structural analysis
All functions included - no external dependencies except standard libraries
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
warnings.filterwarnings('ignore')

# ===============================
# Global Constants
# ===============================
DELTA_PERCENTILE = 94.0
LOCAL_WINDOW_SIZE = 15
LOCAL_JUMP_PERCENTILE = 91.0
WINDOW_SIZE = 30
MULTI_SCALE_WINDOWS = [3, 5, 10, 20, 40]

# MD-specific parameters
CONTACT_CUTOFF = 8.0  # Angstroms for contact map
RMSD_JUMP_THRESHOLD = 2.0  # Angstroms for significant structural change
DIHEDRAL_JUMP_THRESHOLD = 30.0  # Degrees for significant dihedral change

# ===============================
# Configuration Classes
# ===============================
@dataclass
class MDConfig:
    """Configuration parameters for MD Lambda³ analysis."""
    n_paths: int = 7            # Number of structure tensor paths
    jump_scale: float = 1.5     # Sensitivity of jump detection
    use_union: bool = True      # Whether to use union of jumps across paths
    w_topo: float = 0.3         # Weight for topological anomaly score
    w_pulse: float = 0.2        # Weight for pulsation score
    w_structure: float = 0.3    # Weight for structural features
    w_dynamics: float = 0.2     # Weight for dynamic features
    
    # MD-specific parameters
    use_dihedrals: bool = True  # Use dihedral angles as features
    use_contacts: bool = True   # Use contact maps
    use_rmsd: bool = True       # Use RMSD-based features
    use_rg: bool = True         # Use radius of gyration
    use_sasa: bool = False      # Use solvent accessible surface area (optional)

@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical analysis"""
    scales: List[float] = None  # [0.1, 0.5, 1.0, 2.0, 10.0]
    local_features: List[str] = None  # ['helix', 'sheet', 'loop']
    cross_scale_coupling: bool = True
    bayesian_integration: bool = True
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [0.1, 0.5, 1.0, 2.0, 10.0]
        if self.local_features is None:
            self.local_features = ['helix', 'sheet', 'loop']

@dataclass
class Lambda3Result:
    """Data class for Lambda³ analysis results"""
    paths: Dict[int, np.ndarray]
    topological_charges: Dict[int, float]
    stabilities: Dict[int, float]
    energies: Dict[int, float]
    entropies: Dict[int, Dict[str, float]]
    jump_structures: Optional[Dict] = None

@dataclass
class MDLambda3Result:
    """Data class for storing results of MD Lambda³ structural analysis."""
    paths: Dict[int, np.ndarray]                   # Structure tensor paths from MD features
    topological_charges: Dict[int, float]          # Topological charge Q_Λ for each path
    stabilities: Dict[int, float]                  # Topological stability σ_Q for each path
    energies: Dict[int, float]                     # Pulsation/energy metrics for each path
    entropies: Dict[int, Dict[str, float]]         # Multi-type entropies
    classifications: Dict[int, str]                # Path-level classification labels
    jump_structures: Optional[Dict] = None         # Structural transition info
    md_features: Optional[Dict] = None             # Original MD features used

# ===============================
# JIT-compiled Core Functions
# ===============================

@njit
def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calculate pairwise distance matrix for a set of coordinates."""
    n_atoms = coords.shape[0]
    dist_matrix = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
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
def compute_pulsation_energy_from_path(path: np.ndarray) -> Tuple[float, float, float]:
    """Calculate pulsation energy from path data."""
    if len(path) < 2:
        return 0.0, 0.0, 0.0
    
    diff = np.diff(path)
    abs_diff = np.abs(diff)
    threshold = np.mean(abs_diff) + 2.0 * np.std(abs_diff)
    
    pos_mask = diff > threshold
    neg_mask = diff < -threshold
    
    pos_intensity = np.sum(diff[pos_mask]) if np.any(pos_mask) else 0.0
    neg_intensity = np.sum(np.abs(diff[neg_mask])) if np.any(neg_mask) else 0.0
    jump_intensity = pos_intensity + neg_intensity
    
    asymmetry = (pos_intensity - neg_intensity) / (pos_intensity + neg_intensity + 1e-10)
    
    n_jumps = np.sum(pos_mask) + np.sum(neg_mask)
    pulsation_power = jump_intensity * n_jumps / len(path)
    
    return jump_intensity, asymmetry, pulsation_power

@njit(parallel=True)
def compute_topological_charge_jit(path: np.ndarray, n_segments: int = 10) -> Tuple[float, float]:
    """Topological charge calculation."""
    n = len(path)
    closed_path = np.empty(n + 1)
    closed_path[:-1] = path
    closed_path[-1] = path[0]
    
    theta = np.empty(n)
    for i in prange(n):
        theta[i] = np.arctan2(closed_path[i+1], closed_path[i])
    
    Q_Lambda = 0.0
    for i in range(n-1):
        diff = theta[i+1] - theta[i]
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        Q_Lambda += diff
    Q_Lambda /= (2 * np.pi)
    
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
    entropies[1] = -np.log(sum_p2) if sum_p2 > 0 else 0.0
    
    # Tsallis (q=1.5)
    sum_p15 = 0.0
    for p in norm_path:
        sum_p15 += p ** 1.5
    entropies[2] = (1.0 - sum_p15) / 0.5
    
    # Max, Min, Variance
    entropies[3] = np.max(norm_path)
    entropies[4] = np.min(norm_path)
    mean_p = np.mean(norm_path)
    var = 0.0
    for p in norm_path:
        var += (p - mean_p) ** 2
    entropies[5] = var / len(norm_path)
    
    return entropies

# ===============================
# Main Detector Class
# ===============================

class MDLambda3Detector:
    """Enhanced MD-specific Lambda³ detector with hierarchical analysis"""
    
    def __init__(self, 
                 config: MDConfig = None,
                 hierarchical_config: Optional[HierarchicalConfig] = None,
                 base_window_size: int = 50,
                 verbose: bool = True):
        self.config = config or MDConfig()
        self.hierarchical_config = hierarchical_config or HierarchicalConfig()
        self.base_window_size = base_window_size
        self.verbose = verbose
        
        # Adaptive parameters (will be updated based on data)
        self.LOCAL_WINDOW_SIZE = 32
        self.WINDOW_SIZE = 25
        self.DELTA_PERCENTILE = 95
        self.MULTI_SCALE_WINDOWS = [10, 20, 40, 80, 200, 400]
        self._adaptive_metrics = {}
        self._analysis_cache = {}
    
    # ===== Feature Extraction Methods =====
    
    def extract_md_features(self, trajectory: np.ndarray, 
                          backbone_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract both global and local MD features"""
        n_frames, n_atoms, _ = trajectory.shape
        features = {}
        
        # Global features
        features.update(self._extract_global_features(trajectory, backbone_indices))
        
        # Local features
        if self.hierarchical_config.local_features:
            features.update(self._extract_local_features(trajectory, backbone_indices))
        
        # Multi-scale features
        for scale in self.hierarchical_config.scales:
            scaled_features = self._extract_scaled_features(trajectory, backbone_indices, scale)
            for key, value in scaled_features.items():
                features[f'{key}_scale_{scale}'] = value
        
        return features
    
    def _extract_global_features(self, trajectory: np.ndarray, 
                                backbone_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract global structural features"""
        n_frames, n_atoms, _ = trajectory.shape
        features = {}
        
        # Contact-based features (memory-optimized)
        if self.config.use_contacts:
            if self.verbose:
                print("  - Calculating contact-based features (memory-optimized)...")
            
            # Lightweight feature arrays
            per_atom_contacts = np.zeros((n_frames, n_atoms), dtype=np.uint16)
            contact_changes = np.zeros(n_frames)
            
            # First frame
            prev_contact_map = (calculate_contact_map(trajectory[0], CONTACT_CUTOFF) > 0.5)
            per_atom_contacts[0] = np.sum(prev_contact_map, axis=1)
            
            # Process frames
            for i in range(1, n_frames):
                if self.verbose and i % 10000 == 0:
                    print(f"    Processing frame {i}/{n_frames}...")
                
                # Current frame contact map
                current_contact_map = (calculate_contact_map(trajectory[i], CONTACT_CUTOFF) > 0.5)
                
                # Per-atom contacts
                per_atom_contacts[i] = np.sum(current_contact_map, axis=1)
                
                # Contact changes
                contact_changes[i] = np.sum(current_contact_map != prev_contact_map)
                
                # Update reference
                prev_contact_map = current_contact_map
            
            features['per_atom_contacts'] = per_atom_contacts
            features['contact_changes'] = contact_changes
            
            # Cache contact_changes for other methods
            if not hasattr(self, '_analysis_cache'):
                self._analysis_cache = {}
            self._analysis_cache['contact_changes'] = contact_changes
            
            if self.verbose:
                mem_usage_gb = (per_atom_contacts.nbytes + contact_changes.nbytes) / 1e9
                print(f"  - Contact features memory usage: ~{mem_usage_gb:.3f} GB")
        
        # RMSD to first frame
        if self.config.use_rmsd:
            rmsd = self._calculate_rmsd_vectorized(
                trajectory[:, backbone_indices], 
                trajectory[0, backbone_indices]
            )
            features['rmsd'] = rmsd
            
            # Local RMSD (frame-to-frame)
            local_rmsd = np.zeros(n_frames)
            for i in range(1, n_frames):
                diff = trajectory[i, backbone_indices] - trajectory[i-1, backbone_indices]
                local_rmsd[i] = np.sqrt(np.mean(diff**2))
            features['local_rmsd'] = local_rmsd
        
        # Radius of gyration
        if self.config.use_rg:
            rg = np.array([
                self._calculate_radius_of_gyration(trajectory[i])
                for i in range(n_frames)
            ])
            features['radius_of_gyration'] = rg
        
        # Principal components
        pc_projections = self._calculate_pc_projections(
            trajectory[:, backbone_indices]
        )
        features['pc_projections'] = pc_projections
        
        # Add frequency features
        freq_features = self._extract_md_frequency_features(trajectory, backbone_indices)
        features.update(freq_features)
        
        return features
    
    def _extract_local_features(self, trajectory: np.ndarray, 
                               backbone_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract local structural features"""
        n_frames = len(trajectory)
        local_features = {}
        
        # Segment-based RMSD
        n_backbone = len(backbone_indices)
        segment_size = max(10, n_backbone // 10)
        segment_rmsds = []
        
        for i in range(0, n_backbone, segment_size):
            segment_idx = backbone_indices[i:i+segment_size]
            segment_rmsd = self._calculate_rmsd_vectorized(
                trajectory[:, segment_idx], 
                trajectory[0, segment_idx]
            )
            segment_rmsds.append(segment_rmsd)
        
        local_features['segment_rmsds'] = np.array(segment_rmsds).T
        
        # Local contact densities
        local_contacts = self._compute_local_contact_density(trajectory, backbone_indices)
        local_features['local_contact_density'] = local_contacts
        
        # Helix propensity
        if 'helix' in self.hierarchical_config.local_features:
            local_features['helix_propensity'] = self._estimate_helix_propensity(
                trajectory, backbone_indices
            )
        
        return local_features
    
    def _extract_scaled_features(self, trajectory: np.ndarray, 
                                backbone_indices: np.ndarray, 
                                scale: float) -> Dict[str, np.ndarray]:
        """Extract features at a specific temporal scale"""
        stride = max(1, int(scale))
        scaled_traj = trajectory[::stride]
        
        features = {}
        
        # Scaled RMSD
        features['rmsd'] = self._calculate_rmsd_vectorized(
            scaled_traj[:, backbone_indices], 
            scaled_traj[0, backbone_indices]
        )
        
        # Scaled Rg
        features['rg'] = np.array([
            self._calculate_radius_of_gyration(scaled_traj[i])
            for i in range(len(scaled_traj))
        ])
        
        return features
    
    # ===== Lambda Construction Methods =====
    
    def construct_lambda_from_md_features(self, features: Dict[str, np.ndarray], 
                                         n_paths: int = 7) -> Lambda3Result:
        """Construct Lambda structure tensor from MD features including frequency"""
        
        # Select and normalize features (exclude frequency features from matrix)
        feature_matrix = []
        feature_names = []
        freq_features = {}
        
        for name, data in features.items():
            if 'freq_' in name:
                # Store frequency features separately
                freq_features[name] = data[0] if len(data) > 0 else 0.0
            elif data.ndim == 1 and len(data) > 1:
                # Only include time-series features
                feature_matrix.append(data)
                feature_names.append(name)
            elif data.ndim == 2:
                if name == 'per_atom_contacts':
                    # For per-atom contacts, use mean across atoms
                    mean_contacts = np.mean(data, axis=1)
                    feature_matrix.append(mean_contacts)
                    feature_names.append('mean_atom_contacts')
                elif name == 'pc_projections' and data.shape[1] > 0:
                    # For PC projections, add each PC separately (up to 3)
                    for i in range(min(3, data.shape[1])):
                        feature_matrix.append(data[:, i])
                        feature_names.append(f'pc{i}')
        
        if not feature_matrix:
            # Fallback if no valid features
            if features:
                # Get n_frames from any available feature
                for feat_name, feat_data in features.items():
                    if hasattr(feat_data, '__len__'):
                        n_frames = len(feat_data)
                        break
                else:
                    n_frames = 1000  # Default fallback
            else:
                n_frames = 1000  # Default fallback
                
            feature_matrix = [np.random.randn(n_frames)]
            feature_names = ['random']
        
        feature_matrix = np.column_stack(feature_matrix)
        n_frames, n_features = feature_matrix.shape
        
        # Normalize
        normalized_features = (feature_matrix - np.mean(feature_matrix, axis=0)) / (
            np.std(feature_matrix, axis=0) + 1e-8
        )
        
        # Construct Lambda paths
        Lambda_matrix = np.zeros((n_paths, n_frames))
        
        for p in range(n_paths):
            if p == 0:
                # Path 0: RMSD-based
                if 'rmsd' in features and len(features['rmsd']) == n_frames:
                    Lambda_matrix[p] = features['rmsd'] / (np.max(features['rmsd']) + 1e-8)
                else:
                    # Fallback
                    weights = np.random.randn(n_features)
                    weights = weights / np.linalg.norm(weights)
                    Lambda_matrix[p] = normalized_features @ weights
            elif p == 1:
                # Path 1: Rg-based
                if 'radius_of_gyration' in features and len(features['radius_of_gyration']) == n_frames:
                    Lambda_matrix[p] = (features['radius_of_gyration'] - 
                                      np.mean(features['radius_of_gyration'])) / (
                                      np.std(features['radius_of_gyration']) + 1e-8)
                else:
                    # Fallback
                    weights = np.random.randn(n_features)
                    weights = weights / np.linalg.norm(weights)
                    Lambda_matrix[p] = normalized_features @ weights
            elif p == 2:
                # Path 2: Contact-based
                if 'contact_changes' in features and len(features['contact_changes']) == n_frames:
                    Lambda_matrix[p] = features['contact_changes'] / (
                        np.max(features['contact_changes']) + 1e-8)
                else:
                    # Fallback
                    weights = np.random.randn(n_features)
                    weights = weights / np.linalg.norm(weights)
                    Lambda_matrix[p] = normalized_features @ weights
            elif p == 3:
                # Path 3: PC1
                if 'pc_projections' in features:
                    pc_proj = features['pc_projections']
                    if pc_proj.ndim == 2 and pc_proj.shape[1] > 0:
                        Lambda_matrix[p] = pc_proj[:, 0]
                    else:
                        # Fallback if no valid PC projections
                        weights = np.random.randn(n_features)
                        weights = weights / np.linalg.norm(weights)
                        Lambda_matrix[p] = normalized_features @ weights
                else:
                    # Fallback if no PC projections
                    weights = np.random.randn(n_features)
                    weights = weights / np.linalg.norm(weights)
                    Lambda_matrix[p] = normalized_features @ weights
            elif p == 4:
                # Path 4: Frequency-modulated path
                if freq_features:
                    # Create a frequency-modulated path
                    base_path = normalized_features @ np.random.randn(n_features)
                    # Modulate by average frequency energy
                    avg_freq_energy = np.mean([v for k, v in freq_features.items() if 'energy' in k])
                    Lambda_matrix[p] = base_path * (1 + avg_freq_energy / 10)
                else:
                    # Fallback to random projection
                    weights = np.random.randn(n_features)
                    weights = weights / np.linalg.norm(weights)
                    Lambda_matrix[p] = normalized_features @ weights
            else:
                # Remaining paths: Random projections
                weights = np.random.randn(n_features)
                weights = weights / np.linalg.norm(weights)
                linear_comb = normalized_features @ weights
                
                if p % 2 == 0:
                    Lambda_matrix[p] = np.tanh(linear_comb * 2)
                else:
                    Lambda_matrix[p] = linear_comb ** 3
        
        # Normalize paths
        paths = {}
        for p in range(n_paths):
            path_norm = np.linalg.norm(Lambda_matrix[p])
            if path_norm > 0:
                paths[p] = Lambda_matrix[p] / path_norm
            else:
                paths[p] = Lambda_matrix[p]
        
        # Compute structural properties
        charges, stabilities = self._compute_topology(paths)
        energies = self._compute_energies(paths)
        entropies = self._compute_entropies(paths)
        
        return Lambda3Result(
            paths=paths,
            topological_charges=charges,
            stabilities=stabilities,
            energies=energies,
            entropies=entropies
        )
    
    def construct_hierarchical_lambda(self, features: Dict[str, np.ndarray]) -> Dict:
        """Construct Lambda tensor with hierarchical structure"""
        
        # Separate features by scale
        scale_features = {scale: {} for scale in self.hierarchical_config.scales}
        global_features = {}
        
        for key, value in features.items():
            found_scale = False
            for scale in self.hierarchical_config.scales:
                if f'scale_{scale}' in key:
                    base_key = key.replace(f'_scale_{scale}', '')
                    scale_features[scale][base_key] = value
                    found_scale = True
                    break
            if not found_scale:
                global_features[key] = value
        
        # Construct Lambda for each scale
        hierarchical_lambdas = {}
        
        # Global Lambda
        global_lambda = self.construct_lambda_from_md_features(global_features)
        hierarchical_lambdas['global'] = global_lambda
        
        # Scale-specific Lambdas
        for scale, scale_feat in scale_features.items():
            if scale_feat:
                scale_lambda = self.construct_lambda_from_md_features(scale_feat, n_paths=3)
                hierarchical_lambdas[f'scale_{scale}'] = scale_lambda
        
        # Cross-scale coupling
        if self.hierarchical_config.cross_scale_coupling:
            coupling = self._analyze_cross_scale_coupling(hierarchical_lambdas)
            hierarchical_lambdas['coupling'] = coupling
        
        # Store original features for later use
        hierarchical_lambdas['_features'] = features
        
        return hierarchical_lambdas
    
    def detect_anomalies(self, result: MDLambda3Result) -> np.ndarray:
        """
        Detect anomalies in MD trajectory (backward compatibility).
        
        Args:
            result: MDLambda3Result object
            
        Returns:
            Array of anomaly scores for each frame.
        """
        n_frames = len(result.paths[0])
        scores = np.zeros(n_frames)
        
        # Jump-based anomalies
        if result.jump_structures:
            jump_mask = result.jump_structures['integrated']['unified_jumps'].astype(float)
            scores += self.config.w_structure * jump_mask
        
        # Topological anomalies
        for i, (Q, sigma) in enumerate(zip(
            result.topological_charges.values(),
            result.stabilities.values()
        )):
            if abs(Q) > 0.5 or sigma > 2.0:
                path = result.paths[i]
                path_anomaly = np.abs(path - np.mean(path)) / (np.std(path) + 1e-8)
                scores += self.config.w_topo * path_anomaly
        
        # Energy anomalies
        energies = list(result.energies.values())
        mean_energy = np.mean(energies)
        for i, energy in enumerate(energies):
            if energy > mean_energy * 2:
                path = result.paths[i]
                energy_anomaly = np.abs(path) * (energy / mean_energy)
                scores += self.config.w_pulse * energy_anomaly
        
        # MD-specific anomalies
        if result.md_features:
            if 'rmsd' in result.md_features:
                rmsd = result.md_features['rmsd']
                rmsd_anomaly = (rmsd - np.mean(rmsd)) / (np.std(rmsd) + 1e-8)
                rmsd_anomaly[rmsd_anomaly < 0] = 0
                scores += self.config.w_dynamics * rmsd_anomaly
        
        # Normalize
        scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-10)
        
        return scores
    
    # ===== Property Computation Methods =====
    
    def _compute_topology(self, paths: Dict[int, np.ndarray]) -> Tuple[Dict, Dict]:
        """Compute topological charges and stabilities"""
        charges = {}
        stabilities = {}
        
        for i, path in paths.items():
            Q, sigma = compute_topological_charge_jit(path)
            charges[i] = Q
            stabilities[i] = sigma
            
        return charges, stabilities
    
    def _compute_energies(self, paths: Dict[int, np.ndarray], 
                         jump_structures: Optional[Dict] = None) -> Dict[int, float]:
        """Compute pulsation energies"""
        energies = {}
        
        for i, path in paths.items():
            basic_energy = np.sum(path**2)
            
            # Add pulsation energy if jump structures available
            if jump_structures and 'path_jumps' in jump_structures and i in jump_structures['path_jumps']:
                pulse_power = jump_structures['path_jumps'][i].get('pulse_power', 0)
                energies[i] = basic_energy + 0.3 * pulse_power
            else:
                # Compute from path directly
                jump_int, _, pulse_pow = compute_pulsation_energy_from_path(path)
                energies[i] = basic_energy + 0.3 * pulse_pow
                
        return energies
    
    def _compute_entropies(self, paths: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
        """Compute various entropy measures"""
        entropies = {}
        entropy_keys = ["shannon", "renyi_2", "tsallis_1.5", "max", "min", "var"]
        
        for i, path in paths.items():
            all_entropies = compute_all_entropies_jit(path)
            entropy_dict = {}
            for j, key in enumerate(entropy_keys):
                entropy_dict[key] = all_entropies[j]
            entropies[i] = entropy_dict
            
        return entropies
    
    # ===== Anomaly Detection Methods =====
    
    def detect_hierarchical_anomalies(self, hierarchical_lambdas: Dict, 
                                     features: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Detect anomalies at each scale and integrate"""
        anomaly_scores = {}
        
        # Extract features from hierarchical_lambdas if not provided
        if features is None and '_features' in hierarchical_lambdas:
            features = hierarchical_lambdas['_features']
        
        # Detect anomalies at each scale
        for scale_key, scale_lambda in hierarchical_lambdas.items():
            if scale_key in ['coupling', '_features']:
                continue
                
            if hasattr(scale_lambda, 'paths'):
                # Detect structural transitions
                # Only pass features for global scale
                transitions = self._detect_structural_transitions(
                    scale_lambda.paths,
                    features if scale_key == 'global' and features is not None else None
                )
                
                # Compute anomaly scores
                scores = self._compute_anomaly_scores(scale_lambda, transitions)
                anomaly_scores[scale_key] = scores
        
        # Integrate scores
        if self.hierarchical_config.bayesian_integration:
            integrated_scores = self._bayesian_score_integration(anomaly_scores)
        else:
            integrated_scores = self._weighted_score_integration(anomaly_scores)
        
        return {
            'scale_specific': anomaly_scores,
            'integrated': integrated_scores,
            'coupling': hierarchical_lambdas.get('coupling', {})
        }
    
    def _detect_structural_transitions(self, paths: Dict[int, np.ndarray], 
                                      md_features: Dict[str, np.ndarray] = None) -> Dict:
        """Detect structural transitions with adaptive parameters"""
        jump_data = {'path_jumps': {}, 'integrated': {}}
        
        # Dynamic percentile based on MD metrics (if available)
        base_percentile = self.DELTA_PERCENTILE
        if md_features and 'rmsd' in md_features:
            rmsd_std = np.std(md_features['rmsd'])
            if rmsd_std > 2.0:
                base_percentile = 90.0
            elif rmsd_std < 0.5:
                base_percentile = 96.0
        
        # Detect jumps in each path
        for p, path in paths.items():
            diff, threshold = calculate_diff_and_threshold(path, base_percentile)
            pos_jumps, neg_jumps = detect_jumps(diff, threshold)
            
            # Use adaptive local window
            local_std = calculate_local_std(path, self.LOCAL_WINDOW_SIZE)
            score = np.abs(diff) / (local_std + 1e-8)
            
            # Adaptive local percentile
            local_percentile = LOCAL_JUMP_PERCENTILE
            if md_features and 'rmsd' in md_features:
                rmsd_std = np.std(md_features['rmsd'])
                local_percentile = LOCAL_JUMP_PERCENTILE - (rmsd_std - 1.0) * 2.0
                local_percentile = np.clip(local_percentile, 85.0, 95.0)
            
            local_threshold = np.percentile(score, local_percentile)
            local_jumps = (score > local_threshold).astype(int)
            
            # Use adaptive window for tension calculation
            rho_t = calculate_rho_t(path, self.WINDOW_SIZE)
            
            # Calculate pulsation energy
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
                'percentile_used': base_percentile,
                'local_percentile_used': local_percentile
            }
        
        # Integrate jumps
        n_frames = len(paths[0])
        unified_jumps = np.zeros(n_frames, dtype=bool)
        
        for p_data in jump_data['path_jumps'].values():
            unified_jumps |= (p_data['pos_jumps'] | p_data['neg_jumps']).astype(bool)
        
        jump_data['integrated'] = {
            'unified_jumps': unified_jumps,
            'n_jumps': np.sum(unified_jumps),
            'base_percentile': base_percentile
        }
        
        return jump_data
    
    def _compute_anomaly_scores(self, lambda_result: Lambda3Result, 
                               transitions: Dict) -> np.ndarray:
        """Compute anomaly scores from Lambda analysis with frequency features"""
        n_frames = len(lambda_result.paths[0])
        scores = np.zeros(n_frames)
        
        # Jump-based anomalies
        if transitions['integrated']['unified_jumps'] is not None:
            jump_mask = transitions['integrated']['unified_jumps'].astype(float)
            scores += 0.3 * jump_mask
        
        # Topological anomalies
        for i, (Q, sigma) in enumerate(zip(
            lambda_result.topological_charges.values(),
            lambda_result.stabilities.values()
        )):
            if abs(Q) > 0.5 or sigma > 2.0:
                path = lambda_result.paths[i]
                path_anomaly = np.abs(path - np.mean(path)) / (np.std(path) + 1e-8)
                scores += 0.2 * path_anomaly
        
        # Energy anomalies
        energies = list(lambda_result.energies.values())
        mean_energy = np.mean(energies)
        for i, energy in enumerate(energies):
            if energy > mean_energy * 2:
                path = lambda_result.paths[i]
                energy_anomaly = np.abs(path) * (energy / mean_energy)
                scores += 0.1 * energy_anomaly
        
        # Frequency-based anomalies
        for i, path in lambda_result.paths.items():
            freq_features = self._extract_frequency_features_from_path(path)
            
            # High frequency anomaly (rapid oscillations)
            if freq_features['freq_hf_lf_ratio'] > 2.0:
                hf_anomaly = np.abs(path) * freq_features['freq_hf_lf_ratio'] / 2.0
                scores += 0.1 * hf_anomaly
            
            # Low frequency anomaly (slow drift)
            if freq_features['freq_peak'] < 0.01 and freq_features['freq_energy'] > 0.5:
                lf_anomaly = np.abs(path) * freq_features['freq_energy']
                scores += 0.1 * lf_anomaly
        
        # Normalize
        scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-10)
        
        return scores
    
    # ===== Integration Methods =====
    
    def _bayesian_score_integration(self, anomaly_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Integrate anomaly scores using Bayesian approach"""
        weights = {}
        
        for scale_key, scores in anomaly_scores.items():
            variance = np.var(scores)
            weights[scale_key] = 1.0 / (variance + 1e-6)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted average
        min_length = min(len(scores) for scores in anomaly_scores.values())
        integrated = np.zeros(min_length)
        
        for scale_key, scores in anomaly_scores.items():
            integrated += weights[scale_key] * scores[:min_length]
        
        return integrated
    
    def _weighted_score_integration(self, anomaly_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple weighted integration"""
        scale_weights = {
            'global': 1.0,
            'scale_0.1': 0.5,
            'scale_0.5': 0.7,
            'scale_1.0': 1.0,
            'scale_2.0': 0.8,
            'scale_10.0': 0.6
        }
        
        min_length = min(len(scores) for scores in anomaly_scores.values())
        integrated = np.zeros(min_length)
        total_weight = 0
        
        for scale_key, scores in anomaly_scores.items():
            weight = scale_weights.get(scale_key, 0.5)
            integrated += weight * scores[:min_length]
            total_weight += weight
        
        return integrated / total_weight
    
    def _compute_jump_anomaly_scores(self, jump_structures: Dict) -> np.ndarray:
        """Compute anomaly scores based on structural jumps."""
        unified_jumps = jump_structures['integrated']['unified_jumps']
        scores = unified_jumps.astype(float)
        
        # Weight by jump intensity
        for p, p_data in jump_structures['path_jumps'].items():
            jump_mask = (p_data['pos_jumps'] | p_data['neg_jumps']).astype(bool)
            if 'jump_intensity' in p_data:
                scores[jump_mask] *= (1 + p_data['jump_intensity'] / 10)
            
        return scores
    
    def _compute_topological_anomaly_scores(self, result: Lambda3Result) -> np.ndarray:
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
    
    def _compute_energy_anomaly_scores(self, result: Lambda3Result) -> np.ndarray:
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
    
    def _compute_md_specific_anomalies(self, result: Lambda3Result, 
                                      md_features: Optional[Dict] = None) -> np.ndarray:
        """Compute MD-specific anomaly scores."""
        n_frames = len(result.paths[0])
        scores = np.zeros(n_frames)
        
        if md_features is None:
            return scores
        
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
    
    # ===== Helper Methods =====
    
    def _extract_frequency_features_from_path(self, path: np.ndarray) -> Dict[str, float]:
        """Extract frequency features from a single path"""
        # FFT calculation
        fft = np.fft.fft(path)
        fft_abs = np.abs(fft)
        fft_freqs = np.fft.fftfreq(len(path))
        
        # Positive frequencies only
        pos_mask = fft_freqs > 0
        pos_freqs = fft_freqs[pos_mask]
        pos_fft = fft_abs[pos_mask]
        
        features = {}
        
        if len(pos_fft) > 0:
            # Peak frequency and amplitude
            peak_idx = np.argmax(pos_fft)
            features['freq_peak'] = pos_freqs[peak_idx]
            features['freq_peak_amp'] = pos_fft[peak_idx]
            
            # Frequency energy
            features['freq_energy'] = np.sum(pos_fft ** 2)
            
            # Spectral centroid
            if np.sum(pos_fft) > 0:
                features['freq_centroid'] = np.sum(pos_freqs * pos_fft) / np.sum(pos_fft)
            else:
                features['freq_centroid'] = 0.0
            
            # Spectral entropy
            if np.sum(pos_fft) > 0:
                norm_fft = pos_fft / np.sum(pos_fft)
                features['freq_entropy'] = -np.sum(norm_fft * np.log(norm_fft + 1e-10))
            else:
                features['freq_entropy'] = 0.0
            
            # High/Low frequency ratio
            mid_point = len(pos_fft) // 2
            if mid_point > 0:
                low_energy = np.sum(pos_fft[:mid_point] ** 2)
                high_energy = np.sum(pos_fft[mid_point:] ** 2)
                features['freq_hf_lf_ratio'] = high_energy / (low_energy + 1e-10)
            else:
                features['freq_hf_lf_ratio'] = 0.0
        else:
            # Default values
            features = {
                'freq_peak': 0.0,
                'freq_peak_amp': 0.0,
                'freq_energy': 0.0,
                'freq_centroid': 0.0,
                'freq_entropy': 0.0,
                'freq_hf_lf_ratio': 0.0
            }
        
        return features
    
    def _extract_md_frequency_features(self, trajectory: np.ndarray,
                                      backbone_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency features from MD trajectory"""
        n_frames = len(trajectory)
        
        freq_features = {}
        
        # RMSD frequency analysis
        if hasattr(self, '_calculate_rmsd_vectorized'):
            rmsd = self._calculate_rmsd_vectorized(
                trajectory[:, backbone_indices],
                trajectory[0, backbone_indices]
            )
            rmsd_freq = self._extract_frequency_features_from_path(rmsd)
            # Store as time series (repeat scalar value for all frames)
            for key, val in rmsd_freq.items():
                freq_features[f'rmsd_{key}'] = np.full(n_frames, val)
        
        # Rg frequency analysis
        rg = np.array([
            self._calculate_radius_of_gyration(trajectory[i])
            for i in range(n_frames)
        ])
        rg_freq = self._extract_frequency_features_from_path(rg)
        for key, val in rg_freq.items():
            freq_features[f'rg_{key}'] = np.full(n_frames, val)
        
        # PC frequency analysis
        pc_projections = self._calculate_pc_projections(trajectory[:, backbone_indices])
        if pc_projections.shape[1] > 0:  # Check if PC projection succeeded
            for i in range(min(3, pc_projections.shape[1])):
                pc_freq = self._extract_frequency_features_from_path(pc_projections[:, i])
                for key, val in pc_freq.items():
                    freq_features[f'pc{i}_{key}'] = np.full(n_frames, val)
        
        # Contact frequency analysis (use cached data if available)
        if hasattr(self, '_analysis_cache') and 'contact_changes' in self._analysis_cache:
            contact_changes = self._analysis_cache['contact_changes']
        else:
            contact_changes = self._calculate_contact_changes(trajectory[:, backbone_indices])
        
        contact_freq = self._extract_frequency_features_from_path(contact_changes)
        for key, val in contact_freq.items():
            freq_features[f'contact_{key}'] = np.full(n_frames, val)
        
        return freq_features
    
    def _calculate_rmsd_vectorized(self, positions: np.ndarray, 
                                   reference: np.ndarray) -> np.ndarray:
        """Calculate RMSD for each frame"""
        diff = positions - reference[np.newaxis, :, :]
        squared_distances = np.sum(diff**2, axis=(1, 2))
        rmsd = np.sqrt(squared_distances / len(reference))
        return rmsd
    
    def _calculate_radius_of_gyration(self, positions: np.ndarray) -> float:
        """Calculate radius of gyration"""
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        rg = np.sqrt(np.mean(distances**2))
        return rg
    
    def _calculate_contact_changes(self, positions: np.ndarray, 
                                   cutoff: float = 8.0) -> np.ndarray:
        """Calculate contact map changes"""
        n_frames, n_atoms, _ = positions.shape
        
        # If already computed in global features, return it
        if hasattr(self, '_analysis_cache') and 'contact_changes' in self._analysis_cache:
            return self._analysis_cache['contact_changes']
        
        contact_changes = np.zeros(n_frames)
        
        # Reference contact map
        ref_contact_map = (calculate_contact_map(positions[0], cutoff) > 0.5)
        
        for i in range(1, n_frames):
            current_contact_map = (calculate_contact_map(positions[i], cutoff) > 0.5)
            contact_changes[i] = np.sum(current_contact_map != ref_contact_map)
            ref_contact_map = current_contact_map  # Update reference for frame-to-frame changes
        
        return contact_changes
    
    def _calculate_pc_projections(self, positions: np.ndarray, n_components: int = 3):
        """Calculate principal component projections"""
        n_frames, n_atoms, _ = positions.shape
        
        # Reshape to 2D
        reshaped = positions.reshape(n_frames, -1)
        
        # Center the data
        mean_structure = np.mean(reshaped, axis=0)
        centered = reshaped - mean_structure
        
        # Check if we have enough data
        if n_frames < n_components:
            n_components = n_frames
        
        if n_frames < 2:
            # Return empty array if not enough frames
            return np.zeros((n_frames, 0))
        
        try:
            # Simple PCA using SVD
            U, S, Vt = np.linalg.svd(centered.T / np.sqrt(n_frames - 1), full_matrices=False)
            
            # Project onto PCs
            projections = centered @ U[:, :n_components]
            
            return projections
        except np.linalg.LinAlgError:
            # If SVD fails, return zeros
            return np.zeros((n_frames, n_components))
    
    def _compute_local_contact_density(self, trajectory: np.ndarray, 
                                      backbone_indices: np.ndarray) -> np.ndarray:
        """Compute local contact density"""
        n_frames = len(trajectory)
        n_backbone = len(backbone_indices)
        
        # Divide into regions
        n_regions = 10
        region_size = n_backbone // n_regions
        local_densities = np.zeros((n_frames, n_regions))
        
        cutoff = 8.0
        
        for frame_idx in range(n_frames):
            positions = trajectory[frame_idx, backbone_indices]
            
            for region_idx in range(n_regions):
                start = region_idx * region_size
                end = min((region_idx + 1) * region_size, n_backbone)
                
                region_pos = positions[start:end]
                distances = np.linalg.norm(
                    region_pos[:, np.newaxis] - region_pos[np.newaxis, :],
                    axis=2
                )
                contacts = np.sum(distances < cutoff) - len(region_pos)
                local_densities[frame_idx, region_idx] = contacts / (
                    len(region_pos) * (len(region_pos) - 1)
                )
        
        return local_densities
    
    def _estimate_helix_propensity(self, trajectory: np.ndarray, 
                                   backbone_indices: np.ndarray) -> np.ndarray:
        """Estimate helix propensity"""
        n_frames = len(trajectory)
        
        # Assume every 4th backbone atom is Cα
        ca_indices = backbone_indices[::4]
        helix_scores = np.zeros(n_frames)
        
        for frame_idx in range(n_frames):
            ca_positions = trajectory[frame_idx, ca_indices]
            
            # Check i to i+4 distances
            helix_distances = []
            for i in range(len(ca_positions) - 4):
                dist = np.linalg.norm(ca_positions[i] - ca_positions[i+4])
                helix_distances.append(dist)
            
            # Ideal distance is ~6.0 Å
            if helix_distances:
                helix_scores[frame_idx] = np.mean(
                    np.exp(-0.5 * ((np.array(helix_distances) - 6.0) / 1.0)**2)
                )
        
        return helix_scores
    
    def _analyze_cross_scale_coupling(self, hierarchical_lambdas: Dict) -> Dict:
        """Analyze coupling between scales"""
        coupling_metrics = {}
        
        if 'global' in hierarchical_lambdas and hasattr(hierarchical_lambdas['global'], 'paths'):
            global_paths = hierarchical_lambdas['global'].paths
            
            for scale_key, scale_lambda in hierarchical_lambdas.items():
                if scale_key.startswith('scale_') and hasattr(scale_lambda, 'paths'):
                    scale_paths = scale_lambda.paths
                    
                    correlations = []
                    for path_idx in range(min(len(global_paths), len(scale_paths))):
                        if path_idx in global_paths and path_idx in scale_paths:
                            global_path = global_paths[path_idx]
                            scale_path = scale_paths[path_idx]
                            
                            min_len = min(len(global_path), len(scale_path))
                            if min_len > 1:
                                corr = np.corrcoef(
                                    global_path[:min_len], 
                                    scale_path[:min_len]
                                )[0, 1]
                                correlations.append(corr)
                    
                    coupling_metrics[scale_key] = {
                        'mean_correlation': np.mean(correlations) if correlations else 0.0,
                        'std_correlation': np.std(correlations) if correlations else 0.0
                    }
        
        return coupling_metrics
    
    def _classify_md_structures(self, paths: Dict, charges: Dict, stabilities: Dict,
                               jump_structures: Dict, md_features: Dict) -> Dict[int, str]:
        """Classify structural states based on Lambda³ analysis."""
        classifications = {}
        
        for i in paths.keys():
            Q = charges[i]
            sigma = stabilities[i]
            
            # Basic classification based on topology
            if Q < -0.5:
                base = "Anti-matter-like structure"
            elif Q > 0.5:
                base = "Matter-like structure"
            else:
                base = "Neutral structure"
            
            # Modifiers based on stability
            tags = []
            
            if sigma > 2.5:
                tags.append("Unstable/Chaotic")
            elif sigma < 0.5:
                tags.append("Super-stable")
            
            # MD-specific tags
            if 'rmsd' in md_features:
                rmsd_mean = np.mean(md_features['rmsd'])
                if rmsd_mean > 5.0:
                    tags.append("Highly deformed")
                elif rmsd_mean < 1.0:
                    tags.append("Near-native")
            
            # Jump characteristics
            if jump_structures and i in jump_structures.get('path_jumps', {}):
                jump_data = jump_structures['path_jumps'][i]
                if 'pulse_power' in jump_data:
                    if jump_data['pulse_power'] > 5:
                        tags.append("High-frequency pulsation")
                    elif jump_data['pulse_power'] < 0.1:
                        tags.append("Static")
            
            # Complete classification
            if tags:
                classifications[i] = base + " - " + "/".join(tags)
            else:
                classifications[i] = base
        
        return classifications
    
    def _update_adaptive_parameters(self, features: Dict[str, np.ndarray]):
        """Update adaptive parameters based on data characteristics"""
        n_frames = len(features.get('rmsd', []))
        
        # Compute adaptive window sizes using MD metrics
        adaptive_params = self._compute_md_adaptive_window_size(
            features,
            n_frames,
            base_window=self.base_window_size,
            min_window=10
        )
        
        # Update instance parameters
        self.LOCAL_WINDOW_SIZE = adaptive_params['local']
        self.WINDOW_SIZE = adaptive_params['transition']
        self.MULTI_SCALE_WINDOWS = adaptive_params['multiscale']
        
        # Store metrics for later use
        self._adaptive_metrics = adaptive_params['md_metrics']
        
        # Update percentile based on volatility
        rmsd_volatility = adaptive_params['md_metrics']['rmsd_volatility']
        if rmsd_volatility > 0.5:
            self.DELTA_PERCENTILE = 90.0  # More sensitive
        elif rmsd_volatility < 0.1:
            self.DELTA_PERCENTILE = 96.0  # Less sensitive
        else:
            self.DELTA_PERCENTILE = 94.0  # Default
        
        if self.verbose:
            print(f"  Adaptive parameters updated:")
            print(f"    LOCAL_WINDOW_SIZE: {self.LOCAL_WINDOW_SIZE}")
            print(f"    WINDOW_SIZE: {self.WINDOW_SIZE}")
            print(f"    DELTA_PERCENTILE: {self.DELTA_PERCENTILE}")
            print(f"    Scale factor: {adaptive_params['md_metrics']['scale_factor']:.3f}")
    
    def _compute_md_adaptive_window_size(self,
                                        features: Dict[str, np.ndarray],
                                        trajectory_length: int,
                                        base_window: int = 30,
                                        min_window: int = 10,
                                        max_window: int = None) -> Dict[str, any]:
        """
        Compute adaptive window sizes based on MD trajectory characteristics
        """
        if max_window is None:
            max_window = max(100, min(trajectory_length // 10, 500))
        
        # 1. RMSD volatility analysis
        rmsd_volatility = 0.0
        if 'rmsd' in features:
            rmsd = features['rmsd']
            rmsd_std = np.std(rmsd)
            rmsd_mean = np.mean(rmsd)
            rmsd_volatility = rmsd_std / (rmsd_mean + 1e-10)
        
        # 2. Contact change frequency
        contact_change_rate = 0.0
        if 'contact_changes' in features:
            changes = features['contact_changes']
            contact_change_rate = np.mean(changes > np.percentile(changes, 75))
        
        # 3. Structural periodicity (PC space analysis)
        periodicity = 0.0
        if 'pc_projections' in features:
            pc1 = features['pc_projections'][:, 0]
            fft = np.fft.fft(pc1)
            fft_abs = np.abs(fft[1:len(fft)//2])
            if len(fft_abs) > 0:
                periodicity = np.max(fft_abs) / (np.mean(fft_abs) + 1e-10)
        
        # 4. Local structural stability (Rg variation)
        rg_stability = 1.0
        if 'radius_of_gyration' in features:
            rg = features['radius_of_gyration']
            local_windows = []
            window_size = min(50, trajectory_length // 20)
            for i in range(0, len(rg) - window_size, window_size // 2):
                local_windows.append(np.std(rg[i:i+window_size]))
            if local_windows:
                rg_stability = np.std(local_windows) / (np.mean(local_windows) + 1e-10)
        
        # === Window size calculation ===
        scale_factor = 1.0
        
        # RMSD variation → smaller window for fast changes
        if rmsd_volatility > 0.5:
            scale_factor *= 0.7
        elif rmsd_volatility < 0.1:
            scale_factor *= 1.5
        
        # Frequent contact changes → smaller window
        if contact_change_rate > 0.3:
            scale_factor *= 0.8
        elif contact_change_rate < 0.05:
            scale_factor *= 1.3
        
        # Strong periodicity → adjust to period
        if periodicity > 5.0:
            scale_factor *= 1.2
        
        # Unstable structure → adaptive window
        if rg_stability > 1.0:
            scale_factor *= 0.9
        
        # === Purpose-specific window sizes ===
        
        # Local statistics
        local_window = int(base_window * scale_factor)
        local_window = np.clip(local_window, min_window, max_window)
        
        # Jump detection (more sensitive)
        jump_window = int(local_window * 0.6)
        jump_window = np.clip(jump_window, min_window // 2, max_window // 3)
        
        # Entropy calculation (more stable)
        entropy_window = int(local_window * 1.3)
        entropy_window = np.clip(entropy_window, min_window * 2, max_window)
        
        # Multi-scale analysis (MD time scales)
        multiscale_windows = []
        for scale in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
            window = int(local_window * scale)
            window = np.clip(window, min_window, max_window)
            multiscale_windows.append(window)
        
        # Transition detection
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
    
    # ===== Main Analysis Methods =====
    
    def analyze(self, trajectory: np.ndarray, 
                backbone_indices: Optional[np.ndarray] = None) -> MDLambda3Result:
        """
        Standard MD trajectory analysis (backward compatibility).
        
        Args:
            trajectory: (n_frames, n_atoms, 3) array of coordinates
            backbone_indices: Indices of backbone atoms
            
        Returns:
            MDLambda3Result object
        """
        # Extract features
        if self.verbose:
            print("Extracting MD features...")
        features = self._extract_global_features(trajectory, backbone_indices)
        
        # Update adaptive parameters
        if self.verbose:
            print("Computing adaptive parameters...")
        self._update_adaptive_parameters(features)
        
        # Construct Lambda
        if self.verbose:
            print("Constructing Λ structure tensor...")
        lambda_result = self.construct_lambda_from_md_features(features, self.config.n_paths)
        
        # Detect transitions
        if self.verbose:
            print("Detecting structural transitions...")
        jump_structures = self._detect_structural_transitions(lambda_result.paths, features)
        
        # Update Lambda result with jump structures
        lambda_result.jump_structures = jump_structures
        
        # Re-compute energies with jump information
        lambda_result.energies = self._compute_energies(lambda_result.paths, jump_structures)
        
        # Create MDLambda3Result
        classifications = self._classify_md_structures(
            lambda_result.paths,
            lambda_result.topological_charges,
            lambda_result.stabilities,
            jump_structures,
            features
        )
        
        return MDLambda3Result(
            paths=lambda_result.paths,
            topological_charges=lambda_result.topological_charges,
            stabilities=lambda_result.stabilities,
            energies=lambda_result.energies,
            entropies=lambda_result.entropies,
            classifications=classifications,
            jump_structures=jump_structures,
            md_features=features
        )
    
    def analyze_hierarchical(self, trajectory: np.ndarray, 
                           backbone_indices: np.ndarray) -> Dict:
        """Complete hierarchical analysis pipeline"""
        
        if self.verbose:
            print("Extracting hierarchical MD features...")
        
        # 1. Extract multi-scale features
        features = self.extract_md_features(trajectory, backbone_indices)
        
        # 2. Update adaptive parameters based on trajectory characteristics
        if self.verbose:
            print("Computing adaptive parameters...")
        self._update_adaptive_parameters(features)
        adaptive_metrics = self._adaptive_metrics  # Store for later use
        
        # 3. Construct hierarchical Lambda
        if self.verbose:
            print("Constructing hierarchical Λ structure tensor...")
        hierarchical_lambdas = self.construct_hierarchical_lambda(features)
        
        # 4. Detect anomalies
        if self.verbose:
            print("Detecting multi-scale anomalies...")
        anomaly_results = self.detect_hierarchical_anomalies(hierarchical_lambdas, features)
        
        # 5. Analyze events
        event_analysis = self._analyze_events_hierarchically(
            anomaly_results, 
            features
        )
        
        return {
            'features': features,
            'hierarchical_lambdas': hierarchical_lambdas,
            'anomaly_scores': anomaly_results,
            'event_analysis': event_analysis,
            'adaptive_params': {
                'window_size': self.WINDOW_SIZE,
                'local_window': self.LOCAL_WINDOW_SIZE,
                'percentile': self.DELTA_PERCENTILE,
                'scale_factor': adaptive_metrics.get('scale_factor', 1.0),
                'md_metrics': adaptive_metrics
            }
        }
    
    def _analyze_events_hierarchically(self, anomaly_results: Dict, 
                                     features: Dict) -> Dict:
        """Analyze how different events appear at different scales with frequency info"""
        event_signatures = {}
        
        # Expected events
        events = [
            {'name': 'partial_unfold', 'frames': (5000, 7500)},
            {'name': 'helix_break', 'frames': (15000, 17500)},
            {'name': 'major_unfold', 'frames': (25000, 30000)},
            {'name': 'misfold', 'frames': (35000, 37500)},
            {'name': 'aggregation_prone', 'frames': (42500, 45000)}
        ]
        
        scale_scores = anomaly_results['scale_specific']
        
        for event in events:
            start, end = event['frames']
            event_sig = {}
            
            for scale_key, scores in scale_scores.items():
                if start < len(scores) and end < len(scores):
                    event_scores = scores[start:end]
                    event_sig[scale_key] = {
                        'mean_score': np.mean(event_scores),
                        'max_score': np.max(event_scores),
                        'detection_ratio': np.sum(
                            event_scores > np.percentile(scores, 95)
                        ) / len(event_scores)
                    }
            
            # Add frequency analysis for each event
            if 'rmsd' in features and start < len(features['rmsd']) and end < len(features['rmsd']):
                event_rmsd = features['rmsd'][start:end]
                event_freq = self._extract_frequency_features_from_path(event_rmsd)
                event_sig['frequency_signature'] = event_freq
            
            event_signatures[event['name']] = event_sig
        
        return event_signatures


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
                if result.jump_structures and i in result.jump_structures.get('path_jumps', {}):
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
# Utility Functions
# ===============================

def extract_md_features(trajectory: np.ndarray, 
                       config: MDConfig,
                       backbone_indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Extract MD-specific features from trajectory (backward compatibility).
    """
    detector = MDLambda3Detector(config=config, verbose=False)
    features = detector._extract_global_features(trajectory, backbone_indices)
    
    # Add dihedral angles if requested
    if config.use_dihedrals and backbone_indices is not None:
        n_frames = len(trajectory)
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
    
    # Add center of mass motion
    com_trajectory = np.mean(trajectory, axis=1)
    com_velocity = np.zeros_like(com_trajectory)
    com_velocity[1:] = com_trajectory[1:] - com_trajectory[:-1]
    features['com_velocity'] = com_velocity
    
    return features

def construct_lambda_from_md_features(features: Dict[str, np.ndarray], 
                                    n_paths: int = 7) -> np.ndarray:
    """
    Directly construct Lambda structure tensor from MD features (backward compatibility).
    """
    detector = MDLambda3Detector(verbose=False)
    lambda_result = detector.construct_lambda_from_md_features(features, n_paths)
    
    # Convert to matrix format
    Lambda_matrix = np.zeros((n_paths, len(lambda_result.paths[0])))
    for i in range(n_paths):
        Lambda_matrix[i] = lambda_result.paths[i]
    
    return Lambda_matrix

def update_md_global_constants(window_sizes: Dict[str, any]):
    """MD用グローバル定数を動的に更新 (backward compatibility)"""
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
# Demo Function
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
    result = detector.analyze(trajectory, backbone_indices)
    
    print(f"\nAnalysis complete:")
    print(f"  - {len(result.paths)} structure tensor paths constructed")
    print(f"  - {result.jump_structures['integrated']['n_jumps']} structural transitions detected")
    print(f"  - Adaptive percentile used: {result.jump_structures['integrated']['base_percentile']:.1f}")
    print(f"  - RMSD volatility: {result.jump_structures['integrated'].get('rmsd_volatility', 0):.3f}")
    
    # Detect anomalies
    print("\n4. Detecting anomalies...")
    anomaly_scores = detector.detect_anomalies(result)
    top_anomalies = np.argsort(anomaly_scores)[-10:][::-1]
    print(f"Top 10 anomalous frames: {top_anomalies}")
    
    # Expected major events for 50k frames
    print("\nExpected major events:")
    print("  - 5000-7500 (partial_unfold)")
    print("  - 15000-17500 (helix_break)")
    print("  - 25000-30000 (major_unfold)")
    print("  - 35000-37500 (misfold)")
    print("  - 42500-45000 (aggregation_prone)")
    
    # Lysozyme-specific analysis
    print("\n5. Lysozyme-specific analysis:")
    if 'rmsd' in result.md_features:
        rmsd = result.md_features['rmsd']
        print(f"  - RMSD range: {np.min(rmsd):.2f} - {np.max(rmsd):.2f} Å")
        print(f"  - Average RMSD: {np.mean(rmsd):.2f} ± {np.std(rmsd):.2f} Å")
    
    if 'radius_of_gyration' in result.md_features:
        rg = result.md_features['radius_of_gyration']
        print(f"  - Rg range: {np.min(rg):.2f} - {np.max(rg):.2f} Å")
        print(f"  - Average Rg: {np.mean(rg):.2f} ± {np.std(rg):.2f} Å")
    
    # Check specific regions
    print("\n  - Anomaly scores in key regions:")
    
    # Event analysis
    if n_frames >= 50000:
        print(f"    Frames 5000-7500 (partial_unfold): mean score = {np.mean(anomaly_scores[5000:7500]):.3f}")
        print(f"    Frames 15000-17500 (helix_break): mean score = {np.mean(anomaly_scores[15000:17500]):.3f}")
        print(f"    Frames 25000-30000 (major_unfold): mean score = {np.mean(anomaly_scores[25000:30000]):.3f}")
        print(f"    Frames 35000-37500 (misfold): mean score = {np.mean(anomaly_scores[35000:37500]):.3f}")
        print(f"    Frames 42500-45000 (aggregation_prone): mean score = {np.mean(anomaly_scores[42500:45000]):.3f}")
        print(f"    Frames 0-1000 (baseline): mean score = {np.mean(anomaly_scores[0:1000]):.3f}")
        
        # Event detection analysis
        print("\n  - Event detection analysis:")
        events = [
            (5000, 7500, 'partial_unfold'),
            (15000, 17500, 'helix_break'),
            (25000, 30000, 'major_unfold'),
            (35000, 37500, 'misfold'),
            (42500, 45000, 'aggregation_prone')
        ]
        
        for start, end, name in events:
            event_scores = anomaly_scores[start:end]
            max_idx = start + np.argmax(event_scores)
            print(f"    {name}: max score = {np.max(event_scores):.3f} at frame {max_idx}")
    
    # Visualize
    print("\n6. Generating visualizations...")
    fig = detector.visualize_results(result, anomaly_scores)
    if n_frames >= 10000:
        plt.suptitle(f'Lysozyme MD Lambda³ Analysis Results ({n_frames} frames)', fontsize=16)
    else:
        plt.suptitle('Lysozyme MD Lambda³ Analysis Results', fontsize=16)
    
    plt.show()
    
    return detector, result, anomaly_scores

# ===============================
# Example Usage
# ===============================

if __name__ == "__main__":
    # Load trajectory
    trajectory = np.load('lysozyme_50k_trajectory.npy').astype(np.float64)
    backbone_indices = np.load('lysozyme_50k_backbone_indices.npy')
    
    # Example 1: Standard analysis (backward compatible)
    print("=== Standard MD Lambda³ Analysis ===")
    config = MDConfig()
    detector = MDLambda3Detector(config=config, verbose=True)
    
    result = detector.analyze(trajectory, backbone_indices)
    anomaly_scores = detector.detect_anomalies(result)
    
    print(f"\nAnalysis complete:")
    print(f"  - {len(result.paths)} structure tensor paths constructed")
    print(f"  - {result.jump_structures['integrated']['n_jumps']} structural transitions detected")
    print(f"  - Top 10 anomalous frames: {np.argsort(anomaly_scores)[-10:][::-1]}")
    
    # Example 2: Hierarchical analysis (new feature)
    print("\n\n=== Hierarchical MD Lambda³ Analysis ===")
    hierarchical_config = HierarchicalConfig(
        scales=[0.1, 0.5, 1.0, 2.0, 10.0],
        local_features=['helix', 'sheet', 'loop'],
        cross_scale_coupling=True,
        bayesian_integration=True
    )
    
    detector_hier = MDLambda3Detector(
        config=config,
        hierarchical_config=hierarchical_config,
        verbose=True
    )
    
    hier_results = detector_hier.analyze_hierarchical(trajectory, backbone_indices)
    
    # Print hierarchical results
    print("\n=== Hierarchical Event Analysis ===")
    for event_name, signatures in hier_results['event_analysis'].items():
        print(f"\n{event_name}:")
        for scale_key, metrics in signatures.items():
            if isinstance(metrics, dict) and 'mean_score' in metrics:
                print(f"  {scale_key}: mean={metrics['mean_score']:.3f}, "
                      f"max={metrics['max_score']:.3f}, "
                      f"detection={metrics['detection_ratio']:.3f}")
            elif scale_key == 'frequency_signature':
                print(f"  Frequency signature:")
                for freq_key, freq_val in metrics.items():
                    print(f"    {freq_key}: {freq_val:.3f}")
    
    # Print frequency features summary
    print("\n=== Global Frequency Features ===")
    for key, val in hier_results['features'].items():
        if 'freq_' in key and hasattr(val, '__len__') and len(val) == 1:
            print(f"  {key}: {val[0]:.3f}")
