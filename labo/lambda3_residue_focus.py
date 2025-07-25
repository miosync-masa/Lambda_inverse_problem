"""
Lambda³ Residue-Level Focus Analysis Extension v2.0
Two-stage hierarchical analysis with adaptive windows and async strong bonds
Author: Lambda³ Project (Enhanced by Tamaki & Mamichi)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from numba import njit, jit
import warnings

warnings.filterwarnings('ignore')

# Import from main Lambda³ module
try:
    from lambda3_md_fixed import (
        MDLambda3Result,
        compute_structural_coherence,
        detect_local_anomalies
    )
except ImportError:
    print("Warning: Main Lambda³ module not found. Some features may be limited.")

# ===============================
# Data Classes
# ===============================

@dataclass
class ResidueEvent:
    """Single residue-level event"""
    residue_id: int
    residue_name: str
    start_frame: int
    end_frame: int
    peak_lambda_f: float
    propagation_delay: int  # Frames from macro event start
    role: str  # 'initiator', 'propagator', 'responder'
    adaptive_window: int = 100  # 追加: アダプティブウィンドウサイズ

@dataclass 
class ResidueLevelAnalysis:
    """Results of residue-level analysis for one macro event"""
    event_name: str
    macro_start: int
    macro_end: int
    residue_events: List[ResidueEvent]
    causality_chain: List[Tuple[int, int, float]]  # (res1, res2, correlation)
    initiator_residues: List[int]
    key_propagation_paths: List[List[int]]
    # 追加: 同期なき強い結びつき
    async_strong_bonds: List[Dict]
    sync_network: List[Dict]
    network_stats: Dict

@dataclass
class TwoStageLambda3Result:
    """Complete two-stage analysis results"""
    macro_result: 'MDLambda3Result'  # From main analysis
    residue_analyses: Dict[str, ResidueLevelAnalysis]
    global_residue_importance: Dict[int, float]
    suggested_intervention_points: List[int]
    # 追加: ネットワーク統計
    global_network_stats: Dict

# ===============================
# Residue Mapping Functions
# ===============================

def create_residue_mapping(n_atoms: int = 1079, n_residues: int = 129) -> Dict[int, List[int]]:
    """
    Create mapping from residue ID to atom indices.
    This is a simplified version - in practice, would read from PDB.
    """
    atoms_per_residue = n_atoms // n_residues
    residue_atoms = {}
    
    for res_id in range(n_residues):
        start_atom = res_id * atoms_per_residue
        end_atom = min(start_atom + atoms_per_residue, n_atoms)
        residue_atoms[res_id] = list(range(start_atom, end_atom))
    
    # Handle remaining atoms
    if n_atoms % n_residues != 0:
        remaining_start = n_residues * atoms_per_residue
        residue_atoms[n_residues-1].extend(range(remaining_start, n_atoms))
    
    return residue_atoms

def get_residue_names() -> Dict[int, str]:
    """
    Get residue names for lysozyme.
    In practice, would read from PDB file.
    """
    # Simplified - using residue number as name
    return {i: f"RES{i+1}" for i in range(129)}

# ===============================
# Adaptive Window Calculation
# ===============================

@njit
def compute_residue_adaptive_window(
    anomaly_scores: np.ndarray,
    min_window: int = 30,
    max_window: int = 300,
    base_window: int = 50
) -> int:
    """
    各residueの活性パターンに基づいてウィンドウサイズを決定
    """
    if len(anomaly_scores) == 0:
        return base_window
        
    # 異常スコアの統計量
    n_events = np.sum(anomaly_scores > 1.0)
    event_density = n_events / len(anomaly_scores)
    score_volatility = np.std(anomaly_scores) / (np.mean(anomaly_scores) + 1e-10)
    
    # スケールファクター計算
    scale_factor = 1.0
    
    # イベント密度による調整
    if event_density > 0.1:  # 頻繁にイベントが起きる
        scale_factor *= 0.7
    elif event_density < 0.02:  # まれにしか起きない
        scale_factor *= 2.0
    
    # 変動性による調整
    if score_volatility > 2.0:  # 激しく変動
        scale_factor *= 0.8
    elif score_volatility < 0.5:  # 安定
        scale_factor *= 1.3
    
    # 最終的なウィンドウサイズ
    adaptive_window = int(base_window * scale_factor)
    return max(min_window, min(max_window, adaptive_window))

# ===============================
# Residue-Level Lambda³ Computation
# ===============================

@njit
def compute_residue_com(trajectory: np.ndarray, atom_indices: np.ndarray) -> np.ndarray:
    """Compute center of mass for a residue across trajectory."""
    n_frames = trajectory.shape[0]
    com = np.zeros((n_frames, 3))
    
    for frame in range(n_frames):
        for atom_idx in atom_indices:
            com[frame] += trajectory[frame, atom_idx]
        com[frame] /= len(atom_indices)
    
    return com

def compute_residue_lambda_structures(
    trajectory: np.ndarray,
    start_frame: int,
    end_frame: int,
    residue_atoms: Dict[int, List[int]],
    window_size: int = 50
) -> Dict[str, np.ndarray]:
    """
    Compute Lambda³ structures at residue level for a specific time window.
    """
    print(f"\n🔬 Computing residue-level Lambda³ for frames {start_frame}-{end_frame}")
    
    n_residues = len(residue_atoms)
    n_frames = end_frame - start_frame
    
    # Initialize arrays
    residue_lambda_f = np.zeros((n_frames-1, n_residues, 3))
    residue_lambda_f_mag = np.zeros((n_frames-1, n_residues))
    residue_rho_t = np.zeros((n_frames, n_residues))
    residue_coupling = np.zeros((n_frames, n_residues, n_residues))
    
    # Compute residue COMs
    residue_coms = np.zeros((n_frames, n_residues, 3))
    for res_id, atoms in residue_atoms.items():
        residue_coms[:, res_id] = compute_residue_com(
            trajectory[start_frame:end_frame], 
            np.array(atoms)
        )
    
    # 1. Residue-level ΛF
    for frame in range(n_frames-1):
        residue_lambda_f[frame] = residue_coms[frame+1] - residue_coms[frame]
        residue_lambda_f_mag[frame] = np.linalg.norm(residue_lambda_f[frame], axis=1)
    
    # 2. Residue-level ρT (local tension)
    for frame in range(n_frames):
        for res_id in range(n_residues):
            # Local window
            local_start = max(0, frame - window_size//2)
            local_end = min(n_frames, frame + window_size//2)
            
            # Local COM variance
            local_coms = residue_coms[local_start:local_end, res_id]
            if len(local_coms) > 1:
                cov = np.cov(local_coms.T)
                residue_rho_t[frame, res_id] = np.trace(cov)
    
    # 3. Residue-residue coupling (simplified)
    for frame in range(n_frames):
        for res_i in range(n_residues):
            for res_j in range(res_i+1, n_residues):
                # Distance-based coupling
                dist = np.linalg.norm(residue_coms[frame, res_i] - residue_coms[frame, res_j])
                residue_coupling[frame, res_i, res_j] = 1.0 / (1.0 + dist)
                residue_coupling[frame, res_j, res_i] = residue_coupling[frame, res_i, res_j]
    
    return {
        'residue_lambda_f': residue_lambda_f,
        'residue_lambda_f_mag': residue_lambda_f_mag,
        'residue_rho_t': residue_rho_t,
        'residue_coupling': residue_coupling,
        'residue_coms': residue_coms
    }

# ===============================
# Enhanced Anomaly Detection
# ===============================

def detect_residue_anomalies(
    residue_structures: Dict[str, np.ndarray],
    sensitivity: float = 1.0  # より低い閾値
) -> Dict[int, np.ndarray]:
    """
    Detect anomalies for each residue with adaptive sensitivity.
    """
    n_frames, n_residues = residue_structures['residue_rho_t'].shape
    residue_anomaly_scores = {}
    
    for res_id in range(n_residues):
        # ΛF magnitude anomalies (already n_frames-1)
        lambda_f_anomaly = detect_local_anomalies(
            residue_structures['residue_lambda_f_mag'][:, res_id],
            window=50
        )
        
        # ρT anomalies (n_frames)
        rho_t_anomaly = detect_local_anomalies(
            residue_structures['residue_rho_t'][:, res_id],
            window=50
        )
        
        # Combined score - align sizes correctly
        min_len = min(len(lambda_f_anomaly), len(rho_t_anomaly))
        combined = (lambda_f_anomaly[:min_len] + rho_t_anomaly[:min_len]) / 2
        
        # Adaptive sensitivity based on residue activity
        residue_activity = np.mean(combined)
        adaptive_sensitivity = sensitivity * (1 + 0.5 * residue_activity)
        
        # Find significant anomalies
        if np.max(combined) > adaptive_sensitivity:
            residue_anomaly_scores[res_id] = combined
    
    return residue_anomaly_scores

# ===============================
# Advanced Causality Detection
# ===============================

@njit
def calculate_structural_causality(
    anomaly_i: np.ndarray,
    anomaly_j: np.ndarray,
    lag_window: int = 200,
    event_threshold: float = 1.0
) -> Tuple[np.ndarray, float, int]:
    """
    構造的因果関係の計算（同期に依存しない）
    Lambda³理論：ΔΛCの伝播パターン
    """
    n_lags = lag_window
    causality_profile = np.zeros(n_lags)
    
    # イベント検出
    events_i = (anomaly_i > event_threshold).astype(np.float64)
    events_j = (anomaly_j > event_threshold).astype(np.float64)
    
    for lag in range(1, n_lags):
        if lag < len(events_i):
            # 構造変化の伝播確率
            cause = events_i[:-lag]
            effect = events_j[lag:]
            
            # 条件付き確率 P(effect|cause)
            cause_mask = cause > 0
            if np.sum(cause_mask) > 0:
                causality_profile[lag] = np.mean(effect[cause_mask])
    
    # 最大因果強度とその遅延
    max_causality = np.max(causality_profile)
    optimal_lag = np.argmax(causality_profile)
    
    return causality_profile, max_causality, optimal_lag

def detect_residue_network_enhanced(
    residue_anomaly_scores: Dict[int, np.ndarray],
    residue_coupling: np.ndarray,
    base_lag_window: int = 100,
    causality_threshold: float = 0.15,
    sync_threshold: float = 0.2
) -> Dict[str, any]:
    """
    拡張版：同期と因果を分離したネットワーク検出
    """
    residue_ids = sorted(residue_anomaly_scores.keys())
    n_residues = len(residue_ids)
    
    # 結果格納
    causal_network = []
    sync_network = []
    async_strong_bonds = []  # 同期なき強い結びつき！
    
    # 各residueのアダプティブウィンドウを計算
    adaptive_windows = {}
    for res_id, scores in residue_anomaly_scores.items():
        adaptive_windows[res_id] = compute_residue_adaptive_window(scores)
    
    print(f"\n🎯 Adaptive Windows for top residues:")
    for res_id, window in list(adaptive_windows.items())[:5]:
        print(f"   Residue {res_id+1}: {window} frames")
    
    for i, res_i in enumerate(residue_ids):
        for j, res_j in enumerate(residue_ids[i+1:], i+1):
            scores_i = residue_anomaly_scores[res_i]
            scores_j = residue_anomaly_scores[res_j]
            
            # ペアに最適なウィンドウサイズを決定
            pair_window = int((adaptive_windows[res_i] + adaptive_windows[res_j]) / 2)
            
            # 1. 構造的因果関係（時間遅れOK）
            causality_ij, max_caus_ij, lag_ij = calculate_structural_causality(
                scores_i, scores_j, pair_window
            )
            causality_ji, max_caus_ji, lag_ji = calculate_structural_causality(
                scores_j, scores_i, pair_window
            )
            
            # 2. 即時同期率（lag=0での相関）
            if len(scores_i) > 10 and len(scores_j) > 10:
                sync_rate = np.corrcoef(scores_i, scores_j)[0, 1]
            else:
                sync_rate = 0.0
            
            # 3. 空間的結合（距離ベース）
            avg_coupling = np.mean(residue_coupling[:, res_i, res_j])
            
            # 動的閾値（residueの活性度に応じて調整）
            activity_i = np.mean(scores_i > 1.0)
            activity_j = np.mean(scores_j > 1.0)
            dynamic_causality_threshold = causality_threshold * (1 - 0.5 * min(activity_i, activity_j))
            
            # ネットワーク分類
            has_causality = max(max_caus_ij, max_caus_ji) > dynamic_causality_threshold
            has_sync = abs(sync_rate) > sync_threshold
            
            if has_causality:
                # 因果ネットワークに追加
                if max_caus_ij > max_caus_ji:
                    causal_network.append({
                        'from': res_i,
                        'to': res_j,
                        'strength': max_caus_ij,
                        'lag': lag_ij,
                        'type': 'causal',
                        'window_used': pair_window
                    })
                else:
                    causal_network.append({
                        'from': res_j,
                        'to': res_i,
                        'strength': max_caus_ji,
                        'lag': lag_ji,
                        'type': 'causal',
                        'window_used': pair_window
                    })
                
                # 同期なき強い結びつきの検出！
                if has_causality and not has_sync:
                    async_strong_bonds.append({
                        'residue_pair': (res_i, res_j),
                        'causality': max(max_caus_ij, max_caus_ji),
                        'sync_rate': sync_rate,
                        'optimal_lag': lag_ij if max_caus_ij > max_caus_ji else lag_ji,
                        'coupling': avg_coupling,
                        'window': pair_window
                    })
            
            if has_sync:
                # 同期ネットワークに追加
                sync_network.append({
                    'residue_pair': (res_i, res_j),
                    'sync_strength': abs(sync_rate),
                    'type': 'synchronous'
                })
    
    return {
        'causal_network': causal_network,
        'sync_network': sync_network,
        'async_strong_bonds': async_strong_bonds,
        'n_causal_links': len(causal_network),
        'n_sync_links': len(sync_network),
        'n_async_bonds': len(async_strong_bonds),
        'adaptive_windows': adaptive_windows
    }

# ===============================
# Event Analysis Functions
# ===============================

def analyze_macro_event(
    trajectory: np.ndarray,
    event_name: str,
    start_frame: int,
    end_frame: int,
    residue_atoms: Dict[int, List[int]],
    residue_names: Dict[int, str],
    sensitivity: float = 1.0,  # 下げた
    correlation_threshold: float = 0.15  # 下げた
) -> ResidueLevelAnalysis:
    """
    Perform detailed residue-level analysis for a single macro event.
    Enhanced with adaptive windows and async bond detection.
    """
    print(f"\n🎯 Analyzing {event_name} at residue level...")
    
    # Compute residue-level Lambda structures
    residue_structures = compute_residue_lambda_structures(
        trajectory, start_frame, end_frame, residue_atoms
    )
    
    # Detect anomalies per residue
    residue_anomaly_scores = detect_residue_anomalies(residue_structures, sensitivity)
    
    # Enhanced network detection with adaptive windows
    print("\n🔬 Enhanced Network Detection with Adaptive Windows...")
    network_results = detect_residue_network_enhanced(
        residue_anomaly_scores,
        residue_structures['residue_coupling'],
        base_lag_window=100,
        causality_threshold=correlation_threshold,
        sync_threshold=0.2
    )
    
    print(f"   Found {network_results['n_causal_links']} causal links")
    print(f"   Found {network_results['n_sync_links']} synchronous links")
    print(f"   Found {network_results['n_async_bonds']} async strong bonds! ✨")
    
    # Show async strong bonds
    if network_results['async_strong_bonds']:
        print("\n   🔥 Top Async Strong Bonds:")
        for bond in network_results['async_strong_bonds'][:3]:
            print(f"      R{bond['residue_pair'][0]+1} ⟷ R{bond['residue_pair'][1]+1}: "
                  f"causality={bond['causality']:.3f}, sync={bond['sync_rate']:.3f}, "
                  f"lag={bond['optimal_lag']} frames, window={bond['window']}")
    
    # Find initiator residues (earliest anomalies)
    initiators = []
    residue_events = []
    
    for res_id, scores in residue_anomaly_scores.items():
        # Find first significant peak
        peaks, properties = find_peaks(scores, height=sensitivity, distance=50)
        
        if len(peaks) > 0:
            first_peak = peaks[0]
            peak_height = properties['peak_heights'][0]
            
            event = ResidueEvent(
                residue_id=res_id,
                residue_name=residue_names.get(res_id, f"RES{res_id}"),
                start_frame=start_frame + first_peak,
                end_frame=start_frame + min(first_peak + 100, len(scores)),
                peak_lambda_f=float(peak_height),
                propagation_delay=first_peak,
                role='initiator' if first_peak < 50 else 'propagator',
                adaptive_window=network_results['adaptive_windows'].get(res_id, 100)
            )
            residue_events.append(event)
            
            if first_peak < 50:  # Early responders
                initiators.append(res_id)
    
    # Convert network results to causality chains
    causality_chains = [
        (link['from'], link['to'], link['strength'])
        for link in network_results['causal_network']
    ]
    
    # Build propagation paths
    propagation_paths = build_propagation_paths(initiators, causality_chains)
    
    print(f"   Found {len(initiators)} initiator residues")
    print(f"   Detected {len(causality_chains)} causal relationships")
    
    return ResidueLevelAnalysis(
        event_name=event_name,
        macro_start=start_frame,
        macro_end=end_frame,
        residue_events=residue_events,
        causality_chain=causality_chains,
        initiator_residues=initiators,
        key_propagation_paths=propagation_paths[:5],  # Top 5 paths
        async_strong_bonds=network_results['async_strong_bonds'],
        sync_network=network_results['sync_network'],
        network_stats={
            'n_causal': network_results['n_causal_links'],
            'n_sync': network_results['n_sync_links'],
            'n_async': network_results['n_async_bonds'],
            'mean_adaptive_window': np.mean(list(network_results['adaptive_windows'].values()))
        }
    )

def build_propagation_paths(
    initiators: List[int],
    causality_chains: List[Tuple[int, int, float]],
    max_depth: int = 5
) -> List[List[int]]:
    """
    Build propagation paths from initiator residues.
    """
    # Build adjacency graph
    graph = {}
    for res1, res2, weight in causality_chains:
        if res1 not in graph:
            graph[res1] = []
        graph[res1].append((res2, weight))
    
    paths = []
    
    def dfs(current: int, path: List[int], depth: int):
        if depth >= max_depth:
            paths.append(path.copy())
            return
        
        if current in graph:
            for neighbor, weight in graph[current]:
                if neighbor not in path:  # Avoid cycles
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()
        else:
            paths.append(path.copy())
    
    # Start from each initiator
    for initiator in initiators:
        dfs(initiator, [initiator], 0)
    
    # Sort by path length and uniqueness
    unique_paths = []
    seen = set()
    for path in sorted(paths, key=len, reverse=True):
        path_tuple = tuple(path)
        if path_tuple not in seen and len(path) > 1:
            seen.add(path_tuple)
            unique_paths.append(path)
    
    return unique_paths

# ===============================
# Two-Stage Analysis Pipeline
# ===============================

def perform_two_stage_analysis(
    trajectory: np.ndarray,
    macro_result: 'MDLambda3Result',
    detected_events: List[Tuple[int, int, str]],
    n_residues: int = 129,
    sensitivity: float = 1.0,  # 下げた
    correlation_threshold: float = 0.15  # 下げた
) -> TwoStageLambda3Result:
    """
    Perform two-stage analysis: macro events → residue-level causality.
    Enhanced with adaptive windows and network statistics.
    """
    print("\n" + "="*60)
    print("=== Two-Stage Lambda³ Analysis (v2.0) ===")
    print("="*60)
    
    # Create residue mapping
    residue_atoms = create_residue_mapping(trajectory.shape[1], n_residues)
    residue_names = get_residue_names()
    
    # Analyze each detected macro event
    residue_analyses = {}
    all_important_residues = {}
    global_async_bonds = []
    
    for start, end, event_name in detected_events:
        print(f"\n📍 Processing {event_name}...")
        
        analysis = analyze_macro_event(
            trajectory,
            event_name,
            start,
            end,
            residue_atoms,
            residue_names,
            sensitivity=sensitivity,
            correlation_threshold=correlation_threshold
        )
        
        residue_analyses[event_name] = analysis
        
        # Collect async bonds
        global_async_bonds.extend(analysis.async_strong_bonds)
        
        # Track globally important residues
        for event in analysis.residue_events:
            res_id = event.residue_id
            if res_id not in all_important_residues:
                all_important_residues[res_id] = 0
            # Weight by both peak intensity and adaptive window
            importance = event.peak_lambda_f * (1 + 0.1 * (100 / event.adaptive_window))
            all_important_residues[res_id] += importance
    
    # Identify key intervention points
    sorted_residues = sorted(all_important_residues.items(), 
                           key=lambda x: x[1], reverse=True)
    intervention_points = [res_id for res_id, score in sorted_residues[:10]]
    
    # Global network statistics
    total_causal_links = sum(a.network_stats['n_causal'] for a in residue_analyses.values())
    total_sync_links = sum(a.network_stats['n_sync'] for a in residue_analyses.values())
    total_async_bonds = sum(a.network_stats['n_async'] for a in residue_analyses.values())
    
    global_network_stats = {
        'total_causal_links': total_causal_links,
        'total_sync_links': total_sync_links,
        'total_async_bonds': total_async_bonds,
        'async_to_causal_ratio': total_async_bonds / (total_causal_links + 1e-10),
        'mean_adaptive_window': np.mean([a.network_stats['mean_adaptive_window'] 
                                       for a in residue_analyses.values()])
    }
    
    print("\n🎯 Global Analysis Complete!")
    print(f"   Key residues identified: {len(all_important_residues)}")
    print(f"   Total causal links: {total_causal_links}")
    print(f"   Total async strong bonds: {total_async_bonds} ({global_network_stats['async_to_causal_ratio']:.1%})")
    print(f"   Mean adaptive window: {global_network_stats['mean_adaptive_window']:.1f} frames")
    print(f"   Suggested intervention points: {intervention_points[:5]}")
    
    return TwoStageLambda3Result(
        macro_result=macro_result,
        residue_analyses=residue_analyses,
        global_residue_importance=all_important_residues,
        suggested_intervention_points=intervention_points,
        global_network_stats=global_network_stats
    )

# ===============================
# Enhanced Visualization Functions
# ===============================

def visualize_residue_causality(
    analysis: ResidueLevelAnalysis,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Enhanced visualization with async bonds and adaptive windows.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Timeline of residue events with adaptive windows
    ax1 = axes[0, 0]
    ax1.set_title(f"{analysis.event_name} - Residue Event Timeline (Adaptive Windows)")
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Residue ID")
    
    for event in analysis.residue_events:
        color = 'red' if event.role == 'initiator' else 'blue'
        width = event.end_frame - event.start_frame
        
        # Show adaptive window size by bar thickness
        height = 0.4 + 0.4 * (100 / event.adaptive_window)  # Inverse: smaller window = thicker bar
        
        ax1.barh(event.residue_id, 
                width,
                left=event.start_frame,
                height=height,
                color=color,
                alpha=0.7)
    
    # 2. Causality network
    ax2 = axes[0, 1]
    ax2.set_title("Causality Network")
    
    # Simple network visualization
    if analysis.key_propagation_paths:
        for i, path in enumerate(analysis.key_propagation_paths[:3]):
            y_offset = i * 0.3
            for j in range(len(path) - 1):
                ax2.arrow(j, y_offset, 0.8, 0,
                         head_width=0.1, head_length=0.1,
                         fc=f'C{i}', ec=f'C{i}')
                ax2.text(j, y_offset + 0.15, f"R{path[j]+1}", 
                        ha='center', fontsize=10)
            # Last residue
            ax2.text(len(path)-1, y_offset + 0.15, f"R{path[-1]+1}", 
                    ha='center', fontsize=10)
    
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.axis('off')
    
    # 3. Async Strong Bonds
    ax3 = axes[1, 0]
    ax3.set_title("Async Strong Bonds (同期なき強い結びつき)")
    
    if analysis.async_strong_bonds:
        bond_data = []
        for bond in analysis.async_strong_bonds[:10]:  # Top 10
            res1, res2 = bond['residue_pair']
            bond_data.append({
                'pair': f"R{res1+1}-R{res2+1}",
                'causality': bond['causality'],
                'sync': abs(bond['sync_rate']),
                'lag': bond['optimal_lag']
            })
        
        # Plot as scatter
        x = [b['sync'] for b in bond_data]
        y = [b['causality'] for b in bond_data]
        colors = [b['lag'] for b in bond_data]
        
        scatter = ax3.scatter(x, y, c=colors, cmap='viridis', s=100, alpha=0.7)
        
        # Add labels for top bonds
        for i, b in enumerate(bond_data[:5]):
            ax3.annotate(b['pair'], (x[i], y[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel("Synchronization Rate")
        ax3.set_ylabel("Causality Strength")
        ax3.axvline(x=0.2, color='r', linestyle='--', alpha=0.5, label='Sync threshold')
        ax3.axhline(y=0.15, color='b', linestyle='--', alpha=0.5, label='Causality threshold')
        ax3.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Optimal Lag (frames)')
    
    # 4. Network Statistics
    ax4 = axes[1, 1]
    ax4.set_title("Network Statistics")
    ax4.axis('off')
    
    stats_text = f"""
    Network Type Distribution:
    - Causal Links: {analysis.network_stats['n_causal']}
    - Synchronous Links: {analysis.network_stats['n_sync']}
    - Async Strong Bonds: {analysis.network_stats['n_async']}
    
    Mean Adaptive Window: {analysis.network_stats['mean_adaptive_window']:.1f} frames
    
    Top Initiator Residues:
    """
    for i, res_id in enumerate(analysis.initiator_residues[:5]):
        stats_text += f"\n    {i+1}. Residue {res_id+1}"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def create_intervention_report(
    two_stage_result: TwoStageLambda3Result,
    save_path: Optional[str] = None
) -> str:
    """
    Create an enhanced report with network insights.
    """
    report = []
    report.append("="*60)
    report.append("Lambda³ Two-Stage Analysis Report (v2.0)")
    report.append("Residue-Level Intervention Recommendations")
    report.append("="*60)
    report.append("")
    
    # Network summary
    report.append("🌐 NETWORK SUMMARY")
    report.append("-"*30)
    stats = two_stage_result.global_network_stats
    report.append(f"Total Causal Links: {stats['total_causal_links']}")
    report.append(f"Total Sync Links: {stats['total_sync_links']}")
    report.append(f"Total Async Strong Bonds: {stats['total_async_bonds']}")
    report.append(f"Async/Causal Ratio: {stats['async_to_causal_ratio']:.1%}")
    report.append(f"Mean Adaptive Window: {stats['mean_adaptive_window']:.1f} frames")
    report.append("")
    
    # Top intervention targets
    report.append("🎯 TOP INTERVENTION TARGETS")
    report.append("-"*30)
    
    for i, res_id in enumerate(two_stage_result.suggested_intervention_points[:5]):
        importance = two_stage_result.global_residue_importance[res_id]
        report.append(f"{i+1}. Residue {res_id+1}: Score = {importance:.2f}")
        
        # Find which events this residue participates in
        events_involved = []
        for event_name, analysis in two_stage_result.residue_analyses.items():
            for res_event in analysis.residue_events:
                if res_event.residue_id == res_id:
                    events_involved.append(event_name)
                    break
        
        if events_involved:
            report.append(f"   Involved in: {', '.join(events_involved)}")
            
        # Check if involved in async bonds
        async_bonds = []
        for event_name, analysis in two_stage_result.residue_analyses.items():
            for bond in analysis.async_strong_bonds:
                if res_id in bond['residue_pair']:
                    async_bonds.append(bond)
        
        if async_bonds:
            report.append(f"   Part of {len(async_bonds)} async strong bonds")
    
    report.append("")
    report.append("📊 EVENT-SPECIFIC FINDINGS")
    report.append("-"*30)
    
    # Key findings per event
    for event_name, analysis in two_stage_result.residue_analyses.items():
        report.append(f"\n{event_name}:")
        
        if analysis.initiator_residues:
            initiators = [f"R{r+1}" for r in analysis.initiator_residues[:3]]
            report.append(f"  Initiators: {', '.join(initiators)}")
        
        if analysis.key_propagation_paths:
            path = analysis.key_propagation_paths[0]
            path_str = " → ".join([f"R{r+1}" for r in path])
            report.append(f"  Key path: {path_str}")
            
        # Network stats
        report.append(f"  Network: {analysis.network_stats['n_causal']} causal, "
                     f"{analysis.network_stats['n_sync']} sync, "
                     f"{analysis.network_stats['n_async']} async bonds")
    
    report.append("")
    report.append("🔥 ASYNC STRONG BONDS (同期なき強い結びつき)")
    report.append("-"*30)
    
    # Collect all async bonds
    all_async_bonds = []
    for analysis in two_stage_result.residue_analyses.values():
        all_async_bonds.extend(analysis.async_strong_bonds)
    
    # Sort by causality strength
    all_async_bonds.sort(key=lambda x: x['causality'], reverse=True)
    
    for i, bond in enumerate(all_async_bonds[:5]):
        res1, res2 = bond['residue_pair']
        report.append(f"{i+1}. R{res1+1} ⟷ R{res2+1}:")
        report.append(f"   Causality: {bond['causality']:.3f}")
        report.append(f"   Sync Rate: {bond['sync_rate']:.3f}")
        report.append(f"   Optimal Lag: {bond['optimal_lag']} frames")
    
    report.append("")
    report.append("💊 DRUG DESIGN IMPLICATIONS")
    report.append("-"*30)
    report.append("1. Target async strong bonds for disrupting pathological cascades")
    report.append("2. Stabilize initiator residues with small adaptive windows")
    report.append("3. Consider time-delayed interventions based on optimal lags")
    report.append("4. Focus on residues with high causality but low synchronization")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text

# ===============================
# ALS-Specific Analysis Functions
# ===============================

def analyze_aggregation_pathway(
    two_stage_result: TwoStageLambda3Result,
    aggregation_event_name: str = 'aggregation_onset'
) -> Dict[str, any]:
    """
    Detailed analysis of aggregation pathway for ALS research.
    Enhanced with async bond detection.
    """
    if aggregation_event_name not in two_stage_result.residue_analyses:
        return {}
    
    analysis = two_stage_result.residue_analyses[aggregation_event_name]
    
    # Identify hydrophobic residues (simplified - would use real data)
    hydrophobic_residues = [10, 15, 28, 45, 62, 63, 78, 92, 108, 115]  # Example
    
    # Find hydrophobic exposure events
    exposed_hydrophobic = []
    for event in analysis.residue_events:
        if event.residue_id in hydrophobic_residues:
            exposed_hydrophobic.append(event)
    
    # Sort by timing
    exposed_hydrophobic.sort(key=lambda x: x.propagation_delay)
    
    # Find async bonds involving hydrophobic residues
    hydrophobic_async_bonds = []
    for bond in analysis.async_strong_bonds:
        if any(res in hydrophobic_residues for res in bond['residue_pair']):
            hydrophobic_async_bonds.append(bond)
    
    # Build aggregation timeline
    aggregation_timeline = {
        'nucleation_site': exposed_hydrophobic[0] if exposed_hydrophobic else None,
        'growth_sequence': exposed_hydrophobic[:5],
        'critical_size_frame': analysis.macro_start + 5000,  # Estimate
        'irreversible_point': analysis.macro_start + 7000,
        'async_nucleation_bonds': hydrophobic_async_bonds[:3]  # Top async bonds
    }
    
    return {
        'exposed_hydrophobic_count': len(exposed_hydrophobic),
        'first_exposure': exposed_hydrophobic[0] if exposed_hydrophobic else None,
        'timeline': aggregation_timeline,
        'intervention_window': (analysis.macro_start, 
                              aggregation_timeline['irreversible_point']),
        'hydrophobic_async_bonds': len(hydrophobic_async_bonds),
        'mean_adaptive_window': np.mean([e.adaptive_window for e in exposed_hydrophobic]) if exposed_hydrophobic else 0
    }

# ===============================
# Main Demo Function
# ===============================

def demo_two_stage_analysis():
    """
    Demo two-stage analysis on 100k lysozyme trajectory.
    Enhanced version with adaptive windows and async bonds.
    """
    print("🔬 Lambda³ Two-Stage Analysis Demo v2.0")
    print("Stage 1: Macro events (✓ Complete)")
    print("Stage 2: Residue-level causality with adaptive windows (Starting...)")
    
    # Load trajectory
    try:
        trajectory = np.load('lysozyme_100k_final_challenge.npy')
        print(f"\n✓ Loaded trajectory: {trajectory.shape}")
        
        # Key events for analysis
        key_events = [
            (40000, 45000, 'domain_shift'),
            (50000, 53000, 'rapid_partial_unfold'),
            (85000, 95000, 'aggregation_onset')
        ]
        
        # Placeholder for macro_result
        macro_result = None
        
        # Perform two-stage analysis with enhanced parameters
        result = perform_two_stage_analysis(
            trajectory,
            macro_result,
            key_events,
            n_residues=129,
            sensitivity=1.0,  # Lower sensitivity
            correlation_threshold=0.15  # Lower threshold
        )
        
        # Generate report
        report = create_intervention_report(result, "lambda3_intervention_report_v2.txt")
        print("\n" + report)
        
        # Visualize key event
        if 'domain_shift' in result.residue_analyses:
            fig = visualize_residue_causality(
                result.residue_analyses['domain_shift'],
                "domain_shift_causality_v2.png"
            )
            plt.show()
        
        return result
        
    except FileNotFoundError:
        print("❌ Error: Trajectory file not found!")
        print("Please run the main Lambda³ analysis first.")
        return None

if __name__ == "__main__":
    print("\n🚀 Lambda³ Residue-Level Focus Extension v2.0")
    print("Enhanced with Adaptive Windows & Async Strong Bonds")
    print("Taking Lambda³ analysis to the atomic scale with smarter detection...")
    
    result = demo_two_stage_analysis()
    
    if result:
        print("\n✨ Two-stage analysis complete!")
        print("New features:")
        print("  - Adaptive window sizing per residue")
        print("  - Async strong bond detection")
        print("  - Enhanced network statistics")
        print("  - Lower thresholds for better sensitivity")
