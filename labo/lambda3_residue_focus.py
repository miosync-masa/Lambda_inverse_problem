"""
Lambda³ Residue-Level Focus Analysis Extension
Two-stage hierarchical analysis: Macro events → Micro (residue) causality
Author: Lambda³ Project
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

@dataclass
class TwoStageLambda3Result:
    """Complete two-stage analysis results"""
    macro_result: 'MDLambda3Result'  # From main analysis
    residue_analyses: Dict[str, ResidueLevelAnalysis]
    global_residue_importance: Dict[int, float]
    suggested_intervention_points: List[int]

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
# Anomaly Detection at Residue Level
# ===============================

def detect_residue_anomalies(
    residue_structures: Dict[str, np.ndarray],
    sensitivity: float = 2.0
) -> Dict[int, np.ndarray]:
    """
    Detect anomalies for each residue.
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
        # lambda_f_anomaly is already (n_frames-1)
        # rho_t_anomaly is (n_frames), so we need to trim it
        min_len = min(len(lambda_f_anomaly), len(rho_t_anomaly))
        combined = (lambda_f_anomaly[:min_len] + rho_t_anomaly[:min_len]) / 2
        
        # Find significant anomalies
        if np.max(combined) > sensitivity:
            residue_anomaly_scores[res_id] = combined
    
    return residue_anomaly_scores

# ===============================
# Causality Chain Detection
# ===============================

def detect_causality_chain(
    residue_anomaly_scores: Dict[int, np.ndarray],
    residue_coupling: np.ndarray,
    lag_window: int = 100
) -> List[Tuple[int, int, float]]:
    """
    Detect causal relationships between residues based on temporal correlation.
    """
    causality_chains = []
    
    residue_ids = sorted(residue_anomaly_scores.keys())
    
    for i, res_i in enumerate(residue_ids):
        for j, res_j in enumerate(residue_ids[i+1:], i+1):
            scores_i = residue_anomaly_scores[res_i]
            scores_j = residue_anomaly_scores[res_j]
            
            # Check for lagged correlation
            max_correlation = 0
            best_lag = 0
            
            for lag in range(0, min(lag_window, len(scores_i)//2)):
                if lag < len(scores_i) and lag < len(scores_j):
                    # Compute correlation with lag
                    valid_len = min(len(scores_i) - lag, len(scores_j))
                    if valid_len > 10:
                        corr = np.corrcoef(scores_i[lag:lag+valid_len], 
                                         scores_j[:valid_len])[0, 1]
                        if abs(corr) > max_correlation:
                            max_correlation = abs(corr)
                            best_lag = lag
            
            # Strong correlation suggests causality
            if max_correlation > 0.5:
                # Also check spatial proximity
                avg_coupling = np.mean(residue_coupling[:, res_i, res_j])
                if avg_coupling > 0.1 or max_correlation > 0.7:
                    causality_chains.append((res_i, res_j, max_correlation))
    
    return sorted(causality_chains, key=lambda x: x[2], reverse=True)

# ===============================
# Event Analysis Functions
# ===============================

def analyze_macro_event(
    trajectory: np.ndarray,
    event_name: str,
    start_frame: int,
    end_frame: int,
    residue_atoms: Dict[int, List[int]],
    residue_names: Dict[int, str]
) -> ResidueLevelAnalysis:
    """
    Perform detailed residue-level analysis for a single macro event.
    """
    print(f"\n🎯 Analyzing {event_name} at residue level...")
    
    # Compute residue-level Lambda structures
    residue_structures = compute_residue_lambda_structures(
        trajectory, start_frame, end_frame, residue_atoms
    )
    
    # Detect anomalies per residue
    residue_anomaly_scores = detect_residue_anomalies(residue_structures)
    
    # Find initiator residues (earliest anomalies)
    initiators = []
    residue_events = []
    
    for res_id, scores in residue_anomaly_scores.items():
        # Find first significant peak
        peaks, properties = find_peaks(scores, height=2.0, distance=50)
        
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
                role='initiator' if first_peak < 50 else 'propagator'
            )
            residue_events.append(event)
            
            if first_peak < 50:  # Early responders
                initiators.append(res_id)
    
    # Detect causality chains
    causality_chains = detect_causality_chain(
        residue_anomaly_scores,
        residue_structures['residue_coupling'],
        lag_window=100
    )
    
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
        key_propagation_paths=propagation_paths[:5]  # Top 5 paths
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
    n_residues: int = 129
) -> TwoStageLambda3Result:
    """
    Perform two-stage analysis: macro events → residue-level causality.
    """
    print("\n" + "="*60)
    print("=== Two-Stage Lambda³ Analysis ===")
    print("="*60)
    
    # Create residue mapping
    residue_atoms = create_residue_mapping(trajectory.shape[1], n_residues)
    residue_names = get_residue_names()
    
    # Analyze each detected macro event
    residue_analyses = {}
    all_important_residues = {}
    
    for start, end, event_name in detected_events:
        print(f"\n📍 Processing {event_name}...")
        
        analysis = analyze_macro_event(
            trajectory,
            event_name,
            start,
            end,
            residue_atoms,
            residue_names
        )
        
        residue_analyses[event_name] = analysis
        
        # Track globally important residues
        for event in analysis.residue_events:
            res_id = event.residue_id
            if res_id not in all_important_residues:
                all_important_residues[res_id] = 0
            all_important_residues[res_id] += event.peak_lambda_f
    
    # Identify key intervention points
    sorted_residues = sorted(all_important_residues.items(), 
                           key=lambda x: x[1], reverse=True)
    intervention_points = [res_id for res_id, score in sorted_residues[:10]]
    
    print("\n🎯 Global Analysis Complete!")
    print(f"   Key residues identified: {len(all_important_residues)}")
    print(f"   Suggested intervention points: {intervention_points[:5]}")
    
    return TwoStageLambda3Result(
        macro_result=macro_result,
        residue_analyses=residue_analyses,
        global_residue_importance=all_important_residues,
        suggested_intervention_points=intervention_points
    )

# ===============================
# Visualization Functions
# ===============================

def visualize_residue_causality(
    analysis: ResidueLevelAnalysis,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize causality network for a single event.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Timeline of residue events
    ax1.set_title(f"{analysis.event_name} - Residue Event Timeline")
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Residue ID")
    
    for event in analysis.residue_events:
        color = 'red' if event.role == 'initiator' else 'blue'
        ax1.barh(event.residue_id, 
                event.end_frame - event.start_frame,
                left=event.start_frame,
                height=0.8,
                color=color,
                alpha=0.7)
    
    # 2. Causality network
    ax2.set_title("Causality Network")
    
    # Simple network visualization
    # In practice, would use networkx or similar
    if analysis.key_propagation_paths:
        for i, path in enumerate(analysis.key_propagation_paths[:3]):
            y_offset = i * 0.3
            for j in range(len(path) - 1):
                ax2.arrow(j, y_offset, 0.8, 0,
                         head_width=0.1, head_length=0.1,
                         fc=f'C{i}', ec=f'C{i}')
                ax2.text(j, y_offset + 0.15, f"R{path[j]}", 
                        ha='center', fontsize=10)
            # Last residue
            ax2.text(len(path)-1, y_offset + 0.15, f"R{path[-1]}", 
                    ha='center', fontsize=10)
    
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def create_intervention_report(
    two_stage_result: TwoStageLambda3Result,
    save_path: Optional[str] = None
) -> str:
    """
    Create a report summarizing intervention recommendations.
    """
    report = []
    report.append("="*60)
    report.append("Lambda³ Two-Stage Analysis Report")
    report.append("Residue-Level Intervention Recommendations")
    report.append("="*60)
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
    
    report.append("")
    report.append("💊 DRUG DESIGN IMPLICATIONS")
    report.append("-"*30)
    report.append("1. Stabilize initiator residues to prevent cascade")
    report.append("2. Disrupt key propagation paths")
    report.append("3. Target high-coupling residue pairs")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text

# ===============================
# Main Demo Function
# ===============================

def demo_two_stage_analysis():
    """
    Demo two-stage analysis on 100k lysozyme trajectory.
    """
    print("🔬 Lambda³ Two-Stage Analysis Demo")
    print("Stage 1: Macro events (✓ Complete)")
    print("Stage 2: Residue-level causality (Starting...)")
    
    # This would normally load the results from the main analysis
    # For demo purposes, we'll use the known events
    
    events = [
        (5000, 15000, 'subtle_breathing'),
        (18000, 22000, 'ligand_binding_effect'),
        (25000, 35000, 'slow_helix_destabilization'),
        (40000, 45000, 'domain_shift'),
        (50000, 53000, 'rapid_partial_unfold'),
        (58000, 65000, 'transient_refolding_attempt'),
        (65000, 75000, 'misfolded_intermediate'),
        (78000, 83000, 'hydrophobic_exposure'),
        (85000, 95000, 'aggregation_onset')
    ]
    
    # Load trajectory
    try:
        trajectory = np.load('lysozyme_100k_final_challenge.npy')
        print(f"\n✓ Loaded trajectory: {trajectory.shape}")
        
        # For demo, analyze just the most interesting events
        key_events = [
            (40000, 45000, 'domain_shift'),
            (50000, 53000, 'rapid_partial_unfold'),
            (85000, 95000, 'aggregation_onset')
        ]
        
        # Placeholder for macro_result (would come from main analysis)
        macro_result = None
        
        # Perform two-stage analysis
        result = perform_two_stage_analysis(
            trajectory,
            macro_result,
            key_events,
            n_residues=129
        )
        
        # Generate report
        report = create_intervention_report(result, "lambda3_intervention_report.txt")
        print("\n" + report)
        
        # Visualize key event
        if 'domain_shift' in result.residue_analyses:
            fig = visualize_residue_causality(
                result.residue_analyses['domain_shift'],
                "domain_shift_causality.png"
            )
            plt.show()
        
        return result
        
    except FileNotFoundError:
        print("❌ Error: Trajectory file not found!")
        print("Please run the main Lambda³ analysis first.")
        return None

# ===============================
# ALS-Specific Analysis Functions
# ===============================

def analyze_aggregation_pathway(
    two_stage_result: TwoStageLambda3Result,
    aggregation_event_name: str = 'aggregation_onset'
) -> Dict[str, any]:
    """
    Detailed analysis of aggregation pathway for ALS research.
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
    
    # Build aggregation timeline
    aggregation_timeline = {
        'nucleation_site': exposed_hydrophobic[0] if exposed_hydrophobic else None,
        'growth_sequence': exposed_hydrophobic[:5],
        'critical_size_frame': analysis.macro_start + 5000,  # Estimate
        'irreversible_point': analysis.macro_start + 7000
    }
    
    return {
        'exposed_hydrophobic_count': len(exposed_hydrophobic),
        'first_exposure': exposed_hydrophobic[0] if exposed_hydrophobic else None,
        'timeline': aggregation_timeline,
        'intervention_window': (analysis.macro_start, 
                              aggregation_timeline['irreversible_point'])
    }

if __name__ == "__main__":
    print("\n🚀 Lambda³ Residue-Level Focus Extension")
    print("Taking Lambda³ analysis to the atomic scale...")
    
    result = demo_two_stage_analysis()
    
    if result:
        print("\n✨ Two-stage analysis complete!")
        print("Next steps:")
        print("  1. Validate findings with experimental data")
        print("  2. Design interventions targeting key residues")
        print("  3. Test on ALS-related proteins")
