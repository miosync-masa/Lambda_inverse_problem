# ==========================================================
# Λ³ Regime-Aware Bayesian Extension (Fixed)
# ----------------------------------------------------
# Fixed version with corrected array operations
# ==========================================================

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, njit
import copy

# Import from main Lambda³ module
from lambda3_zeroshot_tensor_field import (
    L3Config, 
    calc_lambda3_features, 
    Lambda3BayesianLogger,
    extract_interaction_coefficients as extract_base_coefficients,
    load_csv_data
)

# ===============================
# HIERARCHICAL REGIME CONFIGURATION
# ===============================
@dataclass
class HierarchicalRegimeConfig:
    """Configuration for hierarchical regime detection"""
    # Global market regime settings
    n_global_regimes: int = 3  # Bull/Neutral/Bear
    global_regime_names: List[str] = field(default_factory=lambda: ['Bull', 'Neutral', 'Bear'])
    
    # Pair-specific sub-regime settings
    n_sub_regimes: int = 2  # Sub-regimes within each global regime
    enable_sub_regimes: bool = True
    
    # Detection parameters
    min_regime_size: int = 30  # Minimum points for valid regime
    regime_overlap_window: int = 5  # Transition smoothing window
    use_gmm: bool = True  # Use Gaussian Mixture Model vs K-means
    
    # Regime stability criteria
    stability_threshold: float = 0.7  # Minimum probability for regime assignment
    transition_penalty: float = 0.1  # Penalty for frequent transitions
    
    # Feature selection for regime detection
    use_market_features: bool = True  # Include market-wide features
    use_technical_indicators: bool = True  # Include RSI, volatility ratios
    
    # Bayesian settings per regime
    regime_specific_priors: bool = True  # Different priors for each regime
    adaptive_sampling: bool = True  # Adjust MCMC parameters by regime size

# ===============================
# HELPER FUNCTIONS
# ===============================
def calculate_returns(data: np.ndarray) -> np.ndarray:
    """
    Calculate percentage returns safely
    
    Parameters:
    -----------
    data : np.ndarray
        Price or value data
        
    Returns:
    --------
    np.ndarray
        Percentage returns (same length as input)
    """
    returns = np.zeros_like(data)
    if len(data) > 1:
        # Calculate returns for all elements except first
        returns[1:] = (data[1:] - data[:-1]) / (np.abs(data[:-1]) + 1e-8)
    return returns

def create_regime_aware_config(
    base_config: L3Config,
    regime_name: str,
    n_regime_points: int
) -> L3Config:
    """
    Create regime-specific configuration while preserving Lambda³ structure
    
    Parameters:
    -----------
    base_config : L3Config
        Base configuration
    regime_name : str
        'Bull', 'Bear', or 'Neutral'
    n_regime_points : int
        Number of data points in this regime
        
    Returns:
    --------
    L3Config
        Adjusted configuration for regime
    """
    config = copy.deepcopy(base_config)
    
    # Adjust sampling based on regime characteristics
    if regime_name == 'Bull':
        # Bull: More stable, can use fewer samples
        if n_regime_points < 200:
            config.draws = min(4000, base_config.draws)
            config.tune = min(4000, base_config.tune)
        config.target_accept = 0.95
        
    elif regime_name == 'Bear':
        # Bear: More volatile, need more samples
        if n_regime_points > 100:
            config.draws = int(min(12000, base_config.draws * 1.5))
            config.tune = int(min(12000, base_config.tune * 1.5))
        config.target_accept = 0.90
        
    else:  # Neutral
        # Keep base settings but adjust for data size
        if n_regime_points < 150:
            config.draws = min(6000, base_config.draws)
            config.tune = min(6000, base_config.tune)
    
    # Ensure minimum sampling
    config.draws = max(2000, config.draws)
    config.tune = max(2000, config.tune)
    
    return config

def validate_regime_features(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    regime_mask: np.ndarray
) -> bool:
    """
    Validate that features are properly extracted for regime analysis
    """
    required_keys = ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend', 'data']
    
    for name, features in features_dict.items():
        for key in required_keys:
            if key not in features:
                print(f"Warning: Missing {key} in features for {name}")
                return False
            
            if len(features[key]) != len(regime_mask):
                print(f"Warning: Feature length mismatch for {key} in {name}")
                return False
    
    return True

def merge_regime_results_with_base(
    regime_results: Dict[str, Any],
    base_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge regime-aware results with base Lambda³ analysis results
    
    This allows regime analysis to be an extension rather than replacement
    """
    if base_results is None:
        return regime_results
    
    # Create merged results preserving Lambda³ structure
    merged = base_results.copy()
    
    # Add regime-specific enhancements
    merged['regime_analysis'] = regime_results.get('regime_analysis', {})
    merged['global_regimes'] = regime_results.get('global_regimes', None)
    merged['regime_statistics'] = regime_results.get('regime_detector', {}).regime_features if 'regime_detector' in regime_results else {}
    
    # Enhanced pairwise results with regime information
    if 'pairwise_results' in merged and 'regime_specific_results' in regime_results.get('regime_analysis', {}):
        merged['pairwise_results']['regime_breakdown'] = regime_results['regime_analysis']['regime_specific_results']
    
    # Add regime-aware Bayesian logger entries
    if 'bayes_logger' in regime_results:
        regime_logger = regime_results['bayes_logger']
        if 'bayes_logger' in merged:
            # Merge logger results
            merged['bayes_logger'].all_results.update(regime_logger.all_results)
    
    return merged

# ===============================
# HIERARCHICAL REGIME DETECTOR
# ===============================
class HierarchicalRegimeDetector:
    """
    Lambda³ Hierarchical Regime Detection for Financial Markets
    Detects global market regimes and pair-specific sub-regimes
    """
    
    def __init__(self, config: HierarchicalRegimeConfig):
        self.config = config
        self.global_regimes = None
        self.global_regime_model = None
        self.sub_regime_models = {}
        self.regime_features = {}
        self.transition_matrix = None
        
    def detect_global_market_regimes(
        self,
        features_dict: Dict[str, Dict[str, np.ndarray]],
        market_indicators: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Detect global market regimes using all series
        
        Lambda³ principle: Market-wide structural states (Λ_market)
        represent coherent phases of the entire financial system
        """
        print("\n" + "="*60)
        print("GLOBAL MARKET REGIME DETECTION")
        print("="*60)
        
        # Extract market-wide features
        market_features = self._extract_market_wide_features(features_dict, market_indicators)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(market_features)
        
        # Detect regimes
        if self.config.use_gmm:
            print("Using Gaussian Mixture Model for regime detection...")
            model = GaussianMixture(
                n_components=self.config.n_global_regimes,
                covariance_type='full',
                n_init=10,
                max_iter=200,
                random_state=42
            )
            regimes = model.fit_predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            
            # Apply stability threshold
            max_probs = np.max(probabilities, axis=1)
            unstable_mask = max_probs < self.config.stability_threshold
            
            # Store model for later use
            self.global_regime_model = model
            
        else:
            print("Using K-means for regime detection...")
            model = KMeans(
                n_clusters=self.config.n_global_regimes,
                n_init=20,
                random_state=42
            )
            regimes = model.fit_predict(features_scaled)
            unstable_mask = np.zeros(len(regimes), dtype=bool)
            self.global_regime_model = model
        
        # Post-process: smooth transitions
        regimes = self._smooth_regime_transitions(regimes, self.config.regime_overlap_window)
        
        # Calculate regime statistics
        self._calculate_global_regime_statistics(regimes, features_dict)
        
        # Relabel regimes based on characteristics
        regimes = self._relabel_regimes_by_characteristics(regimes, features_dict)
        
        self.global_regimes = regimes
        
        # Print summary
        unique, counts = np.unique(regimes, return_counts=True)
        print("\nGlobal Market Regime Distribution:")
        for r, count in zip(unique, counts):
            if r < len(self.config.global_regime_names):
                regime_name = self.config.global_regime_names[r]
                print(f"  {regime_name}: {count} periods ({count/len(regimes)*100:.1f}%)")
        
        # Calculate transition matrix
        self.transition_matrix = self._calculate_transition_matrix(regimes)
        
        return regimes
    
    def _extract_market_wide_features(
        self,
        features_dict: Dict[str, Dict[str, np.ndarray]],
        market_indicators: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Extract features that capture market-wide structural state"""
        
        feature_list = []
        
        # 1. Aggregate structural changes across all series
        all_pos_jumps = []
        all_neg_jumps = []
        all_tensions = []
        
        for name, features in features_dict.items():
            all_pos_jumps.append(features['delta_LambdaC_pos'])
            all_neg_jumps.append(features['delta_LambdaC_neg'])
            all_tensions.append(features['rho_T'])
        
        # Market-wide jump intensity
        market_pos_jumps = np.mean(all_pos_jumps, axis=0)
        market_neg_jumps = np.mean(all_neg_jumps, axis=0)
        
        # Market-wide tension (average and dispersion)
        market_tension_mean = np.mean(all_tensions, axis=0)
        market_tension_std = np.std(all_tensions, axis=0)
        
        # Jump asymmetry
        jump_asymmetry = market_pos_jumps - market_neg_jumps
        
        # Synchronization measure (how many series jump together)
        jump_sync = np.sum(all_pos_jumps, axis=0) + np.sum(all_neg_jumps, axis=0)
        
        feature_list.extend([
            market_pos_jumps,
            market_neg_jumps,
            market_tension_mean,
            market_tension_std,
            jump_asymmetry,
            jump_sync
        ])
        
        # 2. Rolling statistics (different time scales)
        windows = [5, 20, 50]
        for w in windows:
            # Rolling volatility ratio
            rolling_vol = self._rolling_std(market_tension_mean, w)
            feature_list.append(rolling_vol)
            
            # Rolling jump frequency
            rolling_jumps = self._rolling_mean(market_pos_jumps + market_neg_jumps, w)
            feature_list.append(rolling_jumps)
        
        # 3. Market indicators if provided
        if market_indicators is not None:
            if 'vix' in market_indicators:
                feature_list.append(market_indicators['vix'])
            if 'dollar_index' in market_indicators:
                feature_list.append(market_indicators['dollar_index'])
            if 'term_spread' in market_indicators:
                feature_list.append(market_indicators['term_spread'])
        
        # 4. Cross-series correlations (market coherence)
        if len(features_dict) > 1:
            # Calculate pairwise tension correlations
            corr_window = 20
            correlations = []
            
            for i in range(len(all_tensions[0]) - corr_window + 1):
                window_tensors = [t[i:i+corr_window] for t in all_tensions]
                try:
                    # Average pairwise correlation in this window
                    corr_matrix = np.corrcoef(window_tensors)
                    avg_corr = (corr_matrix.sum() - len(corr_matrix)) / (len(corr_matrix) * (len(corr_matrix) - 1))
                    correlations.append(avg_corr)
                except:
                    correlations.append(0.0)
            
            # Pad to match length
            if correlations:
                correlations = np.pad(correlations, (0, len(all_tensions[0]) - len(correlations)), 'edge')
            else:
                correlations = np.zeros(len(all_tensions[0]))
            feature_list.append(correlations)
        
        # Stack all features
        features = np.column_stack(feature_list)
        
        print(f"Extracted {features.shape[1]} market-wide features")
        
        return features
    
    @staticmethod
    @njit
    def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """JIT-compiled rolling standard deviation"""
        n = data.shape[0]
        result = np.empty(n)
        
        for i in range(n):
            start = max(0, i - window + 1)
            end = i + 1
            subset = data[start:end]
            subset_len = end - start
            
            if subset_len > 1:
                mean = np.mean(subset)
                variance = np.sum((subset - mean) ** 2) / subset_len
                result[i] = np.sqrt(variance)
            else:
                result[i] = 0.0
                
        return result
    
    @staticmethod
    @njit
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """JIT-compiled rolling mean"""
        n = data.shape[0]
        result = np.empty(n)
        
        for i in range(n):
            start = max(0, i - window + 1)
            end = i + 1
            result[i] = np.mean(data[start:end])
            
        return result
    
    def _smooth_regime_transitions(self, regimes: np.ndarray, window: int) -> np.ndarray:
        """Smooth regime transitions to reduce noise"""
        try:
            from scipy.ndimage import median_filter
            return median_filter(regimes, size=window)
        except:
            # Fallback if scipy not available
            return regimes
    
    def _calculate_global_regime_statistics(
        self,
        regimes: np.ndarray,
        features_dict: Dict[str, Dict[str, np.ndarray]]
    ):
        """Calculate statistics for each global regime"""
        
        self.regime_features = {}
        
        for r in range(self.config.n_global_regimes):
            mask = (regimes == r)
            n_points = np.sum(mask)
            
            if n_points > 0:
                stats = {
                    'n_points': int(n_points),
                    'frequency': n_points / len(regimes),
                    'avg_returns': {},
                    'avg_volatility': {},
                    'jump_intensity': {},
                    'regime_name': self.config.global_regime_names[r] if r < len(self.config.global_regime_names) else f'Regime_{r}'
                }
                
                # Calculate per-series statistics
                for name, features in features_dict.items():
                    # Average returns using safe calculation
                    data = features.get('data', np.zeros(len(mask)))
                    returns = calculate_returns(data)
                    stats['avg_returns'][name] = np.mean(returns[mask])
                    
                    # Average volatility (tension)
                    stats['avg_volatility'][name] = np.mean(features['rho_T'][mask])
                    
                    # Jump intensity
                    pos_jumps = np.sum(features['delta_LambdaC_pos'][mask])
                    neg_jumps = np.sum(features['delta_LambdaC_neg'][mask])
                    stats['jump_intensity'][name] = (pos_jumps + neg_jumps) / n_points
                
                self.regime_features[r] = stats
            else:
                # Empty regime
                self.regime_features[r] = {
                    'n_points': 0,
                    'frequency': 0.0,
                    'avg_returns': {},
                    'avg_volatility': {},
                    'jump_intensity': {},
                    'regime_name': self.config.global_regime_names[r] if r < len(self.config.global_regime_names) else f'Regime_{r}'
                }
    
    def _relabel_regimes_by_characteristics(
        self,
        regimes: np.ndarray,
        features_dict: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Relabel regimes based on market characteristics"""
        
        # Calculate aggregate market return for each regime
        regime_returns = []
        
        for r in range(self.config.n_global_regimes):
            mask = (regimes == r)
            if np.sum(mask) > 0:
                # Use average return across equity indices
                returns = []
                for name in ['Dow Jones', 'Nikkei 225', 'S&P 500']:
                    if name in features_dict:
                        data = features_dict[name]['data']
                        # Use safe return calculation
                        ret = calculate_returns(data)
                        returns.append(np.mean(ret[mask]))
                
                # If no equity indices found, use all series
                if not returns:
                    for name, features in features_dict.items():
                        data = features['data']
                        ret = calculate_returns(data)
                        returns.append(np.mean(ret[mask]))
                
                avg_return = np.mean(returns) if returns else 0
                regime_returns.append((r, avg_return))
            else:
                regime_returns.append((r, 0))
        
        # Sort by return: highest = Bull (0), middle = Neutral (1), lowest = Bear (2)
        regime_returns.sort(key=lambda x: x[1], reverse=True)
        
        # Create relabeling map
        relabel_map = {old_r: new_r for new_r, (old_r, _) in enumerate(regime_returns)}
        
        # Apply relabeling
        new_regimes = np.array([relabel_map.get(r, r) for r in regimes])
        
        return new_regimes
    
    def _calculate_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime transition probability matrix"""
        n_regimes = self.config.n_global_regimes
        trans_mat = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            if regimes[i] < n_regimes and regimes[i+1] < n_regimes:
                trans_mat[regimes[i], regimes[i+1]] += 1
        
        # Normalize rows
        row_sums = trans_mat.sum(axis=1)
        trans_mat = trans_mat / (row_sums[:, np.newaxis] + 1e-8)
        
        return trans_mat
    
    def detect_pair_specific_subregimes(
        self,
        pair_features: Dict[str, np.ndarray],
        global_regime_mask: np.ndarray,
        pair_name: Tuple[str, str]
    ) -> Dict[int, np.ndarray]:
        """
        Detect sub-regimes within each global regime for a specific pair
        
        Lambda³: Pair-specific structural variations within global market states
        """
        if not self.config.enable_sub_regimes:
            return {}
        
        sub_regimes = {}
        
        for g_regime in range(self.config.n_global_regimes):
            mask = (global_regime_mask == g_regime)
            n_points = np.sum(mask)
            
            if n_points < self.config.min_regime_size * 2:
                # Not enough data for sub-regime detection
                sub_regimes[g_regime] = np.zeros(n_points, dtype=int)
                continue
            
            # Extract features for this global regime
            sub_features = []
            for key in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T']:
                if key in pair_features:
                    sub_features.append(pair_features[key][mask])
            
            if not sub_features:
                continue
                
            X = np.column_stack(sub_features)
            
            # Normalize
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X)
            except:
                sub_regimes[g_regime] = np.zeros(n_points, dtype=int)
                continue
            
            # Detect sub-regimes
            n_sub = min(self.config.n_sub_regimes, n_points // 30)
            if n_sub < 2:
                sub_regimes[g_regime] = np.zeros(n_points, dtype=int)
                continue
            
            try:
                if self.config.use_gmm:
                    model = GaussianMixture(
                        n_components=n_sub,
                        covariance_type='diag',
                        n_init=5,
                        random_state=42
                    )
                else:
                    model = KMeans(
                        n_clusters=n_sub,
                        n_init=10,
                        random_state=42
                    )
                
                sub_labels = model.fit_predict(X_scaled)
                sub_regimes[g_regime] = sub_labels
            except:
                sub_regimes[g_regime] = np.zeros(n_points, dtype=int)
            
        return sub_regimes
    
    def get_regime_specific_config(self, regime: int, base_config: L3Config) -> L3Config:
        """
        Get regime-specific Bayesian configuration
        
        Different regimes may need different priors and sampling parameters
        """
        if not self.config.regime_specific_priors:
            return base_config
        
        # Use helper function
        regime_name = self.config.global_regime_names[regime] if regime < len(self.config.global_regime_names) else 'Unknown'
        n_points = self.regime_features[regime]['n_points'] if hasattr(self, 'regime_features') and regime in self.regime_features else 100
        
        return create_regime_aware_config(base_config, regime_name, n_points)

# ===============================
# REGIME-AWARE BAYESIAN ANALYSIS
# ===============================
class RegimeAwareBayesianAnalysis:
    """
    Regime-specific Bayesian analysis for Lambda³ framework
    """
    
    def __init__(
        self,
        hierarchical_config: HierarchicalRegimeConfig,
        bayes_logger: Optional[Lambda3BayesianLogger] = None
    ):
        self.h_config = hierarchical_config
        self.regime_detector = HierarchicalRegimeDetector(hierarchical_config)
        self.bayes_logger = bayes_logger
        self.regime_results = {}
        self.transition_dynamics = {}
        
    def run_regime_aware_analysis(
        self,
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict[str, np.ndarray]],
        base_config: L3Config,
        max_pairs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete regime-aware Lambda³ analysis
        
        Process:
        1. Detect global market regimes
        2. Regime-specific pairwise Bayesian analysis
        3. Regime transition dynamics
        4. Cross-regime comparison
        """
        
        print("\n" + "="*80)
        print("LAMBDA³ REGIME-AWARE BAYESIAN ANALYSIS")
        print("="*80)
        
        # Stage 1: Global regime detection
        global_regimes = self.regime_detector.detect_global_market_regimes(
            features_dict
        )
        
        # Stage 2: Regime-specific analysis
        self._run_regime_specific_pairwise_analysis(
            series_dict,
            features_dict,
            global_regimes,
            base_config,
            max_pairs
        )
        
        # Stage 3: Regime transition dynamics
        self._analyze_regime_transition_dynamics(
            features_dict,
            global_regimes
        )
        
        # Stage 4: Cross-regime comparison
        comparison_results = self._compare_across_regimes()
        
        # Compile results
        results = {
            'global_regimes': global_regimes,
            'regime_detector': self.regime_detector,
            'regime_specific_results': self.regime_results,
            'transition_dynamics': self.transition_dynamics,
            'cross_regime_comparison': comparison_results,
            'regime_statistics': self.regime_detector.regime_features,
            'transition_matrix': self.regime_detector.transition_matrix
        }
        
        # Generate summary report
        self._generate_regime_summary_report(results)
        
        return results
    
    def _run_regime_specific_pairwise_analysis(
        self,
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict[str, np.ndarray]],
        global_regimes: np.ndarray,
        base_config: L3Config,
        max_pairs: Optional[int] = None
    ):
        """Run pairwise analysis for each regime separately"""
        
        from itertools import combinations
        series_names = list(series_dict.keys())
        all_pairs = list(combinations(series_names, 2))
        
        if max_pairs and len(all_pairs) > max_pairs:
            all_pairs = all_pairs[:max_pairs]
            print(f"\nLimiting to {max_pairs} pairs for computational efficiency")
        
        print(f"\nAnalyzing {len(all_pairs)} pairs across {self.h_config.n_global_regimes} regimes")
        
        # Initialize storage
        self.regime_results = {r: {} for r in range(self.h_config.n_global_regimes)}
        
        # Analyze each regime
        for regime_idx in range(self.h_config.n_global_regimes):
            regime_mask = (global_regimes == regime_idx)
            n_regime_points = np.sum(regime_mask)
            regime_name = self.h_config.global_regime_names[regime_idx] if regime_idx < len(self.h_config.global_regime_names) else f'Regime_{regime_idx}'
            
            if n_regime_points < self.h_config.min_regime_size:
                print(f"\nSkipping {regime_name} regime: insufficient data ({n_regime_points} points)")
                continue
            
            print(f"\n{'='*60}")
            print(f"ANALYZING {regime_name.upper()} REGIME ({n_regime_points} points)")
            print(f"{'='*60}")
            
            # Get regime-specific configuration
            regime_config = self.regime_detector.get_regime_specific_config(
                regime_idx, base_config
            )
            
            # Analyze each pair in this regime
            for pair_idx, (name_a, name_b) in enumerate(all_pairs):
                if pair_idx % 10 == 0:
                    print(f"  Progress: {pair_idx}/{len(all_pairs)} pairs...")
                
                try:
                    # Extract regime-specific data
                    regime_data = {
                        name_a: series_dict[name_a][regime_mask],
                        name_b: series_dict[name_b][regime_mask]
                    }
                    
                    regime_features = {
                        name_a: self._extract_regime_features(features_dict[name_a], regime_mask),
                        name_b: self._extract_regime_features(features_dict[name_b], regime_mask)
                    }
                    
                    # Check if we have enough data
                    if len(regime_data[name_a]) < 10 or len(regime_data[name_b]) < 10:
                        continue
                    
                    # Check for sub-regimes
                    # Create combined features for sub-regime detection
                    # Note: Features are already regime-specific, so we don't need additional masking
                    pair_features_combined = {
                        'delta_LambdaC_pos': regime_features[name_a]['delta_LambdaC_pos'],
                        'delta_LambdaC_neg': regime_features[name_a]['delta_LambdaC_neg'],
                        'rho_T': regime_features[name_a]['rho_T']
                    }
                    
                    # Create sub-regime mask for already filtered data
                    sub_regime_mask = np.full(len(regime_features[name_a]['data']), regime_idx)
                    
                    # Detect sub-regimes if enabled
                    sub_regimes = self.regime_detector.detect_pair_specific_subregimes(
                        pair_features_combined,
                        sub_regime_mask,
                        (name_a, name_b)
                    )
                    
                    # Fit Bayesian model for this regime
                    trace, model = self._fit_regime_specific_bayesian(
                        regime_data,
                        regime_features,
                        regime_config,
                        regime_name,
                        (name_a, name_b)
                    )
                    
                    # Extract results
                    interaction_coeffs = self._extract_regime_interaction_coefficients(
                        trace, [name_a, name_b]
                    )
                    
                    # Store results
                    pair_key = f"{name_a}_vs_{name_b}"
                    self.regime_results[regime_idx][pair_key] = {
                        'trace': trace,
                        'model': model,
                        'interaction_coefficients': interaction_coeffs,
                        'sub_regimes': sub_regimes,
                        'series_names': [name_a, name_b],
                        'n_points': n_regime_points
                    }
                    
                except Exception as e:
                    print(f"    Error in {name_a} vs {name_b}: {str(e)}")
                    continue
            
            # Summary for this regime
            self._summarize_regime_results(regime_idx, regime_name)
    
    def _extract_regime_features(
        self,
        features: Dict[str, np.ndarray],
        mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Extract features for regime-specific analysis"""
        regime_features = {}
        
        for key in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'data']:
            if key in features:
                regime_features[key] = features[key][mask]
        
        # Reset time trend for regime
        regime_features['time_trend'] = np.arange(np.sum(mask))
        
        return regime_features
    
    def _fit_regime_specific_bayesian(
        self,
        regime_data: Dict[str, np.ndarray],
        regime_features: Dict[str, Dict[str, np.ndarray]],
        config: L3Config,
        regime_name: str,
        pair: Tuple[str, str]
    ) -> Tuple[Any, Any]:
        """
        Fit Bayesian model with regime-specific priors
        Lambda³: Complete structural tensor interaction preserved
        """
        
        name_a, name_b = pair
        data_a = regime_data[name_a]
        data_b = regime_data[name_b]
        
        feats_a = regime_features[name_a]
        feats_b = regime_features[name_b]
        
        with pm.Model() as model:
            # ==================================
            # REGIME-SPECIFIC PRIOR ADJUSTMENTS
            # ==================================
            if regime_name == 'Bull':
                # Bull market: positive bias, lower volatility, stronger coupling
                prior_mu_0 = 0.5
                prior_sigma_0 = 1.5
                prior_sigma_interact = 1.5
                prior_sigma_obs = 0.5
                prior_rho_center = 0.3  # Higher baseline correlation
            elif regime_name == 'Bear':
                # Bear market: negative bias, higher volatility, volatile coupling
                prior_mu_0 = -0.5
                prior_sigma_0 = 2.5
                prior_sigma_interact = 2.5
                prior_sigma_obs = 1.5
                prior_rho_center = 0.1  # Lower baseline correlation
            else:  # Neutral
                # Neutral: balanced priors
                prior_mu_0 = 0.0
                prior_sigma_0 = 2.0
                prior_sigma_interact = 2.0
                prior_sigma_obs = 1.0
                prior_rho_center = 0.0
            
            # ==================================
            # COMPLETE STRUCTURAL TENSOR MODEL
            # ==================================
            
            # Series A independent terms
            beta_0_a = pm.Normal('beta_0_a', mu=prior_mu_0, sigma=prior_sigma_0)
            beta_time_a = pm.Normal('beta_time_a', mu=0, sigma=1)
            beta_dLC_pos_a = pm.Normal('beta_dLC_pos_a', mu=0, sigma=prior_sigma_interact * 1.5)
            beta_dLC_neg_a = pm.Normal('beta_dLC_neg_a', mu=0, sigma=prior_sigma_interact * 1.5)
            beta_rhoT_a = pm.Normal('beta_rhoT_a', mu=0, sigma=prior_sigma_interact)
            
            # Series B independent terms
            beta_0_b = pm.Normal('beta_0_b', mu=prior_mu_0, sigma=prior_sigma_0)
            beta_time_b = pm.Normal('beta_time_b', mu=0, sigma=1)
            beta_dLC_pos_b = pm.Normal('beta_dLC_pos_b', mu=0, sigma=prior_sigma_interact * 1.5)
            beta_dLC_neg_b = pm.Normal('beta_dLC_neg_b', mu=0, sigma=prior_sigma_interact * 1.5)
            beta_rhoT_b = pm.Normal('beta_rhoT_b', mu=0, sigma=prior_sigma_interact)
            
            # Interaction terms - regime-adjusted priors
            # A → B influence
            beta_interact_ab_pos = pm.Normal('beta_interact_ab_pos', mu=0, sigma=prior_sigma_interact)
            beta_interact_ab_neg = pm.Normal('beta_interact_ab_neg', mu=0, sigma=prior_sigma_interact)
            beta_interact_ab_stress = pm.Normal('beta_interact_ab_stress', mu=0, sigma=prior_sigma_interact * 0.75)
            
            # B → A influence
            beta_interact_ba_pos = pm.Normal('beta_interact_ba_pos', mu=0, sigma=prior_sigma_interact)
            beta_interact_ba_neg = pm.Normal('beta_interact_ba_neg', mu=0, sigma=prior_sigma_interact)
            beta_interact_ba_stress = pm.Normal('beta_interact_ba_stress', mu=0, sigma=prior_sigma_interact * 0.75)
            
            # Time lag terms
            if len(data_a) > 1:
                lag_data_a = np.concatenate([[0], data_a[:-1]])
                lag_data_b = np.concatenate([[0], data_b[:-1]])
                
                # Regime-specific lag priors
                lag_sigma = prior_sigma_interact * 0.5 if regime_name == 'Bull' else prior_sigma_interact
                beta_lag_ab = pm.Normal('beta_lag_ab', mu=0, sigma=lag_sigma)
                beta_lag_ba = pm.Normal('beta_lag_ba', mu=0, sigma=lag_sigma)
            else:
                lag_data_a = np.zeros_like(data_a)
                lag_data_b = np.zeros_like(data_b)
                beta_lag_ab = 0
                beta_lag_ba = 0
            
            # ==================================
            # COMPLETE MEAN MODEL
            # ==================================
            
            # Mean model for series A
            mu_a = (
                beta_0_a
                + beta_time_a * feats_a['time_trend']
                + beta_dLC_pos_a * feats_a['delta_LambdaC_pos']
                + beta_dLC_neg_a * feats_a['delta_LambdaC_neg']
                + beta_rhoT_a * feats_a['rho_T']
                # B → A interaction (3 components)
                + beta_interact_ba_pos * feats_b['delta_LambdaC_pos']
                + beta_interact_ba_neg * feats_b['delta_LambdaC_neg']
                + beta_interact_ba_stress * feats_b['rho_T']
                # Lag effect
                + beta_lag_ba * lag_data_b
            )
            
            # Mean model for series B
            mu_b = (
                beta_0_b
                + beta_time_b * feats_b['time_trend']
                + beta_dLC_pos_b * feats_b['delta_LambdaC_pos']
                + beta_dLC_neg_b * feats_b['delta_LambdaC_neg']
                + beta_rhoT_b * feats_b['rho_T']
                # A → B interaction (3 components)
                + beta_interact_ab_pos * feats_a['delta_LambdaC_pos']
                + beta_interact_ab_neg * feats_a['delta_LambdaC_neg']
                + beta_interact_ab_stress * feats_a['rho_T']
                # Lag effect
                + beta_lag_ab * lag_data_a
            )
            
            # ==================================
            # OBSERVATION MODEL WITH CORRELATION
            # ==================================
            
            # Regime-specific observation noise
            sigma_a = pm.HalfNormal('sigma_a', sigma=prior_sigma_obs)
            sigma_b = pm.HalfNormal('sigma_b', sigma=prior_sigma_obs)
            
            # Correlation structure - regime-adjusted
            if regime_name == 'Bull':
                # Bull: expect higher correlation
                rho_ab = pm.Beta('rho_ab', alpha=6, beta=4) * 2 - 1  # Bias toward positive
            elif regime_name == 'Bear':
                # Bear: expect more volatile correlation
                rho_ab = pm.Uniform('rho_ab', lower=-1, upper=1)  # Flat prior
            else:  # Neutral
                # Neutral: weakly informative centered at 0
                rho_ab_raw = pm.Normal('rho_ab_raw', mu=prior_rho_center, sigma=0.3)
                rho_ab = pm.Deterministic('rho_ab', pm.math.tanh(rho_ab_raw))
            
            # Covariance matrix
            cov_matrix = pm.math.stack([
                [sigma_a**2, rho_ab * sigma_a * sigma_b],
                [rho_ab * sigma_a * sigma_b, sigma_b**2]
            ])
            
            # Joint observation
            y_combined = pm.math.stack([data_a, data_b]).T
            mu_combined = pm.math.stack([mu_a, mu_b]).T
            
            y_obs = pm.MvNormal('y_obs', mu=mu_combined, cov=cov_matrix, observed=y_combined)
            
            # ==================================
            # REGIME-ADAPTIVE SAMPLING
            # ==================================
            
            # Adjust sampling based on regime and data size
            n_points = len(data_a)
            
            if n_points < 100:
                # Small regime: reduce sampling to avoid overfitting
                actual_draws = min(config.draws, 3000)
                actual_tune = min(config.tune, 3000)
            else:
                actual_draws = config.draws
                actual_tune = config.tune
            
            # Sample from posterior
            trace = pm.sample(
                draws=actual_draws,
                tune=actual_tune,
                target_accept=config.target_accept,
                return_inferencedata=True,
                cores=4,
                chains=4
            )
            
            # Log regime-specific information
            print(f"    Regime: {regime_name}, Points: {n_points}, "
                  f"Draws: {actual_draws}, Tune: {actual_tune}")
        
        # HDI logging if logger provided
        if self.bayes_logger is not None:
            model_id = f"regime_{regime_name}_{name_a}_vs_{name_b}"
            self.bayes_logger.log_trace(
                trace,
                model_id=model_id,
                model_type=f"regime_specific_{regime_name.lower()}",
                series_names=[name_a, name_b],
                verbose=False
            )
        
        return trace, model
    
    def _extract_regime_interaction_coefficients(
        self,
        trace: Any,
        series_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract complete interaction coefficients from regime-specific trace
        Lambda³: Full structural tensor interaction mapping
        """
        summary = az.summary(trace)
        name_a, name_b = series_names[:2]
        
        # Safe parameter extraction helper
        def safe_extract(param_name: str, default: float = 0.0) -> float:
            return summary.loc[param_name, 'mean'] if param_name in summary.index else default
        
        # Extract all structural tensor interaction components
        results = {
            'self_effects': {},
            'cross_effects': {},
            'lag_effects': {},
            'correlation': None,
            'regime_specific': {}
        }
        
        # Self effects for series A
        results['self_effects'][name_a] = {
            'intercept': safe_extract('beta_0_a'),
            'time': safe_extract('beta_time_a'),
            'pos_jump': safe_extract('beta_dLC_pos_a'),
            'neg_jump': safe_extract('beta_dLC_neg_a'),
            'tension': safe_extract('beta_rhoT_a')
        }
        
        # Self effects for series B
        results['self_effects'][name_b] = {
            'intercept': safe_extract('beta_0_b'),
            'time': safe_extract('beta_time_b'),
            'pos_jump': safe_extract('beta_dLC_pos_b'),
            'neg_jump': safe_extract('beta_dLC_neg_b'),
            'tension': safe_extract('beta_rhoT_b')
        }
        
        # Cross effects A → B
        results['cross_effects'][f'{name_a}_to_{name_b}'] = {
            'pos_jump': safe_extract('beta_interact_ab_pos'),
            'neg_jump': safe_extract('beta_interact_ab_neg'),
            'tension': safe_extract('beta_interact_ab_stress')
        }
        
        # Cross effects B → A
        results['cross_effects'][f'{name_b}_to_{name_a}'] = {
            'pos_jump': safe_extract('beta_interact_ba_pos'),
            'neg_jump': safe_extract('beta_interact_ba_neg'),
            'tension': safe_extract('beta_interact_ba_stress')
        }
        
        # Lag effects
        results['lag_effects'] = {
            f'{name_a}_to_{name_b}': safe_extract('beta_lag_ab'),
            f'{name_b}_to_{name_a}': safe_extract('beta_lag_ba')
        }
        
        # Correlation
        results['correlation'] = safe_extract('rho_ab')
        
        # Calculate aggregate interaction strengths
        # A → B total strength
        strength_a_to_b = sum(abs(v) for v in results['cross_effects'][f'{name_a}_to_{name_b}'].values())
        # B → A total strength  
        strength_b_to_a = sum(abs(v) for v in results['cross_effects'][f'{name_b}_to_{name_a}'].values())
        
        results['regime_specific'] = {
            'interaction_strength': (strength_a_to_b + strength_b_to_a) / 2,
            'asymmetry': abs(strength_a_to_b - strength_b_to_a),
            'dominant_direction': f'{name_a}→{name_b}' if strength_a_to_b > strength_b_to_a else f'{name_b}→{name_a}',
            'strength_a_to_b': strength_a_to_b,
            'strength_b_to_a': strength_b_to_a
        }
        
        # Extract observation model parameters
        results['observation_model'] = {
            'sigma_a': safe_extract('sigma_a'),
            'sigma_b': safe_extract('sigma_b'),
            'correlation': results['correlation']
        }
        
        return results
    
    def _summarize_regime_results(self, regime_idx: int, regime_name: str):
        """Summarize results for a specific regime"""
        regime_results = self.regime_results[regime_idx]
        
        if not regime_results:
            return
        
        print(f"\n{regime_name} Regime Summary:")
        print("-" * 40)
        
        # Find strongest interactions
        interactions = []
        for pair_key, results in regime_results.items():
            if 'interaction_coefficients' in results:
                strength = results['interaction_coefficients']['regime_specific']['interaction_strength']
                interactions.append((pair_key, strength))
        
        interactions.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top 5 interactions in {regime_name}:")
        for pair, strength in interactions[:5]:
            print(f"  {pair}: {strength:.4f}")
    
    def _analyze_regime_transition_dynamics(
        self,
        features_dict: Dict[str, Dict[str, np.ndarray]],
        global_regimes: np.ndarray
    ):
        """Analyze dynamics at regime transition points"""
        
        print("\n" + "="*60)
        print("REGIME TRANSITION DYNAMICS ANALYSIS")
        print("="*60)
        
        # Find transition points
        transitions = []
        for i in range(1, len(global_regimes)):
            if global_regimes[i] != global_regimes[i-1]:
                transitions.append({
                    'index': i,
                    'from_regime': int(global_regimes[i-1]),
                    'to_regime': int(global_regimes[i]),
                    'from_name': self.h_config.global_regime_names[global_regimes[i-1]] if global_regimes[i-1] < len(self.h_config.global_regime_names) else f'Regime_{global_regimes[i-1]}',
                    'to_name': self.h_config.global_regime_names[global_regimes[i]] if global_regimes[i] < len(self.h_config.global_regime_names) else f'Regime_{global_regimes[i]}'
                })
        
        print(f"Found {len(transitions)} regime transitions")
        
        # Analyze structural changes around transitions
        window = 10  # Look at ±10 periods around transition
        
        for trans in transitions[:5]:  # Analyze first 5 transitions
            idx = trans['index']
            print(f"\nTransition at index {idx}: {trans['from_name']} → {trans['to_name']}")
            
            # Calculate average structural changes before/after
            for name, features in list(features_dict.items())[:3]:  # First 3 series
                pre_jumps = np.sum(features['delta_LambdaC_pos'][max(0, idx-window):idx])
                post_jumps = np.sum(features['delta_LambdaC_pos'][idx:min(len(features['delta_LambdaC_pos']), idx+window)])
                
                pre_tension = np.mean(features['rho_T'][max(0, idx-window):idx]) if idx > 0 else 0
                post_tension = np.mean(features['rho_T'][idx:min(len(features['rho_T']), idx+window)]) if idx < len(features['rho_T']) else 0
                
                print(f"  {name}: Jumps {pre_jumps}→{post_jumps}, Tension {pre_tension:.3f}→{post_tension:.3f}")
        
        self.transition_dynamics = {
            'transitions': transitions,
            'transition_matrix': self.regime_detector.transition_matrix
        }
    
    def _compare_across_regimes(self) -> Dict[str, Any]:
        """Compare interaction patterns across different regimes"""
        
        comparison = {
            'interaction_changes': {},
            'regime_specific_patterns': {},
            'stability_analysis': {}
        }
        
        # Compare same pair across regimes
        all_pairs = set()
        for regime_results in self.regime_results.values():
            all_pairs.update(regime_results.keys())
        
        for pair_key in all_pairs:
            pair_comparison = {}
            
            for regime_idx, regime_results in self.regime_results.items():
                if pair_key in regime_results:
                    coeffs = regime_results[pair_key]['interaction_coefficients']
                    regime_name = self.h_config.global_regime_names[regime_idx] if regime_idx < len(self.h_config.global_regime_names) else f'Regime_{regime_idx}'
                    pair_comparison[regime_name] = coeffs['regime_specific']['interaction_strength']
            
            if len(pair_comparison) > 1:
                comparison['interaction_changes'][pair_key] = pair_comparison
        
        return comparison
    
    def _generate_regime_summary_report(self, results: Dict[str, Any]):
        """Generate comprehensive regime analysis report"""
        
        print("\n" + "="*80)
        print("REGIME-AWARE LAMBDA³ ANALYSIS SUMMARY")
        print("="*80)
        
        # 1. Regime distribution
        print("\nMarket Regime Distribution:")
        for regime_idx, stats in self.regime_detector.regime_features.items():
            print(f"  {stats['regime_name']}: {stats['frequency']*100:.1f}% ({stats['n_points']} periods)")
        
        # 2. Transition matrix
        print("\nRegime Transition Probabilities:")
        trans_mat = results['transition_matrix']
        regime_names = self.h_config.global_regime_names
        
        print(f"{'From/To':<10}", end='')
        for name in regime_names:
            print(f"{name:>10}", end='')
        print()
        
        for i, from_name in enumerate(regime_names):
            print(f"{from_name:<10}", end='')
            for j in range(len(regime_names)):
                print(f"{trans_mat[i,j]:>10.3f}", end='')
            print()
        
        # 3. Key findings
        print("\nKey Regime-Specific Findings:")
        
        # Find pairs with largest regime-dependent changes
        max_change = 0
        max_change_pair = None
        max_change_regimes = None
        
        for pair_key, regime_strengths in results['cross_regime_comparison']['interaction_changes'].items():
            if len(regime_strengths) > 1:
                values = list(regime_strengths.values())
                change = max(values) - min(values)
                if change > max_change:
                    max_change = change
                    max_change_pair = pair_key
                    max_change_regimes = regime_strengths
        
        if max_change_pair:
            print(f"\nLargest regime-dependent interaction change:")
            print(f"  Pair: {max_change_pair}")
            for regime, strength in max_change_regimes.items():
                print(f"    {regime}: {strength:.4f}")
        
        # 4. Theoretical interpretation
        print("\n" + "-"*80)
        print("LAMBDA³ THEORETICAL INSIGHTS:")
        print("-"*80)
        print("• Market regimes represent distinct configurations in financial tensor space")
        print("• Structural tensor interactions (Λ) adapt to regime-specific dynamics")
        print("• Regime transitions coincide with ΔΛC pulsation clusters")
        print("• Correlation structures (ρ_ab) exhibit regime-dependent patterns")
        print("• Complete bidirectional interactions preserved across all regimes")

# ===============================
# VISUALIZATION FUNCTIONS
# ===============================
def plot_regime_timeline(
    global_regimes: np.ndarray,
    regime_names: List[str],
    series_data: Optional[Dict[str, np.ndarray]] = None,
    highlight_transitions: bool = True
):
    """Plot regime timeline with optional overlay of series data"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                            gridspec_kw={'height_ratios': [1, 2]})
    
    # Top panel: Regime timeline
    ax1 = axes[0]
    
    # Create colormap
    colors = ['green', 'gray', 'red']  # Bull, Neutral, Bear
    
    # Plot regime blocks
    for i in range(len(global_regimes)):
        regime = int(global_regimes[i])
        if regime < len(colors):
            ax1.axvspan(i, i+1, facecolor=colors[regime], alpha=0.6)
    
    # Add regime labels
    ax1.set_xlim(0, len(global_regimes))
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Market Regime')
    ax1.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=regime_names[i]) 
                      for i in range(min(len(colors), len(regime_names)))]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Highlight transitions
    if highlight_transitions:
        for i in range(1, len(global_regimes)):
            if global_regimes[i] != global_regimes[i-1]:
                ax1.axvline(x=i, color='black', linestyle='--', alpha=0.5)
    
    ax1.set_title('Market Regime Timeline')
    
    # Bottom panel: Series data (if provided)
    ax2 = axes[1]
    
    if series_data:
        for name, data in list(series_data.items())[:3]:  # Plot first 3 series
            # Normalize for visualization
            if len(data) > 0:
                data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)
                ax2.plot(data_norm, label=name, alpha=0.7)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Normalized Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    ax2.set_xlim(0, len(global_regimes))
    
    plt.tight_layout()
    plt.show()

def plot_regime_interaction_heatmap(
    regime_results: Dict[int, Dict[str, Any]],
    regime_names: List[str]
):
    """Plot interaction strength heatmaps for each regime"""
    
    n_regimes = len(regime_results)
    if n_regimes == 0:
        print("No regime results to plot")
        return
    
    fig, axes = plt.subplots(1, n_regimes, figsize=(6*n_regimes, 5))
    
    if n_regimes == 1:
        axes = [axes]
    
    # Get all unique pairs
    all_pairs = set()
    for regime_results_dict in regime_results.values():
        all_pairs.update(regime_results_dict.keys())
    all_pairs = sorted(list(all_pairs))
    
    if not all_pairs:
        print("No pair results found")
        return
    
    # Extract series names
    series_names = set()
    for pair in all_pairs:
        names = pair.split('_vs_')
        series_names.update(names)
    series_names = sorted(list(series_names))
    
    for regime_idx, ax in enumerate(axes):
        # Create interaction matrix for this regime
        n_series = len(series_names)
        interaction_matrix = np.zeros((n_series, n_series))
        
        regime_data = regime_results.get(regime_idx, {})
        
        for pair_key, results in regime_data.items():
            if 'interaction_coefficients' in results:
                names = pair_key.split('_vs_')
                if len(names) == 2 and names[0] in series_names and names[1] in series_names:
                    i = series_names.index(names[0])
                    j = series_names.index(names[1])
                    strength = results['interaction_coefficients']['regime_specific']['interaction_strength']
                    interaction_matrix[i, j] = strength
                    interaction_matrix[j, i] = strength  # Symmetric for visualization
        
        # Plot heatmap
        max_val = np.max(np.abs(interaction_matrix)) if np.any(interaction_matrix) else 1
        im = ax.imshow(interaction_matrix, cmap='RdBu_r', aspect='auto',
                       vmin=-max_val, vmax=max_val)
        
        ax.set_xticks(range(n_series))
        ax.set_yticks(range(n_series))
        ax.set_xticklabels(series_names, rotation=45, ha='right')
        ax.set_yticklabels(series_names)
        
        regime_name = regime_names[regime_idx] if regime_idx < len(regime_names) else f'Regime {regime_idx}'
        ax.set_title(f'{regime_name} Regime')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Interaction Strength by Market Regime', fontsize=16)
    plt.tight_layout()
    plt.show()

# ===============================
# INTEGRATION WITH MAIN ANALYSIS
# ===============================
def run_lambda3_regime_aware_analysis(
    data_source: Union[str, Dict[str, np.ndarray]],
    base_config: L3Config = None,
    hierarchical_config: HierarchicalRegimeConfig = None,
    target_series: List[str] = None,
    max_pairs: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Lambda³ analysis with hierarchical regime detection
    
    This extends the standard run_lambda3_analysis with regime-aware
    capabilities while preserving complete structural tensor interactions
    """
    
    if base_config is None:
        base_config = L3Config()
    
    if hierarchical_config is None:
        hierarchical_config = HierarchicalRegimeConfig()
    
    # Load data
    if isinstance(data_source, str):
        series_dict = load_csv_data(data_source)
    else:
        series_dict = data_source
    
    if target_series:
        series_dict = {k: v for k, v in series_dict.items() if k in target_series}
    
    series_names = list(series_dict.keys())
    
    if verbose:
        print("="*80)
        print("Lambda³ Regime-Aware Comprehensive Analysis")
        print(f"Series: {len(series_names)}, Length: {len(next(iter(series_dict.values())))}")
        print("="*80)
    
    # Initialize Bayesian logger
    bayes_logger = Lambda3BayesianLogger(hdi_prob=base_config.hdi_prob)
    
    # Extract features - IMPORTANT: Disable hierarchical mode for regime analysis
    # to ensure consistent feature dimensions
    # Note: Hierarchical features may have different dimensions than raw data,
    # which causes issues with regime masking. The regime analysis will still
    # capture hierarchical dynamics through the regime detection itself.
    feature_config = copy.deepcopy(base_config)
    feature_config.hierarchical = False  # Force non-hierarchical mode
    
    features_dict = {}
    for name, data in series_dict.items():
        features_dict[name] = calc_lambda3_features(data, feature_config)
        
    # Verify feature dimensions match data dimensions
    for name in series_names:
        data_len = len(series_dict[name])
        feat_len = len(features_dict[name]['data'])
        if data_len != feat_len:
            print(f"WARNING: Data/Feature length mismatch for {name}: {data_len} vs {feat_len}")
            print("This may indicate hierarchical features are being used. Regime analysis requires non-hierarchical features.")
    
    # Run regime-aware analysis
    regime_analyzer = RegimeAwareBayesianAnalysis(
        hierarchical_config,
        bayes_logger
    )
    
    regime_results = regime_analyzer.run_regime_aware_analysis(
        series_dict,
        features_dict,
        base_config,
        max_pairs
    )
    
    # Visualize results
    if verbose:
        try:
            plot_regime_timeline(
                regime_results['global_regimes'],
                hierarchical_config.global_regime_names,
                series_dict
            )
            
            plot_regime_interaction_heatmap(
                regime_results['regime_specific_results'],
                hierarchical_config.global_regime_names
            )
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
    
    # Compile complete results
    results = {
        'series_dict': series_dict,
        'series_names': series_names,
        'features_dict': features_dict,
        'regime_analysis': regime_results,
        'config': base_config,
        'hierarchical_config': hierarchical_config,
        'bayes_logger': bayes_logger
    }
    
    return results

# ===============================
# EXAMPLE USAGE
# ===============================
if __name__ == "__main__":
    print("Lambda³ Regime-Aware Extension Module (Fixed)")
    print("="*60)
    
    # Example configuration
    h_config = HierarchicalRegimeConfig(
        n_global_regimes=3,
        global_regime_names=['Bull', 'Neutral', 'Bear'],
        n_sub_regimes=2,
        use_gmm=True,
        min_regime_size=50,
        regime_specific_priors=True,
        adaptive_sampling=True
    )
    
    print("Configuration:")
    print(f"  Global regimes: {h_config.n_global_regimes} ({', '.join(h_config.global_regime_names)})")
    print(f"  Sub-regimes per global: {h_config.n_sub_regimes}")
    print(f"  Detection method: {'GMM' if h_config.use_gmm else 'K-means'}")
    print(f"  Minimum regime size: {h_config.min_regime_size}")
    print(f"  Regime-specific priors: {h_config.regime_specific_priors}")
    print(f"  Adaptive sampling: {h_config.adaptive_sampling}")
    
    print("\nFIXED: Array operations corrected for safe return calculations")
    print("Complete structural tensor interactions preserved across all regimes")
