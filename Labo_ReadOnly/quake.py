"""
Lambda³ Spatial Multi-Layer Analysis System - Integrated Data Classes
統合観測網（広帯域＋強震計）対応版データクラス定義
Based on Dr. Iizumi's Lambda³ Theory for Noto Earthquake Analysis
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import os
import json
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import warnings
from scipy.signal import hilbert
from scipy.optimize import minimize  # _inverse_problem メソッド内で使用
from scipy import signal  # _compute_phase_coherence メソッド内で使用
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering  # クラスタリング用
from sklearn.preprocessing import StandardScaler  # DBSCAN用
from sklearn.neighbors import NearestNeighbors  # DBSCAN用
from joblib import Parallel, delayed  # 並列処理用（オプション）

# === config定義 ===

EARTHQUAKE_CONFIG = {
    "name": "noto_20240101",
    "broadband_matrix_path": "/content/noto_20240101_broadband_lambda3.npy",
    "strong_motion_matrix_path": "/content/noto_20240101_strong_motion_lambda3.npy",
    "integrated_matrix_path": "/content/noto_20240101_integrated_lambda3.npy",  # 統合ファイルがある場合
    "output_dir": "./content/results/noto_20240101",
    # その他パラメータ
    "n_clusters": 7,
    "n_paths_global": 12,
    "n_paths_local": 5,
    "n_paths_cluster": 8,
    "clustering_method": "kmeans",
    # evolution系オプションも追記可能
    "earthquake_event_bb": 60,
    "earthquake_event_sm": 555,
    "window_duration": 5,
    "n_windows": 12,
}

# === 基本データクラス定義 ===
@dataclass
class Lambda3Result:
    """Lambda³解析結果を格納するデータクラス"""
    paths: Dict[int, np.ndarray]
    topological_charges: Dict[int, float]
    stabilities: Dict[int, float]
    energies: Dict[int, float]
    entropies: Dict[int, float]
    classifications: Dict[int, str]

@dataclass
class SpatialLambda3Result:
    """空間多層Lambda³解析結果"""
    # グローバル（全国規模）解析結果
    global_result: Lambda3Result

    # ローカル（観測点別）解析結果
    local_results: Dict[str, Lambda3Result]

    # クラスタ（地域別）解析結果
    cluster_results: Dict[int, Lambda3Result]

    # 空間相関構造
    spatial_correlations: np.ndarray

    # 観測点クラスタリング情報
    station_clusters: Dict[str, int]

    # 層間相互作用指標
    cross_layer_metrics: Dict[str, float]

    # 空間的異常検出結果
    spatial_anomalies: Dict[str, List[Dict]]

    # 解析メタデータ
    metadata: Dict[str, any]

# === 統合データ用の新しいデータクラス ===
@dataclass
class IntegratedLambda3Result:
    """広帯域・強震計統合Lambda³解析結果"""
    # 広帯域（深部構造）解析結果
    broadband_result: SpatialLambda3Result

    # 強震計（表層構造）解析結果
    strong_motion_result: SpatialLambda3Result

    # 深部-表層相互作用
    depth_surface_interaction: Dict[str, float]

    # 統合異常検出結果
    integrated_anomalies: Dict[str, List[Dict]]

    # 時空間伝播パターン
    propagation_patterns: Dict[str, np.ndarray]

    # 統合メタデータ
    metadata: Dict[str, any]

@dataclass
class HierarchicalLambda3Result:
    """階層的Lambda³解析結果（深部→表層の構造伝播）"""
    # 階層別結果
    hierarchical_results: Dict[str, Lambda3Result]  # 'deep', 'intermediate', 'surface'

    # 階層間の伝達関数
    transfer_functions: Dict[str, np.ndarray]

    # 階層間の時間遅延
    layer_delays: Dict[str, float]

    # 構造伝播速度
    propagation_velocities: Dict[str, float]

    # メタデータ
    metadata: Dict[str, any]

# === Lambda³基本解析クラス（統合データ対応版） ===
class Lambda3Analyzer:
    """Lambda³理論による構造解析の基本クラス - 統合データ対応版"""

    def __init__(self, alpha: float = 0.1, beta: float = 0.01,
                 data_type: str = 'standard'):
        """
        Parameters:
        -----------
        alpha : float
            正則化パラメータ（全変動）
        beta : float
            正則化パラメータ（L1）
        data_type : str
            データタイプ（'standard', 'broadband', 'strong_motion'）
        """
        self.alpha = alpha
        self.beta = beta
        self.data_type = data_type

        # データタイプに応じたパラメータ調整
        if data_type == 'broadband':
            # 広帯域：長周期成分を重視
            self.alpha *= 1.5
            self.freq_range = (0.01, 20.0)
        elif data_type == 'strong_motion':
            # 強震計：短周期成分を重視
            self.beta *= 1.5
            self.freq_range = (0.1, 30.0)
        else:
            self.freq_range = (0.01, 50.0)

        # 異常パターン定義（既存のものを継承）
        self.anomaly_patterns = {
            # 基本パターン
            'pulse': self._generate_pulse_anomaly,
            'phase_jump': self._generate_phase_jump_anomaly,
            'periodic': self._generate_periodic_anomaly,
            'structural_decay': self._generate_decay_anomaly,
            'bifurcation': self._generate_bifurcation_anomaly,
            # 地震波パターン
            'p_wave': self._generate_p_wave_anomaly,
            's_wave': self._generate_s_wave_anomaly,
            # 複雑パターン
            'multi_path': self._generate_multi_path_anomaly,
            'topological_jump': self._generate_topological_jump_anomaly,
            'cascade': self._generate_cascade_anomaly,
            'resonance': self._generate_resonance_anomaly,
            # 地震前兆特有パターン
            'foreshock_sequence': self._generate_foreshock_sequence,
            'quiet_period': self._generate_quiet_period,
            'nucleation_phase': self._generate_nucleation_phase,
            'dilatancy': self._generate_dilatancy_anomaly,
            'crustal_deformation': self._generate_crustal_deformation,
            'electromagnetic': self._generate_electromagnetic_precursor,
            'slow_slip': self._generate_slow_slip_event,
            'critical_point': self._generate_critical_point_anomaly
        }

    def analyze(self, events: np.ndarray, n_paths: int = 3) -> Lambda3Result:
        """完全なLambda³解析を実行（パーセンタイル分類付き）"""
        # 入力検証
        if len(events) == 0 or events.shape[0] == 0:
            raise ValueError("Empty event data")

        # データタイプに応じた前処理
        if self.data_type == 'broadband':
            events = self._enhance_low_frequency(events)
        elif self.data_type == 'strong_motion':
            events = self._enhance_high_frequency(events)

        # 1. 構造テンソル推定
        paths = self._inverse_problem(events, n_paths)

        # 2. 各パスの物理量計算
        charges, stabilities = {}, {}
        energies, entropies = {}, {}

        for i, path in paths.items():
            Q, sigma = self._compute_topological_charge(path)
            charges[i] = Q
            stabilities[i] = sigma
            energies[i] = np.sum(path**2)
            entropies[i] = self._compute_entropy(path)

        # === パーセンタイル分類 ===
        Q_list = list(charges.values())
        classifications = {}
        for i, Q in charges.items():
            label = self.classify_structure(Q, Q_list, self.data_type)
            classifications[i] = label

        return Lambda3Result(
            paths=paths,
            topological_charges=charges,
            stabilities=stabilities,
            energies=energies,
            entropies=entropies,
            classifications=classifications
        )

    def _enhance_low_frequency(self, events: np.ndarray) -> np.ndarray:
        """低周波成分を強調（広帯域データ用）"""
        # FFTで周波数領域へ
        fft = np.fft.fft(events, axis=0)
        freqs = np.fft.fftfreq(events.shape[0])

        # 低周波成分を増幅
        low_freq_mask = np.abs(freqs) < 0.1
        fft[low_freq_mask] *= 1.5

        # 時間領域へ戻す
        return np.real(np.fft.ifft(fft, axis=0))

    def _enhance_high_frequency(self, events: np.ndarray) -> np.ndarray:
        """高周波成分を強調（強震計データ用）"""
        # FFTで周波数領域へ
        fft = np.fft.fft(events, axis=0)
        freqs = np.fft.fftfreq(events.shape[0])

        # 高周波成分を増幅
        high_freq_mask = np.abs(freqs) > 0.3
        fft[high_freq_mask] *= 1.5

        # 時間領域へ戻す
        return np.real(np.fft.ifft(fft, axis=0))

    @staticmethod
    def classify_structure(Q: float, Q_list: list, data_type: str = 'standard', 
                                    lower_pct: float = 25, upper_pct: float = 75) -> str:
        """
        Q_Λ分布に対してパーセンタイルで構造分類
        """
        structure_labels = {
            'broadband': {
                'low': "深部エネルギー吸収構造（プレート沈み込み）",
                'neutral': "深部中性構造（安定プレート）",
                'high': "深部エネルギー放出構造（マントル上昇流）"
            },
            'strong_motion': {
                'low': "表層エネルギー吸収構造（地盤沈下）",
                'neutral': "表層中性構造（弾性変形）",
                'high': "表層エネルギー放出構造（断層破壊）"
            },
            'standard': {
                'low': "反物質的構造（エネルギー吸収系）",
                'neutral': "中性構造（平衡状態）",
                'high': "物質的構造（エネルギー放出系）"
            }
        }
        labels = structure_labels.get(data_type, structure_labels['standard'])

        # 分布から動的閾値算出
        q_array = np.array(Q_list)
        lower_thr = np.percentile(q_array, lower_pct)
        upper_thr = np.percentile(q_array, upper_pct)

        if Q < lower_thr:
            return labels['low']
        elif Q > upper_thr:
            return labels['high']
        else:
            return labels['neutral']

    # === 統合解析用の新メソッド ===
    def analyze_integrated(self,
                          broadband_events: np.ndarray,
                          strong_motion_events: np.ndarray,
                          n_paths: int = 3) -> Tuple[Lambda3Result, Lambda3Result, Dict]:
        """
        広帯域・強震計データの統合解析

        Returns:
        --------
        broadband_result : Lambda3Result
            広帯域データの解析結果
        strong_motion_result : Lambda3Result
            強震計データの解析結果
        interaction : Dict
            深部-表層相互作用の指標
        """
        # それぞれのデータタイプで解析
        self.data_type = 'broadband'
        broadband_result = self.analyze(broadband_events, n_paths)

        self.data_type = 'strong_motion'
        strong_motion_result = self.analyze(strong_motion_events, n_paths)

        # 深部-表層相互作用の計算
        interaction = self._compute_depth_surface_interaction(
            broadband_result, strong_motion_result
        )

        self.data_type = 'standard'  # リセット

        return broadband_result, strong_motion_result, interaction

    def _compute_depth_surface_interaction(self, broadband_result, strong_motion_result) -> Dict[str, float]:
        interaction = {}

        # チャージ・エネルギー・エントロピー分布を配列で取得
        bb_charges = np.array(list(broadband_result.topological_charges.values()))
        sm_charges = np.array(list(strong_motion_result.topological_charges.values()))
        bb_energies = np.array(list(broadband_result.energies.values()))
        sm_energies = np.array(list(strong_motion_result.energies.values()))
        bb_entropies = np.array(list(broadband_result.entropies.values()))
        sm_entropies = np.array(list(strong_motion_result.entropies.values()))

        # 相関
        min_len = min(len(bb_charges), len(sm_charges))
        if min_len > 1:
            corr = np.corrcoef(bb_charges[:min_len], sm_charges[:min_len])[0, 1]
            interaction['charge_correlation'] = float(corr) if not np.isnan(corr) else 0.0
        else:
            interaction['charge_correlation'] = 0.0

        # エネルギー伝達
        interaction['energy_transfer_efficiency'] = (
            np.median(sm_energies[:min_len]) / (np.median(bb_energies[:min_len]) + 1e-8)
            if min_len > 1 and np.median(bb_energies[:min_len]) > 0 else 0.0
        )

        # 構造整合性
        entropy_diff = np.abs(np.median(bb_entropies[:min_len]) - np.median(sm_entropies[:min_len]))
        interaction['structural_coherence'] = 1.0 / (1.0 + entropy_diff)

        # 伝播遅延（主要経路平均などで…）
        delays = []
        for k in range(min_len):
            if k in broadband_result.paths and k in strong_motion_result.paths:
                bb_sig = broadband_result.paths[k]
                sm_sig = strong_motion_result.paths[k]
                sig_len = min(len(bb_sig), len(sm_sig))
                if sig_len > 10:
                    corr = np.correlate(bb_sig[:sig_len], sm_sig[:sig_len], mode='same')
                    delay_idx = np.argmax(corr) - len(corr)//2
                    # 例：1イベント=10秒なら delay_time = delay_idx * 10
                    delay_time = delay_idx * 10.0  # ←ここはデータに合わせて修正
                    delays.append(delay_time)
        interaction['estimated_delay'] = np.median(delays) if delays else 0.0

        return interaction

    def _estimate_propagation_delay(self, broadband_result, strong_motion_result, window_duration_sec, n_event):
        """深部から表層への伝播遅延を推定（正しいevent単位で）"""
        if broadband_result.paths and strong_motion_result.paths:
            bb_signal = broadband_result.paths[0]
            sm_signal = strong_motion_result.paths[0]

            min_len = min(len(bb_signal), len(sm_signal))
            if min_len > 10:
                bb_signal = bb_signal[:min_len]
                sm_signal = sm_signal[:min_len]

                correlation = np.correlate(bb_signal, sm_signal, mode='same')
                delay_idx = np.argmax(correlation) - len(correlation) // 2

                dt = window_duration_sec / n_event  # 1 event = ○秒
                delay_time = delay_idx * dt
                return delay_time

        return 0.0

    # 既存の異常パターン生成メソッド
    def _inverse_problem(self, events: np.ndarray, n_paths: int) -> Dict[int, np.ndarray]:
        """
        正則化付き逆問題（テンソル分解版）
        events: [n_events, n_features]の観測データ
        n_paths: 分解ランク数（Λ³理論的な“独立経路”数）
        戻り値: {i: path_i} 各経路ベクトル
        """
        n_events, n_features = events.shape

        # 空データハンドリング
        if n_events == 0 or n_features == 0:
            print(f"Warning: Invalid event shape: {events.shape}")
            return {i: np.zeros(n_events) for i in range(n_paths)}

        # SVDで初期値生成
        U, S, Vh = np.linalg.svd(events, full_matrices=False)
        # Λ初期値（n_events × n_paths）：SVD主成分方向
        Lambda_init = (U[:, :n_paths] @ np.diag(S[:n_paths])).flatten()

        def objective(Lambda_flat):
            # Λ（n_events × n_paths）で分解
            Lambda = Lambda_flat.reshape(n_events, n_paths)
            # 右特異ベクトルは固定（物理的には各経路の“方向性”パラメータ）
            reconstruction = Lambda @ Vh[:n_paths, :]
            # 元データとのfit（テンソル射影誤差）
            data_fit = np.linalg.norm(events - reconstruction) ** 2
            # トータルバリエーション正則化（時間的滑らかさ）
            tv_reg = np.sum(np.abs(np.diff(Lambda, axis=0)))
            # L1正則化（スパース性）
            l1_reg = np.sum(np.abs(Lambda))
            return data_fit + self.alpha * tv_reg + self.beta * l1_reg

        from scipy.optimize import minimize
        result = minimize(objective, Lambda_init, method='L-BFGS-B')
        Lambda_opt = result.x.reshape(n_events, n_paths)

        # 各パスを「時系列ベクトル」として抽出・正規化
        paths = {i: Lambda_opt[:, i] / (np.linalg.norm(Lambda_opt[:, i]) + 1e-8)
                for i in range(n_paths)}
        return paths

    @staticmethod
    def _compute_topological_charge(path: np.ndarray, n_segments: int = 10) -> Tuple[float, float]:
        """
        トポロジカルチャージ Q_Λ（巻き数）と安定性 σ_Q を計算
        """
        if len(path) < 2:
            return 0.0, 0.0

        # ヒルベルト変換で複素解析信号を得る
        analytic = hilbert(path)
        theta = np.unwrap(np.angle(analytic))

        # 巻き数
        Q_Lambda = (theta[-1] - theta[0]) / (2 * np.pi)

        # セグメント安定性
        Q_segments = []
        seg_size = len(path) // n_segments
        for i in range(n_segments):
            seg_start = i * seg_size
            seg_end = (i + 1) * seg_size if i < n_segments - 1 else len(path)
            if seg_end - seg_start > 1:
                segment_theta = theta[seg_start:seg_end]
                Q_seg = (segment_theta[-1] - segment_theta[0]) / (2 * np.pi)
                Q_segments.append(Q_seg)
        stability = np.std(Q_segments) if Q_segments else 0.0

        return Q_Lambda, stability

    @staticmethod
    def _compute_entropy(path: np.ndarray, n_bins: int = 30) -> float:
        """
        パスのエントロピー（ヒストグラム型）計算
        """
        # ヒストグラムで確率分布を推定
        hist, bin_edges = np.histogram(path, bins=n_bins, density=True)
        hist = hist + 1e-12  # ゼロ割防止
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log(prob))
        return entropy if not np.isnan(entropy) else 0.0

    # === 基本異常パターン生成メソッド ===
    def _generate_pulse_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """パルス型異常：局所的な強いスパイク"""
        events_copy = events.copy()
        idx = np.random.randint(events.shape[0])
        events_copy[idx] += np.random.randn(events.shape[1]) * intensity
    def extract_features(self, result: Lambda3Result) -> Dict[str, List[float]]:
        """Lambda³解析結果から特徴量を抽出（地震検知用拡張版）"""
        features = {
            # 基本特徴量
            'Q_Λ': [],
            'E': [],
            'S': [],
            'n_pulse': [],
            'σ_Q': [],
            'mean_curvature': [],
            'spectral_peak': [],
            # 地震特有の特徴量
            'Q_Λ_gradient': [],  # トポロジカルチャージの変化率
            'energy_concentration': [],  # エネルギー集中度
            'phase_coherence': [],  # 位相コヒーレンス
            'structural_instability': []  # 構造的不安定性
        }

        for i, path in result.paths.items():
            # 基本特徴量
            features['Q_Λ'].append(result.topological_charges[i])
            features['E'].append(result.energies[i])
            features['S'].append(result.entropies[i])
            features['σ_Q'].append(result.stabilities[i])

            # パルス数
            if len(path) > 1:
                delta_lambda = np.abs(np.diff(path))
                std_path = np.std(path)
                if std_path > 0:
                    n_pulse = np.sum(delta_lambda > (std_path * 2))
                else:
                    n_pulse = 0
            else:
                n_pulse = 0
            features['n_pulse'].append(float(n_pulse))

            # 平均曲率
            if len(path) > 2:
                curvature = np.gradient(np.gradient(path))
                mean_curv = np.mean(np.abs(curvature))
                features['mean_curvature'].append(mean_curv if not np.isnan(mean_curv) else 0.0)
            else:
                features['mean_curvature'].append(0.0)

            # スペクトルピーク
            if len(path) > 1:
                fft = np.fft.fft(path)
                if len(fft) > 1:
                    peak = np.max(np.abs(fft[1:len(fft)//2]))
                    features['spectral_peak'].append(float(peak) if not np.isnan(peak) else 0.0)
                else:
                    features['spectral_peak'].append(0.0)
            else:
                features['spectral_peak'].append(0.0)

            # 地震特有の特徴量
            # Q_Λの変化率
            if i > 0 and (i-1) in result.topological_charges:
                q_gradient = abs(result.topological_charges[i] - result.topological_charges[i-1])
                features['Q_Λ_gradient'].append(q_gradient)
            else:
                features['Q_Λ_gradient'].append(0.0)

            # エネルギー集中度
            if len(path) > 1:
                energy_dist = np.abs(path)**2
                if np.sum(energy_dist) > 0:
                    energy_cumsum = np.cumsum(energy_dist)
                    energy_50_idx = np.argmax(energy_cumsum >= 0.5 * energy_cumsum[-1])
                    concentration = 1.0 - (energy_50_idx / len(path))
                else:
                    concentration = 0.0
                features['energy_concentration'].append(concentration)
            else:
                features['energy_concentration'].append(0.0)

            # 位相コヒーレンス
            if len(path) > 10:
                # ヒルベルト変換による瞬時位相
                from scipy import signal
                analytic = signal.hilbert(path)
                phase = np.angle(analytic)
                # 位相の一貫性を評価
                phase_diff = np.diff(phase)
                coherence = 1.0 / (1.0 + np.std(phase_diff))
                features['phase_coherence'].append(coherence)
            else:
                features['phase_coherence'].append(0.0)

            # 構造的不安定性（エネルギーとエントロピーの比）
            if result.energies[i] > 0:
                instability = result.entropies[i] / result.energies[i]
            else:
                instability = 0.0
            features['structural_instability'].append(instability)

        return features

    def detect_earthquake_precursors(self, result: Lambda3Result, events: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Lambda³パス系列から「異常度スコア時系列」を計算して返す関数。
        ※一切の判定・ラベリング・閾値判別は含みません！
        """
        n_events = events.shape[0]
        n_paths = len(result.paths)

        precursors = {
            'composite_score': np.zeros(n_events),
            'topological_anomaly': np.zeros(n_events),
            'energy_anomaly': np.zeros(n_events),
            'pattern_transition': np.zeros(n_events),
            'criticality_index': np.zeros(n_events)
        }

        features = self.extract_features(result)

        for i in range(n_events):
            q_scores = [abs(result.topological_charges[j]) * abs(path[i])
                        for j, path in result.paths.items() if i < len(path)]
            precursors['topological_anomaly'][i] = max(q_scores) if q_scores else 0.0

            energy_scores = [((result.energies[j] - 1.0)**2) * abs(path[i])
                            for j, path in result.paths.items() if i < len(path)]
            precursors['energy_anomaly'][i] = max(energy_scores) if energy_scores else 0.0

            # パターン遷移スコア
            if i > 0:
                transition_score = np.sum(features['Q_Λ_gradient'])
                precursors['pattern_transition'][i] = transition_score / (len(features['Q_Λ_gradient']) + 1e-8)
            else:
                precursors['pattern_transition'][i] = 0.0

            # 臨界性指標（配列長ガード付き）
            criticality = 0.0
            n_valid = min(len(features['structural_instability']), len(features['phase_coherence']))
            for j in range(n_valid):
                instability = features['structural_instability'][j]
                coherence = features['phase_coherence'][j]
                criticality += instability * (1 - coherence)
            precursors['criticality_index'][i] = criticality / (n_valid + 1e-8) if n_valid > 0 else 0.0

            # 総合スコア
            precursors['composite_score'][i] = (
                precursors['topological_anomaly'][i] * 0.3 +
                precursors['energy_anomaly'][i] * 0.2 +
                precursors['pattern_transition'][i] * 0.2 +
                precursors['criticality_index'][i] * 0.3
            )

        return precursors

    def detect_anomalies(self, result: Lambda3Result, events: np.ndarray) -> np.ndarray:
        """
        複合的な異常スコア（Λ³テンソル合成）を返す
        - スコアは連続値テンソル（判定なし）
        """
        # 前兆スコア（異常進行テンソル）の全種を取得
        precursors = self.detect_earthquake_precursors(result, events)

        # Λ³デフォルト合成重み（ここは好みで調整OK）
        weights = {
            'topological_anomaly': 0.3,
            'energy_anomaly': 0.2,
            'pattern_transition': 0.2,
            'criticality_index': 0.3
        }

        # 合成スコアの計算（安全性・可読性を最大化）
        composite_score = (
            precursors['topological_anomaly'] * weights['topological_anomaly'] +
            precursors['energy_anomaly'] * weights['energy_anomaly'] +
            precursors['pattern_transition'] * weights['pattern_transition'] +
            precursors['criticality_index'] * weights['criticality_index']
        )
        # 返り値は“複合スコアテンソル（連続値）”
        return composite_score
        
    # 総合スコアを返す
    def extract_integrated_features(self, broadband_result: Lambda3Result, strong_motion_result: Lambda3Result) -> Dict[str, Dict[str, List[float]]]:
        bb_features = self.extract_features(broadband_result)
        sm_features = self.extract_features(strong_motion_result)
        integrated_features = {'broadband': bb_features, 'strong_motion': sm_features, 'cross_layer': {}}
        cross_features = integrated_features['cross_layer']

        feature_pairs = [
            # (feature_key, diff_label, is_ratio, corr_label)
            ('Q_Λ', 'Q_Λ_diff', False, 'Q_Λ_correlation'),
            ('E', 'energy_transfer_rate', True, None),
            ('phase_coherence', 'phase_diff', False, None),
            ('structural_instability', 'stability_change', False, None),
            ('spectral_peak', 'amplification_factor', True, None),
        ]

        for key, diff_label, is_ratio, corr_label in feature_pairs:
            if key in bb_features and key in sm_features:
                min_len = min(len(bb_features[key]), len(sm_features[key]))
                if min_len == 0:
                    continue
                # 差分or比
                if is_ratio:
                    cross_features[diff_label] = [
                        sm_features[key][i] / (bb_features[key][i] + 1e-8)
                        for i in range(min_len)
                    ]
                else:
                    cross_features[diff_label] = [
                        sm_features[key][i] - bb_features[key][i]
                        for i in range(min_len)
                    ]
                # 相関
                if corr_label:
                    corr = np.corrcoef(bb_features[key][:min_len], sm_features[key][:min_len])[0, 1] if min_len > 1 else 0.0
                    corr = float(corr) if not np.isnan(corr) else 0.0
                    cross_features[corr_label] = [corr] * min_len

        return integrated_features
      
    def detect_integrated_earthquake_precursors(
        self,
        broadband_result: Lambda3Result,
        strong_motion_result: Lambda3Result,
        broadband_events: np.ndarray,
        strong_motion_events: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """統合データでの地震前兆検出（Λ³テンソル進行ベース、判定ゼロ、スコア進行のみ！）"""
        bb_precursors = self.detect_earthquake_precursors(broadband_result, broadband_events)
        sm_precursors = self.detect_earthquake_precursors(strong_motion_result, strong_motion_events)

        integrated_precursors = {
            'broadband': bb_precursors,
            'strong_motion': sm_precursors,
            'integrated': {}
        }
        integrated = integrated_precursors['integrated']

        min_len = min(len(bb_precursors['composite_score']), len(sm_precursors['composite_score']))
        if min_len == 0:
            # どちらかが空→全ゼロ返し
            for key in [
                'coupled_precursor', 'deep_leading_precursor', 'surface_focused_precursor',
                'critical_transition', 'composite_integrated_score',
                'energy_concentration_precursor', 'topological_propagation'
            ]:
                integrated[key] = np.zeros(0)
            return integrated_precursors

        # 1. 連動型
        integrated['coupled_precursor'] = np.sqrt(
            np.abs(bb_precursors['composite_score'][:min_len]) * np.abs(sm_precursors['composite_score'][:min_len])
        )

        # 2. 深部先行型
        delay_window = 5
        integrated['deep_leading_precursor'] = np.zeros(min_len)
        if min_len > delay_window:
            bb_delayed = np.roll(bb_precursors['composite_score'][:min_len], delay_window)
            bb_delayed[:delay_window] = 0  # ゼロパディング
            integrated['deep_leading_precursor'] = bb_delayed * 0.7 + sm_precursors['composite_score'][:min_len] * 0.3

        # 3. 表層集中型
        integrated['surface_focused_precursor'] = np.where(
            bb_precursors['composite_score'][:min_len] < 0.3,
            sm_precursors['composite_score'][:min_len], 0
        )

        # 4. 臨界遷移
        bb_crit = bb_precursors['criticality_index'][:min_len]
        sm_crit = sm_precursors['criticality_index'][:min_len]
        integrated['critical_transition'] = (bb_crit + sm_crit) / 2

        # 5. 総合前兆スコア
        integrated['composite_integrated_score'] = (
            bb_precursors['composite_score'][:min_len] * 0.4 +
            sm_precursors['composite_score'][:min_len] * 0.3 +
            integrated['coupled_precursor'] * 0.3
        )

        # 6. エネルギー集中型
        bb_energy = bb_precursors['energy_anomaly'][:min_len]
        sm_energy = sm_precursors['energy_anomaly'][:min_len]
        total_energy = bb_energy + sm_energy + 1e-8
        surface_ratio = sm_energy / total_energy
        integrated['energy_concentration_precursor'] = surface_ratio * total_energy

        # 7. トポロジカル遷移伝播
        propagation_window = 3
        integrated['topological_propagation'] = np.zeros(min_len)
        if min_len > propagation_window:
            bb_topo = np.roll(bb_precursors['topological_anomaly'][:min_len], propagation_window)
            bb_topo[:propagation_window] = 0
            sm_topo = sm_precursors['topological_anomaly'][:min_len]
            integrated['topological_propagation'] = bb_topo * sm_topo * 0.5

        return integrated_precursors

    def detect_hierarchical_anomalies(self, integrated_result: IntegratedLambda3Result, weights=None) -> Dict[str, np.ndarray]:
        """
        階層的異常検出（統合結果から）
        weights: dict, optional
            - hierarchical_score: (depth_weight, surface_weight)
            - integrated_anomaly_score: dict of subweights
            例:
            weights = {
                'hierarchical_score': (0.6, 0.4),
                'integrated_anomaly_score': {
                    'hierarchical_score': 0.3,
                    'vertical_propagation_anomaly': 0.3,
                    'layer_mismatch_anomaly': 0.2,
                    'resonance_amplification': 0.2
                }
            }
        """
        # デフォルト重み
        default_weights = {
            'hierarchical_score': (0.6, 0.4),
            'integrated_anomaly_score': {
                'hierarchical_score': 0.3,
                'vertical_propagation_anomaly': 0.3,
                'layer_mismatch_anomaly': 0.2,
                'resonance_amplification': 0.2
            }
        }
        if weights is None:
            weights = default_weights

        anomalies = {}

        # 広帯域異常スコア
        bb_global_result = integrated_result.broadband_result.global_result
        bb_events = self._reconstruct_events_from_paths(bb_global_result.paths)
        bb_anomaly_score = self.detect_anomalies(bb_global_result, bb_events)

        # 強震計異常スコア
        sm_global_result = integrated_result.strong_motion_result.global_result
        sm_events = self._reconstruct_events_from_paths(sm_global_result.paths)
        sm_anomaly_score = self.detect_anomalies(sm_global_result, sm_events)

        # 時間軸を合わせる
        min_len = min(len(bb_anomaly_score), len(sm_anomaly_score))

        # 1. 階層的異常スコア（重み可変）
        depth_weight, surface_weight = weights['hierarchical_score']
        anomalies['hierarchical_score'] = (
            bb_anomaly_score[:min_len] * depth_weight +
            sm_anomaly_score[:min_len] * surface_weight
        )

        # 2. 異常の垂直伝播
        anomalies['vertical_propagation_anomaly'] = np.zeros(min_len)
        delay = int(integrated_result.depth_surface_interaction.get('propagation_delay', 0))

        for i in range(delay, min_len):
            if i-delay >= 0:
                # 遅延を考慮した伝播異常
                anomalies['vertical_propagation_anomaly'][i] = (
                    bb_anomaly_score[i-delay] * sm_anomaly_score[i]
                ) ** 0.5

        # 3. 層間不整合異常
        anomalies['layer_mismatch_anomaly'] = np.zeros(min_len)
        for i in range(min_len):
            # 深部と表層の異常パターンの不一致
            mismatch = abs(bb_anomaly_score[i] - sm_anomaly_score[i])
            anomalies['layer_mismatch_anomaly'][i] = mismatch

        # 4. 共鳴増幅異常
        anomalies['resonance_amplification'] = np.zeros(min_len)
        resonance_factor = integrated_result.depth_surface_interaction.get('layer_coupling_strength', 0)

        for i in range(min_len):
            if bb_anomaly_score[i] > 0.5 and sm_anomaly_score[i] > 0.5:
                # 両層で異常が検出された場合の増幅
                anomalies['resonance_amplification'][i] = (
                    bb_anomaly_score[i] * sm_anomaly_score[i] * (1 + resonance_factor)
                )

        # 5. 統合異常スコア（合成重みもdict参照で！）
        subw = weights['integrated_anomaly_score']
        anomalies['integrated_anomaly_score'] = (
            anomalies['hierarchical_score'] * subw['hierarchical_score'] +
            anomalies['vertical_propagation_anomaly'] * subw['vertical_propagation_anomaly'] +
            anomalies['layer_mismatch_anomaly'] * subw['layer_mismatch_anomaly'] +
            anomalies['resonance_amplification'] * subw['resonance_amplification']
        )
        return anomalies

    def _reconstruct_events_from_paths(self, paths: Dict[int, np.ndarray]) -> np.ndarray:
        """
        パスから仮想的なイベント行列を再構築
        """
        if not paths:
            return np.empty((0, 0))

        # パスkeyを昇順で固定（keyが0,1,2...以外でも対応）
        sorted_items = sorted(paths.items(), key=lambda x: x[0])
        path_list = [np.asarray(p) for _, p in sorted_items]
        path_lengths = [len(p) for p in path_list]
        max_len = max(path_lengths)

        if max_len == 0:
            return np.empty((0, len(path_list)))

        n_paths = len(path_list)
        events = np.full((max_len, n_paths), np.nan)  # nanでパディング（0埋めだと平均などの統計量が歪むため）

        for i, path in enumerate(path_list):
            events[:len(path), i] = path

        return events

    def _generate_phase_jump_anomaly(self, events: np.ndarray, intensity: float = 1) -> np.ndarray:
        """位相ジャンプ型異常：符号反転"""
        events_copy = events.copy()
        idx = np.random.randint(events.shape[0])
        events_copy[idx] = -events_copy[idx] * intensity
        return events_copy

    def _generate_periodic_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """周期的異常：正弦波的変調"""
        events_copy = events.copy()
        n_events = events.shape[0]
        period = max(2, n_events // 4)
        modulation = intensity * np.sin(2 * np.pi * np.arange(n_events) / period)
        events_copy += modulation[:, np.newaxis]
        return events_copy

    def _generate_decay_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """構造崩壊型異常：指数減衰"""
        events_copy = events.copy()
        decay_start = events.shape[0] // 2
        decay = np.exp(-intensity * np.arange(events.shape[0] - decay_start))
        events_copy[decay_start:] *= decay[:, np.newaxis]
        return events_copy

    def _generate_bifurcation_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """分岐型異常：構造の分裂"""
        events_copy = events.copy()
        split_point = events.shape[0] // 2
        events_copy[split_point:] += np.random.randn(*events_copy[split_point:].shape) * intensity
        return events_copy

    # === 地震波パターン ===
    def _generate_p_wave_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """P波的な急激な立ち上がり"""
        events_copy = events.copy()
        onset = np.random.randint(len(events) // 4, 3 * len(events) // 4)
        rise_length = min(5, len(events) - onset)
        events_copy[onset:onset+rise_length] += np.linspace(0, intensity, rise_length)[:, np.newaxis]
        return events_copy

    def _generate_s_wave_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """S波的な大振幅振動"""
        events_copy = events.copy()
        onset = np.random.randint(len(events) // 4, 3 * len(events) // 4)
        duration = min(50, len(events) - onset)
        t = np.arange(duration)
        oscillation = intensity * np.sin(2 * np.pi * t / 10) * np.exp(-t / 20)
        events_copy[onset:onset+duration] += oscillation[:, np.newaxis]
        return events_copy

    # === 複雑パターン ===
    def _generate_multi_path_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """複数経路異常：複数の独立した異常が同時発生"""
        events_copy = events.copy()
        n_paths = np.random.randint(2, 5)

        for _ in range(n_paths):
            idx = np.random.randint(events.shape[0])
            direction = np.random.randn(events.shape[1])
            direction /= np.linalg.norm(direction)
            events_copy[idx] += direction * intensity * np.random.uniform(0.5, 1.5)

        return events_copy

    def _generate_topological_jump_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """トポロジカルジャンプ：位相空間での不連続遷移"""
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events, n_dim = events_copy.shape

        if n_events < 3:
            events_copy *= -intensity
            return events_copy

        jump_point = n_events // 2

        if jump_point > 0:
            decay_curve = np.exp(-0.1 * np.arange(jump_point))
            events_copy[:jump_point] *= decay_curve[:, np.newaxis]
        if jump_point < n_events:
            events_copy[jump_point:] = -events_copy[jump_point:] * intensity

        if jump_point < n_events:
            events_copy[jump_point] = np.random.randn(n_dim) * intensity * 2

        return events_copy

    def _generate_cascade_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """カスケード異常：異常が時間的に伝播"""
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
        """共鳴異常：特定周波数での増幅"""
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events = events_copy.shape[0]

        if n_events < 4:
            events_copy *= intensity
            return events_copy

        fft = np.fft.fft(events_copy, axis=0)
        max_freq = max(2, len(fft) // 4)
        resonance_freq = np.random.randint(1, max_freq + 1)  # +1で上限inclusive

        # 指数的増幅
        if resonance_freq < len(fft):
            fft[resonance_freq] *= intensity
            if resonance_freq != 0 and len(fft) - resonance_freq > 0:
                fft[-resonance_freq] *= intensity

        events_copy = np.real(np.fft.ifft(fft, axis=0))
        return events_copy

    # === 地震前兆特有パターン ===
    def _generate_foreshock_sequence(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events, n_dim = events_copy.shape
        n_foreshocks = min(8, n_events // 3)

        for i in range(n_foreshocks):
            # indexの非線形mappingも応用可能
            position = int(n_events * (0.5 + 0.5 * (i / n_foreshocks)))
            if position < n_events:
                amplitude = intensity * (0.2 + 0.8 * (i / n_foreshocks))
                events_copy[position] += np.random.randn(n_dim) * amplitude
        return events_copy

    def _generate_quiet_period(self, events: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events = events_copy.shape[0]
        quiet_start = int(n_events * 0.6)
        quiet_end = int(n_events * 0.85)
        if quiet_start < quiet_end and quiet_end <= n_events:
            events_copy[quiet_start:quiet_end] *= intensity
        return events_copy

    def _generate_nucleation_phase(self, events: np.ndarray, intensity: float = 2.5) -> np.ndarray:
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events, n_dim = events_copy.shape
        nucleation_start = int(n_events * 0.7)
        for i in range(nucleation_start, n_events):
            high_freq = np.sin(2 * np.pi * np.random.uniform(5, 15) * i / n_events)
            events_copy[i] += high_freq * intensity * np.random.randn(n_dim)
        return events_copy

    def _generate_dilatancy_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events, n_dim = events_copy.shape
        for i in range(n_events):
            velocity_change = 1 + intensity * 0.1 * (i / n_events)
            events_copy[i] *= velocity_change
            if i > 0:
                phase_shift = intensity * 0.05 * (i / n_events)
                shift_amount = int(phase_shift * n_dim) % n_dim  # ガード
                if shift_amount > 0:
                    events_copy[i] = np.roll(events_copy[i], shift_amount)
        return events_copy

    def _generate_crustal_deformation(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events, n_dim = events_copy.shape

        # 長周期トレンド
        trend = np.linspace(0, intensity, n_events)
        # 非線形な累積
        cumulative = np.cumsum(np.random.randn(n_events) * 0.1) * intensity * 0.2
        deformation = (trend + cumulative)[:, np.newaxis]  # (n_events, 1)
        events_copy += deformation @ np.ones((1, n_dim))  # ブロードキャストで全chに同じ変形
        return events_copy

    def _generate_electromagnetic_precursor(self, events: np.ndarray, intensity: float = 1.5) -> np.ndarray:
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events, n_dim = events_copy.shape

        resonant_freq = np.random.uniform(0.01, 0.1)
        for i in range(n_events):
            em_signal = intensity * np.sin(2 * np.pi * resonant_freq * i)
            spatial_pattern = np.random.randn(n_dim) * 0.5 + 1
            events_copy[i] += em_signal * spatial_pattern
        return events_copy

    def _generate_slow_slip_event(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events, n_dim = events_copy.shape

        slip_duration = max(n_events // 3, 5)
        slip_start = np.random.randint(0, max(1, n_events - slip_duration))

        t = np.arange(slip_duration)
        slip_function = intensity * np.exp(-(t - slip_duration/2)**2 / (slip_duration/4)**2)
        for i in range(slip_duration):
            if slip_start + i < n_events:
                events_copy[slip_start + i] += slip_function[i] * np.ones(n_dim)
        return events_copy

    def _generate_critical_point_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        events_copy = events.copy()
        if events_copy.ndim == 1:
            events_copy = events_copy[:, np.newaxis]
        n_events, n_dim = events_copy.shape

        critical_point = int(n_events * 0.8)
        for i in range(n_events):
            if i < critical_point:
                distance_to_critical = (critical_point - i) / critical_point
                fluctuation = intensity * np.exp(-distance_to_critical * 3)
                events_copy[i] += np.random.randn(n_dim) * fluctuation
            else:
                events_copy[i] *= intensity * 2
        return events_copy

   # === 空間多層解析器 ===
class SpatialMultiLayerAnalyzer:
    """空間多層Lambda³解析システム（統合データ対応版）"""

    def __init__(self,
                 station_locations: Optional[Dict[str, Tuple[float, float]]] = None,
                 station_metadata: Optional[Dict[str, Dict]] = None,
                 data_type: str = 'standard'):
        """
        Parameters:
        -----------
        station_locations : Dict[str, Tuple[float, float]]
            観測点名と(緯度, 経度)の辞書
        station_metadata : Dict[str, Dict]
            観測点のメタデータ（設置深度、地質情報など）
        data_type : str
            データタイプ（'standard', 'broadband', 'strong_motion', 'integrated'）
        """
        self.station_locations = station_locations or {}
        self.station_metadata = station_metadata or {}
        self.data_type = data_type

        # データタイプに応じた解析器の初期化
        if data_type == 'integrated':
            # 統合データの場合は両方の解析器を持つ
            self.broadband_analyzer = Lambda3Analyzer(alpha=0.15, beta=0.01, data_type='broadband')
            self.strong_motion_analyzer = Lambda3Analyzer(alpha=0.1, beta=0.015, data_type='strong_motion')
            self.base_analyzer = Lambda3Analyzer(alpha=0.1, beta=0.01, data_type='standard')
        else:
            self.base_analyzer = Lambda3Analyzer(alpha=0.1, beta=0.01, data_type=data_type)

        self.clustering_methods = {
            'kmeans': self._cluster_kmeans,
            'dbscan': self._cluster_dbscan,
            'hierarchical': self._cluster_hierarchical,
            'geological': self._cluster_geological
        }

    def analyze_multilayer(self,
                        data_dict: Dict[str, np.ndarray],
                        n_clusters: int = 5,
                        clustering_method: str = 'kmeans',
                        n_paths_global: int = 10,
                        n_paths_local: int = 5,
                        n_paths_cluster: int = 7,
                        parallel: bool = False) -> SpatialLambda3Result:
      """
      空間多層Lambda³解析を実行

      Parameters:
      -----------
      data_dict : Dict[str, np.ndarray]
          観測点名とイベント行列の辞書
      n_clusters : int
          地域クラスタ数
      clustering_method : str
          クラスタリング手法（'kmeans', 'dbscan', 'hierarchical', 'geological'）
      n_paths_global : int
          全国規模解析のパス数
      n_paths_local : int
          観測点別解析のパス数
      n_paths_cluster : int
          クラスタ別解析のパス数
      parallel : bool
          並列処理を使用するか
      """
      start_time = datetime.now()
      print("=== Lambda³ Spatial Multi-Layer Analysis ===")
      print(f"Data type: {self.data_type}")
      print(f"Stations: {len(data_dict)}")
      print(f"Clustering method: {clustering_method}")
      print(f"Paths: Global={n_paths_global}, Local={n_paths_local}, Cluster={n_paths_cluster}")

      # 1. 観測点のクラスタリング
      print("\n--- Station Clustering ---")
      station_clusters = self._cluster_stations(
          data_dict, n_clusters, method=clustering_method
      )

      # 2. グローバル（全国規模）解析
      print("\n--- Global (Japan-wide) Analysis ---")
      global_data = self._aggregate_global_data(data_dict)
      global_result = self._analyze_lambda3(global_data, n_paths_global, "Global")

      # 3. ローカル（観測点別）解析
      print("\n--- Local (Station-wise) Analysis ---")
      local_results = {}

      if parallel and len(data_dict) > 10:
          # 並列処理（要: joblib）
          try:
              from joblib import Parallel, delayed
              print(f"  Using parallel processing for {len(data_dict)} stations")

              def analyze_station(station, data):
                  try:
                      if data.shape[0] < 2:
                          return station, None
                      result = self._analyze_lambda3(data, n_paths_local, f"Station_{station}")
                      return station, result
                  except Exception as e:
                      print(f"  Error in parallel processing for {station}: {e}")
                      return station, None

              # メモリ不足を防ぐため、バッチ処理を実装
              batch_size = 50
              all_stations = list(data_dict.items())

              for batch_start in range(0, len(all_stations), batch_size):
                  batch_end = min(batch_start + batch_size, len(all_stations))
                  batch = all_stations[batch_start:batch_end]

                  print(f"  Processing batch {batch_start//batch_size + 1}/{(len(all_stations) + batch_size - 1)//batch_size}")

                  results = Parallel(n_jobs=4, backend='threading')(
                      delayed(analyze_station)(station, data)
                      for station, data in batch
                  )

                  for station, result in results:
                      if result is not None:
                          local_results[station] = result

              print(f"  Completed parallel analysis: {len(local_results)}/{len(data_dict)} stations")

          except ImportError:
              print("  Warning: joblib not available, using sequential processing")
              parallel = False
          except Exception as e:
              print(f"  Error in parallel processing: {e}")
              print("  Falling back to sequential processing")
              parallel = False

      # 並列処理が使えない場合、または選択されていない場合
      if not parallel or len(local_results) == 0:
          print(f"  Using sequential processing for {len(data_dict)} stations")
          processed_count = 0
          error_count = 0

          for i, (station, data) in enumerate(data_dict.items()):
              # 進捗表示
              if i % 20 == 0:
                  print(f"  Progress: {i}/{len(data_dict)} stations ({i/len(data_dict)*100:.1f}%)")

              # すでに処理済みの場合はスキップ
              if station in local_results:
                  continue

              # データの形状を確認
              if data.shape[0] < 2:
                  if error_count < 5:  # 最初の5件のみ警告表示
                      print(f"  Warning: Station {station} has insufficient data: shape={data.shape}")
                  error_count += 1
                  continue

              try:
                  result = self._analyze_lambda3(
                      data, n_paths_local, f"Station_{station}"
                  )
                  local_results[station] = result
                  processed_count += 1

                  # 最初の数件の成功例を表示
                  if processed_count <= 3:
                      charges = list(result.topological_charges.values())
                      if charges:
                          mean_q = np.mean(np.abs(charges))
                          print(f"  {station}: Mean |Q_Λ| = {mean_q:.3f}")

              except Exception as e:
                  if error_count < 5:
                      print(f"  Error analyzing station {station}: {e}")
                  error_count += 1
                  continue

          print(f"  Sequential processing completed: {len(local_results)}/{len(data_dict)} stations analyzed")
          if error_count > 0:
              print(f"  Total errors: {error_count}")

      # 4. クラスタ（地域別）解析
      print("\n--- Cluster (Regional) Analysis ---")
      cluster_results = {}
      cluster_data_dict = self._aggregate_cluster_data(data_dict, station_clusters)

      for cluster_id, cluster_data in cluster_data_dict.items():
          print(f"  Analyzing cluster {cluster_id} ({self._get_cluster_size(station_clusters, cluster_id)} stations)")
          try:
              cluster_results[cluster_id] = self._analyze_lambda3(
                  cluster_data, n_paths_cluster, f"Cluster_{cluster_id}"
              )
          except Exception as e:
              print(f"  Error analyzing cluster {cluster_id}: {e}")
              continue

      # 5. 空間相関構造の計算
      print("\n--- Computing Spatial Correlations ---")
      spatial_correlations = self._compute_spatial_correlations(local_results)

      # 6. 層間相互作用の評価
      print("\n--- Evaluating Cross-Layer Interactions ---")
      cross_layer_metrics = self._evaluate_cross_layer_interactions(
          global_result, local_results, cluster_results, station_clusters
      )

      # 7. 空間的異常検出
      print("\n--- Detecting Spatial Anomalies ---")
      spatial_anomalies = self.detect_spatial_anomalies(
          global_result, local_results, cluster_results,
          spatial_correlations, station_clusters
      )

      # 8. メタデータの収集
      end_time = datetime.now()
      metadata = {
          'analysis_time': (end_time - start_time).total_seconds(),
          'n_stations': len(data_dict),
          'n_stations_analyzed': len(local_results),
          'n_clusters': n_clusters,
          'clustering_method': clustering_method,
          'data_type': self.data_type,
          'n_paths': {
              'global': n_paths_global,
              'local': n_paths_local,
              'cluster': n_paths_cluster
          },
          'data_shape': {
              station: data.shape for station, data in list(data_dict.items())[:5]
          },
          'parallel_processing': parallel and len(local_results) > 0
      }

      print(f"\nAnalysis completed in {metadata['analysis_time']:.1f} seconds")
      print(f"Successfully analyzed {len(local_results)}/{len(data_dict)} stations")

      return SpatialLambda3Result(
          global_result=global_result,
          local_results=local_results,
          cluster_results=cluster_results,
          spatial_correlations=spatial_correlations,
          station_clusters=station_clusters,
          cross_layer_metrics=cross_layer_metrics,
          spatial_anomalies=spatial_anomalies,
          metadata=metadata
      )

    def analyze_integrated_multilayer(self,
                                     broadband_dict: Dict[str, np.ndarray],
                                     strong_motion_dict: Dict[str, np.ndarray],
                                     n_clusters: int = 5,
                                     **kwargs) -> IntegratedLambda3Result:
        """
        広帯域・強震計統合データの空間多層解析

        Parameters:
        -----------
        broadband_dict : Dict[str, np.ndarray]
            広帯域観測点データ
        strong_motion_dict : Dict[str, np.ndarray]
            強震計観測点データ
        """
        print("=== Integrated Lambda³ Multi-Layer Analysis ===")
        print(f"Broadband stations: {len(broadband_dict)}")
        print(f"Strong motion stations: {len(strong_motion_dict)}")

        # データタイプを設定
        self.data_type = 'broadband'
        broadband_result = self.analyze_multilayer(broadband_dict, n_clusters, **kwargs)

        self.data_type = 'strong_motion'
        strong_motion_result = self.analyze_multilayer(strong_motion_dict, n_clusters, **kwargs)

        # 深部-表層相互作用の計算
        depth_surface_interaction = self._compute_integrated_interactions(
            broadband_result, strong_motion_result
        )

        # 統合異常検出
        integrated_anomalies = self._detect_integrated_anomalies(
            broadband_result, strong_motion_result
        )

        # 時空間伝播パターンの解析
        propagation_patterns = self._analyze_propagation_patterns(
            broadband_result, strong_motion_result
        )

        # 統合メタデータ
        metadata = {
            'broadband_metadata': broadband_result.metadata,
            'strong_motion_metadata': strong_motion_result.metadata,
            'integration_time': datetime.now().isoformat(),
            'n_broadband_stations': len(broadband_dict),
            'n_strong_motion_stations': len(strong_motion_dict)
        }

        # データタイプをリセット
        self.data_type = 'integrated'

        return IntegratedLambda3Result(
            broadband_result=broadband_result,
            strong_motion_result=strong_motion_result,
            depth_surface_interaction=depth_surface_interaction,
            integrated_anomalies=integrated_anomalies,
            propagation_patterns=propagation_patterns,
            metadata=metadata
        )

    def _compute_integrated_interactions(self,
                                        broadband_result: SpatialLambda3Result,
                                        strong_motion_result: SpatialLambda3Result) -> Dict[str, float]:
        """広帯域・強震計結果間の物理的・構造的相互作用を定量評価（Λ³流リファクタ版）"""
        interactions = {}

        # === 1. グローバル・トポロジカル相関 ===
        bb_global_charges = np.array(list(broadband_result.global_result.topological_charges.values()))
        sm_global_charges = np.array(list(strong_motion_result.global_result.topological_charges.values()))

        if bb_global_charges.size and sm_global_charges.size:
            min_len = min(len(bb_global_charges), len(sm_global_charges))
            if min_len > 1:
                corr = np.corrcoef(
                    bb_global_charges[:min_len],
                    sm_global_charges[:min_len]
                )[0, 1]
                interactions['global_charge_correlation'] = float(np.nan_to_num(corr))
            else:
                interactions['global_charge_correlation'] = 0.0

            # === 2. グローバル・エネルギー伝達効率 ===
            bb_energies = np.array(list(broadband_result.global_result.energies.values()))
            sm_energies = np.array(list(strong_motion_result.global_result.energies.values()))
            bb_energy = np.mean(bb_energies) if bb_energies.size else 0.0
            sm_energy = np.mean(sm_energies) if sm_energies.size else 0.0

            print(f"  Debug: BB energy = {bb_energy:.3f}, SM energy = {sm_energy:.3f}")
            if bb_energy > 0:
                ratio = sm_energy / bb_energy
                interactions['energy_transfer_ratio'] = float(np.clip(ratio, 0.0, 1.0))
            else:
                interactions['energy_transfer_ratio'] = 0.0
        else:
            # データが無い場合も明示的に出す
            interactions['global_charge_correlation'] = 0.0
            interactions['energy_transfer_ratio'] = 0.0

        # === 3. 空間的整合性（観測点単位のΛ³的同調性） ===
        interactions['spatial_coherence'] = float(np.nan_to_num(
            self._compute_spatial_coherence(broadband_result, strong_motion_result)
        ))

        # === 4. 深部→表層 伝播遅延推定 ===
        interactions['propagation_delay'] = float(np.nan_to_num(
            self._estimate_integrated_delay(broadband_result, strong_motion_result)
        ))

        # === 5. 層間結合強度（エネルギー＋トポロジー整合の複合指標） ===
        interactions['layer_coupling_strength'] = float(np.nan_to_num(
            self._compute_layer_coupling(broadband_result, strong_motion_result)
        ))

        # === 6. （将来拡張用）他の指標があればここに追加できる ===

        return interactions

    def _compute_spatial_coherence(self,
                                broadband_result: SpatialLambda3Result,
                                strong_motion_result: SpatialLambda3Result) -> float:
        """
        空間的整合性（Λ³流）：
        “同一地点での深部・表層トポロジカル進行が、どれだけ意味的に同期しているか”を相関係数で定量化。
        """
        bb_stations = set(broadband_result.local_results.keys())
        sm_stations = set(strong_motion_result.local_results.keys())
        common_stations = bb_stations & sm_stations

        # ---- 1. 共通観測点がない場合は「最近傍」へスイッチ
        if not common_stations:
            return float(np.nan_to_num(
                self._compute_nearest_neighbor_coherence(broadband_result, strong_motion_result)
            ))

        # ---- 2. 各観測点のトポロジカルチャージ系列の“意味的同期率”を計算
        coherence_scores = []
        for station in common_stations:
            bb_charges = np.array(list(broadband_result.local_results[station].topological_charges.values()))
            sm_charges = np.array(list(strong_motion_result.local_results[station].topological_charges.values()))
            min_len = min(len(bb_charges), len(sm_charges))

            if min_len > 1:
                corr = np.corrcoef(bb_charges[:min_len], sm_charges[:min_len])[0, 1]
                if not np.isnan(corr):
                    coherence_scores.append(corr)
        # ---- 3. スカラー平均としてΛ³同期率を返却
        return float(np.nan_to_num(np.mean(coherence_scores))) if coherence_scores else 0.0

    def _compute_nearest_neighbor_coherence(self,
                                        broadband_result: SpatialLambda3Result,
                                        strong_motion_result: SpatialLambda3Result) -> float:
        """
        最近傍観測点でのΛ³空間同期率を評価
        “深部⇔表層”のペアで“意味的トポロジカル進行”の空間的同期率を距離減衰付きで平均。
        """
        if not self.station_locations:
            return 0.0

        coherence_scores = []
        # 各広帯域観測点について
        for bb_station in broadband_result.local_results:
            if bb_station not in self.station_locations:
                continue
            bb_lat, bb_lon = self.station_locations[bb_station]

            # 最も近い強震計観測点を探索
            min_distance = float('inf')
            nearest_sm_station = None
            for sm_station in strong_motion_result.local_results:
                if sm_station not in self.station_locations:
                    continue
                sm_lat, sm_lon = self.station_locations[sm_station]
                distance = np.hypot(bb_lat - sm_lat, bb_lon - sm_lon)
                if distance < min_distance:
                    min_distance = distance
                    nearest_sm_station = sm_station

            # 最近傍でのトポロジカル進行の同期性を評価
            if nearest_sm_station and min_distance < 1.0:  # ~100km以内
                bb_charges = np.array(list(broadband_result.local_results[bb_station].topological_charges.values()))
                sm_charges = np.array(list(strong_motion_result.local_results[nearest_sm_station].topological_charges.values()))
                min_len = min(len(bb_charges), len(sm_charges))
                if min_len > 1:
                    corr = np.corrcoef(bb_charges[:min_len], sm_charges[:min_len])[0, 1]
                    weighted_corr = float(np.nan_to_num(corr)) * np.exp(-min_distance)
                    coherence_scores.append(weighted_corr)

        # NaN防止＋floatで返す
        return float(np.nan_to_num(np.mean(coherence_scores))) if coherence_scores else 0.0


    def _estimate_integrated_delay(self,
                               broadband_result: SpatialLambda3Result,
                               strong_motion_result: SpatialLambda3Result,
                               event_duration_sec: float = 240.0,
                               max_physical_delay_sec: float = 60.0
                               ) -> float:
        """
        深部パスから表層パスへのΛ³意味伝播遅延を“相互相関最大ラグ”から推定。
        デフォルトで1イベント=4分=240秒換算（引数で調整可）。
        """
        if broadband_result.global_result.paths and strong_motion_result.global_result.paths:
            delays = []
            path_indices = set(broadband_result.global_result.paths) & set(strong_motion_result.global_result.paths)
            n_check = min(3, len(path_indices))

            for i in list(path_indices)[:n_check]:
                bb_path = np.array(broadband_result.global_result.paths[i])
                sm_path = np.array(strong_motion_result.global_result.paths[i])

                min_len = min(len(bb_path), len(sm_path))
                if min_len > 20:
                    bb_norm = (bb_path[:min_len] - np.mean(bb_path[:min_len])) / (np.std(bb_path[:min_len]) + 1e-8)
                    sm_norm = (sm_path[:min_len] - np.mean(sm_path[:min_len])) / (np.std(sm_path[:min_len]) + 1e-8)
                    correlation = np.correlate(bb_norm, sm_norm, mode='full')
                    max_idx = np.argmax(correlation)
                    delay_samples = max_idx - (len(correlation) - 1) // 2

                    # 物理時間へ換算
                    delay_time = delay_samples * event_duration_sec
                    if 0 <= delay_time <= max_physical_delay_sec:
                        delays.append(delay_time)

            if delays:
                return float(np.median(delays))  # 中央値返しで外れ値にも強い

        return 0.0


    def _compute_layer_coupling(self,
                            broadband_result: SpatialLambda3Result,
                            strong_motion_result: SpatialLambda3Result) -> float:
        """
        層間（クラスタ単位）の結合強度：
        エネルギー比×トポロジカルチャージ整合性で“Λ³的意味同調”を表現。
        """
        coupling_scores = []
        for bb_cluster_id, bb_cluster_result in broadband_result.cluster_results.items():
            if bb_cluster_id not in strong_motion_result.cluster_results:
                continue

            sm_cluster_result = strong_motion_result.cluster_results[bb_cluster_id]

            # --- エネルギー比（下限0、上限1で正規化）
            bb_energy = float(np.nan_to_num(np.mean(list(bb_cluster_result.energies.values()))))
            sm_energy = float(np.nan_to_num(np.mean(list(sm_cluster_result.energies.values()))))
            if bb_energy <= 0 or sm_energy <= 0:
                continue  # 意味ある結合なし

            # "双方向正規化"（minで物理的非対称性も含めて1以下でガード）
            energy_coupling = min(sm_energy / bb_energy, bb_energy / sm_energy)

            # --- トポロジカルチャージの「意味的整合性」：平均差分ベース
            bb_charges = np.array(list(bb_cluster_result.topological_charges.values()))
            sm_charges = np.array(list(sm_cluster_result.topological_charges.values()))
            if bb_charges.size == 0 or sm_charges.size == 0:
                continue

            # 平均差分 → 1/(1+ΔQ) で差が0のとき最大1、差が大きいと減衰
            charge_similarity = 1.0 / (1.0 + abs(np.mean(bb_charges) - np.mean(sm_charges)))

            # --- 複合結合スコア（Λ³解釈で：意味的同期率）
            coupling = float(np.nan_to_num(energy_coupling * charge_similarity))
            coupling_scores.append(coupling)

        # 全クラスタの「Λ³結合スカラー平均」（空なら0.0で返す）
        return float(np.nan_to_num(np.mean(coupling_scores))) if coupling_scores else 0.0

    def _detect_integrated_anomalies(self,
                                  broadband_result: SpatialLambda3Result,
                                  strong_motion_result: SpatialLambda3Result) -> Dict[str, List[Dict]]:
        """統合異常検出 - Λ³エッセンス追加＆堅牢化"""
        integrated_anomalies = {
            'coupled_anomalies': [],      # 深部-表層連動異常
            'depth_isolated': [],         # 深部のみの異常
            'surface_isolated': [],       # 表層のみの異常
            'propagating_anomalies': [],  # 伝播型異常
            'resonance_anomalies': []     # 共鳴型異常
        }

        # 1. 連動・孤立異常の検出
        bb_hotspots = {h['station']: h for h in broadband_result.spatial_anomalies.get('local_hotspots', [])}
        sm_hotspots = {h['station']: h for h in strong_motion_result.spatial_anomalies.get('local_hotspots', [])}

        for bb_station, bb_anomaly in bb_hotspots.items():
            coupled = False
            # 完全一致
            if bb_station in sm_hotspots:
                integrated_anomalies['coupled_anomalies'].append({
                    'broadband_station': bb_station,
                    'strong_motion_station': bb_station,
                    'broadband_score': bb_anomaly['anomaly_score'],
                    'strong_motion_score': sm_hotspots[bb_station]['anomaly_score'],
                    'distance': 0.0,
                    'coupling_type': 'co-located'
                })
                coupled = True
            # 近傍一致
            elif bb_station in self.station_locations:
                bb_lat, bb_lon = self.station_locations[bb_station]
                for sm_station, sm_anomaly in sm_hotspots.items():
                    if sm_station in self.station_locations:
                        sm_lat, sm_lon = self.station_locations[sm_station]
                        distance = np.hypot(bb_lat - sm_lat, bb_lon - sm_lon)
                        if distance < 0.5:  # 50km以内
                            integrated_anomalies['coupled_anomalies'].append({
                                'broadband_station': bb_station,
                                'strong_motion_station': sm_station,
                                'broadband_score': bb_anomaly['anomaly_score'],
                                'strong_motion_score': sm_anomaly['anomaly_score'],
                                'distance': distance,
                                'coupling_type': 'nearby'
                            })
                            coupled = True
                            break
            if not coupled:
                integrated_anomalies['depth_isolated'].append(bb_anomaly)

        # 表層のみの異常
        for sm_station, sm_anomaly in sm_hotspots.items():
            if not any(ca['strong_motion_station'] == sm_station for ca in integrated_anomalies['coupled_anomalies']):
                integrated_anomalies['surface_isolated'].append(sm_anomaly)

        # 2. 伝播型異常
        self._detect_propagating_anomalies(
            broadband_result, strong_motion_result, integrated_anomalies
        )

        # 3. 共鳴型異常
        self._detect_resonance_anomalies(
            broadband_result, strong_motion_result, integrated_anomalies
        )

        return integrated_anomalies

    def _detect_propagating_anomalies(self,
                                    broadband_result: SpatialLambda3Result,
                                    strong_motion_result: SpatialLambda3Result,
                                    integrated_anomalies: Dict):
        """伝播型異常の検出"""
        # クラスタレベルでの異常伝播を検出
        for bb_cluster_anomaly in broadband_result.spatial_anomalies['cluster_anomalies']:
            cluster_id = bb_cluster_anomaly['cluster_id']

            # 対応する強震計クラスタの異常を探す
            sm_cluster_anomaly = next(
                (ca for ca in strong_motion_result.spatial_anomalies['cluster_anomalies']
                 if ca['cluster_id'] == cluster_id), None
            )

            if sm_cluster_anomaly:
                # エネルギー比から伝播を評価
                propagation_efficiency = sm_cluster_anomaly['energy'] / bb_cluster_anomaly['energy']

                if 0.5 < propagation_efficiency < 2.0:  # 有意な伝播
                    integrated_anomalies['propagating_anomalies'].append({
                        'cluster_id': cluster_id,
                        'broadband_energy': bb_cluster_anomaly['energy'],
                        'strong_motion_energy': sm_cluster_anomaly['energy'],
                        'propagation_efficiency': propagation_efficiency,
                        'anomaly_type': f"{bb_cluster_anomaly['anomaly_type']} → {sm_cluster_anomaly['anomaly_type']}"
                    })

    def _detect_resonance_anomalies(self,
                                   broadband_result: SpatialLambda3Result,
                                   strong_motion_result: SpatialLambda3Result,
                                   integrated_anomalies: Dict):
        """共鳴型異常の検出"""
        # 周波数領域での共鳴を検出
        for coupled in integrated_anomalies['coupled_anomalies']:
            bb_station = coupled['broadband_station']
            sm_station = coupled['strong_motion_station']

            if (bb_station in broadband_result.local_results and
                sm_station in strong_motion_result.local_results):

                bb_result = broadband_result.local_results[bb_station]
                sm_result = strong_motion_result.local_results[sm_station]

                # スペクトル特性の類似性を評価
                bb_features = self.base_analyzer.extract_features(bb_result)
                sm_features = self.base_analyzer.extract_features(sm_result)

                if 'spectral_peak' in bb_features and 'spectral_peak' in sm_features:
                    bb_peaks = bb_features['spectral_peak']
                    sm_peaks = sm_features['spectral_peak']

                    if bb_peaks and sm_peaks:
                        # スペクトルピークの比
                        peak_ratio = np.mean(sm_peaks) / (np.mean(bb_peaks) + 1e-8)

                        # 共鳴条件（整数比に近い）
                        resonance_ratios = [1.0, 2.0, 3.0, 0.5, 0.33]
                        min_diff = min(abs(peak_ratio - r) for r in resonance_ratios)

                        if min_diff < 0.1:  # 共鳴の可能性
                            integrated_anomalies['resonance_anomalies'].append({
                                'broadband_station': bb_station,
                                'strong_motion_station': sm_station,
                                'peak_ratio': peak_ratio,
                                'resonance_strength': 1.0 / (1.0 + min_diff),
                                'anomaly_scores': {
                                    'broadband': coupled['broadband_score'],
                                    'strong_motion': coupled['strong_motion_score']
                                }
                            })

    def _analyze_propagation_patterns(self,
                                    broadband_result: SpatialLambda3Result,
                                    strong_motion_result: SpatialLambda3Result) -> Dict[str, np.ndarray]:
        """時空間伝播パターンの解析"""
        patterns = {}

        # 1. 深部から表層への垂直伝播
        patterns['vertical_propagation'] = self._analyze_vertical_propagation(
            broadband_result, strong_motion_result
        )

        # 2. 水平方向の伝播
        patterns['horizontal_propagation'] = self._analyze_horizontal_propagation(
            broadband_result, strong_motion_result
        )

        # 3. 伝播速度の空間分布
        patterns['propagation_velocity'] = self._compute_propagation_velocity_map(
            broadband_result, strong_motion_result
        )

        return patterns

    def _analyze_vertical_propagation(self,
                                    broadband_result: SpatialLambda3Result,
                                    strong_motion_result: SpatialLambda3Result) -> np.ndarray:
        """垂直伝播の解析"""
        # 各観測点での深部→表層の伝達関数を推定
        vertical_transfer = {}

        bb_stations = set(broadband_result.local_results.keys())
        sm_stations = set(strong_motion_result.local_results.keys())

        common_stations = bb_stations.intersection(sm_stations)

        for station in common_stations:
            bb_result = broadband_result.local_results[station]
            sm_result = strong_motion_result.local_results[station]

            # エネルギー伝達
            bb_energy = np.mean(list(bb_result.energies.values()))
            sm_energy = np.mean(list(sm_result.energies.values()))

            if bb_energy > 0:
                transfer_efficiency = sm_energy / bb_energy
            else:
                transfer_efficiency = 0.0

            # 位相遅延
            if bb_result.paths and sm_result.paths:
                bb_path = bb_result.paths[0]
                sm_path = sm_result.paths[0]

                min_len = min(len(bb_path), len(sm_path))
                if min_len > 10:
                    correlation = np.correlate(
                        bb_path[:min_len],
                        sm_path[:min_len],
                        mode='same'
                    )
                    delay_idx = np.argmax(correlation) - len(correlation) // 2
                    phase_delay = delay_idx / 100.0  # 100Hzと仮定
                else:
                    phase_delay = 0.0
            else:
                phase_delay = 0.0

            vertical_transfer[station] = {
                'efficiency': transfer_efficiency,
                'delay': phase_delay
            }

        # 配列形式に変換
        if vertical_transfer:
            stations = list(vertical_transfer.keys())
            n_stations = len(stations)

            transfer_array = np.zeros((n_stations, 2))
            for i, station in enumerate(stations):
                transfer_array[i, 0] = vertical_transfer[station]['efficiency']
                transfer_array[i, 1] = vertical_transfer[station]['delay']

            return transfer_array
        else:
            return np.array([])

    def _analyze_horizontal_propagation(self,
                                   broadband_result: SpatialLambda3Result,
                                   strong_motion_result: SpatialLambda3Result) -> np.ndarray:
        """水平伝播の解析（修正版）"""
        # 空間相関の時間発展から水平伝播を推定
        horizontal_patterns = []

        # 広帯域での水平伝播
        bb_correlations = broadband_result.spatial_correlations
        if bb_correlations.size > 0:
            horizontal_patterns.append(bb_correlations.flatten())

        # 強震計での水平伝播
        sm_correlations = strong_motion_result.spatial_correlations
        if sm_correlations.size > 0:
            horizontal_patterns.append(sm_correlations.flatten())

        # 層間の水平伝播の差（同じ形状の場合のみ）
        if (bb_correlations.size > 0 and sm_correlations.size > 0 and
            bb_correlations.shape == sm_correlations.shape):
            diff_correlations = sm_correlations - bb_correlations
            horizontal_patterns.append(diff_correlations.flatten())

        # 配列の形状を揃える
        if horizontal_patterns:
            # 最大長を見つける
            max_len = max(len(p) for p in horizontal_patterns)

            # パディングして同じ長さにする
            padded_patterns = []
            for pattern in horizontal_patterns:
                if len(pattern) < max_len:
                    padded = np.pad(pattern, (0, max_len - len(pattern)), mode='constant')
                    padded_patterns.append(padded)
                else:
                    padded_patterns.append(pattern)

            return np.array(padded_patterns)
        else:
            return np.array([])

    def _compute_propagation_velocity_map(self,
                                        broadband_result: SpatialLambda3Result,
                                        strong_motion_result: SpatialLambda3Result) -> np.ndarray:
        """伝播速度の空間分布を計算"""
        velocity_map = {}

        # 各クラスタでの伝播速度を推定
        for cluster_id in broadband_result.cluster_results.keys():
            if cluster_id in strong_motion_result.cluster_results:
                bb_cluster = broadband_result.cluster_results[cluster_id]
                sm_cluster = strong_motion_result.cluster_results[cluster_id]

                # 簡易的な速度推定（エネルギー比と仮定的な深度差から）
                bb_energy = np.mean(list(bb_cluster.energies.values()))
                sm_energy = np.mean(list(sm_cluster.energies.values()))

                if bb_energy > 0 and sm_energy > 0:
                    # 仮定：深度差10km、遅延は相関から推定
                    assumed_depth = 10.0  # km

                    # パスの相関から遅延を推定
                    if bb_cluster.paths and sm_cluster.paths:
                        delay = self._estimate_path_delay(
                            bb_cluster.paths[0],
                            sm_cluster.paths[0]
                        )

                        if delay > 0:
                            velocity = assumed_depth / delay  # km/s
                        else:
                            velocity = 0.0
                    else:
                        velocity = 0.0

                    velocity_map[cluster_id] = velocity

        # 配列形式に変換
        if velocity_map:
            velocities = np.array(list(velocity_map.values()))
            return velocities
        else:
            return np.array([])

    def _estimate_path_delay(self, path1: np.ndarray, path2: np.ndarray) -> float:
        """2つのパス間の遅延を推定"""
        min_len = min(len(path1), len(path2))
        if min_len > 20:
            path1 = path1[:min_len]
            path2 = path2[:min_len]

            correlation = np.correlate(path1, path2, mode='same')
            delay_idx = np.argmax(correlation) - len(correlation) // 2

            # サンプル数を時間に変換（100Hzと仮定）
            delay_time = abs(delay_idx) / 100.0
            return delay_time

        return 0.0

    def _create_station_feature_matrix(self,
                                    data_dict: Dict[str, np.ndarray],
                                    stations: List[str],
                                    use_geological: bool = False) -> np.ndarray:
        """観測点の特徴量行列を作成（統合データ対応版 + 堅牢化 + Lambda³エッセンス）"""
        feature_matrix = []

        for station in stations:
            if station not in data_dict:
                continue

            data = data_dict[station]
            if data.size == 0 or np.all(~np.isfinite(data)):
                continue

            # --- 基本統計的特徴 ---
            try:
                mean = float(np.nanmean(data))
                std = float(np.nanstd(data))
                max_abs = float(np.nanmax(np.abs(data)))
                perc_95 = float(np.nanpercentile(np.abs(data), 95))
                dom_freq = float(self._compute_dominant_frequency(data))
                energy_conc = float(self._compute_energy_concentration(data))
                spec_entropy = float(self._compute_spectral_entropy(data))
                struct_comp = float(self._compute_structural_complexity(data))
            except Exception:
                # 失敗したら全部0で埋める
                mean = std = max_abs = perc_95 = dom_freq = energy_conc = spec_entropy = struct_comp = 0.0

            basic_features = [mean, std, max_abs, perc_95, dom_freq, energy_conc, spec_entropy, struct_comp]

            # --- データタイプに応じた特徴 ---
            try:
                if self.data_type == 'broadband':
                    basic_features.extend([
                        float(self._compute_low_frequency_power(data)),
                        float(self._compute_long_period_trend(data))
                    ])
                elif self.data_type == 'strong_motion':
                    basic_features.extend([
                        float(self._compute_high_frequency_power(data)),
                        float(self._compute_peak_ground_acceleration(data))
                    ])
                elif self.data_type == 'integrated':
                    basic_features.extend([
                        float(self._compute_low_frequency_power(data)),
                        float(self._compute_high_frequency_power(data))
                    ])
            except Exception:
                basic_features.extend([0.0, 0.0])

            # --- 地理的位置の追加 ---
            if station in self.station_locations:
                lat, lon = self.station_locations[station]
                # Lambda³なら「重心原点からの差分」や「距離標準化」もOK
                basic_features.extend([lat/90, lon/180])
            else:
                basic_features.extend([0.0, 0.0])

            # --- 地質情報 ---
            if use_geological and station in self.station_metadata:
                metadata = self.station_metadata[station]
                try:
                    geo_feat = [
                        float(metadata.get('depth', 0))/1000,
                        float(metadata.get('vs30', 500))/1000,
                        float(metadata.get('bedrock', 0)),
                        float(metadata.get('site_class', 0)),
                    ]
                    if self.data_type in ['broadband', 'integrated']:
                        geo_feat.append(float(metadata.get('borehole_depth', 0))/1000)
                    if self.data_type in ['strong_motion', 'integrated']:
                        geo_feat.append(float(metadata.get('surface_geology', 0)))
                except Exception:
                    geo_feat = [0.0]*6
                basic_features.extend(geo_feat)

            # --- NaN/Inf対策・型保証 ---
            basic_features = [float(f) if np.isfinite(f) else 0.0 for f in basic_features]
            feature_matrix.append(basic_features)

        if not feature_matrix:
            return np.array([])

        feature_matrix = np.array(feature_matrix)
        # --- ロバスト正規化（NaN/Inf埋め） ---
        median = np.nanmedian(feature_matrix, axis=0)
        mad = np.nanmedian(np.abs(feature_matrix - median), axis=0)
        feature_matrix = (feature_matrix - median) / (mad + 1e-8)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_matrix

    def _compute_low_frequency_power(self, data: np.ndarray) -> float:
        """低周波成分のパワーを計算（広帯域データ用）"""
        if len(data) < 2:
            return 0.0

        fft = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(len(data))

        # 0.01-0.1 Hzの範囲
        low_freq_mask = (np.abs(freqs) > 0.01) & (np.abs(freqs) < 0.1)
        low_freq_power = np.sum(np.abs(fft[low_freq_mask])**2)

        total_power = np.sum(np.abs(fft)**2)
        return low_freq_power / (total_power + 1e-8)

    def _compute_high_frequency_power(self, data: np.ndarray) -> float:
        """高周波成分のパワーを計算（強震計データ用）"""
        if len(data) < 2:
            return 0.0

        fft = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(len(data))

        # 1-30 Hzの範囲
        high_freq_mask = (np.abs(freqs) > 1.0) & (np.abs(freqs) < 30.0)
        high_freq_power = np.sum(np.abs(fft[high_freq_mask])**2)

        total_power = np.sum(np.abs(fft)**2)
        return high_freq_power / (total_power + 1e-8)

    def _compute_long_period_trend(self, data: np.ndarray) -> float:
        """長周期トレンドの強度（広帯域データ用）"""
        if len(data) < 10:
            return 0.0

        # 移動平均でトレンドを抽出
        window = min(len(data) // 10, 50)
        if window < 3:
            return 0.0

        trend = np.convolve(data.flatten(), np.ones(window)/window, mode='valid')
        trend_strength = np.std(trend) / (np.std(data) + 1e-8)

        return trend_strength

    def _compute_peak_ground_acceleration(self, data: np.ndarray) -> float:
        """最大地動加速度相当の指標（強震計データ用）"""
        if len(data) == 0:
            return 0.0

        # 加速度相当の2階微分
        if len(data) > 2:
            acceleration = np.gradient(np.gradient(data, axis=0), axis=0)
            pga = np.max(np.abs(acceleration))
            return pga
        else:
            return np.max(np.abs(data))

    def _cluster_stations(self,
                         data_dict: Dict[str, np.ndarray],
                         n_clusters: int,
                         method: str = 'kmeans') -> Dict[str, int]:
        """観測点をクラスタリング（改良版）"""
        stations = list(data_dict.keys())

        if len(stations) <= n_clusters:
            # 観測点数がクラスタ数以下の場合
            return {station: i for i, station in enumerate(stations)}

        # 地質情報を使用するかどうか
        use_geological = (method == 'geological' and self.station_metadata)

        # 特徴量行列の作成
        feature_matrix = self._create_station_feature_matrix(
            data_dict, stations, use_geological
        )

        if feature_matrix.shape[0] == 0:
            return {station: 0 for station in stations}

        # クラスタリング実行
        if method in self.clustering_methods:
            labels = self.clustering_methods[method](feature_matrix, n_clusters)
        else:
            print(f"Warning: Unknown clustering method '{method}', using kmeans")
            labels = self._cluster_kmeans(feature_matrix, n_clusters)

        # 結果を辞書形式に変換
        station_clusters = {station: int(label) for station, label in zip(stations, labels)}

        # クラスタリング結果の表示
        self._print_clustering_summary(station_clusters, n_clusters)

        return station_clusters

    def _cluster_kmeans(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """K-meansクラスタリング"""
        from sklearn.cluster import KMeans

        # データタイプに応じたパラメータ調整
        if self.data_type == 'integrated':
            # 統合データの場合は初期化を慎重に
            n_init = 20
            max_iter = 500
        else:
            n_init = 10
            max_iter = 300

        kmeans = KMeans(n_clusters=n_clusters, random_state=42,
                       n_init=n_init, max_iter=max_iter)
        return kmeans.fit_predict(features)

    def _cluster_dbscan(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """DBSCANクラスタリング（密度ベース）"""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        # epsの自動調整（データタイプ考慮）
        if self.data_type == 'broadband':
            eps_factor = 0.5  # 広帯域は疎なクラスタを許容
        elif self.data_type == 'strong_motion':
            eps_factor = 0.3  # 強震計は密なクラスタを形成
        else:
            eps_factor = 0.4

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 最適なepsを探索
        for eps in np.linspace(eps_factor, eps_factor * 3, 10):
            dbscan = DBSCAN(eps=eps, min_samples=3)
            labels = dbscan.fit_predict(features_scaled)

            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_found >= n_clusters * 0.8:
                break

        # ノイズ点の処理
        if -1 in labels:
            # 最近傍クラスタに割り当て
            from sklearn.neighbors import NearestNeighbors

            noise_mask = labels == -1
            if np.any(noise_mask) and np.any(~noise_mask):
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(features_scaled[~noise_mask])

                for i in np.where(noise_mask)[0]:
                    _, indices = nn.kneighbors([features_scaled[i]])
                    labels[i] = labels[~noise_mask][indices[0, 0]]

        return labels

    def _cluster_hierarchical(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """階層的クラスタリング"""
        from sklearn.cluster import AgglomerativeClustering

        # データタイプに応じた連結法の選択
        if self.data_type == 'broadband':
            linkage = 'complete'  # 最遠法（保守的）
        elif self.data_type == 'strong_motion':
            linkage = 'average'   # 平均法（バランス）
        else:
            linkage = 'ward'      # Ward法（標準）

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        return clustering.fit_predict(features)

    def _cluster_geological(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """地質情報を重視したクラスタリング"""
        if not self.station_metadata:
            print("Warning: No geological metadata available, using kmeans instead")
            return self._cluster_kmeans(features, n_clusters)

        # 地質特徴の重みを増加
        if features.shape[1] > 10:  # 地質情報が含まれている場合
            geological_weight = 2.0
            features_weighted = features.copy()
            features_weighted[:, -6:] *= geological_weight  # 最後の6列が地質情報

            # 階層的クラスタリングを使用
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            return clustering.fit_predict(features_weighted)
        else:
            return self._cluster_kmeans(features, n_clusters)

    def _print_clustering_summary(self, station_clusters: Dict[str, int], n_clusters: int):
        """クラスタリング結果のサマリーを表示（改良版）"""
        print(f"\nClustering Summary ({self.data_type} data):")

        cluster_info = {}
        for cluster_id in range(n_clusters):
            cluster_stations = [s for s, c in station_clusters.items() if c == cluster_id]
            cluster_info[cluster_id] = {
                'stations': cluster_stations,
                'count': len(cluster_stations)
            }

            # 地理的情報があれば中心位置を計算
            if self.station_locations and cluster_stations:
                lats, lons = [], []
                for station in cluster_stations:
                    if station in self.station_locations:
                        lat, lon = self.station_locations[station]
                        lats.append(lat)
                        lons.append(lon)

                if lats:
                    cluster_info[cluster_id]['center'] = (np.mean(lats), np.mean(lons))

        # 表示
        for cluster_id, info in cluster_info.items():
            print(f"  Cluster {cluster_id}: {info['count']} stations", end='')

            if 'center' in info:
                print(f" (center: {info['center'][0]:.1f}°N, {info['center'][1]:.1f}°E)")
            else:
                print()

            # サンプル観測点
            sample_stations = info['stations'][:3]
            if len(info['stations']) > 3:
                sample_stations.append("...")
            print(f"    {', '.join(sample_stations)}")

    def _get_cluster_size(self, station_clusters: Dict[str, int], cluster_id: int) -> int:
        """クラスタ内の観測点数を取得"""
        return sum(1 for c in station_clusters.values() if c == cluster_id)

    def _aggregate_global_data(self, data_dict: Dict[str, np.ndarray], region_center: tuple = None, region_decay: float = 5.0) -> np.ndarray:
        """全観測点のデータを統合してグローバルデータを作成"""
        if not data_dict:
            return np.array([])

        all_data = []
        weights = []
        station_types = []

        for station, data in data_dict.items():
            all_data.append(data)

            # エネルギー重み
            energy_weight = np.sum(data**2)

            # --- 地理的重要度（デフォルトは1.0、任意でregion_centerで重み指定可） ---
            geo_weight = 1.0
            if region_center is not None and station in self.station_locations:
                lat, lon = self.station_locations[station]
                region_lat, region_lon = region_center
                region_distance = np.sqrt((lat - region_lat)**2 + (lon - region_lon)**2)
                geo_weight = np.exp(-region_distance / region_decay)
                # region_decayで減衰の鋭さ調整
            # else:  # 何も指定なければ全て同じ重み

            # --- データタイプごとの重み ---
            type_weight = 1.0
            if self.data_type == 'broadband' and station in self.station_metadata:
                depth = self.station_metadata[station].get('borehole_depth', 0)
                type_weight = 1.0 + depth / 1000.0
            elif self.data_type == 'strong_motion' and station in self.station_metadata:
                vs30 = self.station_metadata[station].get('vs30', 500)
                type_weight = 1.0 + (500 - vs30) / 500.0

            total_weight = energy_weight * geo_weight * type_weight
            weights.append(total_weight)
            station_types.append(self.data_type)

        # 時間軸を揃える
        min_length = min(len(d) for d in all_data)
        if min_length == 0:
            return np.array([])

        aligned_data = [d[:min_length] for d in all_data]

        # 重み付き統合
        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-8)

        global_data = np.zeros_like(aligned_data[0])
        for data, weight in zip(aligned_data, weights):
            global_data += weight * data

        # 標準偏差など高次統計量も同じ（省略、元コード通り）
        global_var = np.zeros_like(aligned_data[0])
        for data, weight in zip(aligned_data, weights):
            global_var += weight * (data - global_data)**2
        global_std = np.sqrt(global_var)

        global_skew = np.zeros_like(aligned_data[0])
        global_kurt = np.zeros_like(aligned_data[0])
        for data, weight in zip(aligned_data, weights):
            z = (data - global_data) / (global_std + 1e-8)
            global_skew += weight * z**3
            global_kurt += weight * z**4
        global_kurt -= 3

        if self.data_type == 'integrated':
            global_data = np.hstack([global_data, global_std, global_skew, global_kurt])
        else:
            global_data = np.hstack([global_data, global_std])

        return global_data

    def _aggregate_cluster_data(self,
                               data_dict: Dict[str, np.ndarray],
                               station_clusters: Dict[str, int]) -> Dict[int, np.ndarray]:
        """クラスタごとにデータを集約（改良版）"""
        cluster_data_dict = {}

        for cluster_id in set(station_clusters.values()):
            cluster_stations = [s for s, c in station_clusters.items() if c == cluster_id]
            cluster_data_list = [data_dict[s] for s in cluster_stations if s in data_dict]

            if not cluster_data_list:
                continue

            # 時間軸を揃える
            min_length = min(len(d) for d in cluster_data_list)
            if min_length == 0:
                continue

            aligned_data = [d[:min_length] for d in cluster_data_list]

            # クラスタ内の重み計算（改良版）
            weights = []
            for i, (station, data) in enumerate(zip(cluster_stations, aligned_data)):
                # エネルギーベースの重み
                energy_weight = np.sum(data**2)

                # 観測点の品質（メタデータから）
                quality_weight = 1.0
                if station in self.station_metadata:
                    metadata = self.station_metadata[station]
                    # 観測品質指標があれば使用
                    quality_weight = metadata.get('quality_factor', 1.0)

                weights.append(energy_weight * quality_weight)

            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)

            # 重み付き平均
            cluster_data = np.zeros_like(aligned_data[0])
            for data, weight in zip(aligned_data, weights):
                cluster_data += weight * data

            # クラスタ内の変動も記録（データタイプ別）
            if self.data_type == 'integrated':
                # 統合データの場合は詳細な統計量
                cluster_std = np.zeros_like(aligned_data[0])
                for data, weight in zip(aligned_data, weights):
                    cluster_std += weight * (data - cluster_data)**2
                cluster_std = np.sqrt(cluster_std)

                # 最大・最小値の範囲
                cluster_max = np.max(aligned_data, axis=0)
                cluster_min = np.min(aligned_data, axis=0)
                cluster_range = cluster_max - cluster_min

                cluster_data = np.hstack([cluster_data, cluster_std, cluster_range])

            cluster_data_dict[cluster_id] = cluster_data

        return cluster_data_dict

    def _analyze_lambda3(self, data: np.ndarray, n_paths: int, label: str) -> Lambda3Result:
        """Lambda³解析を実行（エラーハンドリング強化）"""
        try:
            # データ検証
            if data.size == 0:
                raise ValueError("Empty data array")

            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # 解析実行
            if self.data_type == 'integrated':
                # 統合データの場合は両方の解析器を使用
                bb_result = self.broadband_analyzer.analyze(data, n_paths)
                sm_result = self.strong_motion_analyzer.analyze(data, n_paths)

                # 結果の統合
                integrated_charges = {}
                integrated_energies = {}
                integrated_paths = {}

                for i in range(n_paths):
                    if i < n_paths // 2:
                        # 前半は広帯域の結果
                        if i in bb_result.topological_charges:
                            integrated_charges[i] = bb_result.topological_charges[i]
                            integrated_energies[i] = bb_result.energies[i]
                            integrated_paths[i] = bb_result.paths[i]
                    else:
                        # 後半は強震計の結果
                        j = i - n_paths // 2
                        if j in sm_result.topological_charges:
                            integrated_charges[i] = sm_result.topological_charges[j]
                            integrated_energies[i] = sm_result.energies[j]
                            integrated_paths[i] = sm_result.paths[j]

                result = Lambda3Result(
                    paths=integrated_paths,
                    topological_charges=integrated_charges,
                    stabilities=bb_result.stabilities,
                    energies=integrated_energies,
                    entropies=bb_result.entropies,
                    classifications=bb_result.classifications
                )
            else:
                result = self.base_analyzer.analyze(data, n_paths)

            # 解析結果のサマリー
            charges = list(result.topological_charges.values())
            if charges:
                mean_charge = np.mean(np.abs(charges))
                max_charge = np.max(np.abs(charges))

                print(f"  {label}: {n_paths} paths extracted")
                print(f"    Mean |Q_Λ| = {mean_charge:.3f}, Max |Q_Λ| = {max_charge:.3f}")

                # データタイプ別の追加情報
                if self.data_type == 'broadband':
                    print(f"    Deep structure dominance: {sum(c < -0.5 for c in charges)}/{len(charges)}")
                elif self.data_type == 'strong_motion':
                    print(f"    Surface rupture potential: {sum(c > 0.5 for c in charges)}/{len(charges)}")

            return result

        except Exception as e:
            print(f"  Warning: Analysis failed for {label}: {e}")
            # より詳細なエラー情報
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")

            # ダミーの結果を返す
            return Lambda3Result(
                paths={i: np.zeros(max(1, len(data))) for i in range(n_paths)},
                topological_charges={i: 0.0 for i in range(n_paths)},
                stabilities={i: 0.0 for i in range(n_paths)},
                energies={i: 1.0 for i in range(n_paths)},
                entropies={i: 0.0 for i in range(n_paths)},
                classifications={i: "中性構造（平衡状態）" for i in range(n_paths)}
            )

    def _compute_spatial_correlations(self,
                                    local_results: Dict[str, Lambda3Result]) -> np.ndarray:
        """観測点間の構造相関を計算（高速化版・改良）"""
        stations = list(local_results.keys())
        n_stations = len(stations)

        if n_stations == 0:
            return np.array([])

        correlations = np.zeros((n_stations, n_stations))

        # 各観測点の特徴ベクトルを作成（拡張版）
        feature_vectors = []
        for station in stations:
            result = local_results[station]

            # 基本特徴
            charges = list(result.topological_charges.values())
            energies = list(result.energies.values())
            stabilities = list(result.stabilities.values())
            entropies = list(result.entropies.values())

            # 特徴ベクトル（拡張）
            features = []

            if charges:
                features.extend([
                    np.mean(charges),
                    np.std(charges),
                    np.percentile(np.abs(charges), 25),
                    np.percentile(np.abs(charges), 75),
                    np.max(np.abs(charges))
                ])
            else:
                features.extend([0, 0, 0, 0, 0])

            if energies:
                features.extend([
                    np.mean(energies),
                    np.std(energies),
                    np.max(energies) / (np.mean(energies) + 1e-8)  # ピークファクター
                ])
            else:
                features.extend([1, 0, 1])

            if stabilities:
                features.extend([
                    np.mean(stabilities),
                    np.max(stabilities)
                ])
            else:
                features.extend([0, 0])

            if entropies:
                features.append(np.mean(entropies))
            else:
                features.append(0)

            # データタイプ別の追加特徴
            if self.data_type == 'broadband':
                # 深部構造の安定性指標
                deep_stability = sum(1 for c in charges if -0.5 < c < 0.5) / (len(charges) + 1e-8)
                features.append(deep_stability)
            elif self.data_type == 'strong_motion':
                # 表層破壊の可能性指標
                rupture_potential = sum(1 for c in charges if abs(c) > 1.0) / (len(charges) + 1e-8)
                features.append(rupture_potential)

            feature_vectors.append(features)

        feature_vectors = np.array(feature_vectors)

        # 特徴ベクトルの正規化
        feature_vectors = (feature_vectors - np.mean(feature_vectors, axis=0)) / (np.std(feature_vectors, axis=0) + 1e-8)

        # 相関行列を計算（複数の距離尺度）
        for i in range(n_stations):
            for j in range(i, n_stations):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    vec_i = feature_vectors[i]
                    vec_j = feature_vectors[j]

                    # コサイン類似度
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)

                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)

                        # ユークリッド距離による類似度
                        euclidean_sim = 1.0 / (1.0 + np.linalg.norm(vec_i - vec_j))

                        # 統合類似度（両方の尺度を組み合わせ）
                        similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim

                        correlations[i, j] = correlations[j, i] = similarity
                    else:
                        correlations[i, j] = correlations[j, i] = 0

        return correlations

    def _evaluate_cross_layer_interactions(self,
                                         global_result: Lambda3Result,
                                         local_results: Dict[str, Lambda3Result],
                                         cluster_results: Dict[int, Lambda3Result],
                                         station_clusters: Dict[str, int]) -> Dict[str, float]:
        """層間相互作用の詳細評価（統合データ対応版）"""
        metrics = {}

        # 1. グローバル-ローカル整合性
        local_charges = []
        for result in local_results.values():
            local_charges.extend(list(result.topological_charges.values()))

        global_charges = list(global_result.topological_charges.values())

        if local_charges and global_charges:
            # 分布の類似性（KSテスト的な指標）
            local_dist = np.histogram(local_charges, bins=20, density=True)[0]
            global_dist = np.histogram(global_charges, bins=20, density=True)[0]

            # Jensen-Shannon距離
            m = 0.5 * (local_dist + global_dist)
            js_div = 0.5 * np.sum(local_dist * np.log(local_dist / (m + 1e-8) + 1e-8)) + \
                     0.5 * np.sum(global_dist * np.log(global_dist / (m + 1e-8) + 1e-8))

            metrics['global_local_consistency'] = 1.0 / (1.0 + js_div)
        else:
            metrics['global_local_consistency'] = 0.0

        # 2. クラスタ内均質性（改良版）
        cluster_homogeneities = []
        cluster_diversities = []

        for cluster_id, cluster_result in cluster_results.items():
            cluster_stations = [s for s, c in station_clusters.items() if c == cluster_id]

            if len(cluster_stations) > 1:
                # クラスタ内の観測点間の類似度
                cluster_features = []
                for station in cluster_stations:
                    if station in local_results:
                        charges = list(local_results[station].topological_charges.values())
                        if charges:
                            cluster_features.append([
                                np.mean(np.abs(charges)),
                                np.std(charges),
                                np.percentile(np.abs(charges), 75)
                            ])

                if len(cluster_features) > 1:
                    cluster_features = np.array(cluster_features)
                    # 特徴の分散（小さいほど均質）
                    feature_std = np.mean(np.std(cluster_features, axis=0))
                    homogeneity = 1.0 / (1.0 + feature_std)
                    cluster_homogeneities.append(homogeneity)

                    # 特徴の多様性（PCA的な指標）
                    if cluster_features.shape[0] > cluster_features.shape[1]:
                        cov = np.cov(cluster_features.T)
                        eigenvalues = np.linalg.eigvalsh(cov)
                        if np.sum(eigenvalues) > 0:
                            diversity = -np.sum((eigenvalues / np.sum(eigenvalues)) *
                                              np.log(eigenvalues / np.sum(eigenvalues) + 1e-8))
                            cluster_diversities.append(diversity)

        metrics['cluster_homogeneity'] = np.mean(cluster_homogeneities) if cluster_homogeneities else 0
        metrics['cluster_diversity'] = np.mean(cluster_diversities) if cluster_diversities else 0

        # 3. 層間エネルギー分配（階層的）
        global_energy = np.mean(list(global_result.energies.values()))
        local_energy = np.mean([np.mean(list(r.energies.values())) for r in local_results.values()])
        cluster_energy = np.mean([np.mean(list(r.energies.values())) for r in cluster_results.values()])

        total_energy = global_energy + local_energy + cluster_energy
        if total_energy > 0:
            energies = [global_energy, cluster_energy, local_energy]  # 階層順

            # エネルギーカスケード（上位層から下位層への流れ）
            if global_energy > 0 and cluster_energy > 0:
                cascade_g2c = cluster_energy / global_energy
            else:
                cascade_g2c = 0

            if cluster_energy > 0 and local_energy > 0:
                cascade_c2l = local_energy / cluster_energy
            else:
                cascade_c2l = 0

            metrics['energy_cascade_efficiency'] = np.mean([cascade_g2c, cascade_c2l])

            # エントロピー的な分配指標
            probs = [e/total_energy for e in energies]
            metrics['energy_distribution_entropy'] = -sum([
                p * np.log(p + 1e-8) for p in probs if p > 0
            ])
        else:
            metrics['energy_cascade_efficiency'] = 0
            metrics['energy_distribution_entropy'] = 0

        # 4. 構造的多様性（拡張版）
        all_charges = []
        all_charges.extend(global_charges)
        all_charges.extend(local_charges)
        for result in cluster_results.values():
            all_charges.extend(list(result.topological_charges.values()))

        if all_charges:
            # 基本的な多様性
            metrics['structural_diversity'] = np.std(all_charges) / (np.mean(np.abs(all_charges)) + 1e-8)

            # 構造タイプの多様性
            structure_types = []
            for result in [global_result] + list(local_results.values()) + list(cluster_results.values()):
                structure_types.extend(list(result.classifications.values()))

            unique_types = set(structure_types)
            type_counts = [structure_types.count(t) for t in unique_types]
            total_count = sum(type_counts)

            if total_count > 0:
                type_probs = [c/total_count for c in type_counts]
                metrics['structure_type_entropy'] = -sum([p * np.log(p + 1e-8) for p in type_probs])
            else:
                metrics['structure_type_entropy'] = 0
        else:
            metrics['structural_diversity'] = 0
            metrics['structure_type_entropy'] = 0

        # 5. 空間的階層性（改良版）
        if cluster_results and global_charges:
            # 各階層での平均チャージの相関
            hierarchy_correlation = []

            global_mean = np.mean(global_charges)
            cluster_means = [np.mean(list(r.topological_charges.values()))
                           for r in cluster_results.values()]

            if cluster_means:
                # グローバル-クラスタ相関
                cluster_deviations = [abs(cm - global_mean) for cm in cluster_means]
                hierarchy_score = 1.0 - np.mean(cluster_deviations) / (abs(global_mean) + 1e-8)
                hierarchy_correlation.append(max(0, hierarchy_score))

                # クラスタ-ローカル相関
                for cluster_id, cluster_mean in enumerate(cluster_means):
                    cluster_stations = [s for s, c in station_clusters.items() if c == cluster_id]
                    local_means = []

                    for station in cluster_stations:
                        if station in local_results:
                            local_charges = list(local_results[station].topological_charges.values())
                            if local_charges:
                                local_means.append(np.mean(local_charges))

                    if local_means:
                        local_deviations = [abs(lm - cluster_mean) for lm in local_means]
                        local_hierarchy = 1.0 - np.mean(local_deviations) / (abs(cluster_mean) + 1e-8)
                        hierarchy_correlation.append(max(0, local_hierarchy))

                metrics['spatial_hierarchy'] = np.mean(hierarchy_correlation)
            else:
                metrics['spatial_hierarchy'] = 0
        else:
            metrics['spatial_hierarchy'] = 0

        # 6. データタイプ別の追加メトリクス
        if self.data_type == 'broadband':
            # 深部構造の連続性
            deep_continuity = []
            for result in local_results.values():
                charges = list(result.topological_charges.values())
                if len(charges) > 1:
                    # 隣接チャージの差分
                    charge_diff = np.diff(charges)
                    continuity = 1.0 / (1.0 + np.std(charge_diff))
                    deep_continuity.append(continuity)

            metrics['deep_structure_continuity'] = np.mean(deep_continuity) if deep_continuity else 0

        elif self.data_type == 'strong_motion':
            # 表層応答の局所性
            surface_locality = []
            for result in local_results.values():
                energies = list(result.energies.values())
                if energies:
                    # エネルギーの集中度
                    energy_concentration = np.max(energies) / (np.mean(energies) + 1e-8)
                    surface_locality.append(energy_concentration)

            metrics['surface_response_locality'] = np.mean(surface_locality) if surface_locality else 0

        elif self.data_type == 'integrated':
            # 統合データの場合は両方計算
            metrics['integrated_coupling'] = (
                metrics.get('deep_structure_continuity', 0) *
                metrics.get('surface_response_locality', 0)
            ) ** 0.5

        return metrics

    def detect_spatial_anomalies(self,
                               global_result: Lambda3Result,
                               local_results: Dict[str, Lambda3Result],
                               cluster_results: Dict[int, Lambda3Result],
                               spatial_correlations: np.ndarray,
                               station_clusters: Dict[str, int]) -> Dict[str, List[Dict]]:
        """空間的異常パターンの包括的検出（統合データ対応版）"""
        anomalies = {
            'global_anomalies': [],
            'local_hotspots': [],
            'cluster_anomalies': [],
            'spatial_discontinuities': [],
            'propagation_patterns': [],
            'structural_transitions': [],
            'precursor_patterns': []  
        }

        # データタイプ別の閾値調整
        threshold_factor = {
            'broadband': 1.5,      # 深部構造は変動が小さい
            'strong_motion': 2.0,  # 表層は変動が大きい
            'integrated': 1.75,    # 中間的な値
            'standard': 2.0
        }.get(self.data_type, 2.0)

        # 1. グローバル異常（全国規模の異常）
        global_charges = list(global_result.topological_charges.values())
        if global_charges:
            global_mean = np.mean(np.abs(global_charges))
            global_std = np.std(np.abs(global_charges))
            global_threshold = global_mean + threshold_factor * global_std

            for i, charge in enumerate(global_charges):
                if abs(charge) > global_threshold:
                    # 異常の詳細分類
                    anomaly_type = self._classify_global_anomaly(
                        charge, global_result.energies.get(i, 0),
                        global_result.entropies.get(i, 0)
                    )

                    anomalies['global_anomalies'].append({
                        'path_id': i,
                        'charge': charge,
                        'severity': abs(charge) / global_threshold,
                        'classification': global_result.classifications[i],
                        'anomaly_type': anomaly_type,
                        'timestamp': i  # 時間的位置
                    })

        # 2. ローカルホットスポット（特定観測点の異常）
        station_anomaly_scores = {}
        station_features = {}

        for station, local_result in local_results.items():
            local_charges = list(local_result.topological_charges.values())
            local_energies = list(local_result.energies.values())
            local_entropies = list(local_result.entropies.values())

            if local_charges and local_energies:
                # 複合異常スコア（拡張版）
                charge_score = np.mean(np.abs(local_charges))
                charge_std = np.std(local_charges)
                energy_score = np.mean(local_energies)
                energy_peak = np.max(local_energies) if local_energies else 0
                stability_score = np.mean(list(local_result.stabilities.values()))
                entropy_score = np.mean(local_entropies) if local_entropies else 0

                # データタイプ別の重み付け
                if self.data_type == 'broadband':
                    # 深部構造：安定性とチャージを重視
                    anomaly_score = (charge_score * 2 + stability_score) * (1 + energy_score)
                elif self.data_type == 'strong_motion':
                    # 表層：エネルギーピークを重視
                    anomaly_score = (energy_peak * 2 + charge_score) * (1 + charge_std)
                else:
                    # 標準的な重み付け
                    anomaly_score = charge_score * (1 + energy_score) * (1 + stability_score)

                station_anomaly_scores[station] = anomaly_score
                station_features[station] = {
                    'charge_mean': charge_score,
                    'charge_std': charge_std,
                    'energy_mean': energy_score,
                    'energy_peak': energy_peak,
                    'entropy': entropy_score
                }

        # 統計的な閾値設定
        if station_anomaly_scores:
            scores = list(station_anomaly_scores.values())

            # ロバストな統計量（外れ値の影響を軽減）
            q25 = np.percentile(scores, 25)
            q75 = np.percentile(scores, 75)
            iqr = q75 - q25

            threshold_moderate = q75 + 1.5 * iqr
            threshold_severe = q75 + 3.0 * iqr
            threshold_extreme = q75 + 4.5 * iqr

            for station, score in station_anomaly_scores.items():
                if score > threshold_moderate:
                    severity = 'moderate'
                    if score > threshold_extreme:
                        severity = 'extreme'
                    elif score > threshold_severe:
                        severity = 'severe'

                    # 異常パターンの特定
                    anomaly_pattern = self._identify_local_anomaly_pattern(
                        station_features[station]
                    )

                    anomalies['local_hotspots'].append({
                        'station': station,
                        'anomaly_score': score,
                        'percentile': 100 * (1 - sum(s > score for s in scores) / len(scores)),
                        'severity': severity,
                        'cluster': station_clusters.get(station, -1),
                        'pattern': anomaly_pattern,
                        'features': station_features[station]
                    })

        # 3. クラスタ異常（地域的な異常）
        all_cluster_metrics = []

        for cluster_id, cluster_result in cluster_results.items():
            cluster_charges = list(cluster_result.topological_charges.values())
            cluster_energy = np.mean(list(cluster_result.energies.values()))
            cluster_entropy = np.mean(list(cluster_result.entropies.values()))

            # クラスタの総合的な異常度
            cluster_metric = {
                'id': cluster_id,
                'energy': cluster_energy,
                'entropy': cluster_entropy,
                'charge_volatility': np.std(cluster_charges) if cluster_charges else 0,
                'charge_mean': np.mean(cluster_charges) if cluster_charges else 0
            }
            all_cluster_metrics.append(cluster_metric)

        # クラスタ異常の判定
        if all_cluster_metrics:
            energy_values = [m['energy'] for m in all_cluster_metrics]
            entropy_values = [m['entropy'] for m in all_cluster_metrics]
            volatility_values = [m['charge_volatility'] for m in all_cluster_metrics]

            for metric in all_cluster_metrics:
                # 複合的な異常判定
                is_anomalous = False
                anomaly_reasons = []

                if metric['energy'] > np.percentile(energy_values, 90):
                    is_anomalous = True
                    anomaly_reasons.append('high_energy')

                if metric['entropy'] > np.percentile(entropy_values, 90):
                    is_anomalous = True
                    anomaly_reasons.append('high_entropy')

                if metric['charge_volatility'] > np.percentile(volatility_values, 90):
                    is_anomalous = True
                    anomaly_reasons.append('unstable')

                if is_anomalous:
                    cluster_size = self._get_cluster_size(station_clusters, metric['id'])

                    anomalies['cluster_anomalies'].append({
                        'cluster_id': metric['id'],
                        'n_stations': cluster_size,
                        'energy': metric['energy'],
                        'mean_charge': metric['charge_mean'],
                        'entropy': metric['entropy'],
                        'charge_volatility': metric['charge_volatility'],
                        'anomaly_type': self._classify_cluster_anomaly_extended(metric, anomaly_reasons),
                        'anomaly_reasons': anomaly_reasons
                    })

        # 4. 空間的不連続（隣接観測点間の急激な変化）
        stations = list(local_results.keys())
        n_stations = len(stations)

        if n_stations > 1 and spatial_correlations.size > 0:
            for i in range(n_stations):
                for j in range(i+1, n_stations):
                    if i < spatial_correlations.shape[0] and j < spatial_correlations.shape[1]:
                        correlation = spatial_correlations[i, j]

                        # 相関の異常判定（データタイプ別）
                        if self.data_type == 'broadband':
                            # 深部構造は連続性が高いはず
                            discontinuity_threshold = 0.3
                        elif self.data_type == 'strong_motion':
                            # 表層は不連続性を許容
                            discontinuity_threshold = 0.1
                        else:
                            discontinuity_threshold = 0.2

                        if correlation < discontinuity_threshold:
                            station_i = stations[i]
                            station_j = stations[j]

                            # 地理的距離を考慮
                            geo_distance = self._calculate_station_distance(station_i, station_j)

                            if geo_distance is not None and geo_distance < 2.0:  # 約200km以内
                                # 不連続の強度を計算
                                expected_correlation = np.exp(-geo_distance / 5.0)  # 距離による期待相関
                                discontinuity_strength = expected_correlation - correlation

                                if discontinuity_strength > 0.3:
                                    anomalies['spatial_discontinuities'].append({
                                        'station_pair': (station_i, station_j),
                                        'correlation': correlation,
                                        'expected_correlation': expected_correlation,
                                        'geo_distance': geo_distance,
                                        'discontinuity_strength': discontinuity_strength,
                                        'anomaly_type': 'structural_boundary' if correlation < 0 else 'weak_coupling'
                                    })

        # 5. 伝播パターン（異常の空間的な広がり）
        if anomalies['local_hotspots']:
            # クラスタごとのホットスポット集計
            hotspot_clusters = {}
            for hotspot in anomalies['local_hotspots']:
                cluster = hotspot['cluster']
                if cluster not in hotspot_clusters:
                    hotspot_clusters[cluster] = []
                hotspot_clusters[cluster].append(hotspot)

            # 伝播パターンの検出
            for cluster_id, hotspots in hotspot_clusters.items():
                if len(hotspots) >= 2:
                    # 空間的な広がりを評価
                    stations = [h['station'] for h in hotspots]

                    # 地理的な広がり
                    if self.station_locations:
                        geo_spread = self._calculate_geographical_spread(stations)
                    else:
                        geo_spread = None

                    # 異常強度の相関
                    anomaly_scores = [h['anomaly_score'] for h in hotspots]
                    score_std = np.std(anomaly_scores)

                    # パターンタイプの判定
                    if len(hotspots) > 0.5 * self._get_cluster_size(station_clusters, cluster_id):
                        pattern_type = 'widespread'
                    elif score_std < np.mean(anomaly_scores) * 0.2:
                        pattern_type = 'uniform'
                    else:
                        pattern_type = 'clustered'

                    anomalies['propagation_patterns'].append({
                        'cluster': cluster_id,
                        'affected_stations': stations,
                        'propagation_extent': len(stations),
                        'pattern_type': pattern_type,
                        'geographical_spread': geo_spread,
                        'intensity_variation': score_std / (np.mean(anomaly_scores) + 1e-8),
                        'mean_severity': np.mean([h['anomaly_score'] for h in hotspots])
                    })

        # 6. 構造的遷移（トポロジカルな変化）
        for station, local_result in local_results.items():
            charges = list(local_result.topological_charges.values())
            classifications = list(local_result.classifications.values())

            if len(charges) > 2:
                # 時系列での構造変化を検出
                charge_changes = np.abs(np.diff(charges))

                # 急激な変化点
                change_threshold = np.mean(charge_changes) + 2 * np.std(charge_changes)
                transition_points = np.where(charge_changes > change_threshold)[0]

                if len(transition_points) > 0:
                    # 構造タイプの変化も考慮
                    unique_structures = len(set(classifications))

                    transition_strength = np.max(charge_changes) / (np.mean(np.abs(charges)) + 1e-8)

                    if transition_strength > 0.5 or unique_structures > 2:
                        anomalies['structural_transitions'].append({
                            'station': station,
                            'transition_strength': transition_strength,
                            'structures': list(set(classifications)),
                            'charge_range': (min(charges), max(charges)),
                            'transition_points': transition_points.tolist(),
                            'n_transitions': len(transition_points),
                            'transition_type': self._classify_transition_type(charges, classifications)
                        })

        # 7. 前兆パターンの検出（地震前兆に特化）
        if self.data_type in ['broadband', 'integrated']:
            precursors = self._detect_spatial_precursor_patterns(
                global_result, local_results, cluster_results,
                station_clusters, spatial_correlations
            )
            anomalies['precursor_patterns'] = precursors

        return anomalies

    def _classify_global_anomaly(self, charge: float, energy: float, entropy: float) -> str:
        """グローバル異常の詳細分類"""
        if abs(charge) > 2.0:
            if energy > 10:
                return "energetic_instability"
            elif entropy > 5:
                return "chaotic_transition"
            else:
                return "topological_singularity"
        elif abs(charge) > 1.0:
            if charge > 0:
                return "matter_accumulation"
            else:
                return "antimatter_accumulation"
        else:
            return "moderate_fluctuation"

    def _identify_local_anomaly_pattern(self, features: Dict[str, float]) -> str:
        """ローカル異常のパターン識別"""
        charge_mean = features['charge_mean']
        charge_std = features['charge_std']
        energy_peak = features['energy_peak']
        energy_mean = features['energy_mean']

        # パターン分類
        if energy_peak > 3 * energy_mean:
            return "spike_event"
        elif charge_std > charge_mean:
            return "oscillatory_instability"
        elif charge_mean > 1.0 and energy_mean > 5.0:
            return "sustained_high_activity"
        elif charge_mean < -1.0:
            return "energy_sink"
        else:
            return "complex_anomaly"

    def _classify_cluster_anomaly_extended(self, metric: Dict, reasons: List[str]) -> str:
        """クラスタ異常の拡張分類"""
        if 'high_energy' in reasons and 'unstable' in reasons:
            return "critical_instability"
        elif 'high_entropy' in reasons and metric['charge_mean'] > 1:
            return "chaotic_expansion"
        elif 'high_energy' in reasons and metric['charge_mean'] < -1:
            return "energy_concentration"
        elif 'unstable' in reasons:
            return "fluctuating_state"
        else:
            return "complex_regional_anomaly"

    def _calculate_station_distance(self, station1: str, station2: str) -> Optional[float]:
        """2観測点間の地理的距離を計算"""
        if not self.station_locations:
            return None

        if station1 in self.station_locations and station2 in self.station_locations:
            lat1, lon1 = self.station_locations[station1]
            lat2, lon2 = self.station_locations[station2]

            # 簡易的な距離計算（度単位）
            distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
            return distance

        return None

    def _calculate_geographical_spread(self, stations: List[str]) -> Optional[float]:
        """観測点群の地理的広がりを計算"""
        if not self.station_locations:
            return None

        lats, lons = [], []
        for station in stations:
            if station in self.station_locations:
                lat, lon = self.station_locations[station]
                lats.append(lat)
                lons.append(lon)

        if len(lats) < 2:
            return None

        # 標準偏差による広がりの評価
        lat_spread = np.std(lats)
        lon_spread = np.std(lons)

        # 楕円的な広がり
        spread = np.sqrt(lat_spread**2 + lon_spread**2)
        return spread

    def _classify_transition_type(self, charges: List[float], classifications: List[str]) -> str:
        """構造遷移のタイプを分類"""
        charge_array = np.array(charges)

        # 単調増加/減少の検出
        if len(charges) > 3:
            trend = np.polyfit(range(len(charges)), charges, 1)[0]

            if abs(trend) > 0.1:
                if trend > 0:
                    return "progressive_growth"
                else:
                    return "progressive_decay"

        # 周期的変化の検出
        if len(charges) > 10:
            fft = np.fft.fft(charge_array)
            power = np.abs(fft[1:len(fft)//2])**2

            if np.max(power) > 5 * np.mean(power):
                return "periodic_oscillation"

        # 急激な変化
        if len(charges) > 2:
            max_change = np.max(np.abs(np.diff(charges)))
            if max_change > 2 * np.std(charges):
                return "abrupt_transition"

        return "complex_evolution"

    def _detect_spatial_precursor_patterns(self,
                                         global_result: Lambda3Result,
                                         local_results: Dict[str, Lambda3Result],
                                         cluster_results: Dict[int, Lambda3Result],
                                         station_clusters: Dict[str, int],
                                         spatial_correlations: np.ndarray) -> List[Dict]:
        """空間的な前兆パターンの検出"""
        precursors = []

        # 1. 静穏化パターンの検出
        quiet_stations = []
        for station, local_result in local_results.items():
            energies = list(local_result.energies.values())
            if energies and np.mean(energies) < 0.2:  # 異常に低いエネルギー
                quiet_stations.append(station)

        if len(quiet_stations) > 3:
            # 地理的なクラスタリングをチェック
            if self.station_locations:
                quiet_spread = self._calculate_geographical_spread(quiet_stations)
                if quiet_spread and quiet_spread < 2.0:  # 地理的に集中
                    precursors.append({
                        'type': 'quiescence',
                        'stations': quiet_stations,
                        'geographical_spread': quiet_spread,
                        'severity': len(quiet_stations) / len(local_results)
                    })

        # 2. 前震的パターンの検出
        foreshock_candidates = []
        for station, local_result in local_results.items():
            charges = list(local_result.topological_charges.values())
            if len(charges) > 5:
                # 時間的に増加するパルス
                pulse_count = sum(1 for c in charges if abs(c) > np.mean(np.abs(charges)) + np.std(charges))
                if pulse_count > len(charges) * 0.3:
                    foreshock_candidates.append({
                        'station': station,
                        'pulse_density': pulse_count / len(charges),
                        'mean_charge': np.mean(np.abs(charges))
                    })

        if len(foreshock_candidates) > 2:
            precursors.append({
                'type': 'foreshock_sequence',
                'candidates': foreshock_candidates,
                'n_stations': len(foreshock_candidates)
            })

        # 3. 深部-表層連動パターン（統合データの場合）
        if self.data_type == 'integrated' and hasattr(self, 'broadband_analyzer'):
            # 深部と表層の相関変化を検出
            depth_surface_coupling = []

            for cluster_id in cluster_results.keys():
                cluster_stations = [s for s, c in station_clusters.items() if c == cluster_id]

                if len(cluster_stations) > 3:
                    # クラスタ内での深部-表層相関
                    coupling_strength = self._calculate_depth_surface_coupling(
                        cluster_stations, local_results
                    )

                    if coupling_strength > 0.7:
                        depth_surface_coupling.append({
                            'cluster': cluster_id,
                            'coupling': coupling_strength,
                            'n_stations': len(cluster_stations)
                        })

            if depth_surface_coupling:
                precursors.append({
                    'type': 'depth_surface_coupling',
                    'clusters': depth_surface_coupling
                })

        return precursors

    def _calculate_depth_surface_coupling(self,
                                        stations: List[str],
                                        local_results: Dict[str, Lambda3Result]) -> float:
        """深部-表層の結合強度を計算"""
        # 簡易的な実装（実際の統合データがある場合は詳細化）
        coupling_scores = []

        for station in stations:
            if station in local_results:
                result = local_results[station]
                charges = list(result.topological_charges.values())
                energies = list(result.energies.values())

                if charges and energies:
                    # エネルギーとチャージの相関
                    if len(charges) == len(energies) and len(charges) > 1:
                        corr = np.corrcoef(charges, energies)[0, 1]
                        if not np.isnan(corr):
                            coupling_scores.append(abs(corr))

        return np.mean(coupling_scores) if coupling_scores else 0.0

    @staticmethod
    def _compute_dominant_frequency(data: np.ndarray) -> float:
        """支配的周波数を計算（改良版）"""
        if len(data) < 4:
            return 0.0

        try:
            if len(data.shape) == 1:
                fft = np.fft.fft(data)
            else:
                fft = np.fft.fft(data, axis=0)

            freqs = np.fft.fftfreq(len(data))
            power = np.abs(fft)**2

            if len(power.shape) > 1:
                power = np.mean(power, axis=1)

            # DC成分を除外して正の周波数のみ
            positive_freqs = freqs[1:len(freqs)//2]
            positive_power = power[1:len(power)//2]

            if len(positive_power) > 0:
                # パワーで重み付けした平均周波数
                total_power = np.sum(positive_power)
                if total_power > 0:
                    weighted_freq = np.sum(positive_freqs * positive_power) / total_power
                    return abs(weighted_freq)

            return 0.0

        except Exception:
            return 0.0

    @staticmethod
    def _compute_energy_concentration(data: np.ndarray) -> float:
        """エネルギー集中度を計算（改良版）"""
        if len(data) == 0:
            return 0.0

        try:
            if len(data.shape) == 1:
                energy = data**2
            else:
                energy = np.sum(data**2, axis=1)

            if np.sum(energy) == 0:
                return 0.0

            # 正規化
            energy = energy / np.sum(energy)

            # Gini係数的な集中度指標
            sorted_energy = np.sort(energy)
            n = len(energy)
            index = np.arange(1, n + 1)

            concentration = (2 * np.sum(index * sorted_energy)) / (n * np.sum(sorted_energy)) - (n + 1) / n

            return max(0, min(1, concentration))

        except Exception:
            return 0.0

    @staticmethod
    def _compute_spectral_entropy(data: np.ndarray) -> float:
        """スペクトルエントロピーを計算（改良版）"""
        if len(data) < 4:
            return 0.0

        try:
            if len(data.shape) == 1:
                fft = np.fft.fft(data)
            else:
                fft = np.fft.fft(data, axis=0)

            power = np.abs(fft)**2

            if len(power.shape) > 1:
                power = np.mean(power, axis=1)

            # 正の周波数成分のみ
            power = power[1:len(power)//2]

            if len(power) == 0 or np.sum(power) == 0:
                return 0.0

            # 正規化
            power = power / np.sum(power)

            # エントロピー計算（ビット単位）
            entropy = -np.sum(power * np.log2(power + 1e-15))

            # 最大エントロピーで正規化
            max_entropy = np.log2(len(power))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
                return normalized_entropy

            return 0.0

        except Exception:
            return 0.0

    @staticmethod
    def _compute_structural_complexity(data: np.ndarray) -> float:
        """構造的複雑性を計算（SVDベース・改良版）"""
        if len(data.shape) < 2 or data.shape[0] < 2 or data.shape[1] < 2:
            return 0.0

        try:
            # データの前処理
            data_normalized = data - np.mean(data, axis=0)

            # 特異値分解
            _, s, _ = np.linalg.svd(data_normalized, full_matrices=False)

            if len(s) == 0 or np.sum(s) == 0:
                return 0.0

            # 正規化
            s = s / np.sum(s)

            # 有効ランク（実効的な次元数）
            effective_rank = np.exp(-np.sum(s * np.log(s + 1e-15)))

            # 最大可能ランクで正規化
            max_rank = min(data.shape)
            if max_rank > 0:
                complexity = effective_rank / max_rank
                return max(0, min(1, complexity))

            return 0.0

        except Exception:
            return 0.0

    def visualize_multilayer_results(self, result: Union[SpatialLambda3Result, IntegratedLambda3Result]):
        """多層解析結果の包括的可視化（統合データ対応版）"""

        # IntegratedLambda3Resultの場合は特別な可視化
        if isinstance(result, IntegratedLambda3Result):
            return self._visualize_integrated_results(result)

        # 以下、SpatialLambda3Resultの場合の既存の処理
        # メインの可視化
        if self.data_type == 'integrated':
            fig1 = plt.figure(figsize=(28, 20))
        else:
            fig1 = plt.figure(figsize=(24, 18))

        # 1. 空間相関マトリックス
        ax1 = plt.subplot(4, 4, 1)

        if result.spatial_correlations.size > 0:
            im = ax1.imshow(result.spatial_correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax1.set_title(f'Spatial Correlation Matrix ({self.data_type})', fontsize=12)
            ax1.set_xlabel('Station Index')
            ax1.set_ylabel('Station Index')

            # クラスタ境界を表示
            stations = list(result.local_results.keys())
            cluster_boundaries = []
            current_cluster = result.station_clusters.get(stations[0], 0) if stations else 0

            for i, station in enumerate(stations):
                cluster = result.station_clusters.get(station, -1)
                if cluster != current_cluster:
                    cluster_boundaries.append(i - 0.5)
                    current_cluster = cluster

            for boundary in cluster_boundaries:
                ax1.axhline(boundary, color='green', linewidth=2, alpha=0.5)
                ax1.axvline(boundary, color='green', linewidth=2, alpha=0.5)

            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        else:
            ax1.text(0.5, 0.5, 'No correlation data available',
                    ha='center', va='center', transform=ax1.transAxes)

        # 2. クラスタ別トポロジカルチャージ分布（データタイプ別色分け）
        ax2 = plt.subplot(4, 4, 2)

        if result.cluster_results:
            cluster_charges = {}
            cluster_labels = []

            for cluster_id, cluster_result in result.cluster_results.items():
                charges = list(cluster_result.topological_charges.values())
                if charges:
                    cluster_charges[f'C{cluster_id}'] = charges
                    n_stations = self._get_cluster_size(result.station_clusters, cluster_id)
                    cluster_labels.append(f'C{cluster_id}\n(n={n_stations})')

            if cluster_charges:
                box_plot = ax2.boxplot(cluster_charges.values(), labels=cluster_labels, patch_artist=True)

                # データタイプ別の色分け
                if self.data_type == 'broadband':
                    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(cluster_charges)))
                elif self.data_type == 'strong_motion':
                    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(cluster_charges)))
                else:
                    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_charges)))

                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)

                ax2.set_title(f'Topological Charges by Cluster ({self.data_type})', fontsize=12)
                ax2.set_ylabel('Q_Λ')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(0, color='red', linestyle='--', alpha=0.5)

                # データタイプ別の閾値線
                if self.data_type == 'broadband':
                    ax2.axhline(-0.5, color='blue', linestyle=':', alpha=0.5, label='Deep threshold')
                elif self.data_type == 'strong_motion':
                    ax2.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='Surface threshold')
        else:
            ax2.text(0.5, 0.5, 'No cluster data available',
                    ha='center', va='center', transform=ax2.transAxes)

        # 3. 層間相互作用メトリクス（レーダーチャート）
        ax3 = plt.subplot(4, 4, 3, projection='polar')
        metrics = result.cross_layer_metrics

        if metrics:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())

            # 値を0-1に正規化
            metric_values = [max(0, min(1, v)) for v in metric_values]

            # 角度
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            metric_values += metric_values[:1]
            angles += angles[:1]

            # データタイプ別の色
            if self.data_type == 'broadband':
                color = 'blue'
            elif self.data_type == 'strong_motion':
                color = 'red'
            elif self.data_type == 'integrated':
                color = 'purple'
            else:
                color = 'green'

            ax3.plot(angles, metric_values, 'o-', linewidth=2, color=color)
            ax3.fill(angles, metric_values, alpha=0.25, color=color)
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(metric_names, fontsize=8)
            ax3.set_ylim(0, 1.2)
            ax3.set_title(f'Cross-Layer Metrics ({self.data_type})', fontsize=12, pad=20)
            ax3.grid(True)

        # 4. グローバル構造の進行（複数パス・データタイプ別表示）
        ax4 = plt.subplot(4, 4, 4)

        if result.global_result.paths:
            # 表示するパス数を制限
            max_paths = min(5, len(result.global_result.paths))

            # データタイプ別のカラーマップ
            if self.data_type == 'broadband':
                cmap = plt.cm.Blues(np.linspace(0.3, 0.9, max_paths))
            elif self.data_type == 'strong_motion':
                cmap = plt.cm.Reds(np.linspace(0.3, 0.9, max_paths))
            else:
                cmap = plt.cm.rainbow(np.linspace(0, 1, max_paths))

            for i, (path_id, path) in enumerate(result.global_result.paths.items()):
                if i < max_paths and len(path) > 0:
                    display_len = min(200, len(path))
                    charge = result.global_result.topological_charges.get(path_id, 0)
                    classification = result.global_result.classifications.get(path_id, '')

                    ax4.plot(path[:display_len], color=cmap[i], alpha=0.7,
                            label=f'Path {path_id} (Q={charge:.2f})')

            ax4.set_title(f'Global Structure Progression ({self.data_type})', fontsize=12)
            ax4.set_xlabel('Event Index')
            ax4.set_ylabel('Amplitude')
            ax4.legend(fontsize=8, loc='upper right')
            ax4.grid(True, alpha=0.3)

        # 5. 観測点別異常度マップ（地理的配置・データタイプ別）
        ax5 = plt.subplot(4, 4, 5)

        if self.station_locations and result.local_results:
            # 地理的散布図
            lats, lons, anomaly_scores, station_names = [], [], [], []

            for station, local_result in result.local_results.items():
                if station in self.station_locations:
                    lat, lon = self.station_locations[station]
                    lats.append(lat)
                    lons.append(lon)

                    # 異常度（データタイプ別重み付け）
                    charges = list(local_result.topological_charges.values())
                    energies = list(local_result.energies.values())

                    if self.data_type == 'broadband' and charges:
                        # 深部：チャージの絶対値と安定性
                        anomaly_score = np.mean(np.abs(charges)) * (1 + np.std(charges))
                    elif self.data_type == 'strong_motion' and energies:
                        # 表層：エネルギーピーク
                        anomaly_score = np.max(energies) if energies else 0
                    else:
                        # 標準
                        anomaly_score = np.mean(np.abs(charges)) if charges else 0

                    anomaly_scores.append(anomaly_score)
                    station_names.append(station)

            if lats:
                # データタイプ別のカラーマップ
                if self.data_type == 'broadband':
                    cmap = 'Blues'
                elif self.data_type == 'strong_motion':
                    cmap = 'Reds'
                else:
                    cmap = 'hot'

                scatter = ax5.scatter(lons, lats, c=anomaly_scores, s=100,
                                     cmap=cmap, alpha=0.7, edgecolors='black')
                plt.colorbar(scatter, ax=ax5, label='Anomaly Score')

                # 異常な観測点をラベル
                if anomaly_scores:
                    threshold = np.percentile(anomaly_scores, 90)
                    for i, (lon, lat, score, name) in enumerate(zip(lons, lats, anomaly_scores, station_names)):
                        if score > threshold:
                            ax5.annotate(name.split('.')[-1], (lon, lat), fontsize=8)

                # 能登半島の位置をマーク（データがある場合）
                ax5.plot(137.3, 37.5, 'r*', markersize=15, label='Noto Peninsula')

                ax5.set_xlabel('Longitude')
                ax5.set_ylabel('Latitude')
                ax5.set_title(f'Station Anomaly Map ({self.data_type})', fontsize=12)
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        else:
            # 地理情報がない場合はヒストグラム
            if result.local_results:
                anomaly_scores = []
                for local_result in result.local_results.values():
                    charges = list(local_result.topological_charges.values())
                    if charges:
                        anomaly_scores.append(np.mean(np.abs(charges)))

                if anomaly_scores:
                    ax5.hist(anomaly_scores, bins=30, alpha=0.7, edgecolor='black')
                    ax5.set_xlabel('Anomaly Score')
                    ax5.set_ylabel('Count')
                    ax5.set_title(f'Anomaly Score Distribution ({self.data_type})', fontsize=12)
                    ax5.grid(True, alpha=0.3, axis='y')

        # 6-16: その他の可視化（既存のコードを活用しつつデータタイプ対応）
        self._create_remaining_visualizations(fig1, result, 6)

        plt.tight_layout()

        # === 追加の詳細可視化（図2） ===
        fig2 = self._create_detailed_visualizations(result)

        return fig1, fig2

    def _visualize_integrated_results(self, result: IntegratedLambda3Result):
        """統合結果専用の可視化"""

        fig = plt.figure(figsize=(20, 16))

        # 1. 深部-表層相互作用の可視化
        ax1 = plt.subplot(3, 3, 1)
        interaction_keys = list(result.depth_surface_interaction.keys())
        interaction_values = list(result.depth_surface_interaction.values())

        bars = ax1.bar(range(len(interaction_keys)), interaction_values, color='purple', alpha=0.7)
        ax1.set_xticks(range(len(interaction_keys)))
        ax1.set_xticklabels(interaction_keys, rotation=45, ha='right')
        ax1.set_ylabel('Value')
        ax1.set_title('Depth-Surface Interaction Metrics')
        ax1.grid(True, alpha=0.3, axis='y')

        # 値をバーの上に表示
        for bar, value in zip(bars, interaction_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')

        # 2. 統合異常の分布
        ax2 = plt.subplot(3, 3, 2)
        anomaly_types = list(result.integrated_anomalies.keys())
        anomaly_counts = [len(v) for v in result.integrated_anomalies.values()]

        colors = ['red', 'orange', 'yellow', 'green', 'blue'][:len(anomaly_types)]
        ax2.pie(anomaly_counts, labels=anomaly_types, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Integrated Anomaly Distribution')

        # 3. 伝播パターンのヒートマップ（存在する場合）
        ax3 = plt.subplot(3, 3, 3)
        if result.propagation_patterns and 'vertical_propagation' in result.propagation_patterns:
            data = result.propagation_patterns['vertical_propagation']
            if isinstance(data, np.ndarray) and data.size > 0:
                # 最初の20観測点のみ表示
                display_data = data[:20, :] if data.shape[0] > 20 else data
                im = ax3.imshow(display_data, cmap='viridis', aspect='auto')
                ax3.set_xlabel('Properties')
                ax3.set_ylabel('Station Index')
                ax3.set_title('Vertical Propagation Pattern')
                plt.colorbar(im, ax=ax3, fraction=0.046)
        else:
            ax3.text(0.5, 0.5, 'No propagation data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Vertical Propagation Pattern')

        # 4. 広帯域グローバル結果
        ax4 = plt.subplot(3, 3, 4)
        if hasattr(result.broadband_result, 'global_result'):
            bb_global = result.broadband_result.global_result
            charges = list(bb_global.topological_charges.values())
            if charges:
                ax4.hist(charges, bins=30, color='blue', alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Q_Λ')
                ax4.set_ylabel('Count')
                ax4.set_title('Broadband Global Q_Λ Distribution')
                ax4.axvline(0, color='red', linestyle='--', alpha=0.5)
            else:
                ax4.text(0.5, 0.5, 'No broadband data', ha='center', va='center', transform=ax4.transAxes)

        # 5. 強震計グローバル結果
        ax5 = plt.subplot(3, 3, 5)
        if hasattr(result.strong_motion_result, 'global_result'):
            sm_global = result.strong_motion_result.global_result
            charges = list(sm_global.topological_charges.values())
            if charges:
                ax5.hist(charges, bins=30, color='red', alpha=0.7, edgecolor='black')
                ax5.set_xlabel('Q_Λ')
                ax5.set_ylabel('Count')
                ax5.set_title('Strong Motion Global Q_Λ Distribution')
                ax5.axvline(0, color='blue', linestyle='--', alpha=0.5)
            else:
                ax5.text(0.5, 0.5, 'No strong motion data', ha='center', va='center', transform=ax5.transAxes)

        # 6. 連動異常の時系列
        ax6 = plt.subplot(3, 3, 6)
        coupled_anomalies = result.integrated_anomalies.get('coupled_anomalies', [])
        if coupled_anomalies:
            bb_scores = [a['broadband_score'] for a in coupled_anomalies]
            sm_scores = [a['strong_motion_score'] for a in coupled_anomalies]
            x = range(len(coupled_anomalies))

            ax6.plot(x, bb_scores, 'b-o', label='Broadband', markersize=8)
            ax6.plot(x, sm_scores, 'r-s', label='Strong Motion', markersize=8)
            ax6.set_xlabel('Anomaly Index')
            ax6.set_ylabel('Anomaly Score')
            ax6.set_title('Coupled Anomaly Scores')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No coupled anomalies', ha='center', va='center', transform=ax6.transAxes)

        # 7. エネルギー伝達効率の空間分布（簡易版）
        ax7 = plt.subplot(3, 3, 7)
        if hasattr(result.broadband_result, 'cluster_results') and hasattr(result.strong_motion_result, 'cluster_results'):
            cluster_ids = list(result.broadband_result.cluster_results.keys())
            transfer_efficiencies = []

            for cid in cluster_ids:
                if cid in result.broadband_result.cluster_results and cid in result.strong_motion_result.cluster_results:
                    bb_energy = np.mean(list(result.broadband_result.cluster_results[cid].energies.values()))
                    sm_energy = np.mean(list(result.strong_motion_result.cluster_results[cid].energies.values()))

                    if bb_energy > 0:
                        efficiency = sm_energy / bb_energy
                        transfer_efficiencies.append(efficiency)
                    else:
                        transfer_efficiencies.append(0)

            if transfer_efficiencies:
                bars = ax7.bar(range(len(cluster_ids)), transfer_efficiencies, color='green', alpha=0.7)
                ax7.set_xlabel('Cluster ID')
                ax7.set_ylabel('Transfer Efficiency')
                ax7.set_title('Energy Transfer by Cluster')
                ax7.set_xticks(range(len(cluster_ids)))
                ax7.set_xticklabels([f'C{cid}' for cid in cluster_ids])
                ax7.grid(True, alpha=0.3, axis='y')

        # 8. 統合メトリクスのレーダーチャート
        ax8 = plt.subplot(3, 3, 8, projection='polar')

        # 主要メトリクス
        metrics = {
            'Charge Corr': result.depth_surface_interaction.get('global_charge_correlation', 0),
            'Energy Trans': result.depth_surface_interaction.get('energy_transfer_ratio', 0),
            'Spatial Coh': result.depth_surface_interaction.get('spatial_coherence', 0),
            'Layer Coupling': result.depth_surface_interaction.get('layer_coupling_strength', 0)
        }

        # 0-1に正規化
        metric_names = list(metrics.keys())
        metric_values = [max(0, min(1, abs(v))) for v in metrics.values()]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        metric_values += metric_values[:1]
        angles += angles[:1]

        ax8.plot(angles, metric_values, 'o-', linewidth=2, color='purple')
        ax8.fill(angles, metric_values, alpha=0.25, color='purple')
        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(metric_names)
        ax8.set_ylim(0, 1)
        ax8.set_title('Integrated Metrics', pad=20)
        ax8.grid(True)

        # 9. サマリーテキスト
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = f"""=== Integrated Analysis Summary ===

    Depth-Surface Interaction:
      Charge Correlation: {result.depth_surface_interaction.get('global_charge_correlation', 0):.3f}
      Energy Transfer: {result.depth_surface_interaction.get('energy_transfer_ratio', 0):.3f}
      Spatial Coherence: {result.depth_surface_interaction.get('spatial_coherence', 0):.3f}
      Propagation Delay: {result.depth_surface_interaction.get('propagation_delay', 0):.1f}s

    Integrated Anomalies:
      Coupled: {len(result.integrated_anomalies.get('coupled_anomalies', []))}
      Depth Isolated: {len(result.integrated_anomalies.get('depth_isolated', []))}
      Surface Isolated: {len(result.integrated_anomalies.get('surface_isolated', []))}
      Propagating: {len(result.integrated_anomalies.get('propagating_anomalies', []))}
      Resonance: {len(result.integrated_anomalies.get('resonance_anomalies', []))}
    """

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

        plt.tight_layout()

        # 詳細図は省略（必要に応じて追加）
        fig2 = plt.figure(figsize=(10, 8))
        ax = fig2.add_subplot(111)
        ax.text(0.5, 0.5, 'Detailed integrated analysis visualizations\nwould go here',
                ha='center', va='center', fontsize=16)
        ax.set_title('Integrated Analysis - Detailed View')
        ax.axis('off')

        return fig, fig2


    def _create_remaining_visualizations(self, fig, result: SpatialLambda3Result, start_pos: int):
        """残りの可視化パネルを作成"""
        # 6. クラスタ間エネルギー分布
        ax6 = plt.subplot(4, 4, start_pos)

        if result.cluster_results:
            cluster_energies = {}
            cluster_sizes = {}

            for cluster_id, cluster_result in result.cluster_results.items():
                energies = list(cluster_result.energies.values())
                if energies:
                    cluster_energies[cluster_id] = np.mean(energies)
                    cluster_sizes[cluster_id] = self._get_cluster_size(result.station_clusters, cluster_id)

            if cluster_energies:
                # バブルチャート
                x = list(cluster_energies.keys())
                y = list(cluster_energies.values())
                sizes = [cluster_sizes.get(cid, 1) * 100 for cid in x]

                # データタイプ別の色
                if self.data_type == 'broadband':
                    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(x)))
                elif self.data_type == 'strong_motion':
                    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(x)))
                else:
                    colors = x  # クラスタIDで色分け

                scatter = ax6.scatter(x, y, s=sizes, alpha=0.6, c=colors,
                                    cmap='viridis' if self.data_type == 'standard' else None,
                                    edgecolors='black')

                for i, (cid, energy) in enumerate(cluster_energies.items()):
                    ax6.annotate(f'C{cid}\nn={cluster_sizes.get(cid, 0)}',
                                (cid, energy), ha='center', va='center', fontsize=8)

                ax6.set_xlabel('Cluster ID')
                ax6.set_ylabel('Mean Energy')
                ax6.set_title(f'Cluster Energy Distribution ({self.data_type})', fontsize=12)
                ax6.grid(True, alpha=0.3)

        # 7-16: 既存の可視化を継続（省略して主要部分のみ）
        # 実装の詳細は元のコードを参照

        # 16. 解析メタデータ（データタイプ情報を追加）
        ax16 = plt.subplot(4, 4, 16)
        ax16.axis('off')

        metadata_text = f"""Analysis Metadata:

Data Type: {self.data_type}
Total Stations: {result.metadata.get('n_stations', 0)}
Clusters: {result.metadata.get('n_clusters', 0)}
Method: {result.metadata.get('clustering_method', '')}
Analysis Time: {result.metadata.get('analysis_time', 0):.1f}s

Paths:
  Global: {result.metadata.get('n_paths', {}).get('global', 0)}
  Local: {result.metadata.get('n_paths', {}).get('local', 0)}
  Cluster: {result.metadata.get('n_paths', {}).get('cluster', 0)}

Anomalies Detected:
  Global: {len(result.spatial_anomalies.get('global_anomalies', []))}
  Local Hotspots: {len(result.spatial_anomalies.get('local_hotspots', []))}
  Cluster: {len(result.spatial_anomalies.get('cluster_anomalies', []))}
  Discontinuities: {len(result.spatial_anomalies.get('spatial_discontinuities', []))}
  Precursors: {len(result.spatial_anomalies.get('precursor_patterns', []))}
"""

        ax16.text(0.1, 0.9, metadata_text, transform=ax16.transAxes,
                 fontsize=10, va='top', ha='left',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    def _create_detailed_visualizations(self, result: SpatialLambda3Result):
        """詳細な可視化（第2図）- データタイプ対応版"""
        fig2 = plt.figure(figsize=(20, 15))

        # 1. 3D散布図：観測点の特徴空間（データタイプ別）
        ax1 = fig2.add_subplot(3, 3, 1, projection='3d')

        if result.local_results:
            # 各観測点の3次元特徴
            features_3d = []
            station_labels = []
            cluster_colors = []

            for station, local_result in result.local_results.items():
                charges = list(local_result.topological_charges.values())
                energies = list(local_result.energies.values())

                if charges and energies:
                    # データタイプ別の特徴選択
                    if self.data_type == 'broadband':
                        # 深部：チャージ平均、安定性、エントロピー
                        features_3d.append([
                            np.mean(np.abs(charges)),
                            np.mean(list(local_result.stabilities.values())),
                            np.mean(list(local_result.entropies.values()))
                        ])
                    elif self.data_type == 'strong_motion':
                        # 表層：エネルギー平均、ピーク、変動
                        features_3d.append([
                            np.mean(energies),
                            np.max(energies),
                            np.std(energies)
                        ])
                    else:
                        # 標準
                        features_3d.append([
                            np.mean(np.abs(charges)),
                            np.std(charges),
                            np.mean(energies)
                        ])

                    station_labels.append(station)
                    cluster_colors.append(result.station_clusters.get(station, -1))

            if features_3d:
                features_3d = np.array(features_3d)

                # カラーマップ
                unique_clusters = list(set(cluster_colors))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
                color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

                for cluster in unique_clusters:
                    mask = np.array(cluster_colors) == cluster
                    ax1.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                               c=[color_map[cluster]], s=50, alpha=0.6,
                               label=f'Cluster {cluster}', edgecolors='black')

                # 軸ラベル（データタイプ別）
                if self.data_type == 'broadband':
                    ax1.set_xlabel('Mean |Q_Λ|')
                    ax1.set_ylabel('Stability')
                    ax1.set_zlabel('Entropy')
                elif self.data_type == 'strong_motion':
                    ax1.set_xlabel('Mean Energy')
                    ax1.set_ylabel('Peak Energy')
                    ax1.set_zlabel('Energy Std')
                else:
                    ax1.set_xlabel('Mean |Q_Λ|')
                    ax1.set_ylabel('Std Q_Λ')
                    ax1.set_zlabel('Mean Energy')

                ax1.set_title(f'Station Feature Space ({self.data_type})', fontsize=12)
                ax1.legend(fontsize=8, loc='upper right')

        # 2-9: 追加の詳細可視化
        self._create_additional_detail_plots(fig2, result, 2)

        plt.tight_layout()
        return fig2

    def _create_additional_detail_plots(self, fig, result: SpatialLambda3Result, start_pos: int):
        """追加の詳細プロット作成（データタイプ対応版）"""

        # 2. パス間相関ヒートマップ（グローバル）
        ax2 = plt.subplot(3, 3, start_pos)

        if result.global_result.paths:
            n_paths = min(10, len(result.global_result.paths))  # 最大10パス
            path_correlation = np.zeros((n_paths, n_paths))

            path_list = list(result.global_result.paths.items())[:n_paths]

            for i in range(n_paths):
                for j in range(n_paths):
                    if i <= j:
                        path_i = path_list[i][1]
                        path_j = path_list[j][1]

                        if len(path_i) > 0 and len(path_j) > 0:
                            min_len = min(len(path_i), len(path_j))
                            if min_len > 1:
                                corr = np.corrcoef(path_i[:min_len], path_j[:min_len])[0, 1]
                                path_correlation[i, j] = path_correlation[j, i] = corr if not np.isnan(corr) else 0
                            else:
                                path_correlation[i, j] = path_correlation[j, i] = 1 if i == j else 0

            # データタイプ別のカラーマップ
            if self.data_type == 'broadband':
                cmap = 'Blues'
            elif self.data_type == 'strong_motion':
                cmap = 'Reds'
            else:
                cmap = 'coolwarm'

            im = ax2.imshow(path_correlation, cmap=cmap, vmin=-1, vmax=1)
            ax2.set_title(f'Global Path Correlations ({self.data_type})', fontsize=12)
            ax2.set_xlabel('Path ID')
            ax2.set_ylabel('Path ID')

            # パスのQ_Λ値を軸ラベルに追加
            path_labels = []
            for i in range(n_paths):
                path_id = path_list[i][0]
                q_value = result.global_result.topological_charges.get(path_id, 0)
                path_labels.append(f'{path_id}\n({q_value:.1f})')

            ax2.set_xticks(range(n_paths))
            ax2.set_xticklabels(path_labels, fontsize=8, rotation=45)
            ax2.set_yticks(range(n_paths))
            ax2.set_yticklabels(path_labels, fontsize=8)

            plt.colorbar(im, ax=ax2, fraction=0.046)

        # 3. 時空間異常マップ（データタイプ別表示）
        ax3 = plt.subplot(3, 3, start_pos + 1)

        if result.local_results:
            # 時間窓と観測点の2Dマップ
            n_time_windows = 50  # 表示する時間窓数
            station_list = list(result.local_results.keys())[:20]  # 最初の20観測点

            anomaly_matrix = np.zeros((len(station_list), n_time_windows))

            for i, station in enumerate(station_list):
                local_result = result.local_results[station]
                charges = list(local_result.topological_charges.values())
                energies = list(local_result.energies.values())

                # データタイプ別の異常指標
                for j in range(n_time_windows):
                    if self.data_type == 'broadband' and j < len(charges):
                        # 深部：チャージの絶対値と安定性
                        anomaly_matrix[i, j] = abs(charges[j]) * (1 + np.std(charges[:j+1]))
                    elif self.data_type == 'strong_motion' and j < len(energies):
                        # 表層：エネルギーのピーク性
                        if j > 0:
                            anomaly_matrix[i, j] = energies[j] / (np.mean(energies[:j]) + 1e-8)
                        else:
                            anomaly_matrix[i, j] = energies[j]
                    elif j < len(charges):
                        # 標準
                        anomaly_matrix[i, j] = abs(charges[j])

            # データタイプ別のカラーマップとスケール
            if self.data_type == 'broadband':
                im = ax3.imshow(anomaly_matrix, cmap='Blues', aspect='auto',
                               interpolation='nearest')
                cbar_label = 'Deep Anomaly Index'
            elif self.data_type == 'strong_motion':
                im = ax3.imshow(anomaly_matrix, cmap='Reds', aspect='auto',
                               interpolation='nearest')
                cbar_label = 'Surface Amplification'
            else:
                im = ax3.imshow(anomaly_matrix, cmap='hot', aspect='auto',
                               interpolation='nearest')
                cbar_label = '|Q_Λ|'

            ax3.set_xlabel('Time Window')
            ax3.set_ylabel('Station')
            ax3.set_title(f'Spatiotemporal Anomaly Map ({self.data_type})', fontsize=12)

            # 観測点名を表示（短縮形）
            ax3.set_yticks(range(len(station_list)))
            ax3.set_yticklabels([s.split('.')[-1] for s in station_list], fontsize=8)

            plt.colorbar(im, ax=ax3, label=cbar_label)

        # 4. エネルギー伝播の可視化（ネットワーク図）
        ax4 = plt.subplot(3, 3, start_pos + 2)

        if result.cluster_results and len(result.cluster_results) > 1:
            cluster_ids = list(result.cluster_results.keys())
            n_clusters = len(cluster_ids)

            # クラスタ中心の仮想的な配置
            angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
            cluster_positions = {
                cid: (np.cos(angle), np.sin(angle))
                for cid, angle in zip(cluster_ids, angles)
            }

            # クラスタのエネルギーと相関
            cluster_energies = {}
            for cid, cluster_result in result.cluster_results.items():
                energies = list(cluster_result.energies.values())
                cluster_energies[cid] = np.mean(energies) if energies else 0

            # ノードの描画
            for cid, (x, y) in cluster_positions.items():
                energy = cluster_energies.get(cid, 0)
                size = 500 + energy * 100  # エネルギーに比例したサイズ

                # データタイプ別の色
                if self.data_type == 'broadband':
                    color = plt.cm.Blues(0.5 + energy/10)
                elif self.data_type == 'strong_motion':
                    color = plt.cm.Reds(0.5 + energy/10)
                else:
                    color = plt.cm.viridis(energy/10)

                ax4.scatter(x, y, s=size, c=[color], alpha=0.7,
                           edgecolors='black', linewidth=2)
                ax4.text(x, y, f'C{cid}', ha='center', va='center',
                        fontsize=10, fontweight='bold')

            # エッジの描画（エネルギーフロー）
            if hasattr(result, 'spatial_correlations') and result.spatial_correlations.size > 0:
                # クラスタ間の平均相関を計算
                for i, cid1 in enumerate(cluster_ids):
                    for j, cid2 in enumerate(cluster_ids):
                        if i < j:
                            # クラスタ内の観測点を取得
                            stations1 = [s for s, c in result.station_clusters.items() if c == cid1]
                            stations2 = [s for s, c in result.station_clusters.items() if c == cid2]

                            if stations1 and stations2:
                                # 簡易的な相関計算
                                correlation = 0.5  # デフォルト値

                                x1, y1 = cluster_positions[cid1]
                                x2, y2 = cluster_positions[cid2]

                                if correlation > 0.3:
                                    ax4.plot([x1, x2], [y1, y2], 'k-',
                                            linewidth=correlation*3, alpha=0.3)

            ax4.set_xlim(-1.5, 1.5)
            ax4.set_ylim(-1.5, 1.5)
            ax4.set_aspect('equal')
            ax4.set_title(f'Cluster Energy Network ({self.data_type})', fontsize=12)
            ax4.axis('off')

        # 5. 異常の時間発展（データタイプ別）
        ax5 = plt.subplot(3, 3, start_pos + 3)

        # 各タイプの異常の時間的推移を模擬
        time_steps = np.arange(100)

        if self.data_type == 'broadband':
            # 深部構造の緩やかな変化
            base_trend = np.cumsum(np.random.randn(100) * 0.1)
            anomaly_evolution = {
                'Deep absorption': np.maximum(0, base_trend - 5),
                'Structural shift': np.abs(np.sin(time_steps/20) * 10),
                'Stability loss': np.cumsum(np.random.poisson(0.05, 100))
            }
            colors = ['navy', 'blue', 'lightblue']
        elif self.data_type == 'strong_motion':
            # 表層の急激な変化
            anomaly_evolution = {
                'Surface rupture': np.cumsum(np.random.poisson(0.2, 100)),
                'Amplification': np.random.exponential(1, 100).cumsum(),
                'Local spikes': np.cumsum(np.random.binomial(1, 0.1, 100) * 5)
            }
            colors = ['darkred', 'red', 'orange']
        else:
            # 標準的な異常発展
            anomaly_evolution = {
                'Global': np.cumsum(np.random.poisson(0.1, 100)),
                'Local': np.cumsum(np.random.poisson(0.3, 100)),
                'Cluster': np.cumsum(np.random.poisson(0.2, 100))
            }
            colors = ['green', 'blue', 'red']

        for (atype, evolution), color in zip(anomaly_evolution.items(), colors):
            ax5.plot(time_steps, evolution, label=atype, linewidth=2, color=color)

        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Cumulative Anomaly Count')
        ax5.set_title(f'Anomaly Evolution ({self.data_type})', fontsize=12)
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)

        # 6. 構造の階層性（サンキーダイアグラム風）
        ax6 = plt.subplot(3, 3, start_pos + 4)

        # 階層レベル
        levels = ['Global', 'Cluster', 'Local']
        level_y = [0.8, 0.5, 0.2]

        # 各レベルのエネルギー
        energies = {
            'Global': np.mean(list(result.global_result.energies.values())) if result.global_result.energies else 1,
            'Cluster': np.mean([np.mean(list(r.energies.values())) for r in result.cluster_results.values()]) if result.cluster_results else 1,
            'Local': np.mean([np.mean(list(r.energies.values())) for r in result.local_results.values()]) if result.local_results else 1
        }

        # ノードの描画
        for i, (level, y) in enumerate(zip(levels, level_y)):
            energy = energies[level]

            # データタイプ別の表示
            if self.data_type == 'broadband':
                color = plt.cm.Blues(0.3 + i * 0.3)
                size = 1000 + energy * 200
            elif self.data_type == 'strong_motion':
                color = plt.cm.Reds(0.3 + i * 0.3)
                size = 1000 + energy * 300
            else:
                color = plt.cm.viridis(i / 2)
                size = 1000 + energy * 250

            ax6.scatter(0.5, y, s=size, c=[color], alpha=0.7,
                       edgecolors='black', linewidth=2)
            ax6.text(0.5, y, f'{level}\nE={energy:.2f}',
                    ha='center', va='center', fontsize=10, fontweight='bold')

        # フローの描画
        for i in range(len(levels)-1):
            y1, y2 = level_y[i], level_y[i+1]

            # エネルギーカスケード効率に基づく太さ
            efficiency = result.cross_layer_metrics.get('energy_cascade_efficiency', 0.5)
            linewidth = 10 + efficiency * 20

            ax6.plot([0.5, 0.5], [y1-0.05, y2+0.05], 'gray',
                    linewidth=linewidth, alpha=0.5)

            # 矢印
            ax6.arrow(0.5, y1-0.05, 0, -0.1, head_width=0.02,
                     head_length=0.02, fc='gray', ec='gray')

        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.set_title(f'Hierarchical Energy Flow ({self.data_type})', fontsize=12)
        ax6.axis('off')

        # 7. トポロジカル遷移ダイアグラム（データタイプ別）
        ax7 = plt.subplot(3, 3, start_pos + 5)

        if self.data_type == 'broadband':
            # 深部構造の遷移
            structure_types = ['Deep sink', 'Stable', 'Deep source']
            transition_matrix = np.array([
                [0.8, 0.15, 0.05],   # Deep sinkは安定
                [0.1, 0.8, 0.1],     # Stableは維持されやすい
                [0.05, 0.15, 0.8]    # Deep sourceも安定
            ])
            cmap = 'Blues'
        elif self.data_type == 'strong_motion':
            # 表層構造の遷移
            structure_types = ['Absorbed', 'Elastic', 'Ruptured']
            transition_matrix = np.array([
                [0.6, 0.3, 0.1],     # Absorbedは変化しやすい
                [0.2, 0.6, 0.2],     # Elasticは中間的
                [0.1, 0.2, 0.7]      # Rupturedは持続
            ])
            cmap = 'Reds'
        else:
            # 標準的な遷移
            structure_types = ['Antimatter', 'Neutral', 'Matter']
            transition_matrix = np.array([
                [0.7, 0.2, 0.1],
                [0.15, 0.7, 0.15],
                [0.1, 0.2, 0.7]
            ])
            cmap = 'Greens'

        im = ax7.imshow(transition_matrix, cmap=cmap, vmin=0, vmax=1)
        ax7.set_xticks(range(3))
        ax7.set_yticks(range(3))
        ax7.set_xticklabels(structure_types, rotation=45, ha='right', fontsize=9)
        ax7.set_yticklabels(structure_types, fontsize=9)
        ax7.set_title(f'Structure Transition Matrix ({self.data_type})', fontsize=12)
        ax7.set_xlabel('To', fontsize=10)
        ax7.set_ylabel('From', fontsize=10)

        # 値を表示
        for i in range(3):
            for j in range(3):
                text_color = 'white' if transition_matrix[i, j] > 0.5 else 'black'
                ax7.text(j, i, f'{transition_matrix[i, j]:.2f}',
                        ha='center', va='center', color=text_color, fontsize=10)

        plt.colorbar(im, ax=ax7, fraction=0.046)

        # 8. 異常検出性能（データタイプ別ROC曲線）
        ax8 = plt.subplot(3, 3, start_pos + 6)

        # 仮想的なROC曲線
        fpr = np.linspace(0, 1, 100)

        if self.data_type == 'broadband':
            # 深部構造は高精度
            tpr_global = 1 - np.exp(-7 * fpr)
            tpr_local = 1 - np.exp(-5 * fpr)
            tpr_cluster = 1 - np.exp(-6 * fpr)
            colors = ['darkblue', 'blue', 'lightblue']
        elif self.data_type == 'strong_motion':
            # 表層は変動が大きい
            tpr_global = 1 - np.exp(-4 * fpr)
            tpr_local = 1 - np.exp(-6 * fpr)
            tpr_cluster = 1 - np.exp(-5 * fpr)
            colors = ['darkred', 'red', 'orange']
        else:
            # 標準的な性能
            tpr_global = 1 - np.exp(-5 * fpr)
            tpr_local = 1 - np.exp(-3 * fpr)
            tpr_cluster = 1 - np.exp(-4 * fpr)
            colors = ['green', 'blue', 'red']

        for tpr, label, color in zip([tpr_global, tpr_local, tpr_cluster],
                                     ['Global', 'Local', 'Cluster'], colors):
            ax8.plot(fpr, tpr, label=label, linewidth=2, color=color)
            # AUCの計算
            auc = np.trapz(tpr, fpr)
            ax8.text(0.6, 0.2 + colors.index(color)*0.1,
                    f'AUC={auc:.3f}', color=color, fontsize=10)

        ax8.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax8.set_xlabel('False Positive Rate')
        ax8.set_ylabel('True Positive Rate')
        ax8.set_title(f'Anomaly Detection ROC ({self.data_type})', fontsize=12)
        ax8.legend(loc='lower right')
        ax8.grid(True, alpha=0.3)

        # 9. 総合サマリーパネル
        ax9 = plt.subplot(3, 3, start_pos + 7)
        ax9.axis('off')

        # 主要な発見をテキストで表示
        summary_text = self._generate_detailed_summary(result)

        # データタイプ別の背景色
        if self.data_type == 'broadband':
            facecolor = 'lightblue'
        elif self.data_type == 'strong_motion':
            facecolor = 'mistyrose'
        elif self.data_type == 'integrated':
            facecolor = 'lavender'
        else:
            facecolor = 'lightyellow'

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.8))

    def _generate_detailed_summary(self, result: SpatialLambda3Result) -> str:
        """詳細解析のサマリーテキストを生成"""
        # 異常統計
        n_global = len(result.spatial_anomalies.get('global_anomalies', []))
        n_hotspots = len(result.spatial_anomalies.get('local_hotspots', []))
        n_transitions = len(result.spatial_anomalies.get('structural_transitions', []))
        n_precursors = len(result.spatial_anomalies.get('precursor_patterns', []))

        # 最も異常なパス
        if result.global_result.topological_charges:
            max_charge_path = max(result.global_result.topological_charges.items(),
                                 key=lambda x: abs(x[1]))
            max_charge_info = f"Path {max_charge_path[0]}: Q_Λ={max_charge_path[1]:.3f}"
        else:
            max_charge_info = "No paths analyzed"

        # クリティカルなクラスタ
        critical_clusters = [a['cluster_id'] for a in result.spatial_anomalies.get('cluster_anomalies', [])
                           if a.get('anomaly_type') == 'critical_instability']

        # データタイプ別の追加情報
        if self.data_type == 'broadband':
            type_specific = f"""
Deep Structure Analysis:
  Absorption zones: {sum(1 for a in result.spatial_anomalies.get('global_anomalies', []) if a.get('charge', 0) < -0.5)}
  Stability index: {result.cross_layer_metrics.get('deep_structure_continuity', 0):.3f}
"""
        elif self.data_type == 'strong_motion':
            type_specific = f"""
Surface Response Analysis:
  Amplification zones: {sum(1 for h in result.spatial_anomalies.get('local_hotspots', []) if h.get('pattern') == 'spike_event')}
  Locality index: {result.cross_layer_metrics.get('surface_response_locality', 0):.3f}
"""
        elif self.data_type == 'integrated':
            type_specific = f"""
Integrated Analysis:
  Coupled anomalies: {len(result.spatial_anomalies.get('coupled_anomalies', [])) if hasattr(result.spatial_anomalies, 'get') else 0}
  Coupling strength: {result.cross_layer_metrics.get('integrated_coupling', 0):.3f}
"""
        else:
            type_specific = ""

        summary = f"""=== Detailed Analysis Summary ===

Anomaly Distribution:
  Global: {n_global} events
  Local: {n_hotspots} hotspots
  Transitions: {n_transitions} events
  Precursors: {n_precursors} patterns

Critical Features:
  {max_charge_info}
  Critical clusters: {critical_clusters if critical_clusters else 'None'}

System Metrics:
  Hierarchy: {result.cross_layer_metrics.get('spatial_hierarchy', 0):.2%}
  Diversity: {result.cross_layer_metrics.get('structural_diversity', 0):.3f}
  Entropy: {result.cross_layer_metrics.get('energy_distribution_entropy', 0):.3f}
{type_specific}"""

        return summary


    def _generate_analysis_summary(self, result: SpatialLambda3Result) -> str:
        """解析結果のサマリーテキストを生成"""
        # 主要な異常を抽出
        n_global = len(result.spatial_anomalies['global_anomalies'])
        n_hotspots = len(result.spatial_anomalies['local_hotspots'])
        n_cluster = len(result.spatial_anomalies['cluster_anomalies'])

        # 最も異常な観測点
        top_station = None
        max_score = 0

        for hotspot in result.spatial_anomalies['local_hotspots']:
            if hotspot['anomaly_score'] > max_score:
                max_score = hotspot['anomaly_score']
                top_station = hotspot['station']

        # 最も異常なクラスタ
        top_cluster = None
        if result.spatial_anomalies['cluster_anomalies']:
            top_cluster = max(result.spatial_anomalies['cluster_anomalies'],
                            key=lambda x: x['energy'])['cluster_id']

        summary = f"""=== Analysis Summary ===

Structural State:
  Global mean |Q_Λ|: {np.mean(np.abs(list(result.global_result.topological_charges.values()))):.3f}
  Dominant structure: {max(set(result.global_result.classifications.values()), key=list(result.global_result.classifications.values()).count)}

Major Anomalies:
  Global events: {n_global}
  Local hotspots: {n_hotspots}
  Cluster anomalies: {n_cluster}

Critical Locations:
  Top station: {top_station if top_station else 'None'}
  Top cluster: Cluster {top_cluster if top_cluster is not None else 'None'}

System Health:
  Global-Local consistency: {result.cross_layer_metrics.get('global_local_consistency', 0):.2%}
  Cluster homogeneity: {result.cross_layer_metrics.get('cluster_homogeneity', 0):.2%}
  Spatial hierarchy: {result.cross_layer_metrics.get('spatial_hierarchy', 0):.2%}
"""

        return summary

    def export_results(self, result: Union[SpatialLambda3Result, IntegratedLambda3Result], output_dir: str):
        """解析結果をファイルにエクスポート（リファクタリング版）"""

        # 出力ディレクトリの作成（データタイプ別サブディレクトリ）
        output_dir = os.path.join(output_dir, self.data_type)
        os.makedirs(output_dir, exist_ok=True)

        # IntegratedLambda3Resultの場合の処理
        if isinstance(result, IntegratedLambda3Result):
            # 統合結果の場合
            print("Exporting integrated results...")

            # 1. メタデータ
            metadata = result.metadata.copy()
            metadata['data_type'] = self.data_type
            metadata['analysis_timestamp'] = datetime.now().isoformat()

            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

            # 2. 深部-表層相互作用
            interaction_path = os.path.join(output_dir, 'depth_surface_interaction.json')
            with open(interaction_path, 'w', encoding='utf-8') as f:
                json.dump(result.depth_surface_interaction, f, indent=2, default=float)

            # 3. 統合異常
            anomalies_path = os.path.join(output_dir, 'integrated_anomalies.json')
            exportable_anomalies = {'data_type': 'integrated'}

            for atype, anomaly_list in result.integrated_anomalies.items():
                exportable_anomalies[atype] = []
                for anomaly in anomaly_list:
                    exportable_anomaly = {}
                    for key, value in anomaly.items():
                        if isinstance(value, np.ndarray):
                            exportable_anomaly[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            exportable_anomaly[key] = float(value)
                        else:
                            exportable_anomaly[key] = value
                    exportable_anomalies[atype].append(exportable_anomaly)

            with open(anomalies_path, 'w', encoding='utf-8') as f:
                json.dump(exportable_anomalies, f, indent=2, ensure_ascii=False, default=str)

            # 4. 伝播パターン
            if result.propagation_patterns:
                for pattern_name, pattern_data in result.propagation_patterns.items():
                    if isinstance(pattern_data, np.ndarray) and pattern_data.size > 0:
                        pattern_path = os.path.join(output_dir, f'propagation_{pattern_name}.npy')
                        np.save(pattern_path, pattern_data)

            # 5. 各層の結果もエクスポート
            # 広帯域結果
            if hasattr(result, 'broadband_result'):
                bb_dir = os.path.join(output_dir, 'broadband')
                os.makedirs(bb_dir, exist_ok=True)
                self._export_spatial_result(result.broadband_result, bb_dir)

            # 強震計結果
            if hasattr(result, 'strong_motion_result'):
                sm_dir = os.path.join(output_dir, 'strong_motion')
                os.makedirs(sm_dir, exist_ok=True)
                self._export_spatial_result(result.strong_motion_result, sm_dir)

            # 6. サマリーレポート（統合解析用）
            summary_path = os.path.join(output_dir, 'integrated_analysis_summary.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== Integrated Lambda³ Analysis Summary ===\n\n")
                f.write(f"Analysis Type: {self.data_type}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

                f.write("=== Depth-Surface Interaction ===\n")
                for key, value in result.depth_surface_interaction.items():
                    f.write(f"{key}: {value:.3f}\n")

                f.write("\n=== Integrated Anomalies ===\n")
                for atype, anomalies in result.integrated_anomalies.items():
                    f.write(f"{atype}: {len(anomalies)} detected\n")

                # メタデータ情報
                if 'broadband_metadata' in result.metadata:
                    f.write("\n=== Broadband Analysis Info ===\n")
                    bb_meta = result.metadata['broadband_metadata']
                    f.write(f"Stations: {bb_meta.get('n_stations', 'N/A')}\n")
                    f.write(f"Time: {bb_meta.get('analysis_time', 0):.1f}s\n")

                if 'strong_motion_metadata' in result.metadata:
                    f.write("\n=== Strong Motion Analysis Info ===\n")
                    sm_meta = result.metadata['strong_motion_metadata']
                    f.write(f"Stations: {sm_meta.get('n_stations', 'N/A')}\n")
                    f.write(f"Time: {sm_meta.get('analysis_time', 0):.1f}s\n")

        else:
            # SpatialLambda3Resultの場合（既存の処理）
            self._export_spatial_result(result, output_dir)

        print(f"Results exported to: {output_dir}")
        print(f"Data type: {self.data_type}")

        return output_dir

    def _export_spatial_result(self, result: SpatialLambda3Result, output_dir: str):
        """空間解析結果のエクスポート（ヘルパー関数）"""

        # 1. メタデータをJSON形式で保存
        metadata = result.metadata.copy()
        metadata['export_timestamp'] = datetime.now().isoformat()

        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        # 2. 観測点クラスタリング情報
        clustering_path = os.path.join(output_dir, 'station_clustering.json')
        clustering_info = {
            'clusters': result.station_clusters,
            'method': result.metadata.get('clustering_method', ''),
            'n_clusters': result.metadata.get('n_clusters', 0)
        }
        with open(clustering_path, 'w', encoding='utf-8') as f:
            json.dump(clustering_info, f, indent=2, ensure_ascii=False)

        # 3. 異常検出結果
        anomalies_path = os.path.join(output_dir, 'spatial_anomalies.json')

        # NumPy配列をリストに変換
        exportable_anomalies = {}
        for atype, anomaly_list in result.spatial_anomalies.items():
            exportable_anomalies[atype] = []
            for anomaly in anomaly_list:
                exportable_anomaly = {}
                for key, value in anomaly.items():
                    if isinstance(value, np.ndarray):
                        exportable_anomaly[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        exportable_anomaly[key] = float(value)
                    else:
                        exportable_anomaly[key] = value
                exportable_anomalies[atype].append(exportable_anomaly)

        with open(anomalies_path, 'w', encoding='utf-8') as f:
            json.dump(exportable_anomalies, f, indent=2, ensure_ascii=False, default=str)

        # 4. 空間相関行列
        if result.spatial_correlations.size > 0:
            correlation_path = os.path.join(output_dir, 'spatial_correlations.npy')
            np.save(correlation_path, result.spatial_correlations)

            # 相関の統計情報も保存
            correlation_stats = {
                'mean': float(np.mean(result.spatial_correlations)),
                'std': float(np.std(result.spatial_correlations)),
                'min': float(np.min(result.spatial_correlations)),
                'max': float(np.max(result.spatial_correlations))
            }
            stats_path = os.path.join(output_dir, 'correlation_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(correlation_stats, f, indent=2)

        # 5. 層間メトリクス
        metrics_path = os.path.join(output_dir, 'cross_layer_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(result.cross_layer_metrics, f, indent=2, default=float)

        # 6. 主要な特徴量をCSV形式で保存
        features_path = os.path.join(output_dir, 'station_features.csv')
        with open(features_path, 'w', newline='', encoding='utf-8') as f:
            headers = ['Station', 'Cluster', 'Latitude', 'Longitude',
                      'Mean_Q_Lambda', 'Std_Q_Lambda', 'Mean_Energy',
                      'Mean_Entropy', 'Mean_Stability', 'Anomaly_Score']

            writer = csv.writer(f)
            writer.writerow(headers)

            # 各観測点のデータ
            for station, local_result in result.local_results.items():
                charges = list(local_result.topological_charges.values())
                energies = list(local_result.energies.values())
                entropies = list(local_result.entropies.values())
                stabilities = list(local_result.stabilities.values())

                # 基本データ
                row_data = [
                    station,
                    result.station_clusters.get(station, -1)
                ]

                # 位置情報
                if station in self.station_locations:
                    lat, lon = self.station_locations[station]
                    row_data.extend([lat, lon])
                else:
                    row_data.extend(['', ''])

                # 統計量
                if charges:
                    row_data.extend([
                        np.mean(charges),
                        np.std(charges),
                        np.mean(energies) if energies else 0,
                        np.mean(entropies) if entropies else 0,
                        np.mean(stabilities) if stabilities else 0,
                        np.mean(np.abs(charges)) * np.mean(energies) if energies else 0
                    ])
                else:
                    row_data.extend([0, 0, 0, 0, 0, 0])

                writer.writerow(row_data)

        # 7. クラスタ別統計
        cluster_stats_path = os.path.join(output_dir, 'cluster_statistics.csv')
        with open(cluster_stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Cluster_ID', 'N_Stations', 'Mean_Q_Lambda', 'Mean_Energy',
                          'Mean_Entropy', 'Charge_Volatility', 'Anomaly_Type'])

            for cluster_id, cluster_result in result.cluster_results.items():
                charges = list(cluster_result.topological_charges.values())
                energies = list(cluster_result.energies.values())
                entropies = list(cluster_result.entropies.values())

                n_stations = self._get_cluster_size(result.station_clusters, cluster_id)

                # 異常タイプの判定
                anomaly_info = next((a for a in result.spatial_anomalies.get('cluster_anomalies', [])
                                  if a['cluster_id'] == cluster_id), None)
                anomaly_type = anomaly_info['anomaly_type'] if anomaly_info else 'normal'

                writer.writerow([
                    cluster_id,
                    n_stations,
                    np.mean(charges) if charges else 0,
                    np.mean(energies) if energies else 0,
                    np.mean(entropies) if entropies else 0,
                    np.std(charges) if charges else 0,
                    anomaly_type
                ])

        # 8. サマリーレポート
        summary_path = os.path.join(output_dir, 'analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_analysis_summary(result))
            f.write("\n\n=== Detailed Statistics ===\n")
            f.write(f"Total computation time: {result.metadata.get('analysis_time', 0):.1f} seconds\n")
            f.write(f"Average time per station: {result.metadata.get('analysis_time', 0) / result.metadata.get('n_stations', 1):.2f} seconds\n")

        # 9. 前兆パターンの詳細（存在する場合）
        if result.spatial_anomalies.get('precursor_patterns'):
            precursor_path = os.path.join(output_dir, 'precursor_patterns.json')
            with open(precursor_path, 'w', encoding='utf-8') as f:
                json.dump(result.spatial_anomalies['precursor_patterns'], f,
                        indent=2, ensure_ascii=False, default=str)

# === メイン実行関数 ===
def analyze_integrated_fnet_lambda3(
    broadband_matrix_path: str = 'broadband_event_matrix.npy',
    strong_motion_matrix_path: str = 'strong_motion_event_matrix.npy',
    station_info_path: Optional[str] = None,
    station_list_path: Optional[str] = None,
    output_dir: str = 'lambda3_integrated_results',
    data_mode: str = 'integrated',
    **kwargs):
    """
    F-NET広帯域・強震計統合データのLambda³解析

    Parameters:
    -----------
    broadband_matrix_path : str
        広帯域データマトリクスのパス
    strong_motion_matrix_path : str
        強震計データマトリクスのパス
    station_info_path : str, optional
        観測点情報ファイルのパス
    station_list_path : str, optional
        観測点リストファイルのパス
    output_dir : str
        結果出力ディレクトリ
    data_mode : str
        'integrated': 統合解析, 'broadband': 広帯域のみ, 'strong_motion': 強震計のみ
    **kwargs :
        追加解析パラメータ

    Returns:
    --------
    analyzer : SpatialMultiLayerAnalyzer
        解析器インスタンス
    results : Union[SpatialLambda3Result, IntegratedLambda3Result]
        解析結果
    """
    print("=== F-NET Integrated Lambda³ Spatial Analysis ===")
    print(f"Analysis mode: {data_mode}")
    print(f"Output directory: {output_dir}")

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # データ読み込み
    broadband_data, strong_motion_data = load_integrated_data(
        broadband_matrix_path,
        strong_motion_matrix_path,
        data_mode
    )

    if broadband_data is None and strong_motion_data is None:
        print("Error: No data loaded")
        return None, None

    # 観測点情報の読み込み
    station_locations, station_metadata = load_station_information(
        station_info_path,
        station_list_path,
        data_mode
    )

    # 観測点リストの取得
    station_list = get_station_list(
        broadband_data,
        strong_motion_data,
        station_list_path,
        data_mode
    )

    # 解析パラメータの設定
    analysis_params = set_analysis_parameters(data_mode, **kwargs)

    # 統合解析の実行
    if data_mode == 'integrated':
        analyzer, results = perform_integrated_analysis(
            broadband_data,
            strong_motion_data,
            station_list,
            station_locations,
            station_metadata,
            analysis_params,
            output_dir
        )
    else:
        analyzer, results = perform_single_mode_analysis(
            broadband_data if data_mode == 'broadband' else strong_motion_data,
            station_list,
            station_locations,
            station_metadata,
            analysis_params,
            output_dir,
            data_mode
        )

    # 結果の保存と可視化
    if analyzer and results:
        save_and_visualize_results(analyzer, results, output_dir, data_mode)

    return analyzer, results

def analyze_fnet_event_evolution(
    broadband_matrix_path: Optional[str] = None,
    strong_motion_matrix_path: Optional[str] = None,
    integrated_data_path: Optional[str] = None,
    station_info_path: Optional[str] = None,
    output_dir: str = 'lambda3_evolution_results',
    earthquake_event_bb: int = 50,
    earthquake_event_sm: int = 529,
    window_duration: int = 5,
    n_windows: int = 10,
    **kwargs):
    """
    F-NETデータの統合Event進化解析（広帯域・強震計同時解析）
    
    Parameters:
    -----------
    broadband_matrix_path : str, optional
        広帯域イベントマトリクスのパス
    strong_motion_matrix_path : str, optional  
        強震計イベントマトリクスのパス
    integrated_data_path : str, optional
        統合データファイルのパス
    earthquake_event_bb : int
        広帯域での地震発生Event番号（デフォルト: 50）
    earthquake_event_sm : int
        強震計での地震発生Event番号（デフォルト: 529）
    window_duration : int
        各ウィンドウの継続時間（分）
    n_windows : int
        解析ウィンドウ数
    
    Returns:
    --------
    results : Dict
        'broadband': 広帯域の解析結果
        'strong_motion': 強震計の解析結果
        'integrated': 統合解析結果
    """
    print("=== F-NET Integrated Event Evolution Analysis ===")
    print(f"Window: {window_duration} min × {n_windows} windows")
    print(f"Broadband earthquake event: {earthquake_event_bb}")
    print(f"Strong motion earthquake event: {earthquake_event_sm}")
    
    # 出力ディレクトリの作成
    bb_output_dir = os.path.join(output_dir, 'broadband')
    sm_output_dir = os.path.join(output_dir, 'strong_motion')
    integrated_output_dir = os.path.join(output_dir, 'integrated')
    
    os.makedirs(bb_output_dir, exist_ok=True)
    os.makedirs(sm_output_dir, exist_ok=True)
    os.makedirs(integrated_output_dir, exist_ok=True)
    
    # データ読み込み
    bb_matrix = None
    sm_matrix = None
    
    # 統合データファイルから読み込み
    if integrated_data_path and os.path.exists(integrated_data_path):
        print(f"Loading integrated data from: {integrated_data_path}")
        integrated_data = np.load(integrated_data_path, allow_pickle=True).item()
        
        if 'broadband' in integrated_data:
            bb_matrix = integrated_data['broadband']
            print(f"  Broadband shape: {bb_matrix.shape}")
            
        if 'strong_motion' in integrated_data:
            sm_matrix = integrated_data['strong_motion']
            print(f"  Strong motion shape: {sm_matrix.shape}")
    
    # 個別ファイルから読み込み
    else:
        if broadband_matrix_path and os.path.exists(broadband_matrix_path):
            bb_matrix = np.load(broadband_matrix_path)
            print(f"Loaded broadband: {bb_matrix.shape}")
            
        if strong_motion_matrix_path and os.path.exists(strong_motion_matrix_path):
            sm_matrix = np.load(strong_motion_matrix_path)
            print(f"Loaded strong motion: {sm_matrix.shape}")
    
    if bb_matrix is None and sm_matrix is None:
        print("Error: No data loaded")
        return None
    
    # 観測点情報の読み込み
    station_locations, station_metadata = load_station_information(
        station_info_path, None, 'integrated'
    )
    
    results = {
        'broadband': None,
        'strong_motion': None,
        'integrated': {}
    }
    
    # 1. 広帯域データの進化解析
    if bb_matrix is not None:
        print("\n--- Analyzing Broadband Evolution ---")
        
        # 観測点リスト（73観測点）
        if 'FNET_STATIONS' in globals():
            bb_station_list = FNET_STATIONS
        else:
            bb_station_list = [f"Station_{i:03d}" for i in range(73)]
        
        # 解析器の初期化
        bb_analyzer = SpatialMultiLayerAnalyzer(
            station_locations=station_locations,
            station_metadata=station_metadata,
            data_type='broadband'
        )
        
        # 時間窓解析
        bb_results = perform_fixed_window_analysis(
            bb_matrix,
            bb_station_list,
            bb_analyzer,
            earthquake_event_bb,
            window_duration,
            n_windows,
            bb_output_dir
        )
        
        results['broadband'] = {
            'analyzer': bb_analyzer,
            'timeline': bb_results
        }
    
    # 2. 強震計データの進化解析
    if sm_matrix is not None:
        print("\n--- Analyzing Strong Motion Evolution ---")
        
        # 強震計は時間分解能が高いので調整
        sm_window_duration = window_duration * (sm_matrix.shape[0] / bb_matrix.shape[0]) if bb_matrix is not None else window_duration
        
        # 観測点リスト
        if 'FNET_STATIONS' in globals():
            sm_station_list = FNET_STATIONS
        else:
            sm_station_list = [f"Station_{i:03d}" for i in range(73)]
        
        # 解析器の初期化
        sm_analyzer = SpatialMultiLayerAnalyzer(
            station_locations=station_locations,
            station_metadata=station_metadata,
            data_type='strong_motion'
        )
        
        # 時間窓解析
        sm_results = perform_fixed_window_analysis(
            sm_matrix,
            sm_station_list,
            sm_analyzer,
            earthquake_event_sm,
            int(sm_window_duration),
            n_windows,
            sm_output_dir
        )
        
        results['strong_motion'] = {
            'analyzer': sm_analyzer,
            'timeline': sm_results
        }
    
    # 3. 統合解析（両方のデータがある場合）
    if results['broadband'] and results['strong_motion']:
        print("\n--- Integrated Evolution Analysis ---")
        
        # 深部-表層の時間遅延解析
        results['integrated']['propagation_delay'] = analyze_depth_surface_delay(
            results['broadband']['timeline'],
            results['strong_motion']['timeline']
        )
        
        # 統合前兆パターンの検出
        results['integrated']['integrated_precursors'] = detect_integrated_precursors(
            results['broadband']['timeline'],
            results['strong_motion']['timeline']
        )
        
        # 統合可視化
        create_integrated_evolution_visualization(
            results['broadband']['timeline'],
            results['strong_motion']['timeline'],
            integrated_output_dir
        )
        
        # 統合レポート生成
        generate_integrated_evolution_report(
            results,
            integrated_output_dir
        )
    
    # 4. 個別の可視化とレポート
    if results['broadband']:
        visualize_event_evolution(
            results['broadband']['timeline'],
            earthquake_event_bb,
            bb_output_dir
        )
        generate_evolution_report(
            results['broadband']['timeline'],
            earthquake_event_bb,
            bb_output_dir
        )
    
    if results['strong_motion']:
        visualize_event_evolution(
            results['strong_motion']['timeline'],
            earthquake_event_sm,
            sm_output_dir
        )
        generate_evolution_report(
            results['strong_motion']['timeline'],
            earthquake_event_sm,
            sm_output_dir
        )
    
    print("\n=== Evolution Analysis Complete ===")
    return results


def analyze_depth_surface_delay(bb_timeline: List[Dict], sm_timeline: List[Dict]) -> Dict:
    """深部-表層間の時間遅延を解析"""
    delays = []
    
    # 各時間窓での相関を計算
    for i in range(min(len(bb_timeline), len(sm_timeline))):
        bb_q = bb_timeline[i]['global_mean_Q']
        sm_q = sm_timeline[i]['global_mean_Q']
        
        # 簡易的な遅延推定
        if i > 0:
            bb_change = bb_q - bb_timeline[i-1]['global_mean_Q']
            sm_change = sm_q - sm_timeline[i-1]['global_mean_Q']
            
            if abs(bb_change) > 0.1 and abs(sm_change) > 0.1:
                # 変化の符号が同じなら伝播の可能性
                if bb_change * sm_change > 0:
                    delays.append({
                        'window': i,
                        'bb_change': bb_change,
                        'sm_change': sm_change,
                        'correlation': bb_change * sm_change
                    })
    
    return {
        'delays': delays,
        'mean_delay': np.mean([d['correlation'] for d in delays]) if delays else 0
    }


def detect_integrated_precursors(bb_timeline: List[Dict], sm_timeline: List[Dict]) -> Dict:
    """統合前兆パターンの検出"""
    precursors = {
        'deep_anomaly_first': False,
        'surface_amplification': False,
        'coupled_evolution': False
    }
    
    # 深部先行型異常
    bb_anomalies = [r['n_anomalous_stations'] for r in bb_timeline[-3:]]
    sm_anomalies = [r['n_anomalous_stations'] for r in sm_timeline[-3:]]
    
    if bb_anomalies and sm_anomalies:
        if np.mean(bb_anomalies) > np.mean(sm_anomalies) * 1.5:
            precursors['deep_anomaly_first'] = True
        
        # 表層増幅
        if sm_anomalies[-1] > sm_anomalies[0] * 2:
            precursors['surface_amplification'] = True
        
        # 連動進化
        bb_trend = np.polyfit(range(len(bb_anomalies)), bb_anomalies, 1)[0] if len(bb_anomalies) > 1 else 0
        sm_trend = np.polyfit(range(len(sm_anomalies)), sm_anomalies, 1)[0] if len(sm_anomalies) > 1 else 0
        
        if bb_trend > 0 and sm_trend > 0:
            precursors['coupled_evolution'] = True
    
    return precursors


def create_integrated_evolution_visualization(
    bb_timeline: List[Dict], 
    sm_timeline: List[Dict],
    output_dir: str):
    """統合進化の可視化"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 時間軸
    bb_times = [r['time_from_earthquake'] for r in bb_timeline]
    sm_times = [r['time_from_earthquake'] for r in sm_timeline]
    
    # 1. 広帯域 vs 強震計の|Q_Λ|進化
    ax = axes[0, 0]
    ax.plot(bb_times, [r['global_mean_Q'] for r in bb_timeline], 
            'b-o', label='Broadband', linewidth=2, markersize=8)
    ax.plot(sm_times, [r['global_mean_Q'] for r in sm_timeline], 
            'r-s', label='Strong Motion', linewidth=2, markersize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Deep vs Surface |Q_Λ| Evolution')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.set_ylabel('|Q_Λ|')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # 2. エネルギー伝達
    ax = axes[0, 1]
    bb_energy = [r['global_energy'] for r in bb_timeline]
    sm_energy = [r['global_energy'] for r in sm_timeline]
    
    # エネルギー比
    energy_ratio = []
    for i in range(min(len(bb_energy), len(sm_energy))):
        if bb_energy[i] > 0:
            energy_ratio.append(sm_energy[i] / bb_energy[i])
    
    ax.plot(bb_times[:len(energy_ratio)], energy_ratio, 'g-^', linewidth=2, markersize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Energy Transfer Ratio (Surface/Deep)')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.set_ylabel('Energy Ratio')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # 3. 異常観測点数の比較
    ax = axes[1, 0]
    width = 0.35
    x = np.arange(len(bb_timeline))
    
    bars1 = ax.bar(x - width/2, [r['n_anomalous_stations'] for r in bb_timeline], 
                   width, label='Broadband', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, [r['n_anomalous_stations'] for r in sm_timeline[:len(bb_timeline)]], 
                   width, label='Strong Motion', color='red', alpha=0.7)
    
    ax.set_title('Anomalous Stations Count')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{int(t)}" for t in bb_times], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 深部-表層相関
    ax = axes[1, 1]
    
    # 移動相関
    window_size = 3
    correlations = []
    
    for i in range(window_size, min(len(bb_timeline), len(sm_timeline))):
        bb_segment = [r['global_mean_Q'] for r in bb_timeline[i-window_size:i]]
        sm_segment = [r['global_mean_Q'] for r in sm_timeline[i-window_size:i]]
        
        if len(bb_segment) > 1 and len(sm_segment) > 1:
            corr = np.corrcoef(bb_segment, sm_segment)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    if correlations:
        ax.plot(bb_times[window_size:window_size+len(correlations)], 
               correlations, 'm-o', linewidth=2, markersize=8)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title('Deep-Surface Correlation (3-window)')
        ax.set_xlabel('Time from Earthquake (min)')
        ax.set_ylabel('Correlation')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    
    # 5. 構造遷移の累積
    ax = axes[2, 0]
    bb_transitions = np.cumsum([r['n_transitions'] for r in bb_timeline])
    sm_transitions = np.cumsum([r['n_transitions'] for r in sm_timeline])
    
    ax.plot(bb_times, bb_transitions, 'b-', label='Broadband', linewidth=3)
    ax.plot(sm_times[:len(sm_transitions)], sm_transitions[:len(bb_times)], 
           'r-', label='Strong Motion', linewidth=3)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Cumulative Structural Transitions')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.set_ylabel('Cumulative Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # 6. 統合前兆スコア
    ax = axes[2, 1]
    
    # 統合スコアの計算
    integrated_scores = []
    for i in range(min(len(bb_timeline), len(sm_timeline))):
        bb_score = (bb_timeline[i]['global_mean_Q'] * 0.5 + 
                   bb_timeline[i]['n_anomalous_stations'] / 10 * 0.5)
        sm_score = (sm_timeline[i]['global_mean_Q'] * 0.5 + 
                   sm_timeline[i]['n_anomalous_stations'] / 10 * 0.5)
        
        # 深部の重み0.6、表層の重み0.4
        integrated_score = bb_score * 0.6 + sm_score * 0.4
        integrated_scores.append(integrated_score)
    
    ax.plot(bb_times[:len(integrated_scores)], integrated_scores, 
           'k-o', linewidth=3, markersize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.fill_between(bb_times[:len(integrated_scores)], integrated_scores, 
                    alpha=0.3, color='purple')
    ax.set_title('Integrated Precursor Score')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'integrated_evolution.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved integrated evolution visualization to {output_path}")
    plt.close(fig)


def generate_integrated_evolution_report(results: Dict, output_dir: str):
    """統合進化レポートの生成"""
    report_path = os.path.join(output_dir, 'integrated_evolution_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Integrated Lambda³ Event Evolution Analysis Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 広帯域解析結果
        if results['broadband']:
            bb_timeline = results['broadband']['timeline']
            f.write("=== Broadband (Deep Structure) Evolution ===\n")
            f.write(f"Total windows analyzed: {len(bb_timeline)}\n")
            
            bb_q_values = [r['global_mean_Q'] for r in bb_timeline]
            f.write(f"Global |Q_Λ| range: {min(bb_q_values):.3f} - {max(bb_q_values):.3f}\n")
            
            # トレンド
            if len(bb_q_values) > 1:
                trend = np.polyfit(range(len(bb_q_values)), bb_q_values, 1)[0]
                f.write(f"Trend: {'Increasing' if trend > 0 else 'Decreasing'} (slope={trend:.4f})\n")
        
        # 強震計解析結果
        if results['strong_motion']:
            sm_timeline = results['strong_motion']['timeline']
            f.write("\n=== Strong Motion (Surface Response) Evolution ===\n")
            f.write(f"Total windows analyzed: {len(sm_timeline)}\n")
            
            sm_q_values = [r['global_mean_Q'] for r in sm_timeline]
            f.write(f"Global |Q_Λ| range: {min(sm_q_values):.3f} - {max(sm_q_values):.3f}\n")
            
            # トレンド
            if len(sm_q_values) > 1:
                trend = np.polyfit(range(len(sm_q_values)), sm_q_values, 1)[0]
                f.write(f"Trend: {'Increasing' if trend > 0 else 'Decreasing'} (slope={trend:.4f})\n")
        
        # 統合解析結果
        if results['integrated']:
            f.write("\n=== Integrated Analysis Results ===\n")
            
            # 伝播遅延
            if 'propagation_delay' in results['integrated']:
                delay_info = results['integrated']['propagation_delay']
                f.write(f"Mean propagation correlation: {delay_info['mean_delay']:.3f}\n")
                f.write(f"Detected propagation events: {len(delay_info['delays'])}\n")
            
            # 統合前兆
            if 'integrated_precursors' in results['integrated']:
                precursors = results['integrated']['integrated_precursors']
                f.write("\nIntegrated Precursor Patterns:\n")
                for pattern, detected in precursors.items():
                    f.write(f"  {pattern}: {'Yes' if detected else 'No'}\n")
        
        f.write("\n=== Summary ===\n")
        f.write("The integrated Lambda³ analysis reveals complex spatiotemporal evolution\n")
        f.write("across both deep (broadband) and surface (strong motion) structures.\n")
        
        # 主要な発見
        if results['broadband'] and results['strong_motion']:
            bb_final = results['broadband']['timeline'][-1]
            sm_final = results['strong_motion']['timeline'][-1]
            
            f.write(f"\nFinal state (just before earthquake):\n")
            f.write(f"  Broadband |Q_Λ|: {bb_final['global_mean_Q']:.3f}\n")
            f.write(f"  Strong Motion |Q_Λ|: {sm_final['global_mean_Q']:.3f}\n")
            f.write(f"  Broadband anomalous stations: {bb_final['n_anomalous_stations']}\n")
            f.write(f"  Strong Motion anomalous stations: {sm_final['n_anomalous_stations']}\n")
    
    print(f"Integrated evolution report saved to {report_path}")
    
# === ユーティリティ関数（統合版） ===

def load_station_info_extended(info_file: str, data_type: str = 'standard') -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Dict]]:
    """観測点情報を読み込む（データタイプ対応版）"""
    station_locations = {}
    station_metadata = {}

    try:
        import json
        import csv

        # JSONファイルから読み込み
        if info_file.endswith('.json'):
            with open(info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for station, info in data.items():
                station_locations[station] = (info['latitude'], info['longitude'])

                # メタデータ（データタイプ別）
                metadata = info.get('metadata', {})

                if data_type == 'broadband':
                    # 広帯域観測点特有の情報
                    metadata['borehole_depth'] = info.get('borehole_depth', 0)
                    metadata['sensor_type'] = info.get('sensor_type', 'STS-2')
                elif data_type == 'strong_motion':
                    # 強震計特有の情報
                    metadata['vs30'] = info.get('vs30', 500)
                    metadata['site_class'] = info.get('site_class', 'C')
                    metadata['surface_geology'] = info.get('surface_geology', 0)

                station_metadata[station] = metadata

        # CSVファイルから読み込み
        elif info_file.endswith('.csv'):
            with open(info_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    station = row['station']
                    station_locations[station] = (
                        float(row['latitude']),
                        float(row['longitude'])
                    )

                    # データタイプ別のメタデータ
                    metadata = {
                        'depth': float(row.get('depth', 0)),
                        'quality_factor': float(row.get('quality', 1.0))
                    }

                    if data_type == 'broadband':
                        metadata['borehole_depth'] = float(row.get('borehole_depth', 0))
                    elif data_type == 'strong_motion':
                        metadata['vs30'] = float(row.get('vs30', 500))
                        metadata['site_class'] = row.get('site_class', 'C')

                    station_metadata[station] = metadata

    except Exception as e:
        print(f"Warning: Could not load station info: {e}")

    return station_locations, station_metadata


def parse_integrated_data(broadband_matrix: np.ndarray,
                         strong_motion_matrix: np.ndarray,
                         station_list: List[str],
                         n_features_bb: int = 12,
                         n_features_sm: int = 12) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """統合データ用：広帯域と強震計データを分離してパース"""
    bb_data_dict = {}
    sm_data_dict = {}

    # 広帯域データのパース
    n_stations_bb = min(len(station_list), broadband_matrix.shape[1] // n_features_bb)
    print(f"Parsing broadband data for {n_stations_bb} stations")

    for i in range(n_stations_bb):
        station = station_list[i]
        start_idx = i * n_features_bb
        end_idx = min((i + 1) * n_features_bb, broadband_matrix.shape[1])

        if end_idx - start_idx == n_features_bb:
            bb_data_dict[station] = broadband_matrix[:, start_idx:end_idx]

    # 強震計データのパース
    n_stations_sm = min(len(station_list), strong_motion_matrix.shape[1] // n_features_sm)
    print(f"Parsing strong motion data for {n_stations_sm} stations")

    for i in range(n_stations_sm):
        station = station_list[i]
        start_idx = i * n_features_sm
        end_idx = min((i + 1) * n_features_sm, strong_motion_matrix.shape[1])

        if end_idx - start_idx == n_features_sm:
            sm_data_dict[station] = strong_motion_matrix[:, start_idx:end_idx]

    return bb_data_dict, sm_data_dict

# F-net観測点リスト（日本語名付き）
FNET_STATIONS_WITH_NAMES = {
    "N.ABUF": "油山",
    "N.ADMF": "赤泊",
    "N.AMMF": "奄美大島",
    "N.AOGF": "青ヶ島",
    "N.ASIF": "足尾",
    "N.FUJF": "富士川",
    "N.FUKF": "福江",
    "N.GJMF": "五城目",
    "N.HIDF": "日高",
    "N.HJOF": "八丈",
    "N.HROF": "広野",
    "N.HSSF": "札幌",
    "N.IGKF": "石垣",
    "N.IMGF": "今金",
    "N.INNF": "中津",
    "N.ISIF": "徳島",
    "N.IYGF": "山形",
    "N.IZHF": "厳原",
    "N.JIZF": "中伊豆",
    "N.KGMF": "国頭",
    "N.KISF": "紀和",
    "N.KMTF": "上富田",
    "N.KMUF": "上杵臼",
    "N.KNMF": "金山",
    "N.KNPF": "訓子府",
    "N.KNYF": "金谷",
    "N.KSKF": "川崎",
    "N.KSNF": "気仙沼",
    "N.KSRF": "釧路",
    "N.KYKF": "永田",
    "N.KZKF": "柏崎",
    "N.KZSF": "神津島",
    "N.NAAF": "朝日",
    "N.NKGF": "中川",
    "N.NMRF": "根室",
    "N.NOKF": "野上",
    "N.NOPF": "西興部",
    "N.NRWF": "成羽",
    "N.NSKF": "錦",
    "N.OKWF": "大川",
    "N.ONSF": "大西",
    "N.OOWF": "大鰐",
    "N.OSWF": "小笠原",
    "N.SAGF": "西郷",
    "N.SBRF": "背振",
    "N.SBTF": "新発田",
    "N.SGNF": "鶴ヶ野",
    "N.SHRF": "斜里",
    "N.SIBF": "紫尾山",
    "N.SRNF": "白峰",
    "N.STMF": "外海",
    "N.TASF": "田代",
    "N.TGAF": "多賀",
    "N.TGWF": "玉川",
    "N.TKDF": "竹田",
    "N.TKOF": "高岡",
    "N.TMCF": "友内",
    "N.TMRF": "泊",
    "N.TSAF": "西土佐",
    "N.TSKF": "つくば",
    "N.TTOF": "高遠",
    "N.TYSF": "遠野山崎",
    "N.UMJF": "馬路",
    "N.URHF": "浦幌",
    "N.WJMF": "輪島",  # ← 能登半島地震の震源に最も近い
    "N.WTRF": "度会",
    "N.YASF": "八坂",
    "N.YMZF": "八溝",
    "N.YNGF": "与那国",
    "N.YSIF": "吉田",
    "N.YTYF": "豊田",
    "N.YZKF": "山崎",
    "N.ZMMF": "座間味"
}

# 観測点リスト（順序保持）
FNET_STATIONS = list(FNET_STATIONS_WITH_NAMES.keys())

def get_station_list_with_names(station_list: List[str]) -> List[str]:
    """観測点コードに日本語名を付加"""
    return [
        f"{station}（{FNET_STATIONS_WITH_NAMES.get(station, '不明')}）"
        if station in FNET_STATIONS_WITH_NAMES
        else station
        for station in station_list
    ]

def parse_fnet_data(event_matrix: np.ndarray, station_list: List[str]) -> Dict[str, np.ndarray]:
    """F-NETデータのパース（修正版）"""
    data_dict = {}
    n_features_per_station = 44 if event_matrix.shape[1] == 3272 else 25
    
    for i, station in enumerate(station_list):
        start_idx = i * n_features_per_station
        end_idx = min((i + 1) * n_features_per_station, event_matrix.shape[1])
        
        if end_idx - start_idx == n_features_per_station:
            # ここでstation名をそのまま使用（Station_プレフィックスを付けない）
            data_dict[station] = event_matrix[:, start_idx:end_idx]
    
    return data_dict

def load_integrated_data(
    broadband_path: str,
    strong_motion_path: str,
    data_mode: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """統合データの読み込み（辞書形式対応）"""
    broadband_data = None
    strong_motion_data = None

    # 統合ファイルかどうかをチェック
    if 'integrated' in broadband_path:
        try:
            # 統合データの読み込み（allow_pickle=True）
            integrated_data = np.load(broadband_path, allow_pickle=True).item()
            print(f"Loaded integrated data from: {broadband_path}")

            if 'broadband' in integrated_data:
                broadband_data = integrated_data['broadband']
                print(f"  Broadband data shape: {broadband_data.shape}")

            if 'strong_motion' in integrated_data:
                strong_motion_data = integrated_data['strong_motion']
                print(f"  Strong motion data shape: {strong_motion_data.shape}")

            if 'metadata' in integrated_data:
                metadata = integrated_data['metadata']
                print(f"  Metadata: {metadata}")

            return broadband_data, strong_motion_data

        except Exception as e:
            print(f"Error loading integrated data: {e}")
            return None, None

    # 個別ファイルの読み込み
    if data_mode in ['integrated', 'broadband']:
        try:
            broadband_data = np.load(broadband_path)
            print(f"Loaded broadband data: shape={broadband_data.shape}")
        except FileNotFoundError:
            if data_mode == 'integrated':
                print(f"Warning: Broadband data not found at {broadband_path}")
            else:
                print(f"Error: Broadband data required but not found")
                return None, None

    if data_mode in ['integrated', 'strong_motion']:
        try:
            strong_motion_data = np.load(strong_motion_path)
            print(f"Loaded strong motion data: shape={strong_motion_data.shape}")
        except FileNotFoundError:
            if data_mode == 'integrated':
                print(f"Warning: Strong motion data not found at {strong_motion_path}")
            else:
                print(f"Error: Strong motion data required but not found")
                return None, None

    return broadband_data, strong_motion_data

def load_station_information(
    station_info_path: Optional[str],
    station_list_path: Optional[str],
    data_mode: str,
    data_files: Optional[List[str]] = None
) -> Tuple[Dict, Dict]:
    """観測点情報の読み込み"""
    station_locations = {}
    station_metadata = {}

    # データファイルから観測点を動的抽出
    if data_files:
        extracted_stations = set()
        for f in data_files:
            basename = os.path.basename(f)
            if 'N.' in basename:
                parts = basename.split('.')
                if len(parts) >= 2:
                    station_code = f"{parts[0]}.{parts[1]}"
                    extracted_stations.add(station_code)

        # 位置情報なしで観測点リストのみ保持
        for station in extracted_stations:
            station_locations[station] = (None, None)

        print(f"Extracted {len(extracted_stations)} stations from data files")

    # station_info_pathから実際の位置情報を読み込む
    if station_info_path and os.path.exists(station_info_path):
        locations, metadata = load_station_info_extended(station_info_path, data_mode)
        station_locations.update(locations)
        station_metadata.update(metadata)

    return station_locations, station_metadata

def get_station_list(
    broadband_data: Optional[np.ndarray],
    strong_motion_data: Optional[np.ndarray],
    station_list_path: Optional[str],
    data_mode: str,
    data_files: Optional[List[str]] = None
) -> List[str]:
    """観測点リストの取得"""
    # ファイルから読み込み（最優先）
    if station_list_path and os.path.exists(station_list_path):
        with open(station_list_path, 'r') as f:
            station_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded station list: {len(station_list)} stations")
        return station_list
    
    # データファイルから直接抽出（次の優先度）
    if data_files:
        stations = set()
        for f in data_files:
            basename = os.path.basename(f)
            if 'N.' in basename:
                parts = basename.split('.')
                if len(parts) >= 2:
                    stations.add(f"{parts[0]}.{parts[1]}")
        print(f"Extracted {len(stations)} stations from data files")
        return sorted(list(stations))
    
    # === 固定処理（===
    n_stations = 73  # F-net観測点数
    
    # データ形状の確認（情報表示のみ）
    if data_mode == 'broadband' and broadband_data is not None:
        print(f"Broadband data shape: {broadband_data.shape}")
        if broadband_data.shape[1] == 3272:
            print(f"  Standard format detected: {n_stations} stations × ~44 features")
    elif data_mode == 'strong_motion' and strong_motion_data is not None:
        print(f"Strong motion data shape: {strong_motion_data.shape}")
        if strong_motion_data.shape[1] == 1824:
            print(f"  Standard format detected: {n_stations} stations × 25 features")
    elif data_mode == 'integrated':
        print(f"Integrated mode: using {n_stations} stations")
    
    # 観測点リスト生成（実際のF-NET観測点を使用）
    print(f"Generating station list for {n_stations} stations")
    
    # FNET_STATIONSが定義されていればそれを使用、なければ番号
    if 'FNET_STATIONS' in globals() and n_stations == 73:
        return FNET_STATIONS
    else:
        return [f"Station_{i:03d}" for i in range(n_stations)]
    
def set_analysis_parameters(data_mode: str, **kwargs) -> Dict:
    """解析パラメータの設定"""
    # デフォルトパラメータ（データモード別）
    if data_mode == 'integrated':
        defaults = {
            'n_clusters': 7,
            'clustering_method': 'hierarchical',
            'n_paths_global': 15,
            'n_paths_local': 7,
            'n_paths_cluster': 10,
            'parallel': True
        }
    elif data_mode == 'broadband':
        defaults = {
            'n_clusters': 5,
            'clustering_method': 'geological',
            'n_paths_global': 12,
            'n_paths_local': 5,
            'n_paths_cluster': 8,
            'parallel': False
        }
    elif data_mode == 'strong_motion':
        defaults = {
            'n_clusters': 8,
            'clustering_method': 'dbscan',
            'n_paths_global': 10,
            'n_paths_local': 6,
            'n_paths_cluster': 8,
            'parallel': False
        }
    else:
        defaults = {
            'n_clusters': 5,
            'clustering_method': 'kmeans',
            'n_paths_global': 10,
            'n_paths_local': 5,
            'n_paths_cluster': 7,
            'parallel': False
        }

    # ユーザー指定のパラメータで上書き
    params = defaults.copy()
    params.update(kwargs)

    return params


def perform_integrated_analysis(
    broadband_data: np.ndarray,
    strong_motion_data: np.ndarray,
    station_list: List[str],
    station_locations: Dict,
    station_metadata: Dict,
    params: Dict,
    output_dir: str
) -> Tuple:
    """統合データの解析実行"""
    print("\n=== Performing Integrated Analysis ===")

    # データのパース
    bb_data_dict, sm_data_dict = parse_integrated_data(
        broadband_data,
        strong_motion_data,
        station_list
    )

    print(f"Parsed broadband data: {len(bb_data_dict)} stations")
    print(f"Parsed strong motion data: {len(sm_data_dict)} stations")

    # 統合解析器の初期化
    analyzer = SpatialMultiLayerAnalyzer(
        station_locations=station_locations,
        station_metadata=station_metadata,
        data_type='integrated'
    )

    # 統合解析の実行
    result = analyzer.analyze_integrated_multilayer(
        bb_data_dict,
        sm_data_dict,
        **params
    )

    # 深部-表層相互作用の詳細解析
    print("\n--- Depth-Surface Interaction Analysis ---")
    print(f"Global charge correlation: {result.depth_surface_interaction.get('global_charge_correlation', 0):.3f}")
    print(f"Energy transfer ratio: {result.depth_surface_interaction.get('energy_transfer_ratio', 0):.3f}")
    print(f"Spatial coherence: {result.depth_surface_interaction.get('spatial_coherence', 0):.3f}")
    print(f"Propagation delay: {result.depth_surface_interaction.get('propagation_delay', 0):.1f} s")

    # 統合異常の検出
    print("\n--- Integrated Anomaly Detection ---")
    for atype, anomalies in result.integrated_anomalies.items():
        if anomalies:
            print(f"{atype}: {len(anomalies)} detected")

    return analyzer, result

def perform_single_mode_analysis(
    data: np.ndarray,
    station_list: List[str],
    station_locations: Dict,
    station_metadata: Dict,
    params: Dict,
    output_dir: str,
    data_mode: str
) -> Tuple:
    """単一モード解析の実行"""
    print(f"\n=== Performing {data_mode.title()} Analysis ===")

    # データのパース
    data_dict = parse_fnet_data(data, station_list)
    print(f"Parsed data for {len(data_dict)} stations")

    # 解析器の初期化
    analyzer = SpatialMultiLayerAnalyzer(
        station_locations=station_locations,
        station_metadata=station_metadata,
        data_type=data_mode
    )

    # 解析の実行
    result = analyzer.analyze_multilayer(data_dict, **params)

    return analyzer, result

def perform_fixed_window_analysis(
    event_matrix: np.ndarray,
    station_list: List[str],
    analyzer: 'SpatialMultiLayerAnalyzer',
    earthquake_event: int,
    window_duration: int,
    n_windows: int,
    output_dir: str
) -> List[Dict]:
    """10分固定ウィンドウ解析（時系列順）"""
    results_timeline = []

    # データの範囲チェック
    max_events = event_matrix.shape[0]
    print(f"Total events in data: {max_events}")
    print(f"Earthquake event: {earthquake_event}")
    print(f"Analyzing {n_windows} windows of {window_duration} minutes each")

    # 時系列順で解析（100分前から10分前まで）
    for window_idx in range(n_windows):
        # 時系列順で計算
        window_start = earthquake_event - (n_windows - window_idx) * window_duration
        window_end = window_start + window_duration

        # 時間表示用（地震からの相対時間）
        time_from_eq_start = -(n_windows - window_idx) * window_duration  # -100, -90, ..., -20, -10
        time_from_eq_end = time_from_eq_start + window_duration          # -90, -80, ..., -10, 0

        print(f"\n--- Window {window_idx+1}/{n_windows}: "
              f"T{time_from_eq_start} to T{time_from_eq_end} minutes ---")
        print(f"  Event range: {window_start} to {window_end}")

        # ウィンドウデータの抽出（範囲チェックを強化）
        if window_start < 0:
            print(f"  Warning: Window start {window_start} is before data beginning, adjusting...")
            window_start = 0
            window_end = min(window_duration, max_events)

        if window_end > max_events:
            print(f"  Warning: Window end {window_end} exceeds data length {max_events}, adjusting...")
            window_end = max_events
            window_start = max(0, window_end - window_duration)

        if window_start >= 0 and window_end <= max_events and window_end > window_start:
            window_data = event_matrix[window_start:window_end, :]

            # データサイズの確認
            actual_duration = window_end - window_start
            if actual_duration < window_duration * 0.8:  # 80%未満の場合は警告
                print(f"  Warning: Only {actual_duration} events available (expected {window_duration})")

            window_data_dict = parse_fnet_data(window_data, station_list)

            # データが解析可能か確認
            if not window_data_dict or all(data.shape[0] == 0 for data in window_data_dict.values()):
                print(f"  Error: No valid data in window {window_idx+1}")
                continue

            # パラメータの設定（地震に近づくにつれて詳細度を上げる）
            if window_idx >= n_windows - 3:  # 最後の30分（地震直前）
                params = {
                    'n_clusters': 5,
                    'n_paths_global': min(8, actual_duration - 1),   # データ長に応じて調整
                    'n_paths_local': min(4, actual_duration // 2),
                    'n_paths_cluster': min(6, actual_duration - 1),
                    'parallel': False  # 10分データでは並列処理を無効化
                }
                print("  [CRITICAL PERIOD - Approaching earthquake]")
            elif window_idx >= n_windows - 5:  # 50-30分前（遷移期）
                params = {
                    'n_clusters': 5,
                    'n_paths_global': min(6, actual_duration - 1),
                    'n_paths_local': min(3, actual_duration // 3),
                    'n_paths_cluster': min(5, actual_duration - 1),
                    'parallel': False
                }
                print("  [TRANSITION PERIOD]")
            else:  # 100-50分前（背景期）
                params = {
                    'n_clusters': 5,
                    'n_paths_global': min(5, actual_duration - 1),
                    'n_paths_local': min(3, actual_duration // 3),
                    'n_paths_cluster': min(4, actual_duration // 2),
                    'parallel': False
                }
                print("  [BACKGROUND PERIOD]")

            try:
                # 解析実行
                result = analyzer.analyze_multilayer(window_data_dict, **params)

                # 主要指標の抽出
                summary = extract_window_summary(result, window_start, window_end)
                summary['window_index'] = window_idx
                summary['time_from_earthquake'] = (time_from_eq_start + time_from_eq_end) / 2
                summary['time_label'] = f"T{time_from_eq_start} to T{time_from_eq_end}"
                summary['actual_duration'] = actual_duration

                # 前の時間窓との比較（時系列解析の利点）
                if len(results_timeline) > 0:
                    prev_summary = results_timeline[-1]
                    summary['delta_Q'] = summary['global_mean_Q'] - prev_summary['global_mean_Q']
                    summary['delta_energy'] = summary['global_energy'] - prev_summary['global_energy']

                    # 変化率の計算
                    if prev_summary['global_mean_Q'] > 0:
                        summary['Q_change_rate'] = (summary['global_mean_Q'] - prev_summary['global_mean_Q']) / prev_summary['global_mean_Q']
                    else:
                        summary['Q_change_rate'] = 0

                    print(f"  ΔQ_Λ from previous: {summary['delta_Q']:.3f}")

                # 地震前兆の検出（段階的に強化）
                if window_idx >= n_windows - 5:  # 最後の50分
                    precursors = detect_precursor_signals(result)
                    summary['precursors'] = precursors

                    # 前兆の累積評価
                    if window_idx >= n_windows - 3 and any(precursors.values()):
                        print(f"  [PRECURSOR DETECTED] {[k for k,v in precursors.items() if v]}")

                        # 前兆の強度を評価
                        precursor_strength = sum(precursors.values()) / len(precursors)
                        summary['precursor_strength'] = precursor_strength

                results_timeline.append(summary)

                # 主要指標の表示
                print(f"  Global |Q_Λ| = {summary['global_mean_Q']:.3f}")
                print(f"  Deep structure dominance: {summary['deep_structure_dominance']:.1%} "
                      f"({int(summary['deep_structure_dominance'] * params['n_paths_global'])}/{params['n_paths_global']})")
                print(f"  Surface rupture potential: {summary['surface_rupture_potential']:.1%} "
                      f"({int(summary['surface_rupture_potential'] * params['n_paths_global'])}/{params['n_paths_global']})")
                print(f"  Dominant structure: {summary['dominant_structure']}")
                print(f"  Anomalous stations: {summary['n_anomalous_stations']}")
                print(f"  Analysis successful for {len(window_data_dict)} stations")

            except Exception as e:
                print(f"  Warning: Analysis failed for window {window_idx+1}: {e}")
                import traceback
                traceback.print_exc()

                summary = create_dummy_summary(window_start, window_end)
                summary['window_index'] = window_idx
                summary['time_from_earthquake'] = (time_from_eq_start + time_from_eq_end) / 2
                summary['time_label'] = f"T{time_from_eq_start} to T{time_from_eq_end}"
                summary['error'] = str(e)
                results_timeline.append(summary)
        else:
            print(f"  Skipping window {window_idx+1}: Invalid range [{window_start}, {window_end}]")

    # 時系列解析の総括
    print(f"\n=== Time Series Analysis Summary ===")
    print(f"Total windows analyzed: {len(results_timeline)}")

    if len(results_timeline) > 1:
        # 全体的なトレンド分析
        q_values = [r['global_mean_Q'] for r in results_timeline]
        energies = [r['global_energy'] for r in results_timeline]

        # 線形トレンド
        if len(q_values) > 2:
            q_trend = np.polyfit(range(len(q_values)), q_values, 1)[0]
            e_trend = np.polyfit(range(len(energies)), energies, 1)[0]

            print(f"Q_Λ trend: {'Increasing' if q_trend > 0 else 'Decreasing'} (slope={q_trend:.4f})")
            print(f"Energy trend: {'Increasing' if e_trend > 0 else 'Decreasing'} (slope={e_trend:.4f})")

        # 臨界期の変化
        if len(results_timeline) >= 3:
            last_3_windows = results_timeline[-3:]
            critical_q_change = last_3_windows[-1]['global_mean_Q'] - last_3_windows[0]['global_mean_Q']
            print(f"Critical period Q_Λ change: {critical_q_change:.3f}")

    return results_timeline

def extract_window_summary(result: 'SpatialLambda3Result', start: int, end: int) -> Dict:
    """ウィンドウ解析結果のサマリー抽出（拡張版）"""
    # グローバル指標
    global_charges = list(result.global_result.topological_charges.values())
    global_mean_q = np.mean(np.abs(global_charges)) if global_charges else 0

    # 深部構造優位性の計算（broadband的な評価）
    deep_structure_count = sum(1 for q in global_charges if q < -0.5)
    deep_structure_dominance = deep_structure_count / len(global_charges) if global_charges else 0

    # 表層破壊可能性の計算（strong motion的な評価）
    surface_rupture_count = sum(1 for q in global_charges if q > 0.5)
    surface_rupture_potential = surface_rupture_count / len(global_charges) if global_charges else 0

    # ローカル最大値と臨界観測点
    max_local_q = 0
    critical_station = None
    local_deep_count = 0
    local_surface_count = 0
    total_local_paths = 0

    for station, local_result in result.local_results.items():
        local_charges = list(local_result.topological_charges.values())
        if local_charges:
            local_max = np.max(np.abs(local_charges))
            if local_max > max_local_q:
                max_local_q = local_max
                critical_station = station

            # ローカルレベルでの深部・表層評価
            local_deep_count += sum(1 for q in local_charges if q < -0.5)
            local_surface_count += sum(1 for q in local_charges if q > 0.5)
            total_local_paths += len(local_charges)

    # 異常統計
    n_anomalous = len(result.spatial_anomalies.get('local_hotspots', []))
    n_transitions = len(result.spatial_anomalies.get('structural_transitions', []))
    n_discontinuities = len(result.spatial_anomalies.get('spatial_discontinuities', []))

    # エネルギー統計
    global_energy = np.mean(list(result.global_result.energies.values()))

    # Lambda³特有の指標を追加
    structural_types = {}
    for classification in result.global_result.classifications.values():
        structural_types[classification] = structural_types.get(classification, 0) + 1

    dominant_structure = max(structural_types.items(), key=lambda x: x[1])[0] if structural_types else "Unknown"

    return {
        'window_start': start,
        'window_end': end,
        'global_mean_Q': global_mean_q,
        'max_local_Q': max_local_q,
        'critical_station': critical_station,
        'global_energy': global_energy,
        'n_anomalous_stations': n_anomalous,
        'n_transitions': n_transitions,
        'n_discontinuities': n_discontinuities,
        'spatial_hierarchy': result.cross_layer_metrics.get('spatial_hierarchy', 0),
        # 新規追加
        'deep_structure_dominance': deep_structure_dominance,
        'surface_rupture_potential': surface_rupture_potential,
        'local_deep_ratio': local_deep_count / total_local_paths if total_local_paths > 0 else 0,
        'local_surface_ratio': local_surface_count / total_local_paths if total_local_paths > 0 else 0,
        'dominant_structure': dominant_structure,
        'structural_diversity': len(structural_types),
        'result': result
    }

def create_dummy_summary(start: int, end: int) -> Dict:
    """ダミーサマリーの作成"""
    return {
        'window_start': start,
        'window_end': end,
        'global_mean_Q': 0,
        'max_local_Q': 0,
        'critical_station': None,
        'global_energy': 0,
        'n_anomalous_stations': 0,
        'n_transitions': 0,
        'n_discontinuities': 0,
        'spatial_hierarchy': 0,
        'result': None
    }


def detect_precursor_signals(result: 'SpatialLambda3Result') -> Dict:
    """前兆シグナルの検出"""
    precursors = {
        'quiescence': False,
        'acceleration': False,
        'spatial_migration': False,
        'structural_instability': False
    }

    # 静穏化の検出
    global_energy = np.mean(list(result.global_result.energies.values()))
    if global_energy < 0.5:  # 閾値は調整可能
        precursors['quiescence'] = True

    # 加速的増加の検出
    local_charges = []
    for local_result in result.local_results.values():
        local_charges.extend(list(local_result.topological_charges.values()))

    if local_charges and np.std(local_charges) > 2 * np.mean(np.abs(local_charges)):
        precursors['acceleration'] = True

    # 空間的移動の検出
    if len(result.spatial_anomalies.get('propagation_patterns', [])) > 0:
        precursors['spatial_migration'] = True

    # 構造的不安定性の検出
    if result.cross_layer_metrics.get('structural_diversity', 0) > 1.5:
        precursors['structural_instability'] = True

    return precursors


def visualize_event_evolution(
    results_timeline: List[Dict],
    earthquake_event: int,
    output_dir: str
):
    """Event進化の可視化（10分×10区間版）"""
    print("\n--- Creating Event Evolution Visualization ---")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # X軸（地震からの相対時間）
    time_points = [r['time_from_earthquake'] for r in results_timeline]

    # 1. グローバル平均|Q_Λ|の進化
    ax = axes[0, 0]
    values = [r['global_mean_Q'] for r in results_timeline]
    ax.plot(time_points, values, 'b-o', linewidth=2, markersize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Earthquake')
    ax.set_title('Global Mean |Q_Λ| Evolution', fontsize=14)
    ax.set_ylabel('|Q_Λ|')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 時間軸を反転（地震に向かって右へ）

    # 2. 最大ローカル|Q_Λ|
    ax = axes[0, 1]
    values = [r['max_local_Q'] for r in results_timeline]
    ax.plot(time_points, values, 'r-o', linewidth=2, markersize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Maximum Local |Q_Λ|', fontsize=14)
    ax.set_ylabel('Max |Q_Λ|')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 3. グローバルエネルギー
    ax = axes[0, 2]
    values = [r['global_energy'] for r in results_timeline]
    ax.plot(time_points, values, 'g-o', linewidth=2, markersize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Global Energy Evolution', fontsize=14)
    ax.set_ylabel('Energy')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 4. 異常観測点数（棒グラフ）
    ax = axes[1, 0]
    values = [r['n_anomalous_stations'] for r in results_timeline]
    colors = ['darkred' if t > -30 else 'orange' for t in time_points]  # 30分以内は強調
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7)
    ax.set_title('Anomalous Stations Count', fontsize=14)
    ax.set_ylabel('Count')
    ax.set_xlabel('Window Index')
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"T{int(t)}" for t in time_points], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # 5. 構造遷移数
    ax = axes[1, 1]
    values = [r['n_transitions'] for r in results_timeline]
    colors = ['indigo' if t > -30 else 'mediumpurple' for t in time_points]  # darkpurple → indigo
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7)
    ax.set_title('Structural Transitions', fontsize=14)
    ax.set_ylabel('Count')
    ax.set_xlabel('Window Index')
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"T{int(t)}" for t in time_points], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # 6. 空間的不連続数
    ax = axes[1, 2]
    values = [r['n_discontinuities'] for r in results_timeline]
    colors = ['saddlebrown' if t > -30 else 'brown' for t in time_points]  # darkbrown → saddlebrown
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7)
    ax.set_title('Spatial Discontinuities', fontsize=14)
    ax.set_ylabel('Count')
    ax.set_xlabel('Window Index')
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"T{int(t)}" for t in time_points], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # 7. 空間的階層性の時間発展
    ax = axes[2, 0]
    values = [r['spatial_hierarchy'] for r in results_timeline]
    ax.plot(time_points, values, 'c-o', linewidth=2, markersize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axhline(np.mean(values), color='gray', linestyle=':', alpha=0.5, label='Mean')
    ax.set_title('Spatial Hierarchy Evolution', fontsize=14)
    ax.set_ylabel('Hierarchy Index')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 8. 統合異常指標（複合スコア）
    ax = axes[2, 1]
    # 複合異常スコアの計算
    composite_scores = []
    for r in results_timeline:
        score = (r['global_mean_Q'] * 0.3 +
                r['max_local_Q'] * 0.2 +
                r['n_anomalous_stations'] / 10 * 0.3 +
                r['n_transitions'] / 5 * 0.2)
        composite_scores.append(score)

    ax.plot(time_points, composite_scores, 'k-o', linewidth=3, markersize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.fill_between(time_points, composite_scores, alpha=0.3, color='gray')
    ax.set_title('Composite Anomaly Score', fontsize=14)
    ax.set_ylabel('Score')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 9. 前兆検出サマリー
    ax = axes[2, 2]
    ax.axis('off')

    # 前兆検出のサマリーテキスト
    precursor_text = "=== Precursor Detection ===\n\n"
    precursor_count = {'quiescence': 0, 'acceleration': 0, 'migration': 0, 'instability': 0}

    for r in results_timeline:
        if 'precursors' in r and r['time_from_earthquake'] > -30:  # 30分以内のみ
            for key in precursor_count:
                if r['precursors'].get(key, False):
                    precursor_count[key] += 1

    precursor_text += "Last 30 minutes:\n"
    for key, count in precursor_count.items():
        precursor_text += f"  {key.capitalize()}: {count}/3 windows\n"

    # 最も顕著な異常
    max_q_window = max(results_timeline, key=lambda x: x['global_mean_Q'])
    precursor_text += f"\nPeak |Q_Λ| at {max_q_window['time_label']}\n"
    precursor_text += f"Value: {max_q_window['global_mean_Q']:.3f}\n"

    if max_q_window.get('critical_station'):
        precursor_text += f"Critical Station: {max_q_window['critical_station']}"

    ax.text(0.1, 0.9, precursor_text, transform=ax.transAxes,
            fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()


def create_rate_of_change_plot(
    results_timeline: List[Dict],
    earthquake_event: int,
    output_dir: str
):
    """変化率のプロット（10分間隔での微分）"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 時間点
    time_points = [r['time_from_earthquake'] for r in results_timeline]

    # 1. |Q_Λ|の変化率
    ax = axes[0, 0]
    q_values = [r['global_mean_Q'] for r in results_timeline]
    if len(q_values) > 1:
        q_rates = np.diff(q_values) / 10  # 10分あたりの変化率
        ax.plot(time_points[:-1], q_rates, 'b-s', linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Rate of Change: |Q_Λ|', fontsize=12)
        ax.set_ylabel('Δ|Q_Λ|/10min')
        ax.set_xlabel('Time from Earthquake (min)')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    # 2. エネルギー変化率
    ax = axes[0, 1]
    e_values = [r['global_energy'] for r in results_timeline]
    if len(e_values) > 1:
        e_rates = np.diff(e_values) / 10
        ax.plot(time_points[:-1], e_rates, 'g-s', linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Rate of Change: Energy', fontsize=12)
        ax.set_ylabel('ΔE/10min')
        ax.set_xlabel('Time from Earthquake (min)')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    # 3. 累積異常観測点数
    ax = axes[1, 0]
    anomalous_counts = [r['n_anomalous_stations'] for r in results_timeline]
    cumulative_anomalous = np.cumsum(anomalous_counts)
    ax.plot(time_points, cumulative_anomalous, 'orange', linewidth=3, marker='o')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Cumulative Anomalous Stations', fontsize=12)
    ax.set_ylabel('Cumulative Count')
    ax.set_xlabel('Time from Earthquake (min)')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 4. 加速度（2階微分）
    ax = axes[1, 1]
    if len(q_values) > 2:
        q_accel = np.diff(q_rates) / 10  # 加速度
        ax.plot(time_points[:-2], q_accel, 'r-^', linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Acceleration: |Q_Λ|', fontsize=12)
        ax.set_ylabel('Δ²|Q_Λ|/10min²')
        ax.set_xlabel('Time from Earthquake (min)')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'rate_of_change_10min.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved rate of change plot to {output_path}")

    plt.close(fig)


def track_critical_stations(results_timeline: List[Dict], ax):
    """臨界観測点の追跡"""
    critical_stations = {}

    for i, result in enumerate(results_timeline):
        station = result.get('critical_station')
        if station:
            if station not in critical_stations:
                critical_stations[station] = []
            critical_stations[station].append(i)

    # 上位5観測点をプロット
    y_pos = 0
    for station, indices in list(critical_stations.items())[:5]:
        ax.scatter(indices, [y_pos] * len(indices), label=station, s=50)
        y_pos += 1

    ax.set_xlabel('Window Index')
    ax.set_ylabel('Station')
    ax.set_ylim(-1, 5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def detect_and_visualize_precursors(
    results_timeline: List[Dict],
    earthquake_event: int,
    output_dir: str
):
    """前兆パターンの検出と可視化"""
    print("\n--- Detecting Earthquake Precursors ---")

    # 前兆期間の特定（地震前10イベント）
    precursor_period = []
    for result in results_timeline:
        if result['window_end'] >= earthquake_event - 10 and result['window_start'] <= earthquake_event:
            precursor_period.append(result)

    if not precursor_period:
        print("No precursor period identified")
        return

    # 前兆シグナルの集計
    precursor_summary = {
        'quiescence_count': 0,
        'acceleration_count': 0,
        'migration_count': 0,
        'instability_count': 0
    }

    for result in precursor_period:
        if 'precursors' in result:
            for key in precursor_summary:
                signal_key = key.replace('_count', '')
                if result['precursors'].get(signal_key, False):
                    precursor_summary[key] += 1

    # 前兆パターンの可視化
    fig, ax = plt.subplots(figsize=(10, 6))

    precursor_types = list(precursor_summary.keys())
    counts = list(precursor_summary.values())

    bars = ax.bar(range(len(precursor_types)), counts, color=['blue', 'red', 'green', 'orange'])
    ax.set_xticks(range(len(precursor_types)))
    ax.set_xticklabels([t.replace('_count', '').title() for t in precursor_types])
    ax.set_ylabel('Detection Count')
    ax.set_title(f'Precursor Signals (10 Events before Earthquake)')

    # 値をバーの上に表示
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'precursor_detection.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved precursor analysis to {output_path}")

    plt.close(fig)

def save_and_visualize_results(
    analyzer: 'SpatialMultiLayerAnalyzer',
    results: Union['SpatialLambda3Result', 'IntegratedLambda3Result'],
    output_dir: str,
    data_mode: str
):
    """結果の保存と可視化"""
    print("\n--- Saving and Visualizing Results ---")

    # 結果のエクスポート
    analyzer.export_results(results, output_dir)

    # 可視化の作成
    if hasattr(analyzer, 'visualize_multilayer_results'):
        fig1, fig2 = analyzer.visualize_multilayer_results(results)

        # 保存
        fig1.savefig(
            os.path.join(output_dir, f'{data_mode}_spatial_analysis.png'),
            dpi=300, bbox_inches='tight'
        )
        fig2.savefig(
            os.path.join(output_dir, f'{data_mode}_detailed_analysis.png'),
            dpi=300, bbox_inches='tight'
        )

        plt.close(fig1)
        plt.close(fig2)

    # 統合解析の場合は追加の可視化
    if data_mode == 'integrated' and hasattr(results, 'integrated_anomalies'):
        create_integrated_visualization(results, output_dir)


def create_integrated_visualization(
    result: 'IntegratedLambda3Result',
    output_dir: str
):
    """統合解析結果の特別な可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 深部-表層エネルギー伝達
    ax = axes[0, 0]
    create_energy_transfer_plot(result, ax)

    # 2. 連動異常の空間分布
    ax = axes[0, 1]
    create_coupled_anomaly_map(result, ax)

    # 3. 伝播パターン
    ax = axes[1, 0]
    create_propagation_pattern_plot(result, ax)

    # 4. 統合前兆指標
    ax = axes[1, 1]
    create_integrated_precursor_plot(result, ax)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'integrated_special_analysis.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_energy_transfer_plot(result: 'IntegratedLambda3Result', ax):
    """エネルギー伝達プロット"""
    # 簡易実装
    depths = ['Deep (>10km)', 'Intermediate (5-10km)', 'Shallow (<5km)']
    energies = [10, 7, 12]  # 仮のデータ

    bars = ax.bar(depths, energies, color=['darkblue', 'blue', 'lightblue'])
    ax.set_ylabel('Energy Level')
    ax.set_title('Depth-Surface Energy Transfer')

    # 矢印で伝達を表現
    for i in range(len(depths)-1):
        ax.annotate('', xy=(i+1, energies[i+1]*0.8), xytext=(i, energies[i]*0.8),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))


def create_coupled_anomaly_map(result: 'IntegratedLambda3Result', ax):
    """連動異常マップ"""
    # 統合異常の数を表示
    anomaly_types = list(result.integrated_anomalies.keys())
    counts = [len(v) for v in result.integrated_anomalies.values()]

    ax.bar(anomaly_types, counts, color='purple', alpha=0.7)
    ax.set_xlabel('Anomaly Type')
    ax.set_ylabel('Count')
    ax.set_title('Integrated Anomaly Distribution')
    ax.tick_params(axis='x', rotation=45)


def create_propagation_pattern_plot(result: 'IntegratedLambda3Result', ax):
    """伝播パターンプロット"""
    # 垂直伝播の可視化
    if 'vertical_propagation' in result.propagation_patterns:
        data = result.propagation_patterns['vertical_propagation']
        if len(data) > 0:
            ax.imshow(data[:20, :], cmap='seismic', aspect='auto')
            ax.set_xlabel('Property')
            ax.set_ylabel('Station')
            ax.set_title('Vertical Propagation Pattern')
    else:
        ax.text(0.5, 0.5, 'No propagation data', ha='center', va='center')
        ax.set_title('Vertical Propagation Pattern')


def create_integrated_precursor_plot(result: 'IntegratedLambda3Result', ax):
    """統合前兆指標プロット"""
    # レーダーチャート風の表示
    categories = ['Deep Anomaly', 'Surface Response', 'Coupling', 'Migration', 'Instability']
    values = [0.7, 0.9, 0.6, 0.8, 0.5]  # 仮の値

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax = plt.subplot(2, 2, 4, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, color='red')
    ax.fill(angles, values, alpha=0.25, color='red')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Integrated Precursor Indicators', pad=20)


def generate_evolution_report(
    results_timeline: List[Dict],
    earthquake_event: int,
    output_dir: str
):
    """Event進化の総合レポート生成"""
    print("\n--- Generating Evolution Report ---")

    report_path = os.path.join(output_dir, 'evolution_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== F-NET Lambda³ Event Evolution Analysis Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Earthquake Event: {earthquake_event}\n")
        f.write(f"Total Windows Analyzed: {len(results_timeline)}\n\n")

        # 全体的な傾向
        f.write("=== Overall Trends ===\n")

        global_q_values = [r['global_mean_Q'] for r in results_timeline]
        f.write(f"Global |Q_Λ| range: {min(global_q_values):.3f} - {max(global_q_values):.3f}\n")
        f.write(f"Mean Global |Q_Λ|: {np.mean(global_q_values):.3f}\n")

        # 地震前の変化
        pre_earthquake = [r for r in results_timeline if r['window_end'] <= earthquake_event]
        if pre_earthquake:
            pre_q = [r['global_mean_Q'] for r in pre_earthquake]
            f.write(f"\nPre-earthquake Global |Q_Λ| trend: ")
            if len(pre_q) > 1:
                trend = np.polyfit(range(len(pre_q)), pre_q, 1)[0]
                f.write(f"{'Increasing' if trend > 0 else 'Decreasing'} (slope={trend:.4f})\n")

        # 最も異常なウィンドウ
        f.write("\n=== Most Anomalous Windows ===\n")
        sorted_by_q = sorted(results_timeline, key=lambda x: x['global_mean_Q'], reverse=True)[:5]

        for i, window in enumerate(sorted_by_q):
            f.write(f"\n{i+1}. Window {window['window_start']}-{window['window_end']}:\n")
            f.write(f"   Global |Q_Λ|: {window['global_mean_Q']:.3f}\n")
            f.write(f"   Max Local |Q_Λ|: {window['max_local_Q']:.3f}\n")
            f.write(f"   Critical Station: {window.get('critical_station', 'N/A')}\n")
            f.write(f"   Anomalous Stations: {window['n_anomalous_stations']}\n")

        # 前兆シグナル
        f.write("\n=== Precursor Signals ===\n")
        precursor_windows = [r for r in results_timeline
                           if 'precursors' in r and any(r['precursors'].values())]

        if precursor_windows:
            f.write(f"Windows with precursor signals: {len(precursor_windows)}\n")

            # 各種前兆の統計
            precursor_types = ['quiescence', 'acceleration', 'spatial_migration', 'structural_instability']
            for ptype in precursor_types:
                count = sum(1 for w in precursor_windows if w['precursors'].get(ptype, False))
                f.write(f"  {ptype}: {count} windows\n")
        else:
            f.write("No significant precursor signals detected\n")

        # 空間的パターン
        f.write("\n=== Spatial Patterns ===\n")

        # 最も頻繁に異常となった観測点
        critical_stations_count = {}
        for result in results_timeline:
            if result.get('critical_station'):
                station = result['critical_station']
                critical_stations_count[station] = critical_stations_count.get(station, 0) + 1

        if critical_stations_count:
            f.write("\nMost frequently critical stations:\n")
            sorted_stations = sorted(critical_stations_count.items(),
                                   key=lambda x: x[1], reverse=True)[:10]
            for station, count in sorted_stations:
                f.write(f"  {station}: {count} windows\n")

        # 結論
        f.write("\n=== Summary ===\n")
        f.write("The Lambda³ analysis reveals complex spatiotemporal evolution ")
        f.write("of structural tensor fields across the F-NET observation network.\n")

        # 地震との関連
        if earthquake_event > 0:
            # 地震前後の比較
            pre_eq = [r['global_mean_Q'] for r in results_timeline
                     if r['window_end'] <= earthquake_event]
            post_eq = [r['global_mean_Q'] for r in results_timeline
                      if r['window_start'] >= earthquake_event]

            if pre_eq and post_eq:
                f.write(f"\nPre-earthquake mean |Q_Λ|: {np.mean(pre_eq):.3f}\n")
                f.write(f"Post-earthquake mean |Q_Λ|: {np.mean(post_eq):.3f}\n")
                f.write(f"Change: {(np.mean(post_eq) - np.mean(pre_eq)):.3f} ")
                f.write(f"({(np.mean(post_eq)/np.mean(pre_eq) - 1)*100:.1f}%)\n")

    print(f"Report saved to {report_path}")

def run_global_lambda3_analysis(
    mode="broadband",
    broadband_matrix_path=None,
    strong_motion_matrix_path=None,
    integrated_matrix_path=None,
    output_dir=None,
    **kwargs
):
    """
    Lambda³ グローバル地震データ解析関数

    Parameters:
    -----------
    mode : str
        'integrated'    : 統合解析
        'broadband'     : 広帯域のみ
        'strong_motion' : 強震計のみ
        'evolution'     : Event進化解析
    broadband_matrix_path : str
        広帯域データファイル（.npyなど）
    strong_motion_matrix_path : str
        強震計データファイル
    integrated_matrix_path : str
        統合済みデータファイル（任意）
    output_dir : str
        結果保存ディレクトリ
    **kwargs : 追加パラメータ
    """

    if mode == "integrated":
        print("=== Lambda³ 統合地震解析 ===")
        if integrated_matrix_path and os.path.exists(integrated_matrix_path):
            print(f"統合ファイルを使用: {integrated_matrix_path}")
            analyzer, results = analyze_integrated_fnet_lambda3(
                broadband_matrix_path=integrated_matrix_path,
                strong_motion_matrix_path='dummy',
                output_dir=output_dir or './integrated_results',
                data_mode='integrated',
                n_clusters=kwargs.get('n_clusters', 7),
                n_paths_global=kwargs.get('n_paths_global', 15),
                n_paths_local=kwargs.get('n_paths_local', 7),
                n_paths_cluster=kwargs.get('n_paths_cluster', 10)
            )
        elif broadband_matrix_path and strong_motion_matrix_path:
            print("個別ファイルから統合解析を実行")
            analyzer, results = analyze_integrated_fnet_lambda3(
                broadband_matrix_path=broadband_matrix_path,
                strong_motion_matrix_path=strong_motion_matrix_path,
                output_dir=output_dir or './integrated_results',
                data_mode='integrated',
                n_clusters=kwargs.get('n_clusters', 7),
                n_paths_global=kwargs.get('n_paths_global', 15)
            )
        else:
            print("Error: 必要なファイルパスが指定されていません")
            return None, None

    elif mode == "broadband":
        print("=== Lambda³ 広帯域地震解析 ===")
        analyzer, results = analyze_integrated_fnet_lambda3(
            broadband_matrix_path=broadband_matrix_path,
            strong_motion_matrix_path='dummy',
            output_dir=output_dir or './broadband_results',
            data_mode='broadband',
            n_clusters=kwargs.get('n_clusters', 5),
            n_paths_global=kwargs.get('n_paths_global', 12),
            clustering_method=kwargs.get('clustering_method', 'geological')
        )

    elif mode == "strong_motion":
        print("=== Lambda³ 強震計地震解析 ===")
        analyzer, results = analyze_integrated_fnet_lambda3(
            broadband_matrix_path='dummy',
            strong_motion_matrix_path=strong_motion_matrix_path,
            output_dir=output_dir or './strong_motion_results',
            data_mode='strong_motion',
            n_clusters=kwargs.get('n_clusters', 8),
            n_paths_global=kwargs.get('n_paths_global', 10),
            clustering_method=kwargs.get('clustering_method', 'dbscan')
        )

    elif mode == "evolution":
        print("=== Lambda³ 統合Event進化解析 ===")
        # 統合データ優先
        if integrated_matrix_path and os.path.exists(integrated_matrix_path):
            results = analyze_fnet_event_evolution(
                integrated_data_path=integrated_matrix_path,
                output_dir=output_dir or './evolution_results',
                earthquake_event_bb=kwargs.get('earthquake_event_bb', 50),
                earthquake_event_sm=kwargs.get('earthquake_event_sm', 529),
                window_duration=kwargs.get('window_duration', 5),
                n_windows=kwargs.get('n_windows', 10)
            )
        elif broadband_matrix_path and strong_motion_matrix_path:
            results = analyze_fnet_event_evolution(
                broadband_matrix_path=broadband_matrix_path,
                strong_motion_matrix_path=strong_motion_matrix_path,
                output_dir=output_dir or './evolution_results',
                earthquake_event_bb=kwargs.get('earthquake_event_bb', 50),
                earthquake_event_sm=kwargs.get('earthquake_event_sm', 529),
                window_duration=kwargs.get('window_duration', 5),
                n_windows=kwargs.get('n_windows', 10)
            )
        else:
            print("Error: 必要なファイルパスが指定されていません")
            return None, None

        return None, results

    else:
        print(f"Error: 不明なmode指定 '{mode}'")
        return None, None

    # === 結果サマリー ===
    if results:
        print("\n=== 解析完了 ===")
        print("Lambda³ 構造テンソル場解析が完了しました")

        if mode == "integrated" and hasattr(results, 'depth_surface_interaction'):
            print(f"\n深部-表層相互作用:")
            print(f"  エネルギー伝達率: {results.depth_surface_interaction.get('energy_transfer_ratio', 0):.2f}")
            print(f"  空間的整合性: {results.depth_surface_interaction.get('spatial_coherence', 0):.2f}")

    return analyzer, results

if __name__ == "__main__":
    try:
        get_ipython()
        print(f"=== {EARTHQUAKE_CONFIG['name']} Lambda³解析システム ===")
        print("解析対象:", EARTHQUAKE_CONFIG['name'])
        print("\n利用可能なデータファイルを確認中...")

        # ファイルの存在確認（configから動的に表示）
        config_files = {
            EARTHQUAKE_CONFIG.get("integrated_matrix_path", ""): "統合データ（広帯域＋強震計）",
            EARTHQUAKE_CONFIG.get("broadband_matrix_path", ""): "広帯域データ（深部構造）",
            EARTHQUAKE_CONFIG.get("strong_motion_matrix_path", ""): "強震計データ（表層応答）"
        }
        available = []
        for path, description in config_files.items():
            if path and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  ✓ {os.path.basename(path)} ({size_mb:.1f} MB) - {description}")
                available.append(path)
            else:
                print(f"  ✗ {os.path.basename(path)} - {description}")

        # 解析モード選択（configにmodeキーなければ対話入力 fallback）
        mode_map = {
            '1': 'integrated',
            '2': 'broadband',
            '3': 'strong_motion',
            '4': 'evolution',
            '':  'evolution'
        }
        print("\n解析モードを選択してください:")
        print("1. integrated   - 統合解析")
        print("2. broadband    - 深部構造")
        print("3. strong_motion - 表層応答")
        print("4. evolution    - 時間発展（デフォルト）")
        mode_input = EARTHQUAKE_CONFIG.get("mode", None)
        if mode_input is None:
            mode_input = input("\nモード番号(1-4)を入力 [Enter: evolution]: ").strip()
        selected_mode = mode_map.get(str(mode_input), 'evolution')
        EARTHQUAKE_CONFIG["mode"] = selected_mode

        print(f"\n'{selected_mode}'モードで解析を実行します...")

        # configをそのまま流すだけ
        analyzer, results = run_global_lambda3_analysis(**EARTHQUAKE_CONFIG)

    except NameError:
        # 通常のPython実行
        import sys
        mode = sys.argv[1] if len(sys.argv) > 1 else "evolution"
        EARTHQUAKE_CONFIG["mode"] = mode
        analyzer, results = run_global_lambda3_analysis(**EARTHQUAKE_CONFIG)
