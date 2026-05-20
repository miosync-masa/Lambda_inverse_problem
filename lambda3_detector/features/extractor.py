"""
Lambda³特徴量抽出（基本／高度／周波数強化版）。
"""

from typing import Dict, Optional

import numpy as np

from ..config import Lambda3Result
from ..core.pulsation_jit import compute_pulsation_energy_from_path


class Lambda3FeatureExtractor:
    """Lambda³特徴量抽出の統一インターフェース（周波数特徴強化版）"""

    def extract_basic_features(self, result: Lambda3Result, events: np.ndarray = None) -> Dict[str, np.ndarray]:
        """基本特徴量の抽出（更新版）"""
        n_paths = len(result.paths)
        paths_matrix = np.stack(list(result.paths.values()))

        # Lambda³コア物理量
        basic_features = {
            'Q_Λ': np.array([result.topological_charges[i] for i in range(n_paths)]),
            'E': np.array([result.energies[i] for i in range(n_paths)]),
            'σ_Q': np.array([result.stabilities[i] for i in range(n_paths)])
        }

        # エントロピー特徴
        for i in range(n_paths):
            ent = result.entropies[i]
            if isinstance(ent, dict):
                basic_features[f'S_shannon_{i}'] = np.array([ent.get('shannon', 0)])
                basic_features[f'S_renyi_{i}'] = np.array([ent.get('renyi_2', 0)])
                basic_features[f'S_tsallis_{i}'] = np.array([ent.get('tsallis_1.5', 0)])

        # 拍動特徴
        for i in range(n_paths):
            path = paths_matrix[i]
            jump_int, asymm, pulse_pow = compute_pulsation_energy_from_path(path)
            basic_features[f'jump_int_{i}'] = np.array([jump_int])
            basic_features[f'asymm_{i}'] = np.array([asymm])
            basic_features[f'pulse_pow_{i}'] = np.array([pulse_pow])

        # === 新規：パス空間での周波数特徴 ===
        for i in range(n_paths):
            path = paths_matrix[i]
            freq_features = self._extract_frequency_features_from_path(path)
            for fname, fval in freq_features.items():
                basic_features[f'{fname}_{i}'] = np.array([fval])

        return basic_features

    def extract_advanced_features(self, result: Lambda3Result, events: np.ndarray) -> Dict[str, np.ndarray]:
        """高度な特徴量の生成（周波数特徴強化版）"""
        # resultが辞書の場合はそのまま使用、Lambda3Resultの場合は属性にアクセス
        if isinstance(result, dict):
            features = result.copy()
            n_paths = max(int(k.split('_')[-1]) for k in features.keys()
                        if any(k.startswith(prefix) for prefix in ['S_shannon_', 'S_renyi_', 'S_tsallis_'])) + 1
            n_events = events.shape[0]

            if 'Q_Λ' in features:
                Qs = features['Q_Λ']
                Es = features.get('E', np.zeros(n_paths))
                Sigmas = features.get('σ_Q', np.ones(n_paths))
            else:
                Qs = np.zeros(n_paths)
                Es = np.zeros(n_paths)
                Sigmas = np.ones(n_paths)
        else:
            n_paths = len(result.paths)
            n_events = events.shape[0]
            paths_matrix = np.stack(list(result.paths.values()))
            Qs = np.array([result.topological_charges[i] for i in range(n_paths)])
            Es = np.array([result.energies[i] for i in range(n_paths)])
            Sigmas = np.array([result.stabilities[i] for i in range(n_paths)])

        # 2. 物理的に意味のある組み合わせ特徴
        features = {
            "Q_Λ": Qs,
            "σ_Q": Sigmas,
            "E": Es,
            "Q_Λ/σ_Q": Qs / (Sigmas + 1e-8),
            "Q_Λ×E": Qs * Es,
            "sq_Q_Λ": Qs ** 2,
            "E×σ_Q": Es * Sigmas,
            "sqrt_σ_Q": np.sqrt(Sigmas + 1e-8),
            "log_σ_Q": np.log(Sigmas + 1e-8),
            "sqrt_Q_Λ": np.sqrt(np.abs(Qs) + 1e-8),
            "log_Q_Λ": np.log(np.abs(Qs) + 1e-8),
        }

        # === 新規：イベント空間での周波数特徴 ===
        # 各イベント特徴量のFFT解析
        event_fft_features = self._extract_event_frequency_features(events)
        for fname, fvals in event_fft_features.items():
            features[f'event_{fname}'] = fvals

        # 3. エントロピー、拍動、統計特徴（パスごと）
        if hasattr(result, 'paths'):
            paths_matrix = np.stack(list(result.paths.values()))

            # === 新規：パスごとの周波数特徴 ===
            path_freq_features = {}
            for i in range(n_paths):
                path = paths_matrix[i]
                freq_feats = self._extract_frequency_features_from_path(path)
                for fname, fval in freq_feats.items():
                    if fname not in path_freq_features:
                        path_freq_features[fname] = []
                    path_freq_features[fname].append(fval)

            # パス周波数特徴の統計量
            for fname, fvals in path_freq_features.items():
                features[f'path_{fname}_mean'] = np.array([np.mean(fvals)])
                features[f'path_{fname}_std'] = np.array([np.std(fvals)])
                features[f'path_{fname}_max'] = np.array([np.max(fvals)])

            for i in range(n_paths):
                # 既存のエントロピー特徴
                ent = result.entropies[i]
                if isinstance(ent, dict):
                    features[f'S_shannon_{i}'] = np.array([ent.get('shannon', 0)])
                    features[f'S_renyi_{i}'] = np.array([ent.get('renyi_2', 0)])
                    features[f'S_tsallis_{i}'] = np.array([ent.get('tsallis_1.5', 0)])

                # 既存の拍動エネルギー
                path = paths_matrix[i]
                jump_int, asymm, pulse_pow = compute_pulsation_energy_from_path(path)
                features[f'jump_int_{i}'] = np.array([jump_int])
                features[f'asymm_{i}'] = np.array([asymm])
                features[f'pulse_pow_{i}'] = np.array([pulse_pow])

                # 歪度と尖度
                mean_path = np.mean(path)
                std_path = np.std(path)
                if std_path > 1e-10:
                    skew = np.sum((path - mean_path)**3) / (len(path) * std_path**3)
                    kurt = np.sum((path - mean_path)**4) / (len(path) * std_path**4) - 3
                else:
                    skew = 0.0
                    kurt = 0.0
                features[f'skew_{i}'] = np.array([np.nan_to_num(skew)])
                features[f'kurt_{i}'] = np.array([np.nan_to_num(kurt)])

                # 自己相関
                if len(path) > 1:
                    ac = np.correlate(path - mean_path, path - mean_path, mode='full')[len(path)-1:]
                    if np.var(path) > 1e-10:
                        ac = ac / (np.var(path) * np.arange(len(path), 0, -1))
                        features[f'autocorr_{i}'] = np.array([np.mean(ac[:5])])
                    else:
                        features[f'autocorr_{i}'] = np.array([0.0])

        elif isinstance(result, dict):
            for k, v in result.items():
                if k not in features:
                    features[k] = v

        # 4. 主要特徴のペアワイズ組み合わせ
        main_keys = ["Q_Λ", "σ_Q", "E"]
        for i, f1 in enumerate(main_keys):
            for j, f2 in enumerate(main_keys):
                if i < j and len(features[f1]) == len(features[f2]):
                    features[f'{f1}×{f2}'] = features[f1] * features[f2]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        r = features[f1] / (features[f2] + 1e-10)
                        features[f'{f1}/{f2}'] = np.nan_to_num(r, nan=0.0, posinf=10.0, neginf=-10.0)

        # === 新規：Lambda³物理量と周波数特徴の相互作用 ===
        if 'event_freq_peak_amp' in features:
            # Q_Λと周波数振幅の相互作用
            features['Q_Λ×freq_amplitude'] = Qs.mean() * features['event_freq_peak_amp']
            features['σ_Q×freq_amplitude'] = Sigmas.mean() * features['event_freq_peak_amp']
            features['E×freq_amplitude'] = Es.mean() * features['event_freq_peak_amp']

            # 周波数エネルギーとの相互作用
            if 'event_freq_energy' in features:
                features['Q_Λ×freq_energy'] = Qs.mean() * features['event_freq_energy']
                features['Q_Λ/freq_energy'] = Qs.mean() / (features['event_freq_energy'] + 1e-10)

        # 5. 非線形変換（全特徴に適用）
        for k, v in list(features.items()):
            # 周波数関連特徴は既に非線形なのでスキップ
            if 'freq' not in k and 'fft' not in k:
                features[f'log_{k}'] = np.log1p(np.abs(v))
                features[f'sqrt_{k}'] = np.sqrt(np.abs(v))
                features[f'sq_{k}'] = v ** 2
                features[f'sig_{k}'] = 1 / (1 + np.exp(-v))

        # 6. イベントレベルの統計量
        features['event_mean'] = np.mean(events, axis=1)
        features['event_std'] = np.std(events, axis=1)

        return features

    def _extract_frequency_features_from_path(self, path: np.ndarray) -> Dict[str, float]:
        """パスの周波数特徴を抽出"""
        # FFT計算
        fft = np.fft.fft(path)
        fft_abs = np.abs(fft)
        fft_freqs = np.fft.fftfreq(len(path))

        # 正の周波数のみ
        pos_mask = fft_freqs > 0
        pos_freqs = fft_freqs[pos_mask]
        pos_fft = fft_abs[pos_mask]

        features = {}

        if len(pos_fft) > 0:
            # ピーク周波数とその振幅
            peak_idx = np.argmax(pos_fft)
            features['freq_peak'] = pos_freqs[peak_idx]
            features['freq_peak_amp'] = pos_fft[peak_idx]

            # 周波数エネルギー
            features['freq_energy'] = np.sum(pos_fft ** 2)

            # スペクトル重心
            if np.sum(pos_fft) > 0:
                features['freq_centroid'] = np.sum(pos_freqs * pos_fft) / np.sum(pos_fft)
            else:
                features['freq_centroid'] = 0.0

            # スペクトルエントロピー
            if np.sum(pos_fft) > 0:
                norm_fft = pos_fft / np.sum(pos_fft)
                features['freq_entropy'] = -np.sum(norm_fft * np.log(norm_fft + 1e-10))
            else:
                features['freq_entropy'] = 0.0

            # 高周波/低周波比率
            mid_point = len(pos_fft) // 2
            if mid_point > 0:
                low_energy = np.sum(pos_fft[:mid_point] ** 2)
                high_energy = np.sum(pos_fft[mid_point:] ** 2)
                features['freq_hf_lf_ratio'] = high_energy / (low_energy + 1e-10)
            else:
                features['freq_hf_lf_ratio'] = 0.0
        else:
            # デフォルト値
            features = {
                'freq_peak': 0.0,
                'freq_peak_amp': 0.0,
                'freq_energy': 0.0,
                'freq_centroid': 0.0,
                'freq_entropy': 0.0,
                'freq_hf_lf_ratio': 0.0
            }

        return features

    def _extract_event_frequency_features(self, events: np.ndarray) -> Dict[str, np.ndarray]:
        """イベント空間での周波数特徴を抽出"""
        n_events, n_features = events.shape

        # 各特徴次元でFFT
        all_ffts = np.fft.fft(events, axis=0)
        all_fft_abs = np.abs(all_ffts)

        features = {}

        # 全体的な周波数特徴
        # ピーク振幅（各特徴の最大値の平均）
        features['freq_peak_amp'] = np.mean(np.max(all_fft_abs[1:n_events//2], axis=0))

        # 周波数エネルギー（全特徴の平均）
        features['freq_energy'] = np.mean(np.sum(all_fft_abs ** 2, axis=0))

        # 支配的周波数の分散（特徴間の周波数パターンの違い）
        peak_freqs = []
        for f in range(n_features):
            fft_f = all_fft_abs[1:n_events//2, f]
            if len(fft_f) > 0 and np.max(fft_f) > 0:
                peak_idx = np.argmax(fft_f)
                peak_freqs.append(peak_idx / len(fft_f))

        if peak_freqs:
            features['freq_dispersion'] = np.std(peak_freqs)
        else:
            features['freq_dispersion'] = 0.0

        # 低周波成分の割合
        low_freq_cutoff = n_events // 10
        if low_freq_cutoff > 1:
            low_freq_energy = np.sum(all_fft_abs[1:low_freq_cutoff] ** 2)
            total_energy = np.sum(all_fft_abs[1:n_events//2] ** 2)
            features['low_freq_ratio'] = low_freq_energy / (total_energy + 1e-10)
        else:
            features['low_freq_ratio'] = 0.0

        # 特徴間の周波数相関
        if n_features > 1:
            freq_corrs = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    corr = np.corrcoef(all_fft_abs[:, i], all_fft_abs[:, j])[0, 1]
                    freq_corrs.append(corr)
            features['freq_correlation'] = np.mean(freq_corrs)
        else:
            features['freq_correlation'] = 0.0

        # これらはスカラー値なので、イベント数に合わせて拡張
        for fname in list(features.keys()):
            features[fname] = np.full(n_events, features[fname])

        return features

    def project_to_event_space(self,
                              features: Dict[str, np.ndarray],
                              paths_matrix: np.ndarray,
                              event_indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """パス特徴量をイベント空間に射影"""
        n_paths, n_events = paths_matrix.shape

        if event_indices is None:
            event_indices = np.arange(n_events)

        event_features = {}

        for name, vals in features.items():
            if vals.shape[0] == n_paths:  # パス特徴量
                event_scores = np.zeros(len(event_indices))
                for i, evt_idx in enumerate(event_indices):
                    event_scores[i] = np.sum(np.abs(paths_matrix[:, evt_idx]) * vals)
                event_features[name] = event_scores
            elif len(vals) == len(event_indices):  # 既にイベント特徴量
                event_features[name] = vals

        return event_features
