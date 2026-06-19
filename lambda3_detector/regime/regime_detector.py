"""
RegimeAwareDetector: semi-supervised (normal-label only) regime-aware
streaming detector for Lambda³。

Pipeline:
  1. clean = events[~anomaly_mask_expanded]    # anomaly window + margin 除外
  2. z-normalize using clean statistics
  3. GMM(K).fit(clean) → 各 frame の regime cluster
  4. 6 scorer を全 clean data で calibrate (共通 baseline)
  5. 各 scorer の raw score 系列を clean 全期間で計算
     → regime ごとに per-scorer percentile threshold を fit
  6. streaming: 全 frame について gmm.predict → regime 別 threshold で OR voting

Tier 0 (zero-shot streaming) との違い:
  - calibration が「先頭 15%」ではなく「全期間から異常窓を除いた clean」
  - threshold が「single value per scorer」ではなく「regime ごと per scorer」
  - 全 frame で normalize と GMM 推論を行うため、計算量は ~2x

Anomaly の "shape" は使用しない (window 除外のみ):
  - anomaly_mask は train-time マスクのみに使用
  - GMM/scorer は anomaly frame を一切見ない
  - 学術的分類: semi-supervised, normal-label only
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

try:
    from sklearn.mixture import GaussianMixture
except ImportError as e:
    raise ImportError(
        "RegimeAwareDetector requires scikit-learn. "
        "Install via `pip install scikit-learn`."
    ) from e

from ..streaming import (
    StreamingGradualScorer,
    StreamingJumpScorer,
    StreamingKernelScorer,
    StreamingReconstructionScorer,
    StreamingScorer,
    StreamingStructuralDriftScorer,
    StreamingStructuralScorer,
)


def compute_robust_threshold(scores: np.ndarray,
                             method: str = 'percentile',
                             percentile: float = 99.0,
                             iqr_k: float = 3.0,
                             mad_k: float = 2.5,
                             trim_fraction: float = 0.01,
                             cap_ratio: float = 5.0,
                             cap_quantile: float = 90.0) -> float:
    """Robust threshold from a 1-D array of positive scores。

    Methods:
        'percentile'         : np.percentile(scores, percentile)         ← baseline
        'trimmed_percentile' : 上位 trim_fraction を除外後の percentile
        'iqr'                : Q3 + iqr_k * IQR     (Tukey の outlier rule)
        'mad'                : median + mad_k * 1.4826 * MAD
        'capped'             : min(percentile, cap_ratio * cap_quantile)
                               normal continuous tail なら p99 ≈ 2-3 × p90 → p99 そのまま
                               isolated outlier 混入なら p99 ≫ 5 × p90 → p90 × 5 にクリップ
                               1 ハイパラ (cap_ratio=5.0)、解釈可能、per-scorer 独立に効く。
                               Adaptive method selection 相当の挙動を線形式で実現。

    sample が少ない (< 5) 場合は inf を返して該当 scorer/regime を無効化する。
    """
    if scores.size < 5:
        return float('inf')

    if method == 'percentile':
        return float(np.percentile(scores, percentile))

    if method == 'trimmed_percentile':
        # 上位 trim_fraction を除外してから percentile
        trim_cut = float(np.percentile(scores, 100.0 * (1.0 - trim_fraction)))
        trimmed = scores[scores <= trim_cut]
        if trimmed.size < 5:
            return trim_cut
        return float(np.percentile(trimmed, percentile))

    if method == 'iqr':
        q1, q3 = np.percentile(scores, [25.0, 75.0])
        iqr = float(q3 - q1)
        if iqr <= 0:
            # 分布が degenerate (median = Q3) → fallback to percentile
            return float(np.percentile(scores, percentile))
        return float(q3 + iqr_k * iqr)

    if method == 'mad':
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        if mad <= 0:
            return float(np.percentile(scores, percentile))
        return med + mad_k * 1.4826 * mad

    if method == 'capped':
        p_main = float(np.percentile(scores, percentile))
        p_cap = float(np.percentile(scores, cap_quantile))
        if p_cap <= 0:
            # cap base 0 → cap 不能、percentile そのまま
            return p_main
        return float(min(p_main, cap_ratio * p_cap))

    raise ValueError(f"unknown threshold_method: {method!r}")


def expand_anomaly_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    """anomaly window を前後 margin frame 拡張する (boundary leakage 防止)。

    Args:
        mask: (n,) bool, True = known anomaly frame
        margin: 拡張幅 (frame 数)

    Returns:
        expanded mask (n,) bool, train から除外する frame が True
    """
    n = len(mask)
    if margin <= 0:
        return mask.copy()
    # dilate: 各 True 位置の前後 margin frame を True に
    out = np.zeros(n, dtype=bool)
    positions = np.where(mask)[0]
    for idx in positions:
        s = max(0, idx - margin)
        e = min(n, idx + margin + 1)
        out[s:e] = True
    return out


def adaptive_anomaly_mask(events: np.ndarray,
                          anomaly_mask: np.ndarray,
                          base_margin: int = 50,
                          max_margin: int = 300,
                          max_exclusion_ratio: float = 0.4,
                          recovery_window: int = 30,
                          variance_ratio: float = 2.0) -> tuple:
    """Adaptive anomaly margin expansion。

    2 つの問題を同時に解決:
      (1) gradual leak: anomaly window 前後の transient (recovery) 期間が
          baseline variance に戻るまで margin 延長 (max_margin で上限)
      (2) clean shrink: 総除外率が max_exclusion_ratio を超えたら
          base_margin より縮小 (多窓 file での clean サンプル不足防止)

    アルゴリズム:
      1. baseline variance = events[mask_far_from_anomaly] の std
      2. 各 window の前後 max_margin 範囲を walk:
           local variance > variance_ratio * baseline_variance なら margin 延長
           recovered (or max_margin 到達) で break
      3. base_margin 最低保証で OR 合成
      4. 総除外率 > max_exclusion_ratio なら margin を縮小して再構築

    Args:
        events: (n,) or (n, d) z-normalize 前の生 events
        anomaly_mask: (n,) bool
        base_margin: 最低保証 margin (default 50)
        max_margin: 適応延長の上限 (default 300)
        max_exclusion_ratio: 総除外率の上限 (default 0.4 = 40%)
        recovery_window: local variance を測る windowsize (default 30)
        variance_ratio: baseline の何倍までを "recovered" とみなすか (default 2.0)

    Returns:
        (expanded_mask, info_dict)
        info_dict: {
            'baseline_std': float,
            'avg_margin_pre': float, 'avg_margin_post': float,
            'exclusion_ratio_pre_cap': float,
            'exclusion_ratio_final': float,
            'effective_margin_after_cap': int,
        }
    """
    n = len(anomaly_mask)
    if events.ndim == 1:
        sig_1d = events
    else:
        sig_1d = events.mean(axis=1)

    # 1. baseline variance from frames far from any anomaly
    safe_mask = expand_anomaly_mask(anomaly_mask, max_margin)
    safe_frames = sig_1d[~safe_mask]
    if len(safe_frames) < 100:
        # 安全 baseline 不足 → fallback to base_margin
        return expand_anomaly_mask(anomaly_mask, base_margin), {
            'baseline_std': float('nan'),
            'avg_margin_pre': float(base_margin),
            'avg_margin_post': float(base_margin),
            'exclusion_ratio_pre_cap': float(safe_mask.sum() / n),
            'exclusion_ratio_final': float(
                expand_anomaly_mask(anomaly_mask, base_margin).sum() / n
            ),
            'effective_margin_after_cap': int(base_margin),
            'fallback': True,
        }

    baseline_std = float(np.std(safe_frames))
    threshold_std = variance_ratio * baseline_std

    # 2. find contiguous anomaly windows
    diff = np.diff(
        np.concatenate([[False], anomaly_mask, [False]]).astype(np.int8)
    )
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    expanded = anomaly_mask.copy()
    margins_pre = []
    margins_post = []
    half_win = recovery_window // 2

    def _local_std(i: int) -> float:
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        seg = sig_1d[lo:hi]
        if len(seg) < 5:
            return float('inf')
        return float(np.std(seg))

    for s_idx, e_idx in zip(starts, ends):
        # Forward (after window end): extend until recovered
        pe = 0
        for off in range(1, max_margin + 1):
            i = e_idx + off
            if i >= n:
                break
            if _local_std(i) <= threshold_std and off >= base_margin:
                pe = off
                break
            expanded[i] = True
            pe = off
        margins_post.append(pe)

        # Backward (before window start): extend until normalcy
        pb = 0
        for off in range(1, max_margin + 1):
            i = s_idx - off
            if i < 0:
                break
            if _local_std(i) <= threshold_std and off >= base_margin:
                pb = off
                break
            expanded[i] = True
            pb = off
        margins_pre.append(pb)

    # 3. ensure base_margin minimum
    base_expanded = expand_anomaly_mask(anomaly_mask, base_margin)
    expanded = expanded | base_expanded

    excl_pre_cap = float(expanded.sum()) / n
    avg_pre = float(np.mean(margins_pre)) if margins_pre else float(base_margin)
    avg_post = float(np.mean(margins_post)) if margins_post else float(base_margin)
    eff_margin = max_margin

    # 4. cap total exclusion
    if excl_pre_cap > max_exclusion_ratio:
        # margin を base_margin から段階的に縮小して max_exclusion_ratio 以下を目指す
        for m in range(base_margin, -1, -5):
            candidate = expand_anomaly_mask(anomaly_mask, m)
            if candidate.sum() / n <= max_exclusion_ratio:
                expanded = candidate
                eff_margin = m
                break
        else:
            expanded = anomaly_mask.copy()
            eff_margin = 0

    excl_final = float(expanded.sum()) / n

    info = {
        'baseline_std': baseline_std,
        'avg_margin_pre': avg_pre,
        'avg_margin_post': avg_post,
        'exclusion_ratio_pre_cap': excl_pre_cap,
        'exclusion_ratio_final': excl_final,
        'effective_margin_after_cap': int(eff_margin),
        'fallback': False,
    }
    return expanded, info


#: Mapping from short scorer name → factory. Used for per-scorer ablation
#: ("--scorers jump,kernel" or "--exclude-scorers kernel" in CLI).
SCORER_FACTORIES: Dict[str, Callable[[float], Callable]] = {
    'jump':    lambda p: (lambda: StreamingJumpScorer(percentile=p)),
    'gradual': lambda p: (lambda: StreamingGradualScorer(
        window_sizes=[50, 200, 500], percentile=p)),
    'drift':   lambda p: (lambda: StreamingStructuralDriftScorer(
        local_window=200, percentile=p)),
    'recon':   lambda p: (lambda: StreamingReconstructionScorer(
        n_components=5, delay_window=20, percentile=p)),
    'kernel':  lambda p: (lambda: StreamingKernelScorer(
        kernel='polynomial', degree=3, coef0=1.0, percentile=p)),
    'struct':  lambda p: (lambda: StreamingStructuralScorer(
        delay_window=20, percentile=p)),
}

#: Canonical ordering (for reproducibility, deterministic results).
SCORER_NAMES: List[str] = ['jump', 'gradual', 'drift', 'recon', 'kernel', 'struct']


def build_scorer_factories(scorer_names: Optional[List[str]] = None,
                           percentile: float = 99.0) -> List[Callable]:
    """Build scorer factory list from short names.

    Args:
        scorer_names: subset of SCORER_NAMES (default = all 6 in canonical order)
        percentile: passed to each scorer

    Raises:
        ValueError: unknown scorer name or empty list
    """
    if scorer_names is None:
        scorer_names = SCORER_NAMES
    if not scorer_names:
        raise ValueError("scorer_names must be non-empty")
    out = []
    for name in scorer_names:
        if name not in SCORER_FACTORIES:
            raise ValueError(
                f"unknown scorer name {name!r}; valid: {SCORER_NAMES}"
            )
        out.append(SCORER_FACTORIES[name](percentile))
    return out


def _default_scorer_factories(percentile: float = 99.0) -> List[Callable]:
    """Backward-compat: build all 6 default scorer factories."""
    return build_scorer_factories(None, percentile=percentile)


class RegimeAwareDetector:
    """K-regime GMM + per-regime threshold + OR voting。"""

    def __init__(self,
                 K: Union[int, str] = 'auto',
                 K_max: int = 5,
                 mask_margin: int = 50,
                 margin_adaptive: bool = False,
                 margin_max: int = 300,
                 margin_max_exclusion_ratio: float = 0.4,
                 margin_recovery_window: int = 30,
                 margin_variance_ratio: float = 2.0,
                 percentile: float = 99.0,
                 threshold_method: str = 'trimmed_percentile',
                 iqr_k: float = 3.0,
                 mad_k: float = 2.5,
                 trim_fraction: float = 0.01,
                 cap_ratio: float = 5.0,
                 cap_quantile: float = 90.0,
                 cap_min_regime_size: int = 300,
                 scorer_factories: Optional[List[Callable]] = None,
                 normalize: bool = True,
                 random_state: int = 0,
                 min_frames_per_regime: int = 50):
        """
        Args:
            K: GMM 成分数。int で固定 (1-K_max)、'auto' で BIC 自動選択。
               'auto' のとき K_min=1 から K_max まで全て fit し、
               「最小クラスタが min_frames_per_regime 以上」かつ BIC 最小の K を選ぶ。
               artificialWithAnomaly の synthetic data 等で K=1 が最適なケースに対応。
            K_max: K='auto' の最大候補数 (default 5)
            mask_margin: anomaly window の前後 margin (frame)。
                margin_adaptive=True のときは「base_margin = 最低保証」として機能。
            margin_adaptive: True で adaptive_anomaly_mask を使用 (default False)
                gradual leak (anomaly 影響が window 外に染み出す) と
                clean shrink (多窓 file で除外過剰) を同時対処。
                NAB 実験では効果なし (NAB ラベルは tight なため variance recovery が即起こる)。
                seasonal drift / long-tail recovery を持つ domain で有効と想定。
            margin_max: 適応延長の上限 frame (default 300)
            margin_max_exclusion_ratio: 総除外率の cap (default 0.4)
            margin_recovery_window: local variance を測る window (default 30)
            margin_variance_ratio: baseline の何倍までを recovered と見るか (default 2.0)
            percentile: 'percentile' / 'trimmed_percentile' 用 percentile (default 99)
            threshold_method: regime ごとの threshold 計算手法
                ★ Default: 'trimmed_percentile' (NAB 全 52 file 加重 72.02 確定)
                - 'trimmed_percentile' : 上位 trim_fraction (default 1%) 除外後の percentile
                                         ← RECOMMENDED (training 内 rare outlier を除去)
                - 'percentile'         : np.percentile(scores, percentile)  (baseline 71.29)
                - 'iqr'                : Q3 + iqr_k * IQR (experimental, NAB 66.88)
                - 'mad'                : median + mad_k * 1.4826 * MAD (experimental, NAB 66.09)
                - 'capped'             : min(p99, cap_ratio * p_cap_quantile) (experimental,
                                         small regime で誤発動するため非推奨, NAB 70.69)
            iqr_k: 'iqr' method の係数 (default 3.0、Tukey extreme outlier)
            mad_k: 'mad' method の係数 (default 2.5、~99% 相当)
            trim_fraction: 'trimmed_percentile' method の上位除外割合 (default 0.01)
            cap_min_regime_size: 'capped' を有効化する最小 regime サンプル数 (default 300)。
                これ未満の regime では cap を無効化し percentile に fallback。
                小 regime では sample 数不足で p99/p90 > 5 が自然変動でも発生し、
                cap 誤発動が FP cascade を引き起こすため。
            scorer_factories: 各 scorer を返す callable list (None で default 6)
            normalize: clean 統計量で z-normalize
            random_state: GMM 再現性
            min_frames_per_regime: regime k のサンプルがこれ未満ならその K は不採用
                (default 50: K=3 fixed のとき k_size=4-19 の noise regime が
                 inf threshold を出す問題を BIC で根本解決)
        """
        if isinstance(K, str):
            if K.lower() != 'auto':
                raise ValueError(f"K must be int or 'auto', got {K!r}")
            self.K: Union[int, str] = 'auto'
        else:
            self.K = max(1, int(K))
        self.K_max = int(K_max)
        self.mask_margin = int(mask_margin)
        self.margin_adaptive = bool(margin_adaptive)
        self.margin_max = int(margin_max)
        self.margin_max_exclusion_ratio = float(margin_max_exclusion_ratio)
        self.margin_recovery_window = int(margin_recovery_window)
        self.margin_variance_ratio = float(margin_variance_ratio)
        self.percentile = float(percentile)
        valid_methods = {'percentile', 'trimmed_percentile', 'iqr', 'mad', 'capped'}
        if threshold_method not in valid_methods:
            raise ValueError(
                f"threshold_method must be one of {valid_methods}, got {threshold_method!r}"
            )
        self.threshold_method = threshold_method
        self.iqr_k = float(iqr_k)
        self.mad_k = float(mad_k)
        self.trim_fraction = float(trim_fraction)
        self.cap_ratio = float(cap_ratio)
        self.cap_quantile = float(cap_quantile)
        self.cap_min_regime_size = int(cap_min_regime_size)
        self.normalize = bool(normalize)
        self.random_state = int(random_state)
        self.min_frames_per_regime = int(min_frames_per_regime)
        self.scorer_factories = (
            scorer_factories if scorer_factories is not None
            else _default_scorer_factories(percentile=percentile)
        )

        # fit 結果
        self.gmm: Optional[GaussianMixture] = None
        self.scorers: List[StreamingScorer] = []
        self.thresholds_per_regime: Dict[int, Dict[str, float]] = {}
        self.K_eff: int = 0
        self.clean_mu: Optional[np.ndarray] = None
        self.clean_sd: Optional[np.ndarray] = None
        self.cal_clean_frames: int = 0
        self.bic_per_K: Dict[int, float] = {}
        self.margin_info: Optional[Dict] = None

    def _fit_gmm_adaptive(self, clean: np.ndarray) -> tuple:
        """K∈[1, K_upper] を試して、最小クラスタが min_frames_per_regime
        以上を満たす範囲で BIC 最小の K を選ぶ。

        K が int 指定の場合: K 固定 (clean サンプル数で自動縮小のみ)。
        K='auto' の場合: BIC 自動選択。

        Returns:
            (gmm, K_eff, bic_per_K_dict)
        """
        n_clean = len(clean)
        # 上限: clean サンプル数で物理的に制限 (各 cluster 最低 min_frames_per_regime)
        K_physical_max = max(1, n_clean // self.min_frames_per_regime)

        if self.K == 'auto':
            K_upper = min(self.K_max, K_physical_max)
            candidates = list(range(1, K_upper + 1))
        else:
            K_target = min(int(self.K), K_physical_max)
            candidates = [max(1, K_target)]

        bic_per_K: Dict[int, float] = {}
        best_K = 1
        best_bic = float('inf')
        best_gmm = None

        for K in candidates:
            try:
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type='full',
                    random_state=self.random_state,
                    reg_covar=1e-6,
                    max_iter=200,
                )
                gmm.fit(clean)
            except Exception:
                continue
            # 全 cluster が min_frames_per_regime 以上か?
            labels = gmm.predict(clean)
            sizes = np.bincount(labels, minlength=K)
            min_size = int(sizes.min())
            bic = float(gmm.bic(clean))
            bic_per_K[K] = bic
            if min_size < self.min_frames_per_regime:
                # noise cluster あり → 不採用 (K='auto' 時)
                continue
            if bic < best_bic:
                best_bic = bic
                best_K = K
                best_gmm = gmm

        # fallback: K=1 (clean 全体を 1 cluster と見做す)
        if best_gmm is None:
            best_K = 1
            best_gmm = GaussianMixture(
                n_components=1, covariance_type='full',
                random_state=self.random_state, reg_covar=1e-6, max_iter=200,
            ).fit(clean)
            if 1 not in bic_per_K:
                bic_per_K[1] = float(best_gmm.bic(clean))

        return best_gmm, best_K, bic_per_K

    def fit_predict(self, events: np.ndarray, anomaly_mask: np.ndarray) -> dict:
        """One-shot: regime fit (offline) + per-frame streaming OR voting。

        Args:
            events: (n,) or (n, d) full time series
            anomaly_mask: (n,) bool, True = known anomaly frame (除外対象)

        Returns:
            dict containing:
                'score'    : (n,) max-normalized continuous score (>=1 = flagged)
                'binary'   : (n,) 0/1 OR voting result
                'per_scorer': dict[name -> (n,) raw]
                'thresholds_per_regime': dict[k -> dict[name -> float]]
                'regimes'  : (n,) int assigned regime per frame
                'K_eff'    : actually used K (sample count で縮小されうる)
                'cal_clean_frames': clean サンプル数
                'mask_margin': used margin
        """
        n = len(events)
        anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
        if anomaly_mask.shape != (n,):
            raise ValueError(
                f"anomaly_mask shape {anomaly_mask.shape} != events length {n}"
            )

        # 1. マスク拡張 (固定 margin or adaptive)
        if self.margin_adaptive:
            expanded_mask, self.margin_info = adaptive_anomaly_mask(
                events,
                anomaly_mask,
                base_margin=self.mask_margin,
                max_margin=self.margin_max,
                max_exclusion_ratio=self.margin_max_exclusion_ratio,
                recovery_window=self.margin_recovery_window,
                variance_ratio=self.margin_variance_ratio,
            )
        else:
            expanded_mask = expand_anomaly_mask(anomaly_mask, self.mask_margin)
            self.margin_info = None
        clean_idx = np.where(~expanded_mask)[0]
        if len(clean_idx) < max(self.min_frames_per_regime * 2, 100):
            raise ValueError(
                f"clean data too small ({len(clean_idx)} frames) for regime fitting "
                f"(margin={self.mask_margin}, total={n})"
            )

        # 2. z-normalize (clean 統計量で全期間を変換)
        X = events if events.ndim > 1 else events.reshape(-1, 1)
        if self.normalize:
            self.clean_mu = X[~expanded_mask].mean(axis=0)
            self.clean_sd = X[~expanded_mask].std(axis=0) + 1e-10
            X_norm = (X - self.clean_mu) / self.clean_sd
            events_used = X_norm
        else:
            self.clean_mu = np.zeros(X.shape[1])
            self.clean_sd = np.ones(X.shape[1])
            events_used = X.astype(np.float64)

        clean = events_used[~expanded_mask]
        self.cal_clean_frames = len(clean)

        # 3. GMM fit: K 固定 or BIC 自動選択
        self.gmm, K_eff, self.bic_per_K = self._fit_gmm_adaptive(clean)
        self.K_eff = K_eff
        regime_labels_clean = self.gmm.predict(clean)

        # 4. 全 clean data で scorer を calibrate (共通 baseline)
        self.scorers = [f() for f in self.scorer_factories]
        for s in self.scorers:
            s.calibrate(clean)

        # 5. 全 clean data 上で scorer の raw score 系列を計算
        #    regime 別に percentile を切る
        self.thresholds_per_regime = {k: {} for k in range(K_eff)}
        clean_scores_by_scorer: Dict[str, np.ndarray] = {}
        for s in self.scorers:
            raw = np.array(
                [float(s.score(clean, t)) for t in range(len(clean))],
                dtype=np.float64,
            )
            clean_scores_by_scorer[s.name] = raw

        for k in range(K_eff):
            mask_k = (regime_labels_clean == k)
            n_k = int(mask_k.sum())
            if n_k < self.min_frames_per_regime:
                # サンプル不足 → 無限大 threshold で実質無効化
                for s in self.scorers:
                    self.thresholds_per_regime[k][s.name] = float('inf')
                continue
            # cap_ratio adaptive: 小 regime では cap 無効化 (percentile fallback)
            cap_ratio_k = (
                self.cap_ratio if n_k >= self.cap_min_regime_size else float('inf')
            )
            for s in self.scorers:
                scores_k = clean_scores_by_scorer[s.name][mask_k]
                positive = scores_k[scores_k > 1e-12]
                self.thresholds_per_regime[k][s.name] = compute_robust_threshold(
                    positive,
                    method=self.threshold_method,
                    percentile=self.percentile,
                    iqr_k=self.iqr_k,
                    mad_k=self.mad_k,
                    trim_fraction=self.trim_fraction,
                    cap_ratio=cap_ratio_k,
                    cap_quantile=self.cap_quantile,
                )

        # 6. streaming: 全 frame の regime を一括予測、frame ごとに OR voting
        regimes_all = self.gmm.predict(events_used).astype(np.int32)
        combined = np.zeros(n, dtype=np.float64)
        per_scorer: Dict[str, np.ndarray] = {
            s.name: np.zeros(n, dtype=np.float64) for s in self.scorers
        }

        for t in range(n):
            k = int(regimes_all[t])
            best_ratio = 0.0
            for s in self.scorers:
                raw = float(s.score(events_used, t))
                per_scorer[s.name][t] = raw
                thr = self.thresholds_per_regime[k].get(s.name, float('inf'))
                if thr > 0 and np.isfinite(thr):
                    ratio = raw / (thr + 1e-12)
                    if ratio > best_ratio:
                        best_ratio = ratio
            combined[t] = best_ratio

        binary = (combined >= 1.0).astype(np.int32)

        return {
            'score': combined,
            'binary': binary,
            'per_scorer': per_scorer,
            'thresholds_per_regime': self.thresholds_per_regime,
            'regimes': regimes_all,
            'K_eff': K_eff,
            'K_requested': self.K,
            'bic_per_K': self.bic_per_K,
            'cal_clean_frames': self.cal_clean_frames,
            'mask_margin': self.mask_margin,
            'normalized': self.normalize,
        }
