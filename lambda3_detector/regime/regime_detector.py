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


def _default_scorer_factories(percentile: float = 99.0) -> List[Callable]:
    return [
        lambda: StreamingJumpScorer(percentile=percentile),
        lambda: StreamingGradualScorer(
            window_sizes=[50, 200, 500], percentile=percentile
        ),
        lambda: StreamingStructuralDriftScorer(
            local_window=200, percentile=percentile
        ),
        lambda: StreamingReconstructionScorer(
            n_components=5, delay_window=20, percentile=percentile
        ),
        lambda: StreamingKernelScorer(
            kernel='polynomial', degree=3, coef0=1.0, percentile=percentile
        ),
        lambda: StreamingStructuralScorer(
            delay_window=20, percentile=percentile
        ),
    ]


class RegimeAwareDetector:
    """K-regime GMM + per-regime threshold + OR voting。"""

    def __init__(self,
                 K: Union[int, str] = 'auto',
                 K_max: int = 5,
                 mask_margin: int = 50,
                 percentile: float = 99.0,
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
            mask_margin: anomaly window の前後 margin (frame)
            percentile: regime ごとの threshold percentile
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
        self.percentile = float(percentile)
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

        # 1. マスク拡張
        expanded_mask = expand_anomaly_mask(anomaly_mask, self.mask_margin)
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
            if int(mask_k.sum()) < self.min_frames_per_regime:
                # サンプル不足 → 無限大 threshold で実質無効化
                for s in self.scorers:
                    self.thresholds_per_regime[k][s.name] = float('inf')
                continue
            for s in self.scorers:
                scores_k = clean_scores_by_scorer[s.name][mask_k]
                positive = scores_k[scores_k > 1e-12]
                if len(positive) > 5:
                    self.thresholds_per_regime[k][s.name] = float(
                        np.percentile(positive, self.percentile)
                    )
                else:
                    self.thresholds_per_regime[k][s.name] = float('inf')

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
