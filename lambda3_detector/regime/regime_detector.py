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

from typing import Callable, Dict, List, Optional

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
                 K: int = 3,
                 mask_margin: int = 50,
                 percentile: float = 99.0,
                 scorer_factories: Optional[List[Callable]] = None,
                 normalize: bool = True,
                 random_state: int = 0,
                 min_frames_per_regime: int = 20):
        """
        Args:
            K: GMM 成分数 (v1 固定、サンプル数が足りなければ自動縮小)
            mask_margin: anomaly window の前後 margin (frame)
            percentile: regime ごとの threshold percentile
            scorer_factories: 各 scorer を返す callable list (None で default 6)
            normalize: clean 統計量で z-normalize
            random_state: GMM 再現性
            min_frames_per_regime: regime k のサンプルがこれ未満なら threshold=inf
        """
        self.K = int(K)
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

        # 3. GMM fit (K_eff = min(K, clean_size // min_frames_per_regime / 2))
        K_eff = max(1, min(self.K, len(clean) // (self.min_frames_per_regime * 2)))
        self.K_eff = K_eff
        self.gmm = GaussianMixture(
            n_components=K_eff,
            covariance_type='full',
            random_state=self.random_state,
            reg_covar=1e-6,
            max_iter=200,
        )
        self.gmm.fit(clean)
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
            'cal_clean_frames': self.cal_clean_frames,
            'mask_margin': self.mask_margin,
            'normalized': self.normalize,
        }
