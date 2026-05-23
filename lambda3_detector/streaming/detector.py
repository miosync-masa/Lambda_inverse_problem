"""Lambda3StreamingDetector: calibrate → stream → Binary OR voting。

設計：
  各 StreamingScorer は per-frame raw score と threshold (calibration で固定)
  を持つ。最終出力は per-frame の "max-normalized score":
      out[t] = max_k ( raw_score_k(t) / threshold_k )
  out[t] >= 1.0 のとき「フラグ」とみなす (Binary OR voting と等価)。

  NAB Sweeper は連続値の score 列を期待するので、この max-normalized 値を
  そのまま渡せば threshold sweep が機能する。Sweeper の best threshold が
  1.0 に近ければ「Binary OR voting がそのまま最適」を示す。

  Calibration phase (先頭 cal_ratio*n フレーム) では out[t] = 0 とする
  (NAB probationary period と整合: probation 中は detection しない)。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import StreamingScorer


class Lambda3StreamingDetector:
    """各 StreamingScorer を OR voting で統合する detector。"""

    def __init__(self,
                 scorers: List[StreamingScorer],
                 calibration_ratio: float = 0.15,
                 min_calibration: int = 50,
                 normalize: bool = True):
        """
        normalize: True なら calibration 区間の feature 毎の (mean, std) を
            測って全 events を z-normalize してから各 scorer に渡す。
            scorer は scale-invariant な空間で動作する。
            NAB の disk_write_bytes (値域 数百万) のような巨大スケール file で
            一部 scorer が threshold 爆発して dead-zero になるのを防ぐ。
        """
        if not scorers:
            raise ValueError("at least one scorer required")
        self.scorers = list(scorers)
        self.calibration_ratio = float(calibration_ratio)
        self.min_calibration = int(min_calibration)
        self.normalize = bool(normalize)

    def fit_predict(self, events: np.ndarray) -> dict:
        """Calibrate on the first cal_ratio*n frames, then stream.

        Returns dict containing:
            'score'    : (n,) max-normalized continuous score (>=1.0 = flagged)
            'binary'   : (n,) 0/1 binary OR voting result
            'per_scorer': dict[scorer_name -> (n,) raw scores]
            'thresholds': dict[scorer_name -> float]
            'cal_end'  : int, index where streaming starts
            'normalized': bool, whether pre-normalization was applied
        """
        n = len(events)
        cal_end = max(self.min_calibration, int(n * self.calibration_ratio))
        cal_end = min(cal_end, n - 1)

        # ===== Pre-normalize (z-norm per feature, cal-window-fixed) =====
        if self.normalize:
            X = events if events.ndim > 1 else events.reshape(-1, 1)
            mu = X[:cal_end].mean(axis=0)
            sd = X[:cal_end].std(axis=0) + 1e-10
            X_norm = (X - mu) / sd
            events_used = X_norm.reshape(events.shape) if events.ndim == 1 else X_norm
        else:
            events_used = events

        events_cal = events_used[:cal_end]

        # ===== Calibration phase =====
        for s in self.scorers:
            s.calibrate(events_cal)

        # ===== Streaming phase =====
        per_scorer_scores = {s.name: np.zeros(n, dtype=np.float64)
                             for s in self.scorers}
        combined = np.zeros(n, dtype=np.float64)

        for t in range(n):
            if t < cal_end:
                # Probationary period: emit 0 (no detection)
                continue
            best_ratio = 0.0
            for s in self.scorers:
                raw = float(s.score(events_used, t))
                per_scorer_scores[s.name][t] = raw
                thr = s.threshold
                if thr > 0 and np.isfinite(thr):
                    ratio = raw / (thr + 1e-12)
                    if ratio > best_ratio:
                        best_ratio = ratio
            combined[t] = best_ratio

        binary = (combined >= 1.0).astype(np.int32)

        return {
            'score': combined,
            'binary': binary,
            'per_scorer': per_scorer_scores,
            'thresholds': {s.name: float(s.threshold) for s in self.scorers},
            'cal_end': int(cal_end),
            'normalized': self.normalize,
        }
