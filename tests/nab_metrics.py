"""
NAB 公式 Sweeper を用いた scoring wrapper。

NAB の `nab.sweeper.Sweeper` をオプティマイザとして使い、
detector の anomaly_score 列を 3 profile (standard / low FP / low FN) で評価する。

normalization:
    score_normalized = 100 * (best_raw - null_baseline) / (perfect - null_baseline)
    null_baseline = threshold=1.1 (=何も検出しない) 時の Sweeper score
    perfect       = #windows * tpWeight

Usage::
    from tests.nab_metrics import score_all_profiles
    scores = score_all_profiles(sample, anomaly_scores)
    # scores = {'standard': NABScore, 'reward_low_FP_rate': ..., 'reward_low_FN_rate': ...}
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# NAB をパッケージ root の ./NAB に置く想定 (tests/ から見ると ../NAB)
_NAB_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'NAB')
)
if _NAB_ROOT not in sys.path:
    sys.path.insert(0, _NAB_ROOT)

from nab.sweeper import Sweeper  # noqa: E402

_PROFILES_PATH = os.path.join(_NAB_ROOT, 'config', 'profiles.json')
_PROFILES_CACHE: Dict[str, Dict] = {}


def load_profiles() -> Dict[str, Dict]:
    if not _PROFILES_CACHE:
        with open(_PROFILES_PATH) as f:
            _PROFILES_CACHE.update(json.load(f))
    return _PROFILES_CACHE


@dataclass
class NABScore:
    profile: str
    best_threshold: float
    best_raw: float
    null_baseline: float
    perfect: float
    normalized: float        # 0=null detector, 100=perfect detector
    tp: int
    fp: int
    fn: int
    n_windows: int


def _normalize_to_unit(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s)
    lo, hi = float(np.min(s[finite])), float(np.max(s[finite]))
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    out = np.where(finite, (s - lo) / (hi - lo), 0.0)
    return np.clip(out, 0.0, 1.0)


def score_with_sweeper(sample, anomaly_scores: np.ndarray,
                       profile_name: str = 'standard') -> NABScore:
    """1 file × 1 profile のスコア。

    Args:
        sample: NABSample (timestamps / windows_ts / name を参照)
        anomaly_scores: (n,) float. NaN は 0 に置換。range は自動で [0,1] に min-max。
        profile_name: 'standard' | 'reward_low_FP_rate' | 'reward_low_FN_rate'
    """
    profiles = load_profiles()
    cost = profiles[profile_name]['CostMatrix']
    sweeper = Sweeper(probationPercent=0.15, costMatrix=cost)

    scores01 = _normalize_to_unit(anomaly_scores)

    timestamps = sample.timestamps
    window_limits = [(s, e) for s, e in sample.windows_ts]

    anomaly_list = sweeper.calcSweepScore(
        timestamps, scores01.tolist(), window_limits, sample.name
    )
    threshold_scores = sweeper.calcScoreByThreshold(anomaly_list)
    # 最初の entry が threshold=1.1 (null baseline, 全 FN)
    null = threshold_scores[0]
    best = max(threshold_scores, key=lambda r: r.score)

    n_windows = len(sample.windows_ts)
    perfect = n_windows * cost['tpWeight']
    if perfect - null.score > 1e-12:
        normalized = 100.0 * (best.score - null.score) / (perfect - null.score)
    else:
        normalized = 0.0

    return NABScore(
        profile=profile_name,
        best_threshold=float(best.threshold),
        best_raw=float(best.score),
        null_baseline=float(null.score),
        perfect=float(perfect),
        normalized=float(normalized),
        tp=int(best.tp), fp=int(best.fp), fn=int(best.fn),
        n_windows=n_windows,
    )


def score_all_profiles(sample, anomaly_scores: np.ndarray
                       ) -> Dict[str, NABScore]:
    return {
        name: score_with_sweeper(sample, anomaly_scores, profile_name=name)
        for name in load_profiles().keys()
    }


def format_nab_score(s: NABScore, label: str = '') -> str:
    return (f"    {label:<24}  {s.profile:<22}  "
            f"raw={s.best_raw:+8.3f}  norm={s.normalized:+7.2f}  "
            f"thr={s.best_threshold:.3f}  TP={s.tp:3d} FP={s.fp:4d} FN={s.fn:3d}")
