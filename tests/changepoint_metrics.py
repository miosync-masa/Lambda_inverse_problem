"""
Change-point detection 用メトリクス。

産業利用シナリオでは AUC より以下が重要:
    - Time-to-Detect (TTD): 異常開始から検出までのフレーム数
    - False Alarm Rate before/after: 正常区間での誤検出
    - Recall in anomaly window: 異常区間でどれだけ検出できたか
    - Localization error: 「最初の検出位置」と true_start の差

閾値の決め方 (threshold_method):
    'percentile' (default, 産業標準):
        threshold = np.percentile(cal_scores, percentile_q)  (default q=99.0)
        - 「calibration 期間で1%の FAR」を実現する闘値
        - 重い尾を持つ score 分布にロバスト
    'mean_plus_sigma' (legacy):
        threshold = mean(cal) + k_threshold * std(cal)  (default k=3)
        - 正規分布前提
        - 外れ値混入で threshold が過大になる弱点

calibration 区間は anomaly_region より前の正常区間から自動切出（既定: n_normal_pre // 2）。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from .changepoint_datasets import ChangePointInfo


@dataclass
class ChangePointMetrics:
    auc: float                      # 参考: 通常 AUC
    detected: bool                  # 異常区間中に1回以上検出したか
    first_detection_idx: Optional[int]  # 全体最初の閾値超え位置（calibrationの後ろから探す）
    ttd: Optional[int]              # max(0, first_detection - true_start)、未検出なら None
    overshoot: Optional[int]        # first_detection が true_start より前に出た余分なフレーム数
    n_false_alarm_pre: int          # [0, true_start) での誤検出件数
    n_false_alarm_post: int         # [true_end, N) での誤検出件数
    far_pre: float                  # 上記の発生率
    far_post: float
    recall_in_window: float         # [true_start, true_end) のうち閾値超えの割合
    threshold: float                # 採用した閾値
    cal_mean: float                 # calibrationの平均
    cal_std: float                  # calibrationのstd

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_changepoint(scores: np.ndarray,
                         labels: np.ndarray,
                         info: ChangePointInfo,
                         *,
                         calibration_frames: Optional[int] = None,
                         threshold_method: str = 'percentile',
                         percentile_q: float = 99.0,
                         k_threshold: float = 3.0) -> ChangePointMetrics:
    """1本のスコア時系列に対する change-point 評価。

    Args:
        scores: (n,) 異常スコア（高いほど異常）
        labels: (n,) 0/1
        info: ChangePointInfo (true_start/end)
        calibration_frames: 閾値学習に使う先頭フレーム数。
            既定: n_normal_pre の半分（必ず正常区間内に収まる）
        threshold_method: 'percentile' (default) or 'mean_plus_sigma' (legacy)
        percentile_q: percentile mode の分位点 (default 99.0)
        k_threshold: mean_plus_sigma mode の倍数 (default 3.0)
    """
    n = len(scores)
    true_start = info.true_start
    true_end = info.true_end

    if calibration_frames is None:
        calibration_frames = max(10, info.n_normal_pre // 2)
    calibration_frames = min(calibration_frames, true_start - 1)
    if calibration_frames < 5:
        raise ValueError("calibration_frames が短すぎます。n_normal_pre を大きく。")

    # === 閾値計算 ===
    cal = scores[:calibration_frames]
    cal_mean = float(np.mean(cal))
    cal_std = float(np.std(cal)) + 1e-12
    if threshold_method == 'percentile':
        threshold = float(np.percentile(cal, percentile_q))
    elif threshold_method == 'mean_plus_sigma':
        threshold = cal_mean + k_threshold * cal_std
    else:
        raise ValueError(
            f"Unknown threshold_method={threshold_method!r}. "
            f"Use 'percentile' or 'mean_plus_sigma'."
        )

    # === 検出マスク ===
    above = scores > threshold

    # === 最初の検出（calibrationを除く全範囲）===
    search_start = calibration_frames  # calibration中の自己検出は除く
    if above[search_start:].any():
        first_detection_idx = int(np.argmax(above[search_start:])) + search_start
    else:
        first_detection_idx = None

    # === TTD と overshoot ===
    if first_detection_idx is None:
        ttd = None
        overshoot = None
    elif first_detection_idx >= true_start:
        ttd = first_detection_idx - true_start
        overshoot = 0
    else:
        ttd = 0
        overshoot = true_start - first_detection_idx  # 早すぎる検出（FA）

    # === false alarms ===
    pre_mask = above[search_start:true_start]
    post_mask = above[true_end:]
    n_fa_pre = int(pre_mask.sum())
    n_fa_post = int(post_mask.sum())
    far_pre = float(n_fa_pre / max(len(pre_mask), 1))
    far_post = float(n_fa_post / max(len(post_mask), 1))

    # === 異常区間内 recall ===
    in_window_above = above[true_start:true_end]
    recall = float(in_window_above.mean())

    # === detected (any) ===
    detected = bool(in_window_above.any())

    # === 全体AUC ===
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float('nan')

    return ChangePointMetrics(
        auc=auc,
        detected=detected,
        first_detection_idx=first_detection_idx,
        ttd=ttd,
        overshoot=overshoot,
        n_false_alarm_pre=n_fa_pre,
        n_false_alarm_post=n_fa_post,
        far_pre=far_pre,
        far_post=far_post,
        recall_in_window=recall,
        threshold=threshold,
        cal_mean=cal_mean,
        cal_std=cal_std,
    )


def format_metrics(m: ChangePointMetrics, name: str = "") -> str:
    """1行サマリーフォーマット"""
    ttd_str = f"{m.ttd:>4d}" if m.ttd is not None else "  --"
    over_str = f"{m.overshoot:>3d}" if m.overshoot is not None else " --"
    det_idx_str = f"{m.first_detection_idx}" if m.first_detection_idx is not None else "--"
    return (f"  {name:<14}  AUC={m.auc:.4f}  detected={'Y' if m.detected else 'N'}  "
            f"TTD={ttd_str}  over={over_str}  "
            f"FA(pre/post)={m.n_false_alarm_pre:>3d}/{m.n_false_alarm_post:>3d}  "
            f"recall={m.recall_in_window:.3f}  "
            f"first@{det_idx_str}")
