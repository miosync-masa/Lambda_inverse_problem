"""
Polarity-symmetric scoring wrapper.

問題: 一部の scorer (HybridScorer, KernelScorer, ...) は「異常 = signal が逸脱（大きく）」
を前提に設計されている。「signal absence」型異常 (e.g. decay, fade) では、scorer は
**異常区間で逆に低スコア**を出すことがある (anti-correlation, AUC < 0.5)。
これを scorer の **polarity 問題** と呼ぶ。

解決: calibration 窓（既知正常区間の冒頭）で z-normalize し、絶対値を取る。
    z_t = (score_t - μ_cal) / σ_cal
    symmetric_t = |z_t|

これにより:
  - 正常からの **両方向** の逸脱 (上にも下にも) が等価に「異常」として現れる
  - scorer 間の scale 差も z-norm で吸収され、線形合成しやすくなる

calibration_frames は通常 "anomaly 開始より前の正常区間" の冒頭。change-point eval では
``info.n_normal_pre // 2`` のような半分を使う。
"""

from typing import Dict

import numpy as np


def polarity_symmetric_score(scores: np.ndarray,
                              calibration_frames: int,
                              eps: float = 1e-8) -> np.ndarray:
    """|z| from a calibration window.

    Args:
        scores: (n,) 任意の scorer 生出力
        calibration_frames: 冒頭の既知正常区間フレーム数（baseline 推定用）
        eps: 数値安定化

    Returns:
        symmetric: (n,) absolute z-scores（高いほど baseline から逸脱）
    """
    n = len(scores)
    if calibration_frames < 2:
        raise ValueError("calibration_frames >= 2 required")
    if calibration_frames >= n:
        raise ValueError(
            f"calibration_frames ({calibration_frames}) must be < total length ({n})"
        )
    cal = scores[:calibration_frames]
    mu = float(np.mean(cal))
    sigma = float(np.std(cal)) + eps
    z = (scores - mu) / sigma
    return np.abs(z)


def polarity_symmetric_dict(score_dict: Dict[str, np.ndarray],
                             calibration_frames: int,
                             eps: float = 1e-8) -> Dict[str, np.ndarray]:
    """複数 scorer 出力を一括 symmetric 化"""
    return {
        name: polarity_symmetric_score(s, calibration_frames, eps)
        for name, s in score_dict.items()
    }
