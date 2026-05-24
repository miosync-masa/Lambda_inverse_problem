"""
Regime-aware anomaly detection (semi-supervised, normal-label only)。

Lambda³-S streaming (zero-shot) の限界 = 先頭 15% calibration が複数 regime を
カバーできない (ambient_temp の季節 drift、ec2_cpu の load 多様性、etc.)。

本モジュールは industrial deployment の現実モデル:
    "過去の故障期間 (anomaly window) を **除外** した clean な歴史データを使って
     normal の regime 構造を学び、現在 frame を最も近い regime に対して評価する"

  - anomaly の "形" は学習しない (window 除外のみ、shape 情報不使用)
  - K=3 regime GMM クラスタ
  - 各 regime ごとに 6 scorer の per-threshold を fit
  - streaming: regime 分類 → 該当 regime の threshold で OR voting

正直な分類: **semi-supervised (normal-label only)**。
NAB context では combined_windows.json から anomaly_mask を作る。
production では operator-tagged post-mortem 記録に相当。
"""

from .regime_detector import (
    RegimeAwareDetector,
    adaptive_anomaly_mask,
    compute_robust_threshold,
    expand_anomaly_mask,
)

__all__ = [
    'RegimeAwareDetector',
    'adaptive_anomaly_mask',
    'compute_robust_threshold',
    'expand_anomaly_mask',
]
