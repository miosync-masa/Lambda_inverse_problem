"""
ベンチマークユーティリティ: AUC ／ Top-K 精度の評価。
"""

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from lambda3_detector import Lambda3ZeroShotDetector


def evaluate_performance(detector: Lambda3ZeroShotDetector,
                         events: np.ndarray,
                         labels: np.ndarray,
                         config: Dict[str, bool] = None) -> Dict[str, float]:
    """性能評価ユーティリティ"""

    if config is None:
        config = {
            "use_feature_optimization": True,
            "use_jump_analysis": False,
            "use_kernel_space": False,
            "use_ensemble": False
        }

    # Lambda³解析
    result = detector.analyze(events)

    # 異常検知
    scores = detector.detect_anomalies(result, events, **config)

    # AUC計算
    auc = roc_auc_score(labels, scores)

    # トップ10の精度
    top_10_indices = np.argsort(scores)[-10:]
    top_10_accuracy = np.mean(labels[top_10_indices])

    return {
        "auc": auc,
        "top_10_accuracy": top_10_accuracy,
        "config": config
    }
