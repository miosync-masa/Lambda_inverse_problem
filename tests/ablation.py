"""
Ablation study scaffolding: 各scorerを単独／組合せでAUC評価する。

Usage::

    python -m tests.ablation
"""

from itertools import combinations
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from lambda3_detector import L3Config, Lambda3ZeroShotDetector
from lambda3_detector.scorers import (
    AnomalyScorer,
    HybridScorer,
    JumpScorer,
    KernelScorer,
    ScoreIntegrator,
    StructuralScorer,
)


SCORER_BUILDERS = {
    'jump': JumpScorer,
    'hybrid': HybridScorer,
    'kernel': lambda: KernelScorer(kernel_type=1, degree=7, coef0=1.0),  # Polynomial
    'structural': StructuralScorer,
}


def _standardize(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    if s < 1e-10:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def evaluate_scorer(name: str,
                    scorer: AnomalyScorer,
                    events: np.ndarray,
                    result,
                    labels: np.ndarray) -> Dict[str, float]:
    """単独のscorerについてAUCとTop-10精度を測る。"""
    scores = scorer.score(events, result)
    auc = roc_auc_score(labels, scores)
    top10 = np.mean(labels[np.argsort(scores)[-10:]])
    return {'name': name, 'auc': float(auc), 'top10': float(top10)}


def evaluate_combination(names: List[str],
                          scorers: Dict[str, AnomalyScorer],
                          events: np.ndarray,
                          result,
                          labels: np.ndarray) -> Dict[str, float]:
    """指定したscorer群を等重みで合成した場合の性能。"""
    integrator = ScoreIntegrator(default_weights={n: 1.0 / len(names) for n in names})
    component_scores = {n: scorers[n].score(events, result) for n in names}
    combined = integrator.combine(component_scores)
    auc = roc_auc_score(labels, combined)
    top10 = np.mean(labels[np.argsort(combined)[-10:]])
    return {'names': names, 'auc': float(auc), 'top10': float(top10)}


def run_ablation(events: np.ndarray, labels: np.ndarray, seed: int = 42) -> None:
    """各scorer単独 + 全2/3/4組合せのAUCを出力する。"""
    np.random.seed(seed)
    detector = Lambda3ZeroShotDetector(L3Config())
    result = detector.analyze(events)

    scorers = {name: builder() for name, builder in SCORER_BUILDERS.items()}

    print("\n=== Single scorers ===")
    for name, scorer in scorers.items():
        metrics = evaluate_scorer(name, scorer, events, result, labels)
        print(f"  {metrics['name']:<10}  AUC={metrics['auc']:.4f}  Top10={metrics['top10']:.2f}")

    print("\n=== Combined scorers (equal weights) ===")
    all_names = list(scorers.keys())
    for k in range(2, len(all_names) + 1):
        for combo in combinations(all_names, k):
            metrics = evaluate_combination(list(combo), scorers, events, result, labels)
            label = '+'.join(combo)
            print(f"  {label:<35}  AUC={metrics['auc']:.4f}  Top10={metrics['top10']:.2f}")


if __name__ == "__main__":
    from tests.datasets import create_complex_natural_dataset

    np.random.seed(42)
    events, labels, _ = create_complex_natural_dataset(
        n_events=300, n_features=10, anomaly_ratio=0.15,
    )
    run_ablation(events, labels)
