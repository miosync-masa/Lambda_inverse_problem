"""
拡張ablation: 重み付け統合 / seed頑健性 / データサイズ / 異常率 / kernel mode / スコア相関。

検証1: 重み付き全部もり vs 重み付きjump+hybridのみ
検証2: 5つのseedで再現するか
検証3: kernel auto-select / RBF / Laplacian を試す
検証4: scorerスコア間の相関で冗長性を確認

各データセットについて analyze() は1回しか走らせず、scorerは使い回す。

Usage::

    python -m tests.ablation_v2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from lambda3_detector import L3Config, Lambda3ZeroShotDetector
from lambda3_detector.scorers import (
    HybridScorer,
    JumpScorer,
    KernelScorer,
    ScoreIntegrator,
    StructuralScorer,
)

from tests.datasets import create_complex_natural_dataset


# ===============================
# 重み付け統合の評価
# ===============================

WEIGHT_CONFIGS = {
    'production (j:0.20, h:0.35, k:0.30, s:0.15)': {
        'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15,
    },
    'jump+hybrid weighted (j:0.36, h:0.64)': {
        'jump': 0.20 / 0.55, 'hybrid': 0.35 / 0.55,
    },
    'jump+hybrid equal (j:0.5, h:0.5)': {
        'jump': 0.5, 'hybrid': 0.5,
    },
    'hybrid only': {'hybrid': 1.0},
    'kernel only': {'kernel': 1.0},
    'structural only': {'structural': 1.0},
    'jump only': {'jump': 1.0},
}


@dataclass
class ComponentScores:
    jump: np.ndarray
    hybrid: np.ndarray
    kernel: np.ndarray
    structural: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {'jump': self.jump, 'hybrid': self.hybrid,
                'kernel': self.kernel, 'structural': self.structural}


def _compute_all(events: np.ndarray,
                 detector: Lambda3ZeroShotDetector | None = None,
                 *,
                 kernel: KernelScorer | None = None) -> tuple[ComponentScores, object]:
    """analyze + 4 scorerを1回ずつ実行。"""
    detector = detector or Lambda3ZeroShotDetector(L3Config())
    np.random.seed(0)
    result = detector.analyze(events)

    np.random.seed(0); jump = JumpScorer().score(events, result)
    np.random.seed(0); hybrid = HybridScorer().score(events, result)
    np.random.seed(0)
    k = kernel or KernelScorer(kernel_type=1, degree=7, coef0=1.0)
    kernel_scores = k.score(events, result)
    np.random.seed(0); structural = StructuralScorer().score(events, result)

    return ComponentScores(jump, hybrid, kernel_scores, structural), result


def _eval_weights(components: ComponentScores,
                  labels: np.ndarray,
                  weights: Dict[str, float]) -> Dict[str, float]:
    integrator = ScoreIntegrator(default_weights=weights)
    available = {k: v for k, v in components.as_dict().items() if k in weights}
    scores = integrator.combine(available, weights=weights)
    auc = roc_auc_score(labels, scores)
    top10 = float(np.mean(labels[np.argsort(scores)[-10:]]))
    return {'auc': float(auc), 'top10': top10}


# ===============================
# 検証1+2: 重み付け × seed
# ===============================

def validation_weights_x_seeds(seeds: List[int] = (42, 123, 456, 789, 2024)) -> None:
    print("\n" + "=" * 78)
    print("検証1+2: 重み付け統合 × seed頑健性 (300 events x 10 feats, anomaly 15%)")
    print("=" * 78)

    matrix: Dict[str, List[float]] = {name: [] for name in WEIGHT_CONFIGS}
    matrix_top10: Dict[str, List[float]] = {name: [] for name in WEIGHT_CONFIGS}

    for seed in seeds:
        np.random.seed(seed)
        events, labels, _ = create_complex_natural_dataset(
            n_events=300, n_features=10, anomaly_ratio=0.15,
        )
        components, _ = _compute_all(events)
        print(f"\n  -- seed={seed} --")
        for name, weights in WEIGHT_CONFIGS.items():
            m = _eval_weights(components, labels, weights)
            matrix[name].append(m['auc'])
            matrix_top10[name].append(m['top10'])
            print(f"    {name:<48}  AUC={m['auc']:.4f}  Top10={m['top10']:.2f}")

    print("\n  -- 集計 (mean ± std across seeds) --")
    for name in WEIGHT_CONFIGS:
        aucs = np.array(matrix[name])
        top10s = np.array(matrix_top10[name])
        print(f"    {name:<48}  AUC={aucs.mean():.4f}±{aucs.std():.3f}  "
              f"Top10={top10s.mean():.2f}±{top10s.std():.2f}")


# ===============================
# 検証2': 異常率を変える
# ===============================

def validation_anomaly_ratio(seed: int = 42,
                              ratios: List[float] = (0.01, 0.05, 0.10, 0.15)) -> None:
    print("\n" + "=" * 78)
    print(f"検証2': 異常率の影響 (seed={seed}, 300 events x 10 feats)")
    print("=" * 78)

    key_configs = {
        k: WEIGHT_CONFIGS[k] for k in [
            'production (j:0.20, h:0.35, k:0.30, s:0.15)',
            'jump+hybrid weighted (j:0.36, h:0.64)',
            'hybrid only',
        ]
    }

    for ratio in ratios:
        np.random.seed(seed)
        events, labels, _ = create_complex_natural_dataset(
            n_events=300, n_features=10, anomaly_ratio=ratio,
        )
        n_anom = int(labels.sum())
        components, _ = _compute_all(events)
        print(f"\n  -- anomaly_ratio={ratio:.2f}  (#anomalies={n_anom}) --")
        for name, weights in key_configs.items():
            m = _eval_weights(components, labels, weights)
            print(f"    {name:<48}  AUC={m['auc']:.4f}  Top10={m['top10']:.2f}")


# ===============================
# 検証2'': データサイズを変える
# ===============================

def validation_data_size(seed: int = 42,
                          sizes: List[int] = (200, 500)) -> None:
    print("\n" + "=" * 78)
    print(f"検証2'': データサイズの影響 (seed={seed}, anomaly_ratio=0.15)")
    print("(注: 1000+ events は JIT後でも O(N^2) で長時間 → 200/500のみ計測)")
    print("=" * 78)

    key_configs = {
        k: WEIGHT_CONFIGS[k] for k in [
            'production (j:0.20, h:0.35, k:0.30, s:0.15)',
            'jump+hybrid weighted (j:0.36, h:0.64)',
            'hybrid only',
        ]
    }

    for n_events in sizes:
        np.random.seed(seed)
        events, labels, _ = create_complex_natural_dataset(
            n_events=n_events, n_features=10, anomaly_ratio=0.15,
        )
        components, _ = _compute_all(events)
        print(f"\n  -- n_events={n_events} --")
        for name, weights in key_configs.items():
            m = _eval_weights(components, labels, weights)
            print(f"    {name:<48}  AUC={m['auc']:.4f}  Top10={m['top10']:.2f}")


# ===============================
# 検証3: kernel mode を変える
# ===============================

def validation_kernel_modes(seed: int = 42) -> None:
    print("\n" + "=" * 78)
    print(f"検証3: kernel mode の影響 (seed={seed}, 300 events x 10 feats)")
    print("=" * 78)

    np.random.seed(seed)
    events, labels, _ = create_complex_natural_dataset(
        n_events=300, n_features=10, anomaly_ratio=0.15,
    )
    detector = Lambda3ZeroShotDetector(L3Config())
    np.random.seed(0)
    result = detector.analyze(events)

    kernel_variants = {
        'fixed Polynomial(d=7)': KernelScorer(kernel_type=1, degree=7, coef0=1.0),
        'fixed RBF(gamma=1.0)':  KernelScorer(kernel_type=0, gamma=1.0),
        'fixed Laplacian(g=0.5)': KernelScorer(kernel_type=3, gamma=0.5),
        'auto-select':            KernelScorer(kernel_type=-1),
    }

    for name, scorer in kernel_variants.items():
        np.random.seed(0)
        scores = scorer.score(events, result)
        auc = roc_auc_score(labels, scores)
        top10 = float(np.mean(labels[np.argsort(scores)[-10:]]))
        print(f"  {name:<28}  AUC={auc:.4f}  Top10={top10:.2f}")


# ===============================
# 検証4: スコア相関 (kernel が hybridに包含されるか)
# ===============================

def validation_score_correlations(seed: int = 42) -> None:
    print("\n" + "=" * 78)
    print(f"検証4: scorerスコア間の相関 (seed={seed}, 300 events x 10 feats)")
    print("=" * 78)

    np.random.seed(seed)
    events, labels, _ = create_complex_natural_dataset(
        n_events=300, n_features=10, anomaly_ratio=0.15,
    )
    components, _ = _compute_all(events)

    names = ['jump', 'hybrid', 'kernel', 'structural']
    score_arrs = [components.jump, components.hybrid,
                  components.kernel, components.structural]

    # Pearson
    print("\n  Pearson correlation matrix:")
    print("                " + "  ".join(f"{n:>10}" for n in names))
    for i, ni in enumerate(names):
        row = []
        for j in range(len(names)):
            r = np.corrcoef(score_arrs[i], score_arrs[j])[0, 1]
            row.append(f"{r:>+10.3f}")
        print(f"    {ni:<12}" + "  ".join(row))

    # Spearman相当（rankで再計算）
    from scipy.stats import spearmanr
    print("\n  Spearman correlation matrix:")
    print("                " + "  ".join(f"{n:>10}" for n in names))
    for i, ni in enumerate(names):
        row = []
        for j in range(len(names)):
            rho, _ = spearmanr(score_arrs[i], score_arrs[j])
            row.append(f"{rho:>+10.3f}")
        print(f"    {ni:<12}" + "  ".join(row))

    print("\n  目安: |r| > 0.9 → 強冗長 / 0.7-0.9 → 冗長気味 / <0.5 → 独立情報")


# ===============================
# main
# ===============================

if __name__ == "__main__":
    validation_weights_x_seeds()
    validation_anomaly_ratio()
    validation_data_size()
    validation_kernel_modes()
    validation_score_correlations()
    print("\n" + "=" * 78)
    print("ablation_v2 完了")
    print("=" * 78)
