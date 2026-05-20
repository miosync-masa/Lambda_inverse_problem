"""
Sparse vs Full inverse solver を、4種類の異常シナリオごとに評価。

シナリオ:
  - progressive_degradation  : structural_decay → cascade → topological_jump
  - periodic_burst           : periodic / pulse / resonance を混ぜる
  - chaotic_bifurcation      : bifurcation / multi_path / phase_jump を同時印加
  - partial_anomaly          : partial_periodic / superposition を一部特徴に印加

各シナリオ × 2 seeds で full / sparse を比較。

Usage::
    python -m tests.benchmark_sparse_per_scenario
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from lambda3_detector import L3Config, Lambda3Result
from lambda3_detector.analysis.multiscale_jumps import detect_multiscale_jumps
from lambda3_detector.analysis.physical_quantities import (
    classify_structures,
    compute_jump_aware_topology,
    compute_jump_conditional_entropies,
    compute_pulsation_energies,
)
from lambda3_detector.analysis.structure_tensor import inverse_problem_jump_constrained
from lambda3_detector.analysis.structure_tensor_sparse import solve_inverse_problem_sparse
from lambda3_detector.core.adaptive_params import compute_adaptive_window_size
from lambda3_detector import detector as _detector_mod
from lambda3_detector.scorers import (
    HybridScorer,
    JumpScorer,
    KernelScorer,
    ScoreIntegrator,
    StructuralScorer,
)

from tests.datasets import create_complex_natural_dataset


SCENARIOS = [
    'progressive_degradation',
    'periodic_burst',
    'chaotic_bifurcation',
    'partial_anomaly',
]


def _build_result(events, paths, jump_structures) -> Lambda3Result:
    charges, stabilities = compute_jump_aware_topology(paths, jump_structures)
    energies = compute_pulsation_energies(paths, jump_structures)
    entropies = compute_jump_conditional_entropies(paths, jump_structures)
    classifications = classify_structures(paths, charges, stabilities, jump_structures)
    return Lambda3Result(
        paths=paths, topological_charges=charges, stabilities=stabilities,
        energies=energies, entropies=entropies, classifications=classifications,
        jump_structures=jump_structures,
    )


def _score(events, result, labels) -> Dict[str, float]:
    np.random.seed(0); jump = JumpScorer().score(events, result)
    np.random.seed(0); hybrid = HybridScorer().score(events, result)
    np.random.seed(0); kernel = KernelScorer(kernel_type=1, degree=7, coef0=1.0).score(events, result)
    np.random.seed(0); structural = StructuralScorer().score(events, result)

    integrator = ScoreIntegrator(default_weights={
        'jump': 0.20, 'hybrid': 0.35, 'kernel': 0.30, 'structural': 0.15,
    })
    combined = integrator.combine({
        'jump': jump, 'hybrid': hybrid, 'kernel': kernel, 'structural': structural,
    })
    return {
        'auc_combined': float(roc_auc_score(labels, combined)),
        'auc_jump':     float(roc_auc_score(labels, jump)),
        'auc_hybrid':   float(roc_auc_score(labels, hybrid)),
        'auc_kernel':   float(roc_auc_score(labels, kernel)),
        'auc_struct':   float(roc_auc_score(labels, structural)),
        'top10':        float(np.mean(labels[np.argsort(combined)[-10:]])),
    }


def benchmark_scenario(scenario: str, seed: int, n_events: int = 300,
                        n_features: int = 10, anomaly_ratio: float = 0.15) -> Dict:
    np.random.seed(seed)
    events, labels, details = create_complex_natural_dataset(
        n_events=n_events, n_features=n_features, anomaly_ratio=anomaly_ratio,
        scenario_filter=scenario,
    )
    n_anom = int(labels.sum())

    # adaptive window前処理（detector.analyzeと同じ）
    aw = compute_adaptive_window_size(events)
    _detector_mod._config.update_global_constants(aw)

    config = L3Config()
    jump_structures = detect_multiscale_jumps(events)

    # ----- full -----
    t0 = time.perf_counter()
    paths_full = inverse_problem_jump_constrained(
        events, jump_structures, config.n_paths, config.alpha, config.beta,
    )
    t_full = time.perf_counter() - t0
    metrics_full = _score(events, _build_result(events, paths_full, jump_structures), labels)

    # ----- sparse -----
    t0 = time.perf_counter()
    paths_sparse, stats = solve_inverse_problem_sparse(
        events, jump_structures, config.n_paths, config.alpha, config.beta,
        expand_window=5, n_static_samples=50, verbose=False,
    )
    t_sparse = time.perf_counter() - t0
    metrics_sparse = _score(events, _build_result(events, paths_sparse, jump_structures), labels)

    return {
        'scenario': scenario, 'seed': seed, 'n_anom': n_anom,
        'sparse_reduction': stats['reduction_ratio'],
        'time_full': t_full, 'time_sparse': t_sparse,
        'speedup': t_full / max(t_sparse, 1e-9),
        'full': metrics_full, 'sparse': metrics_sparse,
    }


def main():
    print("=" * 96)
    print("Sparse vs Full per anomaly scenario (300 events x 10 feats, anomaly 15%)")
    print("=" * 96)

    seeds = [42, 123]
    all_results: List[Dict] = []

    for scenario in SCENARIOS:
        print(f"\n■ scenario = {scenario}")
        for seed in seeds:
            r = benchmark_scenario(scenario, seed)
            all_results.append(r)
            f = r['full']; s = r['sparse']
            print(f"  seed={seed:<4}  #anom={r['n_anom']:>2}  "
                  f"reduction={r['sparse_reduction']*100:.1f}%  "
                  f"speedup={r['speedup']:.2f}x")
            print(f"           [full  ] combined={f['auc_combined']:.4f}  "
                  f"hybrid={f['auc_hybrid']:.4f}  jump={f['auc_jump']:.4f}  "
                  f"kernel={f['auc_kernel']:.4f}  struct={f['auc_struct']:.4f}  "
                  f"Top10={f['top10']:.2f}")
            print(f"           [sparse] combined={s['auc_combined']:.4f}  "
                  f"hybrid={s['auc_hybrid']:.4f}  jump={s['auc_jump']:.4f}  "
                  f"kernel={s['auc_kernel']:.4f}  struct={s['auc_struct']:.4f}  "
                  f"Top10={s['top10']:.2f}")
            d_combined = s['auc_combined'] - f['auc_combined']
            d_hybrid   = s['auc_hybrid']   - f['auc_hybrid']
            print(f"           [delta ] Δcombined={d_combined:+.4f}  "
                  f"Δhybrid={d_hybrid:+.4f}")

    print("\n" + "=" * 96)
    print("シナリオ別集計 (mean across seeds)")
    print("=" * 96)
    print(f"  {'scenario':<24}  "
          f"{'full_comb':>10}  {'sprs_comb':>10}  {'Δcomb':>8}  "
          f"{'full_hyb':>10}  {'sprs_hyb':>10}  {'Δhyb':>8}  "
          f"{'speedup':>8}")
    for scenario in SCENARIOS:
        subset = [r for r in all_results if r['scenario'] == scenario]
        if not subset:
            continue
        m = lambda key: np.mean([r['full'][key] for r in subset])
        ms = lambda key: np.mean([r['sparse'][key] for r in subset])
        sp = np.mean([r['speedup'] for r in subset])
        print(f"  {scenario:<24}  "
              f"{m('auc_combined'):>10.4f}  {ms('auc_combined'):>10.4f}  "
              f"{ms('auc_combined') - m('auc_combined'):>+8.4f}  "
              f"{m('auc_hybrid'):>10.4f}  {ms('auc_hybrid'):>10.4f}  "
              f"{ms('auc_hybrid') - m('auc_hybrid'):>+8.4f}  "
              f"{sp:>8.2f}")

    print("\n" + "=" * 96)
    print("シナリオ別 scorer の効きを比較 (sparseのAUC, mean across seeds)")
    print("=" * 96)
    print(f"  {'scenario':<24}  {'jump':>8}  {'hybrid':>8}  "
          f"{'kernel':>8}  {'struct':>8}  {'combined':>10}")
    for scenario in SCENARIOS:
        subset = [r for r in all_results if r['scenario'] == scenario]
        if not subset:
            continue
        ms = lambda key: np.mean([r['sparse'][key] for r in subset])
        print(f"  {scenario:<24}  "
              f"{ms('auc_jump'):>8.4f}  {ms('auc_hybrid'):>8.4f}  "
              f"{ms('auc_kernel'):>8.4f}  {ms('auc_struct'):>8.4f}  "
              f"{ms('auc_combined'):>10.4f}")


if __name__ == "__main__":
    main()
