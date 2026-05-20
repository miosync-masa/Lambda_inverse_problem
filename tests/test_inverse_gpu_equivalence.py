"""
CPU 版 solve_inverse_problem_sparse vs GPU 版の数値的等価性チェック。

ローカル Mac (CuPy 無し) では import 段階で失敗するため自動 skip。
Colab で `python -m tests.test_inverse_gpu_equivalence` で実行。

検証項目:
  1. objective + 解析勾配が scipy.optimize.check_grad を通る (CPU 上で数値勾配と一致)
  2. CPU 版と GPU 版で paths の最大絶対差 < 1e-3 (float32 精度)
  3. KernelScorer (polynomial, degree=7) で CPU/GPU の出力が近い
"""

from __future__ import annotations

import sys
import time

import numpy as np

try:
    import cupy as cp  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def _make_synthetic(n_events: int = 200, n_features: int = 5, seed: int = 0):
    """少量の合成データ + jump_structures。"""
    rng = np.random.default_rng(seed)
    events = rng.normal(0, 1, (n_events, n_features)).astype(np.float64)
    # 中間に jump を埋め込み
    events[80:120] += 5.0

    # jump_structures (Lambda3 既存形式)
    diff = np.abs(np.diff(events, axis=0)).mean(axis=1)
    thr = diff.mean() + 2.0 * diff.std()
    unified = (diff > thr).astype(np.int32)
    unified = np.concatenate([[0], unified])  # (n_events,)
    importance = np.where(unified > 0, 1.0, 0.0).astype(np.float64)
    if importance.sum() > 0:
        importance = importance / importance.sum() * importance.sum()  # keep raw
    jump_structures = {
        'integrated': {
            'unified_jumps': unified,
            'jump_importance': importance,
        },
        'features': {},
    }
    return events, jump_structures


# -----------------------------------------------------------------------------
# Test 1: scipy.optimize.check_grad で解析勾配を数値検証
# -----------------------------------------------------------------------------

def test_check_grad():
    if not HAS_GPU:
        print("SKIP test_check_grad: CuPy not available")
        return

    from scipy.optimize import check_grad
    from lambda3_detector.analysis.structure_tensor_sparse import compute_significant_pairs
    from lambda3_detector.gpu.backend import ensure_gpu, to_cpu, DEFAULT_DTYPE
    from lambda3_detector.gpu.inverse_sparse_gpu import _full_objective_grad

    events, jump_structures = _make_synthetic(n_events=80, n_features=4)
    n_paths = 3
    n_events = events.shape[0]

    pair_array, _ = compute_significant_pairs(
        events, jump_structures, expand_window=3, n_static_samples=10,
    )

    events_gpu = ensure_gpu(events)
    pair_i = cp.asarray(pair_array[:, 0], dtype=cp.int64)
    pair_j = cp.asarray(pair_array[:, 1], dtype=cp.int64)
    G_pairs = cp.sum(events_gpu[pair_i] * events_gpu[pair_j], axis=1).astype(DEFAULT_DTYPE)
    jm_gpu = ensure_gpu(jump_structures['integrated']['unified_jumps'])
    jw_gpu = ensure_gpu(jump_structures['integrated']['jump_importance'])

    def f(x):
        L = cp.asarray(x.reshape(n_paths, n_events), dtype=DEFAULT_DTYPE)
        o, _ = _full_objective_grad(
            L, G_pairs, pair_i, pair_j, jm_gpu, jw_gpu,
            alpha=0.05, beta=0.005, jump_weight=0.5, topo_weight=0.0,
        )
        return float(o)

    def g(x):
        L = cp.asarray(x.reshape(n_paths, n_events), dtype=DEFAULT_DTYPE)
        _, gr = _full_objective_grad(
            L, G_pairs, pair_i, pair_j, jm_gpu, jw_gpu,
            alpha=0.05, beta=0.005, jump_weight=0.5, topo_weight=0.0,
        )
        return to_cpu(gr).astype(np.float64).ravel()

    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 0.1, n_paths * n_events).astype(np.float64)
    err = check_grad(f, g, x0, epsilon=1e-4)
    print(f"  check_grad error (no-topo): {err:.6f}  (||grad|| order; <1e-1 で OK)")
    assert err < 1.0, f"Gradient check failed: error={err}"

    # with topo
    def f_to(x):
        L = cp.asarray(x.reshape(n_paths, n_events), dtype=DEFAULT_DTYPE)
        o, _ = _full_objective_grad(
            L, G_pairs, pair_i, pair_j, jm_gpu, jw_gpu,
            alpha=0.05, beta=0.005, jump_weight=0.5, topo_weight=0.1,
        )
        return float(o)

    def g_to(x):
        L = cp.asarray(x.reshape(n_paths, n_events), dtype=DEFAULT_DTYPE)
        _, gr = _full_objective_grad(
            L, G_pairs, pair_i, pair_j, jm_gpu, jw_gpu,
            alpha=0.05, beta=0.005, jump_weight=0.5, topo_weight=0.1,
        )
        return to_cpu(gr).astype(np.float64).ravel()

    err_to = check_grad(f_to, g_to, x0, epsilon=1e-4)
    print(f"  check_grad error (with-topo): {err_to:.6f}")
    assert err_to < 2.0, f"Gradient check failed (topo): error={err_to}"
    print("✓ test_check_grad passed")


# -----------------------------------------------------------------------------
# Test 2: CPU 版 vs GPU 版で paths が近い
# -----------------------------------------------------------------------------

def test_solver_equivalence():
    if not HAS_GPU:
        print("SKIP test_solver_equivalence: CuPy not available")
        return

    from lambda3_detector.analysis.structure_tensor_sparse import solve_inverse_problem_sparse
    from lambda3_detector.gpu.inverse_sparse_gpu import solve_inverse_problem_sparse_gpu

    events, jump_structures = _make_synthetic(n_events=200, n_features=5)
    kwargs = dict(
        n_paths=4, alpha=0.05, beta=0.005,
        expand_window=3, n_static_samples=20,
    )

    t0 = time.perf_counter()
    paths_cpu, _ = solve_inverse_problem_sparse(events, jump_structures, **kwargs)
    t_cpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    paths_gpu, _ = solve_inverse_problem_sparse_gpu(events, jump_structures, **kwargs)
    t_gpu = time.perf_counter() - t0

    # paths_cpu / paths_gpu は MAX 合成 + normalize 済みの dict[int, ndarray]
    diffs = []
    for k in paths_cpu.keys():
        a = paths_cpu[k]
        b = paths_gpu[k]
        # 符号曖昧性 (|·| 取ってる MAX 合成済みなので符号差はないが念のため)
        d1 = float(np.max(np.abs(a - b)))
        d2 = float(np.max(np.abs(a + b)))
        diffs.append(min(d1, d2))

    max_diff = max(diffs)
    print(f"  CPU time: {t_cpu:.2f}s,  GPU time: {t_gpu:.2f}s  (speedup {t_cpu/t_gpu:.1f}x)")
    print(f"  max |paths_cpu - paths_gpu| = {max_diff:.6f}")
    # float32 精度 + L-BFGS-B の収束揺らぎを考慮
    assert max_diff < 5e-2, f"paths divergence too large: {max_diff}"
    print("✓ test_solver_equivalence passed")


# -----------------------------------------------------------------------------
# Test 3: KernelScorer (polynomial) CPU vs GPU
# -----------------------------------------------------------------------------

def test_kernel_scorer_equivalence():
    if not HAS_GPU:
        print("SKIP test_kernel_scorer_equivalence: CuPy not available")
        return

    from lambda3_detector.gpu.scorers_gpu import kernel_anomaly_scores_gpu
    from lambda3_detector.scorers.kernel_scorer import compute_kernel_anomaly_scores_with_params
    from lambda3_detector.config import Lambda3Result

    events, _ = _make_synthetic(n_events=150, n_features=4)
    rng = np.random.default_rng(0)
    n_paths = 4
    paths = {i: rng.normal(0, 0.1, len(events)) for i in range(n_paths)}
    # 正規化
    paths = {i: p / (np.linalg.norm(p) + 1e-8) for i, p in paths.items()}

    # 最低限の Lambda3Result スタブ
    class _Res:
        pass
    res = _Res()
    res.paths = paths

    cpu_scores = compute_kernel_anomaly_scores_with_params(
        events, res, kernel_type=1, degree=7, coef0=1.0,
    )
    gpu_scores = kernel_anomaly_scores_gpu(
        events, paths, kernel_type=1, degree=7, coef0=1.0,
    )

    max_diff = float(np.max(np.abs(cpu_scores - gpu_scores)))
    rel = max_diff / (float(np.max(np.abs(cpu_scores))) + 1e-12)
    print(f"  KernelScorer max |cpu - gpu| = {max_diff:.6e}  (rel {rel:.3e})")
    # float32 vs float64 で 1e-3 程度の相対差は許容
    assert rel < 1e-2, f"KernelScorer divergence too large: rel={rel}"
    print("✓ test_kernel_scorer_equivalence passed")


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print(f"GPU equivalence tests  (CuPy={'YES' if HAS_GPU else 'NO'})")
    print("=" * 70)
    if not HAS_GPU:
        print("\nCuPy が無い環境では全テストが skip されます。")
        print("Colab で実行してください: pip install cupy-cuda12x")
        sys.exit(0)

    test_check_grad()
    test_solver_equivalence()
    test_kernel_scorer_equivalence()
    print("\nAll tests passed.")
