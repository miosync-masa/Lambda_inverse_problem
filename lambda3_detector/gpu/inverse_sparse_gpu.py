"""
スパース版逆問題の GPU 実装（CuPy + 解析勾配）。

CPU 版 (`analysis/structure_tensor_sparse.solve_inverse_problem_sparse`) と
数値的に等価な目的関数を、scipy.optimize.minimize の jac=True に渡せる
``(obj, grad)`` 形式で評価する。

数値勾配 (CPU 版) では 1 iter あたり O(n_paths * n_events) 回の objective
評価が必要だった (13k 次元で 1.8k 行 21分)。解析勾配を 1 回計算するだけで
全勾配を得るため、1000〜10000 倍の高速化が見込める。

実装:
  - events / pair_array / G[pairs] / jump_mask / jump_weights は GPU 常駐
  - objective + grad は 1 pass で計算 (forward の e_ij を grad で再利用)
  - data_fit grad の scatter-add は CuPy RawKernel (atomicAdd) で実装
  - 各正則化項 (TV / L1 / jump_reg / jump_consistency / topo_reg) は
    全て解析勾配。subgrad は sign() = x/(|x|+eps) で平滑化。
  - L-BFGS-B 自体は scipy (CPU) で動かし、callback で GPU 評価する。
    Λ_flat の host↔device 転送 (n_paths*n_events*4byte) はマイクロ秒級。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from .backend import cp, DEFAULT_DTYPE, ensure_gpu, to_cpu
from ..analysis.structure_tensor_sparse import compute_significant_pairs


# =============================================================================
# CUDA RawKernel: data_fit gradient scatter (atomicAdd)
# =============================================================================
#
# 各ペア (i, j) ∈ Ω の寄与:
#     ∂ data_fit / ∂Λ[p, i] += -2 * e_ij * Λ[p, j]    for all p
# atomicAdd で grad[p, i] に蓄積する。
#
# pair が (i, j), (j, i) の両方を含むため、対称な寄与は重複加算で自然に表現される。

_DATA_FIT_GRAD_KERNEL_SRC = r"""
extern "C" __global__
void data_fit_grad_kernel(
    const float* __restrict__ Lambda,   // (n_paths * n_events,) row-major
    const float* __restrict__ e_pairs,  // (n_pairs,)  = G_ij - <Λ_i, Λ_j>
    const long long* __restrict__ pair_i,    // (n_pairs,)
    const long long* __restrict__ pair_j,    // (n_pairs,)
    float* __restrict__ grad,           // (n_paths * n_events,)
    const int n_paths,
    const int n_events,
    const int n_pairs)
{
    // 各ペア (i, j) のスカラー項 (G_ij - <Λ_i, Λ_j>)^2 の勾配は
    //   ∂/∂Λ[p, i] += -2 e_ij Λ[p, j]
    //   ∂/∂Λ[p, j] += -2 e_ij Λ[p, i]
    // の両方に乗る。対角 (i == j) は同じアドレスに 2 回加算され、
    // ∂(G_ii - Σ Λ_ki^2)^2/∂Λ[p, i] = -4 e_ii Λ[p, i] が自動的に再現される。
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pairs) return;
    int i = (int)pair_i[k];
    int j = (int)pair_j[k];
    float scale = -2.0f * e_pairs[k];
    for (int p = 0; p < n_paths; ++p) {
        float Lpi = Lambda[p * n_events + i];
        float Lpj = Lambda[p * n_events + j];
        atomicAdd(&grad[p * n_events + i], scale * Lpj);
        atomicAdd(&grad[p * n_events + j], scale * Lpi);
    }
}
"""

_data_fit_grad_kernel = None  # lazy init (CuPy が無いとコンパイル不可)


def _get_data_fit_grad_kernel():
    global _data_fit_grad_kernel
    if _data_fit_grad_kernel is None:
        _data_fit_grad_kernel = cp.RawKernel(
            _DATA_FIT_GRAD_KERNEL_SRC, "data_fit_grad_kernel"
        )
    return _data_fit_grad_kernel


# =============================================================================
# Smooth sign: subgrad の平滑化
# =============================================================================

def _smooth_sign(x: "cp.ndarray", eps: float = 1e-8) -> "cp.ndarray":
    """sign(x) を eps で平滑化。x=0 で 0、|x|>>eps で ±1。"""
    return x / (cp.abs(x) + eps)


# =============================================================================
# 各項の objective + gradient
# =============================================================================

def _data_fit_obj_grad(
    Lambda: "cp.ndarray",      # (n_paths, n_events)
    G_pairs: "cp.ndarray",     # (n_pairs,) precomputed gram on selected pairs
    pair_i: "cp.ndarray",      # (n_pairs,) int64
    pair_j: "cp.ndarray",      # (n_pairs,) int64
    grad: "cp.ndarray",        # (n_paths, n_events) in/out — 加算先
) -> float:
    """data_fit = Σ_{(i,j)∈Ω} (G_ij - <Λ_i, Λ_j>)^2 の値と grad 寄与。"""
    n_paths, n_events = Lambda.shape
    n_pairs = pair_i.shape[0]

    # recon_pairs[k] = <Λ[:, pair_i[k]], Λ[:, pair_j[k]]>
    recon_pairs = cp.sum(
        Lambda[:, pair_i] * Lambda[:, pair_j], axis=0
    )  # (n_pairs,) float32
    e_pairs = G_pairs - recon_pairs
    # objective は float64 で reduce（float32 の合算で生じる ULP ノイズが
    # check_grad の数値勾配を埋めてしまうのを防ぐ）
    data_fit = float(cp.sum((e_pairs.astype(cp.float64)) ** 2))

    # grad scatter
    block = 256
    grid = (n_pairs + block - 1) // block
    kernel = _get_data_fit_grad_kernel()
    kernel(
        (grid,), (block,),
        (
            Lambda.ravel(),
            e_pairs.astype(cp.float32, copy=False),
            pair_i, pair_j,
            grad.ravel(),
            np.int32(n_paths), np.int32(n_events), np.int32(n_pairs),
        ),
    )
    return data_fit


def _tv_obj_grad(Lambda: "cp.ndarray", alpha: float,
                 grad: "cp.ndarray") -> float:
    """TV = α (Σ|Λ[p+1,j]-Λ[p,j]| + Σ|Λ[p,j+1]-Λ[p,j]|) の値と grad。"""
    if alpha == 0.0:
        return 0.0
    # vertical (paths 方向)
    dv = Lambda[1:, :] - Lambda[:-1, :]
    abs_dv_sum = float(cp.sum(cp.abs(dv).astype(cp.float64)))
    sgn_dv = _smooth_sign(dv)  # (n_paths-1, n_events)
    # grad[p, j] += α·sign(Λ[p,j]-Λ[p-1,j]) (p>0) - α·sign(Λ[p+1,j]-Λ[p,j]) (p<n_p-1)
    grad[1:, :] += alpha * sgn_dv
    grad[:-1, :] -= alpha * sgn_dv

    # horizontal (time 方向)
    dh = Lambda[:, 1:] - Lambda[:, :-1]
    abs_dh_sum = float(cp.sum(cp.abs(dh).astype(cp.float64)))
    sgn_dh = _smooth_sign(dh)  # (n_paths, n_events-1)
    grad[:, 1:] += alpha * sgn_dh
    grad[:, :-1] -= alpha * sgn_dh

    return alpha * (abs_dv_sum + abs_dh_sum)


def _l1_obj_grad(Lambda: "cp.ndarray", beta: float,
                 grad: "cp.ndarray") -> float:
    """L1 = β·Σ|Λ| の値と grad。"""
    if beta == 0.0:
        return 0.0
    val = float(cp.sum(cp.abs(Lambda).astype(cp.float64)))
    grad += beta * _smooth_sign(Lambda)
    return beta * val


def _jump_reg_obj_grad(
    Lambda: "cp.ndarray", jump_weight: float, grad: "cp.ndarray",
) -> float:
    """Adaptive jump-aware regularizer (per-path)。

    delta = |Λ[p, k+1] - Λ[p, k]|; threshold = mean(delta)+2.5*std(delta) per path
    Σ over delta > threshold of jump_weight*delta。

    threshold は Λ に依存するが慣例どおり stop-gradient で扱う（学習時 RBN 等と同様）。
    """
    n_paths, n_events = Lambda.shape
    if n_events < 2:
        return 0.0

    d = Lambda[:, 1:] - Lambda[:, :-1]        # (n_paths, n_events-1)
    ad = cp.abs(d)
    mean_d = cp.mean(ad, axis=1, keepdims=True)
    std_d = cp.std(ad, axis=1, keepdims=True)
    thr = mean_d + 2.5 * std_d                 # (n_paths, 1)

    mask = ad > thr                            # (n_paths, n_events-1) bool
    total = float(
        cp.sum((ad * mask.astype(ad.dtype)).astype(cp.float64))
    ) * jump_weight

    # 勾配: 越えてる箇所の |d| の subgrad = sign(d)
    contrib = jump_weight * _smooth_sign(d) * mask.astype(d.dtype)
    grad[:, 1:] += contrib
    grad[:, :-1] -= contrib
    return total


def _jump_consistency_obj_grad(
    Lambda: "cp.ndarray",
    jump_mask_gpu: "cp.ndarray",     # (n_events,) float32 0/1
    jump_weights_gpu: "cp.ndarray",  # (n_events,) float32
    grad: "cp.ndarray",
) -> float:
    """
    compute_jump_consistency_term 相当 (CPU 版):
        for p, for i in [1, n_e):
            delta = |Λ[p,i] - Λ[p,i-1]|
            if jump_mask[i] == 1: consistency -= jump_weights[i] * delta
            else:                 consistency += 0.1 * delta
    """
    n_paths, n_events = Lambda.shape
    if n_events < 2:
        return 0.0

    d = Lambda[:, 1:] - Lambda[:, :-1]    # (n_p, n_e-1)
    ad = cp.abs(d)
    # i = 1..n_e-1 の重み (jump_mask[1:], jump_weights[1:])
    jm = jump_mask_gpu[1:].astype(Lambda.dtype)    # (n_e-1,)
    jw = jump_weights_gpu[1:].astype(Lambda.dtype)  # (n_e-1,)
    # 各 i のスカラー係数 c_i: jump_mask==1 → -jw[i], それ以外 → +0.1
    c = cp.where(jm > 0.5, -jw, 0.1 * cp.ones_like(jw))  # (n_e-1,)

    val = float(cp.sum((c[None, :] * ad).astype(cp.float64)))  # broadcast

    # ∂/∂d = c · sign(d)
    contrib = c[None, :] * _smooth_sign(d)  # (n_p, n_e-1)
    grad[:, 1:] += contrib
    grad[:, :-1] -= contrib
    return val


def _topo_reg_obj_grad(
    Lambda: "cp.ndarray", topo_weight: float, grad: "cp.ndarray",
    eps: float = 1e-8,
) -> float:
    """topo_reg = topo_weight · Σ_p Σ (wrap(phase_diff))^2

    phase[i] = atan2(path[i+1], path[i]),  raw_q[m] = phase[m+1] - phase[m],
    q = wrap(raw_q) ∈ (-π, π]

    wrapping は constant w.r.t. Λ なので ∂q/∂Λ = ∂raw_q/∂Λ。
    """
    n_paths, n_events = Lambda.shape
    if topo_weight == 0.0 or n_events < 3:
        return 0.0

    a = Lambda[:, :-1]           # path[i],   shape (n_p, n_e-1)
    b = Lambda[:, 1:]            # path[i+1], shape (n_p, n_e-1)
    D = a * a + b * b + eps
    phase = cp.arctan2(b, a)     # (n_p, n_e-1)
    raw_q = phase[:, 1:] - phase[:, :-1]   # (n_p, n_e-2)
    # wrap to (-π, π]
    q = raw_q - cp.round(raw_q / (2.0 * np.pi)) * (2.0 * np.pi)
    val = topo_weight * float(cp.sum((q * q).astype(cp.float64)))

    # ∂topo/∂phase[i]:
    #   phase[i] が q[m] = phase[m+1] - phase[m] に現れるのは m = i-1 (+) と m = i (-)
    dphase = cp.zeros_like(phase)  # (n_p, n_e-1)
    two_q_tw = 2.0 * topo_weight * q
    dphase[:, 1:] += two_q_tw      # m=i-1 → phase[i] 係数 +1
    dphase[:, :-1] -= two_q_tw     # m=i   → phase[i] 係数 -1

    # ∂phase[i]/∂a = -b/D,  ∂phase[i]/∂b = a/D
    dphase_da = -b / D
    dphase_db = a / D

    # path[i]   = a[i] (phase[i] の a として) かつ b[i-1] (phase[i-1] の b として)
    grad_path = cp.zeros_like(Lambda)
    grad_path[:, :-1] += dphase * dphase_da   # path[i] ← phase[i] (a 側)
    grad_path[:, 1:] += dphase * dphase_db    # path[i+1] ← phase[i] (b 側)

    grad += grad_path
    return val


# =============================================================================
# objective + grad 統合
# =============================================================================

def _full_objective_grad(
    Lambda: "cp.ndarray",
    G_pairs: "cp.ndarray",
    pair_i: "cp.ndarray",
    pair_j: "cp.ndarray",
    jump_mask_gpu: "cp.ndarray",
    jump_weights_gpu: "cp.ndarray",
    alpha: float, beta: float,
    jump_weight: float,
    topo_weight: float,
) -> Tuple[float, "cp.ndarray"]:
    """objective + analytic gradient を一度に計算。"""
    grad = cp.zeros_like(Lambda)
    obj = 0.0
    obj += _data_fit_obj_grad(Lambda, G_pairs, pair_i, pair_j, grad)
    obj += _tv_obj_grad(Lambda, alpha, grad)
    obj += _l1_obj_grad(Lambda, beta, grad)
    obj += _jump_reg_obj_grad(Lambda, jump_weight, grad)
    obj += _jump_consistency_obj_grad(Lambda, jump_mask_gpu, jump_weights_gpu, grad)
    if topo_weight != 0.0:
        obj += _topo_reg_obj_grad(Lambda, topo_weight, grad)
    return obj, grad


# =============================================================================
# 初期値生成 (jump-aware eigh init) GPU 版
# =============================================================================

def _initialize_with_jump_structure_gpu(
    events_gpu: "cp.ndarray",     # (n_events, n_dim) float32
    jump_mask_gpu: "cp.ndarray",
    jump_weights_gpu: "cp.ndarray",
    n_paths: int,
) -> "cp.ndarray":
    """CPU 版と同じ手順を CuPy で。eigh は GPU 上で。"""
    n_events = events_gpu.shape[0]
    gram = events_gpu @ events_gpu.T                          # (n, n)
    # eigh は ascending order を返す → 末尾 n_paths 個が大きい固有値
    _, V = cp.linalg.eigh(gram)
    base_paths = V[:, -n_paths:].T                             # (n_paths, n_events)
    Lambda_init = base_paths.copy()

    # ジャンプ位置で増幅 (偶数 path: ×(1+jw), 奇数 path: ×-(1+jw))
    jm = jump_mask_gpu > 0.5
    jw = jump_weights_gpu
    amp = 1.0 + jw                                              # (n_events,)
    # path index ごとに sign を変える
    signs = cp.where(
        cp.arange(n_paths)[:, None] % 2 == 0, 1.0, -1.0
    ).astype(Lambda_init.dtype)                                 # (n_paths, 1)
    # mask に該当する列だけ × (signs * amp)
    factor = cp.where(jm[None, :], signs * amp[None, :].astype(signs.dtype), 1.0)
    Lambda_init = Lambda_init * factor
    return Lambda_init.astype(DEFAULT_DTYPE)


# =============================================================================
# Public API
# =============================================================================

def solve_inverse_problem_sparse_gpu(
    events: np.ndarray,
    jump_structures: Dict,
    n_paths: int,
    alpha: float,
    beta: float,
    topo_weight: float = 0.1,
    expand_window: int = 5,
    n_static_samples: int = 50,
    use_jump_init: bool = True,
    maxiter: int = 1000,
    verbose: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[str, int]]:
    """CPU 版 solve_inverse_problem_sparse の GPU 版（解析勾配）。

    Returns:
        paths: {i: Λ[i]} (numpy float64, normalized)
        stats: ペア統計
    """
    # ----- pair 構築 (CPU, 軽い) -----
    pair_array, stats = compute_significant_pairs(
        events, jump_structures,
        expand_window=expand_window, n_static_samples=n_static_samples,
    )
    if verbose:
        print(f"  [GPU] sparse pairs: {stats['n_selected_pairs']:,} / "
              f"{stats['n_full_pairs']:,} "
              f"({stats['reduction_ratio']*100:.1f}% reduction)")

    n_events = events.shape[0]
    jump_mask = jump_structures['integrated']['unified_jumps']
    jump_weights = jump_structures['integrated']['jump_importance']

    # ----- GPU 常駐データ -----
    events_gpu = ensure_gpu(events)                            # (n_events, n_dim)
    pair_i = cp.asarray(pair_array[:, 0], dtype=cp.int64)
    pair_j = cp.asarray(pair_array[:, 1], dtype=cp.int64)
    # G_pairs[k] = <events[pair_i[k]], events[pair_j[k]]>
    G_pairs = cp.sum(events_gpu[pair_i] * events_gpu[pair_j], axis=1).astype(DEFAULT_DTYPE)
    jump_mask_gpu = ensure_gpu(jump_mask)
    jump_weights_gpu = ensure_gpu(jump_weights)

    # ----- 初期値 -----
    if use_jump_init:
        Lambda_init_gpu = _initialize_with_jump_structure_gpu(
            events_gpu, jump_mask_gpu, jump_weights_gpu, n_paths,
        )
    else:
        _, V = cp.linalg.eigh(events_gpu @ events_gpu.T)
        Lambda_init_gpu = V[:, -n_paths:].T.astype(DEFAULT_DTYPE).copy()

    # scipy.optimize.minimize は flat x を期待
    Lambda_init_np = to_cpu(Lambda_init_gpu).astype(np.float64).flatten()

    # ----- 共通の objective_and_grad ファクトリ -----
    def make_objective_and_grad(tw: float):
        def f(Lambda_flat_np: np.ndarray):
            Lambda_gpu = cp.asarray(
                Lambda_flat_np.reshape(n_paths, n_events), dtype=DEFAULT_DTYPE,
            )
            obj, grad_gpu = _full_objective_grad(
                Lambda_gpu, G_pairs, pair_i, pair_j,
                jump_mask_gpu, jump_weights_gpu,
                alpha=alpha, beta=beta, jump_weight=0.5, topo_weight=tw,
            )
            return float(obj), to_cpu(grad_gpu).astype(np.float64).ravel()
        return f

    # ----- no-topo solve -----
    res_no = minimize(
        make_objective_and_grad(0.0),
        Lambda_init_np,
        jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'disp': verbose},
    )
    L_no = res_no.x.reshape(n_paths, n_events)

    # ----- with-topo solve -----
    res_to = minimize(
        make_objective_and_grad(topo_weight),
        Lambda_init_np,
        jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'disp': verbose},
    )
    L_to = res_to.x.reshape(n_paths, n_events)

    # ----- MAX 合成 (CPU 版と同じ) -----
    L_max = np.maximum(np.abs(L_no), np.abs(L_to))
    paths = {
        i: path / (np.linalg.norm(path) + 1e-8)
        for i, path in enumerate(L_max)
    }
    return paths, stats
