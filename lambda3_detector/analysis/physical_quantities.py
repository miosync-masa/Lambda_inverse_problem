"""
物理量計算: トポロジカルチャージ、エネルギー、エントロピー（基本＆ジャンプ条件付き）と
パス分類。
"""

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.entropy_jit import compute_all_entropies_jit
from ..core.pulsation_jit import compute_pulsation_energy_from_path
from ..core.topology_jit import compute_topological_charge_jit


def compute_topology(paths: Dict[int, np.ndarray]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """トポロジカル量の計算"""
    charges = {}
    stabilities = {}

    for i, path in paths.items():
        Q, sigma = compute_topological_charge_jit(path)
        charges[i] = Q
        stabilities[i] = sigma

    return charges, stabilities


def compute_energies(paths: Dict[int, np.ndarray]) -> Dict[int, float]:
    """エネルギー計算"""
    energies = {}
    for i, path in paths.items():
        basic_energy = np.sum(path**2)
        jump_int, _, pulse_pow = compute_pulsation_energy_from_path(path)
        energies[i] = basic_energy + 0.3 * pulse_pow

    return energies


def compute_entropies(paths: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
    """エントロピー計算"""
    entropies = {}
    entropy_keys = ["shannon", "renyi_2", "tsallis_1.5", "max", "min", "var"]

    for i, path in paths.items():
        all_entropies = compute_all_entropies_jit(path)
        entropy_dict = {}
        for j, key in enumerate(entropy_keys):
            entropy_dict[key] = all_entropies[j]
        entropies[i] = entropy_dict

    return entropies


def classify_structures(paths: Dict[int, np.ndarray],
                        charges: Dict[int, float],
                        stabilities: Dict[int, float],
                        jump_structures: Optional[Dict] = None) -> Dict[int, str]:
    """構造分類"""
    classifications = {}

    for i in paths.keys():
        Q = charges[i]
        sigma = stabilities[i]

        # 基本分類
        if Q < -0.5:
            base = "反物質的構造（吸収系）"
        elif Q > 0.5:
            base = "物質的構造（放出系）"
        else:
            base = "中性構造（平衡）"

        # 修飾
        tags = []

        if sigma > 2.5:
            tags.append("不安定/カオス的")
        elif sigma < 0.5:
            tags.append("超安定")

        # ジャンプ特性があれば追加
        if jump_structures and i < len(jump_structures['features']):
            feature_data = jump_structures['features'].get(i, list(jump_structures['features'].values())[0])
            pulse_power = feature_data['pulse_power']
            asymmetry = feature_data['asymmetry']

            if pulse_power > 5:
                tags.append("高頻度拍動")
            elif pulse_power < 0.1:
                tags.append("静的構造")

            if abs(asymmetry) > 0.7:
                if asymmetry > 0:
                    tags.append("正方向優位")
                else:
                    tags.append("負方向優位")

        # 分類完成
        if tags:
            classifications[i] = base + "・" + "／".join(tags)
        else:
            classifications[i] = base

    return classifications


def compute_jump_aware_topology(paths: Dict[int, np.ndarray],
                                jump_structures: Dict) -> Tuple[Dict[int, float], Dict[int, float]]:
    """ジャンプ構造を考慮したトポロジカル量計算"""
    charges = {}
    stabilities = {}

    for i, path in paths.items():
        # 基本的なトポロジカルチャージ
        Q, sigma = compute_topological_charge_jit(path)

        # ジャンプ位置での位相変化を考慮
        jump_mask = jump_structures['integrated']['unified_jumps']
        jump_phase_shift = 0.0

        for j in range(1, len(path)):
            if jump_mask[j]:
                # ジャンプ位置での位相変化
                phase_diff = np.arctan2(path[j], path[j-1]) - \
                           np.arctan2(path[j-1], path[j-2] if j > 1 else path[0])
                jump_phase_shift += phase_diff

        # ジャンプ補正したチャージ
        charges[i] = Q + jump_phase_shift / (2 * np.pi)
        stabilities[i] = sigma

    return charges, stabilities


def compute_pulsation_energies(paths: Dict[int, np.ndarray],
                               jump_structures: Dict) -> Dict[int, float]:
    """拍動エネルギーの計算（ジャンプ構造を優先使用）"""
    energies = {}

    for i, path in paths.items():
        # 基本エネルギー
        basic_energy = np.sum(path**2)

        # 対応する特徴のジャンプエネルギーを統合
        total_pulse_power = 0.0
        n_features = len(jump_structures['features'])

        for f_idx, f_data in jump_structures['features'].items():
            pulse_power = f_data['pulse_power']
            total_pulse_power += pulse_power

        avg_pulse_power = total_pulse_power / n_features if n_features > 0 else 0.0

        # 統合エネルギー
        energies[i] = basic_energy + 0.3 * avg_pulse_power

    return energies


def compute_jump_conditional_entropies(paths: Dict[int, np.ndarray],
                                       jump_structures: Dict) -> Dict[int, Dict[str, float]]:
    """ジャンプイベントでの条件付きエントロピー（局所構造解析強化版）"""
    entropies = {}
    jump_mask = jump_structures['integrated']['unified_jumps']

    # ジャンプ周辺の窓サイズ
    JUMP_WINDOW = 5  # ジャンプ前後5イベント

    for i, path in paths.items():
        # 全体のエントロピー
        all_entropies = compute_all_entropies_jit(path)
        entropy_keys = ["shannon", "renyi_2", "tsallis_1.5", "max", "min", "var"]

        # ジャンプ位置と非ジャンプ位置で分離
        jump_indices = np.where(jump_mask)[0]
        non_jump_indices = np.where(~jump_mask)[0]

        entropy_dict = {}

        # 全体エントロピー
        for j, key in enumerate(entropy_keys):
            entropy_dict[key] = all_entropies[j]

        # ジャンプ条件付きエントロピー
        if len(jump_indices) > 0:
            jump_path = path[jump_indices]
            jump_entropies = compute_all_entropies_jit(jump_path)
            for j, key in enumerate(entropy_keys):
                entropy_dict[f"{key}_jump"] = jump_entropies[j]

        # 非ジャンプ条件付きエントロピー
        if len(non_jump_indices) > 0:
            non_jump_path = path[non_jump_indices]
            non_jump_entropies = compute_all_entropies_jit(non_jump_path)
            for j, key in enumerate(entropy_keys):
                entropy_dict[f"{key}_non_jump"] = non_jump_entropies[j]

        # === 新規追加：ジャンプ周辺のエントロピー解析 ===

        # 1. ジャンプ前後の窓内エントロピー
        jump_vicinity_indices = set()
        for jump_idx in jump_indices:
            for offset in range(-JUMP_WINDOW, JUMP_WINDOW + 1):
                vicinity_idx = jump_idx + offset
                if 0 <= vicinity_idx < len(path):
                    jump_vicinity_indices.add(vicinity_idx)

        if jump_vicinity_indices:
            vicinity_indices = np.array(sorted(jump_vicinity_indices))
            vicinity_path = path[vicinity_indices]
            vicinity_entropies = compute_all_entropies_jit(vicinity_path)
            for j, key in enumerate(entropy_keys):
                entropy_dict[f"{key}_vicinity"] = vicinity_entropies[j]

        # 2. ジャンプ前のエントロピー（構造崩壊の前兆）
        pre_jump_indices = []
        for jump_idx in jump_indices:
            for offset in range(1, JUMP_WINDOW + 1):
                pre_idx = jump_idx - offset
                if pre_idx >= 0:
                    pre_jump_indices.append(pre_idx)

        if pre_jump_indices:
            pre_jump_path = path[pre_jump_indices]
            pre_jump_entropies = compute_all_entropies_jit(pre_jump_path)
            for j, key in enumerate(entropy_keys):
                entropy_dict[f"{key}_pre_jump"] = pre_jump_entropies[j]

        # 3. ジャンプ後のエントロピー（構造再編成）
        post_jump_indices = []
        for jump_idx in jump_indices:
            for offset in range(1, JUMP_WINDOW + 1):
                post_idx = jump_idx + offset
                if post_idx < len(path):
                    post_jump_indices.append(post_idx)

        if post_jump_indices:
            post_jump_path = path[post_jump_indices]
            post_jump_entropies = compute_all_entropies_jit(post_jump_path)
            for j, key in enumerate(entropy_keys):
                entropy_dict[f"{key}_post_jump"] = post_jump_entropies[j]

        # 4. エントロピー勾配（ジャンプによる構造変化の急峻さ）
        if len(jump_indices) > 0:
            entropy_gradients = []
            for jump_idx in jump_indices:
                # ジャンプ前後のローカルエントロピーを計算
                pre_start = max(0, jump_idx - JUMP_WINDOW)
                pre_end = jump_idx
                post_start = jump_idx + 1
                post_end = min(len(path), jump_idx + JUMP_WINDOW + 1)

                if pre_end > pre_start and post_end > post_start:
                    pre_local = compute_all_entropies_jit(path[pre_start:pre_end])
                    post_local = compute_all_entropies_jit(path[post_start:post_end])

                    # 各エントロピータイプの勾配
                    gradient = post_local - pre_local
                    entropy_gradients.append(gradient)

            if entropy_gradients:
                mean_gradients = np.mean(entropy_gradients, axis=0)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_gradient"] = mean_gradients[j]

        # 5. 局所エントロピー変動性（ジャンプ周辺の不安定性）
        if len(jump_indices) > 0:
            local_variations = []
            for jump_idx in jump_indices:
                # 各ジャンプ周辺での小窓エントロピー計算
                for window_start in range(max(0, jump_idx - JUMP_WINDOW),
                                        min(len(path) - 2, jump_idx + JUMP_WINDOW)):
                    if window_start + 3 <= len(path):  # 最小3点で計算
                        local_window = path[window_start:window_start + 3]
                        local_ent = compute_all_entropies_jit(local_window)
                        local_variations.append(local_ent)

            if local_variations:
                # 変動性を標準偏差で評価
                variations_std = np.std(local_variations, axis=0)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_local_variation"] = variations_std[j]

        # 6. 遠隔エントロピー（ジャンプから離れた領域）
        if len(jump_vicinity_indices) > 0:
            all_indices = set(range(len(path)))
            remote_indices = sorted(all_indices - jump_vicinity_indices)

            if remote_indices:
                remote_path = path[np.array(remote_indices)]
                remote_entropies = compute_all_entropies_jit(remote_path)
                for j, key in enumerate(entropy_keys):
                    entropy_dict[f"{key}_remote"] = remote_entropies[j]

                # ジャンプ近傍と遠隔の比率（構造的な局所性の指標）
                for j, key in enumerate(entropy_keys):
                    if f"{key}_vicinity" in entropy_dict and entropy_dict[f"{key}_remote"] != 0:
                        ratio = entropy_dict[f"{key}_vicinity"] / (entropy_dict[f"{key}_remote"] + 1e-10)
                        entropy_dict[f"{key}_locality_ratio"] = ratio

        entropies[i] = entropy_dict

    return entropies
