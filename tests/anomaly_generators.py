"""
11種類の異常パターン生成ジェネレータ（地獄モード）。

本番detectorからは切り離されており、評価／合成データ生成からのみ使用する。
``init_anomaly_patterns(detector)`` で detector に attach すると、既存の
``detector.anomaly_patterns[name](events, intensity)`` 呼び出しがそのまま動く。
"""

from typing import Callable, Dict

import numpy as np


def _generate_pulse_anomaly(events: np.ndarray, intensity: float = 3, decay_rate: float = 0.5, n_pulses: int = 2) -> np.ndarray:
    events_copy = events.copy()
    n_events, n_features = events.shape
    n_pulses_safe = min(n_pulses, n_events)
    pulse_indices = np.random.choice(n_events, size=n_pulses_safe, replace=False)
    for idx in pulse_indices:
        pulse = np.zeros(n_features)
        affected_dims = np.random.choice(n_features, size=np.random.randint(1, n_features//2 + 1), replace=False)
        pulse[affected_dims] = np.random.randn(len(affected_dims)) * intensity * np.random.uniform(0.8, 1.2)
        if np.random.rand() < 0.5:
            pulse *= -1
        events_copy[idx] += pulse
        for offset in range(1, 4):
            decay = np.exp(-decay_rate * offset)
            if idx - offset >= 0:
                events_copy[idx - offset] += pulse * decay * np.random.uniform(0.5, 1.0)
            if idx + offset < n_events:
                events_copy[idx + offset] += pulse * decay * np.random.uniform(0.5, 1.0)
    noise_mask = np.random.rand(*events_copy.shape) < 0.01
    events_copy[noise_mask] += np.random.normal(0, intensity/6, np.sum(noise_mask))
    return events_copy


def _generate_phase_jump_anomaly(events: np.ndarray, intensity: float = 3, spread: int = 4) -> np.ndarray:
    events_copy = events.copy()
    n_events, n_features = events.shape
    idx = np.random.randint(n_events)
    events_copy[idx] = -np.sign(events_copy[idx]) * (np.abs(events_copy[idx]) ** np.random.uniform(1.2, 2.0)) * intensity
    events_copy[idx] += np.random.randn(n_features) * intensity * 0.3
    for offset in range(1, spread + 1):
        decay_factor = np.exp(-0.6 * offset)
        random_phase_shift = np.random.uniform(-np.pi, np.pi, n_features)
        modulation_factor = intensity * decay_factor * np.random.uniform(0.6, 1.2)
        if idx - offset >= 0:
            events_copy[idx - offset] += (
                np.sin(events_copy[idx]) * modulation_factor
                + np.cos(random_phase_shift) * modulation_factor * 0.5
            )
        if idx + offset < n_events:
            events_copy[idx + offset] += (
                np.sin(events_copy[idx]) * modulation_factor
                + np.cos(random_phase_shift) * modulation_factor * 0.5
            )
    distant_offset = spread + np.random.randint(1, 3)
    distant_decay = np.exp(-1.2 * distant_offset)
    distant_idx = idx + distant_offset if (idx + distant_offset < n_events) else idx - distant_offset
    if 0 <= distant_idx < n_events:
        events_copy[distant_idx] += np.random.randn(n_features) * intensity * distant_decay
    return events_copy


def _generate_periodic_anomaly(events: np.ndarray, intensity: float = 2, disruption_prob=0.2) -> np.ndarray:
    events_copy = events.copy()
    n_events, n_features = events.shape
    t = np.arange(n_events)
    period = np.random.randint(max(3, n_events // 10), max(4, n_events // 4))
    phase = np.random.uniform(0, 2*np.pi)
    base_signal = intensity * np.sin(2 * np.pi * t / period + phase)
    for f in range(n_features):
        feat_phase = phase + np.random.uniform(-np.pi/5, np.pi/5)
        feat_signal = base_signal * np.random.uniform(0.7, 1.3)
        if np.random.rand() < disruption_prob:
            idx = np.random.randint(2, max(3, n_events-2)) if n_events > 3 else 0
            feat_signal[idx-1:idx+2] += np.random.uniform(2, 4) * intensity * np.random.choice([-1, 1])
        if np.random.rand() < disruption_prob and n_events >= 4:
            low = n_events // 4
            high = 3 * n_events // 4
            if high > low:
                jump_idx = np.random.randint(low, high)
                feat_signal[jump_idx:] *= -1
        if np.random.rand() < disruption_prob:
            del_start = np.random.randint(0, max(1, n_events - n_events // 8))
            del_end = del_start + np.random.randint(2, max(3, n_events // 8))
            del_end = min(n_events, del_end)
            feat_signal[del_start:del_end] = 0
        feat_signal += np.random.normal(0, 0.15 * intensity, n_events)
        events_copy[:, f] += feat_signal
    return events_copy


def _generate_decay_anomaly(events: np.ndarray, intensity: float = 2) -> np.ndarray:
    events_copy = events.copy()
    n_events = events.shape[0]
    decay_start = n_events // 2
    decay_length = n_events - decay_start
    decay = np.exp(-intensity * np.arange(decay_length) / decay_length)
    oscillation = np.sin(np.arange(decay_length) * 0.5) * 0.3
    decay_with_osc = decay * (1 + oscillation)
    n_features = events.shape[1]
    feature_decay_rates = np.random.uniform(0.5, 1.5, n_features)
    for i in range(decay_length):
        events_copy[decay_start + i] *= decay_with_osc[i]
        events_copy[decay_start + i] *= feature_decay_rates
        noise_level = (1 - decay[i]) * intensity * 0.5
        events_copy[decay_start + i] += np.random.normal(0, noise_level, n_features)
    n_spikes = max(1, decay_length // 10)
    spike_positions = np.random.choice(range(decay_start, n_events), n_spikes, replace=False)
    for pos in spike_positions:
        spike_features = np.random.choice(n_features, np.random.randint(1, max(2, n_features//3)), replace=False)
        events_copy[pos, spike_features] *= np.random.uniform(2, 4)
    if decay_length > 5:
        for i in range(decay_start + 2, n_events):
            correlation_loss = 1 - decay[i - decay_start]
            events_copy[i] = (1 - correlation_loss) * events_copy[i] + \
                            correlation_loss * np.random.randn(n_features) * np.std(events_copy[:decay_start])
    return events_copy


def _generate_bifurcation_anomaly(events: np.ndarray, intensity: float = 2) -> np.ndarray:
    events_copy = events.copy()
    n_events = events.shape[0]
    n_features = events.shape[1]
    split_point = n_events // 2
    post_split_length = n_events - split_point
    mode1 = np.random.randn(n_features) * intensity
    mode2 = -mode1 + np.random.randn(n_features) * intensity * 0.5
    for i in range(post_split_length):
        t = i / post_split_length
        bifurcation_strength = np.sqrt(t) * intensity
        if (split_point + i) % 2 == 0:
            events_copy[split_point + i] += mode1 * bifurcation_strength
            rotation_angle = t * np.pi / 4
            events_copy[split_point + i] = _rotate_features(events_copy[split_point + i], rotation_angle)
        else:
            events_copy[split_point + i] += mode2 * bifurcation_strength
            rotation_angle = -t * np.pi / 4
            events_copy[split_point + i] = _rotate_features(events_copy[split_point + i], rotation_angle)
    if split_point > 0 and split_point < n_events:
        events_copy[split_point] *= np.random.uniform(0.1, 0.5)
        events_copy[split_point] += np.random.randn(n_features) * intensity * 2
        chaos_range = min(5, split_point // 10)
        for j in range(max(0, split_point - chaos_range), min(n_events, split_point + chaos_range)):
            distance_from_split = abs(j - split_point)
            chaos_intensity = intensity * np.exp(-distance_from_split / chaos_range)
            events_copy[j] += np.random.randn(n_features) * chaos_intensity
    if post_split_length > 10:
        high_freq = np.random.uniform(0.3, 0.5)
        for i in range(split_point, n_events):
            phase = (i - split_point) * high_freq * 2 * np.pi
            amplitude = intensity * 0.3 * ((i - split_point) / post_split_length)
            events_copy[i] += np.sin(phase) * amplitude * np.random.randn(n_features)
    if n_features > 3:
        correlation_matrix = np.random.randn(n_features, n_features)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        correlation_matrix = np.exp(-np.abs(correlation_matrix))
        for i in range(split_point, n_events):
            events_copy[i] = correlation_matrix @ events_copy[i]
    return events_copy


def _rotate_features(features: np.ndarray, angle: float) -> np.ndarray:
    if len(features) < 2:
        return features
    rotated = features.copy()
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    temp0 = rotated[0] * cos_a - rotated[1] * sin_a
    temp1 = rotated[0] * sin_a + rotated[1] * cos_a
    rotated[0], rotated[1] = temp0, temp1
    return rotated


def _generate_multi_path_anomaly(events: np.ndarray, intensity: float = 2, interaction: float = 0.3) -> np.ndarray:
    events_copy = events.copy()
    n_events, n_features = events.shape
    n_available = n_events
    n_paths = min(np.random.randint(2, 5), n_available)
    path_indices = np.random.choice(n_available, size=n_paths, replace=False)
    paths_directions = np.random.randn(n_paths, n_features)
    paths_directions /= np.linalg.norm(paths_directions, axis=1, keepdims=True)
    base_intensities = intensity * np.random.uniform(0.8, 1.8, size=n_paths)
    polarities = np.random.choice([-1, 1], size=n_paths)
    for i, idx in enumerate(path_indices):
        pulse = paths_directions[i] * base_intensities[i] * polarities[i]
        events_copy[idx] += pulse * np.random.uniform(1.5, 2.5)
        for offset in range(1, 3):
            decay = np.exp(-0.5 * offset)
            spike_factor = (base_intensities[i] ** 2) * decay
            if idx - offset >= 0:
                events_copy[idx - offset] += pulse * spike_factor * np.random.uniform(0.5, 1.2)
            if idx + offset < n_events:
                events_copy[idx + offset] += pulse * spike_factor * np.random.uniform(0.5, 1.2)
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            midpoint = (path_indices[i] + path_indices[j]) // 2
            interaction_vector = (paths_directions[i] * paths_directions[j])
            interaction_strength = intensity * interaction * np.random.uniform(1.0, 2.0)
            if midpoint < n_events:
                events_copy[midpoint] += interaction_vector * interaction_strength * np.random.choice([-1, 1])
                for offset in range(1, 3):
                    interaction_decay = np.exp(-0.4 * offset)
                    if midpoint - offset >= 0:
                        events_copy[midpoint - offset] += interaction_vector * interaction_strength * interaction_decay
                    if midpoint + offset < n_events:
                        events_copy[midpoint + offset] += interaction_vector * interaction_strength * interaction_decay
    if np.random.rand() < 0.3:
        spike_idx = np.random.randint(0, n_events)
        spike_magnitude = intensity * np.random.uniform(3, 5)
        events_copy[spike_idx] += np.random.randn(n_features) * spike_magnitude
    return events_copy


def _generate_partial_periodic_anomaly(events: np.ndarray, intensity: float = 2) -> np.ndarray:
    events_copy = events.copy()
    n_events = events.shape[0]
    if n_events < 4:
        t = np.arange(n_events)
        period = max(2, n_events // 2)
        modulation = intensity * np.sin(2 * np.pi * t / period)
        events_copy += modulation[:, np.newaxis]
        return events_copy
    min_width = max(2, n_events // 6)
    max_width = max(min_width + 1, n_events // 2)
    width = np.random.randint(min_width, max_width)
    if width >= n_events:
        width = n_events - 1
    start = np.random.randint(0, n_events - width)
    end = start + width
    period = max(2, width // 3)
    t = np.arange(width)
    modulation = intensity * np.sin(2 * np.pi * t / period)
    events_copy[start:end] += modulation[:, np.newaxis]
    return events_copy


def _generate_topological_jump_anomaly(events: np.ndarray, intensity: float = 3) -> np.ndarray:
    events_copy = events.copy()
    n_events = events.shape[0]
    if n_events < 3:
        events_copy *= -intensity
        return events_copy
    min_point = max(1, n_events // 3)
    max_point = max(min_point + 1, 2 * n_events // 3)
    if min_point >= max_point:
        jump_point = n_events // 2
    else:
        jump_point = np.random.randint(min_point, max_point)
    if jump_point > 0:
        events_copy[:jump_point] *= np.exp(-0.1 * np.arange(jump_point))[:, np.newaxis]
    if jump_point < n_events:
        events_copy[jump_point:] = -events_copy[jump_point:] * intensity
    if jump_point < n_events:
        events_copy[jump_point] = np.random.randn(events.shape[1]) * intensity * 2
    return events_copy


def _generate_cascade_anomaly(events: np.ndarray, intensity: float = 2) -> np.ndarray:
    events_copy = events.copy()
    n_events = events.shape[0]
    if n_events < 2:
        events_copy *= intensity
        return events_copy
    start_idx = np.random.randint(0, max(1, n_events // 2))
    events_copy[start_idx] += np.random.randn(events.shape[1]) * intensity
    for i in range(start_idx + 1, min(start_idx + 10, n_events)):
        decay = np.exp(-0.3 * (i - start_idx))
        events_copy[i] += events_copy[i-1] * 0.5 * decay
    return events_copy


def _generate_resonance_anomaly(events: np.ndarray, intensity: float = 2) -> np.ndarray:
    events_copy = events.copy()
    n_events = events.shape[0]
    if n_events < 4:
        events_copy *= intensity
        return events_copy
    fft = np.fft.fft(events_copy, axis=0)
    max_freq = max(2, len(fft) // 4)
    resonance_freq = np.random.randint(1, max_freq)
    if resonance_freq < len(fft):
        fft[resonance_freq] *= intensity
        if len(fft) - resonance_freq > 0:
            fft[-resonance_freq] *= intensity
    events_copy = np.real(np.fft.ifft(fft, axis=0))
    return events_copy


def _generate_superposition_anomaly_factory(patterns: Dict[str, Callable]) -> Callable:
    """Build a superposition generator that defers to a patterns dict
    (so we don't need a detector instance handy at call time)."""
    def _generate_superposition_anomaly(events: np.ndarray, intensity: float = 2) -> np.ndarray:
        events_copy = events.copy()
        n_patterns = np.random.randint(2, 4)
        base_patterns = ['pulse', 'periodic', 'phase_jump']
        chosen = np.random.choice(base_patterns, min(n_patterns, len(base_patterns)), replace=False)
        for pattern in chosen:
            weight = np.random.uniform(0.3, 0.7)
            if pattern in patterns:
                events_copy = weight * events_copy + (1 - weight) * patterns[pattern](events_copy, intensity)
        return events_copy
    return _generate_superposition_anomaly


# ===============================
# 公開API
# ===============================

def build_anomaly_patterns() -> Dict[str, Callable]:
    """Return a fresh registry of 11 anomaly-pattern generators."""
    patterns: Dict[str, Callable] = {
        'pulse': _generate_pulse_anomaly,
        'phase_jump': _generate_phase_jump_anomaly,
        'periodic': _generate_periodic_anomaly,
        'structural_decay': _generate_decay_anomaly,
        'bifurcation': _generate_bifurcation_anomaly,
        'multi_path': _generate_multi_path_anomaly,
        'topological_jump': _generate_topological_jump_anomaly,
        'cascade': _generate_cascade_anomaly,
        'partial_periodic': _generate_partial_periodic_anomaly,
        'resonance': _generate_resonance_anomaly,
    }
    # superpositionは他のパターンを呼ぶので最後に追加
    patterns['superposition'] = _generate_superposition_anomaly_factory(patterns)
    return patterns


def init_anomaly_patterns(detector):
    """Attach the 11 anomaly generators to ``detector.anomaly_patterns``."""
    detector.anomaly_patterns = build_anomaly_patterns()
    return detector.anomaly_patterns


def generate_anomalies(events: np.ndarray, pattern: str = 'pulse', intensity: float = 3) -> np.ndarray:
    """Standalone helper mirroring ``detector.generate_anomalies``."""
    patterns = build_anomaly_patterns()
    if pattern not in patterns:
        raise ValueError(f"Unknown anomaly pattern: {pattern}")
    return patterns[pattern](events, intensity)
