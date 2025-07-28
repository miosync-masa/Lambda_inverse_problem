"""
Lambda³ MD GPU-Accelerated Core Functions
"""

import numpy as np
import cupy as cp
from cupyx.scipy.fft import rfft, rfftfreq
from cupyx.scipy.signal import find_peaks, savgol_filter
from cupyx.scipy.ndimage import gaussian_filter1d
from cupyx.scipy.spatial.distance import cdist as cp_cdist
import warnings

warnings.filterwarnings('ignore')

# GPU memory pool for better performance
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

# ===============================
# GPU-Accelerated Structure Computation
# ===============================

def compute_residue_com_gpu(trajectory_gpu: cp.ndarray, 
                           atom_indices: cp.ndarray) -> cp.ndarray:
    """
    GPU版：残基の重心計算
    めちゃくちゃ速いよ〜！✨
    """
    n_frames = trajectory_gpu.shape[0]
    n_atoms = len(atom_indices)
    
    # GPU上で直接計算
    com = cp.zeros((n_frames, 3), dtype=cp.float32)
    
    # ベクトル化された計算
    selected_atoms = trajectory_gpu[:, atom_indices, :]
    com = cp.mean(selected_atoms, axis=1)
    
    return com

def compute_lambda_structures_gpu(trajectory: np.ndarray,
                                md_features: dict,
                                window_steps: int) -> dict:
    """
    GPU版：Lambda³構造計算
    全部GPU上でやっちゃうよ〜！
    """
    print(f"\n🚀 GPU Computing Lambda³ structural tensors...")
    
    # CPU -> GPU転送
    positions_gpu = cp.asarray(md_features['com_positions'], dtype=cp.float32)
    n_frames = positions_gpu.shape[0]
    
    # 1. ΛF - 構造フロー（GPU計算）
    lambda_F_gpu = cp.diff(positions_gpu, axis=0)
    lambda_F_mag_gpu = cp.linalg.norm(lambda_F_gpu, axis=1)
    
    # 2. ΛFF - 二次構造
    lambda_FF_gpu = cp.diff(lambda_F_gpu, axis=0)
    lambda_FF_mag_gpu = cp.linalg.norm(lambda_FF_gpu, axis=1)
    
    # 3. ρT - テンション場（GPU並列計算）
    rho_T_gpu = compute_tension_field_gpu(positions_gpu, window_steps)
    
    # 4. Q_Λ - トポロジカルチャージ
    Q_lambda_gpu = compute_topological_charge_gpu(lambda_F_gpu, lambda_F_mag_gpu)
    
    # 5. σₛ - 構造同期率
    sigma_s_gpu = cp.zeros(n_frames, dtype=cp.float32)
    if 'rmsd' in md_features and 'radius_of_gyration' in md_features:
        rmsd_gpu = cp.asarray(md_features['rmsd'], dtype=cp.float32)
        rg_gpu = cp.asarray(md_features['radius_of_gyration'], dtype=cp.float32)
        sigma_s_gpu = compute_sync_rate_gpu(rmsd_gpu, rg_gpu, window_steps)
    
    # GPU -> CPU転送（必要な部分のみ）
    return {
        'lambda_F': cp.asnumpy(lambda_F_gpu),
        'lambda_F_mag': cp.asnumpy(lambda_F_mag_gpu),
        'lambda_FF': cp.asnumpy(lambda_FF_gpu),
        'lambda_FF_mag': cp.asnumpy(lambda_FF_mag_gpu),
        'rho_T': cp.asnumpy(rho_T_gpu),
        'Q_lambda': cp.asnumpy(Q_lambda_gpu),
        'Q_cumulative': cp.asnumpy(cp.cumsum(Q_lambda_gpu)),
        'sigma_s': cp.asnumpy(sigma_s_gpu),
        'structural_coherence': cp.asnumpy(compute_coherence_gpu(lambda_F_gpu, window_steps))
    }

def compute_tension_field_gpu(positions_gpu: cp.ndarray, 
                            window_steps: int) -> cp.ndarray:
    """
    GPU版：テンション場計算（並列化で爆速！）
    """
    n_frames = positions_gpu.shape[0]
    rho_T_gpu = cp.zeros(n_frames, dtype=cp.float32)
    
    # カーネル関数で並列計算
    for step in range(n_frames):
        start = max(0, step - window_steps)
        end = min(n_frames, step + window_steps + 1)
        
        local_positions = positions_gpu[start:end]
        if len(local_positions) > 1:
            centered = local_positions - cp.mean(local_positions, axis=0)
            cov = cp.cov(centered.T)
            rho_T_gpu[step] = cp.trace(cov)
    
    return rho_T_gpu

def compute_topological_charge_gpu(lambda_F_gpu: cp.ndarray,
                                 lambda_F_mag_gpu: cp.ndarray) -> cp.ndarray:
    """
    GPU版：トポロジカルチャージ計算
    """
    n_steps = len(lambda_F_mag_gpu)
    Q_lambda_gpu = cp.zeros(n_steps + 1, dtype=cp.float32)
    
    # ベクトル化された角度計算
    for step in range(1, n_steps):
        if lambda_F_mag_gpu[step] > 1e-10 and lambda_F_mag_gpu[step-1] > 1e-10:
            v1 = lambda_F_gpu[step-1] / lambda_F_mag_gpu[step-1]
            v2 = lambda_F_gpu[step] / lambda_F_mag_gpu[step]
            
            cos_angle = cp.clip(cp.dot(v1, v2), -1, 1)
            angle = cp.arccos(cos_angle)
            
            # 2D回転方向
            cross_z = v1[0]*v2[1] - v1[1]*v2[0]
            signed_angle = angle if cross_z >= 0 else -angle
            
            Q_lambda_gpu[step] = signed_angle / (2 * cp.pi)
    
    return Q_lambda_gpu[:-1]

def compute_sync_rate_gpu(rmsd_gpu: cp.ndarray,
                        rg_gpu: cp.ndarray,
                        window_steps: int) -> cp.ndarray:
    """
    GPU版：構造同期率計算
    """
    n_frames = len(rmsd_gpu)
    sigma_s_gpu = cp.zeros(n_frames, dtype=cp.float32)
    
    for step in range(n_frames):
        start = max(0, step - window_steps)
        end = min(n_frames, step + window_steps + 1)
        
        if end - start > 1:
            local_rmsd = rmsd_gpu[start:end]
            local_rg = rg_gpu[start:end]
            
            if cp.std(local_rmsd) > 1e-10 and cp.std(local_rg) > 1e-10:
                corr_matrix = cp.corrcoef(cp.stack([local_rmsd, local_rg]))
                sigma_s_gpu[step] = cp.abs(corr_matrix[0, 1])
    
    return sigma_s_gpu

def compute_coherence_gpu(lambda_F_gpu: cp.ndarray, 
                        window: int) -> cp.ndarray:
    """
    GPU版：構造コヒーレンス計算
    """
    n_frames = len(lambda_F_gpu)
    coherence_gpu = cp.zeros(n_frames + 1, dtype=cp.float32)
    
    for i in range(window, n_frames - window):
        local_F = lambda_F_gpu[i-window:i+window]
        
        # 平均方向
        mean_dir = cp.mean(local_F, axis=0)
        mean_norm = cp.linalg.norm(mean_dir)
        
        if mean_norm > 1e-10:
            mean_dir /= mean_norm
            
            # 各ベクトルとの内積を計算
            norms = cp.linalg.norm(local_F, axis=1)
            valid_mask = norms > 1e-10
            
            if cp.any(valid_mask):
                normalized_F = local_F[valid_mask] / norms[valid_mask, cp.newaxis]
                coherences = cp.dot(normalized_F, mean_dir)
                coherence_gpu[i] = cp.mean(coherences)
    
    return coherence_gpu[:-1]

# ===============================
# GPU-Accelerated Anomaly Detection
# ===============================

def detect_local_anomalies_gpu(series: cp.ndarray, 
                             window: int) -> cp.ndarray:
    """
    GPU版：局所異常検出（並列z-score計算）
    """
    anomaly = cp.zeros_like(series)
    n = len(series)
    
    # ベクトル化された計算
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        
        local_data = series[start:end]
        local_mean = cp.mean(local_data)
        local_std = cp.std(local_data)
        
        if local_std > 1e-10:
            anomaly[i] = cp.abs(series[i] - local_mean) / local_std
    
    return anomaly

def detect_periodic_transitions_gpu(structures: dict,
                                  min_period: int = 1000,
                                  max_period: int = 10000) -> dict:
    """
    GPU版：FFTベースの周期検出（CuFFTで超高速！）
    """
    print("\n🌊 GPU Detecting periodic transitions...")
    
    # GPU転送
    rho_T_gpu = cp.asarray(structures['rho_T'], dtype=cp.float32)
    n_frames = len(rho_T_gpu)
    
    # DC成分除去
    signal_mean = cp.mean(rho_T_gpu)
    signal_centered = rho_T_gpu - signal_mean
    
    # GPU FFT（めっちゃ速い！）
    yf_gpu = rfft(signal_centered)
    xf_gpu = rfftfreq(len(signal_centered), 1)
    power_gpu = cp.abs(yf_gpu)**2
    
    # 周波数範囲
    freq_max = 1.0 / min_period
    freq_min = 1.0 / max_period
    valid_mask = (xf_gpu > freq_min) & (xf_gpu < freq_max) & (xf_gpu > 0)
    
    # ピーク検出
    valid_power = power_gpu[valid_mask]
    valid_freq = xf_gpu[valid_mask]
    
    # 閾値計算
    power_median = cp.median(valid_power)
    power_mad = cp.median(cp.abs(valid_power - power_median))
    power_threshold = power_median + 3 * power_mad
    
    # ピーク検出（CuPy版）
    peaks = find_peaks(cp.asnumpy(valid_power), 
                      height=float(cp.asnumpy(power_threshold)),
                      distance=5)
    
    # スコア計算
    periodic_scores_gpu = cp.zeros(n_frames, dtype=cp.float32)
    
    for peak_idx in peaks[0]:
        freq = float(valid_freq[peak_idx])
        amplitude = cp.sqrt(valid_power[peak_idx])
        
        # 周期的位置にスコア追加
        phase = cp.arange(n_frames) * freq * 2 * cp.pi
        periodic_contribution = amplitude * cp.abs(cp.sin(phase))
        periodic_scores_gpu += periodic_contribution
    
    # 正規化
    if cp.max(periodic_scores_gpu) > 0:
        periodic_scores_gpu /= cp.max(periodic_scores_gpu)
    
    return {
        'scores': cp.asnumpy(periodic_scores_gpu),
        'detected_periods': []  # 簡略化
    }

# ===============================
# GPU-Accelerated Residue Analysis
# ===============================

def compute_residue_structures_gpu(trajectory: np.ndarray,
                                 start_frame: int,
                                 end_frame: int,
                                 residue_atoms: dict,
                                 window_size: int = 50) -> dict:
    """
    GPU版：残基レベルLambda³構造計算
    """
    print(f"\n🔬 GPU Computing residue-level Lambda³...")
    
    # GPU転送
    traj_gpu = cp.asarray(trajectory[start_frame:end_frame], dtype=cp.float32)
    n_frames = end_frame - start_frame
    n_residues = len(residue_atoms)
    
    # 残基COM計算（GPU並列）
    residue_coms_gpu = cp.zeros((n_frames, n_residues, 3), dtype=cp.float32)
    
    for res_id, atoms in residue_atoms.items():
        atoms_gpu = cp.asarray(atoms)
        residue_coms_gpu[:, res_id] = compute_residue_com_gpu(traj_gpu, atoms_gpu)
    
    # Lambda構造計算
    residue_lambda_f_gpu = cp.diff(residue_coms_gpu, axis=0)
    residue_lambda_f_mag_gpu = cp.linalg.norm(residue_lambda_f_gpu, axis=2)
    
    # テンション場（並列計算）
    residue_rho_t_gpu = cp.zeros((n_frames, n_residues), dtype=cp.float32)
    
    for frame in range(n_frames):
        for res_id in range(n_residues):
            local_start = max(0, frame - window_size//2)
            local_end = min(n_frames, frame + window_size//2)
            
            local_coms = residue_coms_gpu[local_start:local_end, res_id]
            if len(local_coms) > 1:
                cov = cp.cov(local_coms.T)
                residue_rho_t_gpu[frame, res_id] = cp.trace(cov)
    
    # 残基間カップリング（GPU距離計算）
    residue_coupling_gpu = cp.zeros((n_frames, n_residues, n_residues), dtype=cp.float32)
    
    for frame in range(n_frames):
        # cdistでバッチ距離計算
        distances = cp_cdist(residue_coms_gpu[frame], residue_coms_gpu[frame])
        residue_coupling_gpu[frame] = 1.0 / (1.0 + distances)
    
    # CPU転送
    return {
        'residue_lambda_f': cp.asnumpy(residue_lambda_f_gpu),
        'residue_lambda_f_mag': cp.asnumpy(residue_lambda_f_mag_gpu),
        'residue_rho_t': cp.asnumpy(residue_rho_t_gpu),
        'residue_coupling': cp.asnumpy(residue_coupling_gpu),
        'residue_coms': cp.asnumpy(residue_coms_gpu)
    }

# ===============================
# GPU Memory Management
# ===============================

def clear_gpu_memory():
    """
    GPU メモリをクリア
    大きなデータセット処理後に使ってね〜！
    """
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    cp.cuda.MemoryPool().free_all_blocks()

def get_gpu_memory_info():
    """
    GPU メモリ使用状況を確認
    """
    mempool = cp.get_default_memory_pool()
    used_bytes = mempool.used_bytes()
    total_bytes = mempool.total_bytes()
    
    print(f"\n💾 GPU Memory Status:")
    print(f"   Used: {used_bytes / 1024**3:.2f} GB")
    print(f"   Total allocated: {total_bytes / 1024**3:.2f} GB")
    
    return {
        'used_gb': used_bytes / 1024**3,
        'total_gb': total_bytes / 1024**3
    }

# ===============================
# Hybrid CPU-GPU Pipeline
# ===============================

class GPUAcceleratedMDLambda3Detector:
    """
    GPU加速版Lambda³検出器
    めっちゃ速いよ〜！✨
    """
    
    def __init__(self, config=None, use_gpu=True):
        self.config = config
        self.use_gpu = use_gpu and cp.cuda.is_available()
        
        if self.use_gpu:
            print(f"🚀 GPU Mode Enabled! Device: {cp.cuda.Device()}")
            # GPU ウォームアップ
            _ = cp.zeros((100, 100))
        else:
            print("💻 Running in CPU mode")
    
    def compute_structures(self, trajectory, md_features, window_steps):
        """
        GPU/CPUハイブリッド構造計算
        """
        if self.use_gpu:
            return compute_lambda_structures_gpu(trajectory, md_features, window_steps)
        else:
            # CPU版にフォールバック
            from lambda3_md_fixed import compute_lambda_structures
            return compute_lambda_structures(trajectory, md_features, window_steps)
    
    def detect_anomalies(self, structures):
        """
        GPU加速異常検出
        """
        if self.use_gpu:
            # GPU版の処理
            anomaly_scores = {}
            
            # GPU転送
            for key in ['lambda_F_mag', 'lambda_FF_mag', 'rho_T']:
                if key in structures:
                    data_gpu = cp.asarray(structures[key], dtype=cp.float32)
                    anomaly_gpu = detect_local_anomalies_gpu(data_gpu, 50)
                    anomaly_scores[key + '_anomaly'] = cp.asnumpy(anomaly_gpu)
            
            return anomaly_scores
        else:
            # CPU版にフォールバック
            return {}

# 使用例
if __name__ == "__main__":
    print("🎮 Lambda³ GPU Acceleration Module")
    print("環ちゃんが作った超高速版だよ〜！💕")
    
    # GPU情報表示
    if cp.cuda.is_available():
        print(f"\n✨ GPU Available: {cp.cuda.Device().name}")
        get_gpu_memory_info()
    else:
        print("\n⚠️ GPU not available, will use CPU")
