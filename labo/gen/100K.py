import numpy as np
import MDAnalysis as mda
import urllib.request
import os
import warnings
import time

warnings.filterwarnings('ignore')

def create_lysozyme_100k_final_challenge(n_frames=100000):
    """
    Lambda³の最終テスト用の、より現実的で挑戦的な10万フレームMDトラジェクトリを生成する。
    - 因果関係のあるイベントカスケード
    - 多様な時間スケール
    - 可逆的/不可逆的な変化
    - 安定なミスフォールド中間体
    などを導入。
    """
    start_time = time.time()
    
    # --- 1. 初期設定 ---
    print(f"Creating a {n_frames} frame Final Challenge trajectory...")
    pdb_file = '1AKI.pdb'
    
    if not os.path.exists(pdb_file):
        print("Downloading Lysozyme structure (1AKI.pdb)...")
        urllib.request.urlretrieve('https://files.rcsb.org/download/1AKI.pdb', pdb_file)
    
    u = mda.Universe(pdb_file)
    initial_positions = u.atoms.positions.copy()
    n_atoms = len(u.atoms)
    
    print(f"Atoms: {n_atoms}, Target frames: {n_frames}")
    # float32で計算 (4 bytes/value)
    estimated_gb = n_frames * n_atoms * 3 * 4 / 1e9
    print(f"Estimated memory for trajectory: {estimated_gb:.1f} GB")
    
    # 全体の配列を事前確保
    trajectory = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    print("Memory allocated successfully!")
    
    # --- 2. 物理モデルの準備 ---
    
    # 温度因子から原子ごとの揺らぎを推定
    bfactors = u.atoms.tempfactors
    atom_fluctuations = np.sqrt(bfactors / (8 * np.pi**2))
    atom_fluctuations = np.clip(atom_fluctuations, 0.1, 2.0)
    
    # 集団運動モード（より複雑に）
    # 短周期から長周期まで複数のモードを組み合わせる
    collective_modes = [np.random.randn(n_atoms, 3) for _ in range(5)]
    for mode in collective_modes:
        mode /= np.linalg.norm(mode)

    # 特定の残基グループを定義
    loop_region = u.select_atoms('resid 40-50').indices
    helix_region = u.select_atoms('resid 24-36').indices # α-helix C
    hydrophobic_core = u.select_atoms('resname LEU ILE VAL and backbone').indices

    # --- 3. イベントタイムラインの設計 (10万フレーム版ストーリー) ---
    events = [
        # フェーズ1: 安定期と微小な揺らぎ
        {'start': 5000, 'end': 15000, 'type': 'subtle_breathing'},
        
        # フェーズ2: カスケードの開始
        {'start': 18000, 'end': 22000, 'type': 'ligand_binding_effect'},
        {'start': 25000, 'end': 35000, 'type': 'slow_helix_destabilization'}, # 長期的な緩やか変化
        {'start': 40000, 'end': 45000, 'type': 'domain_shift'},

        # フェーズ3: 破局的イベントと回復の試み
        {'start': 50000, 'end': 53000, 'type': 'rapid_partial_unfold'}, # 短期的な鋭い変化
        {'start': 58000, 'end': 65000, 'type': 'transient_refolding_attempt'}, # 可逆的変化
        {'start': 65000, 'end': 75000, 'type': 'misfolded_intermediate'}, # 安定な偽の状態
        
        # フェーズ4: 凝集への道
        {'start': 78000, 'end': 83000, 'type': 'hydrophobic_exposure'},
        {'start': 85000, 'end': 95000, 'type': 'aggregation_onset'}
    ]
    
    # --- 4. トラジェクトリ生成ループ ---
    trajectory[0] = initial_positions
    current_pos = initial_positions.copy()
    
    print("\nGenerating trajectory...")
    for i in range(1, n_frames):
        if i % 10000 == 0:
            print(f"  Frame {i}/{n_frames} ({i/n_frames*100:.0f}%)")
        
        # 基本的な熱揺らぎ
        thermal_noise = np.random.randn(n_atoms, 3).astype(np.float32)
        thermal_noise *= atom_fluctuations[:, np.newaxis] * 0.02
        
        # 複雑な集団運動
        collective_motion = np.zeros((n_atoms, 3), dtype=np.float32)
        collective_motion += collective_modes[0] * np.sin(2 * np.pi * i / 1000) * 0.05
        collective_motion += collective_modes[1] * np.sin(2 * np.pi * i / 3300) * 0.08
        collective_motion += collective_modes[2] * np.sin(2 * np.pi * i / 7100) * 0.06
        collective_motion += collective_modes[3] * np.sin(2 * np.pi * i / 15000) * 0.10 # 長周期
        collective_motion += collective_modes[4] * np.cos(2 * np.pi * i / 25000) * 0.07 # 長周期

        # イベント処理
        displacement = np.zeros((n_atoms, 3), dtype=np.float32)
        for event in events:
            if event['start'] <= i < event['end']:
                progress = (i - event['start']) / (event['end'] - event['start'])
                
                if event['type'] == 'subtle_breathing':
                    scale = 1 + 0.005 * np.sin(2 * np.pi * progress)
                    center = current_pos.mean(axis=0)
                    displacement += (current_pos - center) * (scale - 1)
                
                elif event['type'] == 'ligand_binding_effect':
                    # 特定のループ領域が少し動く
                    displacement[loop_region] += np.random.randn(len(loop_region), 3) * progress * 0.1
                
                elif event['type'] == 'slow_helix_destabilization':
                    # ヘリックスが1万フレームかけてゆっくり壊れる
                    noise = np.random.randn(len(helix_region), 3) * progress * 0.05
                    displacement[helix_region] += noise
                
                elif event['type'] == 'domain_shift':
                    # タンパク質の上半分（Y座標で定義）が少しずれる
                    domain_atoms = np.where(current_pos[:, 1] > current_pos[:, 1].mean())[0]
                    shift_vector = np.array([0.01, -0.005, 0.01]) * np.sin(np.pi * progress)
                    displacement[domain_atoms] += shift_vector
                
                elif event['type'] == 'rapid_partial_unfold':
                    # 短時間で部分的に大きく広がる
                    scale = 1 + 0.1 * (np.sin(np.pi * progress)**2)
                    center = current_pos.mean(axis=0)
                    displacement += (current_pos - center) * (scale - 1)
                    displacement += np.random.randn(n_atoms, 3) * 0.05 * progress
                
                elif event['type'] == 'transient_refolding_attempt':
                    # 一時的にコンパクトに戻ろうとする
                    scale = 1 - 0.05 * (np.sin(np.pi * progress)**2)
                    center = current_pos.mean(axis=0)
                    displacement += (current_pos - center) * (scale - 1)
                
                elif event['type'] == 'misfolded_intermediate':
                    # 誤った状態で安定化。特定の疎水性コアが少し緩む
                    displacement[hydrophobic_core] += np.random.randn(len(hydrophobic_core), 3) * 0.02
                
                elif event['type'] == 'hydrophobic_exposure':
                    # 疎水性残基が表面に露出し、揺らぎが大きくなる
                    displacement[hydrophobic_core] += np.random.randn(len(hydrophobic_core), 3) * 0.15 * progress

                elif event['type'] == 'aggregation_onset':
                    # 全体が凝集し始める（中心に引き寄せられ、コンパクト化）
                    center = current_pos.mean(axis=0)
                    displacement += (center - current_pos) * 0.002 * progress
        
        # 構造の更新
        current_pos += thermal_noise + collective_motion + displacement
        
        # 構造の維持（逸脱が大きすぎる原子を優しく引き戻す）
        dev_from_initial = current_pos - initial_positions
        dev_norm = np.linalg.norm(dev_from_initial, axis=1)
        too_far = dev_norm > 15.0
        if np.any(too_far):
            # 逸脱の大きさに応じて引き戻す力を変える
            pullback_force = (dev_norm[too_far] - 15.0) / dev_norm[too_far]
            current_pos[too_far] -= dev_from_initial[too_far] * pullback_force[:, np.newaxis] * 0.2
        
        trajectory[i] = current_pos.copy()
    
    # --- 5. 保存 ---
    backbone = u.select_atoms('backbone')
    backbone_indices = backbone.indices
    
    print("\nSaving final challenge trajectory...")
    output_traj_file = 'lysozyme_100k_final_challenge.npy'
    output_idx_file = 'lysozyme_100k_backbone_indices.npy'
    np.save(output_traj_file, trajectory)
    np.save(output_idx_file, backbone_indices)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.1f} seconds!")
    print(f"Files saved:")
    print(f"  - {output_traj_file} ({estimated_gb:.1f} GB)")
    print(f"  - {output_idx_file}")
    
    print("\n🔬 Event Timeline for Final Challenge:")
    for event in events:
        print(f"  - {event['type']}: frames {event['start']}-{event['end']}")
    
    return trajectory, backbone_indices

# --- 実行 ---
if __name__ == "__main__":
    print("="*50)
    print("=== Generating the Lambda³ Final Challenge Dataset ===")
    print("="*50)
    
    traj, backbone = create_lysozyme_100k_final_challenge(100000)
    
    if traj is not None:
        print("\n\nReady for the FINAL TEST!")
        print("Run your Lambda³ analysis on this trajectory.")
