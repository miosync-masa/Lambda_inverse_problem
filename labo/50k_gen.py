import numpy as np
import MDAnalysis as mda
import urllib.request
import os
import warnings
warnings.filterwarnings('ignore')

def create_lysozyme_50k_trajectory(n_frames=50000):
    """Lysozymeの5万フレームMDトラジェクトリを生成"""

    import time
    start_time = time.time()

    # PDBファイルをダウンロード（既にあればスキップ）
    print(f"Creating {n_frames} frame trajectory...")
    pdb_file = '1AKI.pdb'

    if not os.path.exists(pdb_file):
        print("Downloading Lysozyme structure...")
        urllib.request.urlretrieve('https://files.rcsb.org/download/1AKI.pdb', pdb_file)

    # MDAnalysisで読み込み
    u = mda.Universe(pdb_file)
    initial_positions = u.atoms.positions.copy()
    n_atoms = len(u.atoms)

    print(f"Atoms: {n_atoms}, Target frames: {n_frames}")
    print(f"Estimated memory: {n_frames * n_atoms * 3 * 8 / 1e9:.1f} GB")

    # 全体の配列を事前確保
    trajectory = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)  # float32で省メモリ
    print("Memory allocated successfully!")

    # 温度因子から揺らぎを推定
    bfactors = u.atoms.tempfactors
    atom_fluctuations = np.sqrt(bfactors / (8 * np.pi**2))
    atom_fluctuations = np.clip(atom_fluctuations, 0.1, 2.0)

    # 集団運動モード
    collective_modes = [
        np.random.randn(n_atoms, 3) * 0.5,
        np.random.randn(n_atoms, 3) * 0.3,
        np.random.randn(n_atoms, 3) * 0.2,  # 追加モード
    ]
    for mode in collective_modes:
        mode /= np.linalg.norm(mode)

    # 初期構造
    trajectory[0] = initial_positions
    current_pos = initial_positions.copy()

    print("Generating trajectory...")

    # 5万フレーム用のイベント（時間スケール調整）
    events = [
        {'start': 5000, 'end': 7500, 'type': 'partial_unfold'},      # 早めに開始
        {'start': 15000, 'end': 17500, 'type': 'helix_break'},       # 中盤
        {'start': 25000, 'end': 30000, 'type': 'major_unfold'},      # 中盤〜後半
        {'start': 35000, 'end': 37500, 'type': 'misfold'},           # 後半
        {'start': 42500, 'end': 45000, 'type': 'aggregation_prone'}, # 終盤
    ]

    # トラジェクトリ生成
    for i in range(1, n_frames):
        if i % 5000 == 0:
            print(f"  Frame {i}/{n_frames} ({i/n_frames*100:.1f}%)")

        # 基本的な熱揺らぎ
        thermal_noise = np.random.randn(n_atoms, 3).astype(np.float32)
        for j in range(n_atoms):
            thermal_noise[j] *= atom_fluctuations[j] * 0.02

        # 集団運動（複数の周期）
        collective_motion = np.zeros((n_atoms, 3), dtype=np.float32)
        collective_motion += collective_modes[0] * np.sin(2 * np.pi * i / 2500) * 0.1   # 周期を半分に
        collective_motion += collective_modes[1] * np.sin(2 * np.pi * i / 4000) * 0.08  # 周期を半分に
        collective_motion += collective_modes[2] * np.sin(2 * np.pi * i / 6000) * 0.06  # 周期を半分に

        # イベント処理
        for event in events:
            if event['start'] <= i <= event['end']:
                progress = (i - event['start']) / (event['end'] - event['start'])

                if event['type'] == 'partial_unfold':
                    # 部分的アンフォールディング
                    expansion = 1 + progress * 0.02
                    current_pos = (current_pos - initial_positions.mean(axis=0)) * expansion + initial_positions.mean(axis=0)

                elif event['type'] == 'helix_break':
                    # ヘリックス構造の破壊（特定領域）
                    helix_atoms = np.arange(200, 300)  # 仮想的なヘリックス領域
                    current_pos[helix_atoms] += np.random.randn(len(helix_atoms), 3) * progress * 0.5

                elif event['type'] == 'major_unfold':
                    # 大規模アンフォールディング
                    expansion = 1 + progress * 0.05
                    current_pos *= expansion
                    current_pos += np.random.randn(n_atoms, 3) * progress * 0.3

                elif event['type'] == 'misfold':
                    # ミスフォールディング
                    affected = np.random.choice(n_atoms, n_atoms//2, replace=False)
                    current_pos[affected] += np.random.randn(len(affected), 3) * progress * 0.4
                    current_pos *= 0.98  # コンパクト化

                elif event['type'] == 'aggregation_prone':
                    # 凝集傾向状態
                    current_pos *= 0.95
                    cluster_center = np.random.randn(3) * 5
                    current_pos += cluster_center * progress * 0.1

        # 通常の更新
        current_pos += thermal_noise + collective_motion

        # 構造制約（原子同士が近づきすぎない）
        # 簡易版：大きすぎる変位を制限
        deviation = current_pos - initial_positions
        large_dev = np.linalg.norm(deviation, axis=1) > 10.0
        if np.any(large_dev):
            current_pos[large_dev] = initial_positions[large_dev] + deviation[large_dev] * 0.5

        trajectory[i] = current_pos.copy()

    # バックボーンインデックス
    backbone = u.select_atoms('backbone')
    backbone_indices = backbone.indices

    # 保存
    print("\nSaving trajectory...")
    np.save('lysozyme_50k_trajectory.npy', trajectory)
    np.save('lysozyme_50k_backbone_indices.npy', backbone_indices)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds!")
    print(f"Files saved:")
    print(f"  - lysozyme_50k_trajectory.npy ({n_frames * n_atoms * 3 * 4 / 1e9:.1f} GB)")
    print(f"  - lysozyme_50k_backbone_indices.npy")

    # イベント情報も表示
    print("\nEvent timeline:")
    for event in events:
        print(f"  {event['type']}: frames {event['start']}-{event['end']}")

    return trajectory, backbone_indices

# 実行！
if __name__ == "__main__":
    print("=== Creating 50k frame Lysozyme trajectory ===")
    traj, backbone = create_lysozyme_50k_trajectory(50000)

    if traj is not None:
        print("\nReady for Lambda³ analysis!")
        print("Run: detector.analyze(traj, backbone)")
