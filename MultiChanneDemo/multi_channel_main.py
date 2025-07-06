"""
Multi-Channel Lambda³ Detection System - Main Execution Script
マルチチャンネル異常検知システム実行スクリプト
"""
import numpy as np
import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from dataclasses import dataclass

# Lambda³モジュール（それぞれ独立したファイル）
from lambda3_detector import Lambda3ZeroShotDetector, L3Config
from sync_analyzer import SyncAnalyzer  # 独立したsync_analyzerモジュール
from multi_channel_detector import (
    MultiChannelDetector, 
    ChannelConfig, 
    prepare_for_integration,
    save_integration_results
)

@dataclass
class DetectionConfig:
    """検知システムの設定"""
    # Lambda³設定
    n_paths: int = 5
    jump_scale: float = 2.0
    use_adaptive_weights: bool = False
    
    # マルチチャンネル設定
    sync_threshold: float = 0.3
    lag_window: int = 10
    integration_method: str = "weighted_sync"
    
    # 処理設定
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    save_intermediate: bool = True
    visualize: bool = True

def process_single_channel(channel_name: str, 
                         channel_data: np.ndarray,
                         config: DetectionConfig,
                         save_dir: str = "./lambda3_results") -> Tuple[str, Dict]:
    """
    単一チャンネルのLambda³処理
    
    Args:
        channel_name: チャンネル名
        channel_data: チャンネルデータ (n_events, n_features)
        config: 検知設定
        save_dir: 保存ディレクトリ
        
    Returns:
        (channel_name, result_dict)
    """
    print(f"Processing channel: {channel_name}")
    start_time = time.time()
    
    # Lambda³検出器の初期化
    l3_config = L3Config(
        n_paths=config.n_paths,
        jump_scale=config.jump_scale
    )
    detector = Lambda3ZeroShotDetector(l3_config)
    
    # 解析実行
    result = detector.analyze(channel_data)
    scores = detector.detect_anomalies(
        result, 
        channel_data,
        use_adaptive_weights=config.use_adaptive_weights
    )
    
    # 中間結果の保存
    if config.save_intermediate:
        saved_files = detector.save_results(
            result=result,
            anomaly_scores=scores,
            events=channel_data,
            channel_name=channel_name,
            save_dir=save_dir
        )
    
    process_time = time.time() - start_time
    print(f"Channel {channel_name} processed in {process_time:.2f}s")
    
    # 統合用のデータを返す
    return channel_name, {
        'scores': scores,
        'result': result,
        'events': channel_data,
        'process_time': process_time
    }

def process_multi_channel_batch(channel_data_dict: Dict[str, np.ndarray],
                              config: DetectionConfig,
                              save_dir: str = "./lambda3_results") -> Dict[str, Dict]:
    """
    複数チャンネルのバッチ処理
    
    Args:
        channel_data_dict: {channel_name: channel_data}
        config: 検知設定
        save_dir: 保存ディレクトリ
        
    Returns:
        処理結果の辞書
    """
    results = {}
    total_start = time.time()
    
    print(f"\n=== Phase 1: Lambda³ Analysis ===")
    print(f"Processing {len(channel_data_dict)} channels...")
    
    if config.parallel_processing:
        # 並列処理
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            # ジョブの投入
            futures = {
                executor.submit(
                    process_single_channel, 
                    channel_name, 
                    channel_data, 
                    config, 
                    save_dir
                ): channel_name
                for channel_name, channel_data in channel_data_dict.items()
            }
            
            # 結果の収集
            for future in as_completed(futures):
                try:
                    channel_name, result = future.result()
                    results[channel_name] = result
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
    else:
        # 逐次処理
        for channel_name, channel_data in channel_data_dict.items():
            _, result = process_single_channel(
                channel_name, channel_data, config, save_dir
            )
            results[channel_name] = result
    
    # サマリーファイルの作成
    if config.save_intermediate:
        summary = {
            'n_channels': len(results),
            'channels': list(results.keys()),
            'timestamp': str(np.datetime64('now')),
            'config': {
                'n_paths': config.n_paths,
                'jump_scale': config.jump_scale,
                'use_adaptive_weights': config.use_adaptive_weights
            },
            'channel_stats': {}
        }
        
        for channel_name, data in results.items():
            scores = data['scores']
            summary['channel_stats'][channel_name] = {
                'mean_score': float(np.mean(scores)),
                'max_score': float(np.max(scores)),
                'std_score': float(np.std(scores)),
                'n_events': len(scores),
                'process_time': data['process_time']
            }
        
        summary_path = os.path.join(save_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    phase1_time = time.time() - total_start
    print(f"\nPhase 1 completed in {phase1_time:.2f}s")
    
    return results

def run_integration_analysis(lambda3_results: Dict[str, Dict],
                           config: DetectionConfig,
                           channel_configs: Optional[Dict[str, ChannelConfig]] = None,
                           integration_save_dir: str = "./integration_results") -> 'IntegrationResult':
    """
    統合解析の実行
    
    Args:
        lambda3_results: Lambda³処理結果
        config: 検知設定
        channel_configs: チャンネル設定（センサー位置など）
        integration_save_dir: 統合結果の保存ディレクトリ
        
    Returns:
        統合結果
    """
    print(f"\n=== Phase 2-3: Integration Analysis ===")
    integration_start = time.time()
    
    # マルチチャンネル検出器の初期化
    mc_detector = MultiChannelDetector(
        sync_threshold=config.sync_threshold,
        lag_window=config.lag_window,
        integration_method=config.integration_method
    )
    
    # チャンネル設定があれば適用
    if channel_configs:
        mc_detector.set_channel_configs(channel_configs)
    
    # 統合実行
    integration_result = mc_detector.integrate_channels(
        lambda3_results,
        use_jump_sync=True,
        visualize=config.visualize
    )
    
    # 統合結果の保存
    os.makedirs(integration_save_dir, exist_ok=True)
    save_integration_results(integration_result, integration_save_dir)
    
    integration_time = time.time() - integration_start
    print(f"\nIntegration completed in {integration_time:.2f}s")
    
    # 最終レポート
    print(f"\n=== Detection Summary ===")
    print(f"Total anomaly regions detected: {len(integration_result.anomaly_regions)}")
    print(f"Detection confidence: {integration_result.confidence:.2%}")
    
    if integration_result.anomaly_regions:
        print("\nAnomaly regions:")
        for i, (start, end) in enumerate(integration_result.anomaly_regions):
            duration = end - start
            print(f"  Region {i+1}: [{start}:{end}] (duration: {duration} samples)")
    
    if integration_result.defect_locations:
        print("\nEstimated defect locations:")
        for region, info in integration_result.defect_locations.items():
            print(f"  {region}: position={info['estimated_position']}, "
                  f"confidence={info['confidence']:.2%}")
    
    return integration_result

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="Multi-Channel Lambda³ Detection System"
    )
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing channel data files')
    parser.add_argument('--channels', nargs='+', default=None,
                       help='List of channel names to process')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Configuration file (JSON)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--integration-only', action='store_true',
                       help='Run only integration on existing results')
    
    args = parser.parse_args()
    
    # 設定の読み込み
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = DetectionConfig(**config_dict)
    else:
        config = DetectionConfig()
    
    # コマンドライン引数で上書き
    config.parallel_processing = not args.no_parallel
    config.visualize = not args.no_visualize
    
    if args.integration_only:
        # 既存の結果から統合のみ実行
        print("Running integration analysis on existing results...")
        lambda3_results = prepare_for_integration(channels=args.channels)
        
        # TODO: チャンネル設定の読み込み
        channel_configs = None
        
        integration_result = run_integration_analysis(
            lambda3_results, config, channel_configs
        )
    else:
        # フル実行
        # データの読み込み（実装は使用するデータ形式に依存）
        channel_data_dict = load_channel_data(args.data_dir, args.channels)
        
        # Phase 1: Lambda³解析
        lambda3_results = process_multi_channel_batch(
            channel_data_dict, config
        )
        
        # Phase 2-3: 統合解析
        # TODO: チャンネル設定の読み込み
        channel_configs = None
        
        integration_result = run_integration_analysis(
            lambda3_results, config, channel_configs
        )
    
    print("\n=== Processing Complete ===")
    return integration_result

def load_channel_data(data_dir: str, 
                     channels: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    チャンネルデータの読み込み（実装例）
    
    Args:
        data_dir: データディレクトリ
        channels: 読み込むチャンネルのリスト
        
    Returns:
        {channel_name: channel_data}
    """
    channel_data_dict = {}
    
    # .npyファイルを探す
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    for npy_file in npy_files:
        channel_name = os.path.splitext(npy_file)[0]
        
        # チャンネル指定がある場合はフィルタ
        if channels and channel_name not in channels:
            continue
        
        file_path = os.path.join(data_dir, npy_file)
        channel_data_dict[channel_name] = np.load(file_path)
        print(f"Loaded {channel_name}: shape={channel_data_dict[channel_name].shape}")
    
    if not channel_data_dict:
        raise ValueError(f"No channel data found in {data_dir}")
    
    return channel_data_dict

# ===============================
# デモ/テスト用関数
# ===============================
def demo_multi_channel_system():
    """完全なシステムのデモ"""
    print("=== Multi-Channel Lambda³ Detection System Demo ===\n")
    
    # デモ用の3チャンネルデータ生成
    n_events = 1000
    n_features = 10
    
    # 基本的な正常データ
    base_data = np.random.randn(n_events, n_features) * 0.5
    
    # 異常パターンの注入
    anomaly_start = 300
    anomaly_end = 350
    anomaly_pattern = np.random.randn(anomaly_end - anomaly_start, n_features) * 3
    
    # 3チャンネルのデータ（異常が伝播する）
    channel_data_dict = {
        'sensor_1': base_data.copy(),
        'sensor_2': base_data.copy(),
        'sensor_3': base_data.copy()
    }
    
    # センサー1に異常を注入
    channel_data_dict['sensor_1'][anomaly_start:anomaly_end] += anomaly_pattern
    
    # センサー2には2サンプル遅れで伝播
    channel_data_dict['sensor_2'][anomaly_start+2:anomaly_end+2] += anomaly_pattern * 0.8
    
    # センサー3には5サンプル遅れで伝播
    channel_data_dict['sensor_3'][anomaly_start+5:anomaly_end+5] += anomaly_pattern * 0.6
    
    # 設定
    config = DetectionConfig(
        n_paths=5,
        jump_scale=2.0,
        sync_threshold=0.3,
        lag_window=10,
        integration_method="weighted_sync",
        parallel_processing=False,  # デモでは逐次実行
        visualize=True
    )
    
    # センサー位置設定
    channel_configs = {
        'sensor_1': ChannelConfig('sensor_1', position=np.array([0, 0, 0])),
        'sensor_2': ChannelConfig('sensor_2', position=np.array([1, 0, 0])),
        'sensor_3': ChannelConfig('sensor_3', position=np.array([0.5, 1, 0]))
    }
    
    # Phase 1: Lambda³解析
    lambda3_results = process_multi_channel_batch(channel_data_dict, config)
    
    # Phase 2-3: 統合解析
    integration_result = run_integration_analysis(
        lambda3_results, config, channel_configs
    )
    
    return integration_result

if __name__ == "__main__":
    # コマンドライン引数がない場合はデモを実行
    import sys
    if len(sys.argv) == 1:
        demo_multi_channel_system()
    else:
        main()
        
