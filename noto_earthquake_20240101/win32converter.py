import numpy as np
import struct
import glob
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

class WIN32ProperDecoder:
    """
    WIN32フォーマット正式デコーダー
    
    仕様書に基づく正確な実装
    - 1分間 = 60フレーム（1秒ごと）
    - 各フレーム = 全チャンネルのデータ
    - 差分圧縮による効率的な記録
    """
    
    def __init__(self, channel_info: Dict[int, Dict] = None):
        """
        Args:
            channel_info: チャンネルID -> 観測点情報のマッピング
        """
        self.channel_info = channel_info or {}
        self.header_size = 4  # フォーマットID + バージョン + 予備
        self.frame_header_size = 16  # 時刻(8) + フレーム時間長(4) + データブロックサイズ(4)
        
    def decode_win32_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        WIN32ファイルの完全デコード
        
        Returns:
            {
                'channel_XXXX': np.ndarray(6000,),  # 1分間のデータ
                ...
            }
        """
        print(f"WIN32デコード開始: {os.path.basename(filepath)}")
        
        with open(filepath, 'rb') as f:
            raw_data = f.read()
        
        # ファイルヘッダー読み込み
        file_header = self._read_file_header(raw_data)
        
        # 全60フレームをデコード
        decoded_data = {}
        position = self.header_size
        
        for frame_idx in range(60):  # 0-59秒
            if position >= len(raw_data):
                print(f"  警告: フレーム{frame_idx}でデータ終了")
                break
                
            # フレームヘッダー読み込み
            frame_header = self._read_frame_header(raw_data[position:])
            position += self.frame_header_size
            
            # フレーム内の全チャンネルをデコード
            frame_end = position + frame_header['data_block_size']
            
            while position < frame_end and position < len(raw_data):
                # チャンネルブロックをデコード
                channel_data, bytes_read = self._decode_channel_block(
                    raw_data[position:frame_end]
                )
                
                if channel_data is None:
                    break
                    
                # チャンネルIDをキーとしてデータを蓄積
                channel_id = channel_data['channel_id']
                channel_key = f"channel_{channel_id:04X}"
                
                if channel_key not in decoded_data:
                    decoded_data[channel_key] = []
                
                decoded_data[channel_key].extend(channel_data['samples'])
                position += bytes_read
            
            if frame_idx % 10 == 0:
                print(f"  フレーム {frame_idx}/59 完了")
        
        # リストをnumpy配列に変換
        for key in decoded_data:
            decoded_data[key] = np.array(decoded_data[key])
        
        print(f"デコード完了: {len(decoded_data)}チャンネル")
        
        return decoded_data
    
    def _read_file_header(self, data: bytes) -> Dict:
        """ファイルヘッダーの読み込み"""
        if len(data) < 4:
            raise ValueError("データが短すぎます")
            
        return {
            'format_id': data[0],
            'version': data[1],
            'reserved': struct.unpack('>H', data[2:4])[0]
        }
    
    def _read_frame_header(self, data: bytes) -> Dict:
        """フレームヘッダーの読み込み"""
        if len(data) < 16:
            raise ValueError("フレームヘッダーが不完全")
        
        # BCD形式の時刻を読み込み
        time_bcd = data[0:8]
        year = self._bcd_to_int(time_bcd[0:2])
        month = self._bcd_to_int(time_bcd[2:3])
        day = self._bcd_to_int(time_bcd[3:4])
        hour = self._bcd_to_int(time_bcd[4:5])
        minute = self._bcd_to_int(time_bcd[5:6])
        second = self._bcd_to_int(time_bcd[6:7])
        
        frame_time_length = struct.unpack('>I', data[8:12])[0]
        data_block_size = struct.unpack('>I', data[12:16])[0]
        
        return {
            'time': f"{year:04d}/{month:02d}/{day:02d} {hour:02d}:{minute:02d}:{second:02d}",
            'frame_time_length': frame_time_length,
            'data_block_size': data_block_size
        }
    
    def _bcd_to_int(self, bcd_bytes: bytes) -> int:
        """BCD形式をintに変換"""
        result = 0
        for byte in bcd_bytes:
            high = (byte >> 4) & 0x0F
            low = byte & 0x0F
            result = result * 100 + high * 10 + low
        return result
    
    def _decode_channel_block(self, data: bytes) -> Tuple[Optional[Dict], int]:
        """
        チャンネルブロックのデコード
        
        Returns:
            (デコード結果, 読み込みバイト数)
        """
        if len(data) < 8:  # 最小ヘッダーサイズ
            return None, 0
        
        position = 0
        
        # チャンネルブロックヘッダー
        org_id = data[position]
        net_id = data[position + 1]
        channel_id = struct.unpack('>H', data[position + 2:position + 4])[0]
        
        # サンプルサイズとサンプル数
        sample_size_bits = (data[position + 4] >> 4) & 0x0F
        sample_count = struct.unpack('>H', data[position + 4:position + 6])[0] & 0x0FFF
        
        position += 6
        
        # 最初のサンプル（32bit）
        if len(data) < position + 4:
            return None, position
            
        first_sample = struct.unpack('>i', data[position:position + 4])[0]
        position += 4
        
        samples = [first_sample]
        
        # 差分サンプルのデコード
        if sample_size_bits == 0:  # 4bit
            samples.extend(self._decode_4bit_diff(data[position:], sample_count - 1))
            position += ((sample_count - 1) + 1) // 2  # 4bit = 0.5バイト
        elif sample_size_bits == 1:  # 8bit
            samples.extend(self._decode_8bit_diff(data[position:], sample_count - 1))
            position += sample_count - 1
        elif sample_size_bits == 2:  # 16bit
            samples.extend(self._decode_16bit_diff(data[position:], sample_count - 1))
            position += (sample_count - 1) * 2
        elif sample_size_bits == 3:  # 24bit
            samples.extend(self._decode_24bit_diff(data[position:], sample_count - 1))
            position += (sample_count - 1) * 3
        elif sample_size_bits == 4:  # 32bit
            samples.extend(self._decode_32bit_diff(data[position:], sample_count - 1))
            position += (sample_count - 1) * 4
        
        # 累積和で元の値を復元
        for i in range(1, len(samples)):
            samples[i] = samples[i-1] + samples[i]
        
        return {
            'org_id': org_id,
            'net_id': net_id,
            'channel_id': channel_id,
            'samples': samples
        }, position
    
    def _decode_4bit_diff(self, data: bytes, count: int) -> List[int]:
        """4bit差分データのデコード"""
        samples = []
        for i in range(count):
            byte_idx = i // 2
            if byte_idx >= len(data):
                break
                
            if i % 2 == 0:
                # 上位4bit
                diff = (data[byte_idx] >> 4) & 0x0F
                if diff & 0x08:  # 負数の場合
                    diff |= 0xFFFFFFF0
            else:
                # 下位4bit
                diff = data[byte_idx] & 0x0F
                if diff & 0x08:  # 負数の場合
                    diff |= 0xFFFFFFF0
                    
            samples.append(diff)
        return samples
    
    def _decode_8bit_diff(self, data: bytes, count: int) -> List[int]:
        """8bit差分データのデコード"""
        samples = []
        for i in range(min(count, len(data))):
            diff = struct.unpack('b', data[i:i+1])[0]  # signed char
            samples.append(diff)
        return samples
    
    def _decode_16bit_diff(self, data: bytes, count: int) -> List[int]:
        """16bit差分データのデコード"""
        samples = []
        for i in range(count):
            if i * 2 + 2 > len(data):
                break
            diff = struct.unpack('>h', data[i*2:i*2+2])[0]  # signed short
            samples.append(diff)
        return samples
    
    def _decode_24bit_diff(self, data: bytes, count: int) -> List[int]:
        """24bit差分データのデコード"""
        samples = []
        for i in range(count):
            if i * 3 + 3 > len(data):
                break
            # 24bitを32bitに拡張（符号拡張）
            bytes3 = data[i*3:i*3+3]
            if bytes3[0] & 0x80:  # 負数
                diff = struct.unpack('>i', b'\xff' + bytes3)[0]
            else:
                diff = struct.unpack('>i', b'\x00' + bytes3)[0]
            samples.append(diff)
        return samples
    
    def _decode_32bit_diff(self, data: bytes, count: int) -> List[int]:
        """32bit差分データのデコード"""
        samples = []
        for i in range(count):
            if i * 4 + 4 > len(data):
                break
            diff = struct.unpack('>i', data[i*4:i*4+4])[0]  # signed int
            samples.append(diff)
        return samples


class WIN32ToLambda3Converter:
    """
    WIN32データをLambda3解析用に直接変換
    
    Lambda3理論では観測データそのものが意味空間への射影であるため、
    追加の特徴量抽出は行わない
    """
    
    def __init__(self, decoder=None):
        self.decoder = decoder or WIN32ProperDecoder()
        self.processed_files = []
        self.failed_files = []
        
    def convert_directory(self, 
                         input_dir: str,
                         output_dir: str,
                         file_pattern: str = "*.cnt"):
        """
        ディレクトリ内の全WIN32ファイルをLambda3解析用形式に変換
        
        Args:
            input_dir: WIN32ファイルのディレクトリ
            output_dir: 出力ディレクトリ
            file_pattern: ファイルパターン
        """
        print(f"=== WIN32 → Lambda3 直接変換開始 ===")
        print(f"入力: {input_dir}")
        print(f"出力: {output_dir}")
        print(f"Lambda3理論: 観測データ = 意味空間への射影")
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # WIN32ファイルリスト取得
        cnt_files = sorted(glob.glob(os.path.join(input_dir, '**', file_pattern), recursive=True))
        print(f"対象ファイル数: {len(cnt_files)}")
        
        # 時系列でソート
        cnt_files = self._sort_files_by_time(cnt_files)
        
        # 全体のデータ構造
        all_channels_data = {}  # {channel_id: [時系列データのリスト]}
        file_info = []  # 各ファイルの情報
        
        # 各ファイルを処理
        for i, filepath in enumerate(cnt_files):
            print(f"\n処理中 ({i+1}/{len(cnt_files)}): {os.path.basename(filepath)}")
            
            try:
                # WIN32デコード
                decoded = self.decoder.decode_win32_file(filepath)
                
                # ファイル情報を記録
                timestamp = self._extract_timestamp(filepath)
                file_info.append({
                    'filename': os.path.basename(filepath),
                    'timestamp': timestamp,
                    'index': i
                })
                
                # チャンネルごとにデータを蓄積
                for channel_key, data in decoded.items():
                    if channel_key not in all_channels_data:
                        all_channels_data[channel_key] = []
                    all_channels_data[channel_key].append(data)
                
                self.processed_files.append(filepath)
                
            except Exception as e:
                print(f"  エラー: {e}")
                self.failed_files.append(filepath)
                continue
        
        # Lambda3解析用のデータ構造として保存
        self._save_lambda3_format(all_channels_data, file_info, output_dir)
        
        # 処理結果のサマリー
        self._save_summary(output_dir)
        
        print(f"\n=== 変換完了 ===")
        print(f"成功: {len(self.processed_files)}ファイル")
        print(f"失敗: {len(self.failed_files)}ファイル")
    
    def _sort_files_by_time(self, files: List[str]) -> List[str]:
        """ファイル名から時刻を抽出してソート"""
        def get_time_key(filepath):
            filename = os.path.basename(filepath)
            try:
                return filename[:14]
            except:
                return filename
        
        return sorted(files, key=get_time_key)
    
    def _extract_timestamp(self, filepath: str) -> str:
        """ファイル名からタイムスタンプ抽出"""
        filename = os.path.basename(filepath)
        try:
            year = filename[0:4]
            month = filename[4:6]
            day = filename[6:8]
            hour = filename[8:10]
            minute = filename[10:12]
            second = filename[12:14]
            return f"{year}-{month}-{day} {hour}:{minute}:{second}"
        except:
            return filename
    
    def _save_lambda3_format(self, 
                            all_channels_data: Dict[str, List[np.ndarray]], 
                            file_info: List[Dict],
                            output_dir: str):
        """Lambda3解析用フォーマットで保存"""
        print(f"\n=== Lambda3形式での保存 ===")
        
        # チャンネルごとの連続データ
        for channel_key, data_list in all_channels_data.items():
            print(f"\nチャンネル {channel_key}:")
            
            # 時系列データを連結
            continuous_data = np.concatenate(data_list)
            print(f"  総サンプル数: {len(continuous_data)}")
            print(f"  データ範囲: [{np.min(continuous_data)}, {np.max(continuous_data)}]")
            
            # 個別チャンネルファイルとして保存
            channel_file = os.path.join(output_dir, f"{channel_key}_continuous.npy")
            np.save(channel_file, continuous_data)
            
            # メタデータ
            metadata = {
                'channel': channel_key,
                'total_samples': len(continuous_data),
                'n_files': len(data_list),
                'samples_per_file': [len(d) for d in data_list],
                'data_range': {
                    'min': float(np.min(continuous_data)),
                    'max': float(np.max(continuous_data)),
                    'mean': float(np.mean(continuous_data)),
                    'std': float(np.std(continuous_data))
                }
            }
            
            metadata_file = os.path.join(output_dir, f"{channel_key}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # 統合データファイル（全チャンネル）
        print("\n統合データの作成...")
        integrated_data = {}
        
        # 同じ長さに揃える（最小長に合わせる）
        min_length = min(len(np.concatenate(data_list)) for data_list in all_channels_data.values())
        
        for channel_key, data_list in all_channels_data.items():
            continuous_data = np.concatenate(data_list)[:min_length]
            integrated_data[channel_key] = continuous_data
        
        # 統合ファイルとして保存
        integrated_file = os.path.join(output_dir, "all_channels_lambda3.npz")
        np.savez_compressed(integrated_file, **integrated_data)
        print(f"統合ファイル保存: {integrated_file}")
        
        # ファイル情報も保存
        file_info_path = os.path.join(output_dir, "file_sequence.json")
        with open(file_info_path, 'w') as f:
            json.dump(file_info, f, indent=2)
    
    def _save_summary(self, output_dir: str):
        """処理結果のサマリー保存"""
        summary = {
            'processing_time': datetime.now().isoformat(),
            'lambda3_theory': 'Direct observation data as semantic space projection',
            'processed_files': len(self.processed_files),
            'failed_files': len(self.failed_files),
            'failed_list': [os.path.basename(f) for f in self.failed_files],
            'output_format': {
                'individual_channels': '*_continuous.npy',
                'metadata': '*_metadata.json',
                'integrated': 'all_channels_lambda3.npz',
                'file_sequence': 'file_sequence.json'
            }
        }
        
        summary_file = os.path.join(output_dir, "conversion_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nサマリー保存: {summary_file}")


# 使用例
if __name__ == "__main__":
    # 設定
    input_directory = "/content/drive/MyDrive/Colab Notebooks/noto_earthquake_20240101/K-NET"
    output_directory = "/content/drive/MyDrive/Colab Notebooks/noto_earthquake_20240101/K-NET_lambda3"
    
    # デコーダーと変換器を初期化
    decoder = WIN32ProperDecoder()
    converter = WIN32ToLambda3Converter(decoder=decoder)
    
    # 変換実行
    converter.convert_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        file_pattern="*.cnt"
    )
    
    print("\n変換後のデータ構造：")
    print("- 個別チャンネル: channel_XXXX_continuous.npy")
    print("- メタデータ: channel_XXXX_metadata.json")
    print("- 統合データ: all_channels_lambda3.npz")
    print("\nLambda3解析の準備が完了しました。")
