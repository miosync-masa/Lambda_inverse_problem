import numpy as np
import struct
import glob
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

class WIN32ProperDecoder:
    """
    WIN32 Format Official Decoder
    
    Accurate implementation based on specification
    - 1 minute = 60 frames (per second)
    - Each frame = data for all channels
    - Efficient recording with differential compression
    """
    
    def __init__(self, channel_info: Dict[int, Dict] = None):
        """
        Args:
            channel_info: Channel ID -> Station info mapping
        """
        self.channel_info = channel_info or {}
        self.header_size = 4  # Format ID + version + reserved
        self.frame_header_size = 16  # Time(8) + frame duration(4) + data block size(4)
        
    def decode_win32_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Complete WIN32 file decoding
        
        Returns:
            {
                'channel_XXXX': np.ndarray(6000,),  # 1 minute of data
                ...
            }
        """
        print(f"WIN32 decoding started: {os.path.basename(filepath)}")
        
        with open(filepath, 'rb') as f:
            raw_data = f.read()
        
        # Read file header
        file_header = self._read_file_header(raw_data)
        
        # Decode all 60 frames
        decoded_data = {}
        position = self.header_size
        
        for frame_idx in range(60):  # 0-59 seconds
            if position >= len(raw_data):
                print(f"  Warning: Data ended at frame {frame_idx}")
                break
                
            # Read frame header
            frame_header = self._read_frame_header(raw_data[position:])
            position += self.frame_header_size
            
            # Decode all channels in frame
            frame_end = position + frame_header['data_block_size']
            
            while position < frame_end and position < len(raw_data):
                # Decode channel block
                channel_data, bytes_read = self._decode_channel_block(
                    raw_data[position:frame_end]
                )
                
                if channel_data is None:
                    break
                    
                # Accumulate data with channel ID as key
                channel_id = channel_data['channel_id']
                channel_key = f"channel_{channel_id:04X}"
                
                if channel_key not in decoded_data:
                    decoded_data[channel_key] = []
                
                decoded_data[channel_key].extend(channel_data['samples'])
                position += bytes_read
            
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/59 completed")
        
        # Convert lists to numpy arrays
        for key in decoded_data:
            decoded_data[key] = np.array(decoded_data[key])
        
        print(f"Decoding completed: {len(decoded_data)} channels")
        
        return decoded_data
    
    def _read_file_header(self, data: bytes) -> Dict:
        """Read file header"""
        if len(data) < 4:
            raise ValueError("Data too short")
            
        return {
            'format_id': data[0],
            'version': data[1],
            'reserved': struct.unpack('>H', data[2:4])[0]
        }
    
    def _read_frame_header(self, data: bytes) -> Dict:
        """Read frame header"""
        if len(data) < 16:
            raise ValueError("Incomplete frame header")
        
        # Read BCD format time
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
        """Convert BCD format to int"""
        result = 0
        for byte in bcd_bytes:
            high = (byte >> 4) & 0x0F
            low = byte & 0x0F
            result = result * 100 + high * 10 + low
        return result
    
    def _decode_channel_block(self, data: bytes) -> Tuple[Optional[Dict], int]:
        """
        Decode channel block
        
        Returns:
            (decode result, bytes read)
        """
        if len(data) < 8:  # Minimum header size
            return None, 0
        
        position = 0
        
        # Channel block header
        org_id = data[position]
        net_id = data[position + 1]
        channel_id = struct.unpack('>H', data[position + 2:position + 4])[0]
        
        # Sample size and count
        sample_size_bits = (data[position + 4] >> 4) & 0x0F
        sample_count = struct.unpack('>H', data[position + 4:position + 6])[0] & 0x0FFF
        
        position += 6
        
        # First sample (32bit)
        if len(data) < position + 4:
            return None, position
            
        first_sample = struct.unpack('>i', data[position:position + 4])[0]
        position += 4
        
        samples = [first_sample]
        
        # Decode differential samples
        if sample_size_bits == 0:  # 4bit
            samples.extend(self._decode_4bit_diff(data[position:], sample_count - 1))
            position += ((sample_count - 1) + 1) // 2  # 4bit = 0.5 bytes
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
        
        # Restore original values with cumulative sum
        for i in range(1, len(samples)):
            samples[i] = samples[i-1] + samples[i]
        
        return {
            'org_id': org_id,
            'net_id': net_id,
            'channel_id': channel_id,
            'samples': samples
        }, position
    
    def _decode_4bit_diff(self, data: bytes, count: int) -> List[int]:
        """Decode 4-bit differential data"""
        samples = []
        for i in range(count):
            byte_idx = i // 2
            if byte_idx >= len(data):
                break
                
            if i % 2 == 0:
                # Upper 4 bits
                diff = (data[byte_idx] >> 4) & 0x0F
                if diff & 0x08:  # Negative number
                    diff |= 0xFFFFFFF0
            else:
                # Lower 4 bits
                diff = data[byte_idx] & 0x0F
                if diff & 0x08:  # Negative number
                    diff |= 0xFFFFFFF0
                    
            samples.append(diff)
        return samples
    
    def _decode_8bit_diff(self, data: bytes, count: int) -> List[int]:
        """Decode 8-bit differential data"""
        samples = []
        for i in range(min(count, len(data))):
            diff = struct.unpack('b', data[i:i+1])[0]  # signed char
            samples.append(diff)
        return samples
    
    def _decode_16bit_diff(self, data: bytes, count: int) -> List[int]:
        """Decode 16-bit differential data"""
        samples = []
        for i in range(count):
            if i * 2 + 2 > len(data):
                break
            diff = struct.unpack('>h', data[i*2:i*2+2])[0]  # signed short
            samples.append(diff)
        return samples
    
    def _decode_24bit_diff(self, data: bytes, count: int) -> List[int]:
        """Decode 24-bit differential data"""
        samples = []
        for i in range(count):
            if i * 3 + 3 > len(data):
                break
            # Extend 24bit to 32bit (sign extension)
            bytes3 = data[i*3:i*3+3]
            if bytes3[0] & 0x80:  # Negative
                diff = struct.unpack('>i', b'\xff' + bytes3)[0]
            else:
                diff = struct.unpack('>i', b'\x00' + bytes3)[0]
            samples.append(diff)
        return samples
    
    def _decode_32bit_diff(self, data: bytes, count: int) -> List[int]:
        """Decode 32-bit differential data"""
        samples = []
        for i in range(count):
            if i * 4 + 4 > len(data):
                break
            diff = struct.unpack('>i', data[i*4:i*4+4])[0]  # signed int
            samples.append(diff)
        return samples


class WIN32ToLambda3Converter:
    """
    Direct conversion from WIN32 data to Lambda3 analysis format
    
    In Lambda3 theory, observation data itself is a projection to semantic space,
    so no additional feature extraction is performed
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
        Convert all WIN32 files in directory to Lambda3 analysis format
        
        Args:
            input_dir: WIN32 files directory
            output_dir: Output directory
            file_pattern: File pattern
        """
        print(f"=== WIN32 → Lambda3 Direct Conversion Started ===")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Lambda3 Theory: Observation data = Projection to semantic space")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get WIN32 file list
        cnt_files = sorted(glob.glob(os.path.join(input_dir, '**', file_pattern), recursive=True))
        print(f"Target files: {len(cnt_files)}")
        
        # Sort by time sequence
        cnt_files = self._sort_files_by_time(cnt_files)
        
        # Overall data structure
        all_channels_data = {}  # {channel_id: [time series data list]}
        file_info = []  # Information for each file
        
        # Process each file
        for i, filepath in enumerate(cnt_files):
            print(f"\nProcessing ({i+1}/{len(cnt_files)}): {os.path.basename(filepath)}")
            
            try:
                # WIN32 decode
                decoded = self.decoder.decode_win32_file(filepath)
                
                # Record file information
                timestamp = self._extract_timestamp(filepath)
                file_info.append({
                    'filename': os.path.basename(filepath),
                    'timestamp': timestamp,
                    'index': i
                })
                
                # Accumulate data by channel
                for channel_key, data in decoded.items():
                    if channel_key not in all_channels_data:
                        all_channels_data[channel_key] = []
                    all_channels_data[channel_key].append(data)
                
                self.processed_files.append(filepath)
                
            except Exception as e:
                print(f"  Error: {e}")
                self.failed_files.append(filepath)
                continue
        
        # Save as Lambda3 analysis data structure
        self._save_lambda3_format(all_channels_data, file_info, output_dir)
        
        # Save processing summary
        self._save_summary(output_dir)
        
        print(f"\n=== Conversion Completed ===")
        print(f"Success: {len(self.processed_files)} files")
        print(f"Failed: {len(self.failed_files)} files")
    
    def _sort_files_by_time(self, files: List[str]) -> List[str]:
        """Sort files by time extracted from filename"""
        def get_time_key(filepath):
            filename = os.path.basename(filepath)
            try:
                return filename[:14]
            except:
                return filename
        
        return sorted(files, key=get_time_key)
    
    def _extract_timestamp(self, filepath: str) -> str:
        """Extract timestamp from filename"""
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
        """Save in Lambda3 analysis format"""
        print(f"\n=== Saving in Lambda3 Format ===")
        
        # Continuous data for each channel
        for channel_key, data_list in all_channels_data.items():
            print(f"\nChannel {channel_key}:")
            
            # Concatenate time series data
            continuous_data = np.concatenate(data_list)
            print(f"  Total samples: {len(continuous_data)}")
            print(f"  Data range: [{np.min(continuous_data)}, {np.max(continuous_data)}]")
            
            # Save as individual channel file
            channel_file = os.path.join(output_dir, f"{channel_key}_continuous.npy")
            np.save(channel_file, continuous_data)
            
            # Metadata
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
        
        # Create integrated data file (all channels)
        print("\nCreating integrated data...")
        integrated_data = {}
        
        # Align to same length (use minimum length)
        min_length = min(len(np.concatenate(data_list)) for data_list in all_channels_data.values())
        
        for channel_key, data_list in all_channels_data.items():
            continuous_data = np.concatenate(data_list)[:min_length]
            integrated_data[channel_key] = continuous_data
        
        # Save as integrated file
        integrated_file = os.path.join(output_dir, "all_channels_lambda3.npz")
        np.savez_compressed(integrated_file, **integrated_data)
        print(f"Integrated file saved: {integrated_file}")
        
        # Save file information
        file_info_path = os.path.join(output_dir, "file_sequence.json")
        with open(file_info_path, 'w') as f:
            json.dump(file_info, f, indent=2)
    
    def _save_summary(self, output_dir: str):
        """Save processing summary"""
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
        
        print(f"\nSummary saved: {summary_file}")


# Usage example
if __name__ == "__main__":
    # Configuration
    input_directory = "/content/drive/MyDrive/Colab Notebooks/noto_earthquake_20240101/K-NET"
    output_directory = "/content/drive/MyDrive/Colab Notebooks/noto_earthquake_20240101/K-NET_lambda3"
    
    # Initialize decoder and converter
    decoder = WIN32ProperDecoder()
    converter = WIN32ToLambda3Converter(decoder=decoder)
    
    # Execute conversion
    converter.convert_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        file_pattern="*.cnt"
    )
    
    print("\nData structure after conversion:")
    print("- Individual channels: channel_XXXX_continuous.npy")
    print("- Metadata: channel_XXXX_metadata.json")
    print("- Integrated data: all_channels_lambda3.npz")
    print("\nLambda3 analysis preparation completed.")
