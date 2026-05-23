"""
Streaming Lambda³: NAB streaming-style anomaly detection.

設計：
  Calibration phase  -- 先頭 cal_ratio*n フレーム (NAB probationary と整合)
    各 scorer の "正常 baseline" と threshold を hard-fix する。
  Streaming phase    -- 各フレーム t を順次投入、過去のみ参照
    各 scorer が独立に「閾値超え/否」の Binary 判定。
    最終 anomaly = OR over scorers (max-normalized 連続値で出すので
                                    NAB Sweeper の threshold sweep と互換)。

合成 weight チューニング無し、各 axis が独立 binary detector として動くため
"NNNU = Neural Network Non-Use" の真意 (parameter-free voting) を体現する。

Future leakage 厳禁：score(history, t) は events[:t+1] のみ使用。
"""

from .base import StreamingScorer
from .jump_streaming import StreamingJumpScorer
from .gradual_streaming import StreamingGradualScorer
from .drift_streaming import StreamingStructuralDriftScorer
from .reconstruction_streaming import StreamingReconstructionScorer
from .kernel_streaming import StreamingKernelScorer
from .structural_streaming import StreamingStructuralScorer
from .periodic_streaming import StreamingPeriodicScorer
from .detector import Lambda3StreamingDetector

__all__ = [
    'StreamingScorer',
    'StreamingJumpScorer',
    'StreamingGradualScorer',
    'StreamingStructuralDriftScorer',
    'StreamingReconstructionScorer',
    'StreamingKernelScorer',
    'StreamingStructuralScorer',
    'StreamingPeriodicScorer',
    'Lambda3StreamingDetector',
]
