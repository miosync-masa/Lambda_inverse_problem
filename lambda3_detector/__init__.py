"""
Lambda³ Zero-Shot Dual Anomaly Detection System — package facade.

Public surface mirrors the pre-split monolith so existing call sites
(e.g. ``from lambda3_detector import Lambda3ZeroShotDetector, L3Config``)
continue to work unchanged.

For ablation / component-level evaluation, import directly from
sub-packages::

    from lambda3_detector.scorers import (
        JumpScorer, HybridScorer, KernelScorer, StructuralScorer,
    )
"""

from .config import (
    DetectionStrategy,
    L3Config,
    Lambda3Result,
    OptimizationResult,
)
from .detector import Lambda3ZeroShotDetector
from .detector_dual import Lambda3DualDetector
from .features import Lambda3FeatureExtractor, Lambda3FeatureOptimizer

__all__ = [
    'Lambda3ZeroShotDetector',
    'Lambda3DualDetector',
    'L3Config',
    'Lambda3Result',
    'OptimizationResult',
    'DetectionStrategy',
    'Lambda3FeatureExtractor',
    'Lambda3FeatureOptimizer',
]
