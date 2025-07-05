# Lambda³ Zero-Shot Anomaly Detection

A physics-inspired anomaly detection system that requires no training data, based on Lambda³ (Lambda-Cubed) theory.

## Overview

This system detects anomalies by analyzing structural changes (ΔΛC jumps) and topological invariants in data, achieving competitive performance without any labeled training examples.

--
## Key Features

- **Zero-Shot Learning**: No training data required
- **Physics-Based**: Uses topological charge Q_Λ and structure tensors
- **Interpretable**: Provides physical explanations for detected anomalies
- **JIT-Optimized**: Fast execution with Numba compilation

--
## Performance

On synthetic datasets with complex anomaly patterns:
- Basic detection: ~80% AUC
- With feature optimization: ~81% AUC
- No training required

--
## Core Concepts

1. **Structure Tensor (Λ)**: Represents data structure in semantic space
2. **Jump Detection (ΔΛC)**: Identifies sudden structural changes
3. **Topological Charge (Q_Λ)**: Measures structural defects
4. **Multi-Entropy Analysis**: Shannon, Renyi, and Tsallis entropies

--
## Usage

```python
from lambda3_detector import Lambda3ZeroShotDetector

# Initialize detector
detector = Lambda3ZeroShotDetector()

# Analyze data (no training needed)
result = detector.analyze(events, n_paths=5)

# Detect anomalies
anomaly_scores = detector.detect_anomalies(result, events)
```
--
## Requirements

1. **Python 3.8+
2. **NumPy
3. **Numba
4. **scikit-learn
5. **SciPy
6. **matplotlib
7. 
--
## Installation

```bash
pip install -r requirements.txt
```
--
## Theory Background

Lambda³ theory models phenomena without assuming time or causality, using:

1. **Structure tensors (Λ)
2. **Progression vectors (ΛF)
3. **Tension scalars (ρT)

The key insight is that anomalies manifest as topological defects in the structure space, particularly visible in the topological charge Q_Λ.

--
## Citation
If you use this code, please cite:

@software{lambda3_anomaly,
  title={Lambda³ Zero-Shot Anomaly Detection},
  author={Based on Dr. Iizumi's Lambda³ Theory},
  year={2025}
}
--
## License

MIT License





