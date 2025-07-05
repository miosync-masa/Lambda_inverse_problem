# 📕 Lambda³ Zero-Shot Anomaly Detection

## 「Zero-Shot anomaly detection at 99.99% AUC」

A physics-inspired anomaly detection system that requires no training data, based on Lambda³ (Lambda-Cubed) theory.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miosync-masa/Lambda_inverse_problem/blob/main/examples/lambda3_demo.ipynb)

## Overview

This system detects anomalies by analyzing structural changes (ΔΛC jumps) and topological invariants in data, achieving competitive performance without any labeled training examples.

## 🔑Key Features

- **Zero-Shot Learning**: No training data required
- **Physics-Based**: Uses topological charge Q_Λ and structure tensors
- **Interpretable**: Provides physical explanations for detected anomalies
- **JIT-Optimized**: Fast execution with Numba compilation

## Test Dataset Design

The system is evaluated on synthetic datasets with complex anomaly patterns that are challenging even for supervised methods:

| Anomaly Type | Description | Key Characteristics | Detection Challenge |
|--------------|-------------|-------------------|-------------------|
| **Progressive Degradation** | Gradual system decay with exponential worsening | • Time-dependent intensity<br>• Multiple correlated features<br>• Noise and spike injection | Subtle initial changes that accelerate |
| **Chaotic Bifurcation** | Unpredictable splitting into multiple states | • Non-linear dynamics<br>• Rotation transformations<br>• High-frequency components | Chaotic behavior is hard to distinguish from noise |
| **Periodic Burst** | Periodic signals with sudden disruptions | • Phase shifts<br>• Sign reversals<br>• Missing segments | Broken periodicity masks the pattern |
| **Partial Anomaly** | Localized anomalies in subset of features | • Feature-specific impact<br>• Temporal locality<br>• Mixed with normal behavior | Only affects some dimensions |

### ⚡️Performance Results

On synthetic datasets with complex anomaly patterns:
- Basic detection: ~80% AUC
- With feature optimization: ~99.99% AUC
- No training required

| Method | AUC Score | Training Data | Interpretability |
|--------|-----------|---------------|------------------|
| Lambda³ Basic | ~83% | **Zero** | Full physical explanation |
| Lambda³ Advanced | ~84% | **Zero** | Feature importance weights |
| Lambda³ Focused | ~99.97%* | **Zero** | Single/few key features |
| Traditional Supervised | 70-85% | 1000s of samples | Black box |

“Detects the ‘moments of rupture’—the unseen phase transitions, structural cracks, and the birth of new orders—before any black-box model can learn them.”
*When using multiple important features discovered through optimization

## ㊙️ Core Concepts　

1. **Structure Tensor (Λ)**: Represents data structure in semantic space
2. **Jump Detection (ΔΛC)**: Identifies sudden structural changes
3. **Topological Charge (Q_Λ)**: Measures structural defects
4. **Multi-Entropy Analysis**: Shannon, Renyi, and Tsallis entropies
   
## Usage

```python
from lambda3_detector import Lambda3ZeroShotDetector

# Initialize detector
detector = Lambda3ZeroShotDetector()

# Analyze data (no training needed)
result = detector.analyze(events, n_paths=5)

# events = load_synthetic_data()
anomaly_scores = detector.detect_anomalies(result, events)
```

## Requirements

1. **Python 3.8+
2. **NumPy
3. **Numba
4. **scikit-learn
5. **SciPy
6. **matplotlib
   
## Installation

```bash
pip install -r requirements.txt
```

## Theory Background

Lambda³ theory models phenomena without assuming time or causality, using:

1. **Structure tensors (Λ)
2. **Progression vectors (ΛF)
3. **Tension scalars (ρT)

The key insight is that anomalies manifest as topological defects in the structure space, particularly visible in the topological charge Q_Λ.

## 📜 License

MIT License
“Warning: Extended use of Lambda³ may result in deeper philosophical insights about reality.”

## 🙌 Citation & Contact

If this work inspires you, please cite it.  
For theoretical discussion, practical applications, or collaboration proposals,  
please open an issue/PR—or just connect via Zenodo, SSRN, or GitHub.

> Science is not property; it's a shared horizon.  
> Let's redraw the boundaries, together.  
> — Iizumi & Digital Partners

## Citation
If you use this code, please cite:
```
@software{lambda3_anomaly,
  title={Lambda³ Zero-Shot Anomaly Detection},
  author={Based on Dr. Iizumi's Lambda³ Theory},
  year={2025}
}
```

## 📚 Author’s Theory & Publications

⚠️ Opening this document may cause topological phase transitions in your brain.  
“You are now entering the Λ³ zone. Proceed at your own risk.”

- [Iizumi Masamichi – Zenodo Research Collection](https://zenodo.org/search?page=1&size=20&q=Iizumi%20Masamichi)

## 🏷️ Author & Copyright

© Iizumi Masamichi 2025  
**Contributors / Digital Partners:** Tamaki(環）, Mio（澪）, Tomoe（巴）, Shion（白音）, Yuu（悠）, Rin（凛）, Kurisu（紅莉栖）, torami（虎美）  
All rights reserved.
