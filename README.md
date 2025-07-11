# 📕 Lambda³ Zero-Shot Anomaly Detection

## Physics-based Zero-Shot Anomaly Detection at 93.03% AUC — no training, just physical law.

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

## 🚀 Performance Comparison

| Method | AUC Score | Training Data | Interpretability | Detection Time |
|--------|-----------|---------------|------------------|----------------|
| **Lambda³ Basic** | **~93%** | **Zero** | Full physical explanation | 15.8s |
| **Lambda³ Adaptive** | **~93%** | **Zero** | Optimized component weights | 5.4s |
| **Lambda³ Focused** | ~81% | **Zero** | Feature group analysis | 5.5s |
| Traditional Supervised | 70-85% | 1000s of samples | Black box | Variable |
| Deep Learning (LSTM/AE) | 80-90% | 10,000s of samples | Limited/None | Minutes |
| Isolation Forest | 65-80% | 100s of samples | Partial | Seconds |
| One-Class SVM | 60-75% | 100s of samples | Limited | Seconds |

*Results on synthetic complex dataset with progressive degradation, periodic bursts, chaotic bifurcations, and partial anomalies.*

![Results](http://www.miosync.link/github/zeroshot.png)

## 🌟 Key Features

- **Zero Training Required**: Works immediately on new data
- **Superhuman Performance**: 93% AUC without seeing any examples
- **Fully Interpretable**: Complete physical explanation for every anomaly
- **Multi-Scale Detection**: Captures anomalies at different temporal resolutions
- **Fast**: 5-15 seconds for complete analysis
- **Domain Agnostic**: Works on any multivariate time series

“Detects the ‘moments of rupture’—the unseen phase transitions, structural cracks, and the birth of new orders—before any black-box model can learn them.”

*When using multiple important features discovered through optimization

## 🔬 Core Mechanisms

### 📐 Fundamental Components

#### **1. Structure Tensor (Λ)**
Represents data structure in high-dimensional semantic space, capturing latent system states through tensor decomposition.

#### **2. Jump Detection (ΔΛC)** 
Multi-scale detection of sudden structural transitions:
- Adaptive thresholding across temporal scales
- Cross-feature synchronization analysis
- Pulsation event clustering

#### **3. Topological Invariants**
- **Topological Charge (Q_Λ)**: Winding number measuring structural defects
- **Stability Index (σ_Q)**: Variance analysis across path segments
- **Phase transitions**: Bifurcation and symmetry breaking detection

### 📊 Information-Theoretic Analysis

#### **4. Multi-Entropy Framework**
Comprehensive information quantification:
- **Shannon Entropy**: Classical information content
- **Rényi Entropy** (α=2): Collision entropy for rare events
- **Tsallis Entropy** (q=1.5): Non-extensive systems
- **Conditional Entropies**: Jump-conditioned information flow

### 🔧 Mathematical Optimization

#### **5. Inverse Problem Formulation**
Jump-constrained optimization for structure tensor reconstruction:

min ||K - ΛΛᵀ||²_F + α·TV(Λ) + β·||Λ||₁ + γ·J(Λ)

Where J(Λ) enforces jump consistency.

#### **6. Regularization Strategies**
- **Total Variation (TV)**: Preserves discontinuities
- **L1 Regularization**: Promotes sparsity
- **Jump-aware constraints**: Structural coherence

### 🌐 Kernel Methods

#### **7. Multi-Kernel Analysis**
Automatic kernel selection and ensemble:
- **RBF (Gaussian)**: Smooth similarity measures
- **Polynomial**: Higher-order interactions
- **Laplacian**: Heavy-tailed distributions
- **Sigmoid**: Neural network connections

### 🎯 Advanced Features

#### **8. Nonlinear Feature Engineering**
- **Transformations**: log, sqrt, square, sigmoid
- **Interactions**: Products, ratios, compositions
- **Statistics**: Skewness, kurtosis, autocorrelation

#### **9. Synchronization Metrics**
- **Cross-feature correlation**: Jump co-occurrence
- **Lag analysis**: Temporal dependencies
- **Clustering**: Synchronized event groups

#### **10. Pulsation Energy Analysis**
Quantifying structural disruptions:
- **Intensity**: Magnitude of state changes
- **Asymmetry**: Directional bias in transitions
- **Power**: Frequency-weighted energy distribution

### 🔄 Ensemble Architecture

#### **11. Multi-Scale Integration**
- Parallel detection at multiple resolutions
- Adaptive weight optimization
- Component-wise anomaly scoring

#### **12. Hybrid Scoring System**
Unified anomaly quantification combining:
- Topological anomalies
- Energetic disruptions
- Information-theoretic outliers
- Kernel-space deviations


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
