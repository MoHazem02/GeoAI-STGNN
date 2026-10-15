# Spatio-Temporal GNN & Active Learning Pipeline for Satellite Environmental Change Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Compatible-green.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, production-grade micro-proof-of-concept for detecting environmental anomalies (such as wildfires, deforestation, or flood inundation) using multi-spectral satellite time-series. The system integrates **Spatio-Temporal Graph Neural Networks (ST-GNN)** with a human-in-the-loop **Active Learning Pipeline** to minimize manual GIS annotation overhead.

---

## 🛰️ Problem Formulation & GIS Context

In Earth Observation (EO), satellite constellations like **Sentinel-2 (MSI)** revisit Earth locations at high cadence (5-day revisit), yielding multi-spectral spatio-temporal data cubes. However, ground-truth change validation (e.g., post-disaster field inspection or high-resolution drone photo-interpretation) is prohibitively expensive.

This framework solves two interrelated challenges:
1. **Spatio-Temporal Correlation**: Environmental changes are neither purely spatial nor purely temporal. Spatial autocorrelation (Tobler's First Law of Geography) governs neighboring terrain response, while temporal phenology distinguishes natural seasonal cycles from sudden catastrophic disturbances.
2. **Annotation Scarcity**: Active Learning prioritizes satellite observation zones with maximal model uncertainty (via Shannon Entropy, Margin Sampling, or Least Confidence) for domain expert review.

```
+--------------------------------------------------------------------------+
|                         Sentinel-2 Multi-Temporal Data Cube              |
|        [N regions x T timestamps x F bands] (B2, B3, B4, B8 + NDVI/NDWI) |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                  Spatio-Temporal Graph Representation                    |
|        - Spatial Graph A_spatial: RBF kernel over geographic coordinates |
|        - Temporal Graph A_temporal: Inter-month sequential transition    |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                        Spatio-Temporal GNN (ST-GNN)                      |
|        - Spatial Conv: Normalized Message Passing (GCN / ChebConv)       |
|        - Temporal Mechanism: Bidirectional GRU / Temporal 1D Conv        |
|        - Output: Node Change Probability P(Change | Region_i)            |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                     Human-in-the-Loop Active Learner                     |
|        - Query Engine: Shannon Entropy / Margin / Least Confidence       |
|        - Select Top-K highest uncertainty geographic nodes for review    |
|        - Oracle Simulation & Feedback Loop into Labeled Pool             |
+--------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```tree
satellite-change-detection/
├── README.md                     # High-level architecture and technical guide
├── requirements.txt              # Pinned core dependencies
├── main.py                       # End-to-end benchmark & demo runner
├── data/
│   ├── __init__.py
│   └── generate_mock_sentinel.py # Synthetic GIS Sentinel-2 time series generator
├── models/
│   ├── __init__.py
│   └── st_gnn.py                 # Spatio-Temporal Graph Neural Network (PyTorch / PyG)
├── pipeline/
│   ├── __init__.py
│   └── active_learner.py         # Active Learning uncertainty sampling strategies
└── app/
    ├── __init__.py
    └── visualizer.py             # Spatio-temporal analysis & uncertainty visualizer
```

---

## 🔬 Mathematical Details

### 1. Spatial Graph Construction
Given spatial centroids $\mathbf{p}_i = (x_i, y_i) \in \mathbb{R}^2$ for region $i$:
$$A_{\text{spatial}}(i, j) = \exp\left( - \frac{\|\mathbf{p}_i - \mathbf{p}_j\|^2}{2 \sigma_s^2} \right) \cdot \mathbb{I}(\|\mathbf{p}_i - \mathbf{p}_j\| < \tau)$$

Normalized spatial graph Laplacian:
$$\widetilde{A}_{\text{spatial}} = \widetilde{D}^{-1/2} (A_{\text{spatial}} + I_N) \widetilde{D}^{-1/2}$$

### 2. Spatio-Temporal GNN Forward Pass
For node features $X \in \mathbb{R}^{N \times T \times F}$:
1. **Spatial Aggregation (per timestamp $t$):**
   $$H_t = \text{ReLU}\left( \widetilde{A}_{\text{spatial}} X_t W_s + b_s \right)$$
2. **Temporal Modeling:**
   $$Z = \text{GRU}(H_1, H_2, \dots, H_T) \quad \text{or} \quad Z = \text{Conv1D}(H)$$
3. **Change Logits:**
   $$\hat{Y} = \text{Softmax}\left( \text{MLP}(Z) \right)$$

### 3. Active Learning Query Strategies
- **Shannon Entropy**:
  $$\mathcal{U}_{\text{entropy}}(i) = - \sum_{c=1}^C P(y_i = c \mid X, A) \log P(y_i = c \mid X, A)$$
- **Margin Sampling**:
  $$\mathcal{U}_{\text{margin}}(i) = 1 - \left( P(y_i = \hat{c}_1 \mid X, A) - P(y_i = \hat{c}_2 \mid X, A) \right)$$
- **Least Confidence**:
  $$\mathcal{U}_{\text{conf}}(i) = 1 - \max_{c} P(y_i = c \mid X, A)$$

---

## 🚀 Quickstart

### 1. Prerequisites
Python 3.10 or higher. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Executes synthetic spatio-temporal data generation, initial model training (5 epochs), active learning uncertainty query, annotation loop, and post-active learning evaluation:
```bash
python main.py
```

Optional arguments:
```bash
python main.py --num_nodes 10 --num_timestamps 12 --epochs 5 --al_budget 2 --strategy entropy
```

### 3. Run the Interactive Visualizer
```bash
python -m app.visualizer
```
This renders spectral band trajectories, spatial graph topology, and uncertainty distributions directly to your terminal.

---

## 📊 Sample Output Metrics

The pipeline measures classification performance across rounds:
- **Macro F1-Score**: Harmonic mean of precision and recall across all classes
- **Overall Accuracy**: Fraction of correctly categorized spatial regions
- **Precision / Recall**: Detected true environmental change events vs false alarms
- **Uncertainty Reductions**: Tracking entropy drop across active learning iterations
