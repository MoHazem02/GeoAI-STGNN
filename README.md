# 🛰️ GeoAI-STGNN: Spatio-Temporal GNN & Active Learning for Satellite Change Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Compatible-green.svg)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade GeoAI framework for detecting environmental anomalies (wildfires, deforestation, floods) using multi-spectral satellite time-series. Combines **Spatio-Temporal Graph Neural Networks (ST-GNN)** with **Human-in-the-Loop Active Learning** to minimize manual GIS annotation overhead while maximizing detection accuracy.

---

## [Unreleased]

### Added
- Initial release of GeoAI-STGNN framework
- Spatio-Temporal Graph Neural Network implementation
- Synthetic data generator for Sentinel-2-like satellite data
- Active Learning module with multiple uncertainty sampling strategies
- Interactive Streamlit web application
- Comprehensive test suite with pytest
- Detailed documentation including architecture, API, and user guide
- Docker support for containerization
- CI/CD configuration with GitHub Actions
- Setup script for package installation

## 🌍 Problem Formulation & Agentic GeoAI Context

In Earth Observation (EO), satellite constellations like **Sentinel-2 (MSI)** revisit Earth locations at high cadence (5-day revisit), yielding multi-spectral spatio-temporal data cubes. However, ground-truth change validation (e.g., post-disaster field inspection or high-resolution drone photo-interpretation) is prohibitively expensive.

This framework solves two interrelated challenges through an **Agentic GeoAI** approach:
1. **Spatio-Temporal Correlation**: Environmental changes are neither purely spatial nor purely temporal. Spatial autocorrelation (Tobler's First Law of Geography) governs neighboring terrain response, while temporal phenology distinguishes natural seasonal cycles from sudden catastrophic disturbances.
2. **Annotation Scarcity**: Active Learning prioritizes satellite observation zones with maximal model uncertainty (via Shannon Entropy, Margin Sampling, or Least Confidence) for domain expert review.

```mermaid
graph TD
    A[Sentinel-2 Multi-Temporal Data Cube] --> B[Spatio-Temporal Graph Representation]
    B --> C[Spatio-Temporal GNN]
    C --> D[Change Probability Map]
    D --> E[Active Learning Uncertainty Sampling]
    E --> F[Human-in-the-Loop Annotation]
    F --> G[Model Retraining]
    G --> C
```

---

## 🏗️ Repository Structure

```tree
GeoAI-STGNN/
├── README.md                     # High-level architecture and technical guide
├── requirements.txt              # Pinned core dependencies
├── main.py                       # End-to-end benchmark & demo runner
├── app/
│   ├── __init__.py
│   ├── main_app.py               # Interactive Streamlit web application
│   └── visualizer.py             # Spatio-temporal analysis & uncertainty visualizer
├── data/
│   ├── __init__.py
│   └── generate_mock_sentinel.py # Synthetic GIS Sentinel-2 time series generator
├── models/
│   ├── __init__.py
│   └── st_gnn.py                 # Spatio-Temporal Graph Neural Network (PyTorch / PyG)
├── pipeline/
│   ├── __init__.py
│   └── active_learner.py         # Active Learning uncertainty sampling strategies
├── tests/
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_st_gnn.py
│   ├── test_active_learner.py
│   └── test_integration.py
└── docs/
    ├── architecture.md
    ├── api.md
    └── user_guide.md
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
Python 3.8 or higher. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit App
Launch the web-based GUI for interactive experimentation:
```bash
streamlit run app/main_app.py
```

The app provides:
- Configurable dataset parameters (node count, temporal window)
- Model hyperparameters (epochs, learning rate)
- Active learning controls (strategy, budget)
- Interactive spatial graph visualization
- Real-time metrics dashboard
- Active learning panel with uncertainty sampling

### 3. Run the End-to-End Pipeline
Executes synthetic spatio-temporal data generation, initial model training (5 epochs), active learning selection, and evaluation:
```bash
python main.py
```

Optional arguments:
```bash
python main.py --num_nodes 10 --num_timestamps 12 --epochs 5 --al_budget 2 --strategy entropy
```

### 4. Run Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 📊 Sample Output Metrics

The pipeline measures classification performance across rounds:
- **Macro F1-Score**: Harmonic mean of precision and recall across all classes
- **Overall Accuracy**: Fraction of correctly categorized spatial regions
- **Precision / Recall**: Detected true environmental change events vs false alarms
- **Uncertainty Reductions**: Tracking entropy drop across active learning iterations

---

## 🧠 Architecture Overview

### Data Generation
The `data/generate_mock_sentinel.py` module creates synthetic spatio-temporal datasets that mimic real satellite observations:
- Spatial regions distributed in a geographic area with coordinates
- Monthly temporal resolution with seasonal phenology patterns
- Four spectral bands (Blue, Green, Red, NIR) plus derived indices
- Spatial adjacency matrix computed using RBF kernel over distances
- Temporal adjacency matrix representing sequential month relationships
- Ground truth labels indicating stable vs. changed regions

### Models
The `models/st_gnn.py` module implements the core neural network architectures:
- **SpatialGraphConv**: A spatial graph convolution layer that supports both PyG and dense matrix implementations
- **TemporalAttention**: A temporal attention mechanism to weigh the importance of different time steps
- **SpatioTemporalGNN**: The main model combining spatial and temporal processing for change detection

### Active Learning Pipeline
The `pipeline/active_learner.py` module implements active learning strategies to identify the most informative samples for labeling:
- Multiple uncertainty sampling strategies (Entropy, Margin, Least Confidence)
- Integration with the GNN model for uncertainty estimation
- Return of node indices and optionally uncertainty scores

---

## 🌟 Future Research Extensions

1. **Real Satellite Integration**:
   - Connect to Sentinel-2 STAC catalogs via `pystac-client`
   - Process real Cloud-Optimized GeoTIFFs (COGs) using `rasterio` and `xarray`
   - Implement cloud masking with `s2cloudless`

2. **Advanced GNN Architectures**:
   - Graph Attention Networks (GAT) for spatial processing
   - Transformer-based temporal modeling
   - Multi-scale spatial-temporal feature extraction

3. **Uncertainty Quantification**:
   - Bayesian Neural Networks for model uncertainty
   - Ensemble methods for prediction confidence
   - Conformal prediction intervals

4. **Multi-Modal Fusion**:
   - Incorporate SAR data (Sentinel-1) for all-weather monitoring
   - Fuse optical and radar observations
   - Cross-sensor calibration techniques

5. **Scalability Improvements**:
   - Distributed training with PyTorch Distributed
   - Graph sampling techniques for large regions
   - Model compression for edge deployment

---

## 📚 References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*.
2. Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., & Vandergheynst, P. (2017). Geometric deep learning: going beyond Euclidean data. *IEEE Signal Processing Magazine*, 34(4), 18-42.
3. Settles, B. (2009). Active learning literature survey. *University of Wisconsin-Madison Department of Computer Sciences*.
4. Dranguet, A., et al. (2022). Spatio-temporal graph neural networks for wildfire detection in Mediterranean ecosystems. *Remote Sensing of Environment*, 275, 113007.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](docs/contributing.md) for more details.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

---

## 📧 Contact

For questions or feedback, please open an issue or contact the maintainers.
