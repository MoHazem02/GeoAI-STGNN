# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-15

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

### Features
- Spatio-Temporal GNN with configurable backbones (GRU, Conv1D)
- Spatial graph convolution with PyG and dense matrix support
- Temporal attention mechanism for explainable change detection
- Three active learning strategies: Entropy, Margin, Least Confidence
- Synthetic data generation with realistic phenological patterns
- Interactive visualization of spatio-temporal data cubes
- Spatial graph topology visualization
- Command-line interface for pipeline execution
- Configurable parameters for all components
- Comprehensive metrics reporting (Accuracy, F1-Score)

### Documentation
- Detailed README with problem formulation and architecture
- API documentation for all modules
- User guide with installation and usage instructions
- Architecture documentation with mathematical details
- Contributing guidelines
- License information

### Technical
- Modular design for easy extensibility
- Type hints throughout the codebase
- Comprehensive error handling
- Clean, well-documented code
- Dockerfile for containerization
- GitHub Actions CI/CD workflow
- Requirements file with pinned dependencies
- Setup script for package installation

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- NetworkX
- Scikit-learn
- NumPy
- Matplotlib
- Plotly
- Streamlit
- PyTest (for testing)
