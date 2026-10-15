# User Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installing Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/satellite-change-detection.git
   cd satellite-change-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install additional development dependencies (for testing):
   ```bash
   pip install pytest
   ```

## Usage

### Running the Main Pipeline

The main entry point for the pipeline is `main.py`. You can run it with default parameters:

```bash
python main.py
```

Or with custom parameters:

```bash
python main.py --num_nodes 10 --epochs 5 --al_budget 3 --strategy entropy --seed 42
```

### Command-Line Arguments

- `--num_nodes`: Number of spatial regions (default: 8)
- `--num_timestamps`: Number of temporal steps (default: 12)
- `--num_change_regions`: Number of change regions (default: 3)
- `--epochs`: Number of training epochs (default: 5)
- `--al_budget`: Active learning query budget (default: 3)
- `--strategy`: Active learning strategy (entropy, margin, least_confidence) (default: entropy)
- `--lr`: Learning rate (default: 0.01)
- `--seed`: Random seed (default: 42)

### Running Tests

To run all tests:

```bash
python -m pytest tests/ -v
```

To run a specific test file:

```bash
python -m pytest tests/test_data_generation.py -v
```

### Generating Visualizations

To generate visualizations of the synthetic data:

```bash
python -m app.visualizer
```

This will create:
- `spatio_temporal_cube.png`: Spectral time series for a sample node
- `spatial_graph.png`: Spatial graph topology
- A textual change detection report in the console

## Customization

### Adding New Active Learning Strategies

New active learning strategies can be added to the `calculate_uncertainty` function in `pipeline/active_learner.py`.

### Modifying the GNN Architecture

The GNN architecture can be modified in `models/st_gnn.py`. You can change:
- Number of layers
- Hidden dimensions
- Temporal modeling approach (GRU vs Conv1D)
- Dropout rates
- Activation functions

### Changing Data Generation Parameters

The synthetic data generation parameters can be adjusted in `data/generate_mock_sentinel.py`. You can modify:
- Number of spectral bands
- Seasonal phenology patterns
- Change event types and characteristics
- Noise levels
- Spatial distribution of regions

## Examples

### Basic Usage

```bash
# Run with default parameters
python main.py

# Run with more nodes and epochs
python main.py --num_nodes 15 --epochs 10

# Use margin sampling for active learning
python main.py --strategy margin --al_budget 5
```

### Testing Different Strategies

```bash
# Test entropy-based active learning
python main.py --strategy entropy

# Test margin-based active learning
python main.py --strategy margin

# Test least confidence-based active learning
python main.py --strategy least_confidence
```

### Generating Visualizations

```bash
# Generate all visualizations and report
python -m app.visualizer
```

## Docker

You can also run the pipeline using Docker:

1. Build the Docker image:
   ```bash
   docker build -t satcd .
   ```

2. Run the container:
   ```bash
   docker run satcd
   ```

3. To run with custom parameters:
   ```bash
   docker run satcd python main.py --num_nodes 10 --epochs 5
   ```

## Troubleshooting

### Import Errors

If you encounter import errors, make sure you're running Python from the root directory of the project, or that the project directory is in your PYTHONPATH.

### PyTorch Installation Issues

If you have trouble installing PyTorch, visit [pytorch.org](https://pytorch.org/) and follow the installation instructions for your specific OS and CUDA setup.

### PyTorch Geometric Installation Issues

If you have trouble installing PyTorch Geometric, visit [pyg.org](https://pyg.org/) and follow the installation instructions for your specific PyTorch version.

### Performance Issues

For better performance, especially with larger datasets:
- Use a GPU if available
- Increase the `--epochs` parameter for better model training
- Adjust the model hyperparameters in `models/st_gnn.py`
