# API Documentation

## Data Module

### generate_mock_sentinel.py

#### SpatioTemporalData
```python
class SpatioTemporalData:
    x: torch.Tensor
    y: torch.Tensor
    A_spatial: torch.Tensor
    A_temporal: torch.Tensor
    edge_index_spatial: torch.Tensor
    edge_weight_spatial: torch.Tensor
    coordinates: np.ndarray
    timestamps: List[str]
    band_names: List[str]
    metadata: Dict[str, Union[str, int, float, list]]
```

Container for spatio-temporal satellite data.

#### generate_mock_sentinel_dataset
```python
def generate_mock_sentinel_dataset(
    num_nodes: int = 8,
    num_timestamps: int = 12,
    num_change_regions: int = 3,
    noise_std: float = 0.02,
    random_seed: int = 42,
) -> SpatioTemporalData
```

Generate a synthetic spatio-temporal Sentinel-2-like dataset.

## Models Module

### st_gnn.py

#### SpatialGraphConv
```python
class SpatialGraphConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_pyg_if_available: bool = True,
    )
```

Normalized spatial graph convolution layer.

#### SpatioTemporalGNN
```python
class SpatioTemporalGNN(nn.Module):
    def __init__(
        self,
        num_features: int = 4,
        spatial_hidden_dim: int = 32,
        temporal_hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.2,
        temporal_backbone: str = "gru",
        use_pyg: bool = False,
    )
```

Spatio-temporal graph neural network for change detection.

## Pipeline Module

### active_learner.py

#### calculate_uncertainty
```python
def calculate_uncertainty(
    probs: torch.Tensor,
    strategy: str = "entropy",
    eps: float = 1e-7,
) -> torch.Tensor
```

Calculate node-level uncertainty scores.

#### ActiveLearner
```python
class ActiveLearner:
    def __init__(
        self,
        strategy: str = "entropy",
        device: Union[str, torch.device] = "cpu",
    )
    
    def query(
        self,
        model: torch.nn.Module,
        data: "SpatioTemporalData",
        k: int = 3,
        return_scores: bool = False,
    ) -> Union[List[int], Tuple[List[int], torch.Tensor]]
```

Active learning module for querying uncertain nodes.

## Main Module

### main.py

#### evaluate_model
```python
def evaluate_model(
    model: nn.Module,
    data: "SpatioTemporalData",
    device: torch.device,
) -> Dict[str, float]
```

Evaluate model performance.

#### train_model
```python
def train_model(
    model: nn.Module,
    data: "SpatioTemporalData",
    device: torch.device,
    epochs: int = 5,
    lr: float = 0.01,
) -> None
```

Train the model.

#### main
```python
def main() -> None
```

Main entry point for the pipeline.

## App Module

### visualizer.py

#### visualize_spatio_temporal_cube
```python
def visualize_spatio_temporal_cube(
    data: "SpatioTemporalData",
    node_idx: int = 0,
    save_path: str = "spatio_temporal_cube.png",
) -> None
```

Visualize multi-spectral time series for a spatial node.

#### visualize_spatial_graph
```python
def visualize_spatial_graph(
    data: "SpatioTemporalData",
    save_path: str = "spatial_graph.png",
) -> None
```

Visualize spatial graph topology.

#### print_change_report
```python
def print_change_report(data: "SpatioTemporalData") -> None
```

Print a textual summary of change events.
