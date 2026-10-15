"""Tests for synthetic data generation."""

import sys
import os
import torch
import numpy as np

# Add current directory to path for local imports
sys.path.append(os.getcwd())

from data.generate_mock_sentinel import generate_mock_sentinel_dataset, SpatioTemporalData


def test_generate_mock_sentinel_dataset():
    """Test synthetic data generation."""
    # Generate dataset
    data = generate_mock_sentinel_dataset(num_nodes=8, num_timestamps=12, num_change_regions=3)
    
    # Check types
    assert isinstance(data, SpatioTemporalData)
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)
    assert isinstance(data.A_spatial, torch.Tensor)
    assert isinstance(data.A_temporal, torch.Tensor)
    assert isinstance(data.edge_index_spatial, torch.Tensor)
    assert isinstance(data.edge_weight_spatial, torch.Tensor)
    assert isinstance(data.coordinates, np.ndarray)
    assert isinstance(data.timestamps, list)
    assert isinstance(data.band_names, list)
    assert isinstance(data.metadata, dict)
    
    # Check shapes
    assert data.x.shape == (8, 12, 4)  # N, T, F
    assert data.y.shape == (8,)        # N
    assert data.A_spatial.shape == (8, 8)   # N, N
    assert data.A_temporal.shape == (12, 12) # T, T
    assert data.edge_index_spatial.shape[0] == 2  # 2, E
    assert data.edge_weight_spatial.shape[0] == data.edge_index_spatial.shape[1]  # E
    assert data.coordinates.shape == (8, 2)  # N, 2
    assert len(data.timestamps) == 12
    assert len(data.band_names) == 4
    
    # Check values
    assert torch.all(data.x >= 0.01)  # Minimum reflectance
    assert torch.all(data.x <= 0.95)  # Maximum reflectance
    assert torch.all(data.y >= 0)     # Labels non-negative
    assert torch.all(data.y <= 1)     # Binary labels
    assert torch.all(data.A_spatial >= 0)  # Non-negative adjacency
    assert torch.all(data.A_temporal >= 0) # Non-negative adjacency
    assert torch.all(data.edge_weight_spatial >= 0)  # Non-negative edge weights
    
    # Check metadata
    assert "num_nodes" in data.metadata
    assert "num_timestamps" in data.metadata
    assert "num_bands" in data.metadata
    assert "change_regions" in data.metadata
    assert "change_events" in data.metadata
    assert "crs_simulated" in data.metadata
    
    print("test_generate_mock_sentinel_dataset passed!")


def test_spatial_graph_properties():
    """Test spatial graph properties."""
    data = generate_mock_sentinel_dataset(num_nodes=8, num_timestamps=12, num_change_regions=3)
    
    # Check that diagonal elements are non-zero (self-loops)
    assert torch.all(torch.diag(data.A_spatial) > 0)
    
    # Check symmetry (since we use RBF kernel)
    assert torch.allclose(data.A_spatial, data.A_spatial.t())
    
    # Check that edge weights match adjacency matrix
    edge_index = data.edge_index_spatial
    edge_weight = data.edge_weight_spatial
    A_spatial = data.A_spatial
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        weight = edge_weight[i].item()
        assert abs(A_spatial[src, dst].item() - weight) < 1e-6
    
    print("test_spatial_graph_properties passed!")


def test_temporal_graph_properties():
    """Test temporal graph properties."""
    data = generate_mock_sentinel_dataset(num_nodes=8, num_timestamps=12, num_change_regions=3)
    
    # Check that diagonal elements are 1.0
    assert torch.all(torch.diag(data.A_temporal) == 1.0)
    
    # Check symmetry
    assert torch.allclose(data.A_temporal, data.A_temporal.t())
    
    # Check that adjacent months have higher weights than distant ones
    # (this is a property of our temporal graph construction)
    # Fix the index error by ensuring we don't go out of bounds
    for t in range(data.A_temporal.shape[0] - 2):  # -2 to ensure t+2 is within bounds
        assert data.A_temporal[t, t+1] > data.A_temporal[t, t+2]
    
    print("test_temporal_graph_properties passed!")


if __name__ == "__main__":
    test_generate_mock_sentinel_dataset()
    test_spatial_graph_properties()
    test_temporal_graph_properties()
    print("All data generation tests passed!")
