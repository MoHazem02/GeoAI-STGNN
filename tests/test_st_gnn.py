"""Tests for Spatio-Temporal GNN model."""

import sys
import os
import torch
import torch.nn as nn

# Add current directory to path for local imports
sys.path.append(os.getcwd())

from models.st_gnn import SpatioTemporalGNN, SpatialGraphConv


def test_spatial_graph_conv():
    """Test SpatialGraphConv layer."""
    # Test with PyG
    try:
        from torch_geometric.nn import GCNConv
        pyg_available = True
    except ImportError:
        pyg_available = False
    
    conv = SpatialGraphConv(4, 32, use_pyg_if_available=pyg_available)
    
    # Test with edge_index (only if PyG is available)
    if pyg_available:
        x = torch.randn(8, 4)
        edge_index = torch.randint(0, 8, (2, 20))
        edge_weight = torch.rand(20)
        
        out = conv(x, edge_index=edge_index, edge_weight=edge_weight)
        assert out.shape == (8, 32)
    
    # Test with dense adjacency (always works)
    conv_no_pyg = SpatialGraphConv(4, 32, use_pyg_if_available=False)
    x = torch.randn(8, 4)
    adj = torch.rand(8, 8)
    adj = (adj + adj.t()) / 2.0  # Symmetrize
    out = conv_no_pyg(x, adj=adj)
    assert out.shape == (8, 32)
    
    print("test_spatial_graph_conv passed!")


def test_spatio_temporal_gnn():
    """Test SpatioTemporalGNN model."""
    model = SpatioTemporalGNN(
        num_features=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=32,
        num_classes=2,
        dropout=0.2,
        temporal_backbone="gru",
        use_pyg=False,
    )
    
    # Test forward pass
    x = torch.randn(8, 12, 4)
    adj = torch.rand(8, 8)
    adj = (adj + adj.t()) / 2.0  # Symmetrize
    
    out = model(x, A_spatial=adj)
    
    # Check output keys
    assert "logits" in out
    assert "probs" in out
    assert "temporal_attention" in out
    assert "node_embeddings" in out
    
    # Check output shapes
    assert out["logits"].shape == (8, 2)
    assert out["probs"].shape == (8, 2)
    assert out["temporal_attention"].shape == (8, 12)
    assert out["node_embeddings"].shape == (8, 64)  # 32 * 2 for bidirectional GRU
    
    # Check probability sum to 1
    assert torch.allclose(torch.sum(out["probs"], dim=1), torch.ones(8), atol=1e-6)
    
    print("test_spatio_temporal_gnn passed!")


def test_spatio_temporal_gnn_conv1d():
    """Test SpatioTemporalGNN model with Conv1D temporal backbone."""
    model = SpatioTemporalGNN(
        num_features=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=32,
        num_classes=2,
        dropout=0.2,
        temporal_backbone="conv1d",
        use_pyg=False,
    )
    
    # Test forward pass
    x = torch.randn(8, 12, 4)
    adj = torch.rand(8, 8)
    adj = (adj + adj.t()) / 2.0  # Symmetrize
    
    out = model(x, A_spatial=adj)
    
    # Check output keys
    assert "logits" in out
    assert "probs" in out
    assert "temporal_attention" in out
    assert "node_embeddings" in out
    
    # Check output shapes
    assert out["logits"].shape == (8, 2)
    assert out["probs"].shape == (8, 2)
    assert out["temporal_attention"].shape == (8, 12)
    assert out["node_embeddings"].shape == (8, 64)  # 32 * 2 for Conv1D with same padding
    
    # Check probability sum to 1
    assert torch.allclose(torch.sum(out["probs"], dim=1), torch.ones(8), atol=1e-6)
    
    print("test_spatio_temporal_gnn_conv1d passed!")


if __name__ == "__main__":
    test_spatial_graph_conv()
    test_spatio_temporal_gnn()
    test_spatio_temporal_gnn_conv1d()
    print("All ST-GNN tests passed!")
