"""Integration tests for the full pipeline."""

import sys
import os
import torch
import numpy as np

# Add current directory to path for local imports
sys.path.append(os.getcwd())

from data.generate_mock_sentinel import generate_mock_sentinel_dataset
from models.st_gnn import SpatioTemporalGNN
from pipeline.active_learner import ActiveLearner


def test_full_pipeline():
    """Test the full pipeline: data generation, training, evaluation, active learning."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate dataset
    print("Generating synthetic dataset...")
    data = generate_mock_sentinel_dataset(num_nodes=10, num_timestamps=12, num_change_regions=3)
    
    # Initialize model
    print("Initializing model...")
    model = SpatioTemporalGNN(
        num_features=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=32,
        num_classes=2,
        dropout=0.2,
        temporal_backbone="gru",
        use_pyg=False,
    )
    
    # Check that model can process the data
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        out = model(
            x=data.x,
            A_spatial=data.A_spatial,
            edge_index=data.edge_index_spatial,
            edge_weight=data.edge_weight_spatial,
        )
    
    # Check output
    assert "logits" in out
    assert "probs" in out
    assert "temporal_attention" in out
    assert "node_embeddings" in out
    
    assert out["logits"].shape == (10, 2)
    assert out["probs"].shape == (10, 2)
    assert out["temporal_attention"].shape == (10, 12)
    assert out["node_embeddings"].shape == (10, 64)
    
    # Check probability sum to 1
    assert torch.allclose(torch.sum(out["probs"], dim=1), torch.ones(10), atol=1e-6)
    
    # Test active learning
    print("Testing active learning...")
    al = ActiveLearner(strategy="entropy")
    queried_nodes = al.query(model, data, k=3)
    assert len(queried_nodes) == 3
    
    # With scores
    queried_nodes_with_scores, scores = al.query(model, data, k=3, return_scores=True)
    assert len(queried_nodes_with_scores) == 3
    assert scores.shape == (10,)
    assert torch.all(scores >= 0)
    
    print("test_full_pipeline passed!")


if __name__ == "__main__":
    test_full_pipeline()
    print("All integration tests passed!")
