"""Tests for Active Learning module."""

import sys
import os
import torch
import numpy as np

# Add current directory to path for local imports
sys.path.append(os.getcwd())

from pipeline.active_learner import calculate_uncertainty, ActiveLearner
from data.generate_mock_sentinel import generate_mock_sentinel_dataset
from models.st_gnn import SpatioTemporalGNN


def test_calculate_uncertainty():
    """Test uncertainty calculation functions."""
    # Create mock probabilities
    probs = torch.tensor([
        [0.9, 0.1],  # High confidence class 0
        [0.5, 0.5],  # Maximum uncertainty
        [0.1, 0.9],  # High confidence class 1
    ], dtype=torch.float32)
    
    # Test entropy
    entropy = calculate_uncertainty(probs, strategy="entropy")
    expected_entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
    assert torch.allclose(entropy, expected_entropy, atol=1e-6)
    
    # Test margin
    margin = calculate_uncertainty(probs, strategy="margin")
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    expected_margin = 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])
    assert torch.allclose(margin, expected_margin, atol=1e-6)
    
    # Test least confidence
    least_conf = calculate_uncertainty(probs, strategy="least_confidence")
    max_probs, _ = torch.max(probs, dim=1)
    expected_least_conf = 1.0 - max_probs
    assert torch.allclose(least_conf, expected_least_conf, atol=1e-6)
    
    print("test_calculate_uncertainty passed!")


def test_active_learner_query():
    """Test ActiveLearner query functionality."""
    # Generate mock data
    data = generate_mock_sentinel_dataset(num_nodes=8, num_timestamps=12, num_change_regions=3)
    
    # Initialize model
    model = SpatioTemporalGNN(
        num_features=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=32,
        num_classes=2,
        dropout=0.1,
        temporal_backbone="gru",
        use_pyg=False,
    )
    
    # Initialize active learner
    al = ActiveLearner(strategy="entropy")
    
    # Query uncertain nodes
    queried_nodes = al.query(model, data, k=3)
    assert len(queried_nodes) == 3
    assert all(isinstance(node, int) for node in queried_nodes)
    assert all(0 <= node < 8 for node in queried_nodes)
    
    # With scores
    queried_nodes_with_scores, scores = al.query(model, data, k=3, return_scores=True)
    assert len(queried_nodes_with_scores) == 3
    assert scores.shape == (8,)
    assert torch.all(scores >= 0)
    
    print("test_active_learner_query passed!")


def test_all_strategies():
    """Test all active learning strategies."""
    # Generate mock data
    data = generate_mock_sentinel_dataset(num_nodes=8, num_timestamps=12, num_change_regions=3)
    
    # Initialize model
    model = SpatioTemporalGNN(
        num_features=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=32,
        num_classes=2,
        dropout=0.1,
        temporal_backbone="gru",
        use_pyg=False,
    )
    
    strategies = ["entropy", "margin", "least_confidence"]
    
    for strategy in strategies:
        al = ActiveLearner(strategy=strategy)
        queried_nodes = al.query(model, data, k=3)
        assert len(queried_nodes) == 3
        
    print("test_all_strategies passed!")


if __name__ == "__main__":
    test_calculate_uncertainty()
    test_active_learner_query()
    test_all_strategies()
    print("All Active Learning tests passed!")
