"""Active Learning Strategies for Spatio-Temporal Environmental Change Detection.

Implements entropy-based, margin-based, and least-confidence uncertainty sampling
to identify the most informative nodes for expert annotation in satellite monitoring.
"""

import os
import sys
sys.path.append(os.getcwd())

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np


def calculate_uncertainty(
    probs: torch.Tensor,
    strategy: str = "entropy",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Computes node-level uncertainty scores from predicted class probabilities.

    Args:
        probs: Softmax class probabilities [N, C].
        strategy: Sampling method ("entropy", "margin", "least_confidence").
        eps: Numerical stability epsilon.

    Returns:
        uncertainty_scores: Per-node scalar uncertainty [N].
    """
    if strategy == "entropy":
        # Shannon Entropy: H(p) = -Σ p_i * log(p_i)
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=1)  # [N]
        return entropy

    elif strategy == "margin":
        # Margin Sampling: 1 - (p_max - p_second_max)
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]  # [N]
        return 1.0 - margins

    elif strategy == "least_confidence":
        # Least Confidence: 1 - max(p_i)
        max_probs, _ = torch.max(probs, dim=1)  # [N]
        return 1.0 - max_probs

    else:
        raise ValueError(f"Unsupported uncertainty strategy: {strategy}")


class ActiveLearner:
    """Active Learning Module for Environmental Change Detection.

    Selects top-K uncertain nodes for human annotation based on model predictions.
    """

    def __init__(
        self,
        strategy: str = "entropy",
        device: Union[str, torch.device] = "cpu",
    ):
        self.strategy = strategy
        self.device = device

    def query(
        self,
        model: torch.nn.Module,
        data: "SpatioTemporalData",  # Forward reference
        k: int = 3,
        return_scores: bool = False,
    ) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
        """Queries top-K nodes with highest prediction uncertainty.

        Args:
            model: Trained ST-GNN model.
            data: SpatioTemporalData container with features and graph structure.
            k: Number of nodes to query for annotation.
            return_scores: If True, return uncertainty scores alongside indices.

        Returns:
            queried_indices: List of node indices selected for annotation.
            (optional) uncertainty_scores: Tensor of uncertainty values [N].
        """
        model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            out = model(
                x=data.x,
                A_spatial=data.A_spatial,
                edge_index=data.edge_index_spatial,
                edge_weight=data.edge_weight_spatial,
            )
            probs = out["probs"]  # [N, C]

        uncertainty_scores = calculate_uncertainty(probs, strategy=self.strategy)
        top_k_scores, top_k_indices = torch.topk(uncertainty_scores, k=min(k, probs.size(0)))

        queried_indices = top_k_indices.cpu().tolist()

        if return_scores:
            return queried_indices, uncertainty_scores
        return queried_indices


# Example usage
if __name__ == "__main__":
    from data.generate_mock_sentinel import generate_mock_sentinel_dataset
    from models.st_gnn import SpatioTemporalGNN

    print("Testing Active Learning Pipeline...")

    # Generate mock data
    dataset = generate_mock_sentinel_dataset(num_nodes=10, num_timestamps=12, num_change_regions=3)

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
    queried_nodes = al.query(model, dataset, k=3)
    print(f"Top 3 queried nodes for annotation: {queried_nodes}")

    # With scores
    queried_nodes_with_scores, scores = al.query(model, dataset, k=3, return_scores=True)
    print(f"Queried nodes with scores: {list(zip(queried_nodes_with_scores, scores.cpu().numpy()))}")

    print("Active Learning pipeline test passed!")
