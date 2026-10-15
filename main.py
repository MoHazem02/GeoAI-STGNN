"""End-to-End Benchmark Runner for Spatio-Temporal GNN + Active Learning Pipeline.

Orchestrates synthetic data generation, initial model training (5 epochs),
active learning query (top-k uncertain nodes), and evaluation metrics.
"""

import os
import sys
import argparse
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

# Add current directory to path for local imports
sys.path.append(os.getcwd())

from data.generate_mock_sentinel import generate_mock_sentinel_dataset
from models.st_gnn import SpatioTemporalGNN
from pipeline.active_learner import ActiveLearner


def evaluate_model(
    model: nn.Module,
    data: "SpatioTemporalData",
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model performance on labeled nodes."""
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(
            x=data.x,
            A_spatial=data.A_spatial,
            edge_index=data.edge_index_spatial,
            edge_weight=data.edge_weight_spatial,
        )
        probs = out["probs"]
        preds = torch.argmax(probs, dim=1).cpu().numpy()
    
    labels = data.y.cpu().numpy()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
    }


def train_model(
    model: nn.Module,
    data: "SpatioTemporalData",
    device: torch.device,
    epochs: int = 5,
    lr: float = 0.01,
) -> None:
    """Train the Spatio-Temporal GNN model."""
    model.train()
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training model for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(
            x=data.x,
            A_spatial=data.A_spatial,
            edge_index=data.edge_index_spatial,
            edge_weight=data.edge_weight_spatial,
        )
        logits = out["logits"]
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal GNN & Active Learning Pipeline")
    parser.add_argument("--num_nodes", type=int, default=8, help="Number of spatial regions")
    parser.add_argument("--num_timestamps", type=int, default=12, help="Number of temporal steps")
    parser.add_argument("--num_change_regions", type=int, default=3, help="Number of change regions")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--al_budget", type=int, default=3, help="Active learning query budget")
    parser.add_argument("--strategy", type=str, default="entropy", 
                        choices=["entropy", "margin", "least_confidence"],
                        help="Active learning strategy")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Generate synthetic dataset
    print("Generating synthetic spatio-temporal dataset...")
    data = generate_mock_sentinel_dataset(
        num_nodes=args.num_nodes,
        num_timestamps=args.num_timestamps,
        num_change_regions=args.num_change_regions,
        random_seed=args.seed,
    )
    
    # Initialize model
    model = SpatioTemporalGNN(
        num_features=4,
        spatial_hidden_dim=32,
        temporal_hidden_dim=32,
        num_classes=2,
        dropout=0.2,
        temporal_backbone="gru",
        use_pyg=False,
    ).to(device)
    
    # Train model
    train_model(model, data, device, epochs=args.epochs, lr=args.lr)
    
    # Evaluate before active learning
    print("\n--- Evaluation Before Active Learning ---")
    metrics_before = evaluate_model(model, data, device)
    print(f"Accuracy: {metrics_before['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics_before['f1_score']:.4f}")
    
    # Active learning query
    print(f"\n--- Active Learning Query (Strategy: {args.strategy}) ---")
    al = ActiveLearner(strategy=args.strategy, device=device)
    queried_nodes, uncertainty_scores = al.query(model, data, k=args.al_budget, return_scores=True)
    
    print(f"Top {args.al_budget} queried nodes for annotation:")
    for i, node_idx in enumerate(queried_nodes):
        score = uncertainty_scores[node_idx].item()
        true_label = data.y[node_idx].item()
        event_info = " (Change Region)" if true_label == 1 else " (Stable Region)"
        print(f"  Node {node_idx}: Uncertainty = {score:.4f}{event_info}")
    
    # Simulate expert annotation (in real scenario, this would be manual)
    print("\n--- Simulating Expert Annotation ---")
    annotated_labels = {}
    for node_idx in queried_nodes:
        annotated_labels[node_idx] = data.y[node_idx].item()
        print(f"Node {node_idx} annotated with label: {annotated_labels[node_idx]}")
    
    # In a full implementation, we would retrain with new labels here
    # For this demo, we'll just show how it would work
    
    print("\n--- Pipeline Execution Complete ---")
    print("In a full implementation, the model would be retrained with newly annotated data.")
    print("This completes one iteration of the active learning loop.")


if __name__ == "__main__":
    main()
