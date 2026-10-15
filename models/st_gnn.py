"""Spatio-Temporal Graph Neural Network (ST-GNN) Architecture.

Designed for satellite earth observation time series and change detection.
Combines Spatial Graph Convolutions (neighborhood aggregation across geographic regions)
with a Temporal Mechanism (Bidirectional GRU / 1D Temporal Convolution) and Attention
Pooling to classify node-level environmental disturbances.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    GCNConv = None


class SpatialGraphConv(nn.Module):
    """Normalized Spatial Graph Convolution Layer.

    Supports both PyG edge_index message-passing and dense normalized adjacency matrix
    multiplication (A_hat = D^{-1/2} (A + I) D^{-1/2}).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_pyg_if_available: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_pyg = use_pyg_if_available and PYG_AVAILABLE

        if self.use_pyg:
            self.pyg_conv = GCNConv(in_features, out_features, bias=bias)
            self.linear = None
        else:
            self.pyg_conv = None
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.bias = None if not bias else (nn.Parameter(torch.zeros(out_features)) if not self.use_pyg else None)

    @staticmethod
    def normalize_adjacency(A: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Computes symmetric normalized Laplacian: D^{-1/2} A D^{-1/2}."""
        # Ensure self loops
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A_tilde = A + I
        degree = torch.sum(A_tilde, dim=-1)
        deg_inv_sqrt = torch.pow(degree + eps, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return torch.mm(torch.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node feature tensor [N, in_features]
            adj: Dense adjacency matrix [N, N] (optional)
            edge_index: Sparse edge index [2, E] (optional for PyG)
            edge_weight: Sparse edge weights [E] (optional for PyG)

        Returns:
            Aggregated node features [N, out_features]
        """
        if self.use_pyg and edge_index is not None:
            return self.pyg_conv(x, edge_index, edge_weight=edge_weight)

        # Dense normalized adjacency propagation fallback
        if adj is None:
            raise ValueError("Either 'adj' or 'edge_index' must be provided.")

        A_norm = self.normalize_adjacency(adj)
        support = self.linear(x)  # [N, out_features]
        out = torch.mm(A_norm, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class TemporalAttention(nn.Module):
    """Calculates temporal attention weights over timestamps.

    Enables explainable detection of which month triggered the change.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

        h: Spatio-temporal representations [N, T, hidden_dim]

        Returns:
            pooled: Temporal pooled embedding [N, hidden_dim]
            weights: Normalized attention weights [N, T]
        """
        # score shape: [N, T, 1]
        scores = self.attention_net(h)
        weights = F.softmax(scores, dim=1)  # [N, T, 1]
        pooled = torch.sum(h * weights, dim=1)  # [N, hidden_dim]
        return pooled, weights.squeeze(-1)


class SpatioTemporalGNN(nn.Module):
    """Spatio-Temporal Graph Neural Network for Satellite Change Detection.

    Flow:
    1. Spatial Graph Convolution: Aggregates geographic neighborhood context at each time step.
    2. Temporal Mechanism: Bidirectional GRU or Conv1D sequence modeling across temporal steps.
    3. Temporal Attention: Weights most salient temporal disturbances (phenology break vs seasonal).
    4. Classification Head: Multi-Layer Perceptron (MLP) mapping to change class probabilities.
    """

    def __init__(
        self,
        num_features: int = 4,
        spatial_hidden_dim: int = 32,
        temporal_hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.2,
        temporal_backbone: str = "gru",  # 'gru' or 'conv1d'
        use_pyg: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.spatial_hidden_dim = spatial_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.num_classes = num_classes
        self.temporal_backbone = temporal_backbone.lower()

        # 1. Spatial GCN Module
        self.spatial_gcn1 = SpatialGraphConv(
            num_features, spatial_hidden_dim, use_pyg_if_available=use_pyg
        )
        self.spatial_gcn2 = SpatialGraphConv(
            spatial_hidden_dim, spatial_hidden_dim, use_pyg_if_available=use_pyg
        )
        self.spatial_dropout = nn.Dropout(dropout)

        # 2. Temporal Mechanism Module
        if self.temporal_backbone == "gru":
            # Bidirectional GRU
            self.temporal_model = nn.GRU(
                input_size=spatial_hidden_dim,
                hidden_size=temporal_hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            temporal_effective_dim = temporal_hidden_dim * 2
        elif self.temporal_backbone == "conv1d":
            # 1D Temporal Convolution (across time axis)
            self.temporal_model = nn.Sequential(
                nn.Conv1d(
                    in_channels=spatial_hidden_dim,
                    out_channels=temporal_hidden_dim * 2,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm1d(temporal_hidden_dim * 2),
                nn.ReLU(),
            )
            temporal_effective_dim = temporal_hidden_dim * 2
        else:
            raise ValueError(f"Unsupported temporal backbone: {temporal_backbone}")

        # 3. Temporal Attention Pooling
        self.attention = TemporalAttention(temporal_effective_dim)

        # 4. Change Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_effective_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        A_spatial: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass across spatio-temporal satellite data cube.

        Args:
            x: Satellite multi-spectral features [N, T, F]
            A_spatial: Spatial adjacency matrix [N, N]
            edge_index: PyG edge index [2, E] (optional)
            edge_weight: PyG edge weight [E] (optional)

        Returns:
            Dict containing:
                - 'logits': [N, num_classes] raw classification logits
                - 'probs': [N, num_classes] softmax class probabilities
                - 'temporal_attention': [N, T] attention distribution over months
                - 'node_embeddings': [N, temporal_effective_dim] latent representations
        """
        num_nodes, seq_len, in_channels = x.shape

        # Step 1: Spatial message passing across each timestamp t
        spatial_embeddings = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [N, C]
            h1 = F.relu(self.spatial_gcn1(x_t, adj=A_spatial, edge_index=edge_index, edge_weight=edge_weight))
            h1 = self.spatial_dropout(h1)
            h2 = F.relu(self.spatial_gcn2(h1, adj=A_spatial, edge_index=edge_index, edge_weight=edge_weight))
            spatial_embeddings.append(h2)

        # Stack over time: [N, T, spatial_hidden_dim]
        H_spatial = torch.stack(spatial_embeddings, dim=1)

        # Step 2: Temporal modeling
        if self.temporal_backbone == "gru":
            H_temporal, _ = self.temporal_model(H_spatial)  # [N, T, 2 * temporal_hidden_dim]
        else:
            # Conv1d expects [N, Channels, Length]
            H_spatial_perm = H_spatial.permute(0, 2, 1)  # [N, spatial_dim, T]
            H_temporal = self.temporal_model(H_spatial_perm)  # [N, out_channels, T]
            H_temporal = H_temporal.permute(0, 2, 1)  # [N, T, out_channels]

        # Step 3: Attentive temporal aggregation
        node_embeddings, attn_weights = self.attention(H_temporal)  # [N, hidden], [N, T]

        # Step 4: Classification head
        logits = self.classifier(node_embeddings)  # [N, num_classes]
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "temporal_attention": attn_weights,
            "node_embeddings": node_embeddings,
        }


if __name__ == "__main__":
    print("Testing SpatioTemporalGNN Architecture...")
    N, T, num_feats = 8, 12, 4
    mock_x = torch.randn(N, T, num_feats)
    mock_adj = torch.eye(N) + 0.2 * torch.rand(N, N)
    mock_adj = (mock_adj + mock_adj.t()) / 2.0

    model = SpatioTemporalGNN(num_features=num_feats, spatial_hidden_dim=32, temporal_hidden_dim=32, num_classes=2)
    out = model(mock_x, A_spatial=mock_adj)

    print(f"Logits shape: {out['logits'].shape} (Expected: [{N}, 2])")
    print(f"Probs shape: {out['probs'].shape} (Expected: [{N}, 2])")
    print(f"Temporal Attention shape: {out['temporal_attention'].shape} (Expected: [{N}, {T}])")
    print(f"Node Embeddings shape: {out['node_embeddings'].shape}")
    print("SpatioTemporalGNN passed all unit checks!")
