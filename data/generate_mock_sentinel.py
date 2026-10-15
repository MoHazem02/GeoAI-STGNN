"""Synthetic Spatio-Temporal GIS Dataset Generator.

Simulates multi-temporal Sentinel-2 satellite observation cubes across spatial regions
and monthly timestamps, complete with spatial adjacency (distance-based RBF kernel)
and temporal graphs. Injects realistic environmental disturbance events (e.g., wildfire,
flood, or deforestation) causing abrupt spectral band shifts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import networkx as nx

try:
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    PyGData = None


BAND_NAMES = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"]
MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


@dataclass
class SpatioTemporalData:
    """Container for Spatio-Temporal Satellite Graph Data.

    Attributes:
        x: Multi-spectral feature tensor [N, T, F]
        y: Ground truth change labels [N] (0: Stable, 1: Disturbed/Change)
        A_spatial: Spatial adjacency matrix [N, N]
        A_temporal: Temporal adjacency matrix [T, T]
        edge_index_spatial: PyG-compatible spatial edge index [2, E_s]
        edge_weight_spatial: Spatial edge weights [E_s]
        coordinates: Region geographic 2D coordinates [N, 2] (simulated UTM/lat-lon)
        timestamps: List of timestamp string labels (length T)
        band_names: List of spectral band identifiers (length F)
        metadata: Additional domain metadata (change injection info, region IDs)
    """
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

    def to(self, device: torch.device) -> "SpatioTemporalData":
        """Move all PyTorch tensors to designated device."""
        return SpatioTemporalData(
            x=self.x.to(device),
            y=self.y.to(device),
            A_spatial=self.A_spatial.to(device),
            A_temporal=self.A_temporal.to(device),
            edge_index_spatial=self.edge_index_spatial.to(device),
            edge_weight_spatial=self.edge_weight_spatial.to(device),
            coordinates=self.coordinates,
            timestamps=self.timestamps,
            band_names=self.band_names,
            metadata=self.metadata,
        )

    def to_pyg_data(self):
        """Convert container to PyG Data instance (if PyG is available)."""
        if not PYG_AVAILABLE:
            raise RuntimeError("torch_geometric is not installed in the environment.")
        return PyGData(
            x=self.x,
            y=self.y,
            edge_index=self.edge_index_spatial,
            edge_attr=self.edge_weight_spatial,
            coordinates=torch.from_numpy(self.coordinates).float(),
            A_temporal=self.A_temporal,
        )


def compute_spectral_ndvi(red: torch.Tensor, nir: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Calculates Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)
    """
    return (nir - red) / (nir + red + eps)


def build_spatial_graph(
    coordinates: np.ndarray,
    distance_threshold: float = 35.0,
    rbf_sigma: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, nx.Graph]:
    """Constructs spatial graph adjacency via Gaussian RBF kernel over Euclidean distance.

    Args:
        coordinates: Spatial coordinates [N, 2].
        distance_threshold: Cut-off Euclidean distance for edge creation.
        rbf_sigma: Kernel bandwidth parameter for edge weighting.

    Returns:
        A_spatial: Dense symmetric adjacency matrix [N, N].
        edge_index: PyG-compatible edge indices [2, E].
        edge_weight: Edge weights [E].
        G: NetworkX graph object.
    """
    num_nodes = coordinates.shape[0]
    A_spatial = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i, pos=(coordinates[i, 0], coordinates[i, 1]))

    sources, targets, weights = [], [], []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                A_spatial[i, j] = 1.0  # Self-loop
            else:
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                if dist <= distance_threshold:
                    weight = float(np.exp(-(dist ** 2) / (2.0 * (rbf_sigma ** 2))))
                    A_spatial[i, j] = weight
                    sources.append(i)
                    targets.append(j)
                    weights.append(weight)
                    if i < j:
                        G.add_edge(i, j, weight=weight, distance=float(dist))

        # Add self-loops to edge list
        sources.append(i)
        targets.append(i)
        weights.append(1.0)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    A_spatial_tensor = torch.from_numpy(A_spatial)

    return A_spatial_tensor, edge_index, edge_weight, G


def build_temporal_graph(
    num_timestamps: int = 12,
    bidirectional_chain: bool = True,
    decay_gamma: float = 0.5,
) -> torch.Tensor:
    """Constructs temporal graph adjacency matrix representing sequential month connectivity.

    Args:
        num_timestamps: Number of consecutive temporal steps (default 12 for annual cycle).
        decay_gamma: Rate of temporal decay across lag steps.

    Returns:
        A_temporal: Dense temporal adjacency matrix [T, T].
    """
    A_temporal = np.zeros((num_timestamps, num_timestamps), dtype=np.float32)
    for t1 in range(num_timestamps):
        for t2 in range(num_timestamps):
            dt = abs(t1 - t2)
            if dt == 0:
                A_temporal[t1, t2] = 1.0
            elif dt == 1 and bidirectional_chain:
                A_temporal[t1, t2] = 0.8
            elif dt <= 2:
                A_temporal[t1, t2] = np.exp(-decay_gamma * dt)
    return torch.from_numpy(A_temporal)


def generate_mock_sentinel_dataset(
    num_nodes: int = 8,
    num_timestamps: int = 12,
    num_change_regions: int = 3,
    noise_std: float = 0.02,
    random_seed: int = 42,
) -> SpatioTemporalData:
    """Generates a synthetic spatio-temporal Sentinel-2 multi-spectral dataset.

    Simulates:
    1. Spatial Regions: Distributed centroids in a 100x100 km grid.
    2. Natural Phenology: Sinusoidal seasonal vegetation reflectance across 12 months.
       - Blue (B02): Low, slight winter increase (~0.06 - 0.09)
       - Green (B03): Moderate peak in summer (~0.10 - 0.14)
       - Red (B04): Strong chlorophyll absorption in summer (~0.05 - 0.12)
       - NIR (B08): High cellular scattering, massive summer peak (~0.30 - 0.52)
    3. Anomaly Disturbances (Wildfire / Deforestation / Flood):
       - Triggered mid-year (t >= 6) in selected regions.
       - Drastic NIR collapse, sharp Red increase (NDVI crashes).
       - Ground truth label set to 1 for disturbed nodes, 0 for stable.

    Args:
        num_nodes: Number of spatial regions (default 8, typically 5-10).
        num_timestamps: Temporal sequence length (default 12 months).
        num_change_regions: Number of nodes experiencing environmental disturbance.
        noise_std: Standard deviation of sensor/atmospheric noise.
        random_seed: Reproducibility seed.

    Returns:
        SpatioTemporalData container with tensors, adjacencies, and metadata.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 1. Generate 2D spatial coordinates for regions (e.g., in a 100 km x 100 km bounding area)
    # Arrange nodes in a cluster-like geographic spread
    coords = np.zeros((num_nodes, 2), dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    radius = 30.0 + np.random.uniform(-5.0, 5.0, size=num_nodes)
    coords[:, 0] = 50.0 + radius * np.cos(angles) + np.random.normal(0, 3.0, size=num_nodes)
    coords[:, 1] = 50.0 + radius * np.sin(angles) + np.random.normal(0, 3.0, size=num_nodes)

    # 2. Build spatial and temporal graphs
    A_spatial, edge_index, edge_weight, _ = build_spatial_graph(
        coords, distance_threshold=42.0, rbf_sigma=25.0
    )
    A_temporal = build_temporal_graph(num_timestamps=num_timestamps)

    # 3. Simulate seasonal baseline spectral signatures: [T, 4]
    # Month step: 0=Jan, 6=Jul, 11=Dec
    t_steps = np.arange(num_timestamps)
    # Seasonal phenology wave (peaks in month 6/July)
    phenology_factor = 0.5 * (1.0 - np.cos(2.0 * np.pi * t_steps / num_timestamps))

    # Initialize feature tensor: [N, T, 4]
    features = np.zeros((num_nodes, num_timestamps, 4), dtype=np.float32)

    # Ground truth labels: [N] (0: Stable, 1: Changed)
    labels = np.zeros(num_nodes, dtype=np.int64)

    # Select change regions uniformly
    num_change = min(num_change_regions, num_nodes - 1)
    change_node_indices = np.random.choice(num_nodes, size=num_change, replace=False)
    change_events_metadata = []

    for i in range(num_nodes):
        # Base reflectance per region with slight natural terrain variation
        base_blue = 0.07 + np.random.uniform(-0.01, 0.01)
        base_green = 0.11 + np.random.uniform(-0.01, 0.01)
        base_red = 0.09 + np.random.uniform(-0.01, 0.01)
        base_nir = 0.38 + np.random.uniform(-0.02, 0.02)

        # Baseline phenological curves across 12 months
        blue_curve = base_blue + 0.01 * (1.0 - phenology_factor)
        green_curve = base_green + 0.03 * phenology_factor
        red_curve = base_red - 0.04 * phenology_factor  # Chlorophyll absorbs red in summer
        nir_curve = base_nir + 0.14 * phenology_factor  # Peak canopy reflection in summer

        # Inject disturbance if region is selected for change
        if i in change_node_indices:
            labels[i] = 1
            change_month = int(np.random.choice([5, 6, 7]))  # Late spring / summer disturbance
            event_type = str(np.random.choice(["wildfire_burn", "deforestation", "flood_inundation"]))

            if event_type in ["wildfire_burn", "deforestation"]:
                # Abrupt collapse of NIR, increase in Red (loss of canopy, exposed soil/ash)
                nir_curve[change_month:] = 0.12 + np.random.uniform(-0.02, 0.02, size=num_timestamps - change_month)
                red_curve[change_month:] = 0.22 + np.random.uniform(-0.02, 0.02, size=num_timestamps - change_month)
                blue_curve[change_month:] += 0.03
            else:  # flood inundation
                # Water absorbs NIR completely, lowered visible reflectance
                nir_curve[change_month:] = 0.05 + np.random.uniform(-0.01, 0.01, size=num_timestamps - change_month)
                red_curve[change_month:] = 0.06 + np.random.uniform(-0.01, 0.01, size=num_timestamps - change_month)
                green_curve[change_month:] += 0.02

            change_events_metadata.append({
                "region_id": int(i),
                "event_type": event_type,
                "onset_month": int(change_month),
                "onset_month_name": MONTHS[change_month],
            })

        # Add Gaussian sensor noise
        blue_curve += np.random.normal(0, noise_std, size=num_timestamps)
        green_curve += np.random.normal(0, noise_std, size=num_timestamps)
        red_curve += np.random.normal(0, noise_std, size=num_timestamps)
        nir_curve += np.random.normal(0, noise_std, size=num_timestamps)

        # Physical clipping for reflectance [0.01, 0.95]
        features[i, :, 0] = np.clip(blue_curve, 0.01, 0.95)
        features[i, :, 1] = np.clip(green_curve, 0.01, 0.95)
        features[i, :, 2] = np.clip(red_curve, 0.01, 0.95)
        features[i, :, 3] = np.clip(nir_curve, 0.01, 0.95)

    x_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(labels)

    dataset = SpatioTemporalData(
        x=x_tensor,
        y=y_tensor,
        A_spatial=A_spatial,
        A_temporal=A_temporal,
        edge_index_spatial=edge_index,
        edge_weight_spatial=edge_weight,
        coordinates=coords,
        timestamps=MONTHS[:num_timestamps],
        band_names=BAND_NAMES,
        metadata={
            "num_nodes": num_nodes,
            "num_timestamps": num_timestamps,
            "num_bands": 4,
            "change_regions": [int(idx) for idx in change_node_indices],
            "change_events": change_events_metadata,
            "crs_simulated": "EPSG:32632 (UTM Zone 32N local projection scale)",
        },
    )

    return dataset


if __name__ == "__main__":
    print("Testing Synthetic Spatio-Temporal Sentinel Generator...")
    ds = generate_mock_sentinel_dataset(num_nodes=8, num_timestamps=12, num_change_regions=3)
    print(f"Features X shape: {ds.x.shape} (N, T, F)")
    print(f"Labels Y shape: {ds.y.shape} (distribution: {torch.bincount(ds.y).tolist()})")
    print(f"Spatial Adjacency shape: {ds.A_spatial.shape}")
    print(f"Spatial Edge Index shape: {ds.edge_index_spatial.shape}")
    print(f"Temporal Adjacency shape: {ds.A_temporal.shape}")
    print(f"Disturbance Events: {ds.metadata['change_events']}")
    print("Synthetic Sentinel data generator test passed successfully!")
