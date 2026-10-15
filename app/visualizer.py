"""Interactive Terminal Visualizer for Spatio-Temporal Change Detection.

Renders synthetic satellite data cubes, spatial graphs, and model uncertainty
distributions directly in the terminal for rapid prototyping and debugging.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx

# Add current directory to path for local imports
sys.path.append(os.getcwd())

from data.generate_mock_sentinel import generate_mock_sentinel_dataset, BAND_NAMES, MONTHS


def visualize_spatio_temporal_cube(
    data: "SpatioTemporalData",
    node_idx: int = 0,
    save_path: str = "spatio_temporal_cube.png",
) -> None:
    """Visualize multi-spectral time series for a specific spatial node."""
    features = data.x[node_idx].numpy()  # [T, F]
    timestamps = data.timestamps
    band_names = data.band_names
    label = "Change" if data.y[node_idx].item() == 1 else "Stable"
    
    plt.figure(figsize=(10, 6))
    for i, band_name in enumerate(band_names):
        plt.plot(timestamps, features[:, i], marker='o', label=band_name)
    
    plt.title(f"Spectral Time Series for Node {node_idx} ({label})")
    plt.xlabel("Month")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Spatio-temporal cube visualization saved to {save_path}")


def visualize_spatial_graph(
    data: "SpatioTemporalData",
    save_path: str = "spatial_graph.png",
) -> None:
    """Visualize spatial graph topology with node labels."""
    # Create NetworkX graph from edge index
    G = nx.Graph()
    edge_index = data.edge_index_spatial.numpy()
    edge_weight = data.edge_weight_spatial.numpy()
    coordinates = data.coordinates
    
    # Add nodes with positions and labels
    for i in range(coordinates.shape[0]):
        G.add_node(i, pos=(coordinates[i, 0], coordinates[i, 1]), 
                   label="C" if data.y[i].item() == 1 else "S")
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < dst:  # Avoid duplicate edges
            G.add_edge(src, dst, weight=edge_weight[i])
    
    # Draw graph
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=['red' if labels[n] == 'C' else 'blue' for n in G.nodes()],
                           node_size=500, alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels={n: f"{n}({labels[n]})" for n in G.nodes()}, font_size=10)
    nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight']*2 for u, v in G.edges()], alpha=0.5)
    
    plt.title("Spatial Graph Topology\n(Red=C, Blue=S)")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Spatial graph visualization saved to {save_path}")


def print_change_report(data: "SpatioTemporalData") -> None:
    """Print a textual summary of change events in the dataset."""
    print("\n=== CHANGE DETECTION REPORT ===")
    print(f"Total Nodes: {data.x.shape[0]}")
    print(f"Change Nodes: {data.y.sum().item()}")
    print(f"Stable Nodes: {(1 - data.y).sum().item()}")
    
    change_events = data.metadata.get("change_events", [])
    if change_events:
        print("\nDetected Change Events:")
        for event in change_events:
            print(f"  - Node {event['region_id']}: {event['event_type']} in {event['onset_month_name']}")
    else:
        print("\nNo change events detected.")
    
    print("\nSpectral Bands:")
    for i, band in enumerate(data.band_names):
        print(f"  {i+1}. {band}")
    
    print("\nTemporal Coverage:")
    print(f"  Months: {', '.join(data.timestamps)}")
    
    print("\nSpatial Graph:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Edges: {data.edge_index_spatial.shape[1]}")
    
    print("\n=== END REPORT ===")


def main():
    print("Generating synthetic dataset for visualization...")
    data = generate_mock_sentinel_dataset(num_nodes=8, num_timestamps=12, num_change_regions=3)
    
    # Create visualizations
    visualize_spatio_temporal_cube(data, node_idx=0, save_path="spatio_temporal_cube.png")
    visualize_spatial_graph(data, save_path="spatial_graph.png")
    
    # Print report
    print_change_report(data)
    
    print("\nVisualizations and report generated successfully!")


if __name__ == "__main__":
    main()
