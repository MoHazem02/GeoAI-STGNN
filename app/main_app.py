"""Interactive Streamlit App for Spatio-Temporal GNN & Active Learning Pipeline."""

import os
import sys

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, f1_score

# Add current directory to path for local imports
sys.path.append(os.getcwd())

from data.generate_mock_sentinel import generate_mock_sentinel_dataset
from models.st_gnn import SpatioTemporalGNN
from pipeline.active_learner import ActiveLearner

# Set page configuration
st.set_page_config(
    page_title="GeoAI ST-GNN",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stDataFrame {
        border: 1px solid #d1d1d1;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666666;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    st.title("🛰️ Spatio-Temporal GNN & Active Learning for Satellite Change Detection")
    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("🎛️ Configuration")

    # Dataset parameters
    st.sidebar.subheader("Dataset Parameters")
    num_nodes = st.sidebar.slider("Number of Spatial Regions", 5, 15, 8)
    num_timestamps = st.sidebar.slider("Temporal Window (Months)", 6, 24, 12)
    num_change_regions = st.sidebar.slider(
        "Change Regions", 1, min(5, num_nodes - 1), min(3, num_nodes - 1)
    )

    # Model parameters
    st.sidebar.subheader("Model Parameters")
    epochs = st.sidebar.slider("Training Epochs", 1, 20, 5)
    lr = st.sidebar.selectbox("Learning Rate", [0.001, 0.005, 0.01, 0.05, 0.1], index=2)

    # Active Learning parameters
    st.sidebar.subheader("Active Learning")
    strategy = st.sidebar.selectbox(
        "Query Strategy", ["entropy", "margin", "least_confidence"], index=0
    )
    al_budget = st.sidebar.slider("Query Budget (k)", 1, min(10, num_nodes), 3)

    # Random seed
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    # Run button
    run_button = st.sidebar.button("🚀 Run Pipeline", type="primary")

    # Initialize session state
    if "data" not in st.session_state:
        st.session_state.data = None
        st.session_state.model = None
        st.session_state.metrics_history = []
        st.session_state.queried_nodes = []
        st.session_state.annotations = {}

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🌍 Spatial Graph Visualization")
        if st.session_state.data is not None:
            fig = create_spatial_graph_plot(st.session_state.data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline to generate and visualize the spatial graph.")

    with col2:
        st.header("📊 Metrics Dashboard")
        if st.session_state.metrics_history:
            # Display latest metrics
            latest_metrics = st.session_state.metrics_history[-1]
            col2a, col2b = st.columns(2)
            with col2a:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{latest_metrics['accuracy']:.4f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with col2b:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-label">F1-Score</div>
                    <div class="metric-value">{latest_metrics['f1_score']:.4f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Training loss curve
            st.subheader("📉 Training Loss")
            loss_fig = create_loss_plot(st.session_state.metrics_history)
            st.plotly_chart(loss_fig, use_container_width=True)
        else:
            st.info("Run the pipeline to see metrics.")

    # Active Learning Panel
    st.header("🔍 Active Learning Panel")
    if st.session_state.data is not None and st.session_state.model is not None:
        # Query uncertain nodes
        al = ActiveLearner(strategy=strategy)
        queried_nodes, uncertainty_scores = al.query(
            st.session_state.model,
            st.session_state.data,
            k=al_budget,
            return_scores=True,
        )
        st.session_state.queried_nodes = queried_nodes

        # Display queried nodes
        st.subheader(f"Top {al_budget} Uncertain Nodes for Annotation")
        for i, node_idx in enumerate(queried_nodes):
            score = uncertainty_scores[node_idx].item()
            true_label = st.session_state.data.y[node_idx].item()
            event_info = " (Change Region)" if true_label == 1 else " (Stable Region)"

            col_node1, col_node2, col_node3 = st.columns([1, 2, 1])
            with col_node1:
                st.markdown(f"**Node {node_idx}**")
            with col_node2:
                st.progress(score)  # Visualize uncertainty as progress bar
            with col_node3:
                st.markdown(f"`{score:.4f}`")

            # Annotation simulation
            if node_idx not in st.session_state.annotations:
                st.session_state.annotations[
                    node_idx
                ] = true_label  # Simulate expert annotation

            st.caption(
                f"Uncertainty: {score:.4f} | True Label: {true_label} {event_info}"
            )

        # Display annotation summary
        st.subheader("📝 Annotation Summary")
        annotated_count = len(st.session_state.annotations)
        st.success(f"✅ {annotated_count} nodes annotated")

        # Show annotated nodes in a table
        if st.session_state.annotations:
            annotated_data = []
            for node_idx, label in st.session_state.annotations.items():
                annotated_data.append(
                    {
                        "Node ID": node_idx,
                        "Predicted Label": "Change" if label == 1 else "Stable",
                        "True Label": "Change"
                        if st.session_state.data.y[node_idx].item() == 1
                        else "Stable",
                        "Match": "✅"
                        if label == st.session_state.data.y[node_idx].item()
                        else "❌",
                    }
                )
            st.table(annotated_data)
    else:
        st.info("Run the pipeline to activate the Active Learning Panel.")

    # Run pipeline when button is clicked
    if run_button:
        with st.spinner("Running Spatio-Temporal GNN Pipeline..."):
            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Generate dataset
            st.info("Generating synthetic spatio-temporal dataset...")
            data = generate_mock_sentinel_dataset(
                num_nodes=num_nodes,
                num_timestamps=num_timestamps,
                num_change_regions=num_change_regions,
                random_seed=seed,
            )
            st.session_state.data = data

            # Initialize model
            st.info("Initializing Spatio-Temporal GNN model...")
            model = SpatioTemporalGNN(
                num_features=4,
                spatial_hidden_dim=32,
                temporal_hidden_dim=32,
                num_classes=2,
                dropout=0.2,
                temporal_backbone="gru",
                use_pyg=False,
            )
            st.session_state.model = model

            # Train model
            st.info(f"Training model for {epochs} epochs...")
            metrics_history = train_model_with_history(
                model, data, epochs=epochs, lr=lr
            )
            st.session_state.metrics_history = metrics_history

            # Evaluate model
            st.info("Evaluating model performance...")
            metrics = evaluate_model(model, data)
            st.session_state.metrics_history.append(metrics)

            st.success("Pipeline execution completed successfully!")

            # Rerun to update UI
            st.rerun()


def train_model_with_history(model, data, epochs=5, lr=0.01):
    """Train model and return metrics history."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    metrics_history = []

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

        # Calculate metrics for this epoch
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro")

            metrics_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "accuracy": accuracy,
                    "f1_score": f1,
                }
            )

    return metrics_history


def evaluate_model(model, data):
    """Evaluate model performance."""
    model.eval()
    with torch.no_grad():
        out = model(
            x=data.x,
            A_spatial=data.A_spatial,
            edge_index=data.edge_index_spatial,
            edge_weight=data.edge_weight_spatial,
        )
        probs = torch.softmax(out["logits"], dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

    labels = data.y.cpu().numpy()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return {
        "epoch": "Final",
        "loss": 0,  # Not calculated here
        "accuracy": accuracy,
        "f1_score": f1,
    }


def create_spatial_graph_plot(data):
    """Create Plotly figure for spatial graph visualization."""
    # Create NetworkX graph from edge index
    G = nx.Graph()
    edge_index = data.edge_index_spatial.numpy()
    edge_weight = data.edge_weight_spatial.numpy()
    coordinates = data.coordinates
    labels = data.y.numpy()

    # Add nodes with positions and labels
    for i in range(coordinates.shape[0]):
        G.add_node(
            i,
            pos=(coordinates[i, 0], coordinates[i, 1]),
            label="Change" if labels[i] == 1 else "Stable",
        )

    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < dst:  # Avoid duplicate edges
            G.add_edge(src, dst, weight=edge_weight[i])

    # Extract positions
    pos = nx.get_node_attributes(G, "pos")
    node_labels = nx.get_node_attributes(G, "label")

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G[edge[0]][edge[1]]["weight"])

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append("red" if node_labels[node] == "Change" else "blue")
        node_text.append(f"Node {node}<br>Type: {node_labels[node]}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(showscale=False, color=node_colors, size=10, line_width=2),
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Spatial Graph Topology",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Red: Change Regions, Blue: Stable Regions",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
        ),
    )
    return fig


def create_loss_plot(metrics_history):
    """Create Plotly figure for training loss curve."""
    epochs = [m["epoch"] for m in metrics_history if m["epoch"] != "Final"]
    losses = [m["loss"] for m in metrics_history if m["epoch"] != "Final"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            mode="lines+markers",
            name="Training Loss",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="Training Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


if __name__ == "__main__":
    main()
