"""
Digital Twin Page — Interactive grid topology visualization.

Shows the live state of all nodes, connections, voltages,
and fault locations on a network diagram.
"""
import streamlit as st
import plotly.graph_objects as go
from src.ui.styles import PLOTLY_DARK_THEME, COLORS
from src.ui.system import get_per_node_voltages


def render_digital_twin():
    """Render the digital twin network visualization."""
    st.markdown("""
    <div class="page-header">
        <h2>🏗️ Digital Twin</h2>
        <p>Live topology view with per-node voltage and fault status</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Network Diagram ---
    nodes = get_per_node_voltages()
    circuit = st.session_state.get("circuit_model")

    if not nodes or not circuit:
        st.info("Start the system to see the digital twin.")
        return

    _render_topology_graph(nodes, circuit)

    # --- Node Details Table ---
    st.markdown("#### 📋 Node Status")
    _render_node_table(nodes)


def _render_topology_graph(nodes, circuit):
    """Render the grid topology as a Plotly network graph."""
    fig = go.Figure()

    # Build position map from circuit buses
    pos = {}
    for bus in circuit.buses:
        pos[str(bus.id)] = (bus.x, bus.y)

    # Draw connections (lines)
    for line in circuit.lines:
        from_id = str(line.from_bus)
        to_id = str(line.to_bus)
        if from_id in pos and to_id in pos:
            x0, y0 = pos[from_id]
            x1, y1 = pos[to_id]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.2)", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Draw nodes
    fault_loc = st.session_state.get("fault_location")

    for node_id, info in nodes.items():
        if node_id not in pos:
            continue

        x, y = pos[node_id]
        v = info.get("voltage", 0)
        name = info.get("name", node_id)
        status = info.get("status", "UNKNOWN")
        status_val = status.value if hasattr(status, 'value') else str(status)

        # Color by status
        if status_val == "FAULT":
            color = COLORS["danger"]
            size = 28
        elif v < 360:
            color = COLORS["warning"]
            size = 22
        else:
            color = COLORS["success"]
            size = 20

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=size,
                color=color,
                line=dict(width=2, color="rgba(255,255,255,0.3)"),
                symbol="circle",
            ),
            text=f"{name}<br>{v:.1f}V",
            textposition="top center",
            textfont=dict(size=10, color="white"),
            hovertemplate=f"<b>{name}</b><br>ID: {node_id}<br>Voltage: {v:.1f}V<br>Status: {status_val}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        PLOTLY_DARK_THEME,
        height=450,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        title="Grid Topology",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_node_table(nodes):
    """Render node status as a table."""
    import pandas as pd

    rows = []
    for node_id, info in nodes.items():
        status = info.get("status", "UNKNOWN")
        status_val = status.value if hasattr(status, 'value') else str(status)
        rows.append({
            "Node ID": node_id,
            "Name": info.get("name", node_id),
            "Voltage (V)": f"{info.get('voltage', 0):.1f}",
            "Status": status_val,
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
