"""
Fault Analysis Page — Detailed fault tracing, timeline, and zone analysis.

Shows fault events, distance estimation, and per-component impact graphs.
"""
import time
import streamlit as st
import plotly.graph_objects as go
from src.ui.styles import PLOTLY_DARK_THEME, COLORS
from src.ui.system import get_node_histories, get_per_node_voltages


def render_fault_analysis():
    """Render the fault analysis page."""
    st.markdown("""
    <div class="page-header">
        <h2>🔍 Fault Analysis</h2>
        <p>Detailed fault tracing, timeline, zone analysis, and per-component impact</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Active Fault Summary ---
    _render_fault_summary()

    # --- Per-Component Voltage During Fault ---
    st.markdown("#### 📊 Per-Component Voltage Traces")
    _render_per_node_traces()

    # --- Fault Event Timeline ---
    st.markdown("#### 📅 Fault Event Log")
    _render_fault_timeline()


def _render_fault_summary():
    """Show summary of the current or last fault."""
    if st.session_state.fault_active:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Type", st.session_state.fault_type or "—")
        with col2:
            zone = st.session_state.fault_zone or "—"
            st.metric("Zone", zone)
        with col3:
            dist = st.session_state.fault_distance
            st.metric("Distance", f"{dist:.1f}m" if dist else "—")
        with col4:
            trip = "YES" if st.session_state.trip_active else "NO"
            st.metric("Trip", trip)
    else:
        st.success("No active fault — system operating normally.")


def _render_per_node_traces():
    """Show per-node voltage traces from emulator history."""
    histories = get_node_histories()

    if not histories:
        st.info("Start system and wait for data to see per-node traces.")
        return

    fig = go.Figure()

    node_colors = [COLORS["voltage"], COLORS["d1"], COLORS["warning"],
                   COLORS["d3"], COLORS["d4"], COLORS["a4"]]

    for i, (node_id, data) in enumerate(histories.items()):
        if len(data) == 0:
            continue
        color = node_colors[i % len(node_colors)]
        # Get node name
        nodes_info = get_per_node_voltages()
        name = nodes_info.get(node_id, {}).get("name", f"Node {node_id}")

        fig.add_trace(go.Scatter(
            y=data[-500:],
            mode="lines",
            name=f"{name} ({node_id})",
            line=dict(color=color, width=1.2),
        ))

    fig.update_layout(
        **PLOTLY_DARK_THEME,
        height=400,
        xaxis_title="Sample",
        yaxis_title="Voltage (V)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_fault_timeline():
    """Render the fault event log — actual recorded events, not hardcoded."""
    events = st.session_state.get("fault_events", [])

    if not events:
        st.caption("No fault events recorded yet.")
        return

    # Show as table
    import pandas as pd
    rows = []
    for evt in events[-20:]:
        rows.append({
            "Time": time.strftime("%H:%M:%S", time.localtime(evt.get("time", 0))),
            "Type": evt.get("type", "—"),
            "Zone": evt.get("zone", "—"),
            "Distance (m)": f"{evt.get('distance', 0):.1f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
