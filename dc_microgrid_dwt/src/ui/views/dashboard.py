"""
Dashboard Page — Main real-time monitoring view.

Shows voltage waveform, DWT energy breakdown, fault status,
and key system metrics at a glance.
"""
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.ui.styles import PLOTLY_DARK_THEME, COLORS
from src.ui.system import get_per_node_voltages


def render_dashboard():
    """Render the main dashboard page."""
    # --- Header ---
    st.markdown("""
    <div class="page-header">
        <h2>📊 Real-Time Dashboard</h2>
        <p>Live monitoring of DC microgrid voltages, wavelet energy, and fault status</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Top Metrics Row ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        voltage = st.session_state.voltage_data[-1] if st.session_state.voltage_data else 0
        _metric_card("Bus Voltage", f"{voltage:.1f}", "V", COLORS["voltage"])
    with col2:
        d1 = st.session_state.dwt_energy.get("D1", 0)
        _metric_card("D1 Energy", f"{d1:.2f}", "HF Detail", COLORS["d1"])
    with col3:
        eps = st.session_state.health_data.get("eps", 0)
        _metric_card("Throughput", f"{eps:.0f}", "events/s", COLORS["accent"])
    with col4:
        latency = st.session_state.health_data.get("latency", 0)
        _metric_card("Latency", f"{latency:.1f}", "ms", COLORS["warning"])
    with col5:
        status = "🔴 FAULT" if st.session_state.fault_active else "🟢 NORMAL"
        color = COLORS["danger"] if st.session_state.fault_active else COLORS["success"]
        _metric_card("Status", status, "", color)

    # --- Fault Alert Banner ---
    if st.session_state.fault_active:
        fault_type = st.session_state.fault_type
        zone = st.session_state.fault_zone or "Unknown"
        dist = st.session_state.fault_distance
        dist_str = f" at {dist:.1f}m" if dist else ""
        st.markdown(f"""
        <div class="fault-alert">
            <strong>⚠️ FAULT DETECTED</strong>: {fault_type} in Zone {zone}{dist_str}
        </div>
        """, unsafe_allow_html=True)

    # --- Main Graphs ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        _render_voltage_waveform()

    with col_right:
        _render_energy_breakdown()

    # --- Per-Node Voltage Summary ---
    _render_node_voltages()


def _metric_card(label, value, unit, color):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="background: linear-gradient(135deg, {color}, {color}aa); 
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {value}
        </div>
        <div class="metric-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_voltage_waveform():
    """Render real-time voltage waveform plot."""
    st.markdown("#### ⚡ Voltage Waveform")

    data = st.session_state.voltage_data
    if not data:
        st.info("Waiting for voltage data...")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data[-300:],
        mode="lines",
        name="Voltage",
        line=dict(color=COLORS["voltage"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0, 188, 212, 0.1)"
    ))

    # Add threshold lines
    fig.add_hline(y=400, line_dash="dash", line_color=COLORS["success"],
                  annotation_text="Nominal 400V", annotation_position="top left")
    fig.add_hline(y=360, line_dash="dot", line_color=COLORS["warning"],
                  annotation_text="Low Limit", annotation_position="bottom left")

    fig.update_layout(
        **PLOTLY_DARK_THEME,
        height=320,
        xaxis_title="Sample",
        yaxis_title="Voltage (V)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_energy_breakdown():
    """Render DWT energy breakdown as bar chart."""
    st.markdown("#### 🔬 DWT Energy Spectrum")

    energy = st.session_state.dwt_energy
    if not any(energy.values()):
        st.info("Waiting for DWT data...")
        return

    levels = ["D1", "D2", "D3", "D4", "A4"]
    values = [energy.get(k, 0) for k in levels]
    colors = [COLORS["d1"], COLORS["d2"], COLORS["d3"], COLORS["d4"], COLORS["a4"]]

    fig = go.Figure(data=[
        go.Bar(
            x=levels, y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(color="white", size=10),
        )
    ])
    fig.update_layout(
        **PLOTLY_DARK_THEME,
        height=320,
        yaxis_title="Energy",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_node_voltages():
    """Render per-node voltage summary cards."""
    nodes = get_per_node_voltages()
    if not nodes:
        return

    st.markdown("#### 🔌 Per-Node Voltages")
    cols = st.columns(min(len(nodes), 6))
    for i, (node_id, info) in enumerate(nodes.items()):
        with cols[i % len(cols)]:
            v = info.get("voltage", 0)
            name = info.get("name", node_id)
            status = info.get("status", "UNKNOWN")
            status_val = status.value if hasattr(status, 'value') else str(status)

            badge_class = "status-normal" if status_val == "ACTIVE" else "status-fault"
            st.markdown(f"""
            <div class="component-card">
                <div style="font-size: 11px; color: #8892b0;">Node {node_id}</div>
                <div style="font-size: 14px; color: #fff; font-weight: 600;">{name}</div>
                <div style="font-size: 24px; color: {COLORS['voltage']}; font-weight: bold;">{v:.1f}V</div>
                <span class="status-badge {badge_class}">{status_val}</span>
            </div>
            """, unsafe_allow_html=True)
