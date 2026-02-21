"""
Digital Twin Page — Enhanced Circuit Schematic Visualization

Shows a proper electrical circuit schematic with bus bars, cables,
generators, loads, real-time voltage, and fault highlighting.
Adapted from ZAI's enhanced implementation.
"""
import time
import streamlit as st
from src.ui.system import get_per_node_voltages


def render_digital_twin():
    """Render the enhanced digital twin circuit schematic."""
    st.markdown("""
    <div class="page-header">
        <h2>🏗️ Digital Twin — Circuit Schematic</h2>
        <p>Real-time electrical schematic with fault highlighting</p>
    </div>
    """, unsafe_allow_html=True)

    nodes = get_per_node_voltages()
    circuit = st.session_state.get("circuit_model")

    if not nodes or not circuit:
        st.info("Start the system to see the Digital Twin schematic.")
        return

    # Gather fault state
    fault_info = {
        "active": st.session_state.get("fault_active", False),
        "type": st.session_state.get("fault_type", "NONE"),
        "location": st.session_state.get("fault_location"),
        "zone": st.session_state.get("fault_zone"),
        "distance": st.session_state.get("fault_distance"),
        "trip_active": st.session_state.get("trip_active", False),
    }

    # Render circuit schematic via HTML/CSS/JS
    html = _build_schematic_html(nodes, circuit, fault_info)
    st.components.v1.html(html, height=580, scrolling=False)

    # DSP stats card
    _render_dsp_stats()

    # Node details table
    st.markdown("#### 📋 Node Status")
    _render_node_table(nodes)


def _build_schematic_html(nodes, circuit, fault_info):
    """Build an HTML/CSS/JS circuit schematic."""

    # Build bus bar elements
    bus_elements = ""
    for bus in circuit.buses:
        node_id = str(bus.id)
        info = nodes.get(node_id, {})
        v = info.get("voltage", 0)
        name = info.get("name", bus.name)
        status = info.get("status", "UNKNOWN")
        status_val = status.value if hasattr(status, "value") else str(status)

        # Determine styling
        is_fault = status_val == "FAULT"
        is_warn = v < 360 and not is_fault

        color = "#ff4444" if is_fault else ("#ffaa00" if is_warn else "#00e676")
        glow = "0 0 20px #ff4444, 0 0 40px #ff0000" if is_fault else (
            "0 0 10px #ffaa00" if is_warn else "0 0 8px #00e67644"
        )
        fault_badge = (
            f'<div class="fault-badge">⚡ {fault_info.get("type", "")}</div>'
            if is_fault else ""
        )

        bus_elements += f"""
        <div class="bus-node" id="bus-{node_id}"
             style="left:{bus.x}px; top:{bus.y}px; border-color:{color};
                    box-shadow:{glow};">
            <div class="bus-icon">{'⚡' if is_fault else ('🔌' if bus.type == 'Slack' else '◈')}</div>
            <div class="bus-name">{name}</div>
            <div class="bus-voltage" style="color:{color};">{v:.1f}V</div>
            {fault_badge}
        </div>
        """

    # Build cable elements
    cable_elements = ""
    for line in circuit.lines:
        from_bus = next((b for b in circuit.buses if b.id == line.from_bus), None)
        to_bus = next((b for b in circuit.buses if b.id == line.to_bus), None)
        if not from_bus or not to_bus:
            continue

        # Offset to center of the bus node (approx 55px wide, 70px tall)
        x1, y1 = from_bus.x + 55, from_bus.y + 35
        x2, y2 = to_bus.x + 55, to_bus.y + 35

        cable_elements += f"""
        <svg class="cable-svg" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;">
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                  stroke="rgba(255,255,255,0.15)" stroke-width="2"
                  stroke-dasharray="8 4"/>
            <text x="{(x1+x2)//2}" y="{(y1+y2)//2 - 6}"
                  fill="rgba(255,255,255,0.4)" font-size="10" text-anchor="middle">
                {line.r_ohm:.3f}Ω · {line.length_km*1000:.0f}m
            </text>
        </svg>
        """

    # Trip banner
    trip_banner = ""
    if fault_info.get("trip_active"):
        trip_banner = """
        <div class="trip-banner">
            🔴 SYSTEM TRIP ACTIVE — Protective relay opened
        </div>
        """

    return f"""
    <style>
        .schematic-container {{
            position: relative;
            width: 100%;
            height: 520px;
            background: #0a0e1a;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            overflow: hidden;
            font-family: 'Segoe UI', sans-serif;
        }}
        .schematic-container::before {{
            content: '';
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 50px 50px;
        }}
        .bus-node {{
            position: absolute;
            width: 110px;
            padding: 8px 6px;
            background: rgba(20, 25, 45, 0.92);
            border: 2px solid;
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            z-index: 2;
        }}
        .bus-node:hover {{
            transform: scale(1.08);
        }}
        .bus-icon {{
            font-size: 20px;
            margin-bottom: 2px;
        }}
        .bus-name {{
            color: #ccc;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .bus-voltage {{
            font-size: 16px;
            font-weight: 700;
            font-family: 'Courier New', monospace;
        }}
        .fault-badge {{
            position: absolute;
            top: -10px;
            right: -10px;
            background: #ff2222;
            color: white;
            font-size: 9px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 8px;
            animation: pulse 0.8s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.15); }}
        }}
        .trip-banner {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            padding: 8px;
            background: linear-gradient(135deg, #d32f2f, #b71c1c);
            color: white;
            text-align: center;
            font-weight: 700;
            font-size: 13px;
            z-index: 10;
            animation: tripFlash 1s infinite;
        }}
        @keyframes tripFlash {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        .legend {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(10,14,26,0.9);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 10px;
            color: #aaa;
            z-index: 5;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
    </style>

    <div class="schematic-container">
        {trip_banner}
        {cable_elements}
        {bus_elements}

        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#00e676;"></div>Normal</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ffaa00;"></div>Under-voltage</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ff4444;"></div>Fault</div>
        </div>
    </div>
    """


def _render_dsp_stats():
    """Show DSP performance stats if available."""
    hsl = st.session_state.get("high_speed_loop")
    dsp = st.session_state.get("dsp_pipeline")

    if hsl or dsp:
        st.markdown("#### ⚙️ DSP Performance")
        cols = st.columns(4)

        if hsl:
            stats = hsl.get_stats()
            cols[0].metric("Samples", f"{stats['total_samples']:,}")
            cols[1].metric("Trips", stats["total_trips"])
            cols[2].metric("Avg Latency", f"{stats['avg_processing_us']:.1f} µs")
            cols[3].metric("Loop Rate", f"{stats['sample_rate']:,} Hz")
        elif dsp:
            try:
                cols[0].metric("Samples", f"{dsp.total_samples:,}")
                cols[1].metric("Trips", dsp.total_trips)
                cols[2].metric("Avg Latency", f"{dsp.avg_processing_us:.1f} µs")
                cols[3].metric("DSP Path", "C++ Active")
            except Exception:
                cols[0].metric("DSP Path", "C++ Active")


def _render_node_table(nodes):
    """Render node status as a table."""
    import pandas as pd

    rows = []
    for node_id, info in nodes.items():
        status = info.get("status", "UNKNOWN")
        status_val = status.value if hasattr(status, "value") else str(status)
        rows.append({
            "Node ID": node_id,
            "Name": info.get("name", node_id),
            "Voltage (V)": f"{info.get('voltage', 0):.1f}",
            "Status": status_val,
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
