"""
Unified Grid Page — Digital Twin, Circuit Model, and Live Fault Graphs

Combines the digital twin visualization, circuit designer, and component
telemetry into a unified view with live fault graph visualization.
"""
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui.views.digital_twin import render_digital_twin
from src.ui.views.circuit_designer import render_circuit_designer
from src.domain.circuit import CircuitModel


def render_unified_grid():
    """Render unified digital twin and circuit design view."""
    st.markdown("## 🧩 Unified Grid: Digital Twin & Circuit Model")

    twin_tab, circuit_tab, telemetry_tab = st.tabs(
        ["Digital Twin", "Circuit Designer", "Component Telemetry"]
    )

    with twin_tab:
        render_digital_twin()
        render_live_fault_graphs()

        st.markdown("### 🧭 Fault Trace")
        fault_location = st.session_state.get("fault_location")
        if fault_location:
            circuit_model = st.session_state.get("circuit_model")
            trace = _format_fault_trace(circuit_model, fault_location)
            emulator = st.session_state.get("emulator")
            fault_info = emulator.get_fault_info() if emulator and hasattr(emulator, "get_fault_info") else {}
            if fault_info.get("active"):
                st.markdown(
                    f"**Fault Type:** {fault_info.get('type')}  \n"
                    f"**Severity:** {fault_info.get('severity', 0.0):.2f}  \n"
                    f"**Location Node:** {fault_info.get('location')}"
                )

            details = fault_location.get("details", {}) if isinstance(fault_location, dict) else {}
            energy_ratio = details.get("energy_ratio")
            if energy_ratio is not None:
                st.markdown(
                    f"**Wavelet Energy Ratio (D1/D2):** {energy_ratio:.2f}  \n"
                    "Higher ratios indicate closer, sharper transients; lower ratios imply attenuation over distance."
                )

            st.markdown(
                f"**Estimated Distance:** {trace.get('distance_m', 0.0):.1f} m  \n"
                f"**Zone:** {trace.get('zone', 'UNKNOWN')}  \n"
                f"**Confidence:** {trace.get('confidence', 0.0):.2f}"
            )
            if trace.get("line"):
                line = trace["line"]
                st.markdown(
                    f"**Mapped Line:** Line {line['id']} (Bus {line['from_bus']} → Bus {line['to_bus']})  \n"
                    f"**Offset Along Line:** {line['offset_m']:.1f} m (Line length ~{line['length']:.1f} m)"
                )
            st.info(trace.get("explanation", ""))
        else:
            st.info("Run a fault simulation to see the exact circuit trace.")

    with circuit_tab:
        render_circuit_designer()

    with telemetry_tab:
        _render_recording_controls()
        component_history = st.session_state.get("component_history", {})
        if st.session_state.get("replay_mode") and st.session_state.get("selected_recording"):
            recordings = st.session_state.get("recordings", [])
            selected = next(
                (rec for rec in recordings if rec["name"] == st.session_state.selected_recording),
                None
            )
            if selected:
                _render_component_telemetry("Recorded Component Telemetry", selected.get("data", {}))
        else:
            _render_component_telemetry("Live Component Telemetry", component_history)


def render_live_fault_graphs():
    """Render immediate fault visibility graphs in the Unified Grid view."""
    st.markdown("### 📉 Live Fault Graphs")

    if not st.session_state.get("system_running"):
        st.info("Start the system to see live fault graphs.")
        return

    # Graph 1: Main sensor stream
    voltage_data = st.session_state.get("voltage_data", [])
    if voltage_data:
        recent = voltage_data[-250:]
        t = list(range(len(recent)))

        fig_sensor = go.Figure()
        fig_sensor.add_trace(go.Scatter(
            x=t,
            y=recent,
            mode="lines",
            name="Sensor Voltage",
            line=dict(color="#00CC96", width=2)
        ))

        emulator = st.session_state.get("emulator")
        fault_info = emulator.get_fault_info() if emulator and hasattr(emulator, "get_fault_info") else {}
        if fault_info.get("active"):
            mean_v = np.mean(recent) if recent else 400
            fig_sensor.add_hline(
                y=mean_v,
                line_dash="dash",
                line_color="#e94560",
                annotation_text=f"FAULT ACTIVE: {fault_info.get('type', 'UNKNOWN')}"
            )

        fig_sensor.update_layout(
            height=280,
            xaxis_title="Sample",
            yaxis_title="Voltage (V)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_sensor, use_container_width=True)
    else:
        st.info("Waiting for sensor samples. Start the system and inject a fault.")

    # Graph 2: Per-node waveforms from emulator history
    emulator = st.session_state.get("emulator")
    if emulator and hasattr(emulator, "get_topology"):
        topology = emulator.get_topology()
        nodes = list(topology.get("nodes", {}).keys())
        if nodes:
            selected_nodes = st.multiselect(
                "Nodes to visualize",
                options=nodes,
                default=nodes[:min(3, len(nodes))],
                key="live_fault_nodes"
            )

            if selected_nodes and hasattr(emulator, "get_history"):
                fig_nodes = go.Figure()
                for node_id in selected_nodes:
                    history = emulator.get_history(node_id)
                    if len(history) == 0:
                        continue
                    downsampled = history[::100]
                    tx = np.arange(len(downsampled)) / (getattr(emulator, "sample_rate", 20000) / 100.0)
                    fig_nodes.add_trace(go.Scatter(
                        x=tx,
                        y=downsampled,
                        mode="lines",
                        name=f"Node {node_id}"
                    ))

                fig_nodes.update_layout(
                    height=320,
                    xaxis_title="Window Time (s)",
                    yaxis_title="Voltage (V)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig_nodes, use_container_width=True)


def _format_fault_trace(model: CircuitModel, fault_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a fault trace mapping from estimated distance to circuit elements."""
    trace = {
        "distance_m": fault_info.get("distance"),
        "zone": fault_info.get("zone"),
        "confidence": fault_info.get("confidence"),
        "line": None,
        "explanation": None,
    }

    if not model or not model.lines or not model.buses:
        trace["explanation"] = "Circuit model missing line/bus data for fault trace."
        return trace

    bus_positions = {bus.id: (bus.x, bus.y) for bus in model.buses}
    line_lengths = []
    for line in sorted(model.lines, key=lambda l: l.id):
        from_pos = bus_positions.get(line.from_bus, (0.0, 0.0))
        to_pos = bus_positions.get(line.to_bus, (0.0, 0.0))
        length = float(np.hypot(to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]))
        line_lengths.append((line, length))

    total_dist = 0.0
    target = fault_info.get("distance", 0.0) or 0.0
    target_line = None
    for line, length in line_lengths:
        if total_dist + length >= target:
            target_line = (line, total_dist, length)
            break
        total_dist += length

    if not target_line and line_lengths:
        target_line = (line_lengths[-1][0], total_dist, line_lengths[-1][1])

    if target_line:
        line, start_dist, length = target_line
        trace["line"] = {
            "id": line.id,
            "from_bus": line.from_bus,
            "to_bus": line.to_bus,
            "length": length,
            "offset_m": max(0.0, target - start_dist),
        }
        trace["explanation"] = (
            f"Estimated fault distance {target:.1f}m maps to Line {line.id} "
            f"between Bus {line.from_bus} and Bus {line.to_bus} at ~{max(0.0, target - start_dist):.1f}m."
        )
    else:
        trace["explanation"] = "Unable to map fault distance to a specific line."

    return trace


def _render_component_telemetry(title: str, series_by_node: Dict[str, List[Dict[str, Any]]]):
    """Render per-component telemetry graphs."""
    st.markdown(f"### {title}")
    if not series_by_node:
        st.info("No component telemetry available yet.")
        return

    for node_id, series in series_by_node.items():
        if not series:
            continue
        times = [entry["time"] for entry in series]
        voltages = [entry["voltage"] for entry in series]
        currents = [entry["current"] for entry in series]
        powers = [entry["power"] for entry in series]
        transients = [entry for entry in series if entry.get("transient")]

        with st.expander(f"Component {node_id}", expanded=False):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=voltages, name="Voltage (V)", line=dict(color="#4CAF50")))
            fig.add_trace(go.Scatter(x=times, y=currents, name="Current (A)", line=dict(color="#FFA726")))
            fig.add_trace(go.Scatter(x=times, y=powers, name="Power (W)", line=dict(color="#00CC96")))

            if transients:
                fig.add_trace(go.Scatter(
                    x=[entry["time"] for entry in transients],
                    y=[entry["voltage"] for entry in transients],
                    mode="markers",
                    marker=dict(color="#e94560", size=8),
                    name="Transient"
                ))

            fig.update_layout(
                height=280,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_recording_controls():
    """Render recording and replay controls."""
    st.markdown("### ⏺️ Recording & Replay")
    controls = st.columns([1, 1, 1])
    with controls[0]:
        if st.button("Start Recording", width='stretch', disabled=st.session_state.get("recording_enabled", False)):
            st.session_state.recording_enabled = True
            st.session_state.recording_data = {}
    with controls[1]:
        if st.button("Stop Recording", width='stretch', disabled=not st.session_state.get("recording_enabled", False)):
            st.session_state.recording_enabled = False
            recording_data = st.session_state.get("recording_data", {})
            if recording_data:
                recording = {
                    "name": datetime.now().strftime("Recording %H:%M:%S"),
                    "data": recording_data.copy()
                }
                recordings = st.session_state.get("recordings", [])
                recordings.append(recording)
                st.session_state.recordings = recordings
                st.session_state.recording_data = {}
    with controls[2]:
        if st.button("Clear Recordings", width='stretch'):
            st.session_state.recordings = []
            st.session_state.recording_data = {}

    recordings = st.session_state.get("recordings", [])
    if recordings:
        recording_names = [rec["name"] for rec in recordings]
        selected_name = st.selectbox("Replay Session", recording_names)
        st.session_state.selected_recording = selected_name
        st.session_state.replay_mode = st.toggle("Replay Mode", value=st.session_state.get("replay_mode", False))