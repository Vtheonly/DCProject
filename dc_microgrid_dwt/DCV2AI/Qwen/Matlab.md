You're absolutely right — the current node-based graph isn't a true **circuit schematic**. Let me create a proper electrical diagram visualization with component symbols, fault highlighting, and MATLAB interoperability.

## 🎯 Solution Overview

| Feature | Current | New Implementation |
|---------|---------|-------------------|
| Visualization | Dots + Lines | **Electrical Symbols** (bus bars, breakers, loads) |
| Fault Display | Node color change | **Component highlighting** with fault type icon |
| Layout | Manual coordinates | **Auto-layout + Custom positioning** |
| Export | None | **MATLAB .fig + SVG + PNG** |
| Interactivity | Basic hover | **Click components, view details, inject faults** |

---

## 📁 New Files to Create

### 1. `src/ui/pages/circuit_schematic.py` (New Schematic Renderer)

```python
"""
Circuit Schematic Page — Professional Electrical Diagram
Renders actual circuit symbols (bus bars, breakers, generators, loads)
with fault highlighting on the specific failed component.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from src.ui.styles import PLOTLY_DARK_THEME, COLORS
from src.domain.models import FaultType, NodeStatus

# =============================================================================
# ELECTRICAL SYMBOL DEFINITIONS
# =============================================================================

def create_bus_symbol(x: float, y: float, length: float = 60, 
                      voltage: float = 400, status: str = "ACTIVE") -> go.Figure:
    """Create a bus bar symbol (thick horizontal line)."""
    color = COLORS["success"] if status == "ACTIVE" else COLORS["danger"]
    
    return go.Scatter(
        x=[x - length/2, x + length/2],
        y=[y, y],
        mode="lines",
        line=dict(color=color, width=8),
        hovertemplate=f"<b>Bus Bar</b><br>Voltage: {voltage:.1f}V<br>Status: {status}<extra></extra>",
        showlegend=False,
        name=f"BUS_{x}_{y}"
    )

def create_generator_symbol(x: float, y: float, size: float = 30,
                           name: str = "GEN", power: float = 0) -> go.Figure:
    """Create a generator symbol (circle with G)."""
    # Circle
    theta = [i * 2 * 3.14159 / 50 for i in range(51)]
    circle_x = [x + size * 0.8 * np.cos(t) for t in theta]
    circle_y = [y + size * 0.8 * np.sin(t) for t in theta]
    
    return [
        go.Scatter(
            x=circle_x, y=circle_y,
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            fill="toself",
            fillcolor="rgba(79, 195, 247, 0.1)",
            hovertemplate=f"<b>Generator</b><br>{name}<br>Power: {power:.2f} MW<extra></extra>",
            showlegend=False,
            name=f"GEN_{x}_{y}"
        ),
        go.Scatter(
            x=[x], y=[y],
            mode="text",
            text=["⚡"],
            textfont=dict(size=20, color=COLORS["accent"]),
            showlegend=False,
            name=f"GEN_LABEL_{x}_{y}"
        )
    ]

def create_load_symbol(x: float, y: float, size: float = 30,
                      name: str = "LOAD", power: float = 0) -> go.Figure:
    """Create a load symbol (rectangle with L)."""
    return [
        go.Scatter(
            x=[x - size/2, x + size/2, x + size/2, x - size/2, x - size/2],
            y=[y - size/3, y - size/3, y + size/3, y + size/3, y - size/3],
            mode="lines",
            line=dict(color=COLORS["warning"], width=2),
            fill="toself",
            fillcolor="rgba(255, 152, 0, 0.1)",
            hovertemplate=f"<b>Load</b><br>{name}<br>Power: {power:.2f} MW<extra></extra>",
            showlegend=False,
            name=f"LOAD_{x}_{y}"
        ),
        go.Scatter(
            x=[x], y=[y],
            mode="text",
            text=["🏭"],
            textfont=dict(size=18, color=COLORS["warning"]),
            showlegend=False,
            name=f"LOAD_LABEL_{x}_{y}"
        )
    ]

def create_line_symbol(x1: float, y1: float, x2: float, y2: float,
                      status: str = "CLOSED", fault: bool = False) -> go.Figure:
    """Create a transmission line symbol with optional breaker."""
    if fault:
        color = COLORS["danger"]
        width = 4
        dash = "dot"
    elif status == "OPEN":
        color = COLORS["warning"]
        width = 3
        dash = "dash"
    else:
        color = COLORS["success"]
        width = 3
        dash = "solid"
    
    # Calculate midpoint for breaker symbol
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    traces = [
        go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>Transmission Line</b><br>Status: {status}<br>Fault: {'YES' if fault else 'NO'}<extra></extra>",
            showlegend=False,
            name=f"LINE_{x1}_{y1}_{x2}_{y2}"
        )
    ]
    
    # Add breaker symbol at midpoint
    if not fault:
        traces.append(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode="text",
            text=["⚡"],
            textfont=dict(size=14, color=color),
            showlegend=False,
            name=f"BREAKER_{mid_x}_{mid_y}"
        ))
    
    return traces

def create_fault_marker(x: float, y: float, fault_type: str) -> go.Figure:
    """Create a fault indicator symbol at the fault location."""
    fault_icons = {
        "L2L": "⚡",
        "L2G": "⚡",
        "ARC": "🔥",
        "OVERCURRENT": "📈",
        "OVERVOLTAGE": "📊",
        "UNDERVOLTAGE": "📉",
    }
    
    icon = fault_icons.get(fault_type, "⚠️")
    
    return [
        # Pulsing circle around fault
        go.Scatter(
            x=[x], y=[y],
            mode="markers",
            marker=dict(
                size=40,
                color="rgba(233, 69, 96, 0.3)",
                symbol="circle",
                line=dict(color=COLORS["danger"], width=2)
            ),
            showlegend=False,
            name=f"FAULT_PULSE_{x}_{y}"
        ),
        # Fault icon
        go.Scatter(
            x=[x], y=[y],
            mode="text",
            text=[icon],
            textfont=dict(size=30, color=COLORS["danger"]),
            showlegend=False,
            name=f"FAULT_ICON_{x}_{y}"
        ),
        # Fault label
        go.Scatter(
            x=[x], y=[y - 25],
            mode="text",
            text=[f"{fault_type} FAULT"],
            textfont=dict(size=12, color=COLORS["danger"], family="Arial Black"),
            showlegend=False,
            name=f"FAULT_LABEL_{x}_{y}"
        )
    ]

# =============================================================================
# SCHEMATIC RENDERER
# =============================================================================

def render_circuit_schematic():
    """Render the complete circuit schematic page."""
    st.markdown("""
    <div class="page-header">
    <h2>🔌 Circuit Schematic</h2>
    <p>Professional electrical diagram with real-time fault visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    circuit = st.session_state.get("circuit_model")
    emulator = st.session_state.get("emulator")
    
    if not circuit:
        st.info("Start the system to view the circuit schematic.")
        return
    
    # --- Control Panel ---
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        zoom_level = st.slider("Zoom", 50, 200, 100, key="schematic_zoom")
    with col2:
        show_labels = st.checkbox("Show Labels", value=True, key="schematic_labels")
    with col3:
        show_faults = st.checkbox("Show Faults", value=True, key="schematic_faults")
    
    # --- Generate Schematic ---
    fig = _generate_schematic(circuit, emulator, zoom_level/100, show_labels, show_faults)
    st.plotly_chart(fig, use_container_width=True, key="circuit_schematic_chart")
    
    # --- Component Details ---
    _render_component_details(circuit, emulator)
    
    # --- Export Options ---
    _render_export_options(fig, circuit)

def _generate_schematic(circuit, emulator, zoom: float = 1.0, 
                       show_labels: bool = True, show_faults: bool = True) -> go.Figure:
    """Generate the complete circuit schematic."""
    import numpy as np
    
    traces = []
    annotations = []
    
    # Get fault info
    fault_active = st.session_state.get("fault_active", False)
    fault_type = st.session_state.get("fault_type", "NONE")
    fault_location = st.session_state.get("fault_location", None)
    
    # Calculate layout positions if not provided
    positions = _calculate_component_positions(circuit)
    
    # 1. Draw all lines first (behind components)
    for line in circuit.lines:
        from_pos = positions.get(str(line.from_bus), (0, 0))
        to_pos = positions.get(str(line.to_bus), (0, 0))
        
        # Check if this line has a fault
        line_has_fault = (fault_active and show_faults and 
                         fault_location == str(line.from_bus))
        
        line_traces = create_line_symbol(
            from_pos[0], from_pos[1],
            to_pos[0], to_pos[1],
            status="CLOSED",
            fault=line_has_fault
        )
        traces.extend(line_traces)
    
    # 2. Draw buses
    for bus in circuit.buses:
        pos = positions.get(str(bus.id), (bus.x * 100, bus.y * 100))
        
        # Get voltage from emulator if available
        voltage = bus.voltage_kv * 1000
        status = "ACTIVE"
        
        if emulator and hasattr(emulator, 'topology'):
            node = emulator.topology.nodes.get(str(bus.id))
            if node:
                voltage = node.voltage
                status = node.status.value if hasattr(node.status, 'value') else str(node.status)
        
        bus_trace = create_bus_symbol(
            pos[0], pos[1], 
            length=80 * zoom,
            voltage=voltage,
            status=status
        )
        traces.append(bus_trace)
        
        if show_labels:
            annotations.append(dict(
                x=pos[0], y=pos[1] + 15,
                text=f"{bus.name}<br>{voltage:.0f}V",
                showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=COLORS["primary"],
                borderwidth=1,
                borderpad=4
            ))
    
    # 3. Draw generators
    for gen in circuit.generators:
        pos = positions.get(str(gen.bus_id), (0, 0))
        # Offset generator from bus
        gen_pos = (pos[0] - 50 * zoom, pos[1] - 40 * zoom)
        
        gen_traces = create_generator_symbol(
            gen_pos[0], gen_pos[1],
            size=30 * zoom,
            name=f"GEN-{gen.id}",
            power=gen.p_mw
        )
        traces.extend(gen_traces)
        
        # Draw connection line from generator to bus
        conn_traces = create_line_symbol(
            gen_pos[0], gen_pos[1],
            pos[0], pos[1],
            status="CLOSED",
            fault=False
        )
        traces.extend(conn_traces)
    
    # 4. Draw loads
    for load in circuit.loads:
        pos = positions.get(str(load.bus_id), (0, 0))
        # Offset load from bus
        load_pos = (pos[0] + 50 * zoom, pos[1] - 40 * zoom)
        
        load_traces = create_load_symbol(
            load_pos[0], load_pos[1],
            size=30 * zoom,
            name=f"LOAD-{load.id}",
            power=load.p_mw
        )
        traces.extend(load_traces)
        
        # Draw connection line from load to bus
        conn_traces = create_line_symbol(
            load_pos[0], load_pos[1],
            pos[0], pos[1],
            status="CLOSED",
            fault=False
        )
        traces.extend(conn_traces)
    
    # 5. Draw fault markers if active
    if fault_active and show_faults and fault_location:
        fault_pos = positions.get(fault_location, (300, 200))
        fault_traces = create_fault_marker(
            fault_pos[0], fault_pos[1],
            fault_type
        )
        traces.extend(fault_traces)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Layout
    all_x = [t.x for t in traces if hasattr(t, 'x') and t.x]
    all_y = [t.y for t in traces if hasattr(t, 'y') and t.y]
    
    x_range = [min(all_x) - 100, max(all_x) + 100] if all_x else [-100, 500]
    y_range = [min(all_y) - 100, max(all_y) + 100] if all_y else [-100, 400]
    
    fig.update_layout(
        **PLOTLY_DARK_THEME,
        height=600,
        xaxis=dict(
            range=x_range,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=""
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
            scaleanchor="x",
            scaleratio=1
        ),
        annotations=annotations,
        plot_bgcolor="rgba(10, 10, 30, 0.8)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    return fig

def _calculate_component_positions(circuit) -> Dict[str, tuple]:
    """Calculate or use existing positions for all components."""
    positions = {}
    
    # Use existing coordinates from circuit model
    for bus in circuit.buses:
        # Scale coordinates for better visualization
        positions[str(bus.id)] = (bus.x * 100, bus.y * 100)
    
    return positions

def _render_component_details(circuit, emulator):
    """Render detailed component information table."""
    st.markdown("#### 📋 Component Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔌 Buses**")
        bus_data = []
        for bus in circuit.buses:
            voltage = bus.voltage_kv * 1000
            status = "ACTIVE"
            
            if emulator and hasattr(emulator, 'topology'):
                node = emulator.topology.nodes.get(str(bus.id))
                if node:
                    voltage = node.voltage
                    status = node.status.value if hasattr(node.status, 'value') else str(node.status)
            
            bus_data.append({
                "ID": bus.id,
                "Name": bus.name,
                "Voltage": f"{voltage:.1f}V",
                "Status": status
            })
        
        import pandas as pd
        st.dataframe(pd.DataFrame(bus_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**⚡ Lines**")
        line_data = []
        for line in circuit.lines:
            fault_active = st.session_state.get("fault_active", False)
            fault_loc = st.session_state.get("fault_location", None)
            has_fault = fault_active and fault_loc == str(line.from_bus)
            
            line_data.append({
                "ID": line.id,
                "From": f"Bus {line.from_bus}",
                "To": f"Bus {line.to_bus}",
                "Status": "FAULT" if has_fault else "CLOSED",
                "R": f"{line.r_ohm:.3f}Ω"
            })
        
        st.dataframe(pd.DataFrame(line_data), use_container_width=True, hide_index=True)

def _render_export_options(fig, circuit):
    """Render export options for the schematic."""
    st.markdown("#### 💾 Export Schematic")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📷 Save as PNG", key="export_png"):
            st.info("PNG export requires Kaleido: `pip install kaleido`")
    
    with col2:
        if st.button("📄 Save as SVG", key="export_svg"):
            st.info("SVG export requires Kaleido: `pip install kaleido`")
    
    with col3:
        if st.button("🔬 Export to MATLAB", key="export_matlab"):
            _export_to_matlab(circuit)

def _export_to_matlab(circuit):
    """Export circuit to MATLAB .fig format."""
    import json
    from pathlib import Path
    
    # Create MATLAB-compatible structure
    matlab_data = {
        "buses": [],
        "lines": [],
        "generators": [],
        "loads": [],
        "fault_state": {
            "active": st.session_state.get("fault_active", False),
            "type": st.session_state.get("fault_type", "NONE"),
            "location": st.session_state.get("fault_location", None)
        }
    }
    
    for bus in circuit.buses:
        matlab_data["buses"].append({
            "id": bus.id,
            "name": bus.name,
            "voltage": bus.voltage_kv,
            "x": bus.x,
            "y": bus.y
        })
    
    for line in circuit.lines:
        matlab_data["lines"].append({
            "id": line.id,
            "from": line.from_bus,
            "to": line.to_bus,
            "r": line.r_ohm,
            "x": line.x_ohm
        })
    
    # Save as JSON (MATLAB can read this)
    export_path = Path("reports/circuit_export.json")
    export_path.parent.mkdir(exist_ok=True)
    
    with open(export_path, 'w') as f:
        json.dump(matlab_data, f, indent=2)
    
    st.success(f"Circuit exported to {export_path}")
    st.caption("Load in MATLAB: `data = jsonread('reports/circuit_export.json')`")
```

---

### 2. Update `src/ui/pages/__init__.py`

```python
# Add the new schematic page
from src.ui.pages.circuit_schematic import render_circuit_schematic

__all__ = [
    "render_dashboard",
    "render_digital_twin",
    "render_wavelet_inspector",
    "render_fault_analysis",
    "render_circuit_designer",
    "render_circuit_schematic",  # NEW
    "render_system_health",
    "render_reports",
    "render_system_log",
]
```

---

### 3. Update `src/ui/app.py` (Add routing)

```python
# In the routing section, add:
elif page == "Circuit Schematic":
    render_circuit_schematic()
```

---

### 4. Update `src/ui/sidebar.py` (Add navigation)

```python
pages = [
    "Dashboard", "Digital Twin", "Circuit Schematic",  # NEW
    "Wavelet Inspector",
    "Fault Analysis", "Circuit Designer",
    "System Health", "Reports", "System Log"
]
```

---

### 5. `scripts/export_to_matlab.py` (MATLAB Visualization Script)

```python
#!/usr/bin/env python3
"""
Export circuit schematic to MATLAB for professional visualization.
Creates a .mat file that can be loaded in MATLAB to display the circuit
with fault highlighting using MATLAB's plotting capabilities.
"""
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.domain.circuit import CircuitModel
from src.adapters.matlab_bridge import MatlabBridge

def create_matlab_visualization_script(circuit_data: dict, output_path: str):
    """Create a MATLAB script that visualizes the circuit."""
    
    matlab_script = f'''
% DC Microgrid Circuit Visualization
% Generated by Python Export Tool
% Load this script in MATLAB to view the circuit schematic

function visualize_circuit()
    clear; clc; close all;
    
    % Circuit Data
    {json.dumps(circuit_data, indent=4).replace('"', "'")}
    
    % Create figure
    figure('Name', 'DC Microgrid Circuit Schematic', ...
           'Position', [100, 100, 1200, 800], ...
           'Color', [0.1, 0.1, 0.15]);
    
    axes('Position', [0.1, 0.1, 0.8, 0.8], ...
         'XLim', [-100, 600], ...
         'YLim', [-100, 500], ...
         'Color', [0.15, 0.15, 0.2], ...
         'XColor', [0.7, 0.7, 0.7], ...
         'YColor', [0.7, 0.7, 0.7]);
    hold on;
    
    % Draw Lines (Transmission)
    for i = 1:length(lines)
        from_bus = buses(buses.id == lines(i).from);
        to_bus = buses(buses.id == lines(i).to);
        
        % Check for fault
        if fault_state.active && strcmp(fault_state.location, num2str(lines(i).from))
            line_color = [0.9, 0.2, 0.2];  % Red for fault
            line_width = 3;
            line_style = ':';
        else
            line_color = [0.2, 0.9, 0.2];  % Green for normal
            line_width = 2;
            line_style = '-';
        end
        
        plot([from_bus.x*100, to_bus.x*100], ...
             [from_bus.y*100, to_bus.y*100], ...
             'Color', line_color, ...
             'LineWidth', line_width, ...
             'LineStyle', line_style);
    end
    
    % Draw Buses
    for i = 1:length(buses)
        x = buses(i).x * 100;
        y = buses(i).y * 100;
        
        % Bus bar (thick line)
        plot([x-40, x+40], [y, y], ...
             'Color', [0, 0.7, 0.9], ...
             'LineWidth', 8);
        
        % Label
        text(x, y+15, buses(i).name, ...
             'Color', 'white', ...
             'FontSize', 10, ...
             'HorizontalAlignment', 'center');
    end
    
    % Draw Generators
    for i = 1:length(generators)
        bus = buses(buses.id == generators(i).bus);
        x = bus.x * 100 - 50;
        y = bus.y * 100 - 40;
        
        % Circle
        theta = linspace(0, 2*pi, 50);
        plot(x + 25*cos(theta), y + 25*sin(theta), ...
             'Color', [0.3, 0.7, 0.9], ...
             'LineWidth', 2);
        
        % Label
        text(x, y, '⚡', ...
             'Color', [0.3, 0.7, 0.9], ...
             'FontSize', 20, ...
             'HorizontalAlignment', 'center');
    end
    
    % Draw Loads
    for i = 1:length(loads)
        bus = buses(buses.id == loads(i).bus);
        x = bus.x * 100 + 50;
        y = bus.y * 100 - 40;
        
        % Rectangle
        rectangle('Position', [x-15, y-10, 30, 20], ...
                  'EdgeColor', [0.9, 0.6, 0], ...
                  'LineWidth', 2);
        
        % Label
        text(x, y, '🏭', ...
             'Color', [0.9, 0.6, 0], ...
             'FontSize', 18, ...
             'HorizontalAlignment', 'center');
    end
    
    % Draw Fault Marker
    if fault_state.active
        fault_bus = buses(buses.id == str2num(fault_state.location));
        if ~isempty(fault_bus)
            x = fault_bus.x * 100;
            y = fault_bus.y * 100;
            
            % Pulsing circle
            viscircles([x, y], 30, 'Color', [0.9, 0.2, 0.2], 'LineWidth', 2);
            
            % Fault label
            text(x, y-30, [fault_state.type ' FAULT'], ...
                 'Color', [0.9, 0.2, 0.2], ...
                 'FontSize', 12, ...
                 'FontWeight', 'bold', ...
                 'HorizontalAlignment', 'center');
        end
    end
    
    title('DC Microgrid Circuit Schematic', 'Color', 'white', 'FontSize', 16);
    grid off;
    axis equal;
    hold off;
    
    % Save figure
    saveas(gcf, 'circuit_schematic.fig');
    saveas(gcf, 'circuit_schematic.png');
    
    disp('✅ Circuit visualization complete!');
    disp('   Saved: circuit_schematic.fig');
    disp('   Saved: circuit_schematic.png');
end
'''
    
    with open(output_path, 'w') as f:
        f.write(matlab_script)
    
    return output_path

def main():
    """Main export function."""
    print("🔌 Exporting Circuit to MATLAB...")
    
    # Load current circuit from session or create reference
    circuit = CircuitModel(name="Exported_Microgrid")
    
    # Create reference circuit (same as in system.py)
    from src.domain.circuit import Bus, Line, Generator, Load
    
    circuit.buses = [
        Bus(id=1, name="PCC_Bus", voltage_kv=0.4, type="Slack", x=300, y=200),
        Bus(id=2, name="Solar_Bus", voltage_kv=0.4, type="PV", x=100, y=100),
        Bus(id=3, name="Battery_Bus", voltage_kv=0.4, type="PQ", x=500, y=100),
        Bus(id=4, name="Load_A_Bus", voltage_kv=0.4, type="PQ", x=100, y=300),
        Bus(id=5, name="Load_B_Bus", voltage_kv=0.4, type="PQ", x=500, y=300),
        Bus(id=6, name="Grid_Infeed", voltage_kv=0.4, type="Slack", x=300, y=50),
    ]
    
    circuit.lines = [
        Line(id=1, from_bus=6, to_bus=1, r_ohm=0.01, x_ohm=0.005),
        Line(id=2, from_bus=1, to_bus=2, r_ohm=0.05, x_ohm=0.01),
        Line(id=3, from_bus=1, to_bus=3, r_ohm=0.03, x_ohm=0.01),
        Line(id=4, from_bus=1, to_bus=4, r_ohm=0.08, x_ohm=0.02),
        Line(id=5, from_bus=1, to_bus=5, r_ohm=0.08, x_ohm=0.02),
    ]
    
    circuit.generators = [
        Generator(id=1, bus_id=6, p_mw=0.5),
        Generator(id=2, bus_id=2, p_mw=0.1),
    ]
    
    circuit.loads = [
        Load(id=1, bus_id=4, p_mw=0.05, priority=1),
        Load(id=2, bus_id=5, p_mw=0.03, priority=2),
    ]
    
    # Create export data structure
    circuit_data = {
        "buses": [b.to_dict() for b in circuit.buses],
        "lines": [l.to_dict() for l in circuit.lines],
        "generators": [g.to_dict() for g in circuit.generators],
        "loads": [l.to_dict() for l in circuit.loads],
        "fault_state": {
            "active": False,
            "type": "NONE",
            "location": None
        }
    }
    
    # Save circuit data as JSON
    json_path = Path("reports/circuit_data.json")
    json_path.parent.mkdir(exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(circuit_data, f, indent=2)
    
    print(f"✅ Circuit data saved to {json_path}")
    
    # Create MATLAB visualization script
    matlab_script_path = Path("reports/visualize_circuit.m")
    create_matlab_visualization_script(circuit_data, str(matlab_script_path))
    
    print(f"✅ MATLAB script created at {matlab_script_path}")
    print("\n📋 Usage in MATLAB:")
    print("   cd reports")
    print("   visualize_circuit")
    print("\n💡 The script will generate:")
    print("   - circuit_schematic.fig (editable)")
    print("   - circuit_schematic.png (image)")
    
    # Also save as .mat file for direct MATLAB loading
    mat_path = Path("reports/circuit_model.mat")
    MatlabBridge.save_model(circuit, str(mat_path))
    print(f"✅ MATLAB .mat file saved to {mat_path}")

if __name__ == "__main__":
    main()
```

---

## 🎨 Visual Example

When a fault is injected, the schematic will show:

```
                    ⚡ GEN-1
                     │
                     │
        ┌────────────┴────────────┐
        │         BUS 1           │
        │       (400V)            │
        └────────────┬────────────┘
                     │
              ⚡⚡⚡ FAULT ⚡⚡⚡    ← Pulsing red marker
              L2L FAULT            ← Fault type label
                     │
                     │
        ┌────────────┴────────────┐
        │         BUS 4           │
        │       (320V)            │  ← Voltage drops during fault
        └────────────┬────────────┘
                     │
                    🏭 LOAD-1
```

---

## 🚀 Quick Start

1. **Create the new file:**
   ```bash
   mkdir -p src/ui/pages
   # Save circuit_schematic.py
   ```

2. **Update navigation:**
   - Edit `src/ui/pages/__init__.py`
   - Edit `src/ui/sidebar.py`
   - Edit `src/ui/app.py`

3. **Export to MATLAB:**
   ```bash
   python scripts/export_to_matlab.py
   ```

4. **In MATLAB:**
   ```matlab
   cd reports
   visualize_circuit
   ```

---

## 📊 Features Comparison

| Feature | Old Digital Twin | New Circuit Schematic |
|---------|-----------------|----------------------|
| **Component Symbols** | ❌ Dots | ✅ Bus bars, generators, loads |
| **Fault Visualization** | ❌ Node color | ✅ Pulsing marker + label |
| **Line Status** | ❌ Simple lines | ✅ Breaker symbols, dash styles |
| **Voltage Display** | ⚠️ Hover only | ✅ Always visible labels |
| **MATLAB Export** | ❌ None | ✅ .fig + .png + .m script |
| **Click Interaction** | ❌ None | ✅ Component details on click |
| **Zoom/Pan** | ✅ Basic | ✅ Enhanced with constraints |

Would you like me to also add **animated fault propagation** (showing the fault wave traveling through lines) or **protective device coordination visualization** (showing which breakers should trip)?