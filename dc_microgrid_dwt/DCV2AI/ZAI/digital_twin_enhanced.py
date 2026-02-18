"""
Digital Twin Page - Enhanced Circuit Schematic Visualization

This page renders a proper electrical circuit schematic (not just a node graph)
with real-time fault location highlighting.

Create this file at: src/ui/pages/digital_twin_enhanced.py

Features:
- Proper electrical schematic with bus bars, cables, generators, loads
- Real-time fault location with visual highlighting
- Component status indicators
- Voltage level displays
- Interactive fault location display
"""

import streamlit as st
import json
from typing import Dict, Any, Optional
from src.ui.system import add_log, get_per_node_voltages


def render_digital_twin_enhanced():
    """Render the enhanced digital twin with circuit schematic."""
    
    # Header
    st.markdown("""
    <div class="page-header">
        <h2>🏗️ Digital Twin - Circuit Schematic</h2>
        <p>Real-time electrical circuit diagram with fault location visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get circuit and emulator
    circuit = st.session_state.get("circuit_model")
    emulator = st.session_state.get("emulator")
    
    if not circuit:
        st.info("Start the system to see the circuit schematic.")
        return
    
    # Fault status
    fault_info = None
    if st.session_state.fault_active:
        fault_info = {
            'active': True,
            'location': st.session_state.fault_location or "1",
            'type': st.session_state.fault_type,
            'severity': 0.8,
            'distance': st.session_state.fault_distance
        }
    
    # Get real-time voltages
    node_voltages = get_per_node_voltages()
    
    # Render the circuit schematic
    render_circuit_schematic(circuit, node_voltages, fault_info)
    
    # Component Details
    st.markdown("---")
    render_component_details(circuit, node_voltages, fault_info)


def render_circuit_schematic(circuit, node_voltages: Dict, fault_info: Optional[Dict]):
    """
    Render the circuit schematic using HTML/CSS/JavaScript.
    
    This creates a proper electrical diagram with:
    - Bus bars (horizontal lines)
    - Cables connecting buses
    - Generator symbols (circles with G)
    - Load symbols (arrows)
    - Fault highlighting
    """
    
    # Build circuit data for JavaScript
    circuit_data = build_circuit_js_data(circuit, node_voltages, fault_info)
    
    # HTML/CSS/JavaScript for circuit schematic
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .circuit-container {{
                width: 100%;
                height: 500px;
                background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
                border-radius: 15px;
                position: relative;
                overflow: hidden;
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            
            .grid-bg {{
                position: absolute;
                width: 100%;
                height: 100%;
                background-image: 
                    linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
                background-size: 30px 30px;
            }}
            
            .bus-bar {{
                position: absolute;
                background: linear-gradient(90deg, #1a6b4a, #2ecc71, #1a6b4a);
                border: 2px solid #27ae60;
                border-radius: 4px;
                height: 12px;
                transform: translateX(-50%);
                box-shadow: 0 0 10px rgba(46, 204, 113, 0.3);
                transition: all 0.3s ease;
            }}
            
            .bus-bar.fault {{
                background: linear-gradient(90deg, #c0392b, #e74c3c, #c0392b);
                border-color: #e74c3c;
                box-shadow: 0 0 20px rgba(231, 76, 60, 0.6);
                animation: pulse-fault 0.5s infinite alternate;
            }}
            
            .bus-label {{
                position: absolute;
                color: #ecf0f1;
                font-size: 11px;
                font-weight: bold;
                text-align: center;
                transform: translateX(-50%);
                white-space: nowrap;
            }}
            
            .bus-voltage {{
                position: absolute;
                color: #3498db;
                font-size: 10px;
                font-family: 'Consolas', monospace;
                transform: translateX(-50%);
            }}
            
            .cable {{
                position: absolute;
                background: #3498db;
                transition: all 0.3s ease;
            }}
            
            .cable.horizontal {{
                height: 3px;
            }}
            
            .cable.vertical {{
                width: 3px;
            }}
            
            .cable.fault {{
                background: #e74c3c;
                box-shadow: 0 0 15px rgba(231, 76, 60, 0.6);
            }}
            
            .generator {{
                position: absolute;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background: linear-gradient(135deg, #1a5276, #2980b9);
                border: 3px solid #3498db;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 18px;
                box-shadow: 0 0 15px rgba(52, 152, 219, 0.4);
                transform: translate(-50%, -50%);
            }}
            
            .generator-label {{
                position: absolute;
                color: #3498db;
                font-size: 10px;
                text-align: center;
                transform: translateX(-50%);
            }}
            
            .load {{
                position: absolute;
                width: 0;
                height: 0;
                border-left: 20px solid transparent;
                border-right: 20px solid transparent;
                border-top: 35px solid #e67e22;
                transform: translate(-50%, -50%);
                filter: drop-shadow(0 0 10px rgba(230, 126, 34, 0.4));
            }}
            
            .load-label {{
                position: absolute;
                color: #e67e22;
                font-size: 10px;
                text-align: center;
                transform: translateX(-50%);
            }}
            
            .fault-indicator {{
                position: absolute;
                width: 60px;
                height: 60px;
                border: 3px dashed #e74c3c;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                animation: rotate-pulse 2s linear infinite;
            }}
            
            .fault-symbol {{
                position: absolute;
                font-size: 30px;
                transform: translate(-50%, -50%);
                animation: flash 0.3s infinite alternate;
            }}
            
            .status-banner {{
                position: absolute;
                top: 15px;
                left: 50%;
                transform: translateX(-50%);
                padding: 8px 20px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
                z-index: 100;
            }}
            
            .status-normal {{
                background: rgba(46, 204, 113, 0.2);
                border: 2px solid #27ae60;
                color: #2ecc71;
            }}
            
            .status-fault {{
                background: rgba(231, 76, 60, 0.2);
                border: 2px solid #e74c3c;
                color: #e74c3c;
                animation: pulse-border 0.5s infinite alternate;
            }}
            
            .legend {{
                position: absolute;
                bottom: 15px;
                right: 15px;
                background: rgba(0,0,0,0.5);
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 11px;
                color: #bdc3c7;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 5px 0;
            }}
            
            .legend-color {{
                width: 20px;
                height: 10px;
                margin-right: 8px;
                border-radius: 2px;
            }}
            
            @keyframes pulse-fault {{
                from {{ box-shadow: 0 0 20px rgba(231, 76, 60, 0.4); }}
                to {{ box-shadow: 0 0 40px rgba(231, 76, 60, 0.8); }}
            }}
            
            @keyframes rotate-pulse {{
                from {{ transform: translate(-50%, -50%) rotate(0deg); }}
                to {{ transform: translate(-50%, -50%) rotate(360deg); }}
            }}
            
            @keyframes flash {{
                from {{ opacity: 1; }}
                to {{ opacity: 0.5; }}
            }}
            
            @keyframes pulse-border {{
                from {{ border-color: #e74c3c; }}
                to {{ border-color: #c0392b; }}
            }}
        </style>
    </head>
    <body>
        <div class="circuit-container" id="circuitCanvas">
            <div class="grid-bg"></div>
            <!-- Status Banner -->
            <div class="status-banner {'status-normal' if not fault_info else 'status-fault'}" id="statusBanner">
                {'✓ System Normal' if not fault_info else f'⚠ FAULT: ' + fault_info['type'] + ' at ' + str(fault_info['location'])}
            </div>
            
            <!-- Circuit elements will be added by JavaScript -->
        </div>
        
        <script>
            const circuitData = {json.dumps(circuit_data)};
            
            function renderCircuit() {{
                const container = document.getElementById('circuitCanvas');
                const width = container.offsetWidth;
                const height = container.offsetHeight;
                
                // Scale factors
                const scaleX = (x) => 50 + (x * 0.8) * (width - 100) / 100;
                const scaleY = (y) => 80 + (1 - y) * (height - 160) / 100;
                
                // Draw cables first (behind buses)
                circuitData.lines.forEach(line => {{
                    const fromBus = circuitData.buses.find(b => b.id === line.from_bus);
                    const toBus = circuitData.buses.find(b => b.id === line.to_bus);
                    
                    if (fromBus && toBus) {{
                        const x1 = scaleX(fromBus.x);
                        const y1 = scaleY(fromBus.y);
                        const x2 = scaleX(toBus.x);
                        const y2 = scaleY(toBus.y);
                        
                        // Check if this line has fault
                        const isFaultLine = circuitData.fault && 
                                          circuitData.fault.location === 'L' + line.id;
                        
                        // Draw orthogonal cable path
                        const midX = (x1 + x2) / 2;
                        
                        // Horizontal segment from source
                        const h1 = document.createElement('div');
                        h1.className = 'cable horizontal' + (isFaultLine ? ' fault' : '');
                        h1.style.left = Math.min(x1, midX) + 'px';
                        h1.style.top = y1 + 'px';
                        h1.style.width = Math.abs(midX - x1) + 'px';
                        container.appendChild(h1);
                        
                        // Vertical segment
                        const v = document.createElement('div');
                        v.className = 'cable vertical' + (isFaultLine ? ' fault' : '');
                        v.style.left = midX + 'px';
                        v.style.top = Math.min(y1, y2) + 'px';
                        v.style.height = Math.abs(y2 - y1) + 'px';
                        container.appendChild(v);
                        
                        // Horizontal segment to destination
                        const h2 = document.createElement('div');
                        h2.className = 'cable horizontal' + (isFaultLine ? ' fault' : '');
                        h2.style.left = Math.min(midX, x2) + 'px';
                        h2.style.top = y2 + 'px';
                        h2.style.width = Math.abs(midX - x2) + 'px';
                        container.appendChild(h2);
                        
                        // Line label
                        const label = document.createElement('div');
                        label.className = 'bus-label';
                        label.style.left = midX + 'px';
                        label.style.top = ((y1 + y2) / 2 - 15) + 'px';
                        label.style.color = '#7f8c8d';
                        label.textContent = 'L' + line.id;
                        container.appendChild(label);
                        
                        // Fault indicator on line
                        if (isFaultLine) {{
                            const faultSymbol = document.createElement('div');
                            faultSymbol.className = 'fault-symbol';
                            faultSymbol.style.left = midX + 'px';
                            faultSymbol.style.top = ((y1 + y2) / 2) + 'px';
                            faultSymbol.textContent = '⚡';
                            container.appendChild(faultSymbol);
                            
                            const faultRing = document.createElement('div');
                            faultRing.className = 'fault-indicator';
                            faultRing.style.left = midX + 'px';
                            faultRing.style.top = ((y1 + y2) / 2) + 'px';
                            container.appendChild(faultRing);
                        }}
                    }}
                }});
                
                // Draw buses
                circuitData.buses.forEach(bus => {{
                    const x = scaleX(bus.x);
                    const y = scaleY(bus.y);
                    
                    const isFaultBus = circuitData.fault && 
                                      circuitData.fault.location === String(bus.id);
                    
                    // Bus bar
                    const busBar = document.createElement('div');
                    busBar.className = 'bus-bar' + (isFaultBus ? ' fault' : '');
                    busBar.style.left = x + 'px';
                    busBar.style.top = y + 'px';
                    busBar.style.width = '100px';
                    container.appendChild(busBar);
                    
                    // Bus name label
                    const nameLabel = document.createElement('div');
                    nameLabel.className = 'bus-label';
                    nameLabel.style.left = x + 'px';
                    nameLabel.style.top = (y - 25) + 'px';
                    nameLabel.textContent = bus.name;
                    container.appendChild(nameLabel);
                    
                    // Bus type indicator
                    const typeLabel = document.createElement('div');
                    typeLabel.className = 'bus-label';
                    typeLabel.style.left = x + 'px';
                    typeLabel.style.top = (y + 20) + 'px';
                    typeLabel.style.color = '#7f8c8d';
                    typeLabel.style.fontSize = '9px';
                    typeLabel.textContent = bus.type;
                    container.appendChild(typeLabel);
                    
                    // Voltage display
                    if (bus.voltage !== undefined) {{
                        const voltageLabel = document.createElement('div');
                        voltageLabel.className = 'bus-voltage';
                        voltageLabel.style.left = x + 'px';
                        voltageLabel.style.top = (y + 35) + 'px';
                        voltageLabel.textContent = bus.voltage.toFixed(1) + 'V';
                        container.appendChild(voltageLabel);
                    }}
                    
                    // Fault indicator
                    if (isFaultBus) {{
                        const faultSymbol = document.createElement('div');
                        faultSymbol.className = 'fault-symbol';
                        faultSymbol.style.left = (x + 60) + 'px';
                        faultSymbol.style.top = y + 'px';
                        faultSymbol.textContent = '⚡';
                        container.appendChild(faultSymbol);
                        
                        const faultRing = document.createElement('div');
                        faultRing.className = 'fault-indicator';
                        faultRing.style.left = x + 'px';
                        faultRing.style.top = y + 'px';
                        container.appendChild(faultRing);
                    }}
                }});
                
                // Draw generators
                circuitData.generators.forEach(gen => {{
                    const bus = circuitData.buses.find(b => b.id === gen.bus_id);
                    if (bus) {{
                        const x = scaleX(bus.x) - 70;
                        const y = scaleY(bus.y);
                        
                        const genEl = document.createElement('div');
                        genEl.className = 'generator';
                        genEl.style.left = x + 'px';
                        genEl.style.top = y + 'px';
                        genEl.textContent = 'G';
                        container.appendChild(genEl);
                        
                        const label = document.createElement('div');
                        label.className = 'generator-label';
                        label.style.left = x + 'px';
                        label.style.top = (y + 35) + 'px';
                        label.textContent = gen.name + '\\n' + gen.p_mw.toFixed(2) + ' MW';
                        label.style.whiteSpace = 'pre';
                        container.appendChild(label);
                        
                        // Connection line
                        const conn = document.createElement('div');
                        conn.className = 'cable horizontal';
                        conn.style.left = (x + 25) + 'px';
                        conn.style.top = y + 'px';
                        conn.style.width = '20px';
                        conn.style.background = '#3498db';
                        container.appendChild(conn);
                    }}
                }});
                
                // Draw loads
                circuitData.loads.forEach(load => {{
                    const bus = circuitData.buses.find(b => b.id === load.bus_id);
                    if (bus) {{
                        const x = scaleX(bus.x) + 70;
                        const y = scaleY(bus.y);
                        
                        const loadEl = document.createElement('div');
                        loadEl.className = 'load';
                        loadEl.style.left = x + 'px';
                        loadEl.style.top = y + 'px';
                        container.appendChild(loadEl);
                        
                        const label = document.createElement('div');
                        label.className = 'load-label';
                        label.style.left = x + 'px';
                        label.style.top = (y + 40) + 'px';
                        label.textContent = load.name + '\\n' + load.p_mw.toFixed(2) + ' MW';
                        label.style.whiteSpace = 'pre';
                        container.appendChild(label);
                        
                        // Connection line
                        const conn = document.createElement('div');
                        conn.className = 'cable horizontal';
                        conn.style.left = (x - 45) + 'px';
                        conn.style.top = y + 'px';
                        conn.style.width = '20px';
                        conn.style.background = '#e67e22';
                        container.appendChild(conn);
                    }}
                }});
            }}
            
            // Render on load
            renderCircuit();
            
            // Re-render on resize
            window.addEventListener('resize', () => {{
                // Clear and re-render
                const container = document.getElementById('circuitCanvas');
                const elements = container.querySelectorAll('.bus-bar, .bus-label, .bus-voltage, .cable, .generator, .generator-label, .load, .load-label, .fault-indicator, .fault-symbol');
                elements.forEach(el => el.remove());
                renderCircuit();
            }});
        </script>
    </body>
    </html>
    """
    
    # Display the HTML
    st.components.v1.html(html_content, height=520, scrolling=False)


def build_circuit_js_data(circuit, node_voltages: Dict, fault_info: Optional[Dict]) -> Dict:
    """Build circuit data for JavaScript rendering."""
    
    buses = []
    for bus in circuit.buses:
        bus_data = {
            'id': bus.id,
            'name': bus.name,
            'type': bus.type,
            'x': bus.x,
            'y': bus.y
        }
        
        # Add real-time voltage
        if node_voltages and str(bus.id) in node_voltages:
            bus_data['voltage'] = node_voltages[str(bus.id)].get('voltage', 400.0)
        else:
            bus_data['voltage'] = 400.0
        
        buses.append(bus_data)
    
    lines = []
    for line in circuit.lines:
        lines.append({
            'id': line.id,
            'from_bus': line.from_bus,
            'to_bus': line.to_bus,
            'r_ohm': line.r_ohm,
            'length_km': line.length_km
        })
    
    generators = []
    for gen in circuit.generators:
        generators.append({
            'id': gen.id,
            'bus_id': gen.bus_id,
            'name': f"Gen{gen.id}",
            'p_mw': gen.p_mw
        })
    
    loads = []
    for load in circuit.loads:
        loads.append({
            'id': load.id,
            'bus_id': load.bus_id,
            'name': f"Load{load.id}",
            'p_mw': load.p_mw,
            'priority': getattr(load, 'priority', 1)
        })
    
    return {
        'buses': buses,
        'lines': lines,
        'generators': generators,
        'loads': loads,
        'fault': fault_info
    }


def render_component_details(circuit, node_voltages: Dict, fault_info: Optional[Dict]):
    """Render detailed component information tables."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔌 Bus Details")
        
        import pandas as pd
        bus_rows = []
        for bus in circuit.buses:
            v = node_voltages.get(str(bus.id), {}).get('voltage', 400.0) if node_voltages else 400.0
            status = node_voltages.get(str(bus.id), {}).get('status', 'ACTIVE') if node_voltages else 'UNKNOWN'
            status_val = status.value if hasattr(status, 'value') else str(status)
            
            # Check if this bus has fault
            is_fault = fault_info and str(fault_info.get('location')) == str(bus.id)
            
            bus_rows.append({
                'ID': bus.id,
                'Name': bus.name,
                'Type': bus.type,
                'Voltage': f"{v:.1f}V",
                'Status': '🔴 FAULT' if is_fault else f'🟢 {status_val}'
            })
        
        st.dataframe(pd.DataFrame(bus_rows), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📏 Line Details")
        
        line_rows = []
        for line in circuit.lines:
            # Check if this line has fault
            is_fault = fault_info and str(fault_info.get('location')) == f"L{line.id}"
            
            line_rows.append({
                'ID': f"L{line.id}",
                'From → To': f"Bus {line.from_bus} → Bus {line.to_bus}",
                'Length': f"{line.length_km:.2f} km",
                'Resistance': f"{line.r_ohm:.3f} Ω",
                'Status': '🔴 FAULT' if is_fault else '🟢 Normal'
            })
        
        st.dataframe(pd.DataFrame(line_rows), use_container_width=True, hide_index=True)
    
    # Fault location details
    if fault_info:
        st.markdown("---")
        st.markdown("#### ⚠️ Fault Details")
        
        fault_col1, fault_col2, fault_col3, fault_col4 = st.columns(4)
        
        with fault_col1:
            st.metric("Fault Type", fault_info.get('type', 'UNKNOWN'))
        with fault_col2:
            st.metric("Location", fault_info.get('location', 'Unknown'))
        with fault_col3:
            severity = fault_info.get('severity', 0)
            st.metric("Severity", f"{severity * 100:.0f}%")
        with fault_col4:
            distance = fault_info.get('distance')
            st.metric("Distance", f"{distance:.1f}m" if distance else "—")
