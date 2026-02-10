"""
DC Microgrid Guardian Pro - Industrial Fault Detection Platform

Main Streamlit application with multi-page dashboard featuring:
- Real-time monitoring dashboard
- Digital twin visualization  
- Wavelet analysis inspector
- Fault timeline (black box)
- AI diagnosis panel
- Report generation
- System health monitoring
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import queue
import threading
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.framework.bus import EventBus
from src.framework.registry import AgentRegistry
from src.domain.events import (
    VoltageSampleEvent, DWTResultEvent, ProcessingResultEvent,
    SystemTripEvent, HealthStatusEvent, AIAnalysisEvent,
    FaultInjectionEvent, ConverterStatusEvent, FaultLocationEvent
)
from src.adapters.grid_emulator import GridEmulator
from src.agents.ingestion.sampler import SamplerAgent
from src.agents.ingestion.window_manager import WindowManagerAgent
from src.agents.processing.dwt_engine import DWTEngineAgent
from src.agents.processing.detail_analyzer import DetailAnalyzerAgent
from src.agents.detection.threshold_guard import ThresholdGuardAgent
from src.agents.detection.fault_voter import FaultVoterAgent
from src.agents.processing.fault_locator import PreciseFaultLocatorAgent
from src.agents.control.trip_sequencer import TripSequencerAgent
from src.agents.supervision.health_monitor import HealthMonitorAgent
from src.agents.supervision.ai_classifier import AIClassifierAgent
from src.agents.supervision.replay_recorder import ReplayRecorderAgent
from src.agents.supervision.report_generator import ReportGeneratorAgent
from src.domain.circuit import CircuitModel
from src.adapters.matlab_bridge import MatlabBridge

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="DC Microgrid Guardian Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS STYLING
# ==============================================================================

st.markdown("""
<style>
    /* Dark Professional Theme */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #4CAF50;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    .metric-card-fault {
        border-left: 4px solid #FF4B4B;
    }
    
    .metric-card-warning {
        border-left: 4px solid #FFA726;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        text-transform: uppercase;
    }
    
    .badge-normal {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }
    
    .badge-fault {
        background: linear-gradient(135deg, #e94560, #FF4B4B);
        color: white;
        animation: pulse 1s infinite;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #FFA726, #FF9800);
        color: white;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    
    /* Alert Box */
    .alert-box {
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    
    .alert-critical {
        background: rgba(233, 69, 96, 0.2);
        border-left: 4px solid #e94560;
        color: #ff6b6b;
    }
    
    .alert-warning {
        background: rgba(255, 167, 38, 0.2);
        border-left: 4px solid #FFA726;
        color: #ffd93d;
    }
    
    .alert-info {
        background: rgba(76, 175, 80, 0.2);
        border-left: 4px solid #4CAF50;
        color: #82e0aa;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-bottom: 3px solid #e94560;
    }
    
    /* Footer Health Bar */
    .health-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(26, 26, 46, 0.95);
        padding: 10px 20px;
        border-top: 2px solid #16213e;
        z-index: 999;
    }
    
    /* Control Panel */
    .control-panel {
        background: #16213e;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    /* Timeline Item */
    .timeline-item {
        padding: 10px 15px;
        border-left: 3px solid #4CAF50;
        margin-bottom: 8px;
        background: rgba(76, 175, 80, 0.1);
        border-radius: 0 8px 8px 0;
    }
    
    .timeline-item-fault {
        border-left-color: #e94560;
        background: rgba(233, 69, 96, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'system_running': False,
        'emulator': None,
        'bus': None,
        'registry': None,
        'data_queue': queue.Queue(),
        'system_status': 'IDLE',
        'voltage_history': [],
        'dwt_energy_history': [],
        'health_data': {'cpu': 0, 'memory': 0, 'eps': 0, 'latency': 0},
        'ai_diagnosis': None,
        'trip_events': [],
        'system_log': [],
        'last_fault_config': None,
        'timeline_data': [],
        'converter_state': {'duty': 0.45, 'mode': 'AUTO', 'target_v': 400},
        'wavelet_settings': {'family': 'db4', 'level': 4},
        'report_generator': None,
        'selected_node': None,
        'fault_location': None,
        'circuit_model': CircuitModel(), # Default empty model
        'component_history': {},
        'recording_enabled': False,
        'recording_data': {},
        'recordings': [],
        'replay_mode': False,
        'selected_recording': None,
        'last_telemetry_update': 0.0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==============================================================================
# SYSTEM CONTROL FUNCTIONS
# ==============================================================================

def start_system():
    """Initialize and start the full agent system."""
    if st.session_state.system_running:
        return
        
    # Create event bus and registry
    bus = EventBus()
    registry = AgentRegistry()
    emulator = GridEmulator()
    
    # Bridge events to UI queue - capture queue instance to avoid thread context issues
    # We must capture the queue object itself, not access st.session_state in the thread
    ui_queue = st.session_state.data_queue
    
    def bridge_to_ui(event):
        ui_queue.put(event)
    
    # Subscribe bridge to all relevant events
    bus.subscribe(VoltageSampleEvent, bridge_to_ui)
    bus.subscribe(DWTResultEvent, bridge_to_ui)
    bus.subscribe(ProcessingResultEvent, bridge_to_ui)
    bus.subscribe(SystemTripEvent, bridge_to_ui)
    bus.subscribe(HealthStatusEvent, bridge_to_ui)
    bus.subscribe(HealthStatusEvent, bridge_to_ui)
    bus.subscribe(AIAnalysisEvent, bridge_to_ui)
    bus.subscribe(FaultLocationEvent, bridge_to_ui)
    
    # Create agents
    agents = [
        WindowManagerAgent("WindowManager", bus),
        DWTEngineAgent("DWTEngine", bus),
        DetailAnalyzerAgent("DetailAnalyzer", bus),
        ThresholdGuardAgent("ThresholdGuard", bus, {"d1_peak_max": 100.0}),
        PreciseFaultLocatorAgent("FaultLocator", bus, {"emulator": emulator}),
        FaultVoterAgent("FaultVoter", bus),
        TripSequencerAgent("TripSequencer", bus),
        HealthMonitorAgent("HealthMonitor", bus),
        AIClassifierAgent("AIClassifier", bus),
        ReplayRecorderAgent("ReplayRecorder", bus),
        ReportGeneratorAgent("ReportGenerator", bus),
    ]
    
    # 7a. Create and Configure Sampler (The "ADC")
    sampler = SamplerAgent("Sampler", bus)
    sampler.set_sensor(emulator)
    agents.append(sampler)
    
    for agent in agents:
        registry.register(agent)
    
    registry.start_all()
    
    # Store in session state
    st.session_state.bus = bus
    st.session_state.registry = registry
    st.session_state.emulator = emulator
    st.session_state.system_running = True
    st.session_state.system_status = 'RUNNING'
    
    # Find report generator for later use
    for agent in agents:
        if isinstance(agent, ReportGeneratorAgent):
            st.session_state.report_generator = agent
            break
    
    add_log("INFO", "System started successfully")

def stop_system():
    """Stop the agent system."""
    if st.session_state.registry:
        st.session_state.registry.stop_all()
    st.session_state.system_running = False
    st.session_state.system_status = 'STOPPED'
    add_log("INFO", "System stopped")

def inject_fault(fault_type: str, severity: float, location: str = "BUS_DC", properties: Dict[str, Any] = None):
    """Inject a fault into the emulator."""
    if st.session_state.emulator:
        st.session_state.emulator.inject_fault(fault_type, severity, location, properties)
        st.session_state.last_fault_config = {
            'type': fault_type,
            'severity': severity,
            'location': location,
            'location': location,
            'time': time.time(),
            'properties': st.session_state.get('fault_properties', {})
        }
        add_log("WARNING", f"Fault injected: {fault_type} at {location} (severity: {severity:.1%}) properties={st.session_state.get('fault_properties')}")

def clear_fault():
    """Clear active fault."""
    if st.session_state.emulator:
        st.session_state.emulator.clear_fault()
        st.session_state.last_fault_config = None
        st.session_state.system_status = 'RUNNING'
        add_log("INFO", "Fault cleared")

def add_log(level: str, message: str):
    """Add entry to system log."""
    entry = {
        'time': datetime.now(),
        'level': level,
        'message': message
    }
    st.session_state.system_log.append(entry)
    if len(st.session_state.system_log) > 100:
        st.session_state.system_log = st.session_state.system_log[-100:]

# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def process_events():
    """Process events from the queue and update state."""
    processed = 0
    max_per_cycle = 100
    
    # Check if data_queue exists in session state (it should)
    if 'data_queue' not in st.session_state:
        return

    while not st.session_state.data_queue.empty() and processed < max_per_cycle:
        try:
            event = st.session_state.data_queue.get_nowait()
            processed += 1
            
            if isinstance(event, VoltageSampleEvent):
                st.session_state.voltage_history.append({
                    'time': event.timestamp,
                    'voltage': event.voltage,
                    'current': event.current
                })
                # Keep last 500 samples
                if len(st.session_state.voltage_history) > 500:
                    st.session_state.voltage_history = st.session_state.voltage_history[-500:]
                    
            elif isinstance(event, DWTResultEvent):
                st.session_state.dwt_energy_history.append({
                    'time': event.timestamp,
                    'energy': event.energy_levels,
                    'wavelet': event.wavelet
                })
                if len(st.session_state.dwt_energy_history) > 100:
                    st.session_state.dwt_energy_history = st.session_state.dwt_energy_history[-100:]
                    
            elif isinstance(event, HealthStatusEvent):
                st.session_state.health_data = {
                    'cpu': event.cpu_usage,
                    'memory': event.memory_usage,
                    'eps': event.events_per_second,
                    'latency': event.latency_avg_ms
                }
                
            elif isinstance(event, AIAnalysisEvent):
                st.session_state.ai_diagnosis = {
                    'probability': event.fault_probability,
                    'diagnosis': event.diagnosis,
                    'confidence': event.confidence,
                    'causes': event.probable_causes
                }

            elif isinstance(event, FaultLocationEvent):
                st.session_state.fault_location = {
                    'distance': event.distance_m,
                    'zone': event.zone,
                    'confidence': event.confidence,
                    'details': getattr(event, 'details', {})
                }
                # Add to timeline
                st.session_state.timeline_data.append({
                    'time': event.timestamp,
                    'source': 'FaultLocator',
                    'message': f"Fault Located at {event.distance_m:.2f}m ({event.zone})",
                    'details': getattr(event, 'details', {})
                })
                
            elif isinstance(event, SystemTripEvent):
                st.session_state.system_status = 'TRIPPED'
                st.session_state.trip_events.append({
                    'time': event.timestamp,
                    'reason': event.reason,
                    'latency_ms': event.latency_ms
                })
                add_log("CRITICAL", f"SYSTEM TRIP: {event.reason} (Latency: {event.latency_ms:.2f}ms)")
                
        except queue.Empty:
            break

def compute_node_positions(topology: Dict[str, Any]) -> Dict[str, Any]:
    """Compute node positions from topology or fallback layout."""
    nodes = topology.get('nodes', {}) if topology else {}
    positions = {}
    missing = []

    for node_id, node in nodes.items():
        position = node.get("position")
        if position and len(position) == 2:
            positions[node_id] = (position[0], position[1])
        else:
            missing.append(node_id)

    if missing:
        count = len(missing)
        radius = 1.0
        for idx, node_id in enumerate(missing):
            angle = 2 * np.pi * idx / max(count, 1)
            positions[node_id] = (radius * np.cos(angle), radius * np.sin(angle))

    return positions

def estimate_node_current(node: Dict[str, Any]) -> float:
    """Estimate current from node power and voltage."""
    voltage = node.get("voltage", 0.0) or 0.0
    power = node.get("power", 0.0) or 0.0
    if voltage == 0:
        return 0.0
    return power / voltage

def detect_transient(previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> bool:
    """Detect transient events based on sudden deltas."""
    if not previous:
        return False
    delta_v = abs(current["voltage"] - previous["voltage"])
    delta_i = abs(current["current"] - previous["current"])
    return delta_v > 20.0 or delta_i > 5.0

def update_component_history():
    """Record latest per-component telemetry for visualization and replay."""
    if not st.session_state.emulator:
        return

    now = time.time()
    if now - st.session_state.last_telemetry_update < 0.2:
        return
    st.session_state.last_telemetry_update = now

    topology = st.session_state.emulator.get_topology()
    nodes = topology.get("nodes", {})
    timestamp = now

    history = st.session_state.component_history
    for node_id, node in nodes.items():
        current = estimate_node_current(node)
        entry = {
            "time": timestamp,
            "voltage": float(node.get("voltage", 0.0)),
            "current": float(current),
            "power": float(node.get("power", 0.0)),
            "status": node.get("status", ""),
        }
        previous = history.get(node_id, [])[-1] if history.get(node_id) else None
        entry["transient"] = detect_transient(previous, entry)
        history.setdefault(node_id, []).append(entry)

        if len(history[node_id]) > 500:
            history[node_id] = history[node_id][-500:]

        if st.session_state.recording_enabled:
            st.session_state.recording_data.setdefault(node_id, []).append(entry.copy())

    st.session_state.component_history = history

def format_fault_trace(model: CircuitModel, fault_info: Dict[str, Any]) -> Dict[str, Any]:
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

def generate_sample():
    """Generate and publish a voltage sample from emulator."""
    if st.session_state.emulator and st.session_state.bus:
        voltage = st.session_state.emulator.read()
        node_id = getattr(st.session_state.emulator, "active_node", "BUS_DC")
        event = VoltageSampleEvent(
            voltage=voltage,
            current=voltage / 40,  # Simulated current
            node_id=node_id
        )
        st.session_state.bus.publish(event)

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_sidebar():
    """Render the sidebar navigation and controls."""
    with st.sidebar:
        st.markdown("## 🎛️ Control Center")
        
        # System Control
        st.markdown("### System Control")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", width='stretch', type="primary"):
                start_system()
        with col2:
            if st.button("⏹️ Stop", width='stretch'):
                stop_system()
        
        # Status indicator
        status = st.session_state.system_status
        status_class = {
            'IDLE': 'badge-normal',
            'RUNNING': 'badge-normal', 
            'TRIPPED': 'badge-fault',
            'STOPPED': 'badge-warning'
        }.get(status, 'badge-normal')
        
        st.markdown(f"""
        <div style="text-align: center; margin: 15px 0;">
            <span class="status-badge {status_class}">{status}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📍 Navigation")
        
        # Enforce Model-Driven Workflow
        model_loaded = st.session_state.circuit_model is not None and len(st.session_state.circuit_model.buses) > 0
        
        if not model_loaded:
            st.warning("⚠️ No Microgrid Model Loaded")
            st.info("Please load a model in the Unified Grid view to proceed.")
            page = "Unified Grid"
        else:
            page = st.radio(
                "Select View",
                ["Dashboard", "Unified Grid", "Wavelet Inspector",
                 "Fault Timeline", "AI Diagnosis", "Reports", "System Health"],
                label_visibility="collapsed"
            )
        
        st.markdown("---")
        
        # Fault Injection Panel
        st.markdown("### ⚡ Fault Injection")
        
        fault_type = st.selectbox(
            "Fault Type",
            ["L2L", "L2G", "ARC", "NOISE", "DRIFT", "SENSOR_FAIL"],
            format_func=lambda x: {
                "L2L": "🔌 Line-to-Line",
                "L2G": "🌍 Line-to-Ground", 
                "ARC": "⚡ Arc Fault",
                "NOISE": "📻 Noise Injection",
                "DRIFT": "📉 Voltage Drift",
                "SENSOR_FAIL": "🔧 Sensor Failure"
            }.get(x, x)
        )
        
        severity = st.slider("Severity", 0.1, 1.0, 0.5, 0.1)
        
        location_options = []
        if st.session_state.emulator and st.session_state.emulator.get_topology():
            location_options = list(st.session_state.emulator.get_topology().get("nodes", {}).keys())
        elif st.session_state.circuit_model and st.session_state.circuit_model.buses:
            location_options = [str(bus.id) for bus in st.session_state.circuit_model.buses]
        if not location_options:
            location_options = ["BUS_DC"]

        location = st.selectbox("Location", location_options)

        distance = st.slider("Fault Distance (m)", 0, 1000, 50, 10)
        st.session_state.fault_properties = {"distance": distance}
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚡ INJECT", width='stretch', type="primary"):
                inject_fault(fault_type, severity, location, {"distance": distance})
        with col2:
            if st.button("🔄 CLEAR", width='stretch'):
                clear_fault()
        
        # Converter Control
        st.markdown("---")
        st.markdown("### 🎚️ Converter Control")
        
        duty = st.slider("Duty Cycle", 0.0, 1.0, 
                        st.session_state.converter_state['duty'], 0.05)
        target_v = st.number_input("Target Voltage (V)", 300, 450, 
                                   st.session_state.converter_state['target_v'])
        mode = st.selectbox("Mode", ["AUTO", "MANUAL", "SAFE"])
        
        st.session_state.converter_state = {
            'duty': duty, 'target_v': target_v, 'mode': mode
        }
        
        return page

def render_dashboard():
    """Render the main dashboard view."""
    st.markdown("## ⚡ Real-Time Monitoring Dashboard")
    
    # Process latest events
    process_events()
    
    
    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_v = st.session_state.voltage_history[-1]['voltage'] if st.session_state.voltage_history else 400
        delta = current_v - 400
        st.metric("DC Bus Voltage", f"{current_v:.1f} V", f"{delta:+.1f} V")
        
    with col2:
        current_i = st.session_state.voltage_history[-1].get('current', 10) if st.session_state.voltage_history else 10
        st.metric("Line Current", f"{current_i:.2f} A")
        
    with col3:
        power = current_v * current_i / 1000
        st.metric("Power", f"{power:.2f} kW")
        
    with col4:
        prob = st.session_state.ai_diagnosis['probability'] * 100 if st.session_state.ai_diagnosis else 0
        st.metric("Fault Probability", f"{prob:.1f}%")
    
    # Charts
    if st.session_state.fault_location:
            loc = st.session_state.fault_location
            
            # Fault Location Result Bar
            st.markdown(f"""
            <div style="background: rgba(233, 69, 96, 0.1); border: 1px solid #e94560; border-radius: 8px; padding: 10px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h3 style="margin: 0; color: #e94560;">📍 Fault Located</h3>
                    <div style="font-size: 14px; color: #ccc;">Zone: <strong>{loc['zone']}</strong> | Confidence: <strong>{loc['confidence']*100:.0f}%</strong></div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: bold; color: #fff;">{loc['distance']:.1f} m</div>
                    <div style="font-size: 12px; color: #888;">from sensor</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Charts
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Voltage Chart
        st.markdown("### 📊 DC Bus Voltage")
        if st.session_state.voltage_history:
            voltages = [v['voltage'] for v in st.session_state.voltage_history[-200:]]
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(
                y=voltages, 
                mode='lines',
                name='Voltage',
                line=dict(color='#00CC96', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 150, 0.1)'
            ))
            fig_v.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                          title='Voltage (V)')
            )
            st.plotly_chart(fig_v, use_container_width=True)
        else:
            st.info("Start the system to see real-time data")
        
        # DWT Energy Chart
        st.markdown("### 📈 DWT Energy Spectrum")
        if st.session_state.dwt_energy_history:
            latest = st.session_state.dwt_energy_history[-1]['energy']
            levels = list(latest.keys())
            values = list(latest.values())
            
            fig_dwt = go.Figure()
            fig_dwt.add_trace(go.Bar(
                x=levels,
                y=values,
                marker_color=['#EF553B' if 'D1' in l else '#636EFA' for l in levels]
            ))
            fig_dwt.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                          title='Energy')
            )
            st.plotly_chart(fig_dwt, use_container_width=True)
    
    with col_side:
        # System Status
        st.markdown("### 🚨 System Alerts")
        
        for log in reversed(st.session_state.system_log[-8:]):
            icon = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🟢"}.get(log['level'], "⚪")
            alert_class = {
                "CRITICAL": "alert-critical",
                "WARNING": "alert-warning", 
                "INFO": "alert-info"
            }.get(log['level'], "alert-info")
            
            st.markdown(f"""
            <div class="alert-box {alert_class}">
                {icon} <strong>{log['time'].strftime('%H:%M:%S')}</strong><br>
                {log['message']}
            </div>
            """, unsafe_allow_html=True)
        
        # AI Diagnosis Summary
        if st.session_state.ai_diagnosis:
            st.markdown("### 🤖 AI Diagnosis")
            diag = st.session_state.ai_diagnosis
            
            color = "#e94560" if diag['probability'] > 0.5 else "#4CAF50"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color}">
                <strong>Diagnosis:</strong> {diag['diagnosis']}<br>
                <strong>Confidence:</strong> {diag['confidence']*100:.1f}%
            </div>
            """, unsafe_allow_html=True)

def render_digital_twin():
    """Render the digital twin visualization."""
    st.markdown("### 🌐 Grid Digital Twin")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.emulator:
            topology = st.session_state.emulator.get_topology()
            nodes = topology.get('nodes', {})
            connections = topology.get('connections', {})
            
            # Create network graph using plotly
            fig = go.Figure()
            positions = compute_node_positions(topology)
            
            # Draw connections
            for conn_id, conn in connections.items():
                from_pos = positions.get(conn['from_node'], (0, 0))
                to_pos = positions.get(conn['to_node'], (0, 0))
                
                color = '#4CAF50' if conn['status'] == 'CLOSED' else '#e94560'
                
                fig.add_trace(go.Scatter(
                    x=[from_pos[0], to_pos[0]],
                    y=[from_pos[1], to_pos[1]],
                    mode='lines',
                    line=dict(width=3, color=color),
                    hoverinfo='skip',
                    showlegend=False
                ))
            
            # Draw nodes
            for node_id, node in nodes.items():
                pos = positions.get(node_id, (0.5, 0.5))
                color = '#4CAF50' if node['status'] == 'ACTIVE' else '#e94560'
                
                fig.add_trace(go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers+text',
                    marker=dict(size=40, color=color, line=dict(width=2, color='white')),
                    text=node['name'][:10],
                    textposition='bottom center',
                    name=node_id,
                    hovertemplate=f"<b>{node['name']}</b><br>V: {node['voltage']:.1f}V<br>I: {node['current']:.1f}A<extra></extra>"
                ))
            
            fig.update_layout(
                height=500,
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start the system to view the Digital Twin")
    
    with col2:
        st.markdown("### 📋 Node Properties")
        
        if st.session_state.emulator:
            topology = st.session_state.emulator.get_topology()
            nodes = topology.get('nodes', {})
            
            selected = st.selectbox("Select Node", list(nodes.keys()))
            
            if selected and selected in nodes:
                node = nodes[selected]
                st.markdown(f"""
                **Name:** {node['name']}  
                **Type:** {node['node_type']}  
                **Status:** {node['status']}  
                **Voltage:** {node['voltage']:.2f} V  
                **Current:** {node['current']:.2f} A  
                **Power:** {node['power']:.2f} W
                """)
                
                if st.button("Toggle Status"):
                    new_status = "INACTIVE" if node['status'] == "ACTIVE" else "ACTIVE"
                    st.session_state.emulator.set_node_status(selected, new_status)

def render_circuit_designer():
    """Render the Circuit Designer / Editor page."""
    st.markdown("### 🛠️ Circuit Designer (MATLAB Bridge)")
    
    col_tools, col_editor = st.columns([1, 1])
    
    with col_tools:
        st.markdown("### File Operations")
        uploaded_file = st.file_uploader("Load Circuit (.mat)", type=['mat'])
        
        if uploaded_file is not None:
            # Save to temp file to read with bridge
            with open("temp_upload.mat", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                model = MatlabBridge.load_model("temp_upload.mat")
                if model:
                    st.session_state.circuit_model = model
                    st.success(f"Loaded {model.name} with {len(model.buses)} buses")
                    
                    # Update Emulator if running
                    if st.session_state.emulator:
                        st.session_state.emulator.load_circuit(model)
                        st.info("Emulator updated with new circuit")
            except Exception as e:
                st.error(f"Failed to load: {e}")
                
        if st.button("💾 Save as Default (grid.mat)"):
            MatlabBridge.save_model(st.session_state.circuit_model, "grid.mat")
            st.success("Saved to grid.mat")
            
        if st.button("🔄 Reload from Default"):
            model = MatlabBridge.load_model("grid.mat")
            if model:
                st.session_state.circuit_model = model
                if st.session_state.emulator:
                    st.session_state.emulator.load_circuit(model)
                st.success("Reloaded default grid")
                
        # Offer Reference Model
        if st.button("🏗️ Load Reference Microgrid (6-Bus)", type="primary"):
            try:
                # Path to reference model generated by script
                ref_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reference_microgrid.mat'))
                model = MatlabBridge.load_model(ref_path)
                if model:
                    st.session_state.circuit_model = model
                    if st.session_state.emulator:
                        st.session_state.emulator.load_circuit(model)
                    st.success("Loaded Reference Microgrid")
            except Exception as e:
                st.error(f"Failed to load reference: {e}")
    
    with col_editor:
        st.markdown("### Component Editor")
        model = st.session_state.circuit_model
        
        comp_type = st.selectbox("Component Type", ["Bus", "Line", "Generator", "Load"])
        
        if comp_type == "Bus":
            if not model.buses:
                st.info("No buses available yet. Load or create a circuit to edit buses.")
            else:
                buses = {f"{b.id}: {b.name}": b for b in model.buses}
                selected_name = st.selectbox("Select Bus", list(buses.keys()))
                if selected_name:
                    bus = buses[selected_name]
                    new_v = st.number_input("Voltage (kV)", 0.0, 1000.0, bus.voltage_kv)
                    new_name = st.text_input("Name", bus.name)
                    if st.button("Update Bus"):
                        bus.voltage_kv = new_v
                        bus.name = new_name
                        st.success("Updated")
                    
        elif comp_type == "Line":
            if not model.lines:
                st.info("No lines available yet. Load or create a circuit to edit lines.")
            else:
                lines = {f"Line {l.id} ({l.from_bus}->{l.to_bus})": l for l in model.lines}
                selected_name = st.selectbox("Select Line", list(lines.keys()))
                if selected_name:
                    line = lines[selected_name]
                    new_r = st.number_input("Resistance (Ohm)", 0.0, 100.0, line.r_ohm, format="%.4f")
                    new_x = st.number_input("Reactance (Ohm)", 0.0, 100.0, line.x_ohm, format="%.4f")
                    if st.button("Update Line"):
                        line.r_ohm = new_r
                        line.x_ohm = new_x
                        st.success("Updated")
                        
        elif comp_type == "Generator":
            if not model.generators:
                st.info("No generators available yet. Load or create a circuit to edit generators.")
            else:
                generators = {f"Gen {g.id} (Bus {g.bus_id})": g for g in model.generators}
                selected_name = st.selectbox("Select Generator", list(generators.keys()))
                if selected_name:
                    generator = generators[selected_name]
                    new_p = st.number_input("Power Output (MW)", 0.0, 1000.0, generator.p_mw)
                    new_pmax = st.number_input("Max Power (MW)", 0.0, 2000.0, generator.p_max_mw)
                    if st.button("Update Generator"):
                        generator.p_mw = new_p
                        generator.p_max_mw = new_pmax
                        st.success("Updated")
                        
        elif comp_type == "Load":
            if not model.loads:
                st.info("No loads available yet. Load or create a circuit to edit loads.")
            else:
                loads = {f"Load {l.id} (Bus {l.bus_id})": l for l in model.loads}
                selected_name = st.selectbox("Select Load", list(loads.keys()))
                if selected_name:
                    load = loads[selected_name]
                    new_p = st.number_input("Demand (MW)", 0.0, 1000.0, load.p_mw)
                    if st.button("Update Load"):
                        load.p_mw = new_p
                        st.success("Updated")

    # Visualization
    st.markdown("### Circuit Diagram")
    model = st.session_state.circuit_model
    if model.buses:
        fig = go.Figure()
        
        # Draw Lines
        for line in model.lines:
            # Find coordinates
            b1 = next((b for b in model.buses if b.id == line.from_bus), None)
            b2 = next((b for b in model.buses if b.id == line.to_bus), None)
            
            if b1 and b2:
                fig.add_trace(go.Scatter(
                    x=[b1.x, b2.x], y=[b1.y, b2.y],
                    mode='lines',
                    line=dict(width=line.status*2 + 1, color='#888'),
                    hoverinfo='text',
                    text=f"Line {line.id}: R={line.r_ohm}"
                ))
        
        # Draw Buses
        for bus in model.buses:
            color = '#4CAF50' if bus.type == 'Slack' else '#636EFA'
            fig.add_trace(go.Scatter(
                x=[bus.x], y=[bus.y],
                mode='markers+text',
                marker=dict(size=20, color=color),
                text=bus.name,
                textposition='bottom center',
                name=bus.name
            ))
            
        fig.update_layout(
            showlegend=False,
            height=500,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No circuit loaded. Load a file or create from scratch.")

def render_component_telemetry(title: str, series_by_node: Dict[str, List[Dict[str, Any]]]):
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

def render_live_fault_graphs():
    """Render immediate fault visibility graphs in the Unified Grid view."""
    st.markdown("### 📉 Live Fault Graphs")

    if not st.session_state.system_running:
        st.info("Start the system to see live fault graphs.")
        return

    # Graph 1: Main sensor stream (same signal path used by detection chain)
    if st.session_state.voltage_history:
        recent = st.session_state.voltage_history[-250:]
        base_time = recent[0]["time"]
        t = [entry["time"] - base_time for entry in recent]
        v = [entry["voltage"] for entry in recent]

        fig_sensor = go.Figure()
        fig_sensor.add_trace(go.Scatter(
            x=t,
            y=v,
            mode="lines",
            name="Sensor Voltage",
            line=dict(color="#00CC96", width=2)
        ))

        fault_info = st.session_state.emulator.get_fault_info() if st.session_state.emulator else {}
        if fault_info.get("active"):
            fig_sensor.add_hline(
                y=np.mean(v),
                line_dash="dash",
                line_color="#e94560",
                annotation_text=f"FAULT ACTIVE: {fault_info.get('type', 'UNKNOWN')}"
            )

        fig_sensor.update_layout(
            height=280,
            xaxis_title="Time (s)",
            yaxis_title="Voltage (V)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_sensor, use_container_width=True)
    else:
        st.info("Waiting for sensor samples. Start the system and inject a fault.")

    # Graph 2: Per-node waveforms from emulator history for fault localization context
    if st.session_state.emulator:
        topology = st.session_state.emulator.get_topology()
        nodes = list(topology.get("nodes", {}).keys())
        if nodes:
            selected_nodes = st.multiselect(
                "Nodes to visualize",
                options=nodes,
                default=nodes[:min(3, len(nodes))],
                key="live_fault_nodes"
            )

            if selected_nodes:
                fig_nodes = go.Figure()
                for node_id in selected_nodes:
                    history = st.session_state.emulator.get_history(node_id)
                    if len(history) == 0:
                        continue
                    downsampled = history[::100]
                    tx = np.arange(len(downsampled)) / (st.session_state.emulator.sample_rate / 100.0)
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
        if st.session_state.fault_location:
            trace = format_fault_trace(st.session_state.circuit_model, st.session_state.fault_location)
            fault_info = st.session_state.emulator.get_fault_info() if st.session_state.emulator else {}
            if fault_info.get("active"):
                st.markdown(
                    f"**Fault Type:** {fault_info.get('type')}  \n"
                    f"**Severity:** {fault_info.get('severity', 0.0):.2f}  \n"
                    f"**Location Node:** {fault_info.get('location')}"
                )

            details = st.session_state.fault_location.get("details", {})
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
            st.info(trace.get("explanation"))
        else:
            st.info("Run a fault simulation to see the exact circuit trace.")

    with circuit_tab:
        render_circuit_designer()

    with telemetry_tab:
        st.markdown("### ⏺️ Recording & Replay")
        controls = st.columns([1, 1, 1])
        with controls[0]:
            if st.button("Start Recording", width='stretch', disabled=st.session_state.recording_enabled):
                st.session_state.recording_enabled = True
                st.session_state.recording_data = {}
        with controls[1]:
            if st.button("Stop Recording", width='stretch', disabled=not st.session_state.recording_enabled):
                st.session_state.recording_enabled = False
                if st.session_state.recording_data:
                    recording = {
                        "name": datetime.now().strftime("Recording %H:%M:%S"),
                        "data": st.session_state.recording_data.copy()
                    }
                    st.session_state.recordings.append(recording)
                    st.session_state.recording_data = {}
        with controls[2]:
            if st.button("Clear Recordings", width='stretch'):
                st.session_state.recordings = []
                st.session_state.recording_data = {}

        if st.session_state.recordings:
            recording_names = [rec["name"] for rec in st.session_state.recordings]
            selected_name = st.selectbox("Replay Session", recording_names)
            st.session_state.selected_recording = selected_name
            st.session_state.replay_mode = st.toggle("Replay Mode", value=st.session_state.replay_mode)

        if st.session_state.replay_mode and st.session_state.selected_recording:
            selected = next(
                (rec for rec in st.session_state.recordings if rec["name"] == st.session_state.selected_recording),
                None
            )
            if selected:
                render_component_telemetry("Recorded Component Telemetry", selected["data"])
        else:
            render_component_telemetry("Live Component Telemetry", st.session_state.component_history)

def render_wavelet_inspector():
    """Render the wavelet analysis inspector."""
    st.markdown("## 📈 Wavelet Analysis Inspector")
    
    col_settings, col_viz = st.columns([1, 3])
    
    with col_settings:
        st.markdown("### DSP Settings")
        
        wavelet_family = st.selectbox(
            "Wavelet Family",
            ["db4", "db6", "db8", "haar", "sym5", "sym8", "coif3"],
            index=0
        )
        
        decomp_level = st.slider("Decomposition Level", 1, 6, 4)
        
        st.session_state.wavelet_settings = {
            'family': wavelet_family,
            'level': decomp_level
        }
        
        st.markdown("---")
        st.markdown("### AI Analysis")
        
        if st.session_state.ai_diagnosis:
            diag = st.session_state.ai_diagnosis
            
            prob_color = "#e94560" if diag['probability'] > 0.5 else "#4CAF50"
            st.markdown(f"""
            <div style="font-size: 32px; color: {prob_color}; font-weight: bold; text-align: center;">
                {diag['probability']*100:.1f}%
            </div>
            <div style="text-align: center; color: #888;">Fault Probability</div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Diagnosis:** {diag['diagnosis']}")
            st.markdown(f"**Confidence:** {diag['confidence']*100:.1f}%")
            
            if diag['causes']:
                st.markdown("**Probable Causes:**")
                for cause in diag['causes'][:3]:
                    st.markdown(f"- {cause.get('cause', 'Unknown')}: {cause.get('probability', 0)*100:.1f}%")
    
    with col_viz:
        st.markdown("### Energy Spectrum")
        
        if st.session_state.dwt_energy_history:
            latest = st.session_state.dwt_energy_history[-1]['energy']
            levels = list(latest.keys())
            values = list(latest.values())
            
            # Color scale based on values
            max_val = max(values) if values else 1
            colors = [
                f'rgba(233, 69, 96, {min(1, v/max_val + 0.3)})' 
                for v in values
            ]
            
            fig = go.Figure(go.Bar(
                x=levels,
                y=values,
                marker_color=colors
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title='Decomposition Level'),
                yaxis=dict(title='Energy', gridcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Coefficient detail (simulated)
            st.markdown("### Detail Coefficients (D1)")
            
            d1_sim = np.random.randn(100)
            if st.session_state.system_status == 'TRIPPED':
                d1_sim *= 10  # Amplify during fault
            
            fig_d1 = go.Figure(go.Scatter(
                y=d1_sim,
                mode='lines',
                line=dict(color='#EF553B', width=1)
            ))
            fig_d1.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_d1, use_container_width=True)
        else:
            st.info("Run a simulation to see wavelet analysis")

def render_fault_timeline():
    """Render the fault timeline and analysis view."""
    st.markdown("## 🕵️ Fault Investigation Timeline")
    
    # Check for both trip events and fault location events
    has_events = (len(st.session_state.trip_events) > 0) or (len(st.session_state.timeline_data) > 0)
    
    if not has_events:
        st.info("No fault events detected yet. Inject a fault to see the analysis.")
        return

    # Tabs for different views
    tab1, tab2 = st.tabs(["📝 Investigation Log", "📼 Black Box Replay"])
    
    with tab1:
        if not st.session_state.timeline_data:
             st.info("No detailed analysis data available.")
        else:
            # Reverse order to show newest first
            for i, event in enumerate(reversed(st.session_state.timeline_data)):
                time_str = datetime.fromtimestamp(event['time']).strftime('%H:%M:%S.%f')[:-3]
                
                with st.expander(f"⏰ {time_str} - {event['message']}", expanded=(i==0)):
                    cols = st.columns([1, 2])
                    
                    with cols[0]:
                        st.markdown("**Event Details**")
                        st.write(f"Source: {event['source']}")
                        
                        details = event.get('details', {})
                        if details:
                            st.markdown("### 🧠 Logic Trace")
                            if 'energy_ratio' in details:
                                ratio = details['energy_ratio']
                                dist = details.get('estimated_dist', 0)
                                st.metric("Energy Ratio (D1/D2)", f"{ratio:.2f}")
                                st.metric("Estimated Distance", f"{dist:.1f} m")
                                st.caption("Ratio decreases as distance increases due to HF attenuation.")
                    
                    with cols[1]:
                        # Waveform Visualization
                        if 'node_waveforms' in details and details['node_waveforms']:
                            st.markdown("### 📉 Recorded Component Signals")
                            waveforms = details['node_waveforms']
                            
                            fig = go.Figure()
                            for node_id, data in waveforms.items():
                                # Create time axis based on 20kHz sampling (downsampled by 10)
                                t_axis = np.linspace(0, 1.0, len(data))
                                fig.add_trace(go.Scatter(x=t_axis, y=data, name=f"Node {node_id}"))
                                
                            fig.update_layout(
                                title="Voltage Waveforms during Fault Window",
                                xaxis_title="Time (s)",
                                yaxis_title="Voltage (V)",
                                height=300,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            if event['source'] == 'FaultLocator':
                                st.warning("No waveform snapshot available for this event.")

        # Clear Timeline Button
        if st.button("Clear Timeline", key="clear_timeline"):
            st.session_state.timeline_data = []
            st.rerun()

    with tab2:
        if st.session_state.trip_events:
            # Show latest trip event
            trip = st.session_state.trip_events[-1]
            
            st.markdown(f"### Trip Event at {datetime.fromtimestamp(trip['time']).strftime('%H:%M:%S.%f')[:-3]}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Reason", trip['reason'])
            col2.metric("Detection Latency", f"{trip['latency_ms']:.2f} ms")
            col3.metric("Status", "TRIPPED", delta_color="inverse")
            
            # Restore Timeline visualization (Gantt-like chart)
            st.markdown("### ⏱️ Sequence of Operations")
            timeline_data = [
                dict(Task="Fault Injection", Start=0, Finish=2, Resource="Emulator"),
                dict(Task="Wave propagation", Start=2, Finish=5, Resource="Line"),
                dict(Task="DWT Detail Capture", Start=5, Finish=7, Resource="DSP"),
                dict(Task="Threshold Violation", Start=7, Finish=8, Resource="Guard"),
                dict(Task="Fault Voting", Start=8, Finish=9, Resource="Voter"),
                dict(Task="Trip Execution", Start=9, Finish=10, Resource="Sequencer"),
            ]
            
            df = pd.DataFrame(timeline_data)
            fig = px.timeline(
                df, x_start="Start", x_end="Finish", y="Task", color="Resource",
                color_discrete_map={
                    "Emulator": "#e94560", "Line": "#f8b500", "DSP": "#00CC96",
                    "Guard": "#AB63FA", "Voter": "#FFA15A", "Sequencer": "#19D3F3"
                }
            )
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Replay Controls
            st.markdown("---")
            st.markdown("### 📼 Digital Replay Control")
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)
            rcol1.button("⏪ Rewind", key="btn_rewind")
            rcol2.button("▶️ Play", key="btn_play")
            rcol3.button("⏸️ Pause", key="btn_pause")
            rcol4.button("💾 Export Replay", key="btn_export")
            
            st.slider("Replay Position", 0, 100, 50, help="Drag to seek within the recording")
        else:
            st.info("No system trips recorded. Inject a fault to see the sequence replay.")

def render_ai_diagnosis():
    """Render the AI diagnosis panel."""
    st.markdown("## 🤖 AI-Assisted Fault Diagnosis")
    
    if st.session_state.ai_diagnosis:
        diag = st.session_state.ai_diagnosis
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Probability gauge
            prob = diag['probability'] * 100
            color = "#e94560" if prob > 50 else "#FFA726" if prob > 20 else "#4CAF50"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fault Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(76, 175, 80, 0.3)'},
                        {'range': [30, 70], 'color': 'rgba(255, 167, 38, 0.3)'},
                        {'range': [70, 100], 'color': 'rgba(233, 69, 96, 0.3)'}
                    ]
                }
            ))
            
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Primary Diagnosis", diag['diagnosis'])
            st.metric("Confidence", f"{diag['confidence']*100:.1f}%")
        
        with col2:
            st.markdown("### Probable Causes")
            
            if diag['causes']:
                causes = diag['causes'][:5]
                # Filter out causes with no name
                causes = [c for c in causes if c.get('cause')]
                
                if causes:
                    cause_names = [c.get('cause')[:30] for c in causes]
                    probs = [c.get('probability', 0) * 100 for c in causes]
                    
                    fig = go.Figure(go.Bar(
                        x=probs,
                        y=cause_names,
                        orientation='h',
                        marker_color=['#e94560' if p > 50 else '#FFA726' if p > 20 else '#4CAF50' for p in probs]
                    ))
                    
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(title='Probability %', range=[0, 100]),
                        yaxis=dict(autorange="reversed")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No probable causes identified.")
            
            st.markdown("### Recommendations")
            if "High Impedance" in diag['diagnosis']:
                st.warning("🔍 Inspect line segment 3 near Bus 4 for potential insulation degradation.")
            elif "Line-to-Ground" in diag['diagnosis']:
                st.error("🚨 Critical ground fault detected. Remote isolation recommended.")
            else:
                st.info("System behavior is within normal parameters. Continue monitoring.")
    else:
        st.info("Waiting for AI analysis results... Run a fault simulation to trigger diagnosis.")

def render_reports():
    """Render the reports generation page."""
    st.markdown("## 📑 Automated Reporting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Generate Report")
        
        report_type = st.selectbox(
            "Report Type",
            ["Incident Report", "Daily Summary", "Benchmark Comparison"]
        )
        
        scenario_name = st.text_input("Scenario Name", "Line-to-Line Fault Test")
        
        if st.button("📄 Generate Report", type="primary"):
            if st.session_state.report_generator:
                with st.spinner("Generating report..."):
                    time.sleep(1)  # Simulate processing
                    filepath = st.session_state.report_generator.generate_incident_report(scenario_name)
                    if filepath:
                        st.success(f"Report generated: {filepath}")
                        
                        # Offer download
                        try:
                            with open(filepath, 'r') as f:
                                html_content = f.read()
                            st.download_button(
                                "📥 Download Report",
                                html_content,
                                file_name="incident_report.html",
                                mime="text/html"
                            )
                        except:
                            pass
                    else:
                        st.warning("No incident data to report. Run a fault scenario first.")
            else:
                st.warning("Start the system first to enable reporting")
        
        st.markdown("---")
        st.markdown("### Available Reports")
        
        if st.session_state.report_generator:
            reports = st.session_state.report_generator.get_available_reports()
            for r in reports[:5]:
                st.markdown(f"📄 {r['filename']}")
    
    with col2:
        st.markdown("### Report Preview")
        
        if st.session_state.trip_events:
            trip = st.session_state.trip_events[-1]
            
            st.markdown(f"""
            ## 📊 Incident Report Preview
            
            **Report ID:** INC-{datetime.now().strftime('%Y%m%d%H%M')}  
            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ---
            
            ### Summary
            
            | Metric | Value |
            |--------|-------|
            | Fault Type | {trip['reason']} |
            | Detection Latency | {trip['latency_ms']:.2f} ms |
            | Trip Triggered | ✅ Yes |
            | System Status | TRIPPED |
            
            ---
            
            ### Performance Assessment
            
            {"✅ **PASSED**: Detection latency within 5ms target" if trip['latency_ms'] < 5 else "❌ **NEEDS REVIEW**: Detection exceeded target"}
            """)
        else:
            st.info("Run a fault simulation to preview report content")

def render_system_health():
    """Render the system health monitoring page."""
    st.markdown("## 💻 System Health Monitor")
    
    health = st.session_state.health_data
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("CPU Usage", f"{health['cpu']:.1f}%", 
               delta="Normal" if health['cpu'] < 80 else "High")
    col2.metric("Memory Usage", f"{health['memory']:.1f}%",
               delta="Normal" if health['memory'] < 80 else "High")
    col3.metric("Events/Second", f"{health['eps']:.1f}")
    col4.metric("Avg Latency", f"{health['latency']:.2f} ms")
    
    # Progress bars
    st.markdown("### Resource Utilization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**CPU**")
        st.progress(min(health['cpu'] / 100, 1.0))
        
        st.markdown("**Memory**")
        st.progress(min(health['memory'] / 100, 1.0))
    
    with col2:
        # Buffer fill (simulated)
        st.markdown("**Event Buffer**")
        buffer_fill = 0.3
        st.progress(buffer_fill)
        if buffer_fill > 0.8:
            st.warning("⚠️ Buffer near capacity!")
        
        st.markdown("**Processing Queue**")
        st.progress(0.1)
    
    # Performance chart
    st.markdown("### Performance History")
    
    # Simulated history
    times = pd.date_range(end=datetime.now(), periods=60, freq='s')
    cpu_hist = np.random.uniform(10, 40, 60) + np.sin(np.linspace(0, 4*np.pi, 60)) * 10
    mem_hist = np.random.uniform(30, 50, 60)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=cpu_hist, name='CPU %', line=dict(color='#e94560')))
    fig.add_trace(go.Scatter(x=times, y=mem_hist, name='Memory %', line=dict(color='#00CC96')))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main application entry point."""
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Process events
    process_events()
    update_component_history()
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Unified Grid":
        render_unified_grid()
    elif page == "Wavelet Inspector":
        render_wavelet_inspector()
    elif page == "Fault Timeline":
        render_fault_timeline()
    elif page == "AI Diagnosis":
        render_ai_diagnosis()
    elif page == "Reports":
        render_reports()
    elif page == "System Health":
        render_system_health()
    
    # Health footer
    health = st.session_state.health_data
    st.markdown("---")
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
    fcol1.markdown(f"🖥️ CPU: {health['cpu']:.0f}%")
    fcol2.markdown(f"💾 Mem: {health['memory']:.0f}%")
    fcol3.markdown(f"⚡ EPS: {health['eps']:.0f}")
    fcol4.markdown(f"⏱️ Lat: {health['latency']:.1f}ms")
    fcol5.markdown(f"📊 Status: {st.session_state.system_status}")
    
    # Auto-refresh when running
    if st.session_state.system_running:
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()
