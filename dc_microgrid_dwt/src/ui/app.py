import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import threading
import time
import queue
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.framework.bus import EventBus
from src.framework.registry import AgentRegistry
from src.domain.events import VoltageSampleEvent, ProcessingResultEvent, SystemTripEvent
from src.ui.bridge import BridgeAgent
from simulation.advanced_scenarios import AdvancedScenarioRunner, SimulationResult # Import new runner

st.set_page_config(page_title="DC Microgrid Fault Detection", layout="wide")

# Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .status-safe { color: #00FF00; font-weight: bold; }
    .status-trip { color: #FF0000; font-weight: bold; font-size: 2em; animation: blink 1s infinite; }
    @keyframes blink { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# State Management
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'data_queue' not in st.session_state:
    st.session_state.data_queue = None
if 'system_status' not in st.session_state:
    st.session_state.system_status = "SAFE"
if 'voltage_buffer' not in st.session_state:
    st.session_state.voltage_buffer = []  
if 'last_result' not in st.session_state:
    st.session_state.last_result = None  

def run_simulation(bridge_queue, scenario_name):
    # This runs in a separate thread
    runner = AdvancedScenarioRunner()
    
    # Hook Bridge
    bridge = BridgeAgent("UI_Bridge", runner.bus)
    bridge.queue = bridge_queue 
    runner.registry.register(bridge)
    
    # Run in Visual Mode (Real-Time) so charts update
    result = runner.run(scenario_name, visual_mode=True)
    st.session_state.last_result = result
    st.session_state.simulation_running = False

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "System Logs", "Benchmarking"])

st.sidebar.title("Simulation Control")
scenario = st.sidebar.selectbox("Select Scenario", [
    "Baseline (Normal)", 
    "Line-to-Line Fault", 
    "High Noise Stress", 
    "Gradual Drift"
])

if st.sidebar.button("Start Simulation"):
    if not st.session_state.simulation_running:
        st.session_state.simulation_running = True
        st.session_state.system_status = "SAFE"
        st.session_state.voltage_buffer = []
        st.session_state.data_queue = queue.Queue()
        st.session_state.last_result = None
        
        # Start Thread with Scenario
        t = threading.Thread(target=run_simulation, args=(st.session_state.data_queue, scenario))
        t.start()

if page == "System Logs":
    st.title("📜 System Logs")
    st.markdown("Real-time logs from the Observability module. Use the button below to copy.")
    
    # Refresh button
    if st.button("Refresh Logs"):
        pass
        
    # Get logs from Observability singleton
    # Note: Streamlit runs in a separate process/thread from the simulation thread in this simple setup,
    # but since we are threading the simulation *inside* the streamlit app process, they share memory.
    try:
        obs = Observability.get_instance()
        logs_text = obs.get_logs()
        st.code(logs_text, language="text")
    except Exception as e:
        st.error(f"Could not retrieve logs: {e}")

elif page == "Benchmarking":
    st.title("📊 Automated Benchmarking Report")
    
    if st.session_state.last_result:
        res = st.session_state.last_result
        
        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scenario", res.scenario_name)
        m2.metric("Trip Triggered", "YES" if res.trip_triggered else "NO")
        m3.metric("Latency", f"{res.latency_ms:.4f} ms" if res.latency_ms else "N/A")
        m4.metric("Duration", f"{res.duration_s:.2f} s")
        
        st.divider()
        
        # Detailed Findings
        st.subheader("Analysis")
        if res.trip_triggered:
            if res.latency_ms < 5.0:
                st.success(f"✅ PASSED: Fault detected in {res.latency_ms:.4f}ms (< 5ms target)")
            else:
                st.error(f"❌ FAILED: Detection too slow ({res.latency_ms:.4f}ms)")
        elif res.fault_injected:
             st.error("❌ FAILED: Fault injected but NOT detected!")
        else:
             st.info("ℹ️ Normal Operation - No faults injected.")
             
    else:
        st.info("Run a simulation to generate a benchmark report.")

else:
    # Main Dashboard Page
    st.title("⚡ DC Microgrid DWT Fault Detection")

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    status_placeholder = col1.empty()
    peak_placeholder = col2.empty()
    latency_placeholder = col3.empty()

    # Charts
    chart_voltage = st.empty()
    chart_dwt = st.empty()

    # Real-time Loop (Only run loop if on Dashboard)
    if st.session_state.simulation_running:
        voltage_data = []
        d1_data = []
        timestamps = []
        start_time = time.time()
        
        while st.session_state.simulation_running:
            # Drain Queue
            try:
                while not st.session_state.data_queue.empty():
                    evt = st.session_state.data_queue.get_nowait()
                    
                    if isinstance(evt, VoltageSampleEvent):
                        voltage_data.append(evt.voltage)
                        timestamps.append(evt.timestamp)
                    
                    elif isinstance(evt, ProcessingResultEvent):
                        d1_data.append(evt.d1_peak)
                        peak_placeholder.metric("D1 Peak Energy", f"{evt.d1_peak:.2f}")
                    
                    elif isinstance(evt, SystemTripEvent):
                        st.session_state.system_status = "TRIPPED"
                        
            except queue.Empty:
                pass
                
            # Update UI Elements
            if st.session_state.system_status == "SAFE":
                status_placeholder.markdown(f"### SYSTEM STATUS: <span class='status-safe'>NORMAL</span>", unsafe_allow_html=True)
            else:
                status_placeholder.markdown(f"### SYSTEM STATUS: <span class='status-trip'>TRIPPED</span>", unsafe_allow_html=True)
                
            # Update Charts (Throttle updates to avoid lag)
            if len(voltage_data) > 0:
                # Voltage Chart
                 fig_v = go.Figure()
                 subset_v = voltage_data[-200:] # Show last 200 samples
                 fig_v.add_trace(go.Scatter(y=subset_v, mode='lines', name='Voltage', line=dict(color='#00CC96')))
                 fig_v.update_layout(title="DC Bus Voltage (V)", height=300, margin=dict(l=0, r=0, t=30, b=0))
                 chart_voltage.plotly_chart(fig_v, use_container_width=True)

                 # DWT Chart (simulated correlation for visual)
                 if len(d1_data) > 0:
                     fig_d = go.Figure()
                     subset_d = d1_data[-50:] # Slower update rate usually
                     fig_d.add_trace(go.Scatter(y=subset_d, mode='bars', name='D1 Coeff', marker_color='#EF553B'))
                     fig_d.update_layout(title="Wavelet Detail Coeff (D1)", height=300, margin=dict(l=0, r=0, t=30, b=0))
                     chart_dwt.plotly_chart(fig_d, use_container_width=True)
            
            time.sleep(0.1)
