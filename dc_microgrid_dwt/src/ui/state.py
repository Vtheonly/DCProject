"""
Session State Module — DC Microgrid Fault Detection Platform

Initializes and manages all Streamlit session state variables.
Provides a single point of truth for application state defaults.
"""
import queue
import streamlit as st


def init_session_state():
    """Initialize all session state variables with defaults.
    
    This must be called at the start of every Streamlit rerun.
    All UI state, system references, and data buffers are defined here.
    """
    defaults = {
        # --- System References ---
        "system_running": False,
        "emulator": None,
        "event_bus": None,
        "registry": None,
        "bridge_agent": None,
        "data_queue": queue.Queue(),

        # --- Navigation ---
        "current_page": "Dashboard",

        # --- Live Data Buffers ---
        "voltage_data": [],
        "dwt_energy": {"D1": 0, "D2": 0, "D3": 0, "D4": 0, "A4": 0},
        "dwt_coefficients": [],       # Actual DWT coefficients [A4, D4, D3, D2, D1]
        "energy_history": [],          # List of energy dicts over time
        "voltage_history_per_node": {},  # {node_id: [voltage_values]}

        # --- Fault State ---
        "fault_active": False,
        "fault_type": "NONE",
        "fault_location": None,
        "fault_distance": None,
        "fault_zone": None,
        "trip_active": False,
        "fault_events": [],            # List of fault event dicts

        # --- AI Diagnosis ---
        "ai_diagnosis": None,
        "ai_probable_causes": [],

        # --- Health Metrics ---
        "health_data": {
            "cpu": 0, "memory": 0, "eps": 0,
            "latency": 0, "uptime": 0
        },
        "health_history": [],

        # --- C++ DSP Pipeline ---
        "dsp_pipeline": None,          # microgrid_dsp.DSPPipeline if available
        "dsp_available": False,

        # --- Circuit Model ---
        "circuit_model": None,

        # --- System Log ---
        "system_log": [],

        # --- Recording ---
        "recording_active": False,
        "recorded_events": [],

        # --- Fault Properties for injection ---
        "fault_properties": {},
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
