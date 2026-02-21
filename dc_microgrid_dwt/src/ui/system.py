"""
System Control Module — DC Microgrid Fault Detection Platform

Contains all system control functions: start, stop, inject/clear faults,
event processing, and DSP pipeline integration.
"""
import os
import sys
import time
import queue
import logging
import threading
import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.framework.bus import EventBus
from src.framework.registry import AgentRegistry
from src.framework.observability import Observability
from src.adapters.grid_emulator import GridEmulator
from src.adapters.hardware_adc import SimulatedADCSensor, SimulatedRelayDriver
from src.domain.events import (
    VoltageSampleEvent, ProcessingResultEvent, SystemTripEvent,
    DWTCoefficientsEvent, DWTResultEvent, FaultDetectedEvent,
    FaultLocationEvent, HealthStatusEvent, AIAnalysisEvent
)
from src.domain.circuit import CircuitModel, Bus, Line, Generator, Load
from src.agents.ingestion.sampler import SamplerAgent
from src.agents.ingestion.window_manager import WindowManagerAgent
from src.agents.processing.dwt_engine import DWTEngineAgent
from src.agents.processing.detail_analyzer import DetailAnalyzerAgent
from src.agents.processing.dsp_runner import DSPRunnerAgent
from src.agents.processing.fault_locator import PreciseFaultLocatorAgent
from src.agents.detection.threshold_guard import ThresholdGuardAgent
from src.agents.detection.fault_voter import FaultVoterAgent
from src.agents.detection.energy_monitor import EnergyMonitorAgent
from src.agents.control.trip_sequencer import TripSequencerAgent
from src.agents.control.zeta_logic import ZetaLogicAgent
from src.agents.supervision.health_monitor import HealthMonitorAgent
from src.agents.supervision.ai_classifier import AIClassifierAgent
from src.agents.supervision.replay_recorder import ReplayRecorderAgent
from src.agents.supervision.report_generator import ReportGeneratorAgent
from src.ui.bridge import BridgeAgent
from src.adapters.high_speed_loop import HighSpeedDetectionLoop


# Try to import C++ DSP module
try:
    sys.path.insert(0, _project_root)
    import microgrid_dsp
    DSP_AVAILABLE = True
    logger.info("C++ DSP module loaded successfully")
except ImportError:
    DSP_AVAILABLE = False
    logger.info("C++ DSP module not available, using Python fallback")


def add_log(message: str, level: str = "INFO"):
    """Add a timestamped log entry to session state."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    if "system_log" in st.session_state:
        st.session_state.system_log.append(entry)
        # Keep last 200 entries
        if len(st.session_state.system_log) > 200:
            st.session_state.system_log = st.session_state.system_log[-200:]


def create_reference_circuit() -> CircuitModel:
    """Create the 6-bus reference microgrid circuit model."""
    model = CircuitModel(name="Reference_Microgrid_6Bus")
    model.buses = [
        Bus(id=1, name="PCC_Bus", voltage_kv=0.4, type="Slack", x=300, y=200),
        Bus(id=2, name="Solar_Bus", voltage_kv=0.4, type="PV", x=100, y=100),
        Bus(id=3, name="Battery_Bus", voltage_kv=0.4, type="PQ", x=500, y=100),
        Bus(id=4, name="Load_A_Bus", voltage_kv=0.4, type="PQ", x=100, y=300),
        Bus(id=5, name="Load_B_Bus", voltage_kv=0.4, type="PQ", x=500, y=300),
        Bus(id=6, name="Grid_Infeed", voltage_kv=0.4, type="Slack", x=300, y=50),
    ]
    model.lines = [
        Line(id=1, from_bus=6, to_bus=1, r_ohm=0.01, x_ohm=0.005, length_km=0.05),
        Line(id=2, from_bus=1, to_bus=2, r_ohm=0.05, x_ohm=0.01, length_km=0.1),
        Line(id=3, from_bus=1, to_bus=3, r_ohm=0.03, x_ohm=0.01, length_km=0.1),
        Line(id=4, from_bus=1, to_bus=4, r_ohm=0.08, x_ohm=0.02, length_km=0.3),
        Line(id=5, from_bus=1, to_bus=5, r_ohm=0.08, x_ohm=0.02, length_km=0.3),
    ]
    model.generators = [
        Generator(id=1, bus_id=6, p_mw=0.5),
        Generator(id=2, bus_id=2, p_mw=0.1),
    ]
    model.loads = [
        Load(id=1, bus_id=4, p_mw=0.05, priority=1),
        Load(id=2, bus_id=5, p_mw=0.03, priority=2),
    ]
    return model


def start_system():
    """Start the complete fault detection system.
    
    Initializes the EventBus, all agents, the GridEmulator with a circuit
    model, and the C++ DSP pipeline (if available). Starts the emulator
    simulation thread and all agents.
    """
    if st.session_state.system_running:
        add_log("System already running", "WARNING")
        return

    add_log("Starting DC Microgrid Fault Detection System...", "INFO")

    try:
        # 1. Create EventBus and infrastructure
        bus = EventBus()
        obs = Observability()
        registry = AgentRegistry()

        # 2. Create GridEmulator and load circuit
        emulator = GridEmulator(sample_rate=20000, base_voltage=400.0)

        # Load circuit model (use existing or create reference)
        circuit = st.session_state.circuit_model
        if circuit is None:
            circuit = create_reference_circuit()
            st.session_state.circuit_model = circuit
            add_log("Loaded reference 6-bus microgrid circuit", "INFO")

        emulator.load_circuit(circuit)
        add_log(f"Circuit loaded: {len(circuit.buses)} buses, {len(circuit.lines)} lines", "INFO")

        # 3. Create sensor (reads from emulator)
        sensor = SimulatedADCSensor(emulator)
        relay = SimulatedRelayDriver()

        # 3a. Initialize C++ DSP pipeline EARLY (needed for DSPRunnerAgent)
        dsp_pipeline = None
        if DSP_AVAILABLE:
            try:
                dsp_pipeline = microgrid_dsp.create_pipeline(
                    window_size=128, levels=4,
                    sample_rate=20000.0, cutoff=8000.0,
                    trip_threshold=100.0
                )
                st.session_state.dsp_pipeline = dsp_pipeline
                st.session_state.dsp_available = True
                add_log("C++ DSP pipeline initialized (fast path active)", "INFO")
            except Exception as e:
                add_log(f"C++ DSP init failed, using Python fallback: {e}", "WARNING")
                st.session_state.dsp_available = False

        # 4. Create all agents
        # Increase sample rate to 10kHz for high-speed detection
        sampler = SamplerAgent("Sampler", bus, config={"sample_rate": 10000})
        sampler.set_sensor(sensor)

        # Create DSP Runner if pipeline is available
        dsp_runner = None
        high_speed_loop = None
        if dsp_pipeline:
            dsp_runner = DSPRunnerAgent("DSPRunner", bus, config={"dsp_pipeline": dsp_pipeline})
            add_log("Using C++ DSP Fast Path — Python DWT agents disabled", "INFO")

            # Start High-Speed Detection Loop (bypasses EventBus)
            try:
                high_speed_loop = HighSpeedDetectionLoop(
                    sensor, dsp_pipeline, bus,
                    sample_rate=20000, ui_throttle=100,
                )
                high_speed_loop.start()
                add_log("HighSpeedDetectionLoop active @ 20 kHz", "INFO")
            except Exception as e:
                add_log(f"HighSpeedDetectionLoop failed: {e}", "WARNING")
                high_speed_loop = None
        else:
            add_log("Using Python DSP Fallback (C++ unavailable)", "WARNING")

        # Python DWT chain — only when C++ DSP is NOT available
        window_mgr = None
        dwt_engine = None
        detail_analyzer = None
        if not dsp_pipeline:
            window_mgr = WindowManagerAgent("WindowManager", bus, config={"window_size": 128})
            dwt_engine = DWTEngineAgent("DWTEngine", bus, config={
                "wavelet": "db4", "level": 4, "mode": "symmetric"
            })
            detail_analyzer = DetailAnalyzerAgent("DetailAnalyzer", bus)

        fault_locator = PreciseFaultLocatorAgent("FaultLocator", bus)
        fault_locator.emulator = emulator

        threshold_guard = ThresholdGuardAgent("ThresholdGuard", bus, config={
            "d1_peak_threshold": 50.0
        })
        energy_monitor = EnergyMonitorAgent("EnergyMonitor", bus)
        fault_voter = FaultVoterAgent("FaultVoter", bus)

        trip_sequencer = TripSequencerAgent("TripSequencer", bus, config={"relay_driver": relay})
        zeta_logic = ZetaLogicAgent("ZetaLogic", bus)

        health_monitor = HealthMonitorAgent("HealthMonitor", bus, config={
            "check_interval": 2.0
        })
        ai_classifier = AIClassifierAgent("AIClassifier", bus)
        replay_recorder = ReplayRecorderAgent("ReplayRecorder", bus)
        report_generator = ReportGeneratorAgent("ReportGenerator", bus)

        # 5. Create UI bridge agent (downsample 50x to 200Hz)
        bridge = BridgeAgent("UIBridge", bus, config={"downsample_factor": 50})

        # 6. Register all agents
        agents = [
            sampler, fault_locator,
            threshold_guard, energy_monitor, fault_voter,
            trip_sequencer, zeta_logic,
            health_monitor, ai_classifier, replay_recorder, report_generator,
            bridge
        ]
        # Add Python DWT agents only if C++ is NOT available
        if window_mgr:
            agents.extend([window_mgr, dwt_engine, detail_analyzer])
        for agent in agents:
            registry.register(agent)
            
        if dsp_runner:
            registry.register(dsp_runner)

        # 7. Setup and start all agents
        registry.start_all()
        add_log(f"Started {len(agents)} agents", "INFO")

        # 8. Subscribe UI bridge to additional events for visualization
        bus.subscribe(DWTResultEvent, lambda e: _on_dwt_result(e, bridge.get_queue()))
        bus.subscribe(FaultLocationEvent, lambda e: bridge.get_queue().put(e))
        bus.subscribe(HealthStatusEvent, lambda e: bridge.get_queue().put(e))
        bus.subscribe(AIAnalysisEvent, lambda e: bridge.get_queue().put(e))

        # 9. Start emulator simulation thread
        emulator.start()
        add_log("Grid emulator started (simulation thread active)", "INFO")



        # 11. Store references in session state
        st.session_state.event_bus = bus
        st.session_state.registry = registry
        st.session_state.emulator = emulator
        st.session_state.bridge_agent = bridge
        st.session_state.high_speed_loop = high_speed_loop
        st.session_state.system_running = True

        dsp_label = "C++ DSP" if dsp_pipeline else "Python Fallback"
        add_log(f"✅ System started successfully! (DSP: {dsp_label})", "INFO")

    except Exception as e:
        add_log(f"❌ Failed to start system: {e}", "ERROR")
        logger.exception("System start failed")


def stop_system():
    """Stop the fault detection system and clean up resources."""
    if not st.session_state.system_running:
        return

    add_log("Stopping system...", "INFO")

    try:
        # Stop high-speed loop FIRST (it touches the sensor & pipeline)
        hsl = st.session_state.get("high_speed_loop")
        if hsl is not None:
            hsl.stop()
            st.session_state.high_speed_loop = None

        if st.session_state.registry:
            st.session_state.registry.stop_all()
        if st.session_state.emulator:
            st.session_state.emulator.stop()

        st.session_state.system_running = False
        add_log("System stopped", "INFO")
    except Exception as e:
        add_log(f"Error during shutdown: {e}", "ERROR")


def inject_fault(fault_type: str, severity: float, location: str, properties: dict = None):
    """Inject a fault into the grid emulator.
    
    Args:
        fault_type: One of L2L, L2G, ARC, NOISE, DRIFT, SENSOR_FAIL
        severity: Severity 0.0-1.0
        location: Node ID where fault occurs
        properties: Additional properties (e.g., distance)
    """
    emulator = st.session_state.emulator
    if not emulator:
        add_log("Cannot inject fault: system not running", "WARNING")
        return

    props = properties or {}
    emulator.inject_fault(fault_type, severity, location, properties=props)

    st.session_state.fault_active = True
    st.session_state.fault_type = fault_type
    add_log(f"⚡ Fault injected: {fault_type} at {location} (severity: {severity})", "WARNING")


def clear_fault():
    """Clear the active fault and restore normal operation."""
    emulator = st.session_state.emulator
    if not emulator:
        return

    emulator.clear_fault()
    st.session_state.fault_active = False
    st.session_state.fault_type = "NONE"
    st.session_state.fault_location = None
    st.session_state.fault_distance = None
    st.session_state.fault_zone = None
    st.session_state.trip_active = False
    add_log("Fault cleared, system restored", "INFO")


def _on_dwt_result(event, data_queue):
    """Handle DWT result events — store coefficients for visualization."""
    try:
        data_queue.put(event)
    except Exception:
        pass


def process_events():
    """Drain the event queue and update session state with latest data.
    
    This is called once per Streamlit rerun cycle to pull all pending
    events from the background agents into the UI state.
    """
    bridge = st.session_state.get("bridge_agent")
    if not bridge:
        return

    data_queue = bridge.get_queue()
    max_events = 200  # Process at most N events per cycle to prevent blocking

    for _ in range(max_events):
        try:
            event = data_queue.get_nowait()
        except queue.Empty:
            break

        _handle_event(event)


def _handle_event(event):
    """Route a single event to the appropriate session state update."""
    event_type = type(event).__name__

    if event_type == "VoltageSampleEvent":
        voltage = getattr(event, "voltage", getattr(event, "value", 0))
        st.session_state.voltage_data.append(voltage)
        if len(st.session_state.voltage_data) > 500:
            st.session_state.voltage_data = st.session_state.voltage_data[-500:]

        # DSP Processing logic moved to DSPRunnerAgent (background thread)
        # to avoid blocking UI event loop and improve detection speed.

    elif event_type == "ProcessingResultEvent":
        st.session_state.dwt_energy["D1"] = getattr(event, "d1_energy", 0)
        d1_peak = getattr(event, "d1_peak", 0)
        if getattr(event, "is_faulty", False):
            st.session_state.fault_active = True

    elif event_type == "DWTResultEvent":
        energy = getattr(event, "energy_levels", {})
        if energy:
            st.session_state.dwt_energy = energy
            st.session_state.energy_history.append(energy.copy())
            if len(st.session_state.energy_history) > 200:
                st.session_state.energy_history = st.session_state.energy_history[-200:]
        
        # Store actual coefficients for wavelet inspector
        coeffs = getattr(event, "coeffs", None)
        if coeffs:
            st.session_state.dwt_coefficients = coeffs

    elif event_type == "FaultLocationEvent":
        st.session_state.fault_location = getattr(event, "zone", "Unknown")
        st.session_state.fault_distance = getattr(event, "distance_m", 0)
        st.session_state.fault_zone = getattr(event, "zone", "Unknown")

        st.session_state.fault_events.append({
            "time": time.time(),
            "zone": st.session_state.fault_zone,
            "distance": st.session_state.fault_distance,
            "type": st.session_state.fault_type,
        })

    elif event_type == "SystemTripEvent":
        st.session_state.trip_active = True
        add_log(f"🔴 SYSTEM TRIP: {getattr(event, 'reason', 'Unknown')}", "ERROR")

    elif event_type == "HealthStatusEvent":
        st.session_state.health_data = {
            "cpu": getattr(event, "cpu_usage", 0),
            "memory": getattr(event, "memory_usage", 0),
            "eps": getattr(event, "events_per_second", 0),
            "latency": getattr(event, "latency_avg_ms", 0),
            "uptime": getattr(event, "uptime_s", 0),
        }
        st.session_state.health_history.append(st.session_state.health_data.copy())
        if len(st.session_state.health_history) > 100:
            st.session_state.health_history = st.session_state.health_history[-100:]

    elif event_type == "AIAnalysisEvent":
        st.session_state.ai_diagnosis = getattr(event, "diagnosis", None)
        st.session_state.ai_probable_causes = getattr(event, "probable_causes", [])


def get_node_ids() -> list:
    """Get list of node IDs from the loaded topology.
    
    Returns actual node IDs from the emulator if running,
    otherwise returns IDs from the circuit model.
    """
    emulator = st.session_state.get("emulator")
    if emulator and hasattr(emulator, "topology") and emulator.topology.nodes:
        return list(emulator.topology.nodes.keys())

    circuit = st.session_state.get("circuit_model")
    if circuit and circuit.buses:
        return [str(b.id) for b in circuit.buses]

    return ["1", "2", "3", "4", "5", "6"]  # Default reference grid


def get_per_node_voltages() -> dict:
    """Get current voltage for each node from the emulator."""
    emulator = st.session_state.get("emulator")
    result = {}
    if emulator and hasattr(emulator, "topology"):
        for node_id, node in emulator.topology.nodes.items():
            result[node_id] = {
                "voltage": getattr(node, "voltage", 0),
                "name": getattr(node, "name", node_id),
                "status": getattr(node, "status", "UNKNOWN"),
            }
    return result


def get_node_histories() -> dict:
    """Get voltage history arrays for all nodes from the emulator."""
    emulator = st.session_state.get("emulator")
    result = {}
    if emulator and hasattr(emulator, "history"):
        for node_id in emulator.history:
            result[node_id] = emulator.get_history(node_id)
    return result
