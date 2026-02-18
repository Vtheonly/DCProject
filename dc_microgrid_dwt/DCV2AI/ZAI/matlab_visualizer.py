"""
MATLAB Visualization Bridge - Circuit Schematic with Fault Highlighting

This module provides real-time visualization of the DC microgrid circuit
using MATLAB for rendering. It exports circuit state, sends fault data,
and updates the MATLAB figure dynamically.

Create this file at: src/adapters/matlab_visualizer.py

Features:
- Proper electrical schematic rendering (not just node graph)
- Real-time fault location highlighting with visual indicators
- Support for line faults and bus faults
- Integration with Streamlit UI

Requirements:
- MATLAB Engine API for Python (pip install matlabengine)
- Or use the standalone mode that generates .mat files for manual MATLAB loading
"""

import os
import json
import time
import logging
import threading
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import scipy.io as sio

from src.domain.circuit import CircuitModel
from src.domain.events import FaultLocationEvent, FaultDetectedEvent, SystemTripEvent

logger = logging.getLogger(__name__)

# Try to import MATLAB engine
MATLAB_ENGINE_AVAILABLE = False
try:
    import matlab.engine
    MATLAB_ENGINE_AVAILABLE = True
    logger.info("MATLAB Engine API available")
except ImportError:
    logger.info("MATLAB Engine API not available - using file-based mode")


@dataclass
class CircuitVisualizationState:
    """Current state of the circuit for visualization."""
    buses: List[Dict[str, Any]]
    lines: List[Dict[str, Any]]
    generators: List[Dict[str, Any]]
    loads: List[Dict[str, Any]]
    fault: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0


class MatlabCircuitVisualizer:
    """
    Bridge between Python circuit model and MATLAB visualization.
    
    Provides two modes of operation:
    1. Engine Mode: Direct MATLAB engine connection for real-time updates
    2. File Mode: Export .mat files for manual loading in MATLAB
    
    The visualization shows:
    - Proper electrical schematic with bus bars, cables, generators, loads
    - Real-time fault location with pulsing indicators
    - Voltage levels on each bus
    - Component status (normal/fault)
    """
    
    def __init__(self, standalone_mode: bool = None):
        """
        Initialize the visualizer.
        
        Args:
            standalone_mode: If True, use file-based mode even if MATLAB engine
                           is available. If None, auto-detect based on availability.
        """
        if standalone_mode is None:
            standalone_mode = not MATLAB_ENGINE_AVAILABLE
            
        self.standalone_mode = standalone_mode
        self.matlab_engine = None
        self.visualizer_handle = None
        self._lock = threading.Lock()
        
        # State tracking
        self.current_state: Optional[CircuitVisualizationState] = None
        self.last_update_time = 0.0
        self.update_interval = 0.1  # Minimum seconds between updates
        
        # Output directory for file mode
        self.output_dir = Path(tempfile.gettempdir()) / "dc_microgrid_viz"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MATLAB engine if available
        if not standalone_mode and MATLAB_ENGINE_AVAILABLE:
            self._init_matlab_engine()
    
    def _init_matlab_engine(self):
        """Initialize MATLAB engine connection."""
        try:
            # Start MATLAB engine
            self.matlab_engine = matlab.engine.start_matlab()
            
            # Add the visualization script directory to MATLAB path
            script_dir = Path(__file__).parent.parent.parent / "matlab"
            if script_dir.exists():
                self.matlab_engine.addpath(str(script_dir))
            
            logger.info("MATLAB engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MATLAB engine: {e}")
            self.standalone_mode = True
            self.matlab_engine = None
    
    def update_circuit(self, circuit: CircuitModel, emulator=None):
        """
        Update the visualization with current circuit state.
        
        Args:
            circuit: The CircuitModel to visualize
            emulator: Optional GridEmulator for real-time voltage data
        """
        with self._lock:
            # Throttle updates
            now = time.time()
            if now - self.last_update_time < self.update_interval:
                return
            self.last_update_time = now
            
            # Build visualization state
            state = self._build_state(circuit, emulator)
            self.current_state = state
            
            # Export to MATLAB format
            if self.standalone_mode:
                self._export_to_file(state)
            else:
                self._update_matlab_figure(state)
    
    def highlight_fault(self, location: str, fault_type: str, severity: float, 
                       distance: float = None):
        """
        Highlight a fault location on the circuit diagram.
        
        Args:
            location: Bus ID or Line ID where fault occurred
            fault_type: Type of fault (LINE_TO_LINE, LINE_TO_GROUND, ARC_FAULT, etc.)
            severity: Severity level (0.0 - 1.0)
            distance: Distance from measurement point (for line faults)
        """
        with self._lock:
            if self.current_state is None:
                logger.warning("No circuit state to highlight fault on")
                return
            
            # Update fault in state
            fault_data = {
                'active': True,
                'location': str(location),
                'type': fault_type,
                'severity': severity,
                'distance': distance or 0.0,
                'timestamp': time.time()
            }
            self.current_state.fault = fault_data
            
            # Update visualization
            if self.standalone_mode:
                self._export_to_file(self.current_state)
            else:
                self._update_matlab_figure(self.current_state)
            
            logger.info(f"Fault highlighted: {fault_type} at {location}")
    
    def clear_fault(self):
        """Clear the fault highlight and restore normal display."""
        with self._lock:
            if self.current_state:
                self.current_state.fault = None
                
                if self.standalone_mode:
                    self._export_to_file(self.current_state)
                else:
                    self._update_matlab_figure(self.current_state)
            
            logger.info("Fault cleared from visualization")
    
    def _build_state(self, circuit: CircuitModel, emulator=None) -> CircuitVisualizationState:
        """Build visualization state from circuit model."""
        buses = []
        for bus in circuit.buses:
            bus_data = {
                'id': bus.id,
                'name': bus.name,
                'type': bus.type,
                'x': bus.x,
                'y': bus.y,
                'voltage_kv': bus.voltage_kv
            }
            
            # Add real-time voltage if emulator available
            if emulator and hasattr(emulator, 'topology'):
                node_id = str(bus.id)
                if node_id in emulator.topology.nodes:
                    node = emulator.topology.nodes[node_id]
                    bus_data['voltage'] = node.voltage / 1000.0  # V to kV
                    bus_data['status'] = node.status.value if hasattr(node.status, 'value') else str(node.status)
            
            buses.append(bus_data)
        
        lines = []
        for line in circuit.lines:
            line_data = {
                'id': line.id,
                'from_bus': line.from_bus,
                'to_bus': line.to_bus,
                'r_ohm': line.r_ohm,
                'x_ohm': line.x_ohm,
                'length_km': line.length_km,
                'status': line.status
            }
            lines.append(line_data)
        
        generators = []
        for gen in circuit.generators:
            gen_data = {
                'id': gen.id,
                'bus_id': gen.bus_id,
                'p_mw': gen.p_mw,
                'p_max_mw': gen.p_max_mw,
                'status': gen.status
            }
            generators.append(gen_data)
        
        loads = []
        for load in circuit.loads:
            load_data = {
                'id': load.id,
                'bus_id': load.bus_id,
                'p_mw': load.p_mw,
                'priority': load.priority if hasattr(load, 'priority') else 1,
                'status': load.status
            }
            loads.append(load_data)
        
        return CircuitVisualizationState(
            buses=buses,
            lines=lines,
            generators=generators,
            loads=loads,
            timestamp=time.time()
        )
    
    def _export_to_file(self, state: CircuitVisualizationState):
        """Export circuit state to .mat file for MATLAB loading."""
        try:
            # Convert to MATLAB-compatible format
            mat_data = {
                'circuit': {
                    'buses': self._list_to_struct_array(state.buses),
                    'lines': self._list_to_struct_array(state.lines),
                    'generators': self._list_to_struct_array(state.generators),
                    'loads': self._list_to_struct_array(state.loads)
                }
            }
            
            # Add fault data if present
            if state.fault:
                mat_data['fault'] = state.fault
            
            # Save to file
            output_path = self.output_dir / "circuit_state.mat"
            sio.savemat(str(output_path), mat_data)
            
            # Also save as JSON for web visualization
            json_path = self.output_dir / "circuit_state.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'buses': state.buses,
                    'lines': state.lines,
                    'generators': state.generators,
                    'loads': state.loads,
                    'fault': state.fault,
                    'timestamp': state.timestamp
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to export circuit state: {e}")
    
    def _list_to_struct_array(self, items: List[Dict]) -> np.ndarray:
        """Convert list of dicts to numpy structured array for MATLAB."""
        if not items:
            return np.array([])
        
        # Build dtype from first item
        dtype = []
        for key, value in items[0].items():
            if isinstance(value, str):
                dtype.append((key, 'U50'))
            elif isinstance(value, float):
                dtype.append((key, 'f8'))
            else:
                dtype.append((key, 'i8'))
        
        # Create structured array
        arr = np.zeros(len(items), dtype=dtype)
        for i, item in enumerate(items):
            for key, value in item.items():
                arr[i][key] = value
        
        return arr
    
    def _update_matlab_figure(self, state: CircuitVisualizationState):
        """Update the MATLAB figure directly via engine."""
        if self.matlab_engine is None:
            return
        
        try:
            # Export to temp file first
            self._export_to_file(state)
            
            # Call MATLAB visualization function
            mat_path = str(self.output_dir / "circuit_state.mat")
            
            # Load circuit data in MATLAB
            circuit_data = self.matlab_engine.load(mat_path)
            
            # Prepare fault data
            if state.fault:
                fault_data = self.matlab_engine.struct()
                fault_data['active'] = True
                fault_data['location'] = state.fault['location']
                fault_data['type'] = state.fault['type']
                fault_data['severity'] = state.fault['severity']
            else:
                fault_data = self.matlab_engine.struct()
                fault_data['active'] = False
            
            # Call visualization function
            # The MATLAB function DC_Microgrid_Visualize should be on the path
            self.visualizer_handle = self.matlab_engine.DC_Microgrid_Visualize(
                circuit_data['circuit'], fault_data, nargout=1
            )
            
        except Exception as e:
            logger.error(f"Failed to update MATLAB figure: {e}")
    
    def get_json_state(self) -> Optional[str]:
        """Get current circuit state as JSON string for web visualization."""
        if self.current_state is None:
            return None
        
        return json.dumps({
            'buses': self.current_state.buses,
            'lines': self.current_state.lines,
            'generators': self.current_state.generators,
            'loads': self.current_state.loads,
            'fault': self.current_state.fault,
            'timestamp': self.current_state.timestamp
        })
    
    def close(self):
        """Close MATLAB engine if running."""
        if self.matlab_engine:
            try:
                self.matlab_engine.quit()
            except:
                pass
            self.matlab_engine = None


class VisualizationEventHandler:
    """
    Event handler that updates the visualization when faults occur.
    
    Subscribe this handler to the EventBus to automatically update
    the circuit visualization when faults are detected.
    """
    
    def __init__(self, visualizer: MatlabCircuitVisualizer):
        self.visualizer = visualizer
    
    def on_fault_location(self, event: FaultLocationEvent):
        """Handle fault location event."""
        self.visualizer.highlight_fault(
            location=event.zone,  # or event.details.get('node_id')
            fault_type=event.details.get('fault_type', 'UNKNOWN'),
            severity=0.8,
            distance=event.distance_m
        )
    
    def on_fault_detected(self, event: FaultDetectedEvent):
        """Handle fault detection event."""
        # Map fault type string to visualization type
        fault_type = event.fault_type if hasattr(event, 'fault_type') else 'UNKNOWN'
        self.visualizer.highlight_fault(
            location=event.details.get('location', '') if hasattr(event, 'details') else '',
            fault_type=fault_type,
            severity=event.severity if hasattr(event, 'severity') else 0.7
        )
    
    def on_system_trip(self, event: SystemTripEvent):
        """Handle system trip event - highlight the trip location."""
        # Trip events may contain location info in snapshot_data
        if hasattr(event, 'snapshot_data') and event.snapshot_data:
            location = event.snapshot_data.get('fault_location', '')
            fault_type = event.snapshot_data.get('fault_type', 'LINE_TO_LINE')
            self.visualizer.highlight_fault(
                location=location,
                fault_type=fault_type,
                severity=1.0  # Full severity for trip events
            )


def create_reference_circuit_schematic() -> Dict[str, Any]:
    """
    Create a properly laid out reference circuit for schematic visualization.
    
    This function returns a circuit model with coordinates optimized for
    electrical schematic display (not just graph layout).
    
    Returns:
        Dictionary with circuit data ready for visualization
    """
    # Layout for proper electrical schematic:
    # - Main PCC bus at center top
    # - Generation on left
    # - Loads on right and bottom
    # - Clear separation of DC bus sections
    
    return {
        'buses': [
            {'id': 1, 'name': 'Main PCC', 'type': 'Slack', 'x': 0.5, 'y': 0.85, 'voltage_kv': 0.4},
            {'id': 2, 'name': 'Solar Array', 'type': 'PV', 'x': 0.15, 'y': 0.85, 'voltage_kv': 0.4},
            {'id': 3, 'name': 'Battery', 'type': 'PV', 'x': 0.85, 'y': 0.85, 'voltage_kv': 0.4},
            {'id': 4, 'name': 'Load Center A', 'type': 'PQ', 'x': 0.25, 'y': 0.35, 'voltage_kv': 0.4},
            {'id': 5, 'name': 'Load Center B', 'type': 'PQ', 'x': 0.75, 'y': 0.35, 'voltage_kv': 0.4},
            {'id': 6, 'name': 'Grid Infeed', 'type': 'Slack', 'x': 0.5, 'y': 0.95, 'voltage_kv': 0.4},
        ],
        'lines': [
            {'id': 1, 'from_bus': 6, 'to_bus': 1, 'r_ohm': 0.01, 'x_ohm': 0.005, 'length_km': 0.05},
            {'id': 2, 'from_bus': 1, 'to_bus': 2, 'r_ohm': 0.05, 'x_ohm': 0.01, 'length_km': 0.15},
            {'id': 3, 'from_bus': 1, 'to_bus': 3, 'r_ohm': 0.03, 'x_ohm': 0.01, 'length_km': 0.15},
            {'id': 4, 'from_bus': 1, 'to_bus': 4, 'r_ohm': 0.08, 'x_ohm': 0.02, 'length_km': 0.25},
            {'id': 5, 'from_bus': 1, 'to_bus': 5, 'r_ohm': 0.08, 'x_ohm': 0.02, 'length_km': 0.25},
        ],
        'generators': [
            {'id': 1, 'bus_id': 6, 'p_mw': 0.5, 'name': 'Grid'},
            {'id': 2, 'bus_id': 2, 'p_mw': 0.1, 'name': 'Solar PV'},
            {'id': 3, 'bus_id': 3, 'p_mw': 0.2, 'name': 'Battery'},
        ],
        'loads': [
            {'id': 1, 'bus_id': 4, 'p_mw': 0.15, 'name': 'Critical Load', 'priority': 1},
            {'id': 2, 'bus_id': 5, 'p_mw': 0.10, 'name': 'Normal Load', 'priority': 2},
        ]
    }
