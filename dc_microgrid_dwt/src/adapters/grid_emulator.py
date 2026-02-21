"""
Grid Emulator Module - Industrial DC Microgrid Platform

Advanced grid emulator with fault injection capabilities and
Digital Twin topology management. Generates realistic signals
with physics-based fault modeling.
"""
import numpy as np
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.domain.interfaces import IGridEmulator, ISensor
from src.domain.models import (
    GridTopology, GridNode, GridConnection,
    NodeType, NodeStatus, ConnectionStatus, FaultType
)
from src.domain.circuit import CircuitModel, Bus, Line, Generator, Load

logger = logging.getLogger(__name__)


@dataclass
class FaultConfig:
    """Configuration for active fault injection."""
    active: bool = False
    fault_type: FaultType = FaultType.NONE
    severity: float = 0.0
    location: str = ""
    start_time: float = 0.0
    duration: float = float('inf')
    properties: Dict[str, Any] = None  # New field for extra params like distance


class GridEmulator(IGridEmulator, ISensor):
    """
    Advanced Grid Emulator for fault injection and Digital Twin simulation.
    
    Capabilities:
    - Multi-node topology simulation
    - Physics-based fault signal generation
    - Real-time state management
    - Interactive fault injection API
    
    Fault Types Supported:
    - L2L (Line-to-Line): Sudden voltage drop with high-freq transient
    - L2G (Line-to-Ground): Voltage drop with oscillation
    - ARC: Intermittent high-frequency noise
    - NOISE: General high-frequency noise injection
    - DRIFT: Gradual voltage sag/swell
    - SENSOR_FAIL: Sensor reading anomalies
    """
    
    def __init__(self, base_voltage: float = 400.0, sample_rate: int = 20000):
        self.base_voltage = base_voltage
        self.sample_rate = sample_rate
        self.noise_level = 0.5  # Base noise sigma
        self._running = False
        self._thread = None
        self._last_step_time = time.time()
        
        # Fault state        
        # Fault state
        self.fault_config = FaultConfig()
        self.status = "NORMAL"
        self._lock = threading.Lock()
        
        # Time tracking for signal generation
        self._sample_count = 0
        self._start_time = time.time()
        
        # Multi-Node History Buffers (1 second window at 20kHz)
        self.history_size = 20000 
        self.history: Dict[str, np.ndarray] = {} 
        # Active node for reading (sensor location)
        self.active_node = "1" 
        self.history_idx = 0
        
        # Initialize default topology (EMPTY)
        self.topology = None # Wait for load_circuit
        self._init_default_topology()

    def load_circuit(self, circuit: CircuitModel):
        """
        Load a CircuitModel and update the internal GridTopology.
        This effectively "reprograms" the emulator with the new circuit.
        """
        with self._lock:
            # Clear existing topology
            self.topology = GridTopology()
            
            # Map Buses -> GridNodes
            for bus in circuit.buses:
                node_id = str(bus.id)
                node = GridNode(
                    node_id=node_id,
                    node_type=NodeType.BUS,
                    name=bus.name,
                    voltage=float(bus.voltage_kv * 1000.0), # kV -> V
                    current=0.0,
                    power=0.0,
                    position=(bus.x, bus.y)
                )
                self.topology.add_node(node)
                
                # Initialize history buffer for this node
                self.history[node_id] = np.full(self.history_size, 400.0) # Init with 400V
                
            # Map Generators -> Update Nodes properties
            for gen in circuit.generators:
                node_id = str(gen.bus_id)
                if node_id in self.topology.nodes:
                    node = self.topology.nodes[node_id]
                    node.node_type = NodeType.SOURCE
                    node.power = float(gen.p_mw * 1e6) # MW -> W

            # Map Loads -> Update Nodes properties
            for load in circuit.loads:
                node_id = str(load.bus_id)
                if node_id in self.topology.nodes:
                    node = self.topology.nodes[node_id]
                    node.node_type = NodeType.LOAD
                    node.power = float(load.p_mw * 1e6) # MW -> W
            
            # Map Lines -> GridConnections
            for line in circuit.lines:
                conn = GridConnection(
                    connection_id=str(line.id),
                    from_node=str(line.from_bus),
                    to_node=str(line.to_bus),
                    impedance=float(line.r_ohm),
                    status=ConnectionStatus.CLOSED if line.status == 1 else ConnectionStatus.OPEN
                )
                self.topology.add_connection(conn)
            
            logger.info(f"Loaded circuit '{circuit.name}' with {len(circuit.buses)} buses into Emulator.")
        
        # Active node for voltage reading
        if circuit.buses:
            self.active_node = str(circuit.buses[0].id)
        else:
            self.active_node = "1"
        
    def _init_default_topology(self):
        """Initialize default grid topology for Digital Twin."""
        # Intentionally empty - enforced Model-Driven approach
        pass

    def start(self):
        """Start the background simulation thread."""
        if self.topology is None:
            logger.warning("Starting emulator without a loaded circuit model. Waiting for load_circuit().")
            
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Grid Emulator started")

    def stop(self):
        """Stop the background simulation thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Grid Emulator stopped")

    def _run_loop(self):
        """Main simulation loop running in background thread."""
        while self._running:
            try:
                self._run_simulation_step()
                time.sleep(1.0 / self.sample_rate)
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(0.1)

    def _run_simulation_step(self):
        """Execute one step of the physical simulation."""
        with self._lock:
            # Enforced Model-Driven: Do nothing if no circuit loaded
            if not self.topology or not self.topology.nodes:
                return

            t = self._sample_count / self.sample_rate
            self._sample_count += 1
            
            # --- UNIFIED MULTI-NODE SIMULATION ---
            # Instead of just reading one node, we simulate the physics for ALL nodes
            # based on the fault condition.
            
            for node_id, node in self.topology.nodes.items():
                # Base voltage (ideal)
                v = 400.0 # Simplify: All nodes start at nominal
                
                # Add noise
                v += np.random.normal(0, self.noise_level)
                
                # Apply Fault Logic if active
                if self.fault_config.active:
                    # Calculate distance from fault to this node
                    # (Simplified: Manhattan distance from 'location' to 'node_id')
                    # Ideally use graph traversal, but for now we rely on the topology logic
                    
                    dist_to_fault = 0.0
                    if self.fault_config.location == node_id:
                        dist_to_fault = 0.0
                    else:
                        # Heuristic: If we have a 'distance' property on connections, use it.
                        # For now, assume a generic distance based on "how far" it is in the list
                        # This is a PLACEHOLDER for full power flow
                        dist_to_fault = 100.0 # Default "far"
                        
                        # Hack for Reference Grid:
                        # Solar(2)-PCC(1)-Battery(3)-LoadA(4)-LoadB(5)
                        # We can implement a tiny lookup for the reference grid for realism
                        if self.fault_config.location == "1": # PCC
                            if node_id in ["2", "3"]: dist_to_fault = 100
                            elif node_id in ["4", "5"]: dist_to_fault = 300
                            elif node_id == "6": dist_to_fault = 400
                            
                    # Apply fault effect with distance-based attenuation (ALL fault types)
                    v = self._apply_fault_effect(v, t, distance_m=dist_to_fault)
                
                # Update Node State
                node.voltage = v
                
                # Update Buffer using ring buffer logic
                # We use a global index for efficiency
                idx = self._sample_count % self.history_size
                self.history[node_id][idx] = v
                self.history_idx = idx

        
    def get_history(self, node_id: str) -> np.ndarray:
        """Get ordered history buffer for a node."""
        if node_id not in self.history:
            return np.array([])
        
        # Roll buffer so newest is last
        return np.roll(self.history[node_id], -self.history_idx - 1)

    def inject_fault(self, fault_type: str, severity: float, location: str = "BUS_DC", properties: Dict[str, Any] = None):
        """
        Inject a fault into the grid simulation.
        
        Args:
            fault_type: Type of fault (L2L, L2G, ARC, NOISE, DRIFT, SENSOR_FAIL)
            severity: Severity level 0.0 - 1.0
            location: Node ID where fault occurs
            properties: Additional properties like 'distance'
        """
        with self._lock:
            try:
                ft = FaultType(fault_type) if isinstance(fault_type, str) else fault_type
            except ValueError:
                ft = FaultType.NONE
                logger.error(f"Unknown fault type: {fault_type}")
                
            self.fault_config = FaultConfig(
                active=True,
                fault_type=ft,
                severity=min(1.0, max(0.0, severity)),
                location=location,
                start_time=time.time(),
                properties=properties or {}
            )
            
            # Update node status
            if location in self.topology.nodes:
                self.topology.set_node_status(location, NodeStatus.FAULT)
            
            self.status = f"FAULT_{ft.value}"
            logger.warning(f"Fault injected: {ft.value} at {location} (severity: {severity}) props={properties}")

    def clear_fault(self):
        """Clear any active faults and restore normal operation."""
        with self._lock:
            if self.fault_config.active:
                location = self.fault_config.location
                if location in self.topology.nodes:
                    self.topology.set_node_status(location, NodeStatus.ACTIVE)
                    
            self.fault_config = FaultConfig()
            self.status = "NORMAL"
            logger.info("Fault cleared, system restored to normal")

    def read(self) -> float:
        """
        Return the current voltage of the active node (simulated ADC).
        Implements ISensor interface.
        """
        with self._lock:
            # Enforced Model-Driven: Return 0.0 if no circuit loaded
            if not self.topology or not self.topology.nodes:
                return 0.0

            # Determine active node (default to first bus or specific ID)
            # For now, let's assume we read from "Main_PCC_Bus" or similar
            # In a real system, we'd have multiple sensors.
            # We can use a configured 'sensor_node_id' 
            # Determine active node (default to first bus or specific ID)
            target_node = "1" 
            
            if hasattr(self, 'active_node') and self.active_node in self.topology.nodes:
                target_node = self.active_node
            elif "1" not in self.history:
                # Fallback to any node if available
                if not self.history:
                    return 0.0
                target_node = next(iter(self.history.keys()))
                
            if target_node in self.topology.nodes:
                node = self.topology.nodes[target_node]
                val = node.voltage
                return float(val)
                
            return 0.0

    def read_batch(self, count: int) -> List[float]:
        """Read voltage samples."""
        return [self.read() for _ in range(count)]

    def _apply_fault_effect(self, voltage: float, t: float, distance_m: float = None) -> float:
        """Apply fault effects to voltage based on active fault config.
        
        Args:
            voltage: Current voltage value
            t: Time within simulation step
            distance_m: Distance from fault to measurement point (meters).
                        If None, falls back to fault config properties.
        """
        elapsed = time.time() - self.fault_config.start_time
        sev = self.fault_config.severity
        ft = self.fault_config.fault_type
        
        # Get distance factor
        if distance_m is None:
            props = self.fault_config.properties or {}
            distance_m = props.get("distance", 10.0)
        
        # Physics: High frequencies attenuate over distance
        attenuation = 1.0 / (1.0 + (distance_m / 100.0))
        
        if ft == FaultType.LINE_TO_LINE:
            # Sudden voltage drop with high-frequency transient
            voltage *= (1.0 - sev * 0.6)  # Up to 60% drop
            
            # Add damped oscillation (ringing)
            # Distance affects amplitude of the high-freq ringing
            freq = 5000 + np.random.uniform(-500, 500)
            damping = np.exp(-elapsed * 50)
            
            transient = sev * 100 * np.sin(2 * np.pi * freq * t) * damping
            voltage += transient * attenuation
            
        elif ft == FaultType.LINE_TO_GROUND:
            # Voltage drop with lower frequency oscillation
            voltage *= (1.0 - sev * 0.4)
            freq = 1000
            damping = np.exp(-elapsed * 20)
            
            transient = sev * 80 * np.sin(2 * np.pi * freq * t) * damping
            voltage += transient * attenuation
            
        elif ft == FaultType.ARC_FAULT:
            # Intermittent high-frequency bursts
            if np.random.random() < 0.3:  # 30% chance of arc
                arc_noise = np.random.normal(0, 50 * sev)
                high_freq = sev * 30 * np.sin(2 * np.pi * 8000 * t)
                
                # Arcs are local, but measurement is distant
                voltage += (arc_noise + high_freq) * attenuation
                
        elif ft == FaultType.NOISE:
            # High-amplitude noise injection
            voltage += np.random.normal(0, 30 * sev)
            
        elif ft == FaultType.DRIFT:
            # Gradual voltage sag/swell
            drift_rate = 50 * sev  # V/s
            voltage -= drift_rate * elapsed
            voltage = max(voltage, self.base_voltage * 0.5)  # Floor at 50%
            
        elif ft == FaultType.SENSOR_FAILURE:
            # Sensor reading anomalies
            anomaly_type = int(elapsed * 10) % 4
            if anomaly_type == 0:
                voltage = 0.0  # Zero reading
            elif anomaly_type == 1:
                voltage = self.base_voltage * 2  # Stuck high
            elif anomaly_type == 2:
                voltage = np.random.uniform(-100, 100)  # Random
            # else: normal (intermittent)
            
        return voltage

    def read_voltage(self, node_id: str) -> float:
        """Read voltage at a specific node."""
        old_active = self.active_node
        self.active_node = node_id
        voltage = self.read()
        self.active_node = old_active
        return voltage

    def get_topology(self) -> Dict[str, Any]:
        """Get current grid topology as dictionary."""
        with self._lock:
            return self.topology.to_dict()

    def set_node_status(self, node_id: str, status: str):
        """Set status of a specific node."""
        with self._lock:
            try:
                status_enum = NodeStatus(status)
            except ValueError:
                status_enum = NodeStatus.ACTIVE
                
            self.topology.set_node_status(node_id, status_enum)

    def get_status(self) -> str:
        """Get current emulator status."""
        return self.status

    def get_fault_info(self) -> Dict[str, Any]:
        """Get information about active fault."""
        with self._lock:
            if not self.fault_config.active:
                return {"active": False}
                
            return {
                "active": True,
                "type": self.fault_config.fault_type.value,
                "severity": self.fault_config.severity,
                "location": self.fault_config.location,
                "elapsed_s": time.time() - self.fault_config.start_time
            }

    def reset(self):
        """Reset emulator to initial state."""
        with self._lock:
            self.fault_config = FaultConfig()
            self.status = "NORMAL"
            self._sample_count = 0
            self._start_time = time.time()
            self._init_default_topology()
            logger.info("Grid emulator reset to initial state")

    def generate_signal(self, duration_s: float, scenario: str = "NORMAL") -> np.ndarray:
        """
        Generate a complete signal array for a given duration.
        
        Args:
            duration_s: Duration in seconds
            scenario: Scenario name (NORMAL, L2L_FAULT, NOISE, etc.)
            
        Returns:
            numpy array of voltage samples
        """
        samples = int(duration_s * self.sample_rate)
        t = np.linspace(0, duration_s, samples)
        
        # Base signal
        signal = np.ones(samples) * self.base_voltage
        signal += np.random.normal(0, self.noise_level, samples)
        
        if scenario == "L2L_FAULT":
            fault_idx = samples // 5  # Fault at 20%
            signal[fault_idx:] *= 0.4  # 60% drop
            # Add transient
            transient_len = min(100, samples - fault_idx)
            transient = 300 * np.sin(2 * np.pi * 5000 * t[:transient_len])
            transient *= np.exp(-np.linspace(0, 5, transient_len))
            signal[fault_idx:fault_idx + transient_len] += transient
            
        elif scenario == "HIGH_NOISE":
            signal += np.random.normal(0, 25, samples)
            
        elif scenario == "DRIFT":
            drift_idx = samples // 4
            drift = np.linspace(0, 100, samples - drift_idx)
            signal[drift_idx:] -= drift
            
        return signal
