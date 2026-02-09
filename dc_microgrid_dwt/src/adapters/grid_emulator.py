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
        
        # Fault state
        self.fault_config = FaultConfig()
        self.status = "NORMAL"
        self._lock = threading.Lock()
        
        # Time tracking for signal generation
        self._sample_count = 0
        self._start_time = time.time()
        
        # Initialize default topology
        self.topology = GridTopology()
        self._init_default_topology()
        
        # Active node for voltage reading
        self.active_node = "BUS_DC"
        
    def _init_default_topology(self):
        """Initialize default grid topology for Digital Twin."""
        # Create nodes
        nodes = [
            GridNode(
                node_id="SOURCE_A",
                node_type=NodeType.SOURCE,
                name="Solar Array A",
                voltage=400.0,
                current=10.0,
                power=4000.0,
                position=(0.0, 0.5)
            ),
            GridNode(
                node_id="BUS_DC",
                node_type=NodeType.BUS,
                name="Main DC Bus",
                voltage=400.0,
                current=25.0,
                power=10000.0,
                position=(0.5, 0.5)
            ),
            GridNode(
                node_id="LOAD_CRITICAL",
                node_type=NodeType.LOAD,
                name="Critical Load",
                voltage=400.0,
                current=8.0,
                power=3200.0,
                position=(1.0, 0.3)
            ),
            GridNode(
                node_id="BATTERY",
                node_type=NodeType.STORAGE,
                name="Battery Storage",
                voltage=400.0,
                current=-3.0,  # Charging
                power=-1200.0,
                position=(1.0, 0.7),
                properties={"soc": 75.0}
            ),
            GridNode(
                node_id="CONVERTER",
                node_type=NodeType.CONVERTER,
                name="Zeta Converter",
                voltage=400.0,
                current=12.0,
                power=4800.0,
                position=(0.5, 0.2)
            )
        ]
        
        for node in nodes:
            self.topology.add_node(node)
        
        # Create connections
        connections = [
            GridConnection(
                connection_id="LINE_1",
                from_node="SOURCE_A",
                to_node="BUS_DC",
                current_flow=10.0
            ),
            GridConnection(
                connection_id="LINE_2",
                from_node="BUS_DC",
                to_node="LOAD_CRITICAL",
                current_flow=8.0
            ),
            GridConnection(
                connection_id="LINE_3",
                from_node="BUS_DC",
                to_node="BATTERY",
                current_flow=-3.0
            ),
            GridConnection(
                connection_id="LINE_4",
                from_node="BUS_DC",
                to_node="CONVERTER",
                current_flow=12.0
            )
        ]
        
        for conn in connections:
            self.topology.add_connection(conn)

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
        Generate a single voltage sample with fault effects.
        Implements ISensor interface.
        """
        with self._lock:
            self._sample_count += 1
            t = self._sample_count / self.sample_rate
            
            # Base signal
            voltage = self.base_voltage
            
            # Add base noise
            voltage += np.random.normal(0, self.noise_level)
            
            # Apply fault effects
            if self.fault_config.active:
                voltage = self._apply_fault_effect(voltage, t)
                
            # Update topology node voltage
            if self.active_node in self.topology.nodes:
                self.topology.nodes[self.active_node].voltage = voltage
                
            return float(voltage)

    def read_batch(self, count: int) -> List[float]:
        """Read voltage samples."""
        return [self.read() for _ in range(count)]

    def _apply_fault_effect(self, voltage: float, t: float) -> float:
        """Apply fault effects to voltage based on active fault config."""
        elapsed = time.time() - self.fault_config.start_time
        sev = self.fault_config.severity
        ft = self.fault_config.fault_type
        
        # Get distance factor (default 10m if not specified)
        props = self.fault_config.properties or {}
        distance_m = props.get("distance", 10.0)
        
        # Physics: High frequencies attenuate over distance
        # Simple simulation: Reduce transient amplitude based on distance
        # normalized to some reference (e.g., 100m)
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
