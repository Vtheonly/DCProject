"""
Domain Interfaces Module - Industrial DC Microgrid Platform

Contains abstract base classes (interfaces) that define contracts
for agents, sensors, plugins, and other extensible components.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ==============================================================================
# CORE AGENT INTERFACE
# ==============================================================================

class IAgent(ABC):
    """Interface for all agents in the system."""
    
    @abstractmethod
    def start(self):
        """Initialize and start the agent."""
        pass

    @abstractmethod
    def stop(self):
        """Stop and cleanup the agent."""
        pass

    @abstractmethod
    def on_event(self, event: Any):
        """Handle an incoming event."""
        pass


# ==============================================================================
# HARDWARE INTERFACES
# ==============================================================================

class ISensor(ABC):
    """Interface for voltage/current sensors (real or simulated)."""
    
    @abstractmethod
    def read(self) -> float:
        """Read a single sample from the sensor."""
        pass
    
    @abstractmethod
    def read_batch(self, count: int) -> List[float]:
        """Read multiple samples."""
        pass


class IRelay(ABC):
    """Interface for relay/circuit breaker control."""
    
    @abstractmethod
    def open_relay(self) -> bool:
        """Open the relay (disconnect)."""
        pass
    
    @abstractmethod
    def close_relay(self) -> bool:
        """Close the relay (connect)."""
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """Get current relay status."""
        pass


class IConverter(ABC):
    """Interface for DC-DC converter control."""
    
    @abstractmethod
    def set_duty_cycle(self, duty: float) -> bool:
        """Set PWM duty cycle (0.0 - 1.0)."""
        pass
    
    @abstractmethod
    def set_target_voltage(self, voltage: float) -> bool:
        """Set target output voltage."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get converter current state."""
        pass
    
    @abstractmethod
    def set_mode(self, mode: str) -> bool:
        """Set operating mode (AUTO, MANUAL, SAFE)."""
        pass


# ==============================================================================
# PLUGIN INTERFACE
# ==============================================================================

class IPlugin(ABC):
    """Interface for runtime-loadable plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin unique identifier."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version string."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
    
    @abstractmethod
    def initialize(self, bus: Any, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with event bus and config."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cleanup plugin resources."""
        pass
    
    @abstractmethod
    def get_agents(self) -> List[IAgent]:
        """Return list of agents provided by this plugin."""
        pass


# ==============================================================================
# REPORT INTERFACE
# ==============================================================================

class IReporter(ABC):
    """Interface for report generators."""
    
    @abstractmethod
    def generate_report(self, data: Dict[str, Any]) -> bytes:
        """Generate report from data, return bytes (PDF/HTML)."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported output formats."""
        pass


# ==============================================================================
# FAULT DETECTOR INTERFACE
# ==============================================================================

class IFaultDetector(ABC):
    """Interface for fault detection algorithms."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name."""
        pass
    
    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze signal data and return detection result."""
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """Return confidence level of last detection."""
        pass


# ==============================================================================
# DATA RECORDER INTERFACE
# ==============================================================================

class IRecorder(ABC):
    """Interface for event recording (replay system)."""
    
    @abstractmethod
    def record(self, event: Any):
        """Record an event."""
        pass
    
    @abstractmethod
    def get_recording(self, start_time: float, end_time: float) -> List[Any]:
        """Get recorded events in time range."""
        pass
    
    @abstractmethod
    def save_to_file(self, filepath: str) -> bool:
        """Save recording to file."""
        pass
    
    @abstractmethod
    def load_from_file(self, filepath: str) -> bool:
        """Load recording from file."""
        pass


# ==============================================================================
# GRID EMULATOR INTERFACE
# ==============================================================================

class IGridEmulator(ABC):
    """Interface for grid simulation/emulation."""
    
    @abstractmethod
    def inject_fault(self, fault_type: str, severity: float, location: str):
        """Inject a fault into the grid simulation."""
        pass
    
    @abstractmethod
    def clear_fault(self):
        """Clear any active faults."""
        pass
    
    @abstractmethod
    def get_topology(self) -> Dict[str, Any]:
        """Get current grid topology."""
        pass
    
    @abstractmethod
    def set_node_status(self, node_id: str, status: str):
        """Set status of a specific node."""
        pass
    
    @abstractmethod
    def read_voltage(self, node_id: str) -> float:
        """Read voltage at a specific node."""
        pass
