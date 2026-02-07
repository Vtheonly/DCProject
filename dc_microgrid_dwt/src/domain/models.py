"""
Domain Models Module - Industrial DC Microgrid Platform

Contains data models for grid topology, reports, and analysis results.
These are business objects that aren't events but represent system state.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import uuid


# ==============================================================================
# ENUMERATIONS
# ==============================================================================

class NodeType(Enum):
    """Types of nodes in the grid topology."""
    SOURCE = "SOURCE"           # Power source (solar, utility, generator)
    BUS = "BUS"                 # DC bus / busbar
    LOAD = "LOAD"               # Load (critical, non-critical)
    STORAGE = "STORAGE"         # Battery storage
    CONVERTER = "CONVERTER"     # DC-DC converter / inverter
    SENSOR = "SENSOR"           # Measurement point


class NodeStatus(Enum):
    """Status of a grid node."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    FAULT = "FAULT"
    ISOLATED = "ISOLATED"
    MAINTENANCE = "MAINTENANCE"


class ConnectionStatus(Enum):
    """Status of a grid connection."""
    CLOSED = "CLOSED"           # Normal operation
    OPEN = "OPEN"               # Disconnected
    FAULTED = "FAULTED"         # Fault detected on line


class FaultType(Enum):
    """Types of faults that can occur."""
    NONE = "NONE"
    LINE_TO_LINE = "L2L"
    LINE_TO_GROUND = "L2G"
    ARC_FAULT = "ARC"
    OVERCURRENT = "OVERCURRENT"
    OVERVOLTAGE = "OVERVOLTAGE"
    UNDERVOLTAGE = "UNDERVOLTAGE"
    NOISE = "NOISE"
    DRIFT = "DRIFT"
    SENSOR_FAILURE = "SENSOR_FAIL"


class ConverterMode(Enum):
    """Converter operating modes."""
    AUTO = "AUTO"               # PID control
    MANUAL = "MANUAL"           # Direct duty cycle control
    SAFE = "SAFE"               # Protective mode


# ==============================================================================
# GRID TOPOLOGY MODELS
# ==============================================================================

@dataclass
class GridNode:
    """Represents a node in the DC microgrid Digital Twin."""
    node_id: str
    node_type: NodeType
    name: str
    status: NodeStatus = NodeStatus.ACTIVE
    voltage: float = 0.0
    current: float = 0.0
    power: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    position: tuple = (0.0, 0.0)  # For visualization
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "status": self.status.value,
            "voltage": self.voltage,
            "current": self.current,
            "power": self.power,
            "properties": self.properties,
            "position": self.position
        }


@dataclass
class GridConnection:
    """Represents a connection between two nodes."""
    connection_id: str
    from_node: str
    to_node: str
    status: ConnectionStatus = ConnectionStatus.CLOSED
    impedance: float = 0.0
    current_flow: float = 0.0
    max_current: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "status": self.status.value,
            "impedance": self.impedance,
            "current_flow": self.current_flow,
            "max_current": self.max_current
        }


@dataclass
class GridTopology:
    """Complete grid topology model."""
    nodes: Dict[str, GridNode] = field(default_factory=dict)
    connections: Dict[str, GridConnection] = field(default_factory=dict)
    
    def add_node(self, node: GridNode):
        self.nodes[node.node_id] = node
    
    def add_connection(self, conn: GridConnection):
        self.connections[conn.connection_id] = conn
    
    def get_node(self, node_id: str) -> Optional[GridNode]:
        return self.nodes.get(node_id)
    
    def set_node_status(self, node_id: str, status: NodeStatus):
        if node_id in self.nodes:
            self.nodes[node_id].status = status
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "connections": {k: v.to_dict() for k, v in self.connections.items()}
        }


# ==============================================================================
# FAULT ANALYSIS MODELS
# ==============================================================================

@dataclass
class FaultDiagnosis:
    """AI-generated fault diagnosis."""
    fault_type: FaultType
    probability: float
    confidence: float
    probable_causes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class TimelineStep:
    """A single step in the fault timeline."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = 0.0
    step_name: str = ""
    agent: str = ""
    description: str = ""
    latency_from_start_ms: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultTimeline:
    """Complete fault event timeline (Black Box)."""
    timeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fault_start_time: float = 0.0
    trip_time: Optional[float] = None
    total_latency_ms: float = 0.0
    steps: List[TimelineStep] = field(default_factory=list)
    
    def add_step(self, step: TimelineStep):
        self.steps.append(step)
        if self.fault_start_time > 0:
            step.latency_from_start_ms = (step.timestamp - self.fault_start_time) * 1000
    
    def calculate_total_latency(self):
        if self.trip_time and self.fault_start_time:
            self.total_latency_ms = (self.trip_time - self.fault_start_time) * 1000


# ==============================================================================
# REPORT MODELS
# ==============================================================================

@dataclass
class FaultReport:
    """Structured fault incident report."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = field(default_factory=datetime.now)
    scenario_name: str = ""
    fault_type: str = ""
    fault_severity: float = 0.0
    detection_latency_ms: float = 0.0
    trip_triggered: bool = False
    timeline: Optional[FaultTimeline] = None
    diagnosis: Optional[FaultDiagnosis] = None
    system_health_at_fault: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Algorithm benchmarking comparison result."""
    benchmark_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    algorithms_tested: List[str] = field(default_factory=list)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    winner: str = ""
    summary: str = ""


@dataclass
class DailyReport:
    """Daily operational summary report."""
    report_date: datetime = field(default_factory=datetime.now)
    uptime_percent: float = 100.0
    total_faults: int = 0
    false_positives: int = 0
    avg_detection_latency_ms: float = 0.0
    energy_generated_kwh: float = 0.0
    incidents: List[FaultReport] = field(default_factory=list)


# ==============================================================================
# CONVERTER CONTROL MODELS
# ==============================================================================

@dataclass
class ConverterState:
    """Zeta converter current state."""
    duty_cycle: float = 0.45
    target_voltage: float = 400.0
    actual_voltage: float = 400.0
    actual_current: float = 0.0
    mode: ConverterMode = ConverterMode.AUTO
    current_limit: float = 20.0
    temperature: float = 25.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duty_cycle": self.duty_cycle,
            "target_voltage": self.target_voltage,
            "actual_voltage": self.actual_voltage,
            "actual_current": self.actual_current,
            "mode": self.mode.value,
            "current_limit": self.current_limit,
            "temperature": self.temperature
        }
