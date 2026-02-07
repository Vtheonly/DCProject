"""
Domain Events Module - Industrial DC Microgrid Platform

Contains all event types used for inter-agent communication in the
event-driven architecture. Events are immutable data containers that
flow through the EventBus.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


# ==============================================================================
# BASE EVENT
# ==============================================================================

@dataclass
class BaseEvent:
    """Base class for all events in the system."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = "System"
    payload: Optional[Dict[str, Any]] = None


# ==============================================================================
# INGESTION EVENTS (Sensor → Processing)
# ==============================================================================

@dataclass
class VoltageSampleEvent(BaseEvent):
    """Raw voltage sample from ADC/sensor."""
    voltage: float = 0.0
    current: float = 0.0
    node_id: str = "BUS_DC"
    sample_index: int = 0


@dataclass
class WindowReadyEvent(BaseEvent):
    """Windowed data ready for processing."""
    window_data: Any = None  # numpy array expected
    window_id: int = 0


# ==============================================================================
# PROCESSING EVENTS (DSP Results)
# ==============================================================================

@dataclass
class DWTCoefficientsEvent(BaseEvent):
    """DWT decomposition results."""
    coeffs: Any = None  # List of arrays [cA, cD1, cD2, ...]
    wavelet: str = "db4"
    level: int = 4
    window_id: int = 0


@dataclass
class DWTResultEvent(BaseEvent):
    """Enhanced DWT result with energy analysis."""
    coeffs: List[Any] = field(default_factory=list)
    energy_levels: Dict[str, float] = field(default_factory=dict)
    wavelet: str = "db4"
    level: int = 4
    window_id: int = 0


@dataclass
class FFTResultEvent(BaseEvent):
    """FFT analysis results for benchmarking."""
    frequencies: Any = None  # numpy array
    magnitudes: Any = None  # numpy array
    dominant_freq: float = 0.0
    window_id: int = 0


@dataclass
class ProcessingResultEvent(BaseEvent):
    """Aggregated processing metrics."""
    d1_energy: float = 0.0
    d1_peak: float = 0.0
    is_faulty: bool = False
    window_id: int = 0


# ==============================================================================
# DETECTION EVENTS (Fault Analysis)
# ==============================================================================

@dataclass
class FaultDetectedEvent(BaseEvent):
    """Single detector's fault detection."""
    confidence: float = 0.0
    source_agent: str = ""
    fault_type: str = "UNKNOWN"
    severity: float = 0.0


@dataclass
class FaultVotingResultEvent(BaseEvent):
    """Aggregated voting result from multiple detectors."""
    votes_for_fault: int = 0
    total_voters: int = 0
    consensus_reached: bool = False
    fault_type: str = "UNKNOWN"


@dataclass
class AIAnalysisEvent(BaseEvent):
    """AI/ML-based fault diagnosis result."""
    fault_probability: float = 0.0
    diagnosis: str = "Unknown"
    confidence: float = 0.0
    probable_causes: List[Dict[str, float]] = field(default_factory=list)


# ==============================================================================
# CONTROL EVENTS (Trip & Converter)
# ==============================================================================

@dataclass
class SystemTripEvent(BaseEvent):
    """System trip command (open relay)."""
    reason: str = ""
    urgency: int = 10  # 1-10
    latency_ms: float = 0.0
    snapshot_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConverterStatusEvent(BaseEvent):
    """Zeta converter status update."""
    duty_cycle: float = 0.0
    target_voltage: float = 400.0
    actual_voltage: float = 0.0
    mode: str = "AUTO"  # AUTO, MANUAL, SAFE
    current_limit: float = 20.0


@dataclass
class ConverterCommandEvent(BaseEvent):
    """Command to change converter settings."""
    command: str = ""  # SET_DUTY, SET_TARGET, SET_MODE
    value: Any = None


# ==============================================================================
# SUPERVISION EVENTS (Health & Monitoring)
# ==============================================================================

@dataclass
class HealthStatusEvent(BaseEvent):
    """System health metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    events_per_second: float = 0.0
    buffer_fill_percent: float = 0.0
    latency_avg_ms: float = 0.0


@dataclass
class FaultTimelineEntry(BaseEvent):
    """Single entry in fault timeline reconstruction."""
    step_name: str = ""
    step_time: float = 0.0
    agent_name: str = ""
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySnapshotEvent(BaseEvent):
    """Snapshot for replay system."""
    snapshot_id: str = ""
    events: List[Any] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class BenchmarkResultEvent(BaseEvent):
    """Algorithm benchmarking results."""
    algorithm: str = ""
    detection_time_ms: float = 0.0
    accuracy: float = 0.0
    false_positives: int = 0
    false_negatives: int = 0


# ==============================================================================
# GRID TOPOLOGY EVENTS (Digital Twin)
# ==============================================================================

@dataclass
class GridTopologyEvent(BaseEvent):
    """Grid topology update for Digital Twin."""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    updated_node: str = ""


@dataclass
class NodeStatusChangeEvent(BaseEvent):
    """Single node status change."""
    node_id: str = ""
    old_status: str = ""
    new_status: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultInjectionEvent(BaseEvent):
    """Fault injection command for emulator."""
    fault_type: str = ""  # L2L, L2G, ARC, NOISE, DRIFT, SENSOR_FAIL
    severity: float = 0.0
    location: str = ""
    duration: float = 0.0


# ==============================================================================
# LOGGING & AUDIT EVENTS
# ==============================================================================

@dataclass
class LogEvent(BaseEvent):
    """Structured log event."""
    level: str = "INFO"
    message: str = ""
    agent: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent(BaseEvent):
    """Audit-compliant event for regulatory logging."""
    action: str = ""
    actor: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""  # For tamper-resistance
