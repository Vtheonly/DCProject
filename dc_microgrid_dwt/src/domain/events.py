from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

@dataclass
class BaseEvent:
    timestamp: float = 0.0
    payload: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class VoltageSampleEvent(BaseEvent):
    voltage: float = 0.0
    sample_index: int = 0

@dataclass
class WindowReadyEvent(BaseEvent):
    window_data: Any = None  # numpy array expected
    window_id: int = 0

@dataclass
class DWTCoefficientsEvent(BaseEvent):
    coeffs: Any = None # List of arrays [cA, cD1, cD2, ...]
    wavelet: str = ""
    window_id: int = 0

@dataclass
class ProcessingResultEvent(BaseEvent):
    d1_energy: float = 0.0
    d1_peak: float = 0.0
    is_faulty: bool = False

@dataclass
class FaultDetectedEvent(BaseEvent):
    confidence: float = 0.0
    source_agent: str = ""

@dataclass
class SystemTripEvent(BaseEvent):
    reason: str = ""
    urgency: int = 10  # 1-10

@dataclass
class LogEvent(BaseEvent):
    level: str = "INFO"
    message: str = ""
