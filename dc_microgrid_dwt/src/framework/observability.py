import logging
import json
import time
import uuid
import sys
import threading
import collections # Fix NameError
from typing import Dict, Any, Optional
from datetime import datetime

# Thread-local storage for correlation IDs (Tracing)
_trace_context = threading.local()

def get_correlation_id() -> str:
    if not hasattr(_trace_context, 'correlation_id'):
        return None
    return _trace_context.correlation_id

def set_correlation_id(cid: str):
    _trace_context.correlation_id = cid

def clear_correlation_id():
    if hasattr(_trace_context, 'correlation_id'):
        del _trace_context.correlation_id

class JSONFormatter(logging.Formatter):
    """Formats logs as JSON for structured logging."""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        
        # Add correlation ID if present
        cid = get_correlation_id()
        if cid:
            log_obj["correlation_id"] = cid

        # Add extra fields if passed in 'extra' dict
        if hasattr(record, 'props'):
            log_obj.update(record.props)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)

class Observability:
    """Singleton-like registry for system observability."""
    _instance = None
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self.log_buffer_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self._setup_logging()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Observability()
        return cls._instance

    def _setup_logging(self):
        # Configure root logger
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        
        # Remove existing handlers
        for h in root.handlers:
            root.removeHandler(h)
            
        # Console Handler with JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        root.addHandler(handler)

        # In-Memory Handler for UI
        self.log_buffer = collections.deque(maxlen=1000)
        memory_handler = logging.Handler()
        memory_handler.emit = self._log_to_memory
        # Format for UI readability (Time - Level - Message)
        memory_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root.addHandler(memory_handler)

    def _log_to_memory(self, record):
        try:
            msg = self.log_buffer_formatter.format(record)
            self.log_buffer.append(msg)
        except Exception:
            pass

    def get_logs(self) -> str:
        """Retrieve all buffered logs as a single string."""
        return "\n".join(self.log_buffer)



    def log_metric(self, name: str, value: Any, agent: str = "System"):
        """Track a system metric."""
        timestamp = time.time()
        # In a real system, send this to Prometheus/Grafana
        # For now, just log it as a structured event
        logger = logging.getLogger(agent)
        extra = {"metric_name": name, "metric_value": value, "type": "METRIC"}
        logger.info(f"Metric: {name}={value}", extra={"props": extra})

    def log_business_event(self, event_name: str, payload: Dict[str, Any], agent: str):
        """Log a significant business event (e.g. Fault Detection)."""
        logger = logging.getLogger(agent)
        extra = {"event_type": event_name, "payload": payload, "type": "BUSINESS_EVENT"}
        logger.info(f"Event: {event_name}", extra={"props": extra})

    def start_trace(self, trace_id: Optional[str] = None):
        """Start a new trace context."""
        cid = trace_id or str(uuid.uuid4())
        set_correlation_id(cid)
        return cid

    def end_trace(self):
        """Clear trace context."""
        clear_correlation_id()

# Global Helper
def get_logger(name: str):
    return logging.getLogger(name)
