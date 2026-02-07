"""
Replay Recorder Agent - Industrial DC Microgrid Platform

Records all events for later replay and black-box analysis.
Supports file persistence and time-range queries.
"""
import json
import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.framework.base_agent import BaseAgent
from src.domain.events import BaseEvent, SystemTripEvent, FaultDetectedEvent
from src.domain.interfaces import IRecorder

logger = logging.getLogger(__name__)


class ReplayRecorderAgent(BaseAgent, IRecorder):
    """
    Event Replay Recorder Agent.
    
    Features:
    - Records all events to memory buffer
    - Persists snapshots to disk
    - Time-range queries for replay
    - Automatic snapshots on trip events
    - Black-box reconstruction
    
    Implements IRecorder interface.
    """
    
    def setup(self):
        """Initialize recorder with storage."""
        self.recordings_dir = Path(
            self.config.get("recordings_dir", "logs/recordings")
        )
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_events = self.config.get("max_events", 50000)
        self.auto_snapshot_on_trip = self.config.get("auto_snapshot", True)
        
        # In-memory event buffer
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Subscribe to all important events
        self.subscribe(BaseEvent, self.record)
        
        # Special handling for trip events
        if self.auto_snapshot_on_trip:
            self.subscribe(SystemTripEvent, self._on_trip)
            self.subscribe(FaultDetectedEvent, self._on_fault)
        
        # Track fault timeline
        self._current_fault_start: Optional[float] = None
        
        self.logger.info(f"Replay Recorder initialized, saving to {self.recordings_dir}")

    def record(self, event: Any):
        """Record an event to memory buffer."""
        with self._lock:
            # Serialize event
            event_data = self._serialize_event(event)
            self._events.append(event_data)
            
            # Prune if over limit
            if len(self._events) > self.max_memory_events:
                self._events = self._events[-self.max_memory_events:]

    def _serialize_event(self, event: Any) -> Dict[str, Any]:
        """Convert event to serializable dictionary."""
        data = {
            "type": type(event).__name__,
            "timestamp": getattr(event, "timestamp", time.time()),
            "source": getattr(event, "source", "Unknown"),
            "event_id": getattr(event, "event_id", "")
        }
        
        # Add all dataclass fields
        if hasattr(event, "__dataclass_fields__"):
            for field_name in event.__dataclass_fields__:
                value = getattr(event, field_name)
                # Handle non-serializable types
                if isinstance(value, (int, float, str, bool, type(None))):
                    data[field_name] = value
                elif isinstance(value, (list, dict)):
                    try:
                        json.dumps(value)  # Test if serializable
                        data[field_name] = value
                    except:
                        data[field_name] = str(value)
                else:
                    data[field_name] = str(value)
                    
        return data

    def get_recording(self, start_time: float, end_time: float) -> List[Any]:
        """Get recorded events in time range."""
        with self._lock:
            return [
                e for e in self._events
                if start_time <= e.get("timestamp", 0) <= end_time
            ]

    def get_last_n_events(self, count: int) -> List[Dict[str, Any]]:
        """Get the last N recorded events."""
        with self._lock:
            return self._events[-count:]

    def get_events_by_type(self, event_type: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get events of a specific type."""
        with self._lock:
            matching = [e for e in self._events if e.get("type") == event_type]
            return matching[-count:]

    def save_to_file(self, filepath: str = None) -> bool:
        """Save current recording to file."""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.recordings_dir / f"recording_{timestamp}.json"
            
            with self._lock:
                data = {
                    "recorded_at": datetime.now().isoformat(),
                    "event_count": len(self._events),
                    "events": self._events.copy()
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Saved recording to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save recording: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load recording from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                self._events = data.get("events", [])
            
            self.logger.info(f"Loaded {len(self._events)} events from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load recording: {e}")
            return False

    def _on_fault(self, event: FaultDetectedEvent):
        """Track fault start for timeline."""
        if self._current_fault_start is None:
            self._current_fault_start = event.timestamp
            self.logger.info("Fault timeline started")

    def _on_trip(self, event: SystemTripEvent):
        """Auto-snapshot on trip events."""
        self.logger.info("Trip detected - creating snapshot")
        
        # Save snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.recordings_dir / f"trip_snapshot_{timestamp}.json"
        
        # Get events around the trip (30 seconds before)
        trip_time = event.timestamp
        start_time = trip_time - 30.0
        
        with self._lock:
            trip_events = [
                e for e in self._events
                if start_time <= e.get("timestamp", 0)
            ]
            
            snapshot = {
                "trip_event": self._serialize_event(event),
                "fault_start": self._current_fault_start,
                "trip_time": trip_time,
                "latency_ms": (trip_time - self._current_fault_start) * 1000 if self._current_fault_start else None,
                "recorded_at": datetime.now().isoformat(),
                "event_count": len(trip_events),
                "events": trip_events
            }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            self.logger.info(f"Trip snapshot saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save trip snapshot: {e}")
        
        # Reset fault tracking
        self._current_fault_start = None

    def get_timeline(self, around_time: float, window_s: float = 5.0) -> List[Dict[str, Any]]:
        """
        Get timeline of events around a specific time.
        Useful for black-box reconstruction.
        """
        start = around_time - window_s
        end = around_time + window_s
        return self.get_recording(start, end)

    def clear(self):
        """Clear the recording buffer."""
        with self._lock:
            self._events.clear()
        self.logger.info("Recording buffer cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get recorder statistics."""
        with self._lock:
            return {
                "total_events": len(self._events),
                "max_events": self.max_memory_events,
                "fill_percent": len(self._events) / self.max_memory_events * 100,
                "recordings_dir": str(self.recordings_dir)
            }
