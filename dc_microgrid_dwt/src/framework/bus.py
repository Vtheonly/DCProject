"""
Event Bus Module - Industrial DC Microgrid Platform

High-performance event bus with replay capability for the event-driven
architecture. Supports synchronous and asynchronous event publishing.
"""
import logging
import time
from collections import defaultdict, deque
from typing import Callable, Type, Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class EventBus:
    """
    Central event bus for inter-agent communication.
    
    Features:
    - Type-based subscription
    - Synchronous and asynchronous publishing
    - Event history for replay/blackbox functionality
    - Thread-safe operations
    """
    
    def __init__(self, history_size: int = 10000):
        self._subscribers: Dict[Type, List[Callable]] = defaultdict(list)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._event_count = 0
        self._start_time = time.time()
        
        # Metrics
        self._last_event_time = 0.0
        self._events_per_second = 0.0
        self._eps_counter = 0
        self._eps_last_calc = time.time()

    def subscribe(self, event_type: Type, callback: Callable):
        """Subscribe a callback to an event type."""
        with self._lock:
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed {callback.__name__} to {event_type.__name__}")

    def unsubscribe(self, event_type: Type, callback: Callable):
        """Unsubscribe a callback from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass

    def publish(self, event: Any):
        """
        Publish an event to all subscribers.
        Events are archived to history for replay capability.
        """
        # 1. Archive for Replay/Blackbox
        with self._lock:
            self._history.append(event)
            self._event_count += 1
            self._eps_counter += 1
            self._last_event_time = time.time()
            
            # Calculate EPS every second
            now = time.time()
            if now - self._eps_last_calc >= 1.0:
                self._events_per_second = self._eps_counter / (now - self._eps_last_calc)
                self._eps_counter = 0
                self._eps_last_calc = now

        # 2. Dispatch to subscribers
        event_type = type(event)
        handlers = []
        
        with self._lock:
            if event_type in self._subscribers:
                handlers = list(self._subscribers[event_type])
        
        for callback in handlers:
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    f"Error in handler {callback.__name__}: {e}",
                    exc_info=True,
                    extra={"props": {"event": str(event)}}
                )

    def publish_async(self, event: Any):
        """Fire and forget on a separate thread (for slow tasks)."""
        self._executor.submit(self.publish, event)

    def get_history(self, count: Optional[int] = None) -> List[Any]:
        """
        Get event history for replay.
        
        Args:
            count: Optional number of most recent events to return.
                   If None, returns all history.
        """
        with self._lock:
            if count is None:
                return list(self._history)
            return list(self._history)[-count:]

    def get_history_range(self, start_time: float, end_time: float) -> List[Any]:
        """
        Get events within a specific time range.
        
        Args:
            start_time: Start timestamp (unix time)
            end_time: End timestamp (unix time)
        """
        with self._lock:
            result = []
            for event in self._history:
                if hasattr(event, 'timestamp'):
                    if start_time <= event.timestamp <= end_time:
                        result.append(event)
            return result

    def replay(self, events: List[Any], speed: float = 1.0):
        """
        Replay a list of events.
        
        Args:
            events: List of events to replay
            speed: Replay speed multiplier (1.0 = real-time)
        """
        if not events:
            return
            
        prev_time = events[0].timestamp if hasattr(events[0], 'timestamp') else 0
        
        for event in events:
            if hasattr(event, 'timestamp') and speed > 0:
                delay = (event.timestamp - prev_time) / speed
                if delay > 0:
                    time.sleep(delay)
                prev_time = event.timestamp
            
            # Re-publish (but don't re-archive to avoid duplicates)
            event_type = type(event)
            with self._lock:
                handlers = list(self._subscribers.get(event_type, []))
            
            for callback in handlers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Replay error in {callback.__name__}: {e}")

    def clear_history(self):
        """Clear event history."""
        with self._lock:
            self._history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics."""
        with self._lock:
            return {
                "total_events": self._event_count,
                "history_size": len(self._history),
                "history_max": self._history.maxlen,
                "subscribers_count": sum(len(s) for s in self._subscribers.values()),
                "events_per_second": self._events_per_second,
                "uptime_seconds": time.time() - self._start_time
            }

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
