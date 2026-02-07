"""
Health Monitor Agent - Industrial DC Microgrid Platform

Monitors system health metrics including CPU, memory, event throughput,
and processing latency. Publishes HealthStatusEvent for dashboard display.
"""
import time
import threading
import logging
from typing import Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.framework.base_agent import BaseAgent
from src.domain.events import (
    BaseEvent, HealthStatusEvent, VoltageSampleEvent,
    ProcessingResultEvent, SystemTripEvent
)

logger = logging.getLogger(__name__)


class HealthMonitorAgent(BaseAgent):
    """
    System Health Monitor Agent.
    
    Features:
    - CPU and memory usage tracking
    - Event throughput calculation (events per second)
    - Processing latency monitoring
    - Buffer fill level warnings
    - Periodic health status publication
    
    Publishes: HealthStatusEvent
    """
    
    def setup(self):
        """Initialize health monitoring."""
        self.event_count = 0
        self.last_check = time.time()
        self.check_interval = self.config.get("check_interval", 1.0)
        
        # Latency tracking
        self.latencies = []
        self.max_latency_samples = 100
        
        # Event timestamps for throughput
        self.event_timestamps = []
        self.max_timestamp_samples = 1000
        
        # Subscribe to events for counting
        self.subscribe(VoltageSampleEvent, self._count_event)
        self.subscribe(ProcessingResultEvent, self._track_processing)
        self.subscribe(SystemTripEvent, self._on_trip)
        
        # Start monitoring thread
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Health Monitor initialized")

    def teardown(self):
        """Stop monitoring."""
        self._running = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=2.0)

    def _count_event(self, event: BaseEvent):
        """Count incoming events for throughput calculation."""
        self.event_count += 1
        now = time.time()
        self.event_timestamps.append(now)
        
        # Prune old timestamps
        if len(self.event_timestamps) > self.max_timestamp_samples:
            cutoff = now - 5.0  # Keep last 5 seconds
            self.event_timestamps = [t for t in self.event_timestamps if t > cutoff]

    def _track_processing(self, event: ProcessingResultEvent):
        """Track processing latency."""
        if hasattr(event, 'timestamp'):
            latency_ms = (time.time() - event.timestamp) * 1000
            self.latencies.append(latency_ms)
            
            if len(self.latencies) > self.max_latency_samples:
                self.latencies = self.latencies[-self.max_latency_samples:]

    def _on_trip(self, event: SystemTripEvent):
        """Log trip events."""
        self.logger.critical(f"SYSTEM TRIP: {event.reason}")

    def _monitor_loop(self):
        """Background loop for periodic health checks."""
        while self._running:
            try:
                self._publish_health()
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
            
            time.sleep(self.check_interval)

    def _publish_health(self):
        """Calculate and publish health metrics."""
        now = time.time()
        
        # CPU and Memory
        if PSUTIL_AVAILABLE:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
        else:
            cpu = 0.0
            memory = 0.0
        
        # Events per second
        if self.event_timestamps:
            time_span = now - self.event_timestamps[0] if self.event_timestamps else 1.0
            eps = len(self.event_timestamps) / max(time_span, 0.001)
        else:
            eps = 0.0
        
        # Average latency
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        
        # Buffer fill (from event bus stats if available)
        buffer_fill = 0.0
        try:
            bus_stats = self.bus.get_stats()
            if bus_stats.get("history_max", 0) > 0:
                buffer_fill = (bus_stats["history_size"] / bus_stats["history_max"]) * 100
        except:
            pass
        
        # Publish health status
        health_event = HealthStatusEvent(
            source=self.name,
            cpu_usage=cpu,
            memory_usage=memory,
            events_per_second=eps,
            buffer_fill_percent=buffer_fill,
            latency_avg_ms=avg_latency
        )
        
        self.publish(health_event)
        
        # Log warnings if needed
        if cpu > 90:
            self.logger.warning(f"High CPU usage: {cpu}%")
        if memory > 90:
            self.logger.warning(f"High memory usage: {memory}%")
        if avg_latency > 10:
            self.logger.warning(f"High latency: {avg_latency:.2f}ms")
        if buffer_fill > 80:
            self.logger.warning(f"Event buffer near full: {buffer_fill:.1f}%")

    def get_current_health(self) -> Dict[str, Any]:
        """Get current health metrics (for direct query)."""
        if PSUTIL_AVAILABLE:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
        else:
            cpu = 0.0
            memory = 0.0
            
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        
        return {
            "cpu_usage": cpu,
            "memory_usage": memory,
            "events_per_second": self.event_count / max(time.time() - self.last_check, 1),
            "latency_avg_ms": avg_latency
        }
