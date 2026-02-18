"""
High-Speed Detection Loop - NEW FILE

Create this file at: src/adapters/high_speed_loop.py

This module provides a dedicated high-speed loop that bypasses the EventBus
for the critical path, ensuring deterministic timing for fault detection.
"""

import time
import threading
import logging
from src.domain.events import SystemTripEvent, ProcessingResultEvent

logger = logging.getLogger("HighSpeedLoop")


class HighSpeedDetectionLoop:
    """
    Runs outside the EventBus to ensure deterministic timing.
    
    Directly calls C++ DSP pipeline and only publishes significant events:
    - Trip events (immediate)
    - Periodic UI updates (throttled to reduce overhead)
    
    This eliminates the EventBus lock contention and ensures sub-millisecond
    fault detection latency.
    """
    
    def __init__(self, sensor, dsp_pipeline, event_bus, sample_rate=20000):
        """
        Initialize the high-speed detection loop.
        
        Args:
            sensor: ISensor implementation (reads voltage samples)
            dsp_pipeline: C++ DSPPipeline instance
            event_bus: EventBus for publishing significant events
            sample_rate: Target sample rate in Hz (default 20kHz)
        """
        self.sensor = sensor
        self.pipeline = dsp_pipeline
        self.bus = event_bus
        self.sample_rate = sample_rate
        self.interval = 1.0 / sample_rate
        self._running = False
        self._thread = None
        self._sample_count = 0
        self._trip_count = 0
        
        # Throttle UI updates to reduce overhead
        # At 20kHz, updating every 100 samples = 200Hz UI update rate
        self.ui_update_interval = 100
        
        # Performance tracking
        self._total_processing_us = 0

    def start(self):
        """Start the high-speed detection loop in a background thread."""
        if self._running:
            logger.warning("HighSpeedLoop already running")
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"HighSpeedLoop started at {self.sample_rate}Hz")

    def stop(self):
        """Stop the high-speed detection loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            logger.info("HighSpeedLoop stopped")

    def _run(self):
        """
        Main loop - runs at target sample rate.
        
        Uses time.perf_counter() for precise timing to minimize drift.
        """
        next_time = time.perf_counter()
        
        while self._running:
            try:
                # 1. Read Sensor
                voltage = self.sensor.read()
                
                # 2. Process in C++ (Fast Path)
                result = self.pipeline.process_sample(voltage)
                
                # Track processing time
                self._total_processing_us += result.processing_time_us
                
                # 3. Check Trip (Immediate - no EventBus delay)
                if result.trip.triggered:
                    self._trip_count += 1
                    evt = SystemTripEvent(
                        reason="Fast Trip (C++ Direct)",
                        urgency=10,
                        timestamp=time.time()
                    )
                    self.bus.publish(evt)
                    logger.critical(f"FAST TRIP #{self._trip_count} - C++ Direct Path")
                
                # 4. Throttled UI Updates (every N samples)
                self._sample_count += 1
                if self._sample_count % self.ui_update_interval == 0:
                    if result.window_ready:
                        evt = ProcessingResultEvent(
                            d1_energy=result.energy_levels[0],
                            d1_peak=result.d1_peak,
                            is_faulty=result.trip.triggered,
                            timestamp=time.time()
                        )
                        # Attach energy spectrum
                        evt.energy_levels = result.energy_dict()
                        self.bus.publish(evt)

            except Exception as e:
                logger.error(f"HighSpeedLoop error: {e}")

            # 5. Precise Timing - minimize drift
            now = time.perf_counter()
            drift = (next_time - now)
            if drift > 0:
                time.sleep(drift)
            next_time += self.interval

    @property
    def total_samples(self) -> int:
        """Total samples processed."""
        return self._sample_count

    @property
    def total_trips(self) -> int:
        """Total trips detected."""
        return self._trip_count

    @property
    def avg_processing_us(self) -> float:
        """Average processing time in microseconds."""
        if self._sample_count == 0:
            return 0.0
        return self._total_processing_us / self._sample_count
