"""
High-Speed Detection Loop — DC Microgrid Fault Detection Platform

Bypasses the EventBus for the critical sampling path, calling the C++ DSP
pipeline directly in a dedicated thread.  Only trips and throttled UI
updates are published back to the EventBus.

Design rationale (from Qwen / ZAI analysis):
  - EventBus publish() acquires a lock per event → lock contention at 20 kHz
  - Python DWT agents add ~200µs overhead per sample on top of C++ ~50µs
  - This loop eliminates both problems by reading sensor → C++ → trip check
    entirely inside one thread with no Python-level locking on the hot path.
"""

import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class HighSpeedDetectionLoop:
    """Deterministic sampling loop that drives the C++ DSP pipeline directly.

    Parameters
    ----------
    sensor : ISensor
        Sensor (or emulator) that provides ``read() -> float``.
    dsp_pipeline : microgrid_dsp.DSPPipeline
        Compiled C++ DSP pipeline instance.
    bus : EventBus
        Used *only* for publishing trip events and throttled UI updates.
    sample_rate : int
        Target samples per second (default 20 000).
    ui_throttle : int
        Publish a ``ProcessingResultEvent`` every *ui_throttle* samples
        to keep the UI responsive without flooding the bus.
    """

    def __init__(self, sensor, dsp_pipeline, bus, *,
                 sample_rate: int = 20_000, ui_throttle: int = 100):
        self._sensor = sensor
        self._pipeline = dsp_pipeline
        self._bus = bus
        self._sample_rate = sample_rate
        self._ui_throttle = ui_throttle

        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Performance counters
        self._total_samples = 0
        self._total_trips = 0
        self._total_processing_ns = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the high-speed loop in a daemon thread."""
        if self._running:
            logger.warning("HighSpeedDetectionLoop already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="HighSpeedLoop", daemon=True
        )
        self._thread.start()
        logger.info(
            "HighSpeedDetectionLoop started @ %d Hz (UI throttle every %d samples)",
            self._sample_rate, self._ui_throttle,
        )

    def stop(self):
        """Stop the loop and wait for the thread to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info(
            "HighSpeedDetectionLoop stopped — %d samples, %d trips",
            self._total_samples, self._total_trips,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def get_stats(self) -> dict:
        """Return performance statistics."""
        avg_us = 0.0
        if self._total_samples > 0:
            avg_us = (self._total_processing_ns / self._total_samples) / 1_000
        return {
            "total_samples": self._total_samples,
            "total_trips": self._total_trips,
            "avg_processing_us": round(avg_us, 2),
            "sample_rate": self._sample_rate,
            "running": self._running,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self):
        """Main loop — runs entirely in its own thread."""
        # Lazy imports so the module can be imported even without Streamlit
        from src.domain.events import SystemTripEvent, ProcessingResultEvent

        period = 1.0 / self._sample_rate
        throttle_counter = 0

        while self._running:
            loop_start = time.perf_counter()
            try:
                # 1. Read one sample from the sensor / emulator
                voltage = self._sensor.read()

                # 2. Process through C++ DSP pipeline
                t0 = time.perf_counter_ns()
                result = self._pipeline.process_sample(voltage)
                self._total_processing_ns += time.perf_counter_ns() - t0
                self._total_samples += 1

                # 3. Immediate trip — publish to EventBus right away
                if result.trip.triggered:
                    self._total_trips += 1
                    self._bus.publish(SystemTripEvent(
                        reason=f"Fast Trip (DSP Core) — D1 energy={result.trip.d1_energy:.1f}",
                        source="HighSpeedLoop",
                        timestamp=time.time(),
                    ))

                # 4. Throttled UI update
                throttle_counter += 1
                if throttle_counter >= self._ui_throttle and result.window_ready:
                    throttle_counter = 0
                    energy = result.energy_dict() if hasattr(result, 'energy_dict') else {}
                    evt = ProcessingResultEvent(
                        d1_peak=result.d1_peak,
                        d1_energy=energy.get("D1", 0.0),
                        is_faulty=result.trip.triggered,
                        timestamp=time.time(),
                    )
                    # Attach full energy spectrum for dashboard
                    evt.energy_levels = energy
                    self._bus.publish(evt)

            except Exception:
                logger.exception("HighSpeedLoop sample error")

            # 5. Precise sleep to maintain target rate
            elapsed = time.perf_counter() - loop_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)
