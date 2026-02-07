import logging
import numpy as np
import time
import logging
import numpy as np
import time
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from src.framework.bus import EventBus
from src.framework.registry import AgentRegistry
from src.framework.observability import Observability
from src.domain.events import VoltageSampleEvent, SystemTripEvent
# Import Agents
from src.agents.ingestion.sampler import SamplerAgent
from src.agents.ingestion.window_manager import WindowManagerAgent
from src.agents.processing.dwt_engine import DWTEngineAgent
from src.agents.processing.detail_analyzer import DetailAnalyzerAgent
from src.agents.detection.threshold_guard import ThresholdGuardAgent
from src.agents.detection.energy_monitor import EnergyMonitorAgent
from src.agents.detection.fault_voter import FaultVoterAgent
from src.agents.control.zeta_logic import ZetaLogicAgent
from src.agents.control.trip_sequencer import TripSequencerAgent
from src.agents.control.telemetry import TelemetryAgent
from src.adapters.hardware_adc import SimulatedADCSensor
from src.adapters.relay_driver import RelayDriver

logger = logging.getLogger("AdvancedSimulation")

@dataclass
class SimulationResult:
    scenario_name: str
    duration_s: float
    fault_injected: bool
    fault_injected_time: Optional[float] = None
    trip_triggered: bool = False
    trip_time: Optional[float] = None
    latency_ms: Optional[float] = None
    false_positives: int = 0
    logs: List[str] = field(default_factory=list)

class AdvancedScenarioRunner:
    def __init__(self):
        self.obs = Observability.get_instance()
        self.bus = EventBus()
        self.registry = AgentRegistry()
        self.adc = None
        self.relay = None
        self.result = SimulationResult("", 0.0, False)
        
        # Scenario Configs
        self.scenarios = {
            "Baseline (Normal)": self._config_baseline,
            "Line-to-Line Fault": self._config_fault,
            "High Noise Stress": self._config_noise,
            "Gradual Drift": self._config_drift
        }

    def _setup_system(self):
        """Re-initializes the entire agent system for a fresh run."""
        self.bus = EventBus()
        self.registry = AgentRegistry()
        
        # Adapters
        # self.adc = SimulatedADCSensor() # No longer used directly by Sampler
        self.relay = RelayDriver()
        
        # Create Agents (Note: SamplerAgent removed for direct injection)
        agents = [
            # SamplerAgent("Sampler", self.bus, {"sampling_rate": 20000}), # REMOVED
            WindowManagerAgent("WM", self.bus),
            DWTEngineAgent("DWT", self.bus),
            DetailAnalyzerAgent("DA", self.bus),
            ThresholdGuardAgent("Guard", self.bus, {"d1_peak_max": 100.0}), # Default hook
            EnergyMonitorAgent("Energy", self.bus),
            FaultVoterAgent("Voter", self.bus),
            ZetaLogicAgent("Zeta", self.bus),
            TripSequencerAgent("Seq", self.bus, {"relay_driver": self.relay}),
            TelemetryAgent("Telemetry", self.bus)
        ]
        
        for agent in agents:
            self.registry.register(agent)
            
        # Hook for Trip Detection
        self.bus.subscribe(SystemTripEvent, self._on_trip)

    def _on_trip(self, event: SystemTripEvent):
        if not self.result.trip_triggered:
            self.result.trip_triggered = True
            self.result.trip_time = time.time()
            if self.result.fault_injected_time:
                self.result.latency_ms = (self.result.trip_time - self.result.fault_injected_time) * 1000.0
            logger.critical(f"TRIP DETECTED! Latency: {self.result.latency_ms:.4f}ms" if self.result.latency_ms else "TRIP DETECTED!")

    def _config_baseline(self, duration=0.5):
        # 400V DC, Low Noise
        # self.adc.set_pattern("CONST_400V", noise_level=2.0)
        return duration, False

    def _config_fault(self, duration=0.5):
        # Fault at 0.1s
        # self.adc.set_pattern("FAULT_L2L", fault_time=0.1)
        return duration, True

    def _config_noise(self, duration=0.5):
        # 400V DC, Extreme Noise (should NOT trip ideally, or trip if safe)
        # self.adc.set_pattern("CONST_400V", noise_level=30.0) 
        return duration, False

    def _config_drift(self, duration=1.0):
        # Voltage Sag
        # self.adc.set_pattern("SAG_300V", start_time=0.2, rate=50.0) 
        return duration, False

    def run(self, scenario_name: str, visual_mode: bool = False) -> SimulationResult:
        logger.info(f"--- Starting Scenario: {scenario_name} (Visual: {visual_mode}) ---")
        self._setup_system()
        
        if scenario_name not in self.scenarios:
            logger.error(f"Unknown scenario: {scenario_name}")
            return self.result

        config_func = self.scenarios[scenario_name]
        duration, expect_fault = config_func()
        
        self.result.fault_injected = expect_fault
        self.registry.start_all()
        
        start_time = time.time()
        # We assume 20kHz sample rate
        sample_period = 1/20000.0
        
        if expect_fault:
            self.result.fault_injected_time = start_time + 0.1 
        
        total_samples = int(duration * 20000)
        
        try:
            gen = self._get_generator(scenario_name, total_samples)
            
            # Exec loop
            # DIRECT INJECTION
            idx = 0
            fault_triggered = False
            
            for val in gen:
                # ... (same) ...
                
                # ... (same) ...
                evt = VoltageSampleEvent(
                    timestamp=time.time(),
                    voltage=float(val)
                )
                self.bus.publish(evt)
                
                if visual_mode:
                    time.sleep(sample_period) # Real-time simulation for UI
                # else: No Sleep! Run max speed to measure system latency.
                
        except KeyboardInterrupt:
            pass
        finally:
            self.registry.stop_all()
            
        self.result.duration_s = time.time() - start_time
        logger.info(f"Scenario Complete. Trip: {self.result.trip_triggered}, Latency: {self.result.latency_ms}")
        return self.result

    def _get_generator(self, scenario, N):
        # Simple signal generation logic
        t = np.linspace(0, N/20000.0, N)
        v = np.ones(N) * 400.0 # Baseline
        
        if scenario == "Line-to-Line Fault":
            # Fault at 0.1s (sample 2000)
            fault_idx = 2000
            if fault_idx < N:
                v[fault_idx:] = 200.0 # Drop
                # Transient
                v[fault_idx:fault_idx+50] += 300.0 * np.sin(2 * np.pi * 7000 * t[:50]) * np.exp(-np.linspace(0, 5, 50))
        
        elif scenario == "High Noise Stress":
            noise = np.random.normal(0, 20.0, N) # High noise
            v += noise
            
        elif scenario == "Gradual Drift":
            # Drift start at 0.2s
            drift_idx = 4000
            if drift_idx < N:
                decay = np.linspace(0, 100, N - drift_idx)
                v[drift_idx:] -= decay
                
        return iter(v)

# Utility for generator patching
class partial_generator_step:
    def __init__(self, gen):
        self.gen = gen
    def __next__(self):
        try:
            val = next(self.gen)
            return val, 0.0 # Voltage, Current (dummy)
        except StopIteration:
            return 0.0, 0.0

if __name__ == "__main__":
    sim = AdvancedScenarioRunner()
    res = sim.run("Line-to-Line Fault")
    print(res)
