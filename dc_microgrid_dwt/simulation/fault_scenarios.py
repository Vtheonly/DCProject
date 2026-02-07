import sys
import os
import time
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.framework.bus import EventBus
from src.framework.registry import AgentRegistry
from src.framework.observability import Observability # Import
from src.domain.events import VoltageSampleEvent, SystemTripEvent
# Import agents...
from src.agents.ingestion.window_manager import WindowManagerAgent
from src.agents.processing.dwt_engine import DWTEngineAgent
from src.agents.processing.detail_analyzer import DetailAnalyzerAgent
from src.agents.detection.threshold_guard import ThresholdGuardAgent
from src.agents.detection.fault_voter import FaultVoterAgent
from src.agents.control.trip_sequencer import TripSequencerAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Simulation")

class BenchmarkDriver:
    def __init__(self):
        Observability.get_instance() # Init Observability
        self.bus = EventBus()
        self.registry = AgentRegistry()
        self.trip_detected = False
        self.trip_time = None
        self.start_time = 0
        
        # Setup Agents (Subset for test)
        dsp_cfg = {'sampling': {'rate_hz': 20000, 'window_size': 128}, 'wavelet': {'family': 'db4', 'level': 4}}
        
        self.wm = WindowManagerAgent("WM", self.bus, dsp_cfg)
        self.dwt = DWTEngineAgent("DWT", self.bus, dsp_cfg)
        self.da = DetailAnalyzerAgent("DA", self.bus)
        self.guard = ThresholdGuardAgent("Guard", self.bus)
        self.voter = FaultVoterAgent("Voter", self.bus)
        self.seq = TripSequencerAgent("Seq", self.bus)
        
        # Mock Driver
        class MockDriver:
            def open_relay(s):
                self.trip_detected = True
                self.trip_time = time.time()
                logger.critical(f"TRIP DETECTED at {self.trip_time - self.start_time:.6f}s from start")
                
        self.seq.set_driver(MockDriver())
        
        self.agents = [self.wm, self.dwt, self.da, self.guard, self.voter, self.seq]
        for a in self.agents:
            self.registry.register(a)
            
    def run_scenario(self):
        logger.info("Starting Scenario: Line-to-Line Fault at T=0.1s")
        self.registry.start_all()
        
        # Generate Waveform
        # 20kHz = 50us per sample. 0.2s duration = 4000 samples.
        fs = 20000
        t = np.linspace(0, 0.2, 4000)
        voltage = np.ones_like(t) * 400.0 # 400V DC
        
        # Add noise
        voltage += np.random.normal(0, 1.0, size=len(t))
        
        # Fault at 0.1s (sample 2000)
        # Sharp drop to 50V with ringing
        # Use 7kHz ringing to ensure it hits D1 (5-10kHz band)
        fault_idx = 2000
        voltage[fault_idx:] = 50.0 + 100 * np.exp(-1000*(t[fault_idx:] - t[fault_idx])) * np.sin(2 * np.pi * 7000 * (t[fault_idx:] - t[fault_idx]))
        
        # Inject samples
        self.start_time = time.time()
        logger.info("Injecting samples...")
        
        # We need to inject at a reasonable speed, but for simulation we can just push them fast
        # However, the window manager needs chunks.
        
        fault_injection_time_real = 0
        
        for i, v in enumerate(voltage):
            if i == fault_idx:
                fault_injection_time_real = time.time()
                logger.info("Fault Injected in stream")
                
            evt = VoltageSampleEvent(timestamp=time.time(), voltage=v, sample_index=i)
            # Direct publish to bus to bypass sampler thread delay?
            # Or use WM input directly. WM listens to VoltageSampleEvent.
            self.bus.publish(evt)
            
            # Simple spin to allow threads to catch up?
             # time.sleep(0.00001) 
            
            if self.trip_detected:
                break
                
        if self.trip_detected:
            latency = self.trip_time - fault_injection_time_real
            logger.info(f"SUCCESS: Fault cleared. Latency: {latency*1000:.2f}ms (Approx real-time)")
        else:
            logger.error("FAILURE: No trip detected.")
            
        self.registry.stop_all()

if __name__ == "__main__":
    sim = BenchmarkDriver()
    sim.run_scenario()
