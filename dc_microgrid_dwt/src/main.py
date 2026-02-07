import time
import logging
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.framework.bus import EventBus
from src.framework.registry import AgentRegistry
from src.framework.observability import Observability # Import
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

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
# logger = logging.getLogger("Main")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    obs = Observability.get_instance() # Initialize Observability
    logger = logging.getLogger("Main")
    
    logger.info("Booting DC Microgrid DWT Protection System...")
    
    # Load Configs
    # base_path = os.path.dirname(os.path.dirname(__file__))
    # dsp_cfg = load_config(os.path.join(base_path, 'config', 'dsp_settings.yaml'))
    # hard_cfg = load_config(os.path.join(base_path, 'config', 'hardware_map.yaml'))
    
    # Hardcoded for now to avoid file path issues in this snippet, 
    # but in real app use the load_config above
    dsp_cfg = {'sampling': {'rate_hz': 20000, 'window_size': 128}, 'wavelet': {'family': 'db4', 'level': 4}}
    
    # 1. Framework
    bus = EventBus()
    registry = AgentRegistry()
    
    # 2. Adapters
    sensor = SimulatedADCSensor()
    relay = RelayDriver()
    
    # 3. Agents
    # Ingestion
    sampler = SamplerAgent("Sampler", bus, dsp_cfg)
    sampler.set_sensor(sensor)
    
    window_mgr = WindowManagerAgent("WindowManager", bus, dsp_cfg)
    
    # Processing
    dwt = DWTEngineAgent("DWTEngine", bus, dsp_cfg)
    analyzer = DetailAnalyzerAgent("DetailAnalyzer", bus)
    
    # Detection
    guard = ThresholdGuardAgent("ThresholdGuard", bus)
    energy = EnergyMonitorAgent("EnergyMonitor", bus)
    voter = FaultVoterAgent("FaultVoter", bus)
    
    # Control
    zeta = ZetaLogicAgent("ZetaLogic", bus)
    sequencer = TripSequencerAgent("TripSequencer", bus)
    sequencer.set_driver(relay)
    
    telemetry = TelemetryAgent("Telemetry", bus)
    
    # Register
    agents = [sampler, window_mgr, dwt, analyzer, guard, energy, voter, zeta, sequencer, telemetry]
    for a in agents:
        registry.register(a)
        
    # Start
    registry.start_all()
    
    try:
        while True:
            time.sleep(1)
            # Simulate a random fault after 5 seconds
            # if time.time() % 10 > 5:
            #     sensor.set_fault(True)
            # else:
            #     sensor.set_fault(False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        registry.stop_all()

if __name__ == "__main__":
    main()
