import sys
import os
import time
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.framework.bus import EventBus
from src.domain.events import DWTResultEvent, FaultLocationEvent
from src.agents.processing.fault_locator import PreciseFaultLocatorAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def test_fault_location():
    bus = EventBus()
    agent = PreciseFaultLocatorAgent("Locator", bus)
    agent.setup()
    agent.start()

    # Capture events
    received_events = []
    def on_location(event):
        received_events.append(event)
    
    bus.subscribe(FaultLocationEvent, on_location)

    # Test Case 1: Close Fault (e.g., 50m)
    # High D1 energy relative to D2
    # Formula: estimated_dist = 500 / (ratio + 0.01)
    # Target: 50m => ratio + 0.01 = 10 => ratio = 9.99
    # Ratio = D1/D2
    
    logger.info("--- Testing Close Fault (50m) ---")
    d1_energy_close = 10.0
    d2_energy_close = 1.0 # Ratio = 10
    
    # Create fake DWT event
    # Coeffs: [cA, cD4, cD3, cD2, cD1]
    # We need a peak in D1 for TOA. 
    d1_coeffs = np.zeros(100)
    d1_coeffs[50] = 50.0 # Peak at index 50
    
    event_close = DWTResultEvent(
        timestamp=time.time(),
        coeffs=[[], [], [], [], d1_coeffs],
        energy_levels={"D1": d1_energy_close, "D2": d2_energy_close},
        window_id=1
    )
    
    bus.publish(event_close)
    time.sleep(0.1)

    if received_events:
        evt = received_events[-1]
        logger.info(f"Received Event: Distance={evt.distance_m:.2f}m, Zone={evt.zone}")
        assert 40 < evt.distance_m < 60, f"Expected ~50m, got {evt.distance_m}"
    else:
        logger.error("No event received for Close Fault")

    # Test Case 2: Far Fault (e.g., 500m)
    # Low D1 energy (attenuated)
    # Target: 500m => ratio + 0.01 = 1 => ratio = 0.99
    
    logger.info("--- Testing Far Fault (500m) ---")
    d1_energy_far = 1.0
    d2_energy_far = 1.0 # Ratio = 1
    
    event_far = DWTResultEvent(
        timestamp=time.time(),
        coeffs=[[], [], [], [], d1_coeffs], # Same peak for TOA logic
        energy_levels={"D1": d1_energy_far, "D2": d2_energy_far},
        window_id=2
    )
    
    bus.publish(event_far)
    time.sleep(0.1)
    
    if len(received_events) >= 2:
        evt = received_events[-1]
        logger.info(f"Received Event: Distance={evt.distance_m:.2f}m, Zone={evt.zone}")
        assert 400 < evt.distance_m < 600, f"Expected ~500m, got {evt.distance_m}"
    else:
        logger.error("No event received for Far Fault")

    agent.stop()
    logger.info("Verificaton Complete")

if __name__ == "__main__":
    test_fault_location()
