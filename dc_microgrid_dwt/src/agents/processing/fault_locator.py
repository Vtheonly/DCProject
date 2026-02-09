import numpy as np
import logging
from typing import Dict, Any
from src.framework.base_agent import BaseAgent
from src.domain.events import DWTResultEvent, FaultLocationEvent

class PreciseFaultLocatorAgent(BaseAgent):
    """
    Precise Fault Location Agent.
    Uses Traveling Wave arrival time and Frequency Damping Ratios.
    """
    def setup(self):
        self.v_prop = self.config.get("v_propagation", 2.0e8) # m/s
        self.sampling_rate = self.config.get("sampling_rate", 20000)
        self.cable_attenuation_factor = 0.05 # dB/m simulated
        
        self.subscribe(DWTResultEvent, self.on_dwt_result)
        self.logger.info("Precise Fault Locator initialized")

    def on_dwt_result(self, event: DWTResultEvent):
        # 1. Detect Time of Arrival (TOA) 
        # We look for the peak index in the D1 coefficients
        # D1 coefficients are typically high frequency details
        if not event.coeffs or len(event.coeffs) < 2:
            return

        # Assuming coeffs structure matches PyWavelets: [cA, cD_level, ..., cD1]
        # or however DWTResultEvent populates it. Based on typical DWT, last element is D1.
        d1_coeffs = np.array(event.coeffs[-1])
        
        if len(d1_coeffs) == 0:
            return

        if np.max(np.abs(d1_coeffs)) < 10.0: # Noise floor
            return

        peak_idx = np.argmax(np.abs(d1_coeffs))
        # Precise arrival time within the window
        toa = event.timestamp + (peak_idx / self.sampling_rate)

        # 2. Distance Estimation via Damping Ratio
        # Physics: Higher frequencies (D1) attenuate faster than mid frequencies (D2)
        d1_energy = event.energy_levels.get("D1", 0)
        d2_energy = event.energy_levels.get("D2", 1) # Avoid div by zero
        if d2_energy == 0: d2_energy = 0.1

        # Energy Ratio decreases as distance increases
        # Ratio = E_D1 / E_D2. 
        # Near fault: High D1, Ratio is high.
        # Far fault: Low D1 (attenuated), Ratio is low.
        energy_ratio = d1_energy / d2_energy
        
        # Heuristic distance calculation
        # calibrated for the simulation parameters
        # Distance ~= k / ratio
        # Let's use the formula from the request: 500 / (energy_ratio + 0.01)
        estimated_dist = 500 / (energy_ratio + 0.01) 
        estimated_dist = np.clip(estimated_dist, 1.0, 1000.0)

        # 3. Determine Zone
        zone = "ZONE_1" if estimated_dist < 50 else "ZONE_2" if estimated_dist < 200 else "ZONE_OUTER"

        # 4. Publish Location
        location_evt = FaultLocationEvent(
            source=self.name,
            distance_m=float(estimated_dist),
            zone=zone,
            confidence=0.85, # Logic confidence
            time_of_arrival=toa
        )
        self.publish(location_evt)
        
        self.logger.info(f"📍 Fault Located: {estimated_dist:.2f}m in {zone}", 
                         extra={"props": {"distance": estimated_dist, "toa": toa}})
