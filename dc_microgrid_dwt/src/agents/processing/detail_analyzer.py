import numpy as np
from src.framework.base_agent import BaseAgent
from src.domain.events import DWTCoefficientsEvent, ProcessingResultEvent

class DetailAnalyzerAgent(BaseAgent):
    def setup(self):
        self.subscribe(DWTCoefficientsEvent, self.on_coeffs)

    def on_coeffs(self, event: DWTCoefficientsEvent):
        coeffs = event.coeffs
        # coeffs structure from wavedec: [cA, cDn, ..., cD1]
        # The last element is D1 (highest frequency detail)
        d1 = coeffs[-1]
        
        # Calculate features
        energy = np.sum(np.square(d1))
        peak = np.max(np.abs(d1))
        
        # Simple local heuristic (though ThresholdGuard does the real check)
        # We just pass the metrics forward
        
        result_event = ProcessingResultEvent(
            d1_energy=float(energy),
            d1_peak=float(peak),
            is_faulty=False # This agent calculates metrics, Detection agents decide fault
        )
        self.publish(result_event)
