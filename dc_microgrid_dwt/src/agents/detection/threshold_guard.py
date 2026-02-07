import json
from src.framework.base_agent import BaseAgent
from src.domain.events import ProcessingResultEvent, FaultDetectedEvent

class ThresholdGuardAgent(BaseAgent):
    def setup(self):
        # Load thresholds
        self.thresholds = self._load_thresholds()
        self.subscribe(ProcessingResultEvent, self.on_result)

    def _load_thresholds(self):
        # In a real app, load from config/thresholds.json
        # For now, hardcode or use provided config
        return {
            "d1_peak_max": 100.0
        }

    def on_result(self, event: ProcessingResultEvent):
        # Log metric for visualization/observability
        self.log_metric("d1_peak", event.d1_peak)

        if event.d1_peak > self.thresholds['d1_peak_max']:
            self.logger.warning(
                f"Threshold exceeded!", 
                extra={"props": {"peak": event.d1_peak, "threshold": self.thresholds['d1_peak_max']}}
            )
            fault_event = FaultDetectedEvent(
                confidence=1.0, # High confidence if peak is high
                source_agent=self.name
            )
            self.publish(fault_event)
