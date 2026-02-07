from src.framework.base_agent import BaseAgent
from src.domain.events import ProcessingResultEvent, FaultDetectedEvent

class EnergyMonitorAgent(BaseAgent):
    def setup(self):
        # Thresholds
        self.energy_limit = 50.0  # From thresholds.json example
        self.subscribe(ProcessingResultEvent, self.on_result)

    def on_result(self, event: ProcessingResultEvent):
        # Check energy levels
        if event.d1_energy > self.energy_limit:
            self.logger.info(f"High energy detected: {event.d1_energy}")
            # publish fault vote
            fault_event = FaultDetectedEvent(
                confidence=0.7, 
                source_agent=self.name
            )
            self.publish(fault_event)
