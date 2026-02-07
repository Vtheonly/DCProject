from src.framework.base_agent import BaseAgent
from src.domain.events import FaultDetectedEvent, SystemTripEvent

class FaultVoterAgent(BaseAgent):
    def setup(self):
        self.subscribe(FaultDetectedEvent, self.on_fault_signal)
        # Simple logic: Single vote is enough for "Ultra-Fast" requirement
        # But we could implement a timeframe aggregation.
        
    def on_fault_signal(self, event: FaultDetectedEvent):
        if event.confidence > 0.8:
            self.logger.critical(
                f"Fault Confirmed!",
                extra={"props": {"source": event.source_agent, "confidence": event.confidence}}
            )
            trip_event = SystemTripEvent(
                reason=f"Detected by {event.source_agent}",
                urgency=10
            )
            self.publish(trip_event)
