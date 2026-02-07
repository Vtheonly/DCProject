import logging
from src.framework.base_agent import BaseAgent
from src.domain.events import BaseEvent

class TelemetryAgent(BaseAgent):
    def setup(self):
        # Subscribe to ALL events if possible, or specific ones
        # For simplicity, we assume we can subscribe to BaseEvent and get everything
        # (Implementation of EventBus depends on if it supports inheritance-based sub)
        # Our simple EventBus probably needs specific subscriptions.
        pass

    def subscribe_to_all(self, event_types):
        for et in event_types:
            self.subscribe(et, self.log_event)

    def log_event(self, event: BaseEvent):
        # This callback happens on the thread pool if using async publish,
        # but standard publish is sync. We should be careful.
        # Ideally, we put this into a queue/separate thread.
        self.logger.debug(f"Telemetry: {event}")
