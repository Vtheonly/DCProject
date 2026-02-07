import logging
from abc import abstractmethod
from typing import Any
from src.domain.interfaces import IAgent
from src.framework.bus import EventBus
from src.framework.observability import get_logger, Observability

class BaseAgent(IAgent):
    def __init__(self, name: str, bus: EventBus, config: dict = None):
        self.name = name
        self.bus = bus
        self.config = config or {}
        # Use centralized logger getter
        self.logger = get_logger(self.name)
        self.observability = Observability.get_instance()

    def start(self):
        self.logger.info(f"Agent {self.name} starting...", extra={"props": {"lifecycle": "START"}})
        self.setup()

    def stop(self):
        self.logger.info(f"Agent {self.name} stopping...", extra={"props": {"lifecycle": "STOP"}})
        self.teardown()

    def setup(self):
        """Override for initialization logic"""
        pass

    def teardown(self):
        """Override for cleanup logic"""
        pass

    def on_event(self, event: Any):
        """Default handler, can be overridden or used with specific subscriptions"""
        pass

    def subscribe(self, event_type: Any, handler_method=None):
        if handler_method is None:
            handler_method = self.on_event
        self.bus.subscribe(event_type, handler_method)

    def publish(self, event: Any):
        self.bus.publish(event)
        
    def log_metric(self, name: str, value: Any):
        self.observability.log_metric(name, value, agent=self.name)
