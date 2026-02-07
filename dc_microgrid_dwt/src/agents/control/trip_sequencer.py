from src.framework.base_agent import BaseAgent
from src.domain.events import SystemTripEvent

class TripSequencerAgent(BaseAgent):
    def setup(self):
        self.relay_driver = self.config.get('relay_driver')
        if not self.relay_driver:
            self.logger.warning("No Relay Driver configured!")
        self.bus.subscribe(SystemTripEvent, self.on_trip)

    def set_driver(self, driver):
        self.relay_driver = driver

    def on_trip(self, event: SystemTripEvent):
        self.logger.critical("TRIP SEQUENCE INITIATED")
        if self.relay_driver:
            self.relay_driver.open_relay()
