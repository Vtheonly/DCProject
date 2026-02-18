import queue
from src.framework.base_agent import BaseAgent
from src.domain.events import VoltageSampleEvent, ProcessingResultEvent, SystemTripEvent, DWTCoefficientsEvent

class BridgeAgent(BaseAgent):
    def setup(self):
        self.queue = queue.Queue()
        self.downsample_factor = self.config.get('downsample_factor', 20) # Default to skip 95% of events (for 10kHz -> 500Hz UI)
        self.counter = 0

        self.subscribe(VoltageSampleEvent, self.on_voltage)
        self.subscribe(ProcessingResultEvent, self.on_event)
        self.subscribe(SystemTripEvent, self.on_trip)
        # self.subscribe(DWTCoefficientsEvent, self.on_event) # High throughput, be careful

    def on_voltage(self, event: VoltageSampleEvent):
        """Handle high-frequency voltage events with downsampling."""
        self.counter += 1
        if self.counter % self.downsample_factor == 0:
            self.queue.put(event)

    def on_event(self, event):
        # We might need to downsample or filter to avoid flooding the UI
        # For ordinary events, push everything
        self.queue.put(event)

    def on_trip(self, event: SystemTripEvent):
        # Priority event
        self.queue.put(event)

    def get_queue(self):
        return self.queue
