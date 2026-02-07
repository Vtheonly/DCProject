import queue
from src.framework.base_agent import BaseAgent
from src.domain.events import VoltageSampleEvent, ProcessingResultEvent, SystemTripEvent, DWTCoefficientsEvent

class BridgeAgent(BaseAgent):
    def setup(self):
        self.queue = queue.Queue()
        self.subscribe(VoltageSampleEvent, self.on_event)
        self.subscribe(ProcessingResultEvent, self.on_event)
        self.subscribe(SystemTripEvent, self.on_trip)
        # self.subscribe(DWTCoefficientsEvent, self.on_event) # High throughput, be careful

    def on_event(self, event):
        # We might need to downsample or filter to avoid flooding the UI
        # For VoltageSampleEvent, maybe only send every Nth sample or batches?
        # For now, push everything, but Streamlit loop will need to drain fast.
        self.queue.put(event)

    def on_trip(self, event: SystemTripEvent):
        # Priority event
        self.queue.put(event)

    def get_queue(self):
        return self.queue
