import time
import queue
import threading
from src.framework.base_agent import BaseAgent
from src.domain.events import VoltageSampleEvent
from src.domain.interfaces import ISensor

class SamplerAgent(BaseAgent):
    def setup(self):
        self.sampling_rate = self.config.get('sampling', {}).get('rate_hz', 20000)
        self.interval = 1.0 / self.sampling_rate
        self.running = False
        self.sensor: ISensor = None  # Injected via dependency injection or config

    def set_sensor(self, sensor: ISensor):
        self.sensor = sensor

    def start(self):
        super().start()
        self.running = True
        self.thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        super().stop()

    def _sampling_loop(self):
        sample_index = 0
        while self.running:
            start_time = time.time()
            
            if self.sensor:
                val = self.sensor.read()
                event = VoltageSampleEvent(timestamp=start_time, voltage=val, sample_index=sample_index)
                self.publish(event)
                sample_index += 1
            
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)
