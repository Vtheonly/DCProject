import numpy as np
from collections import deque
from src.framework.base_agent import BaseAgent
from src.domain.events import VoltageSampleEvent, WindowReadyEvent

class WindowManagerAgent(BaseAgent):
    def setup(self):
        self.window_size = self.config.get('sampling', {}).get('window_size', 128)
        self.step_size = 1  # Sliding window step
        self.buffer = deque(maxlen=self.window_size)
        self.subscribe(VoltageSampleEvent, self.on_sample)
        self.window_count = 0

    def on_sample(self, event: VoltageSampleEvent):
        self.buffer.append(event.voltage)
        
        if len(self.buffer) == self.window_size:
            # We have a full window
            window_data = np.array(self.buffer)
            # Create a window event
            we = WindowReadyEvent(
                window_data=window_data,
                window_id=self.window_count
            )
            self.publish(we)
            self.window_count += 1
