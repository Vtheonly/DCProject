import random
from src.domain.interfaces import ISensor

class HardwareADCSensor(ISensor):
    def read(self) -> float:
        # Real hardware read code would go here
        # return adc.read_channel(0)
        return 0.0

class SimulatedADCSensor(ISensor):
    def __init__(self):
        self.value = 400.0 # Nominal 400V DC

    def set_fault(self, active: bool):
        if active:
            # Simulate a fault transient
            self.value = random.uniform(300.0, 500.0) # Oscillation
        else:
            self.value = 400.0 + random.uniform(-1.0, 1.0) # Noise

    def read(self) -> float:
        return self.value
