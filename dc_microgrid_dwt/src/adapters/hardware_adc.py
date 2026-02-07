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
        self.pattern = "CONST_400V"
        self.noise_level = 1.0

    def set_fault(self, active: bool):
        if active:
            # Simulate a fault transient
            self.value = random.uniform(300.0, 500.0) # Oscillation
        else:
            self.value = 400.0 + random.uniform(-1.0, 1.0) # Noise

    def set_pattern(self, pattern_name: str, **kwargs):
        """Configure simulation pattern."""
        self.pattern = pattern_name
        if 'noise_level' in kwargs:
            self.noise_level = kwargs['noise_level']
        # The generator will handle actual values in advanced simulation

    def read(self) -> float:
        return self.value
