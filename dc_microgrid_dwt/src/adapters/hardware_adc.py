import random
from typing import List
from src.domain.interfaces import ISensor, IRelay

class HardwareADCSensor(ISensor):
    def read(self) -> float:
        # Real hardware read code would go here
        # return adc.read_channel(0)
        return 0.0
    
    def read_batch(self, count: int) -> List[float]:
        # Real hardware read batch code would go here
        return [0.0 for _ in range(count)]

class SimulatedADCSensor(ISensor):
    """Simulated sensor that reads from the GridEmulator."""
    
    def __init__(self, emulator=None):
        self.emulator = emulator
        self.value = 400.0 # Fallback
        
    def set_sensor(self, emulator):
        """Set the emulator instance if not provided in init."""
        self.emulator = emulator

    def read(self) -> float:
        """Read voltage from the emulator's active node."""
        if self.emulator:
            return self.emulator.read()
        return self.value

    def read_batch(self, count: int) -> List[float]:
        """Read multiple samples."""
        if self.emulator:
            return self.emulator.read_batch(count)
        return [self.value for _ in range(count)]

class SimulatedRelayDriver(IRelay):
    """Simulated relay driver for circuit protection."""
    
    def __init__(self):
        self.state = "CLOSED"
        
    def open_relay(self) -> bool:
        """Open the relay (disconnect)."""
        self.state = "OPEN"
        print(" [RELAY] CIRCUIT BREAKER OPENED!")
        return True
        
    def close_relay(self) -> bool:
        """Close the relay (connect)."""
        self.state = "CLOSED"
        print(" [RELAY] CIRCUIT BREAKER CLOSED")
        return True
        
    def get_status(self) -> str:
        """Get current relay status."""
        return self.state
