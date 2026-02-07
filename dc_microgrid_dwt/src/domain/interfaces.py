from abc import ABC, abstractmethod
from typing import Any

class IAgent(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def on_event(self, event: Any):
        pass

class ISensor(ABC):
    @abstractmethod
    def read(self) -> float:
        pass
