from dataclasses import dataclass
import numpy as np

@dataclass
class SignalWindow:
    data: np.ndarray
    sequence_id: int
    timestamp_start: float
    timestamp_end: float

    @property
    def size(self) -> int:
        return len(self.data)
