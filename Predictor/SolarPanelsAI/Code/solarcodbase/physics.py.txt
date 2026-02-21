
import pvlib
import pandas as pd
import numpy as np
from ..config.settings import Config

class PhysicsEngine:
    @staticmethod
    def detect_nighttime(ghi_series, threshold=Config.IRRADIANCE_THRESHOLD):
        # If Irradiance is near 0, it is Night.
        return ghi_series < threshold
