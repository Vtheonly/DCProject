
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from .physics import PhysicsEngine
from ..config.settings import Config
from ..utils.logger import setup_logger

logger = setup_logger("Preprocessor")

class DataPipeline:
    def __init__(self):
        pass

    def process_and_impute(self, df):
        # 1. Sort by time
        df = df.sort_values(Config.TIME_COL).reset_index(drop=True)

        # 2. Physics-Based Night Filling (The most important step)
        # If GHI (Sun) is 0, Power MUST be 0.
        # This fixes the 18,000 missing rows at night.
        if 'ghi' in df.columns:
            is_night = PhysicsEngine.detect_nighttime(df['ghi'])

            # Count missing before fix
            missing_before = df[Config.TARGET_COL].isna().sum()

            # Fill 0s where it's night and power is missing
            mask = (df[Config.TARGET_COL].isna()) & (is_night)
            df.loc[mask, Config.TARGET_COL] = 0.0

            missing_after = df[Config.TARGET_COL].isna().sum()
            logger.info(f"Physics Imputation recovered {missing_before - missing_after} night-time rows.")

        # 3. Interpolate small daytime gaps (Linear)
        # For remaining missing values (clouds/sensor glitches during day)
        df[Config.TARGET_COL] = df[Config.TARGET_COL].interpolate(method='linear', limit=4)

        # 4. Drop any remaining huge gaps we can't fix
        df = df.dropna(subset=[Config.TARGET_COL, 'ghi', 'temp'])

        return df
