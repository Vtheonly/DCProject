
import pandas as pd
import numpy as np
from ..config.settings import Config

class FeatureEngineer:
    def process(self, df):
        df = df.copy()

        # 1. Time Features (Cyclical)
        dt = pd.to_datetime(df[Config.TIME_COL])
        df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)

        # 2. Lag Features (What happened 15 mins ago? 1 hour ago?)
        # IMPORTANT: Shift target to avoid data leakage
        for lag in Config.LAGS:
            df[f'lag_{lag}'] = df[Config.TARGET_COL].shift(lag)

        # 3. Rolling Means (Trend)
        for w in Config.ROLLING_WINDOWS:
            df[f'roll_mean_{w}'] = df[Config.TARGET_COL].shift(1).rolling(window=w).mean()

        return df.dropna().reset_index(drop=True)
