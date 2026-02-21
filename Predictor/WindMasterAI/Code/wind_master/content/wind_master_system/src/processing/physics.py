import numpy as np
import pandas as pd
from src.core.schema import WindSchema as Schema

class WindPhysicsEngine:
    @staticmethod
    def process(df):
        df = df.copy()
        
        # 🚨 FIX 1: Impute missing values with the median of the column
        # This saves the Turkey data from being deleted.
        for col in [Schema.WIND_DIRECTION, Schema.TEMP, Schema.PRESSURE]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Physics with Clipping
        v3 = df[Schema.WIND_SPEED] ** 3
        df[Schema.WIND_CUBED] = np.clip(v3, 0, 100000)
        
        rad = np.deg2rad(df[Schema.WIND_DIRECTION])
        df[Schema.WIND_U] = df[Schema.WIND_SPEED] * np.cos(rad)
        df[Schema.WIND_V] = df[Schema.WIND_SPEED] * np.sin(rad)
        
        df[Schema.ACTIVE_POWER] = df[Schema.ACTIVE_POWER].clip(lower=0, upper=5000)
        
        # 🚨 FIX 2: Only drop a row if the ABSOLUTELY ESSENTIAL data is missing
        return df.dropna(subset=[Schema.WIND_SPEED, Schema.ACTIVE_POWER])