import numpy as np
from src.core.schema import WindSchema as Schema

class WindPhysicsEngine:
    @staticmethod
    def process(df):
        df = df.copy()
        v3 = df[Schema.WIND_SPEED] ** 3
        df[Schema.WIND_CUBED] = np.clip(v3, 0, 100000) 
        
        rad = np.deg2rad(df[Schema.WIND_DIRECTION].fillna(0))
        df[Schema.WIND_U] = df[Schema.WIND_SPEED] * np.cos(rad)
        df[Schema.WIND_V] = df[Schema.WIND_SPEED] * np.sin(rad)
        
        df[Schema.ACTIVE_POWER] = df[Schema.ACTIVE_POWER].clip(lower=0, upper=5000)
        return df.replace([np.inf, -np.inf], np.nan).dropna()