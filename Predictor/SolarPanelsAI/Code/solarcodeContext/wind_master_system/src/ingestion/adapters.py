import pandas as pd
import numpy as np
from src.core.schema import WindSchema as Schema

class WindAdapter:
    @staticmethod
    def clean_common(df):
        df[Schema.WIND_SPEED] = pd.to_numeric(df[Schema.WIND_SPEED], errors='coerce')
        df[Schema.ACTIVE_POWER] = pd.to_numeric(df[Schema.ACTIVE_POWER], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna(subset=[Schema.WIND_SPEED, Schema.ACTIVE_POWER])

class NRELAdapter(WindAdapter):
    def load(self, path):
        raw_df = pd.read_csv(path, header=None)
        all_blocks = []
        for i in range(0, raw_df.shape[1], 11):
            if i + 11 > raw_df.shape[1]: break
            chunk = raw_df.iloc[:, i:i+11].copy()
            chunk.columns = ['Y','M','D','H','Min', Schema.WIND_SPEED, Schema.WIND_DIRECTION, 't', 'p', 's80', Schema.ACTIVE_POWER]
            all_blocks.append(chunk)
        
        df = pd.concat(all_blocks, axis=0, ignore_index=True)
        df[Schema.WIND_SPEED] = pd.to_numeric(df[Schema.WIND_SPEED], errors='coerce').fillna(0)
        mask = pd.to_numeric(df[Schema.ACTIVE_POWER], errors='coerce').isna()
        
        sim_power = (df.loc[mask, Schema.WIND_SPEED]**3) * 0.18
        df.loc[mask, Schema.ACTIVE_POWER] = np.clip(sim_power, 0, 5000) # Safe Clip
        
        df[Schema.IS_REAL] = 0.0
        df[Schema.TIMESTAMP] = pd.to_datetime(df[['Y','M','D','H','Min']].rename(columns={'Y':'year','M':'month','D':'day','H':'hour','Min':'minute'}), errors='coerce')
        return self.clean_common(df.dropna(subset=[Schema.TIMESTAMP]))

class TurkeyAdapter(WindAdapter):
    def load(self, path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        mapping = {'Date/Time': Schema.TIMESTAMP, 'LV ActivePower (kW)': Schema.ACTIVE_POWER, 'Wind Speed (m/s)': Schema.WIND_SPEED, 'Wind Direction (°)': Schema.WIND_DIRECTION}
        df = df.rename(columns=mapping)
        df[Schema.TIMESTAMP] = pd.to_datetime(df[Schema.TIMESTAMP], dayfirst=True, errors='coerce')
        df[Schema.IS_REAL] = 1.0
        return self.clean_common(df.dropna(subset=[Schema.TIMESTAMP]))