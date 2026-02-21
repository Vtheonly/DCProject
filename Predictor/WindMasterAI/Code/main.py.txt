import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append("/content/wind_master_system")

from src.ingestion.adapters import NRELAdapter, TurkeyAdapter
from src.processing.physics import WindPhysicsEngine
from src.modeling.predictor import WindGRUPredictor
from src.utils.logger import get_logger

logger = get_logger("WIND_MASTER")

def run():
    RAW = "/content/wind_master_system/data/raw"
    datasets = []
    if os.path.exists(f"{RAW}/nrel_wind_70k.csv"):
        datasets.append(NRELAdapter().load(f"{RAW}/nrel_wind_70k.csv"))
    if os.path.exists(f"{RAW}/T1.csv"):
        datasets.append(TurkeyAdapter().load(f"{RAW}/T1.csv"))

    if not datasets: return

    full_df = pd.concat(datasets, axis=0, ignore_index=True)
    full_df = full_df.sort_values('timestamp')
    full_df = WindPhysicsEngine.process(full_df)
    
    logger.info(f"✅ Safe Dataset Ready: {len(full_df)} rows")

    predictor = WindGRUPredictor(input_shape=(24, 6))
    X, y = predictor.transform_data(full_df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    logger.info("🚀 Training Robust GPU-Bi-GRU...")
    predictor.train(X_train, y_train)
    
    preds = predictor.model.predict(X_test)
    predictor.evaluate(y_test, preds)

if __name__ == "__main__":
    run()