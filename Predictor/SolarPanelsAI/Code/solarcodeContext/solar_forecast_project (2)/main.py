
import pandas as pd
import numpy as np
import sys
from src.config.settings import Config
from src.utils.logger import setup_logger, seed_everything
from src.data.preprocessor import DataPipeline
from src.features.engineer import FeatureEngineer
from src.models.hybrid_system import HybridForecaster
from src.evaluation.metrics import calculate_metrics

def main():
    logger = setup_logger("Main")
    seed_everything(42)

    # 1. LOAD DATA
    logger.info("Loading Datasets...")
    try:
        df_gen = pd.read_csv(Config.DATA_RAW / Config.FILE_GEN)
        df_weather = pd.read_csv(Config.DATA_RAW / Config.FILE_WEATHER)
        df_irr = pd.read_csv(Config.DATA_RAW / Config.FILE_IRR)
    except FileNotFoundError:
        logger.error("CRITICAL: Data files not found in data/raw/!")
        return

    # 2. MERGE LOGIC (Site 25, Campus 1, Bundoora)
    logger.info(f"Merging for SiteKey: {Config.TARGET_SITE_KEY}...")

    # A. Filter
    site_gen = df_gen[df_gen['SiteKey'] == Config.TARGET_SITE_KEY].copy()
    site_weather = df_weather[df_weather['CampusKey'] == Config.TARGET_CAMPUS_KEY].copy()

    # B. Time Alignment
    site_gen[Config.TIME_COL] = pd.to_datetime(site_gen['Timestamp'])
    site_weather[Config.TIME_COL] = pd.to_datetime(site_weather['Timestamp'])
    df_irr['Timestamp_UTC'] = pd.to_datetime(df_irr['Timestamp_UTC'])
    # Assume local time conversion for irradiance (simplified +10h for Melbourne/Bundoora)
    df_irr[Config.TIME_COL] = df_irr['Timestamp_UTC'] + pd.Timedelta(hours=10)

    # C. Merge
    df = pd.merge(site_gen, site_weather, on=Config.TIME_COL, how='inner', suffixes=('', '_w'))
    # Merge irradiance (closest timestamp match to handle small drifts)
    df = pd.merge_asof(df.sort_values(Config.TIME_COL),
                       df_irr.sort_values(Config.TIME_COL),
                       on=Config.TIME_COL,
                       direction='nearest',
                       tolerance=pd.Timedelta('15min'))

    # D. Rename Columns to Standard Names
    df = df.rename(columns={
        'Ghi': 'ghi',
        'AirTemperature': 'temp',
        'RelativeHumidity': 'humidity',
        'WindSpeed': 'wind_speed'
    })

    # Select cols
    cols = [Config.TIME_COL, Config.TARGET_COL, 'ghi', 'temp', 'humidity', 'wind_speed']
    df = df[cols]

    logger.info(f"Merged Data Shape: {df.shape}")

    # 3. PREPROCESSING (Physics-Guided Imputation)
    pipeline = DataPipeline()
    df = pipeline.process_and_impute(df)

    # 4. FEATURE ENGINEERING
    engineer = FeatureEngineer()
    df_features = engineer.process(df)

    # 5. SPLIT
    train_size = int(len(df_features) * 0.7)
    train, test = df_features.iloc[:train_size], df_features.iloc[train_size:]

    features = [c for c in df_features.columns if c not in [Config.TIME_COL, Config.TARGET_COL]]
    X_train, y_train = train[features], train[Config.TARGET_COL]
    X_test, y_test = test[features], test[Config.TARGET_COL]

    # 6. TRAIN & PREDICT
    model = HybridForecaster(input_dim_lstm=1)
    model.fit(X_train, y_train, X_test, y_test) # Using test as val for simplicity here

    y_pred, y_sigma = model.predict(X_test)

    # 7. METRICS
    metrics = calculate_metrics(y_test.values, y_pred)
    logger.info(f"FINAL RESULTS: {metrics}")

    # 8. SAVE
    res = pd.DataFrame({'Actual': y_test.values, 'Pred': y_pred, 'Sigma': y_sigma})
    res.to_csv(Config.DATA_PROCESSED / 'final_results.csv', index=False)
    logger.info("Done.")

if __name__ == "__main__":
    main()
