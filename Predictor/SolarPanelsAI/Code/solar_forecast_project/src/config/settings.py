
from pathlib import Path

class Config:
    # Paths
    ROOT_DIR = Path("/content/solar_forecast_project")
    DATA_RAW = ROOT_DIR / "data/raw"
    DATA_PROCESSED = ROOT_DIR / "data/processed"
    MODEL_DIR = ROOT_DIR / "models/checkpoints"

    # File Names (From your analysis)
    FILE_GEN = "Solar_Generation_sub.csv"
    FILE_WEATHER = "Weather_data_sub.csv"
    FILE_IRR = "Solar_Irradiance.csv"

    # Column Mappings
    TIME_COL = "Timestamp"
    TARGET_COL = "SolarGeneration" # The power output

    # Site Configuration (Adjust if needed)
    TARGET_SITE_KEY = 25  # Based on your data analysis
    TARGET_CAMPUS_KEY = 1

    # Physics Constants
    IRRADIANCE_THRESHOLD = 5  # Watts/m^2 (Below this = Night)

    # Feature Engineering Settings
    LAGS = [1, 4, 96]  # 15-min intervals: 15m, 1h, 1day
    ROLLING_WINDOWS = [4, 16, 96]

    # XGBoost Hyperparameters (Paper Table VI)
    XGB_PARAMS = {
        'n_estimators': 150,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42
    }

    # LSTM Hyperparameters
    LSTM_PARAMS = {
        'hidden_dim': 64,
        'num_layers': 2,
        'lookback_window': 24, # 6 hours context
        'learning_rate': 0.001,
        'epochs': 20,
        'batch_size': 64
    }
