import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Ensure the local script can find the wpf_engine folder
sys.path.append(os.getcwd())

from wpf_engine.config.settings import Config
from wpf_engine.data.processor import PhysicsEngine

def load_system():
    print("🔌 Loading System Artifacts...")
    
    # 1. Load Scaler
    scaler_path = os.path.join("saved_models", "global_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # 2. Load Model
    model_path = os.path.join("saved_models", "deep_physics_transformer.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print("✅ System Loaded Successfully.")
    return model, scaler

def predict_scenario(model, scaler, raw_data_df):
    """
    Takes a raw DataFrame (last 144 steps), processes it, and predicts the next 48 hours.
    """
    # 1. Apply Physics Engine (Calculate Vectors, Flux, Momentum)
    processed_df = PhysicsEngine.engineer_features(raw_data_df)
    
    # 2. Scale Data
    # We must use the same features as training
    features = Config.PHYSICS_FEATURES
    processed_df[features] = scaler.transform(processed_df[features])
    
    # 3. Prepare Input Tensor
    # Need exactly 144 steps
    if len(processed_df) < Config.LOOKBACK_STEPS:
        raise ValueError(f"Need at least {Config.LOOKBACK_STEPS} rows of history.")
        
    # Take the last 144 steps
    seq = processed_df[features].values[-Config.LOOKBACK_STEPS:] 
    
    # Reshape for Model: (1, 144, 14)
    X_hist = seq.reshape(1, Config.LOOKBACK_STEPS, len(features)).astype(np.float32)
    
    # Get Anchor (The 'Patv' at the last step)
    target_idx = features.index(Config.TARGET_COL)
    X_anchor = seq[-1, target_idx].reshape(1, 1).astype(np.float32)
    
    # 4. Predict
    inputs = {"history_in": X_hist, "anchor_in": X_anchor}
    prediction_scaled = model.predict(inputs, verbose=0)
    
    # 5. Inverse Scale to get Real kW
    # Manual inverse transform for just the target column
    # Formula: X_real = X_scaled * scale + min
    target_min = scaler.min_[target_idx]
    target_scale = scaler.scale_[target_idx]
    
    prediction_kw = (prediction_scaled - target_min) / target_scale
    
    return prediction_kw.flatten()

if __name__ == "__main__":
    # --- DEMO: Create Dummy Data to simulate a CSV input ---
    print("🎲 Generating Dummy Historical Data (144 steps)...")
    dummy_data = {
        "Wspd": np.random.uniform(5, 15, 144),
        "Wdir": np.random.uniform(0, 360, 144),
        "Etmp": np.random.uniform(20, 30, 144),
        "Itmp": np.random.uniform(30, 40, 144),
        "Ndir": np.random.uniform(-100, 100, 144),
        "Pab1": np.random.uniform(0, 10, 144),
        "Pab2": np.random.uniform(0, 10, 144),
        "Pab3": np.random.uniform(0, 10, 144),
        "Prtv": np.random.uniform(-50, 50, 144),
        "Patv": np.random.uniform(100, 2000, 144), # Previous power
        "TurbID": [1] * 144
    }
    df = pd.DataFrame(dummy_data)

    # --- EXECUTE ---
    try:
        model, scaler = load_system()
        forecast = predict_scenario(model, scaler, df)
        
        print("\n🔮 FORECAST GENERATED (Next 48 Hours / 288 Steps):")
        print(f"   Step 1 (+10m): {forecast[0]:.2f} kW")
        print(f"   Step 2 (+20m): {forecast[1]:.2f} kW")
        print(f"   ...")
        print(f"   Step 288 (+48h): {forecast[-1]:.2f} kW")
        print(f"   Max Predicted: {np.max(forecast):.2f} kW")
    except Exception as e:
        print(f"❌ Error: {e}")
