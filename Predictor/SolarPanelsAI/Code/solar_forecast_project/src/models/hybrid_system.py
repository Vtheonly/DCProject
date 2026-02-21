
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from .xgb_component import XGBModel
from .lstm_component import ResidualLSTM, HeteroscedasticLoss
from ..config.settings import Config
from ..utils.logger import setup_logger

logger = setup_logger("HybridSystem")

class HybridForecaster:
    def __init__(self, input_dim_lstm):
        self.xgb_model = XGBModel()
        self.lstm_model = ResidualLSTM(
            input_dim=input_dim_lstm,
            hidden_dim=Config.LSTM_PARAMS['hidden_dim'],
            num_layers=Config.LSTM_PARAMS['num_layers']
        )
        self.scaler_lstm = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model.to(self.device)

    def fit(self, X_train, y_train, X_val, y_val):
        # Phase 1: XGBoost
        logger.info("Training XGBoost Base Model...")
        self.xgb_model.fit(X_train, y_train, X_val, y_val)

        # Phase 2: Residuals
        # Returns a Pandas Series
        r_train = self.xgb_model.get_residuals(X_train, y_train)

        # FIX: Convert Pandas Series to NumPy array before reshaping
        r_train = r_train.values.reshape(-1, 1)

        # Prepare LSTM Data
        self.scaler_lstm.fit(r_train)
        r_train_scaled = self.scaler_lstm.transform(r_train)

        # Convert to Tensor sequences
        X_lstm, y_lstm = self._create_sequences(r_train_scaled, Config.LSTM_PARAMS['lookback_window'])

        # Phase 3: Train LSTM
        logger.info(f"Training LSTM Residual Corrector on {self.device}...")
        self._train_lstm(X_lstm, y_lstm)

    def predict(self, X_df):
        # 1. Base Prediction
        y_xgb = self.xgb_model.predict(X_df)

        # 2. Residual Correction
        # Creating a dummy tensor for batch prediction speed (Simplified for this demo)
        # In a real production system, you would feed the *actual* past residuals here.
        dummy_resid = np.zeros((len(X_df), Config.LSTM_PARAMS['lookback_window'], 1))
        dummy_tensor = torch.FloatTensor(dummy_resid).to(self.device)

        self.lstm_model.eval()
        with torch.no_grad():
            preds = self.lstm_model(dummy_tensor).cpu().numpy()

        # 3. Scale back and Combine
        delta = preds[:, 0] * self.scaler_lstm.scale_[0]
        # Calculate uncertainty (sigma)
        sigma = np.sqrt(np.exp(preds[:, 1]))

        return y_xgb + delta, sigma

    def _create_sequences(self, data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            x = data[i:(i + seq_len)]
            y = data[i + seq_len]
            xs.append(x)
            ys.append(y)
        # Convert to PyTorch Tensors
        return torch.FloatTensor(np.array(xs)).to(self.device), torch.FloatTensor(np.array(ys)).to(self.device)

    def _train_lstm(self, X, y):
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=Config.LSTM_PARAMS['learning_rate'])
        criterion = HeteroscedasticLoss()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=Config.LSTM_PARAMS['batch_size'], shuffle=True)

        for epoch in range(Config.LSTM_PARAMS['epochs']):
            self.lstm_model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.lstm_model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Log every 5 epochs
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"LSTM Epoch {epoch+1}/{Config.LSTM_PARAMS['epochs']} - Loss: {avg_loss:.4f}")
