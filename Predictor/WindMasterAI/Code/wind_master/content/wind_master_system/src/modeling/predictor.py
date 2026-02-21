import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from src.core.schema import WindSchema as Schema

class WindGRUPredictor:
    def __init__(self, input_shape):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = Sequential([
            Input(shape=input_shape),
            Bidirectional(GRU(64, return_sequences=True)),
            BatchNormalization(),
            GRU(32),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='huber')

    def transform_data(self, df, seq_len=24):
        X = df[Schema.FEATURES].values
        y = df[Schema.ACTIVE_POWER].values.reshape(-1, 1)
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        Xs, ys = [], []
        for i in range(0, len(X_scaled) - seq_len, 4): 
            Xs.append(X_scaled[i:i+seq_len])
            ys.append(y_scaled[i+seq_len])
        return np.array(Xs), np.array(ys)

    def train(self, Xt, yt):
        cbs = [EarlyStopping(patience=3, restore_best_weights=True), TerminateOnNaN()]
        self.model.fit(Xt, yt, epochs=20, batch_size=512, validation_split=0.1, callbacks=cbs, verbose=1)

    def evaluate(self, y_true_scaled, y_pred_scaled):
        y_t = self.scaler_y.inverse_transform(y_true_scaled)
        y_p = self.scaler_y.inverse_transform(y_pred_scaled)
        mask = ~np.isnan(y_p).flatten()
        print(f"MAE: {mean_absolute_error(y_t[mask], y_p[mask]):.2f} kW | R2: {r2_score(y_t[mask], y_p[mask]):.4f}")