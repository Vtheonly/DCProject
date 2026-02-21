
import xgboost as xgb
from ..config.settings import Config
from ..utils.logger import setup_logger

logger = setup_logger("XGBoost")

class XGBModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(**Config.XGB_PARAMS)

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        logger.info("XGBoost training complete.")

    def predict(self, X):
        return self.model.predict(X)

    def get_residuals(self, X, y_true):
        y_pred = self.predict(X)
        return y_true - y_pred
