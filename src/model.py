import xgboost as xgb
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import joblib


def train_model(X_train, y_train_multi, model_params=None):
    """
    Train multi-output model for multiple horizons
    y_train_multi: shape (n_samples, n_horizons) - returns for 1,3,5,10 days
    """
    if model_params is None:
        model_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
    
    # Use multi-output wrapper for XGBoost
    base_model = xgb.XGBRegressor(**model_params)
    multi_model = MultiOutputRegressor(base_model, n_jobs=-1)
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    multi_model.fit(X_train, y_train_multi)
    
    return multi_model


def save_model(model, filepath='models/regression_model.pkl'):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath='models/regression_model.pkl'):
    return joblib.load(filepath)