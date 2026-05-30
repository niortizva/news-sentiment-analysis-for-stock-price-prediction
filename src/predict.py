import pandas as pd
import numpy as np
from src.data_insights import load_data
from src.data_loading import align_news_to_price
from src.feature_engineering import prepare_features, scale_features, get_llm_embeddings
from src.model import load_model
import joblib


def predict_future(ticker, current_date, days_ahead=[1, 3, 5, 10], 
                   model_path='models/regression_model.pkl',
                   scaler_path='models/scaler.pkl'):
    """
    Predict future returns for a specific ticker at a specific date
    """
    # Load latest data
    df_price, df_news = load_data()
    
    # Align and prepare
    df_aligned = align_news_to_price(df_price, df_news)
    
    # Add embeddings (in production, you'd cache or use a lighter model)
    df_with_emb = get_llm_embeddings(df_aligned)
    
    # Prepare features
    df_ready, numeric_cols = prepare_features(df_with_emb)
    
    # Filter for specific ticker and date
    target_date = pd.to_datetime(current_date).normalize()
    feature_row = df_ready[
        (df_ready['ticker'] == ticker) & 
        (df_ready['date'] == target_date)
    ]
    
    if len(feature_row) == 0:
        # raise ValueError(f"No data for {ticker} on {current_date}")
        raise ValueError("**Select prediction date (Trading days only - Mon to Fri)**")
    
    # Load model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Scale features
    X = scaler.transform(feature_row[numeric_cols])
    
    # Predict returns for all horizons
    predicted_returns = model.predict(X)[0]  # shape (n_horizons,)
    
    # Get current price
    current_price = df_price[
        (df_price['ticker'] == ticker) & 
        (df_price['date'] == target_date)
    ]['close'].values[0]
    
    # Convert returns to prices
    predictions = {}
    for i, horizon in enumerate(days_ahead):
        predicted_price = current_price * (1 + predicted_returns[i])
        predictions[f'{horizon}d'] = {
            'return': predicted_returns[i],
            'price': predicted_price
        }
    
    return predictions