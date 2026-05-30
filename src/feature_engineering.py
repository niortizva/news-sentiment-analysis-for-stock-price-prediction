import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


@st.cache_data
def add_technical_indicators(df):
    """Add common technical indicators"""
    # Returns
    df['return_1d'] = df.groupby('ticker')['close'].pct_change(1)
    df['return_5d'] = df.groupby('ticker')['close'].pct_change(5)
    df['return_10d'] = df.groupby('ticker')['close'].pct_change(10)
    
    # Price position
    df['high_low_ratio'] = (df['high'] - df['low']) / df['low']
    df['close_to_high'] = (df['high'] - df['close']) / df['high']
    df['close_to_low'] = (df['close'] - df['low']) / df['low']
    
    # Volume features
    df['volume_ratio'] = df.groupby('ticker')['volume'].pct_change(1)
    df['volume_ma5'] = df.groupby('ticker')['volume'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # Volatility
    df['volatility_5d'] = df.groupby('ticker')['return_1d'].transform(
        lambda x: x.rolling(5, min_periods=2).std()
    )
    
    # Moving averages
    df['ma5'] = df.groupby('ticker')['close'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['ma20'] = df.groupby('ticker')['close'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df['price_ma5_ratio'] = df['close'] / df['ma5']
    df['price_ma20_ratio'] = df['close'] / df['ma20']
    
    return df


@st.cache_data
def create_lag_features(df, target_col='close', lags=[1, 2, 3, 5, 10]):
    """Create lagged target values for autoregressive features"""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby('ticker')[target_col].shift(lag)
    return df


@st.cache_data
def get_llm_embeddings(df, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for news text using sentence transformers
    Caches results to avoid recomputation
    """
    model = SentenceTransformer(model_name)
    
    # Group unique news texts to avoid recomputation
    unique_texts = df['news_text'].unique()
    text_to_embedding = {}
    
    print(f"Generating embeddings for {len(unique_texts)} unique news texts...")
    for text in unique_texts:
        if text == "NO_NEWS":
            # Zero vector for no news
            text_to_embedding[text] = np.zeros(384)  # MiniLM dimension
        else:
            embedding = model.encode(text, show_progress_bar=False)
            text_to_embedding[text] = embedding
    
    # Map back
    embeddings = np.array([text_to_embedding[text] for text in df['news_text']])
    
    # Add as separate columns
    embedding_cols = [f'news_emb_{i}' for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
    
    df = pd.concat([df, embedding_df], axis=1)
    return df


@st.cache_data
def prepare_features(df, target_horizons=[1, 3, 5, 10]):
    """
    Prepare full feature matrix with multiple targets
    """
    # First add technical indicators and lags
    df = add_technical_indicators(df)
    df = create_lag_features(df, target_col='close')
    
    # Create multiple targets (future prices)
    for horizon in target_horizons:
        df[f'target_{horizon}d'] = df.groupby('ticker')['close'].shift(-horizon)
        df[f'target_return_{horizon}d'] = (
            df[f'target_{horizon}d'] - df['close']
        ) / df['close']
    
    # Define numeric features (excluding embeddings and metadata)
    exclude_cols = ['date', 'ticker', 'news_text', 'open', 'high', 'low', 'close', 'volume'] + \
                   [f'target_{h}d' for h in target_horizons] + \
                   [f'target_return_{h}d' for h in target_horizons]
    
    numeric_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Drop rows with NaN in features or targets
    df_clean = df.dropna(subset=numeric_cols + [f'target_return_{h}d' for h in target_horizons])
    
    return df_clean, numeric_cols


@st.cache_data
def scale_features(df_train, df_test, numeric_cols):
    """Standardize features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[numeric_cols])
    X_test_scaled = scaler.transform(df_test[numeric_cols])
    
    return X_train_scaled, X_test_scaled, scaler