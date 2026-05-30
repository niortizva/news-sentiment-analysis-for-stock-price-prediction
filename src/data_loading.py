import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta


@st.cache_data
def align_news_to_price(df_price, df_news, days_lookback=3):
    """
    Aggregate news for each ticker-date:
    - All news in previous `days_lookback` days
    - Combine headlines + summaries
    """
    df_price['date'] = pd.to_datetime(df_price['date']).dt.normalize()
    df_news['date'] = pd.to_datetime(df_news['datetime']).dt.normalize()
    
    # For each price row, collect recent news
    aligned_news = []
    
    for ticker in df_price['ticker'].unique():
        price_ticker = df_price[df_price['ticker'] == ticker].copy()
        news_ticker = df_news[df_news['ticker'] == ticker].copy()
        
        for idx, row in price_ticker.iterrows():
            current_date = row['date']
            start_date = current_date - timedelta(days=days_lookback)
            
            recent_news = news_ticker[
                (news_ticker['date'] >= start_date) & 
                (news_ticker['date'] <= current_date)
            ]
            
            if len(recent_news) > 0:
                # Combine all news from lookback period
                combined_text = " ".join(
                    str(recent_news['headline']) + " " + str(recent_news['summary'])
                ).strip()
            else:
                combined_text = "NO_NEWS"
            
            aligned_news.append({
                'ticker': ticker,
                'date': current_date,
                'news_text': combined_text,
                'news_count': len(recent_news)
            })
    
    df_aligned = pd.DataFrame(aligned_news)
    df_aligned['date'] = pd.to_datetime(df_aligned['date'])
    
    # Merge back
    df_price['date'] = pd.to_datetime(df_price['date'])
    result = df_price.merge(df_aligned, on=['ticker', 'date'], how='left')
    result['news_text'] = result['news_text'].fillna("NO_NEWS")
    result['news_count'] = result['news_count'].fillna(0)
    
    return result