import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with st.spinner("Loading..."):
    from src.data_insights import load_data, data_quality
    from src.predict import predict_future

    price_df, news_df = load_data()

tab0, tab1, tab2 = st.tabs([
    "📓 Project Overview",
    "📊 Data Insights",
    "🔮 Predict"])

with tab0:
    # Introduction
    st.title("News Sentiment Analysis for Stock Price Prediction App 📈")
    st.write("""
            The following app allows you to predict stock prices using an AI model using 
            news sentiment analysis.
            """)
    
    st.markdown("### Project Overview")
    st.markdown("""
                - **Target**: Predict stock close price for 1,3,5,10 day horizons
                - **Model Architecture**: XGBoost with Multi-output for 1,3,5,10 day horizons
                - **News Integration**: SentenceTransformer (all-MiniLM-L6-v2) embeddings of headlines+summaries
                - **Features**: Technical indicators + lag prices + news embeddings
                """)
    
    st.markdown("**Obtained Model Metrics**")
    st.markdown(""" Metrics on the train set for the final model are as follows:
| Horizon (days) | MAE      | R2      |
|----------------|----------|---------|
| 1              | 0.000357 | 0.99887 |
| 3              | 0.000321 | 0.99968 |
| 5              | 0.000358 | 0.99972 |
| 10             | 0.000362 | 0.99985 |""")
    st.markdown(""" Metrics on the test set for the final model are as follows:
| Horizon (days) | MAE    | R2      |
|----------------|--------|---------|
| 1              | 0.0169 | -0.0474 |
| 3              | 0.0299 | -0.0724 |
| 5              | 0.0396 | -0.0779 |
| 10             | 0.0573 | -0.0503 |""")
                
    st.write("""These metrics indicate that the model has some predictive power, 
                especially for shorter horizons, but there is room for improvement. 
                The negative R2 values suggest that the model is not capturing all the 
                variance in the data, which could be due to the complexity of stock price 
                movements and the limitations of the features used. Future iterations 
                could focus on feature engineering, hyperparameter tuning, and exploring 
                more advanced modeling techniques to enhance performance.
             """)
    
    st.markdown("""
## 🚀 **Next Steps & Improvements**

1. **Better LLM fusion**: Replace simple concatenation with cross-attention between price features and news embeddings (would require neural network)
2. **Hyperparameter tuning**: Use Optuna for XGBoost params
3. **Walk-forward validation**: Time-series cross-validation
4. **Sentiment analysis**: Extract sentiment scores from news before embedding
5. **Model caching**: Save precomputed embeddings to avoid recomputation
                """)

with tab1:
    # Exploratory Data Analysis
    st.subheader("📊 Data Insights")
    st.write("""
            The dataset used for this project consists of historical stock prices and
            news articles related to the selected stocks over a specified date range.
            The stock price data includes features such as opening price, closing price, 
            high, low, and volume for each trading day. The new articles contains 
            headlines and summaries of news related to the selected stocks.
            """)
    st.markdown("### Exploratory Data Analysis")
    st.write("""
            This section explores the historical stock price data and news sentiment 
            datasets to uncover patterns, data quality issues, and key insights that will 
            inform our model building process.
            """)

    st.markdown("#### Data Quality and Completeness")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📈 Price Records", f"{len(price_df):,}")
    with col2:
        st.metric("📰 News Articles", f"{len(news_df):,}")

    st.metric(
        "📅 Price Date Range",
        f"{price_df['date'].min().date()} to {price_df['date'].max().date()}")
    st.metric(
        "📰 News Date Range",
        f"{news_df['datetime'].min().date()} to {news_df['datetime'].max().date()}")

    st.markdown("**Price Data Quality**")
    price_quality = data_quality(price_df)
    st.dataframe(price_quality)
    st.write("""
            The price dataset has no missing values, indicating that it is complete and 
            ready for analysis.
            """)

    st.markdown("**News Data Quality**")
    news_quality = data_quality(news_df)
    st.dataframe(news_quality)
    st.write("""
            The news dataset has one missing value for the 'summary' column, which 
            represents a very small percentage of the total records. This missing value can 
            be handled during data preprocessing without significantly impacting the model's
            performance by treating it as an empty string.
            """)

    st.markdown("#### Stock Price Data Analysis")
    # Display columns
    st.subheader("Price Dataset Columns")
    st.code(price_df.columns.tolist(), language='python')

    # Price time series visualization
    st.subheader("📈 Closing Price Time Series For one Stock (AAPL)")

    fig_price = make_subplots(specs=[[{"secondary_y": True}]])

    price_df_apple = price_df[price_df['ticker'] == 'AAPL'].copy()

    fig_price.add_trace(
        go.Scatter(x=price_df_apple['date'], y=price_df_apple['close'], name="Close Price", 
                line=dict(color='#1f77b4', width=2)),
        secondary_y=False
    )

    fig_price.add_trace(
        go.Bar(x=price_df_apple['date'], y=price_df_apple['volume'], name="Volume", 
            opacity=0.3, marker_color='#2ca02c'),
        secondary_y=True
    )

    fig_price.update_layout(
        height=500,
        title="Close Price & Trading Volume Over Time",
        hovermode='x unified',
        showlegend=True
    )
    fig_price.update_xaxes(title_text="Date")
    fig_price.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig_price.update_yaxes(title_text="Volume", secondary_y=True)

    st.plotly_chart(fig_price, use_container_width=True)

    # Returns analysis
    st.subheader("📉 Daily Returns Analysis for APPL")

    price_df_apple['daily_return'] = price_df_apple['close'].pct_change() * 100
    price_df_apple['log_return'] = np.log(price_df_apple['close'] / price_df_apple['close'].shift(1)) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Mean Daily Return", f"{price_df_apple['daily_return'].mean():.3f}%")
        st.metric("Std Dev", f"{price_df_apple['daily_return'].std():.3f}%")
        st.metric("Max Daily Gain", f"{price_df_apple['daily_return'].max():.3f}%")
        
    with col2:
        st.metric("Max Daily Loss", f"{price_df_apple['daily_return'].min():.3f}%")
        st.metric("Skewness", f"{price_df_apple['daily_return'].skew():.3f}")
        st.metric("Kurtosis", f"{price_df_apple['daily_return'].kurt():.3f}")

    fig_returns = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=('Daily Returns Over Time', 'Returns Distribution'))

    fig_returns.add_trace(
        go.Scatter(x=price_df_apple['date'], y=price_df_apple['daily_return'], name="Returns",
                line=dict(color='#ff7f0e', width=1)),
        row=1, col=1
    )

    fig_returns.add_trace(
        go.Histogram(x=price_df_apple['daily_return'].dropna(), nbinsx=50, name="Distribution",
                    marker_color='#ff7f0e', opacity=0.7),
        row=2, col=1
    )

    fig_returns.update_layout(height=600, showlegend=False, hovermode='x unified')
    st.plotly_chart(fig_returns, use_container_width=True)

    st.write("""
            These metrics and visualizations are crucial for understanding the volatility and
            risk associated with the stock, which will inform our feature engineering and model.
            This means we can kept them into account in order to enrich our data features
            and improve the model's ability to capture the underlying patterns in stock price 
            movements. 
            """)

    st.markdown("#### Technical Indicators")

    st.write("""
            Technical indicators are mathematical calculations based on historical price and 
            volume data that can help identify trends, momentum, and potential reversal points 
            in stock prices. In this section, we will calculate and visualize some common 
            technical indicators for the selected stock to gain insights into its price 
            behavior. As mentioned before, these indicators could be used as features in our 
            model to enhance its predictive power.
            """)

    # Moving averages
    st.subheader("📊 Moving Averages")

    price_df_apple['ma_7'] = price_df_apple['close'].rolling(window=7).mean()
    price_df_apple['ma_20'] = price_df_apple['close'].rolling(window=20).mean()
    price_df_apple['ma_50'] = price_df_apple['close'].rolling(window=50).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=price_df_apple['date'], y=price_df_apple['close'], name='Close Price',
                                line=dict(color='#1f77b4', width=2)))
    fig_ma.add_trace(go.Scatter(
        x=price_df_apple['date'], y=price_df_apple['ma_7'], name='7-Day MA',
                                line=dict(color='#ff7f0e', width=1.5)))
    fig_ma.add_trace(go.Scatter(
        x=price_df_apple['date'], y=price_df_apple['ma_20'], name='20-Day MA',
                                line=dict(color='#2ca02c', width=1.5)))
    fig_ma.add_trace(go.Scatter(
        x=price_df_apple['date'], y=price_df_apple['ma_50'], name='50-Day MA',
                                line=dict(color='#d62728', width=1.5)))

    fig_ma.update_layout(
        height=400,
        title="Moving Averages",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    st.subheader("Recent News")
    ticker_news = news_df[news_df['ticker'] == "AAPL"].head(10)
    for _, row in ticker_news.iterrows():
        st.markdown(f"**{row['datetime']}** - {row['headline']}")

with tab2:
    # App Interface
    st.subheader("🔮 Predict")
    st.header("Prediction Settings")

    ticker = st.selectbox("Select Ticker", ["AAPL", "GOOGL", "MSFT", "AMZN"])
    prediction_days = st.multiselect("Prediction Horizons (days)", 
                                    [1, 3, 5, 10], default=[1, 3, 5, 10])
    prediction_days.sort()
    # Date selector
    last_date = price_df[price_df['ticker'] == ticker]['date'].max()
    pred_date = st.date_input("As of date (must have data)", 
                              value=last_date - timedelta(days=31),
                              max_value=last_date)
    
    if st.button("Generate Prediction"):
        with st.spinner("Generating predictions..."):
            predictions = predict_future(ticker, pred_date, prediction_days)

        # Display current price
        current_price = price_df[
            (price_df['ticker'] == ticker) & 
            (price_df['date'] == pd.to_datetime(pred_date))
        ]['close'].values[0]
        
        st.metric("Current Price", f"${current_price:.2f}")

        # Show predictions
        cols = st.columns(len(prediction_days))
        for idx, (horizon, pred) in enumerate(predictions.items()):
            with cols[idx]:
                st.metric(
                    f"In {horizon}",
                    f"${pred['price']:.2f}",
                    delta=f"{pred['return']*100:.2f}%"
                )

        # Plot
        fig = go.Figure()
        historical = price_df[price_df['ticker'] == ticker].tail(30)
        fig.add_trace(go.Scatter(x=historical['date'], y=historical['close'],
                                    mode='lines', name='Historical'))
        
        # Add prediction points
        pred_dates = [pd.to_datetime(pred_date) + timedelta(days=int(d[:-1])) 
                        for d in predictions.keys()]
        pred_prices = [p['price'] for p in predictions.values()]
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices,
                                    mode='markers+lines', name='Predicted',
                                    marker=dict(size=10, color='red')))
        
        st.plotly_chart(fig, use_container_width=True)