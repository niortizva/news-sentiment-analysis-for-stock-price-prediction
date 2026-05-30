import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_data
def load_data():
    """
    Load the stock price and news datasets from CSV files.
    :return: Tuple of (price_df, news_df)
    """
    price_df = pd.read_csv(
        os.path.join(DATA_DIR, "prices.csv"), parse_dates=["date"])
    news_df = pd.read_csv(
        os.path.join(DATA_DIR, "news.csv"), parse_dates=["datetime"])
    return price_df, news_df


def data_quality(df):
    """
    Assess the data quality of a DataFrame by calculating missing values and their
    percentages.
    :param df: Input DataFrame to assess
    :return: DataFrame with columns 'Column', 'Missing Count', and 'Missing %
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    quality_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    return quality_df