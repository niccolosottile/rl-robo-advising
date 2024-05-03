import pandas as pd
import numpy as np

def load_and_filter_data(risky_file, risk_free_file):
    # Load data
    risky = pd.read_csv(risky_file, index_col='Date', parse_dates=True)
    risk_free = pd.read_csv(risk_free_file, index_col='Date', parse_dates=True)

    # Ensure the data is sorted by date
    risky.sort_index(ascending=True, inplace=True)
    risk_free.sort_index(ascending=True, inplace=True)

    risky.dropna(inplace=True)
    risk_free.dropna(inplace=True)

    # Calculate prices, returns, and volatilities 
    combined_prices = pd.concat([risky['Open'], risk_free['Open']], axis=1)
    combined_prices.columns = ['risky', 'risk_free']
    combined_prices.dropna(inplace=True)

    risky_annual_returns = (risky['Adj Close'] - risky['Open'].shift(253)) / risky['Open'].shift(253)
    risk_free_annual_returns = (risk_free['Adj Close'] - risk_free['Open'].shift(253)) /risk_free['Open'].shift(253)
    combined_annual_returns = pd.concat([risky_annual_returns, risk_free_annual_returns], axis=1)
    combined_annual_returns.columns = ['risky', 'risk_free']
    combined_annual_returns.dropna(inplace=True)

    risky_volatility = risky['Adj Close'].pct_change().rolling(window=253).std() * np.sqrt(253)
    risk_free_volatility = risk_free['Adj Close'].pct_change().rolling(window=253).std() * np.sqrt(253)
    combined_volatility = pd.concat([risky_volatility, risk_free_volatility], axis=1)
    combined_volatility.columns = ['risky', 'risk_free']
    combined_volatility.dropna(inplace=True)

    # Remove first 253 days from prices
    combined_prices = combined_prices.iloc[253:, :]

    return combined_prices, combined_annual_returns, combined_volatility