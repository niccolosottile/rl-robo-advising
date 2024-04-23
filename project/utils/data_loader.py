import pandas as pd
import numpy as np

def load_and_filter_data(acwi_file, aggu_file):
    # Load data
    acwi = pd.read_csv(acwi_file, index_col='Date', parse_dates=True)
    aggu = pd.read_csv(aggu_file, index_col='Date', parse_dates=True)

    # Ensure the data is sorted by date
    acwi.sort_index(inplace=True)
    aggu.sort_index(inplace=True)

    # Calculate prices, returns, and volatilities 
    combined_prices = pd.concat([acwi['Open'], aggu['Open']], axis=1)
    combined_prices.columns = ['ACWI', 'AGGU']
    combined_prices.dropna(inplace=True)

    acwi_annual_returns = (acwi['Adj Close'] - acwi['Open'].shift(253)) / acwi['Open'].shift(253)
    aggu_annual_returns = (aggu['Adj Close'] - aggu['Open'].shift(253)) / aggu['Open'].shift(253)
    combined_annual_returns = pd.concat([acwi_annual_returns, aggu_annual_returns], axis=1)
    combined_annual_returns.columns = ['ACWI', 'AGGU']
    combined_annual_returns.dropna(inplace=True)

    acwi_volatility = acwi['Adj Close'].pct_change().rolling(window=253).std() * np.sqrt(253)
    aggu_volatility = aggu['Adj Close'].pct_change().rolling(window=253).std() * np.sqrt(253)
    combined_volatility = pd.concat([acwi_volatility, aggu_volatility], axis=1)
    combined_volatility.columns = ['ACWI', 'AGGU']
    combined_volatility.dropna(inplace=True)

    # Identify days with negative returns
    negative_return_days = combined_annual_returns[(combined_annual_returns < 0).any(axis=1)].index

    # Filter out negative return days from all datasets
    combined_prices = combined_prices.drop(index=negative_return_days, errors='ignore')
    combined_annual_returns = combined_annual_returns.drop(index=negative_return_days, errors='ignore')
    combined_volatility = combined_volatility.drop(index=negative_return_days, errors='ignore')

    # Remove first 253 days from prices
    combined_prices = combined_prices.iloc[249:, :]

    return combined_prices, combined_annual_returns, combined_volatility