import pandas as pd
import numpy as np

def load_and_prepare_data(acwi_file, aggu_file):
    # Load data
    acwi = pd.read_csv(acwi_file, index_col='Date', parse_dates=True)
    aggu = pd.read_csv(aggu_file, index_col='Date', parse_dates=True)

    # Ensure the data is sorted by date
    acwi.sort_index(inplace=True)
    aggu.sort_index(inplace=True)

    # Calculate annual returns using opening price 253 days ago and closing price today
    acwi_annual_returns = (acwi['Adj Close'] - acwi['Open'].shift(253)) / acwi['Open'].shift(253)
    aggu_annual_returns = (aggu['Adj Close'] - aggu['Open'].shift(253)) / aggu['Open'].shift(253)

    # Combine acwi_returns and aggu_returns into a single DataFrame
    combined_returns = pd.concat([acwi_annual_returns, aggu_annual_returns], axis=1)
    combined_returns.columns = ['ACWI', 'AGGU']

    # Drop rows with any NaN values in the combined DataFrame
    combined_returns.dropna(inplace=True)

    return combined_returns