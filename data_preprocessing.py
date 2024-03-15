import pandas as pd
import numpy as np

def load_and_prepare_data(acwi_file, aggu_file):
    # Load data
    acwi = pd.read_csv(acwi_file, index_col='Date', parse_dates=True)
    aggu = pd.read_csv(aggu_file, index_col='Date', parse_dates=True)

    # Calculate log returns
    acwi_returns = np.log(acwi['Adj Close'] / acwi['Adj Close'].shift(1))
    aggu_returns = np.log(aggu['Adj Close'] / aggu['Adj Close'].shift(1))

    # Combine acwi_returns and aggu_returns into a single DataFrame
    combined_returns = pd.concat([acwi_returns, aggu_returns], axis=1)
    combined_returns.columns = ['ACWI', 'AGGU']

    # Drop rows with any NaN values in the combined DataFrame
    combined_returns.dropna(inplace=True)

    return combined_returns