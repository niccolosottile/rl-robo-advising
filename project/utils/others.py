
import numpy as np

def normalize_portfolio(portfolio_choice):
    portfolio_sum = np.sum(portfolio_choice)

    if portfolio_sum > 0:
        # Normalize if sum is greater than 0
        normalized_portfolio_choice = portfolio_choice / portfolio_sum
    else:
        # Assign equal weights to all assets when the sum is 0
        num_assets = len(portfolio_choice)
        normalized_portfolio_choice = np.full(num_assets, 1.0 / num_assets)

    return normalized_portfolio_choice