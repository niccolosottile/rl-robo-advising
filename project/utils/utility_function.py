import numpy as np

def mean_variance_utility(constituents_returns, portfolio_weights, r):
    # Calculate expected return for each asset
    c_t = constituents_returns.mean(axis=0)

    # Calculate expected return for portfolio
    expected_return = np.dot(c_t, portfolio_weights)

    # Calculate covariance between assets
    Q_t = np.cov(constituents_returns.values.T)

    # Calculate variance for portfolio
    variance = np.dot(portfolio_weights, np.dot(Q_t, portfolio_weights))

    return expected_return - r * variance