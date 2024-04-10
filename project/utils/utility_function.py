import numpy as np

def mean_variance_utility(constituents_returns, portfolio_weights, r):
    n_assets = constituents_returns.shape[1]

    # Calculate expected return for each asset
    c_t = []
    for i in range(n_assets):
        c_i_t = constituents_returns.values[:, i].mean()
        c_t.append(c_i_t)
    c_t = np.array(c_t)

    # Calculate expected return for portfolio
    expected_return = np.dot(c_t, portfolio_weights)

    # Calculate covariance between assets
    Q_t = np.cov(constituents_returns.values.T)

    # Calculate variance for portfolio
    variance = np.dot(portfolio_weights, np.dot(Q_t, portfolio_weights))

    return expected_return - r * variance