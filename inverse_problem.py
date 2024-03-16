import numpy as np
import pandas as pd
from optimization_problems import solve_iporeturn,  solve_iporisk

def inverse_problem(constituents_returns, portfolio_allocations, r_g, M, learning_rate):
    """Inverse problem that estimates risk preferences based on generated portfolios."""
    # Initialization
    n_assets = constituents_returns.shape[1]
    n_time_steps = constituents_returns.values.shape[0] # Assuming constituents equally indexed by time
    A = np.ones((1, n_assets))  # Linear constraints since portfolio weights need to sum to 1
    b = np.array([1]) # Bounds for linear constraints
    r_t = r_g # Initial guess for r

    # Iterative process of alternatively learning r and c in online fashion.
    for t in range(1, n_time_steps - 1):
        current_allocations = portfolio_allocations[t]
        
        # Calculate return mean up to current time t for each asset
        c_t = []
        for i in range(n_assets):
            c_i_t = constituents_returns.values[:t+1, i].mean()
            c_t.append(c_i_t)
        c_t = np.array(c_t)

        Q_t = np.cov(constituents_returns.values[:t+1].T) # Measures covariance between assets
        eta_t = learning_rate / (t ** 0.5)

        # Solve for c given r
        c_t = solve_iporeturn(current_allocations, Q_t, A, b, r_t, c_t, M, eta_t)

        # Solve for r given c
        r_t = solve_iporisk(current_allocations, Q_t, A, b, c_t, r_t, M, eta_t)

    return r_t