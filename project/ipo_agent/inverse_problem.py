import numpy as np
import pandas as pd
from project.ipo_agent.optimization_problems import solve_iporeturn,  solve_iporisk

def inverse_problem(constituents_returns, portfolio_allocations, r_g, M, learning_rate, only_last=False):
    """Inverse problem that estimates risk preferences based on generated portfolios."""
    # Initialization
    n_assets = constituents_returns.shape[1]
    n_time_steps = constituents_returns.values.shape[0] # Assuming constituents equally indexed by time
    A = np.ones((1, n_assets))  # Linear constraints since portfolio weights need to sum to 1
    b = np.array([1]) # Bounds for linear constraints
    r_t = r_g if r_g is not None else 15 # Initial guess for r
    offset = 0 if not only_last else n_time_steps - 2 # Don't process all timesteps
    L = 100

    # Iterative process of alternatively learning r and c in online fashion.
    for t in range(1200 + offset, n_time_steps):
        current_allocations = portfolio_allocations[t-1] if not only_last else portfolio_allocations

        # Adjust indeces to use lookback window L for calculating mean returns and covariance (+1 for slicing)
        start_index = max(0, t + 1 - L)
        end_index = t + 1
        
        # Calculate return mean and covariance within the lookback window
        c_t = constituents_returns.values[start_index:end_index].mean(axis=0)
        Q_t = np.cov(constituents_returns.values[start_index:end_index].T)

        eta_t = learning_rate / ((t - 1199) ** 0.5)

        # Solve for c given r
        c_t = solve_iporeturn(current_allocations, Q_t, A, b, r_t, c_t, M, eta_t)

        # Solve for r given c
        r_t = solve_iporisk(current_allocations, Q_t, A, b, c_t, r_t, M, eta_t)

    return r_t