# "Learning Risk Preferences from Investment Portfolios Using Inverse Optimization"
# By Shi Yu Haoran Wang & Chaosheng Dong

import numpy as np
import pandas as pd
from project.ipo_agent.optimization_problems import solve_iporeturn,  solve_iporisk

def inverse_problem(constituents_returns, portfolio_allocations, r_g, M, learning_rate, only_last=False, max_iterations=100, verbose=False):
    """Inverse problem that estimates risk preferences based on generated portfolios."""
    # Initialization
    n_assets = constituents_returns.shape[1]
    n_time_steps = constituents_returns.values.shape[0] # Assuming constituents equally indexed by time
    A = np.ones((1, n_assets))  # Linear constraints since portfolio weights need to sum to 1
    b = np.array([1]) # Bounds for linear constraints
    r_t = r_g # Initial guess for r
    L = 1200 # Lookback window
    offset = 1200 if not only_last else n_time_steps - 2 # Don't process all timesteps (change to starting point of online learning)

    r_values = []

    # Iterative process of alternatively learning r and c in online fashion.
    for t in range(1 + offset, n_time_steps):
        # Derive current allocations
        current_allocations = portfolio_allocations[t-1] if not only_last else portfolio_allocations

        # Adjust indeces to use lookback window L for calculating mean returns and covariance (+1 for slicing)
        start_index = max(0, t + 1 - L)
        end_index = t + 1
        
        # Calculate return mean and covariance within the lookback window
        c_t = constituents_returns.values[start_index:end_index].mean(axis=0)
        Q_t = np.cov(constituents_returns.values[start_index:end_index].T)

        # Set learning rate
        eta_t = learning_rate / ((t - offset) ** 0.5)

        for iteration in range(max_iterations):
            # Remember previous for convergence check
            prev_c_t = np.copy(c_t)
            prev_r_t = np.copy(r_t)

            if verbose and iteration == 0:
                print("Inverse uses:")
                print(Q_t)
                print(c_t)
                print("Initial guess risk: ", r_g)
                print("Target allocation: ", current_allocations)

            # Update c and r iteratively
            #c_t = solve_iporeturn(current_allocations, Q_t, A, b, c_t, r_t, M, eta_t, verbose)
            r_t = solve_iporisk(current_allocations, Q_t, A, b, c_t, r_t, M, eta_t, verbose)

            # Save r_t for current timestep
            r_values.append(r_t.tolist())

            break # Already converged since not using alternate optimisation of IPO-Return and IPO-Risk

            # Check for convergence (using a simple L2 norm for demonstration)
            if np.linalg.norm(prev_c_t - c_t) < 1e-4 and np.linalg.norm(prev_r_t - r_t) < 1e-4:
                break

    return r_values