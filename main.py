import numpy as np
import pandas as pd
from data_preprocessing import load_and_prepare_data
from optimization_problems import solve_iporeturn,  solve_iporisk

# Preparing constituents returns
# File paths
acwi_file = 'Datasets/ACWI.csv'
aggu_file = 'Datasets/AGGU.L.csv'
# Load and prepare data
constituents_returns = load_and_prepare_data(acwi_file, aggu_file)

# Initialization
n_assets = constituents_returns.shape[1]
n_time_steps = constituents_returns.values.shape[0] # Assuming constituents equally indexed by time
initial_c = np.ones((n_time_steps, n_assets))  # Initial guess for c (uniform accross timesteps and assets)
initial_r = np.ones(n_time_steps)  # Initial guess for r (uniform accross timesteps)
A = np.ones((1, n_assets))  # Linear constraints since portfolio weights need to sum to 1
b = np.array([1])  # Bounds for linear constraints

# Hyperparameters (different for each of IPO-Risk and IPO-Return)
M_risk = 10**6
eta_t_risk = 0.01
M_return = 10**6
eta_t_return = 0.01

# Prepare asset allocations
# List of asset allocations for n_assets
asset_allocations = [0.7, 0.3]
# Verify the length of asset_allocations matches n_assets
if len(asset_allocations) != n_assets:
    raise ValueError("Length of asset_allocations must match n_assets")
# Creating portfolio allocations DataFrame with different allocations per asset, constant across time_steps
portfolio_allocations = pd.DataFrame(np.tile(asset_allocations, (n_time_steps, 1)))

# Iterative process of alternatively learning r and c
tolerance = 1e-6  # Convergence tolerance 

for iteration in range(100):  # Max iterations
    prev_c = initial_c.copy()
    prev_r = initial_r.copy()

    for t in range(1, n_time_steps):
        current_allocations = portfolio_allocations.values[t]
        c_t = initial_c[t]
        r_t = initial_r[t]
        Q_t = np.cov(constituents_returns.values[:t+1].T) # Measures covariance between assets (used to reflect combined risk of returns)

        # Solve for c given r
        initial_c[t] = solve_iporeturn(current_allocations, constituents_returns, Q_t, A, b, r_t, c_t, M_return, eta_t_return)

        # Solve for r given c
        initial_r[t] = solve_iporisk(current_allocations, constituents_returns, Q_t, A, b, initial_c[t], r_t, M_risk, eta_t_risk)

    print("Finished pass {}:\n".format(iteration + 1))
    print(initial_c[-1])
    print(initial_r[-1])

    # Check for convergence
    delta_c = np.linalg.norm(initial_c - prev_c)
    delta_r = np.linalg.norm(initial_r - prev_r)
    if delta_c < tolerance and delta_r < tolerance:
        print(f"Converged after {iteration+1} iterations.")
        break