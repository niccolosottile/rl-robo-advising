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
A = np.ones((1, n_assets))  # Linear constraints since portfolio weights need to sum to 1
b = np.array([1])  # Bounds for linear constraints

# Hyperparameters (later should be different for each of IPO-Risk and IPO-Return)
M = 100
learning_rate = 100

# Prepare asset allocations
# List of asset allocations for n_assets
asset_allocations = [0.7, 0.3]
# Verify the length of asset_allocations matches n_assets
if len(asset_allocations) != n_assets:
    raise ValueError("Length of asset_allocations must match n_assets")
# Creating portfolio allocations DataFrame with different allocations per asset, constant across time_steps
portfolio_allocations = pd.DataFrame(np.tile(asset_allocations, (n_time_steps, 1)))

# Iterative process of alternatively learning r and c in online fashion.
for t in range(1, n_time_steps):
    current_allocations = portfolio_allocations.values[t]
    r_t = 1.0 # Initial guess for r
    
    # Calculate return mean up to current time t for each asset
    c_t = []
    for i in range(n_assets):
        c_i_t = constituents_returns.values[:t+1, i].mean()
        c_t.append(c_i_t)

    Q_t = np.cov(constituents_returns.values[:t+1].T) # Measures covariance between assets (used to reflect combined risk of returns)
    eta_t = learning_rate / (t ** 0.5)

    # Solve for c given r
    c_t = solve_iporeturn(current_allocations, constituents_returns, Q_t, A, b, r_t, c_t, M, eta_t)

    # Solve for r given c
    r_t = solve_iporisk(current_allocations, constituents_returns, Q_t, A, b, c_t, r_t, M, eta_t)

print(c_t)
print(r_t)