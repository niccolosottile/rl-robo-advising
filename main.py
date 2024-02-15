import numpy as np
import pandas as pd
from data_preprocessing import load_and_prepare_data
from optimization_problems import solve_iporisk, solve_iporeturn

##### IMPORTANT 
##### For simplicity the model currently assumes that asset-level expected returns c are constant among timesteps.

# File paths
acwi_file = 'Datasets/ACWI.csv'
aggu_file = 'Datasets/AGGU.L.csv'

# Load and prepare data
acwi_returns, aggu_returns = load_and_prepare_data(acwi_file, aggu_file)

# Merge to form constituents returns
constituents_returns = pd.concat([acwi_returns, aggu_returns], axis=1)
constituents_returns.columns = ['ACWI', 'AGGU']

# Initialization
n_assets = constituents_returns.shape[1]
n_time_steps = len(acwi_returns)  # Assuming constituents equally indexed by time
initial_r = np.ones(n_time_steps) * 0.05  # Initial guess for r (uniform accross timesteps)
#initial_c = np.ones(n_assets) * 0.02 # Initial guess for c (uniform accross assets)
initial_c = np.full((n_time_steps, n_assets), 0.02)  # Initial guess for c (uniform accross timesteps and assets)
tolerance = 1e-6  # Convergence tolerance

Q = np.cov(constituents_returns.T) # Measures covariance between assets (reflects combined risk of returns)
A = np.ones((1, n_assets))  # Linear constraints since portfolio weights need to sum to 1
b = np.array([1])  # Bounds for linear constraints

# Hyperparameters
M_risk = 10**6
eta_t_risk = 0.01
M_return = 10**6
eta_t_return = 0.01

# List of asset allocations for n_assets
asset_allocations = [0.7, 0.3]

# Verify the length of asset_allocations matches n_assets
if len(asset_allocations) != n_assets:
    raise ValueError("Length of asset_allocations must match n_assets")

# Creating portfolio allocations DataFrame with different allocations per asset, constant across time_steps
portfolio_allocations = pd.DataFrame(np.tile(asset_allocations, (n_time_steps, 1)))

# Iterative process of alternatively learning r and c
for iteration in range(10):  # Max iterations
    prev_r = initial_r.copy()
    prev_c = initial_c.copy()

    # Solve for r given c
    learned_r = solve_iporisk(portfolio_allocations, constituents_returns, Q, A, b, initial_c, initial_r, M_risk, eta_t_risk)
    
    # Solve for c given r
    learned_c = solve_iporeturn(portfolio_allocations, constituents_returns, Q, A, b, learned_r, initial_c, M_return, eta_t_return)
    
    # Update initial estimates
    initial_r = learned_r
    initial_c = learned_c

    # Check for convergence
    delta_r = np.linalg.norm(learned_r - prev_r)
    delta_c = np.linalg.norm(learned_c - prev_c)
    if delta_r < tolerance and delta_c < tolerance:
        print(f"Converged after {iteration+1} iterations.")
        break

# Use learned_r and learned_c for the second phase of the algorithm