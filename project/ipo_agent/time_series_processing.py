# "Learning Risk Preferences from Investment Portfolios Using Inverse Optimization"
# By Shi Yu Haoran Wang & Chaosheng Dong

import numpy as np
import pandas as pd
from project.utils.data_loader import load_and_prepare_returns
from project.ipo_agent.forward_problem import forward_problem
from project.ipo_agent.inverse_problem import inverse_problem
import os
import json

# File paths
acwi_file = 'project/data/ACWI.csv'
aggu_file = 'project/data/AGGU.L.csv'
theta_values_path = "project/data/ipo_theta_values.json"

# Extracting constituents returns 253-day rolling periods
constituents_returns = load_and_prepare_returns(acwi_file, aggu_file)

n_time_steps = constituents_returns.values.shape[0]

# Investor behaviour parameters
theta = 0.2
r = 0.1
shifted_theta = 0.4
shifted_r = 0.1
apply_shift = False
timestep_shift = 1210
r_g = 1.0

# Hyperparameters
M = 100
learning_rate = 100

def simulate_investor_behaviour(current_timestep):
    """Simulates investor behaviour according to normal distribution"""
    if apply_shift and current_timestep >= timestep_shift:
        sampled_theta = np.random.normal(shifted_theta, shifted_r) # Apply shifted risk profile parameters
    else:
        sampled_theta = np.random.normal(theta, r)  # Sample above mean theta with std of r
    sampled_theta = max(min(sampled_theta, 1), 0.1) # Clip at boundaries of valid theta value
        
    return sampled_theta

# Generate all allocations based on simulated investor behaviour
all_allocations = []

for t in range(1200, n_time_steps):
    current_theta = simulate_investor_behaviour(t)
    
    if t == 1200:
        all_allocations = forward_problem(constituents_returns, current_theta)[:t+1]
    else:
        current_allocations = forward_problem(constituents_returns.iloc[:t+1, :], current_theta, only_last=True)[-1]
        all_allocations.append(current_allocations)

theta_values = inverse_problem(constituents_returns, all_allocations, r_g, M, learning_rate)[:timestep_shift - 1200]

# Compute incremental average
avg_theta_values = []
for i in range(len(theta_values)):
    avg_theta_values.append(np.mean(theta_values[:i+1]))

if not apply_shift:
    for t in range(timestep_shift, n_time_steps):
        theta_values.append(theta_values[-1])

# Write theta values to data file
with open(theta_values_path, 'w') as f:
    json.dump(theta_values, f)
print(f"theta values saved to {theta_values_path}")