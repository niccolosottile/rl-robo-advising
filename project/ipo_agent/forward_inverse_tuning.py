# Using forward-inverse validation as detailed in paper 
# "Learning Risk Preferences from Investment Portfolios Using Inverse Optimization"
# By Shi Yu Haoran Wang & Chaosheng Dong

# Adopt different hyperparameters for IPO-Risk and IPO-Return

import numpy as np
import pandas as pd
from itertools import product
from project.utils.data_loader import load_and_prepare_returns
from project.ipo_agent.inverse_problem import inverse_problem
from project.ipo_agent.forward_problem import forward_problem

# File paths
acwi_file = 'project/data/ACWI.csv'
aggu_file = 'project/data/AGGU.L.csv'

# Extracting constituents returns 253-day rolling periods
constituents_returns = load_and_prepare_returns(acwi_file, aggu_file)

# Define the grid for hyperparameter tuning
M_values = [100, 500, 1000, 5000, 10000]
learning_rates = [100, 500, 1000, 5000, 10000]

# Placeholder for the best hyperparameters found and their corresponding error
best_hyperparams = None
lowest_error = np.inf

known_risk_aversions = np.linspace(0.1, 0.5, 3) 
guessed_risk_aversions = np.linspace(0.2, 0.4, 2) # Need to be different from actual values

# Loop through all combinations of hyperparameters
for learning_rate, M_return, M_risk in product(learning_rates, M_values, M_values):
    errors = []
    for r_s in known_risk_aversions:
        generated_portfolios = forward_problem(constituents_returns, 0.087) # The forward problem to generate portfolio

        for r_g in guessed_risk_aversions:
            estimated_r = inverse_problem(constituents_returns, generated_portfolios, r_g, M_return, M_risk, learning_rate) # The inverse problem to estimate r back
            errors.append((estimated_r - r_s)**2 / r_s**2) # Calculate sum of square error

            print(r_s, r_g, estimated_r)

    avg_error = np.mean(errors) # Calculate the average error for this combination of hyperparameters
    
    print("Hyperparameters: M_return={}, M_risk={}, Learning Rate={}, Avg Error={}".format(M_return, M_risk, learning_rate, avg_error))

    if avg_error < lowest_error:
        lowest_error = avg_error
        best_hyperparams = (M_return, M_risk, learning_rate)

print("Best Hyperparameters: M_return={}, M_risk={}, Learning Rate={}".format(best_hyperparams[0], best_hyperparams[1], best_hyperparams[2]))
print("Lowest Error: {}".format(lowest_error))