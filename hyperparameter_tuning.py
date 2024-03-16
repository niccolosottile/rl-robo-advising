# Using forward-inverse validation as detailed in paper 
# "Learning Risk Preferences from Investment Portfolios Using Inverse Optimization"
# By Shi Yu Haoran Wang & Chaosheng Dong

import numpy as np
import pandas as pd
from itertools import product
from data_preprocessing import load_and_prepare_data
from inverse_problem import inverse_problem
from forward_problem import forward_problem

# File paths
acwi_file = 'Datasets/ACWI.csv'
aggu_file = 'Datasets/AGGU.L.csv'

# Extracting constituents returns 253-day rolling periods
constituents_returns = load_and_prepare_data(acwi_file, aggu_file)

# Define the grid for hyperparameter tuning
M_values = [100, 500, 1000, 5000, 10000]
learning_rates = [100, 500, 1000, 5000, 10000]

# Placeholder for the best hyperparameters found and their corresponding error
best_hyperparams = None
lowest_error = np.inf

# Loop through all combinations of hyperparameters
for M, learning_rate in product(M_values, learning_rates):
    # Assume a function to sample known risk aversions for the forward problem
    known_risk_aversions = np.linspace(0.1, 3, 3) 
    guessed_risk_aversions = np.linspace(0.2, 6, 3) # Need to be different from actual values
    
    errors = []
    for r_s in known_risk_aversions:
        generated_portfolios = forward_problem(constituents_returns, r_s) # The forward problem to generate portfolio

        for r_g in guessed_risk_aversions:
            estimated_r = inverse_problem(constituents_returns, generated_portfolios, r_g, M, learning_rate)  # The inverse problem to estimate r back
            errors.append((estimated_r - r_s)**2 / r_s**2) # Calculate sum of square error

    avg_error = np.mean(errors)  # Calculate the average error for this combination of hyperparameters
    
    print("Hyperparameters: M={}, Learning Rate={}, Avg Error={}".format(M, learning_rate, avg_error))

    if avg_error < lowest_error:
        lowest_error = avg_error
        best_hyperparams = (M, learning_rate)

print("Best Hyperparameters: M={}, Learning Rate={}".format(best_hyperparams[0], best_hyperparams[1]))
print("Lowest Error: {}".format(lowest_error))
