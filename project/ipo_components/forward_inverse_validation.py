import numpy as np
import cvxpy as cp

from project.ipo_components.MVO_optimisation import MVO_optimisation
from project.ipo_components.inverse_MVO_optimisation import inverse_MVO_optimisation
from project.utils.data_loader import load_and_prepare_returns

# Running Forward-Inverse Validation
constituents_returns = load_and_prepare_returns('project/data/ACWI.csv', 'project/data/AGGU.L.csv')
n_time_steps = constituents_returns.shape[0]
risk_profiles = np.linspace(0.1, 1, 10)  # Range of risk profile coefficients

errors = []

for r_original in risk_profiles:
    optimal_portfolio = MVO_optimisation(constituents_returns, r_original)
    r_estimated = inverse_MVO_optimisation(constituents_returns, optimal_portfolio)
    error = np.abs(r_original - r_estimated)
    errors.append(error)

    print(f"Original r: {r_original:.2f}, Estimated r: {r_estimated:.2f}, Error: {error:.2f}")

mean_error = np.mean(errors)

print(f"Mean Error across all tested r values: {mean_error:.2f}")