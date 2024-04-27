import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from project.ipo_components.MVO_optimisation import MVO_optimisation
from project.ipo_components.inverse_MVO_optimisation import inverse_MVO_optimisation
from project.utils.data_loader import load_and_filter_data

# Load portfolio constituents returns data
_, constituents_returns, _ = load_and_filter_data('project/data/ACWI.csv', 'project/data/AGGU.L.csv')
n_time_steps = constituents_returns.shape[0]

# Range of valid risk profiles for given returns
risk_profiles = np.linspace(0.1, 3, 150) 

# Lists to store results
errors = []
results = []

# Perform Forwad-Inverse Validation and record mean absolute error (MAE)
for r_original in risk_profiles:
    optimal_portfolio = MVO_optimisation(constituents_returns, r_original)
    r_estimated = inverse_MVO_optimisation(constituents_returns, optimal_portfolio)
    error = np.abs(r_original - r_estimated)
    errors.append(error)
    results.append((r_original, r_estimated, error))

    print(f"Original r: {r_original:.6f}, Estimated r: {r_estimated:.6f}, Error: {error:.6f}")

mean_error = np.mean(errors)

print(f"Mean Error across all tested r values: {mean_error:.6f}")

# Create DataFrame for results
df = pd.DataFrame(results, columns=['Original r', 'Estimated r', 'Absolute Error'])

# Statistical Summary
error_description = df['Absolute Error'].describe()
print(error_description)

# Plotting errors with scatter plot, trend line and confidence interval of 95%
plt.figure(figsize=(12, 6))
sns.regplot(x='Original r', y='Absolute Error', data=df, color='blue', scatter_kws={'s': 50},
            line_kws={'color': 'red', 'label': 'Regression Line'})
plt.plot(df['Original r'], df['Absolute Error'], marker='o', color='blue', linestyle='--', label='Error Trajectory', alpha=0.7)
plt.title('Error Analysis in Forward-Inverse Validation of IPO Component')
plt.xlabel('Original Risk Profile (r)')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.show()