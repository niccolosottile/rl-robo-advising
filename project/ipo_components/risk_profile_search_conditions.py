import numpy as np
import pandas as pd
from project.ipo_components.MVO_optimisation import MVO_optimisation
from project.utils.data_loader import load_and_filter_data
from collections import defaultdict

def calculate_market_conditions(constituents_returns, constituents_volatility, vol_thresholds, ret_thresholds):
    market_conditions = []
    for t in range(len(constituents_returns)):
        vol = np.array(constituents_volatility.iloc[t])
        ret = np.array(constituents_returns.iloc[t])
        
        vol_condition = (
            0 if vol[0] <= vol_thresholds[0]
            else 2 if vol[0] > vol_thresholds[1]
            else 1
        )
        
        ret_condition = (
            0 if ret[0] <= ret_thresholds[0]
            else 2 if ret[0] > ret_thresholds[1]
            else 1
        )
        
        market_condition = 3 * ret_condition + vol_condition
        market_conditions.append(market_condition)
    
    return market_conditions

def aggregate_returns_by_condition(constituents_returns, market_conditions):
    aggregated_returns = defaultdict(list)
    for t, condition in enumerate(market_conditions):
        if t > 4000:  # Ensure there's at least one previous day to include
            aggregated_data = constituents_returns.iloc[:t+1, :]  # All data up to current day
            aggregated_returns[condition].append(aggregated_data)
    
    return aggregated_returns

def is_valid_portfolio(portfolio):
    # Ensure the portfolio is not a corner solution
    return np.all(portfolio > 0.01) and np.all(portfolio < 0.99)

def find_valid_r_range_for_condition(returns_data, r_min=0.01, r_max=30.0, step=0.01):
    valid_r_lower = None
    valid_r_upper = None

    # Iterate through each risk aversion coefficient value r
    for r in np.arange(r_min, r_max, step):
        all_portfolios_valid = True  # Assume valid unless proven otherwise

        # Calculate portfolio for each timestep's cumulative returns
        for timestep_returns in returns_data:
            portfolio = np.array(MVO_optimisation(timestep_returns, r))
            if not is_valid_portfolio(portfolio):
                all_portfolios_valid = False
                break  # Stop checking further if any portfolio is invalid

        if all_portfolios_valid:
            if valid_r_lower is None:  # First valid r found
                valid_r_lower = r
            valid_r_upper = r  # Update upper bound to last valid r
        elif valid_r_lower is not None:  # Found invalid r after finding valid range
            break  # Exit loop once an invalid r is found after establishing a valid range

    if valid_r_lower is None:
        return (None, None)  # Return None if no valid r found

    return (round(valid_r_lower, 2), round(valid_r_upper, 2))

# Load data for returns and volatility
_, constituents_returns, constituents_volatility = load_and_filter_data('project/data/VTI.csv', 'project/data/^TNX.csv')

# Calculate thresholds for low, medium, and high for returns and volatility
vol_low_threshold = np.array(constituents_volatility.quantile(0.33))
vol_high_threshold = np.array(constituents_volatility.quantile(0.66))
ret_low_threshold = np.array(constituents_returns.quantile(0.33))
ret_high_threshold = np.array(constituents_returns.quantile(0.66))

print("Volatility thresholds: ", vol_low_threshold, vol_high_threshold)
print("Returns thresholds: ", ret_low_threshold, ret_high_threshold)

print(constituents_returns)

market_conditions = calculate_market_conditions(constituents_returns, constituents_volatility, [vol_low_threshold[0], vol_high_threshold[0]], [ret_low_threshold[0], ret_high_threshold[0]])
aggregated_returns = aggregate_returns_by_condition(constituents_returns, market_conditions)

valid_r_ranges = {}
for condition, returns in aggregated_returns.items():
    print("Number of timesteps in condition: ", len(returns))
    valid_r_range = find_valid_r_range_for_condition(returns)
    valid_r_ranges[condition] = valid_r_range
    print(f"Market Condition {condition}: Valid risk profile range: {valid_r_range}")

#print(valid_r_ranges)