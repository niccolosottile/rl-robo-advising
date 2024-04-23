import numpy as np

# Assuming PortfolioEnv is already imported or defined in your context
from project.envs.portfolio_env import PortfolioEnv
from project.utils.data_loader import load_and_filter_data

def validate_environment():
    # Instantiate your environment
    constituents_prices, constituents_returns, constituents_volatility = load_and_filter_data('project/data/ACWI.csv', 'project/data/AGGU.L.csv')

    env = PortfolioEnv(
        constituents_prices=constituents_prices,
        constituents_returns=constituents_returns,
        consitutents_volatility=constituents_volatility,
    )

    # Reset the environment to start
    observation = env.reset()

    # Number of steps to simulate
    num_steps = 100

    # Manually step through the environment
    for step in range(num_steps):
        print(f"Step: {step}")
        
        # Random action as per the environment's action space
        action = env.action_space.sample()

        portfolio_choice = action[:-1] # Portfolio allocation decision
        ask_investor = action[-1] > 0.5 # Decision to ask the investor

        normalized_portfolio_choice = portfolio_choice / np.sum(portfolio_choice)
        
        # Perform a step in the environment
        new_observation, reward, done, info = env.step(action)
        
        # Print the results to inspect
        print(f"Action Taken (formatted): {normalized_portfolio_choice.tolist() if not ask_investor else False}")
        print(f"New Observation: {new_observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        
        # Check if the episode is done and reset if it is
        if done:
            print("Resetting environment\n")
            observation = env.reset()

if __name__ == "__main__":
    validate_environment()
