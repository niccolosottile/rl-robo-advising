import numpy as np

# Assuming PortfolioEnv is already imported or defined in your context
from project.envs.portfolio_env import PortfolioEnv
from project.utils.data_loader import load_and_prepare_prices, load_and_prepare_returns, load_and_prepare_volatility

def validate_environment():
    # Instantiate your environment
    env = PortfolioEnv(
        constituents_prices=load_and_prepare_prices('project/data/ACWI.csv', 'project/data/AGGU.L.csv'),
        constituents_returns=load_and_prepare_returns('project/data/ACWI.csv', 'project/data/AGGU.L.csv'), 
        consitutents_volatility=load_and_prepare_volatility('project/data/ACWI.csv', 'project/data/AGGU.L.csv'), 
        lookback_window_size=100, 
        use_portfolio=False
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
        
        # Perform a step in the environment
        new_observation, reward, done, info = env.step(action)
        
        # Print the results to inspect
        print(f"Action Taken: {action}")
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
