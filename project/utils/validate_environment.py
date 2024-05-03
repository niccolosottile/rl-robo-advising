from project.utils.data_loader import load_and_filter_data
from project.envs.portfolio_env import PortfolioEnv

def validate_environment():
    """Use to visually validate the environment's functioning."""
    # Load prices, returns, and volatilities for environment
    constituents_prices, constituents_returns, constituents_volatility = load_and_filter_data('project/data/VTI.csv', 'project/data/^TNX.csv')

    # Initialise the environment
    env = PortfolioEnv(
        constituents_prices=constituents_prices,
        constituents_returns=constituents_returns,
        consitutents_volatility=constituents_volatility,
    )

    # Reset the environment
    _ = env.reset()

    # Manually step through the environment
    for step in range(100):
        print(f"Step: {step}")
        
        # Random action as per the environment's action space
        action = env.action_space.sample()

        portfolio_choice = action[:-1] # Portfolio allocation decision
        ask_investor = action[-1] > 0.5 # Investor solicitation decision
        
        # Perform a step in the environment
        new_observation, reward, done, info = env.step(action)
        
        # Print the results to inspect
        print(f"Action Taken: {portfolio_choice.tolist() if not ask_investor else False}")
        print(f"New Observation: {new_observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        
        # Check if the episode is done and reset if it is
        if done:
            print("Resetting environment\n")
            _ = env.reset()

if __name__ == "__main__":
    validate_environment()
