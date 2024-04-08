import numpy as np
from stable_baselines3 import PPO
from project.envs.portfolio_env import PortfolioEnv
from project.utils.data_loader import load_and_prepare_prices, load_and_prepare_returns, load_and_prepare_volatility

class DRLAgent:
    def __init__(self, constituents_prices, constituents_returns, constituents_volatility, lookback_window_size=100, use_portfolio=True):
        self.constituents_prices = constituents_prices
        self.constituents_returns = constituents_returns
        self.constituents_volatility = constituents_volatility
        self.lookback_window_size = lookback_window_size
        self.use_portfolio = use_portfolio
        
        # Setup the environment with market data
        self.env = PortfolioEnv(
            constituents_prices = self.constituents_prices,
            constituents_returns = self.constituents_returns, 
            consitutents_volatility = self.constituents_volatility,                     
            lookback_window_size = self.lookback_window_size,
            use_portfolio = self.use_portfolio)

        # Initialize PPO model with a Multi-Layer Perceptron (MLP) policy
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed.")

    def evaluate(self):
        observation = self.env.reset()
        while True:
            action, _ = self.model.predict(observation, deterministic=True)
            observation, reward, done, info = self.env.step(action)
            if done:
                break
        return info

if __name__ == "__main__":
    # Extracting constituents prices
    constituents_prices = load_and_prepare_prices('project/data/ACWI.csv', 'project/data/AGGU.L.csv')
    # Extracting constituents returns 253-day rolling periods
    constituents_returns = load_and_prepare_returns('project/data/ACWI.csv', 'project/data/AGGU.L.csv')
    # Extracting constituents volatility
    constituents_volatility = load_and_prepare_volatility('project/data/ACWI.csv', 'project/data/AGGU.L.csv')

    # Initialise, train, and evaluate DRL agent
    agent = DRLAgent(constituents_prices, constituents_returns, constituents_volatility, use_portfolio=False)
    print("Agent initialised")
    agent.train(total_timesteps=100000)
    print("Agent trained")
    #evaluation_info = agent.evaluate()
    #print("Evaluation results:", evaluation_info)
