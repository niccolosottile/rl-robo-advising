import numpy as np
from stable_baselines3 import PPO
from project.envs.portfolio_env import PortfolioEnv
from project.utils.data_loader import load_and_prepare_returns, load_and_prepare_volatility

class DRLAgent:
    def __init__(self, constituents_returns, market_conditions, lookback_window_size=100):
        self.constituents_returns = constituents_returns
        self.market_conditions = market_conditions
        self.lookback_window_size = lookback_window_size
        
        # Setup the environment with market data
        self.env = PortfolioEnv(
            constituents_returns = self.constituents_returns,
            market_conditions = market_conditions,                      
            lookback_window_size = self.lookback_window_size)

        # Initialize PPO model with a Multi-Layer Perceptron (MLP) policy
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps=1000):
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
    # Extracting constituents returns 253-day rolling periods
    constituents_returns = load_and_prepare_returns('project/data/ACWI.csv', 'project/data/AGGU.L.csv')
    # Extracting constituents volatility
    constituents_volatility = load_and_prepare_volatility('project/data/ACWI.csv', 'project/data/AGGU.L.csv')

    # Initialise, train, and evaluate DRL agent
    agent = DRLAgent(constituents_returns, constituents_volatility)
    agent.train(total_timesteps=100000)
    evaluation_info = agent.evaluate()
    print("Evaluation results:", evaluation_info)
