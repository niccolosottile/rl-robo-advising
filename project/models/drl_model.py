import numpy as np
from stable_baselines3 import PPO
from project.envs.portfolio_env import PortfolioEnv
from project.utils.data_loader import load_and_prepare_prices, load_and_prepare_returns, load_and_prepare_volatility
import matplotlib.pyplot as plt
import json

class DRLAgent:
    def __init__(self, constituents_prices, constituents_returns, constituents_volatility, lookback_window_size, phi, r, use_portfolio=True):
        self.constituents_prices = constituents_prices
        self.constituents_returns = constituents_returns
        self.constituents_volatility = constituents_volatility
        self.lookback_window_size = lookback_window_size
        self.use_portfolio = use_portfolio
        self.phi = phi,
        self.r = r
        
        # Setup the environment with market data
        self.env = PortfolioEnv(
            constituents_prices = self.constituents_prices,
            constituents_returns = self.constituents_returns, 
            consitutents_volatility = self.constituents_volatility,                     
            lookback_window_size = self.lookback_window_size,
            phi=self.phi,
            r=self.r,
            use_portfolio = self.use_portfolio
            )

        # Initialize PPO model with a Multi-Layer Perceptron (MLP) policy
        self.model = PPO("MlpPolicy", self.env, verbose=1)

        print("Agent initialised.")

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed.")

    def evaluate(self, num_episodes=10, true_risk_profile=0.5):
        total_rewards = []
        phi_values = []
        phi_deviations = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            episode_phi_values = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                total_reward += reward
                current_phi = info.get('current_phi', None)
                episode_phi_values.append(current_phi)

            # Calculate and store deviations from true risk profile for this episode
            deviations = [abs(phi - true_risk_profile) for phi in episode_phi_values if phi is not None]
            phi_deviations.append(np.mean(deviations) if deviations else None)

            total_rewards.append(total_reward)
            phi_values.extend(episode_phi_values)  # Extend to flatten list of lists

        avg_reward = np.mean(total_rewards)
        avg_deviation = np.nanmean(phi_deviations)  # Using nanmean to ignore None values
        avg_phi = np.mean(phi_values)
        print(f"Average Reward: {avg_reward}")
        print(f"Average Deviation from True Risk Profile: {avg_deviation}")

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(phi_deviations, label='Deviation from True Risk Profile per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Deviation from True Risk Profile')
        plt.title('Convergence of Estimated Risk Profile to True Risk Profile')
        plt.legend()
        plt.show()

        # More detailed evaluation data
        return {'average_reward': avg_reward, 'average_phi_deviation': avg_deviation, 'average_phi_value': avg_phi}

    def save_model(self, path):
        """Saves the model to the specified path."""
        self.model.save(path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, path):
        """Loads the model from the specified path."""
        self.model = PPO.load(path, env=self.env) 
        print(f"Model loaded from {model_path}")

    def load_phi_values(self, path):
        """Loads phi values from the specified path."""
        with open(path, 'r') as f:
            self.phi_values = json.load(f)
        print(f"Phi values loaded from {path}")

if __name__ == "__main__":
    # Extracting constituents prices
    constituents_prices = load_and_prepare_prices('project/data/ACWI.csv', 'project/data/AGGU.L.csv')
    # Extracting constituents returns 253-day rolling periods
    constituents_returns = load_and_prepare_returns('project/data/ACWI.csv', 'project/data/AGGU.L.csv')
    # Extracting constituents volatility
    constituents_volatility = load_and_prepare_volatility('project/data/ACWI.csv', 'project/data/AGGU.L.csv')

    # Initialise, train, and evaluate DRL agent
    agent = DRLAgent(constituents_prices, constituents_returns, constituents_volatility, 1200, 0.2, 0.1, use_portfolio=False)

    model_path = "project/models/model.zip" # Path to save or load model from
    phi_values_path = "project/data/phi_values.json"  # Path to save or load phi values from
    train_model = False # Option to train or load already trained model

    if train_model:
        agent.train(total_timesteps=1000)
        # Save the trained model
        agent.save_model(model_path)
    else:
        # Load the model for evaluation
        agent.load_model(model_path)

    # Load the phi values for evaluation
    agent.load_phi_values(phi_values_path)
    
    # Load the dynamic agent's phi values
    dynamic_phi_values = agent.phi_values

    # Load the fixed agent's phi values from a JSON file
    with open("project/data/ipo_phi_values.json", 'r') as f:
        fixed_phi_values = json.load(f)

    # Determine the length for x-axis scaling
    max_length = max(len(dynamic_phi_values), len(fixed_phi_values))

    # Create a plot with specified figure size
    plt.figure(figsize=(10, 6))

    # Plotting the estimated risk profile from dynamic agent
    plt.plot(dynamic_phi_values, label='Dynamic Agent Estimated Risk Profile')

    # Plotting the estimated risk profile from fixed agent
    plt.plot(fixed_phi_values, label='Fixed Agent Estimated Risk Profile', linestyle='--')

    # Adding two horizontal lines to represent the change in the true risk profile
    plt.axhline(y=0.2, color='r', linestyle='-', xmin=0, xmax=13/max_length, label='True Risk Profile until timestep 10')
    plt.axhline(y=0.4, color='r', linestyle='-', xmin=13/max_length, xmax=1, label='True Risk Profile after timestep 10')

    # Mark the change point
    plt.axvline(x=10, color='g', linestyle='--', label='Change in Risk Profile')

    # Setting labels and title
    plt.xlabel('Timestep')
    plt.ylabel('Risk Profile (Phi)')
    plt.title('Convergence of Estimated Risk Profile Over Training Timesteps')

    # Adding legend to the plot
    plt.legend()

    # Display the plot
    plt.show()

    # Evaluate the model
    #evaluation_info = agent.evaluate(num_episodes=100)
    #print("Evaluation results:", evaluation_info)
