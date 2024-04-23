import numpy as np
from stable_baselines3 import PPO
from project.envs.portfolio_env import PortfolioEnv
from project.utils.data_loader import load_and_filter_data
import matplotlib.pyplot as plt
import json

class DRLAgent:
    def __init__(self, constituents_prices, constituents_returns, constituents_volatility):
        self.constituents_prices = constituents_prices
        self.constituents_returns = constituents_returns
        self.constituents_volatility = constituents_volatility
        
        # Setup the environment with market data
        self.env = PortfolioEnv(
            constituents_prices = self.constituents_prices,
            constituents_returns = self.constituents_returns, 
            consitutents_volatility = self.constituents_volatility,                     
            lookback_window_size = 1200,
            r=0.1,
            max_theta=1,
            min_theta=0.1
            )

        # Initialize PPO model with a Multi-Layer Perceptron (MLP) policy
        self.model = PPO("MlpPolicy", self.env, verbose=1)

        print("Agent initialised.")

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed.")

    def evaluate(self, num_episodes=10):
        true_risk_profile = 0.5
        total_rewards = []
        theta_values = []
        theta_deviations = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            episode_theta_values = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                total_reward += reward
                current_theta = info.get('current_theta', None)
                episode_theta_values.append(current_theta)

            # Calculate and store deviations from true risk profile for this episode
            deviations = [abs(theta - true_risk_profile) for theta in episode_theta_values if theta is not None]
            theta_deviations.append(np.mean(deviations) if deviations else None)

            total_rewards.append(total_reward)
            theta_values.extend(episode_theta_values)  # Extend to flatten list of lists

        avg_reward = np.mean(total_rewards)
        avg_deviation = np.nanmean(theta_deviations)  # Using nanmean to ignore None values
        avg_theta = np.mean(theta_values)
        print(f"Average Reward: {avg_reward}")
        print(f"Average Deviation from True Risk Profile: {avg_deviation}")

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(theta_deviations, label='Deviation from True Risk Profile per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Deviation from True Risk Profile')
        plt.title('Convergence of Estimated Risk Profile to True Risk Profile')
        plt.legend()
        plt.show()

        # More detailed evaluation data
        return {'average_reward': avg_reward, 'average_theta_deviation': avg_deviation, 'average_theta_value': avg_theta}

    def save_model(self, path):
        """Saves the model to the specified path."""
        self.model.save(path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, path):
        """Loads the model from the specified path."""
        self.model = PPO.load(path, env=self.env) 
        print(f"Model loaded from {model_path}")

    def load_theta_values(self, path):
        """Loads theta values from the specified path."""
        with open(path, 'r') as f:
            self.theta_values = np.array(json.load(f))
        print(f"theta values loaded from {path}")

if __name__ == "__main__":
    # Extracting prices, returns, and volatility data
    constituents_prices, constituents_returns, constituents_volatility = load_and_filter_data('project/data/ACWI.csv', 'project/data/AGGU.L.csv')

    # Initialise, train, and evaluate DRL agent
    agent = DRLAgent(constituents_prices, constituents_returns, constituents_volatility)

    model_path = "project/models/model.zip" # Path to save or load model from
    theta_values_path = "project/data/theta_values.json"  # Path to save or load theta values from
    train_model = True # Option to train or load already trained model

    if train_model:
        agent.train(total_timesteps=100000)
        # Save the trained model
        agent.save_model(model_path)
    else:
        # Load the model for evaluation
        agent.load_model(model_path)

    # Load the theta values for evaluation
    agent.load_theta_values(theta_values_path)
    
    # Load the theta values
    theta_values = agent.theta_values

    # Create a plot with specified figure size
    plt.figure(figsize=(15, 8))

    # Define a color palette
    colours = plt.cm.nipy_spectral(np.linspace(0, 1, theta_values.shape[1]))

    # Plotting the estimated risk profile for each market condition
    for idx in range(theta_values.shape[1]):
        plt.plot(theta_values[:, idx], label=f'Estimated Risk Profile Market {idx+1}' if idx == 0 else "_nolegend_", color=colours[idx])

    # Adding horizontal lines to represent the true risk profile for each market condition
    true_risk_profiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for idx, profile in enumerate(true_risk_profiles):
        plt.axhline(y=profile, linestyle='--', label=f'Risk Profile Market {idx+1}', color=colours[idx])

    # Setting labels and title
    plt.xlabel('Timestep')
    plt.ylabel('Risk Profile (theta)')
    plt.title('Convergence of Estimated Risk Profile Over Training Timesteps by Market Condition')

    # Adding legend to the plot, adjusting location to avoid overlap
    plt.legend(loc='lower right')

    # Display the plot
    plt.show()

    # Evaluate the model
    #evaluation_info = agent.evaluate(num_episodes=100)
    #print("Evaluation results:", evaluation_info)
