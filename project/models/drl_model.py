from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
import torch.nn as nn
import numpy as np
from project.envs.portfolio_env import PortfolioEnv
from project.utils.data_loader import load_and_filter_data
import matplotlib.pyplot as plt
import json

class CustomNetwork(nn.Module):
    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 3, last_layer_dim_vf: int = 1):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network for portfolio weights
        self.portfolio_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_pi - 1)  # Outputs for two portfolio actions
        )
        # Policy network for solicitation action
        self.solicitation_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output for binary solicitation action
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_vf)
        )

    def forward_actor(self, features):
        portfolio_actions = th.softmax(self.portfolio_net(features), dim=-1)
        solicitation_action = th.sigmoid(self.solicitation_net(features))
        return th.cat((portfolio_actions, solicitation_action), dim=-1)

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

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
            r=0.05,
            max_theta=0.17,
            min_theta=0.01
            )

        # Initialize PPO model with a Multi-Layer Perceptron (MLP) policy
        self.model = PPO(CustomActorCriticPolicy, self.env, verbose=1)

        # Initialize the best reward as very low
        self.best_reward = -float('inf') 

        print("Agent initialised.")

    def train(self, total_timesteps=10000):
        for current_total_timesteps in range(0, total_timesteps, 10000):
            print("Total timesteps trained so far: ", current_total_timesteps)

            # Train the model for eval_interval timesteps
            self.model.learn(total_timesteps=10000)

            # Evaluate the model
            evaluation_results = self.evaluate(num_episodes=1)
            average_reward = np.mean(evaluation_results["rewards"])
            
            # If model is better save it
            if average_reward > self.best_reward:
                self.best_reward = average_reward
                self.save_model("project/models/best_model.zip")
                print(f"New best model saved with average reward: {average_reward}")

        print("Training completed.")

    def evaluate(self, num_episodes=1):
        episode_rewards = []
        estimation_errors = []

        for _ in range(num_episodes):
            state, _ = self.env.reset(eval_mode=True)
            terminated = False
            while not terminated:
                action, _ = self.model.predict(state, deterministic=True)
                #print(action)
                next_state, reward, terminated, _, info = self.env.step(action)
                state = next_state
                episode_rewards.append(reward)
                true_theta = info['true_theta']
                estimated_theta = info['estimated_theta']
                error = abs(estimated_theta - true_theta)
                estimation_errors.append(error)

        return {
            "rewards": episode_rewards,
            "estimation_errors": estimation_errors
        }

    def save_model(self, path):
        """Saves the model to the specified path."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Loads the model from the specified path."""
        self.model = PPO.load(path, env=self.env, custom_objects={"policy": CustomActorCriticPolicy})
        print(f"Model loaded from {path}")

    def load_theta_values(self, path):
        """Loads theta values from the specified path."""
        with open(path, 'r') as f:
            self.theta_values = np.array(json.load(f))
        print(f"theta values loaded from {path}")

def plot_evaluation_results(evaluation_info):
    rewards = evaluation_info['rewards']
    estimation_errors = evaluation_info['estimation_errors']

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plotting rewards
    axs[0].plot(rewards, color='blue', label='Rewards')
    axs[0].set_title('Rewards Over Timesteps')
    axs[0].set_xlabel('Timestep')
    axs[0].set_ylabel('Reward')
    axs[0].legend()

    # Plotting estimation errors
    axs[1].plot(estimation_errors, color='red', label='Estimation Errors')
    axs[1].set_title('Estimation Errors Over Timesteps')
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('Absolute Error')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Extracting prices, returns, and volatility data
    constituents_prices, constituents_returns, constituents_volatility = load_and_filter_data('project/data/VTI.csv', 'project/data/^TNX.csv')

    # Initialise, train, and evaluate DRL agent
    agent = DRLAgent(constituents_prices, constituents_returns, constituents_volatility)

    model_path = "project/models/best_model.zip" # Path to save or load model from
    train_model = True # Option to train or load already trained model

    if train_model:
        agent.train(total_timesteps=200000)

    # Load the model for evaluation
    agent.load_model(model_path)

    # Evaluate the model
    evaluation_info = agent.evaluate(num_episodes=1)
    plot_evaluation_results(evaluation_info)

    theta_values_path = "project/data/theta_values.json"  # Path to save or load theta values from

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
    true_risk_profiles = [0.22, 0.335, 0.09, 0.855, 0.68, 0.48]
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