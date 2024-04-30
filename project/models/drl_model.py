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

            # Train the model for evaluation interval timesteps
            self.model.learn(total_timesteps=10000)

            # Evaluate the model
            evaluation_results = self.evaluate(num_episodes=1)
            average_reward = np.mean(evaluation_results["rewards"])
            
            # If model is better save it
            if average_reward > self.best_reward:
                self.best_reward = average_reward
                self.save_model("project/models/best_model.zip")
                print(f"New best model saved with average reward: {average_reward}")

            # Plot results at every interval
            plot_evaluation_results(evaluation_results)

        # Plot estimates of theta values
        plot_theta_values()

        print("Training completed.")

    def evaluate(self, num_episodes=1):
        episode_rewards = []
        estimation_errors = []
        estimated_thetas = []
        true_thetas = []

        for _ in range(num_episodes):
            state, _ = self.env.reset(eval_mode=True)
            terminated = False
            while not terminated:
                action, _ = self.model.predict(state, deterministic=True)
                next_state, reward, terminated, _, info = self.env.step(action)
                state = next_state
    
                true_theta = info['true_theta']
                estimated_theta = info['estimated_theta']
                error = abs(estimated_theta - true_theta)

                episode_rewards.append(reward)
                estimation_errors.append(error)
                estimated_thetas.append(estimated_theta)
                true_thetas.append(true_theta)

        return {
            "rewards": episode_rewards,
            "estimation_errors": estimation_errors,
            "estimated_thetas": estimated_thetas,
            "true_thetas": true_thetas,
        }

    def save_model(self, path):
        """Saves the model to the specified path."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Loads the model from the specified path."""
        self.model = PPO.load(path, env=self.env, custom_objects={"policy": CustomActorCriticPolicy})
        print(f"Model loaded from {path}")

def plot_evaluation_results(evaluation_info):
    estimation_errors = evaluation_info['estimation_errors']
    estimated_thetas = evaluation_info['estimated_thetas']
    true_thetas = evaluation_info['true_thetas']

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Estimation errors
    color = 'tab:red'
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Estimation Error', color=color)
    ax1.plot(estimation_errors, color=color, marker='o', label='Estimation Error')
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Theta', color=color)  # we already handled the x-label with ax1
    ax2.plot(estimated_thetas, color=color, linestyle='--', label='Estimated Theta')
    ax2.plot(true_thetas, color='tab:green', linestyle='-.', label='True Theta')
    ax2.tick_params(axis='y', labelcolor=color)

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Title and grid
    plt.title('Estimation Errors and Theta Values Over Time')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_theta_values():
    path = "project/data/theta_values.json"

    # Load the theta values
    with open(path, 'r') as f:
        theta_values = np.array(json.load(f))
        print(f"theta values loaded from {path}")

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
    plt.ylabel('Risk Profile')
    plt.title('Convergence of Estimated Risk Profile Over Training Timesteps by Market Condition')

    # Adding legend to the plot, adjusting location to avoid overlap
    plt.legend(loc='lower right')

    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Extracting prices, returns, and volatility data
    constituents_prices, constituents_returns, constituents_volatility = load_and_filter_data('project/data/VTI.csv', 'project/data/^TNX.csv')

    # Initialise, train, and evaluate DRL agent
    agent = DRLAgent(constituents_prices, constituents_returns, constituents_volatility)

    model_path = "project/models/best_model.zip" # Path to save or load model from
    train_model = True # Option to train or load already trained model

    if train_model:
        agent.train(total_timesteps=100000)

    # Load the best model for evaluation
    agent.load_model(model_path)

    # Evaluate the model
    evaluation_info = agent.evaluate(num_episodes=1)
    plot_evaluation_results(evaluation_info)