import numpy as np
import gymnasium as gym
from project.ipo_components.inverse_MVO_optimisation import inverse_MVO_optimisation
from project.ipo_components.MVO_optimisation import MVO_optimisation
from project.utils.utility_function import mean_variance_utility
from project.utils.others import normalize_portfolio
import json
import os

# Need to change it such that the investor behaviour depends on the market condition
# Need to develop offline and online training methodologies
class PortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, constituents_prices, constituents_returns, consitutents_volatility, lookback_window_size, phi, r):
        super(PortfolioEnv, self).__init__()

        # Historical data from preprocessed datasets
        self.constituents_prices = constituents_prices # Used to affect cost of soliciting K
        self.constituents_returns = constituents_returns # Used for utility function, and market conditions
        self.constituents_volatility = consitutents_volatility # Used for market conditions

        # Define environment size
        self.lookback_window_size = lookback_window_size # Affects lookback window L of PO and IPO agents
        self.n_assets = self.constituents_returns.shape[1]
        self.n_timesteps = self.constituents_returns.shape[0]
        self.current_timestep = 1

        # Calculate thresholds to derive market conditions
        self.vol_low_threshold = np.array(self.constituents_volatility.quantile(0.33))
        self.vol_high_threshold = np.array(self.constituents_volatility.quantile(0.66))
        self.ret_low_threshold = np.array(self.constituents_returns.quantile(0.33))
        self.ret_high_threshold = np.array(self.constituents_returns.quantile(0.66))

        # State representations
        self.current_portfolio = np.full((self.n_assets,), 1/self.n_assets)  # Start with equally weighted portfolio
        self.current_market_condition = self.get_market_condition()

        # Investor behaviour parameters, set of phi values is {0.1 to 1}
        self.phi = phi # True risk profile
        self.r = r # Bounds size of investor mistakes about true risk profile
        self.K = 0.0008 / 21 # Opportunity cost of soliciting investor choice (converted to daily basis based on monthly trading days)
        self.current_phi = 0 # Current estimate of true risk profile
        self.n_solicited = 0 # Number of times investor is solicited
        self.phi_values = []  # Store phi values for each step
        
        # Investor shift in risk profile parameters:
        self.shifted_phi = 0.4
        self.shifted_r = 0.1
        self.apply_shift = True
        self.timestep_shift = 1210

        # Define observation space based on set of market conditions
        self.observation_space = gym.spaces.Discrete(9)

        # Define action space as a given portfolio allocation plus ask space to solicit investor
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

    def get_state(self):
        return self.current_market_condition

    def get_market_condition(self):
        # Volatility based market condition
        current_volatilities = self.constituents_volatility.iloc[self.current_timestep] # Extract asset volatilities at current timestep
        weighted_volatility = np.dot(self.current_portfolio, current_volatilities) # Calculate weighted portfolio volatility
        vol_low_threshold = np.dot(self.current_portfolio, self.vol_low_threshold)
        vol_high_threshold = np.dot(self.current_portfolio, self.vol_high_threshold)

        vol_condition = (
            0 if weighted_volatility <= vol_low_threshold
            else 2 if weighted_volatility > vol_high_threshold
            else 1
        )
        
        # Returns based market condition
        current_returns = self.constituents_returns.iloc[self.current_timestep]
        weighted_returns = np.dot(self.current_portfolio, current_returns)
        ret_low_threshold = np.dot(self.current_portfolio, self.ret_low_threshold)
        ret_high_threshold = np.dot(self.current_portfolio, self.ret_high_threshold)

        ret_condition = (
            0 if weighted_returns <= ret_low_threshold
            else 2 if weighted_returns > ret_high_threshold
            else 1
        )

        # Combine the two conditions into a single market condition index
        market_condition = 3 * ret_condition + vol_condition
        
        return market_condition

    def simulate_investor_behaviour(self):
        self.n_solicited += 1 # Increase solicited count
        if self.apply_shift and self.current_timestep >= self.timestep_shift:
            sampled_phi = np.random.normal(self.shifted_phi, self.shifted_r) # Apply shifted risk profile parameters
        else:
            sampled_phi = np.random.normal(self.phi, self.r)  # Sample above mean phi with std of r
        sampled_phi = max(min(sampled_phi, 1), 0.1) # Clip at boundaries of valid phi value
            
        return sampled_phi

    def calculate_reward(self, ask_investor):
        # If investor was asked update estimate of phi
        if ask_investor:
            # Generate risk profile corresponding to portfolio using IPO
            inferred_phi = inverse_MVO_optimisation(self.constituents_returns.iloc[:self.current_timestep+1, :], self.current_portfolio)

            # Update estimate of phi
            if self.n_solicited == 1:
                self.current_phi = inferred_phi
            else:
                self.current_phi = self.current_phi + 1/self.n_solicited * (inferred_phi - self.current_phi)

        # Calculate reward using mean-variance utility function and current estimate of phi
        reward = mean_variance_utility(self.constituents_returns.iloc[:self.current_timestep+1, :], self.current_portfolio, self.current_phi)

        # If investor was asked reduce reward
        if ask_investor:
            # Reduce by cost of soliciting K based on portfolio value
            portfolio_value = np.dot(self.constituents_prices.iloc[self.current_timestep], self.current_portfolio)
            reward -= self.K * portfolio_value

        return reward
        
    def step(self, action):
        portfolio_choice = action[:-1] # Portfolio allocation decision
        ask_investor = action[-1] > 0.5 # Decision to ask the investor

        self.current_timestep += 1 # Increment current timestep

        if ask_investor:
            # Simulate current investor risk profile
            investor_phi = self.simulate_investor_behaviour()

            # Generate portfolio corresponding to risk profile using MVO optimisation
            portfolio_choice = MVO_optimisation(self.constituents_returns.iloc[:self.current_timestep+1, :], investor_phi)

        # Normalize the portfolio weights to sum to 1
        normalized_portfolio_choice = normalize_portfolio(portfolio_choice)

        self.current_portfolio = normalized_portfolio_choice # Update current portfolio
        self.current_market_condition = self.get_market_condition() # Retrieve new market condition
        next_state = self.get_state() # Retrieve next state
        reward = self.calculate_reward(ask_investor) # Calculate reward
        terminated = self.current_timestep >= self.n_timesteps - 1 # Check if episode has ended
        truncated = False # Episodes aren't being cut short
        info = {}

        self.phi_values.append(self.current_phi)  # Append current_phi value for evaluation

        return next_state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Use seed to reproduce investor behaviour simulation
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        # Write phi values to file at the end of an episode for evaluation
        if self.phi_values:  # Check if list is not empty
            phi_values_path = "project/data/phi_values.json"
            # Check if the file exists and is not empty
            if not os.path.exists(phi_values_path) or os.path.getsize(phi_values_path) == 0:
                with open(phi_values_path, 'w') as f:
                    json.dump(self.phi_values, f)
                print(f"Phi values saved to {phi_values_path}")
            self.phi_values = []  # Reset phi values for the next episode

        self.n_solicited = 0 # Reset count investor solicited
        self.current_phi = 0 # Reset phi
        self.current_timestep = 1200 # Reset timestep
        self.current_portfolio = np.full((self.n_assets,), 1/self.n_assets) # Reset to equally weighted portfolio
        self.current_market_condition = self.get_market_condition()
        info = {}

        return self.get_state(), info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Current timestep: {self.current_timestep}, Current portfolio: {self.current_portfolio}, Market condition: {self.current_market_condition}")
