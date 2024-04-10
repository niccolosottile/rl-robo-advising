import numpy as np
import gymnasium as gym
from project.ipo_agent.forward_problem import forward_problem
from project.ipo_agent.inverse_problem import inverse_problem
from project.utils.utility_function import mean_variance_utility
from project.utils.others import normalize_portfolio

# Notes:
# Market condition not used at the moment (no reason for market condition as it was used to estimate returns?)
# Consider changing state to market prices or portfolio value that's what market condition is for
class PortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, constituents_prices, constituents_returns, consitutents_volatility, lookback_window_size, use_portfolio=True):
        super(PortfolioEnv, self).__init__()
        self.constituents_prices = constituents_prices
        self.constituents_returns = constituents_returns
        self.constituents_volatility = consitutents_volatility
        self.low_threshold = np.array(self.constituents_volatility.quantile(0.33))
        self.high_threshold = np.array(self.constituents_volatility.quantile(0.66))
        self.lookback_window_size = lookback_window_size # Affects lookback window L of PO and IPO agents
        self.use_portfolio = use_portfolio # Uses portfolio as part of state representation
        self.n_assets = self.constituents_returns.shape[1]
        self.n_timesteps = self.constituents_returns.shape[0]
        self.current_timestep = 1
        
        # Define action space as a given portfolio allocation plus option to solicit investor
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

        # Define observation space
        if self.use_portfolio:
            # Observation space includes both portfolio and market condition if use_portfolio is True
            self.observation_space = gym.spaces.Dict({
                "current_portfolio": gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32),
                "market_condition": gym.spaces.Discrete(3) # Low, Medium, High
            })
        else:
            # Observation space includes only market condition if use_portfolio is False
            self.observation_space = gym.spaces.Discrete(3)

        # State representations
        self.current_portfolio = np.full((self.n_assets,), 1/self.n_assets)  # Start with equally weighted portfolio
        self.current_market_condition = self.get_market_condition()

        # Investor behaviour parameters, set of phi values is {1 to 30}
        self.phi = 0.5 # True risk profile
        self.r = 0.5 # Bounds size of investor mistakes about true risk profile
        self.K = 0.0008 / 21 # Opportunity cost of soliciting investor choice (converted to daily basis based on monthly trading days)
        self.current_phi = 0.5 # Current estimate of true risk profile
        self.n_solicited = 0 # Number of times investor is solicited

        # Hyperparameters for IPO agent
        self.M = 100
        self.learning_rate = 100

    def get_state(self):
            if self.use_portfolio:
                return {
                    "current_portfolio": self.current_portfolio, 
                    "market_condition": self.current_market_condition
                }
            else:
                return self.current_market_condition

    def get_market_condition(self):
        current_volatilities = self.constituents_volatility.iloc[self.current_timestep] # Extract asset volatilities at current timestep
        weighted_volatility = np.dot(self.current_portfolio, current_volatilities) # Calculate weighted portfolio volatility

        # Calculate weighted portfolio thresholds
        low_threshold_weighted = np.dot(self.current_portfolio, self.low_threshold)
        high_threshold_weighted = np.dot(self.current_portfolio, self.high_threshold)
        
        # Determine market condition
        if weighted_volatility <= low_threshold_weighted:
            market_condition = 0 #'low'
        elif weighted_volatility > high_threshold_weighted:
            market_condition = 2 #'high'
        else:
            market_condition = 1 #'medium'
        
        return market_condition

    def simulate_investor_behaviour(self):
        self.n_solicited += 1 # Increase solicited count
        sampled_phi = np.random.normal(self.phi, self.r) # Sample above mean phi with std of r
        sampled_phi = max(min(sampled_phi, 1), 0.1) # Clip at boundaries of valid phi value
            
        return sampled_phi

    def calculate_reward(self, ask_investor):
        if ask_investor:
            # Generate risk profile corresponding to portfolio using IPO
            inferred_phi = inverse_problem(self.constituents_returns.iloc[:self.current_timestep+1, :], self.current_portfolio, self.current_phi, self.M, self.learning_rate, only_last=True, verbose=False)

            # Update estimate of phi
            if self.n_solicited == 1:
                self.current_phi = inferred_phi
            else:
                self.current_phi = self.current_phi + 1/self.n_solicited * (inferred_phi - self.current_phi)

        # Calculate reward using mean-variance utility function and current estimate of phi
        reward = mean_variance_utility(self.constituents_returns.iloc[:self.current_timestep+1, :], self.current_portfolio, self.current_phi)

        if ask_investor:
            # Reduce by cost of soliciting
            portfolio_value = np.dot(self.constituents_prices.iloc[self.current_timestep], self.current_portfolio)
            reward -= self.K * portfolio_value

        return reward
        
    def step(self, action):
        portfolio_choice = action[:-1] # Portfolio allocation decision
        ask_investor = action[-1] > 0.5 # Decision to ask the investor

        self.current_timestep += 1 # Increment current timestep

        if ask_investor:
            # Real environment wouldn't require this
            # Simulate current investor risk profile
            investor_phi = self.simulate_investor_behaviour()

            # Generate portfolio corresponding to risk profile using PO
            portfolio_choice = forward_problem(self.constituents_returns.iloc[:self.current_timestep+1, :], investor_phi, only_last=True, verbose=False)[-1] # Take portfolio at timestep t 

        # Normalize the portfolio weights to sum to 1
        normalized_portfolio_choice = normalize_portfolio(portfolio_choice)

        self.current_portfolio = normalized_portfolio_choice # Update current portfolio
        self.current_market_condition = self.get_market_condition() # Retrieve new market condition
        next_state = self.get_state() # Retrieve next state
        reward = self.calculate_reward(ask_investor) # Calculate reward
        terminated = self.current_timestep >= self.n_timesteps - 1 # Check if episode has ended
        truncated = False # Episodes aren't being cut short
        info = {}

        return next_state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Use seed to reproduce investor behaviour simulation
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        self.n_solicited = 0 # Reset times investor solicited
        self.current_timestep = 1200 # Reset timestep
        self.current_portfolio = np.full((self.n_assets,), 1/self.n_assets) # Reset to equally weighted portfolio
        self.current_market_condition = self.get_market_condition()
        info = {}

        return self.get_state(), info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Current timestep: {self.current_timestep}, Current portfolio: {self.current_portfolio}, Market condition: {self.current_market_condition}")
