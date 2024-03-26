import numpy as np
import gymnasium as gym
from project.ipo_agent.forward_problem import forward_problem
from project.ipo_agent.inverse_problem import inverse_problem
from project.utils.utility_function import mean_variance_utility

# Notes:
# Need to modify returns so that they are daily (if using daily timesteps)
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
        self.phi = 15 # Current estimate of true risk profile
        self.r = 5 # Bounds size of investor mistakes about true risk profile
        self.K = 0.0008 # Opportunity cost of soliciting investor choice
        self.current_phi = None
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
            market_condition = 'low'
        elif weighted_volatility > high_threshold_weighted:
            market_condition = 'high'
        else:
            market_condition = 'medium'
        
        return market_condition

    def simulate_investor_behaviour(self):
        self.n_solicited += 1 # Increase solicited count
        sampled_phi = np.random.normal(self.phi, self.r) # Sample above mean phi with std of r
        sampled_phi = max(min(sampled_phi, 20), 1) # Clip at boundaries of valid phi value
            
        return sampled_phi

    def calculate_reward(self, ask_investor):
        if ask_investor:
            # Generate risk profile corresponding to portfolio using IPO
            inferred_phi = inverse_problem(self.constituents_returns.iloc[:self.current_timestep+1, :], self.current_portfolio, self.current_phi, self.M, self.learning_rate, True)

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

        if ask_investor:
            # Real environment wouldn't require this
            # Simulate current investor risk profile
            investor_phi = self.simulate_investor_behaviour()

            # Generate portfolio corresponding to risk profile using PO
            portfolio_choice = forward_problem(self.constituents_returns.iloc[:self.current_timestep+1, :], investor_phi, True)[-1] # Take portfolio at timestep t 

        # Normalize the portfolio weights to sum to 1
        normalized_portfolio_choice = portfolio_choice / np.sum(portfolio_choice)

        self.current_timestep += 1 # Increment current timestep
        self.current_portfolio = normalized_portfolio_choice # Update current portfolio
        self.current_market_condition = self.get_market_condition() # Retrieve new market condition
        next_state = self.get_state() # Retrieve next state
        reward = self.calculate_reward(ask_investor) # Calculate reward
        done = self.current_timestep >= self.n_timesteps - 1 # Check if episode has ended
        info = {}

        return next_state, reward, done, info

    def reset(self):
        self.current_timestep = 1200 # Reset timestep
        self.current_portfolio = np.full((self.n_assets,), 1/self.n_assets) # Reset to equally weighted portfolio
        self.current_market_condition = self.get_market_condition()

        return self.get_state()

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Current timestep: {self.current_timestep}, Current portfolio: {self.current_portfolio}, Market condition: {self.current_market_condition}")
