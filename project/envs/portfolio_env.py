import numpy as np
import gym

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, constituents_returns, consitutents_volatility, lookback_window_size, use_portfolio=True):
        super(PortfolioEnv, self).__init__()
        self.constituents_returns = constituents_returns
        self.constituents_volatility = consitutents_volatility
        self.lookback_window_size = lookback_window_size # Affects lookback window L of PO and IPO agents
        self.use_portfolio = use_portfolio # Uses portfolio as part of state representation
        self.n_assets = self.constituents_returns.shape[1]
        self.n_timesteps = self.constituents_returns.shape[0]
        self.current_timestep = 0
        
        # Define action space as a given portfolio allocation
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

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

        # Investor behaviour parameters, set of phi values is {0.1 to 3}
        self.phi = 1.5 # Current estimate of true risk profile
        self.r = 1.5 # Bounds size of investor mistakes about true risk profile
        self.K = 0.08 # Opportunity cost of soliciting investor choice
        self.current_phi = None
        self.n_solicited = 0 # Number of times investor is solicited

    def get_market_condition(self):
        pass

    def get_state(self):
        if self.use_portfolio:
            return {
                "current_portfolio": self.current_portfolio, 
                "market_condition": self.current_market_condition
            }
        else:
            return self.current_market_condition

    def simulate_investor_behaviour(self):
        self.n_solicited += 1 # Increase solicited count
        sampled_phi = np.random.normal(self.phi, self.r) # Sample above mean phi with std of r
        sampled_phi = max(min(sampled_phi, 3), 0.1) # Clip at boundaries of valid phi value
        if self.n_solicited == 1:
            self.current_phi = sampled_phi
        else:
            self.current_phi = self.current_phi + 1/self.n_solicited * (sampled_phi - self.current_phi) 
            
        return sampled_phi

    def calculate_reward(self):
        # Use utility function (mean-variance) which takes:
        # phi (with random sampling about r as a separate function for simulation if investor action)
        # state,
        # action,
        # K (if investor action)
        pass
        
    def step(self, action):
        self.current_timestep += 1 # Increment current timestep
        self.current_portfolio = action # Update current portfolio
        self.current_market_condition = self.get_market_condition() # Retrieve new market condition
        next_state = self.get_state() # Retrieve next state
        reward = self.calculate_reward() # Calculate reward
        done = self.current_timestep >= self.n_timesteps # Check if episode has ended
        info = {}

        return next_state, reward, done, info

    def reset(self):
        self.current_timestep = 0 # Reset timestep
        self.current_portfolio = np.full((self.n_assets,), 1/self.n_assets) # Reset to equally weighted portfolio
        self.current_market_condition = self.get_market_condition()

        return self.get_state()

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Current timestep: {self.current_timestep}, Current portfolio: {self.current_portfolio}, Market condition: {self.current_market_condition}")
