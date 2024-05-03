import json
import os
import numpy as np
import gymnasium as gym
from project.ipo_components.MVO_optimisation import MVO_optimisation
from project.ipo_components.inverse_MVO_optimisation import inverse_MVO_optimisation

class PortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, constituents_prices, constituents_returns, consitutents_volatility, r, max_theta, min_theta):
        """Initialises the Robo-Advising environment."""
        super(PortfolioEnv, self).__init__()

        # Historical data from preprocessed datasets
        self.constituents_prices = constituents_prices # Used to affect cost of soliciting K
        self.constituents_returns = constituents_returns # Used for utility function, and market conditions
        self.constituents_volatility = consitutents_volatility # Used for market conditions

        # Define environment hyperparameters
        self.n_assets = self.constituents_returns.shape[1]
        self.n_timesteps = self.constituents_returns.shape[0]
        self.current_timestep = 1
        theta_bounds, market_conditions = self.initialize_theta_bounds_and_conditions() # Derive valid theta bounds and market conditions
        self.market_conditions =  market_conditions # Change to None to work with dynamic market conditions
        self.n_market_conditions = len(set(self.market_conditions)) if self.market_conditions else 9

        # Calculate thresholds to derive market conditions
        self.vol_thresholds = [self.constituents_volatility.quantile(q) for q in [0.33, 0.66, 0.86]]
        self.ret_thresholds = [self.constituents_returns.quantile(q) for q in [0.33, 0.66, 0.86]]

        # State representations
        self.current_portfolio = np.full((self.n_assets,), 1/self.n_assets)  # Start with equally weighted portfolio
        self.current_market_condition = self.get_market_condition()

        # Investor behaviour parameters
        self.r = r # Variance in investor's mistakes
        self.max_theta = max_theta
        self.min_theta = min_theta
        self.base_theta = (max_theta - min_theta) / 2 + min_theta
        self.theta_bounds = theta_bounds
        self.theta = np.array([np.mean(bounds) for bounds in self.theta_bounds.values()]) # Behaviour based on mean of valid bounds
        self.current_theta = np.array([self.base_theta for _ in range(self.n_market_conditions)]) # Current estimates of true risk profile
        self.n_solicited = np.zeros(self.n_market_conditions)  # Count solicitations per market condition
        self.solicited_this_step = {} # Avoids simulating behaviour multiple times
        self.K = 0.00003809523 # Opportunity cost of soliciting investor choice on daily basis
        self.portfolio_value = 55160 # Based on average robo-advisor user portfolio value
        self.solicitation_penalty = self.K * self.portfolio_value * 0.3 # Scaled to encourage exploration in unseen states 
        self.theta_values = []  # Store theta estimates for evaluation purposes

        # Define observation space based on set of market conditions
        self.observation_space = gym.spaces.Discrete(self.n_market_conditions)
        # Define action space as portfolio allocation plus ask investor position
        self.action_space = gym.spaces.Box(low=0.01, high=0.99, shape=(self.n_assets + 1,), dtype=np.float32)

        # Parameters for training and evaluation
        self.eval_mode = False
        self.train_end_step = 2745 + int(0.8 * (self.n_timesteps - 2745))  # 80% of data for training
        self.eval_end_step = self.n_timesteps  # Remaining 20% for evaluation

        # Optimise training process by caching optimal portfolios in reward function
        self.optimal_portfolio_cache = {}  # Cache for storing optimal portfolios
        self.episode_count = 0
        self.caching_threshold = 50  # Start caching after estimate of theta has converged

    def calculate_market_conditions(self):
        """Calculates market conditions at initialisation based on thresholds."""

        # Calculate conditions
        market_conditions = []

        for t in range(len(self.constituents_returns)):
            vol = np.array(self.constituents_volatility.iloc[t])
            ret = np.array(self.constituents_returns.iloc[t])

            vol_condition = (0 if vol[0] <= self.vol_thresholds[0]['risky'] else
                             3 if vol[0] > self.vol_thresholds[2]['risky'] else
                             2 if vol[0] > self.vol_thresholds[1]['risky'] else 1)

            ret_condition = (0 if ret[0] <= self.ret_thresholds[0]['risky'] else
                             3 if ret[0] > self.ret_thresholds[2]['risky'] else
                             2 if ret[0] > self.ret_thresholds[1]['risky'] else 1)

            condition = 4 * ret_condition + vol_condition

            market_conditions.append(condition)

        return market_conditions

    def initialize_theta_bounds_and_conditions(self):
        """Determines theta based on identified conditions and their valid risk profile ranges."""

        # Valid risk profile bounds defined for each market condition based on empirical research
        distinct_conditions = {10: (0.8, 0.91), 14: (0.56, 0.8), 15: (0.4, 0.56),
                               6: (0.27, 0.4), 1: (0.17, 0.27), 9: (0.01, 0.17)}

        # Merges defined based on statistical significancy of market conditions 
        merged_conditions = {0: 6, 2: 6, 3: 6, 4: 9, 5: 9, 7: 6, 8: 9, 11: 6, 12: 9, 13: 1}

        # Calculate market conditions
        market_conditions = self.calculate_market_conditions()

        # Map each timestep to the corresponding or merged market condition
        mapped_conditions = [merged_conditions.get(condition, condition) for condition in market_conditions]

        # Create a mapping to updated indices that can be used in environment
        unique_conditions = sorted(set(mapped_conditions))
        condition_to_index = {condition: i for i, condition in enumerate(unique_conditions)}

        # Map each condition to its index that can be used in environment
        timestep_conditions = [condition_to_index[condition] for condition in mapped_conditions]

        # Create theta bounds for each condition to be used to simulate investor behaviour
        theta_bounds = {condition_to_index[condition]: distinct_conditions.get(condition, (0.01, 0.17)) for condition in condition_to_index}

        return theta_bounds, timestep_conditions

    def get_market_condition(self):
        """Returns market condition based on investor's current portfolio allocation. """
        # Uses empirically defined market conditions if defined
        if self.market_conditions is not None:
            return self.market_conditions[self.current_timestep]

        # Volatility based market condition on portfolio-weighted thresholds
        current_volatilities = np.array(self.constituents_volatility.iloc[self.current_timestep])
        weighted_volatility = np.dot(self.current_portfolio, current_volatilities)
        vol_low_threshold = np.dot(self.current_portfolio, self.vol_thresholds[0])
        vol_medium_threshold = np.dot(self.current_portfolio, self.vol_thresholds[1])
        vol_high_threshold = np.dot(self.current_portfolio, self.vol_thresholds[2])

        vol_condition = (
            0 if weighted_volatility <= vol_low_threshold
            else 3 if weighted_volatility > vol_high_threshold
            else 2 if weighted_volatility > vol_medium_threshold
            else 1
        )
        
        # Returns based market condition on portfolio-weighted thresholds
        current_returns = np.array(self.constituents_returns.iloc[self.current_timestep])
        weighted_returns = np.dot(self.current_portfolio, current_returns)
        ret_low_threshold = np.dot(self.current_portfolio, self.ret_thresholds[0])
        ret_medium_threshold = np.dot(self.current_portfolio, self.ret_thresholds[1])
        ret_high_threshold = np.dot(self.current_portfolio, self.ret_thresholds[2])

        ret_condition = (
            0 if weighted_returns <= ret_low_threshold
            else 3 if weighted_returns > ret_high_threshold
            else 2 if weighted_returns > ret_medium_threshold
            else 1
        )

        # Combine returns and volatility conditions into a market condition
        market_condition = 4 * ret_condition + vol_condition
        
        return market_condition

    def simulate_investor_behaviour(self):
        """Simulates investor risk profile in given market condition."""

        market_condition = self.current_market_condition # Get current market condition

        just_solicited = False # Flags if investor was solicited for first time in timestep

        if self.current_timestep - 1 not in self.solicited_this_step:
            theta_s = self.theta[market_condition] # Assume behavior varies normally around the true risk profile
            sampled_theta = np.random.normal(theta_s, self.r) # Sample about mean theta with standard deviation of r
            self.solicited_this_step[self.current_timestep - 1] = max(min(sampled_theta, self.theta_bounds[market_condition][1]), self.theta_bounds[market_condition][0]) \
                 if self.market_conditions is not None else max(min(sampled_theta, self.max_theta), self.min_theta) # Clip at boundaries of valid theta value
            
            just_solicited = True
            
        return self.solicited_this_step[self.current_timestep - 1], just_solicited

    def calculate_reward(self, ask_investor, just_solicited):
        """Calculates agent reward based on utility function and solicitation penalty."""

        market_condition = self.current_market_condition # Get current market condition
        reward = 0 # Initialise reward

        if ask_investor:
            # Reduce by cost of soliciting K based on portfolio value
            reward -= self.solicitation_penalty

            # If solicited for first time in this timestep update estimate of theta
            if just_solicited:
                # Generate risk profile corresponding to portfolio choice using IPO
                inferred_theta = inverse_MVO_optimisation(self.constituents_returns.iloc[:self.current_timestep, :], self.current_portfolio)
                
                # Apply incremental averaging of theta estimates
                self.n_solicited[market_condition] += 1
                self.current_theta[market_condition] = inferred_theta if self.n_solicited[market_condition] == 1 else \
                    self.current_theta[market_condition] + (inferred_theta - self.current_theta[market_condition]) / self.n_solicited[market_condition]

        # Calculate reward using mean-variance utility function and current estimate of theta
        true_theta = self.theta[market_condition] if self.eval_mode else self.current_theta[market_condition]

        # Start using cached estiamtes after a certain number of episodes
        if self.episode_count > self.caching_threshold:
            if self.current_timestep not in self.optimal_portfolio_cache:
                # Calculate and cache the optimal portfolio using MVO
                optimal_portfolio = MVO_optimisation(self.constituents_returns.iloc[:self.current_timestep, :], true_theta)
                self.optimal_portfolio_cache[self.current_timestep] = optimal_portfolio

            optimal_portfolio = self.optimal_portfolio_cache[self.current_timestep]
        else:
            # Compute optimal portfolio every time before caching starts
            optimal_portfolio = MVO_optimisation(self.constituents_returns.iloc[:self.current_timestep, :], true_theta)

        # Reward is based on difference between current and optimal portfolio
        reward -= np.linalg.norm(np.array(self.current_portfolio) - np.array(optimal_portfolio))

        return reward
        
    def step(self, action):
        """Steps agent through the environment."""

        portfolio_choice = action[:-1] # Portfolio allocation decision
        ask_investor = action[-1] > 0.5 # Asking the investor decision

        self.current_timestep += 1 # Increment current timestep

        just_solicited = False # Used to limit soliciting to once per timestep
        if ask_investor:
            # Simulate investor risk profile and generate corresponding portfolio using MVO
            investor_theta, just_solicited = self.simulate_investor_behaviour()
            portfolio_choice = MVO_optimisation(self.constituents_returns.iloc[:self.current_timestep, :], investor_theta)

        self.current_portfolio = portfolio_choice # Update current portfolio
        
        reward = self.calculate_reward(ask_investor, just_solicited) # Calculate reward
        self.current_market_condition = self.get_market_condition() # Retrieve new market condition
        next_state = self.current_market_condition # Retrieve next state
        terminated = (self.current_timestep >= self.eval_end_step - 1) \
            if self.eval_mode else (self.current_timestep >= self.ending_timestep - 1)    
        truncated = False # Episodes aren't being cut short

        # Provide additional information if evaluating
        if self.eval_mode:
            true_theta = self.theta[self.current_market_condition]
            estimated_theta = inverse_MVO_optimisation(self.constituents_returns.iloc[:self.current_timestep, :], self.current_portfolio)
            info = {'true_theta': true_theta, 'estimated_theta': estimated_theta}
        else:
            info = {}

        self.theta_values.append(self.current_theta.tolist())  # Append theta estimates for evaluation

        return next_state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment to restart after an episode has ended."""

        # Extract arguments
        seed = kwargs.get('seed', None)
        self.eval_mode = kwargs.get('eval_mode', False)

        # Use seed to reproduce investor behaviour simulation
        if seed is not None:
            np.random.seed(seed)

        # Randomly select a new starting point for the next episode
        max_start = self.train_end_step - 200  # Ensure there is room for 200 timesteps
        self.initial_timestep = self.train_end_step + 1 if self.eval_mode else np.random.randint(2745, max_start + 1)
        self.ending_timestep = self.initial_timestep + 200
        self.current_timestep = self.initial_timestep
        self.current_portfolio = np.full((self.n_assets,), 1/self.n_assets) # Reset to equally weighted portfolio
        self.current_market_condition = self.get_market_condition()
        self.theta_values = []  # Reset theta values for the next episode
        self.episode_count += 1  # Increment episode count on each reset
        info = {}

        # Write theta values to file at the end of an episode for evaluation
        if self.theta_values:
            theta_values_path = "project/data/theta_values.json"
            # Check if the file exists and is not empty
            if not os.path.exists(theta_values_path) or os.path.getsize(theta_values_path) == 0:
                with open(theta_values_path, 'w') as f:
                    json.dump(self.theta_values, f)
                print(f"theta values saved to {theta_values_path}")

        return self.current_market_condition, info

    def render(self, mode='human'):
        """Renders environment relevant information at timestep."""

        if mode == 'human':
            print(f"Current timestep: {self.current_timestep}, Current portfolio: {self.current_portfolio}, Market condition: {self.current_market_condition}")
