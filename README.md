# Reinforcement Learning Framework for Portfolio Optimization in Robo-Advising

This repository contains the implementation of a novel Reinforcement Learning (RL) framework aimed at optimizing portfolio strategies within the context of Robo-Advising. The project utilizes various methods including Inverse Portfolio Optimization (IPO) and Proximal Policy Optimization (PPO) to predict near-optimal investment strategies tailored to individual investor risk profiles.

## Project Structure

- `/data`: Contains datasets used in the models.
  - `VTI.csv`: Dataset for Vanguard Total Stock Market ETF.
  - `TNX.csv`: Dataset for CBOE 10-Year Treasury Note Yield.
- `/envs`: Environment setups for portfolio optimization.
  - `portfolioenv.py`: Defines the Reinforcement Learning (RL) environment class.
- `/ipo_components`: Components for Inverse Portfolio Optimization (IPO) used within environment.
  - `MVO_optimisation.py`: Mean-Variance Optimization (MVO) component.
  - `inverse_MVO_optimisation.py`: Inverse MVO component.
  - `forward_inverse_validation.py`: Validates IPO using Forward-Inverse Validation.
  - `risk_profiles_search_conditions.py`: Searches for valid risk profiles under various market conditions.
  - `risk_profiles_search.py`: Searches for risk profiles for each timestep.
  - `visualise_conditions.py`: Visualises nearest neighbour clustering results for market conditions.
- `/models`: Contains the RL models.
  - `drl_model.py`: Deep RL model using PPO with a custom Actor-Critic architecture.
- `/utils`: Data preprocessing functions and the utility functions supporting the model.

## Requirements

- Python 3.8+
- Libraries: numpy, pandas, seaborn, matplotlib, networkx, cvxpy, torch, stable_baselines3, gymnasium

To install necessary libraries, run:

pip install -r requirements.txt
## Usage

To run the main model (e.g., training the agent), execute from the root project folder:

python -m project.models.drl_model

Other scripts can be run similarly by adjusting the path to the desired Python script
## Dataset

The datasets `VTI.csv` and `TNX.csv` are utilized for training the models with over 20 years of historical data on stock and bond markets respectively.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Author

- Niccolò Sottile - Bachelor of Science in Computer Science, University of Bath, 2023-2024