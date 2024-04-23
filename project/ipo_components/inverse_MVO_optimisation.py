from xmlrpc.client import boolean
import cvxpy as cp
import numpy as np

def inverse_MVO_optimisation(constituents_returns, portfolio_allocations):
    # Initialization
    n_assets = constituents_returns.shape[1]
    n_time_steps = constituents_returns.values.shape[0] # Assuming constituents equally indexed by time

    A_upper = np.ones((1, n_assets))   # Enforces that the sum of weights is at least 1
    A_lower = -np.ones((1, n_assets))  # Enforces that the sum of weights does not exceed 1
    A = np.vstack((A_upper, A_lower))
    A = np.vstack([A, np.eye(n_assets)]) # Add non-negativity constraints for each portfolio weight

    b = np.array([0.999, -1]) # Bounds for sum of portfolio weights
    b = np.hstack([b, np.zeros(n_assets)]) # Bounds for non-negativity constraints

    #L = 1200 # Lookback window
    M = 1 # Number used to bound u and A @ x - b

    # Adjust indeces to use lookback window L for calculating mean returns and covariance
    start_index = 0 #max(0, n_time_steps - L)
    end_index = n_time_steps

    # Calculate return mean and covariance within the lookback window
    c = constituents_returns.values[start_index:end_index].mean(axis=0)
    Q = np.cov(constituents_returns.values[start_index:end_index].T)

    # Observed optimal portfolio
    x_star = portfolio_allocations 

    # Variables
    r = cp.Variable()
    x = cp.Variable(n_assets)
    u = cp.Variable(A.shape[0], nonneg=True) # Variable u being negative would allow violation of constraints
    z = cp.Variable(A.shape[0], boolean=True) # Binary z values used to activate constraints
 
    # Objective
    objective = cp.Minimize(cp.sum_squares(x_star - x))

    # Constraints
    constraints = [
        A @ x >= b,
        u <= M * z,
        A @ x - b <= M * (1 - z),
        Q @ x - r * c - A.T @ u == 0,
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    return r.value.tolist()
