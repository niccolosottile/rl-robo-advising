import cvxpy as cp
import numpy as np

def MVO_optimisation(constituents_returns, r):
    n_assets = constituents_returns.shape[1]
    n_time_steps = constituents_returns.values.shape[0]

    A_upper = np.ones((1, n_assets))   # Enforces that the sum of weights is at least 1
    A_lower = -np.ones((1, n_assets))  # Enforces that the sum of weights does not exceed 1
    A = np.vstack((A_upper, A_lower))
    A = np.vstack([A, np.eye(n_assets)]) # Add non-negativity constraints for each portfolio weight

    b = np.array([0.999, -1]) # Bounds for linear constraints
    b = np.hstack([b, np.zeros(n_assets)]) # Bounds for non-negativity constraints

    # Calculate return mean and covariance within the lookback window
    c = constituents_returns.values[:n_time_steps].mean(axis=0)
    Q = np.cov(constituents_returns.values[:n_time_steps].T)

    # Variable for optimal portfolio x
    x = cp.Variable(n_assets)

    # Objective function
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) - r * (c.T @ x))

    # Constraints
    constraints = [A @ x >= b]

    # Problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(solver=cp.MOSEK)

    return x.value.tolist()