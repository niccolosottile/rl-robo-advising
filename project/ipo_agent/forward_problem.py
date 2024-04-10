import cvxpy as cp
import numpy as np

def forward_problem(constituents_returns, r, only_last=False):
    """Forward problem that estimates optimal portfolio allocation given risk profile."""
    n_assets = constituents_returns.shape[1]
    n_time_steps = constituents_returns.shape[0]
    A = np.ones((1, n_assets))  # Linear constraints since portfolio weights need to sum to 1
    b = np.array([1]) # Bounds for linear constraints
    portfolio_allocations = []
    L = 1200 # Lookback window
    offset = 0 if not only_last else n_time_steps - 2 # Don't process all timesteps

    # Generate optimal allocation for each timestep
    for t in range(1 + offset, n_time_steps):
        # Adjust indeces to use lookback window L for calculating mean returns and covariance (+1 for slicing)
        start_index = max(0, t + 1 - L)
        end_index = t + 1
        
        # Calculate return mean and covariance within the lookback window
        c_t = constituents_returns.values[start_index:end_index].mean(axis=0)
        Q_t = np.cov(constituents_returns.values[start_index:end_index].T)

        #if t >= 1275:
            #print("Forward uses:")
            #print(Q_t)
            #print(c_t)
            #print("True risk: ", r)
        
        # Variables
        x_t = cp.Variable(n_assets, nonneg=True)
        
        # Objective
        objective = cp.Minimize(0.5 * cp.quad_form(x_t, Q_t) - r * cp.matmul(c_t.T, x_t))
        
        # Constraints
        constraints = [
            A @ x_t == b
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Store the optimized portfolio allocation for this timestep
        portfolio_allocations.append(x_t.value)

    return portfolio_allocations