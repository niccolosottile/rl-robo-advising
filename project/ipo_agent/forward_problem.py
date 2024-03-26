import cvxpy as cp
import numpy as np

def forward_problem(constituents_returns, r, only_last=False):
    """Forward problem that estimates optimal portfolio allocation given risk profile."""
    n_assets = constituents_returns.shape[1]
    n_time_steps = constituents_returns.shape[0]
    A = np.ones((1, n_assets))  # Linear constraints since portfolio weights need to sum to 1
    b = np.array([1]) # Bounds for linear constraints
    portfolio_allocations = []
    offset = 0 if not only_last else n_time_steps - 1 # Don't process all timesteps

    # Generate optimal allocation for each timestep
    for t in range(1 + offset, n_time_steps):
        # Calculate return mean up to current time t for each asset
        c_t = []
        for i in range(n_assets):
            c_i_t = constituents_returns.values[:t+1, i].mean()
            c_t.append(c_i_t)
        c_t = np.array(c_t)

        Q_t = np.cov(constituents_returns.values[:t+1].T) # Measures covariance between assets
        
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