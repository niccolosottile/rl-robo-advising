import cvxpy as cp

# IMPORTANT
# To keep vectorised process, the model currently assumes that asset-level expected returns c are constant among timesteps.

def solve_iporisk(portfolio_allocations, constituents_returns, Q, A, b, c, rt, M=10**3, eta_t=1):
    """
    Solves the IPO-Risk optimization problem to learn time-varying risk tolerance r.

    Parameters:
    - portfolio_allocations: DataFrame with portfolio allocations.
    - constituents_returns: DataFrame with constituents returns.
    - Q: Covariance matrix of asset returns.
    - A, b: Constraints for the portfolio optimization problem.
    - c: Known expected returns from IPO-Return solution.
    - rt: Current estimates of risk tolerance.
    - M: A large number for the mixed-integer programming constraint.
    - eta_t: Regularization parameter.

    Returns:
    - r: Learned risk tolerances.
    """
    n_assets = constituents_returns.shape[1]
    n_time_steps = portfolio_allocations.shape[0]

    # Variables
    r = cp.Variable(n_time_steps)
    x = cp.Variable((n_assets, n_time_steps))
    u = cp.Variable((n_assets, n_time_steps))
    z = cp.Variable((n_assets, n_time_steps), boolean=True)

    # Objective
    objective = cp.Minimize(0.5 * cp.sum_squares(r - rt) + eta_t * cp.sum_squares(portfolio_allocations.values.T - x))

    # Constraints
    constraints = [
        A @ x >= b,
        u <= M * z,
        A @ x - b <= M * (1 - z),
        Q.T @ x - cp.multiply(r, c) - A.T @ u == 0,
        x >= 0,
        u >= 0,
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return r.value

def solve_iporeturn(portfolio_allocations, constituents_returns, Q, A, b, r, ct, M=10**3, eta_t=1):
    """
    Solves the IPO-Return optimization problem to learn time-varying expected returns c.

    Parameters:
    - portfolio_allocations: DataFrame with portfolio allocations.
    - constituents_returns: DataFrame with constituents returns.
    - Q: Covariance matrix of asset returns.
    - A, b: Constraints for the portfolio optimization problem.
    - r: Fixed risk tolerance factor from IPO-Risk solution.
    - ct: Current estimates of expected returns.
    - M: A large number for the mixed-integer programming constraint.
    - eta_t: Regularization parameter.

    Returns:
    - c: Learned expected returns.
    """
    n_assets = constituents_returns.shape[1]
    n_time_steps = portfolio_allocations.shape[0]

    # Variables
    c = cp.Variable((n_time_steps, n_assets))
    x = cp.Variable((n_assets, n_time_steps))
    u = cp.Variable((n_assets, n_time_steps))
    z = cp.Variable((n_assets, n_time_steps), boolean=True)

    # Objective
    objective = cp.Minimize(0.5 * cp.sum_squares(c - ct) + eta_t * cp.sum_squares(portfolio_allocations.values.T - x))

    # Constraints
    constraints = [
        A @ x >= b,
        u <= M * z,
        A @ x - b <= M * (1 - z),
        Q.T @ x - cp.multiply(r, c) - A.T @ u == 0,
        x >= 0,
        u >= 0,
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return c.value