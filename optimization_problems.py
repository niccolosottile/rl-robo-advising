import cvxpy as cp

# Might need to add constraint to ensure that num_assets dimension values in r matrix are equal.

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
    r = cp.Variable((n_assets, n_time_steps))
    x = cp.Variable((n_assets, n_time_steps))
    u = cp.Variable((1, n_time_steps))
    z = cp.Variable((1, n_time_steps), boolean=True)

    # Explaining variable choices (see respective formula in paper)
    # A is (1, n_assets), x is (n_assets, n_time_steps), hence, (1, n_time_steps) comparison with b which is broadcasted works.
    # u and z need to have the same dimensions for comparison.
    # z needs to be (1, n_time_steps) for comparison with A @ x - b which is (1, n_time_steps).
    # A transposed is (n_assets, 1) with u which needs to be (1, n_time_steps) results in (n_assets, n_time_steps).
    # Q is (n_timesteps, (n_assets, n_assets)) with x which is (n_assets, n_time_steps) results in (n_assets, n_time_steps).
    # r is (n_assets, n_time_steps) and c is (n_assets, n_time_steps) should result in (n_assets, n_time_steps) by element-wise multiplication,
    # each r value at timestep t is multipled with each asset level c value.
 
    # Objective
    objective = cp.Minimize(0.5 * cp.sum_squares(r - rt) + eta_t * cp.sum_squares(portfolio_allocations.values.T - x))

    # Constraints
    constraints = [
        A @ x >= b,
        u <= M * z,
        A @ x - b <= M * (1 - z),
        Q @ x - cp.multiply(r, c) - A.T @ u == 0,
        x >= 0,
        u >= 0,
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

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
    c = cp.Variable((n_assets, n_time_steps))
    x = cp.Variable((n_assets, n_time_steps))
    u = cp.Variable((1, n_time_steps))
    z = cp.Variable((1, n_time_steps), boolean=True)

    # Objective
    objective = cp.Minimize(0.5 * cp.sum_squares(c - ct) + eta_t * cp.sum_squares(portfolio_allocations.values.T - x))

    # Constraints
    constraints = [
        A @ x >= b,
        u <= M * z,
        A @ x - b <= M * (1 - z),
        #Q @ x - cp.multiply(r, c) - A.T @ u == 0,
        x >= 0,
        u >= 0,
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return c.value