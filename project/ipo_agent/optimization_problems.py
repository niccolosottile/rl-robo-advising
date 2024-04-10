import cvxpy as cp

def solve_iporeturn(portfolio_allocations, Q, A, b, c_t, r_t, M=10**3, eta_t=1, verbose = False):
    """
    Solves the IPO-Return optimization problem to learn time-varying expected returns c.

    Parameters:
    - portfolio_allocations: Array with current portfolio allocations.
    - constituents_returns: Array with constituents returns.
    - Q: Covariance matrix of asset returns.
    - A, b: Constraints for the portfolio optimization problem.
    - r_t: Fixed risk tolerance factor from IPO-Risk solution.
    - ct: Current estimates of asset-level expected returns.
    - M: A large number for the mixed-integer programming constraint.
    - eta_t: Regularization parameter.

    Returns:
    - c: Learned asset-level expected returns.
    """
    n_assets = c_t.shape

    # Variables
    c = cp.Variable(n_assets)
    x = cp.Variable(n_assets, nonneg=True) # Avoids short positions
    u = cp.Variable(1, nonneg=True) # Needed for inequality
    z = cp.Variable(1, boolean=True)

    # Objective
    objective = cp.Minimize(0.5 * cp.sum_squares(c - c_t) + eta_t * cp.sum_squares(portfolio_allocations.T - x))

    # Constraints
    constraints = [
        A @ x >= b,
        u <= M * z,
        A @ x - b <= M * (1 - z),
        Q @ x - r_t * c - A.T @ u == 0,
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    if verbose:
        print("IPO-Return", c.value, r_t, x.value)

    return c.value

def solve_iporisk(portfolio_allocations, Q, A, b, c_t, r_t, M=10**3, eta_t=1, verbose = False):
    """
    Solves the IPO-Risk optimization problem to learn time-varying risk tolerance r.

    Parameters:
    - portfolio_allocations: Array with current portfolio allocations.
    - constituents_returns: Array with constituents returns.
    - Q: Covariance matrix of asset returns.
    - A, b: Constraints for the portfolio optimization problem.
    - c: Known expected returns from IPO-Return solution.
    - rt: Current estimate of risk tolerance.
    - M: A large number for the mixed-integer programming constraint.
    - eta_t: Regularization parameter.

    Returns:
    - r: Learned risk tolerance.
    """
    n_assets = c_t.shape[0]

    # Variables
    r = cp.Variable()
    x = cp.Variable(n_assets, nonneg=True) # Avoids short positions
    u = cp.Variable(1, nonneg=True) # Needed for inequality
    z = cp.Variable(1, boolean=True)
 
    # Objective
    objective = cp.Minimize(0.5 * cp.sum_squares(r - r_t) + eta_t * cp.sum_squares(portfolio_allocations.T - x))

    # Constraints
    constraints = [
        A @ x >= b,
        u <= M * z,
        A @ x - b <= M * (1 - z),
        Q @ x - r * c_t - A.T @ u == 0,
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    if verbose:
        print("IPO-Risk", c_t, r.value, x.value)

    return r.value