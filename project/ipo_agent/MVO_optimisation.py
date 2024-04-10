import cvxpy as cp
import numpy as np

A = np.ones((1, 2))  # Linear constraints since portfolio weights need to sum to 1
b = np.array([1]) # Bounds for linear constraints

# Variables obtained through inverse optimisation
c_t = np.array([0.15633644, 0.0384881])
r = 0.0007462636492241547
Q_t = np.array([[0.00042326, 0.00018577],
 [0.00018577, 0.00013334]])

# Variables
x_t = cp.Variable(2, nonneg=True)

# Objective
objective = cp.Minimize(0.5 * cp.quad_form(x_t, Q_t) - r * cp.matmul(c_t.T, x_t))

# Constraints
constraints = [
    A @ x_t == b
]

# Solve
problem = cp.Problem(objective, constraints)
problem.solve()

print(x_t.value)