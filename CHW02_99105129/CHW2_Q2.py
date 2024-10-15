import numpy as np
import cvxpy as cp

A = np.matrix([[1, 2, 0, 1],
               [0, 0, 3, 1],
               [0, 3, 1, 1],
               [2, 1, 2, 5],
               [1, 0, 3, 2]])
c_max = np.array([100, 100, 100, 100, 100])
p = np.array([3, 2, 7, 6])
q = np.array([4, 10, 5, 10])
p_disc = np.array([2, 1, 4, 2])

x = cp.Variable(4)

objective = cp.Maximize(cp.sum(cp.minimum(cp.multiply(p, x), cp.multiply(p, q) + cp.multiply(p_disc, x - q))))
constraints = [x >= 0, A @ x <= c_max]
prob = cp.Problem(objective, constraints)
result = prob.solve()
r = cp.minimum(cp.multiply(p, x), cp.multiply(p, q) + cp.multiply(p_disc, x - q)).value

sum_r = sum(r)
r_average = r / x.value
print("optimal levels = ", x.value)
print("revenue = ", r)
print("total revenue = ",sum_r)
print("average = ", r_average)
