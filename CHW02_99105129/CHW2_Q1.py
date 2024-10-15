import cvxpy as cp
import numpy as np

# Problem data.
from matplotlib import pyplot as plt

m = 30
N = [0, 4, 2, 2, 3, 0, 4, 5, 6, 6, 4, 1, 4, 4, 0, 1, 3, 4, 2, 0, 3, 2, 0, 1]
n = len(N)
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
#objective = cp.Minimize(cp.sum_squares(A @ x - b))

D = np.eye(n)[1:,] - np.eye(n)[:-1,]
D = D.T @ D

RO = [0.1, 1, 10, 100]
T = np.linspace(1, 24, 24, dtype=int)
values = []

for ro in RO:
    objective = cp.Maximize(sum(-x) + N @ cp.log(x) + x.T @ D @ x * (-ro))
    constraints = [0 <= x]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    print(f"ro = {ro}:")
    print("Lamda t = ", x.value)
    values.append(x.value)

    plt.plot(T, x.value, label="ro = " + str(ro))
    plt.title("lambda(t)")
    plt.xlabel("t")
    plt.ylabel("lambda")
    plt.legend()
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    #print(constraints[0].dual_value)

plt.savefig("lambda_t.png")
plt.show()
plt.close()

N_test = (0, 1, 3, 2, 3, 1, 4, 5, 3, 1, 4, 3, 5, 5, 2, 1, 1, 1, 2, 0, 1, 2, 1, 0)

ro_best = 0
max = -np.inf
for i in range(len(RO)):
    v = values[i]
    log_likelihood = (sum(-v) + N_test @ cp.log(v)).value
    if max < log_likelihood:
        max = log_likelihood
        ro_best = RO[i]
    print(f"ro = {RO[i]}: log likelihood =", log_likelihood)

print("\nro best = ", ro_best)