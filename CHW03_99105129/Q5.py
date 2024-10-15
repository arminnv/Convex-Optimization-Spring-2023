import cvxpy as cp
from multi_risk_portfolio_data import *
w = cp.Variable(n)
t = cp.Variable()
risks = [cp.quad_form(w, S) for S in [Sigma_1, Sigma_2, Sigma_3, Sigma_4, Sigma_5, Sigma_6]]
objective = w.T @ mu - gamma * t
cons1 = [cp.sum(w) == 1]
constraints = [risk <= t for risk in risks]
prob = cp.Problem(cp.Maximize(objective), constraints=constraints + cons1).solve()

print("weights:")
print(w.value)

print("gamma")
print(x.dual_value for x in risks)

print("risks")
print(x.value for x in risks)

