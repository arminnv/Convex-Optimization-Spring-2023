import cvxpy as cp
from blend_design_data import *
tta = cp.Variable(k)
cons = [np.log(P)@tta <= np.log(P_spec)] + [np.log(D)@tta <= np.log(D_spec)]\
    + [np.log(A)@tta <= np.log(A_spec)] + [cp.sum(tta)==1, tta>=0]

cp.Problem(cp.Minimize(0), cons).solve()

w = np.exp(np.log(W) @ tta.value)
print(w)
print(tta.value)

