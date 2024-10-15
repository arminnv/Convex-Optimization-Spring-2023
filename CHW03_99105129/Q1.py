import cvxpy as cp
import matplotlib.pyplot as plt
from opt_evac_data import *


n,m = A.shape
f = cp.Variable((m, T-1))
q = cp.Variable((n, T))

node_risk = q.T @ r + cp.square(q).T @ s
edge_risk = cp.abs(f).T @ rtild + cp.square(f).T @ stild

elist = [None]*T
for i in range(T-1):
    elist[i] = edge_risk[i]
elist[-1] = 0
edge_risk = cp.vstack(elist)[:, 0]

risk = node_risk + edge_risk
cons = [q[:,0] == q1, q[:,1:] == A @ f + q[:,:-1], 0 <= q, q <= np.tile(Q,(T,1)).T,cp.abs(f) <= np.tile(F,(T-1,1)).T]
p = cp.Problem(cp.Minimize(sum(risk)), cons).solve(verbose=True, solver=cp.ECOS)

q, f, risk, node_risk = map(lambda x: np.array(x.value), (q, f, risk, node_risk))

print("total risk: ", p)
print("evacuation time:", (node_risk <= 1e-4).nonzero()[0][0] + 1)

fig, axs = plt.subplots(3,1)
axs[0].set_ylabel('Rt')
axs[0].plot(np.arange(1,T+1), risk)
axs[1].set_ylabel('qt')
axs[1].plot(np.arange(1,T+1), q.T)
axs[2].set_ylabel('ft')
axs[2].plot(np.arange(1,T), f.T)
plt.savefig("optimum evacuation")
plt.show()
