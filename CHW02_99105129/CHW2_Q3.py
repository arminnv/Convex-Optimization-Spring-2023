import cvxpy as cp
import matplotlib.pyplot as plt
from veh_speed_sched_data import *


t = cp.Variable(n)

objective = cp.Minimize(cp.sum(a * cp.multiply(cp.square(d), cp.inv_pos(t)) + b*d + c*t))
constraints = [t <= d / smin, t >= d / smax]
constraints += [tau_min[i] <= cp.sum(t[0:i+1]) for i in range(n)]
constraints += [tau_max[i] >= cp.sum(t[0:i+1]) for i in range(n)]

prob = cp.Problem(objective, constraints)
result = prob.solve()
s = d / t.value
print("optimal fuel consumption = ", objective.value)
plt.step(np.arange(n), s)
plt.title("speed per segment")
plt.xlabel("ith segment")
plt.ylabel("speed")
plt.savefig("speed per segment")
plt.show()

