import cvxpy as cp
x = cp.Variable(1)
y = cp.Variable(1)
z = cp.Variable(1)
print(x.shape)

cons = [None]*4
cons[0] = [cp.inv_pos(x) + cp.inv_pos(y) <= 1, x >= 0, y >= 0]
cons[1] = [y >= cp.inv_pos(x), x >= 0, y >= 0]
cons[2] = [cp.quad_over_lin(x + y, cp.sqrt(y)) <= x - y + 5]
cons[3] = [x + z <= 1 + cp.geo_mean(cp.vstack([x - cp.quad_over_lin(z,y), y])), x >= 0, y >= 0]

for constraint in cons:
    prob = cp.Problem(cp.Maximize(1),constraint).solve()