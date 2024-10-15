import numpy.linalg as la
import numpy as np
import pandas
import matplotlib.pyplot as plt
from numpy import mean

path = "iris.csv"
df = pandas.read_csv(path)
columns = df.columns[1:-1]

X = np.matrix([df.loc[:, x] - mean(df.loc[:, x]) for x in columns]).T
y = df.loc[:, "species"]

cov = np.cov(X.T)
eig_values, eig_vectors = la.eig(cov)
PC = eig_vectors[:2]


U = X @ PC.T
u1, u2 = np.array(U[:, 0]), np.array(U[:, 1])

# use colormap
colors = {"setosa": "red", "versicolor": "blue", "virginica": "green"}
colormap = [colors[k] for k in y]
# depict illustration
plt.scatter(u1, u2, s=40, c=colormap)
plt.xlabel("u1")
plt.ylabel("u2")
plt.title("principal component analysis")
plt.savefig("principal component analysis")
plt.show()

if __name__ == "__main__":
    None