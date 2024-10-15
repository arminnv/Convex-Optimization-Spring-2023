import numpy as np

n = 5
X = np.random.rand(n, n)

# A is a symmetric square random matrix of size n
A = X.T @ X
print(f"A = {A}")


def francis(A):
    Q, R = np.linalg.qr(A)
    V = Q.copy()
    for i in range(1000):
        Q, R = np.linalg.qr(R @ Q)
        V = V @ Q
        D = V.T @ A @ V


    eigen_values = np.array([D[i, i] for i in range(len(D))])
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = np.array(V[:, idx])
    return eigen_values, eigen_vectors


eigen_values, eigen_vectors = francis(A)
print(f"\neigen vectors: {eigen_vectors}")
print(f"\neigen values: {eigen_values}")
print(f"\nerror = ||A - V @ D @ V^T|| = {np.linalg.norm(A - eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T)}")


if __name__=="__main__":
    a = 0