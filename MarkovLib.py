import numpy as np
import math

# Failure rate as function of n
def lmbda(gamma, b_0, alpha, n):
    b = b_0 * math.exp(-alpha * n)
    return gamma * (1 - b / (b + 1))

# Get initial P-vector
def getP0(N, init):
    P = np.zeros([N])
    P[init] = 1
    return P

# Fill in the diagonal elements of a transition matrix
def fixA(A):
    N = len(A)
    for i in range(N):
        s = 0
        for j in range(N):
            if i != j:
                s += A[i][j]
        A[i][i] = -s

# Compute N from tau and eta
def compute_n(tau, eta, T):
    if eta == 1.0:
        return int(np.ceil(T / tau))
    s = (T * (1 - eta)) / tau
    if s >= 1:
        return int(200)
    n = np.log(1 - s) / np.log(eta) - 1
    return int(np.ceil(n))