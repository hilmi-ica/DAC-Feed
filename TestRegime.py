import numpy as np
from scipy.optimize import minimize
import MarkovLib as ML

# Constants
o = 3
alpha = 0.3
gamma = 7e-5
lamf = 4e-7
b0_2 = 100
b0_1 = 75
T = 73000
K = 1000

# Find PFD_avg
def compute_PFD_avg(tau_init, eta):
    if tau_init <= 0 or eta <= 0 or eta > 1:
        return 1e6  # heavy penalty for invalid regions

    n = ML.compute_n(tau_init, eta, T)
    if n > 100:
        return 1e6  # heavy penalty if n > 100

    s0, s2, s22 = 1, 0, 2
    tau = tau_init
    P = ML.getP0(n+o, s2)
    PFD = 0
    A = np.zeros([n+o, n+o])

    A[s2, s0] = lamf

    for i in range(n):
        P[s22] += P[s0]
        P[s0] = 0  # Maintenance
        A[s2, o+i] = ML.lmbda(gamma, b0_2, alpha, i)
        dT = tau / K
        if i > 0:
            A[s2, o+i-1] = 0
            tau = eta * tau
        for j in range(i+1):
            A[o+j, s0] = ML.lmbda(gamma, b0_1, alpha, i-j) + lamf
        ML.fixA(A)
        IM = np.eye(n+o) + np.dot(A, dT)
        for k in range(K):
            P = np.dot(P, IM)
            PFD += P[s0]

    PFD_avg = PFD / (K * n)
    return PFD_avg

# Objective function
def objective(x):
    tau_init, eta = x
    return compute_PFD_avg(tau_init, eta)

# Bounds: tau between 1000 and 20000 hours, eta between 0.5 and 1.0
bounds = [(5000, 20000), (0.5, 1.0)] # Relaxed for demonstration purposes

# Initial guess
x0 = [12000, 0.9]

# Optimization
result = minimize(objective, x0, bounds=bounds, method='Powell')

# Print result
print(f"Optimal tau: {result.x[0]:.2f} hours")
print(f"Optimal eta: {result.x[1]:.4f}")
