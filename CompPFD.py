import numpy as np
import matplotlib.pyplot as plt
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

tau0 = 9192.57
eta = 0.8922

# Find PFD_avg
def simulate_intervals(tau_list):
    n = len(tau_list)
    s0, s2, s22 = 1, 0, 2
    P = ML.getP0(n+o, s2)
    A = np.zeros([n+o, n+o])
    A[s2, s0] = lamf

    PFD_per_interval = []

    for i in range(n):
        P[s22] += P[s0]
        P[s0] = 0  # Maintenance
        A[s2, o+i] = ML.lmbda(gamma, b0_2, alpha, i)
        dT = tau_list[i] / K
        if i > 0:
            A[s2, o+i-1] = 0
        for j in range(i+1):
            A[o+j, s0] = ML.lmbda(gamma, b0_1, alpha, i-j) + lamf
        ML.fixA(A)
        IM = np.eye(n+o) + np.dot(A, dT)

        PFD_interval = 0
        for k in range(K):
            P = np.dot(P, IM)
            PFD_interval += P[s0]

        PFD_per_interval.append(PFD_interval / K)

    return PFD_per_interval

# Decreasing interval schedule
n_decreasing = ML.compute_n(tau0, eta, T)
tau_list_decreasing = []
tau = tau0
for i in range(n_decreasing):
    tau_list_decreasing.append(tau)
    tau *= eta

# Constant interval schedule
n_constant = n_decreasing
tau_const = T / (n_constant + 1)
tau_list_constant = [tau_const] * n_constant

# Simulate
PFD_decreasing = simulate_intervals(tau_list_decreasing)
PFD_constant = simulate_intervals(tau_list_constant)

# Plot
plt.figure(figsize=(10,4.5))  # Horizontal aspect ratio for squareness
plt.plot(np.arange(1, n_decreasing+1), PFD_decreasing, '-', label=r'Varying $\tau$', linewidth=3)
plt.plot(np.arange(1, n_constant+1), PFD_constant, '-', label=r'Constant $\tau$', linewidth=3)
plt.xlabel(r'$N$', fontsize=14)
plt.ylabel(r'PFD', fontsize=14)
plt.xlim(1, n_decreasing)
plt.ylim(0.000, 0.006)
plt.grid(True)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()
