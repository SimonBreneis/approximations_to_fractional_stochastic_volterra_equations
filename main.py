import time
import numpy as np
import matplotlib.pyplot as plt
import Data
import ComputationalFinance as cf
import rBergomiBFG
import rBergomiAK
import rHeston
import rHestonAK
import RoughKernel as rk
import rHestonNinomiyaVictoir as nv
import mpmath as mp

A = -mp.matrix([[1.2, 0.1, 0.3, 0.2], [0.2, 2.1, 0.3, 0.2], [0.2, 0.1, 3.3, 0.2], [0.2, 0.1, 0.3, 4.2]])
print(mp.expm(A))
print(nv.exp_matrix(mp.eig(A), 1.))

print("--------------------------------------------------------")

b = mp.matrix([1., 2.5, -2., 0.2])
dt = 0.1
print(nv.ODE_drift(mp.eig(A), b, dt/2))  # A^{-1} (exp(A dt/2) - Id) b
print(mp.inverse(A) * (mp.expm(A * dt/2) - mp.eye(4)) * b)

print("--------------------------------------------------------")

weights = mp.matrix([0.2, 0.5, 1., 2.])
print(nv.ODE_S_drift(mp.eig(A), b, dt/2, weights.T))  # w/2 cdot (A^{-2} exp(A dt/2) b  - A^{-2} b - A^{-1} b dt/2)
print(mp.fdot(weights/2, mp.inverse(A) * (mp.inverse(A) * mp.expm(A * dt/2) * b - mp.inverse(A) * b - b * dt/2)))

print("--------------------------------------------------------")

print(nv.ODE_S_mult(mp.eig(A), dt/2, weights.T))  # w/2 cdot A^{-1} exp(A dt/2)
print((mp.inverse(A) * mp.expm(A * dt/2)).T * weights/2)

'''
Diagonalization tests
nodes = mp.matrix([1., 2., 3., 4.])
weights = mp.matrix([0.2, 0.1, 0.3, 0.2])
A = -mp.matrix([[1.2, 0.1, 0.3, 0.2], [0.2, 2.1, 0.3, 0.2], [0.2, 0.1, 3.3, 0.2], [0.2, 0.1, 0.3, 4.2]])
D, U = mp.eig(A)
for i in range(4):
    U[:, i] = U[:, i]/mp.norm(U[:, i])
eigvalues, eigvectors = nv.diagonalize(nodes, weights, 1.)

A_1 = mp.zeros(4)
for i in range(4):
    A_1[:, i] = mp.lu_solve(U.T, ((U*mp.diag(D)).T)[:, i])
A_1 = A_1.T
# print(A_1)
A_2 = mp.zeros(4)
for i in range(4):
    A_2[:, i] = mp.lu_solve(eigvectors.T, ((eigvectors*mp.diag(eigvalues)).T)[:, i])
A_2 = A_2.T
# print(A_2)

# for i in range(4):
#     print(A * eigvectors[:, i] - eigvalues[i] * eigvectors[:, i])


print(mp.norm(A - A_1))
print(mp.norm(A - A_2))

N = 50
rule = rk.quadrature_rule_geometric_good(0.1, N-1)
nodes = rule[0, :]
weights = rule[1, :]
N = len(nodes)
print(N)
print(mp.eps)
A = -mp.diag(nodes) - mp.matrix([[w for w in weights] for _ in range(N)])
D, U = mp.eig(A)
print(D)
for i in range(N):
    U[:, i] = U[:, i]/mp.norm(U[:, i])
# eigvalues, eigvectors = nv.diagonalize(nodes, weights, 1.)

A_1 = mp.zeros(N)
for i in range(N):
    A_1[:, i] = mp.lu_solve(U.T, ((U*mp.diag(D)).T)[:, i])
A_1 = A_1.T
# print(A_1)

print(mp.norm(A - A_1))


time.sleep(3600)
'''


S, V, V_comp = nv.get_sample_path(0.1, 0.3, -0.7, 0.3, 0.02, 0.02, 1., 1)
plt.plot(np.linspace(0, 1, 1001), S, label="stock price")
plt.plot(np.linspace(0, 1, 1001), V, label="volatility")
plt.plot(np.linspace(0, 1, 1001), V_comp[-1, :], label="low mean reversion")
plt.plot(np.linspace(0, 1, 1001), V_comp[-2, :], label="high mean reversion")
plt.legend(loc="upper left")
plt.show()
time.sleep(3600)

print("Hello World!")
Data.rHeston_smiles_precise()
Data.plot_rBergomi_smiles()
time.sleep(3600)


K = np.exp(-1.3 + 0.01 * np.arange(161))
print("True rough Heston:")
tic = time.perf_counter()
true_heston = rHeston.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=0.01,
                                         N_Riccati=3000, N_fourier=10000, L=50.)
toc = time.perf_counter()
print(true_heston)
print(f"Generating the true smile took {toc - tic} seconds.")

for N in [1, 2, 3, 4, 5, 6, 7, 8]:
    print(f"Approximation with {N} nodes, our scheme:")
    tic = time.perf_counter()
    approximated_heston = rHestonAK.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02,
                                                       T=0.01, N=N, N_Riccati=3000, N_fourier=10000, L=50.)
    toc = time.perf_counter()
    print(approximated_heston)
    print(f"Generating the approximated smile with N={N} took {toc - tic} seconds.")
