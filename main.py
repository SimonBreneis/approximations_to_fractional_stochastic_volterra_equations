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


K = np.exp(-1.3 + 0.01 * np.arange(161))
print("True rough Heston:")
tic = time.perf_counter()
true_heston = rHeston.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50.)
toc = time.perf_counter()
print(true_heston)
print(f"Generating the true smile took {toc-tic} seconds.")

for N in [1, 2, 3, 4, 5, 6, 7, 8]:
    print(f"Approximation with {N} nodes, our scheme:")
    tic = time.perf_counter()
    approximated_heston = rHestonAK.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=1., N=N, N_Riccati=3000, N_fourier=10000, L=50.)
    toc = time.perf_counter()
    print(approximated_heston)
    print(f"Generating the approximated smile with N={N} took {toc-tic} seconds.")
