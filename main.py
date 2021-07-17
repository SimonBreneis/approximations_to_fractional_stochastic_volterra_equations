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


Data.plot_fBm_errors()
K = np.exp(-0.5 + 0.01 * np.arange(81))
print("True rough Heston:")
tic = time.perf_counter()
true_heston = rHeston.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=1., N_Riccati=3000, N_fourier=200000, L=300.)
toc = time.perf_counter()
print(true_heston)
print(f"Generating the true smile took {toc-tic} seconds.")
for N in [1, 2, 4, 8, 16, 32, 64]:
    print("Approximation with N nodes, our scheme:")
    tic = time.perf_counter()
    approximated_heston = rHestonAK.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=1., N=N, N_Riccati=3000, N_fourier=200000, L=300.)
    toc = time.perf_counter()
    print(approximated_heston)
    print(f"Generating the approximated smile with N={N} took {toc-tic} seconds.")
    error_int = np.sqrt(np.sum((true_heston - approximated_heston)**2))
    print(f"Error: {error_int}")
    '''
    print("Approximation with N nodes, AE scheme:")
    tic = time.perf_counter()
    approximated_heston_AE = rHestonAK.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=1., N=N, mode="AE")
    toc = time.perf_counter()
    print(approximated_heston)
    print(f"Generating the approximated smile with N={N} took {toc - tic} seconds.")
    error_int = np.sqrt(np.sum((true_heston - approximated_heston) ** 2))
    print(f"Error: {error_int}")
    '''
