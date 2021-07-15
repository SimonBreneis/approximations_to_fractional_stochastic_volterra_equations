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


K = np.exp(-1.3 + 0.02 * np.arange(81))
tic = time.perf_counter()
rHeston.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=1.)
toc = time.perf_counter()
print(f"Generating the true smile took {toc-tic} seconds.")
for N in [1, 2, 4, 8, 16, 32]:
    tic = time.perf_counter()
    rHestonAK.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=1., N=N)
    toc = time.perf_counter()
    print(f"Generating the approximated smile with N={N} took {toc-tic} seconds.")
