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


k_rHeston = -1.3 + 0.02 * np.arange(81)
approx = rHestonAK.implied_volatility(K=np.exp(k_rHeston), lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=1., N_Riccati=500, N_fourier=1000, N=8)
plt.plot(k_rHeston, Data.rHeston)
plt.plot(k_rHeston, approx)
plt.plot(k_rHeston, Data.rHeston_8)
plt.show()
