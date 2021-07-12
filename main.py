import time
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import stats
import mpmath as mp
import Data
import ComputationalFinance as cf
import rBergomiBFG
import rBergomiAK
import rHeston
import rHestonAK
import fBmAK
import RoughKernel as rk


k_vec = -0.3 + 0.01*np.arange(51)
K_vec = np.exp(k_vec)
implied_volatilities = rHeston.implied_volatility_Fourier(K=K_vec, L=200, N=1000, N_fourier=200**2, T=0.1)
print(implied_volatilities)
plt.plot(k_vec, implied_volatilities)
plt.show()
