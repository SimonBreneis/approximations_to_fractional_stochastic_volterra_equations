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


k_vec = -1.3 + 0.02*np.arange(41)
K_vec = np.exp(k_vec)
implied_volatilities = rHestonAK.implied_volatility_Fourier(K=K_vec, L=200, N_Riccati=1000, N_fourier=200**2, T=1., nu=1., lambda_=0.3, rho=-0.7, theta=0.02, V_0=0.02, N=1)
print(implied_volatilities)
