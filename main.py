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


H = 0.07
T = 0.9
M = 1000000
N_vec = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64])
k_vec = np.array([-0.4 + 0.01*i for i in range(61)])
K_vec = np.exp(k_vec)


res = rBergomiBFG.implied_volatility_call_rBergomi_BFG(H=H, T=T, K=K_vec, m=M)
print("True")
print(res[0])
print(res[1])
print(res[2])


for N_ in N_vec:
    res = rBergomiAK.implied_volatility_call_rBergomi_AK(H=H, T=T, K=K_vec, M=M, N=N_, mode="observation")
    print(f"Approximation, N={N_}:")
    print(res[0])
