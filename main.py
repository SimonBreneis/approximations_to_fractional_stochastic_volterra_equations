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

'''
N_vec = np.array([1, 4, 16, 64])
a_vec = np.zeros(4)
b_vec = np.zeros(4)
m_vec = np.zeros(4)
n_vec = np.zeros(4)
for i in range(4):
    [m_vec[i], n_vec[i], a_vec[i], b_vec[i]] = rk.get_parameters(0.07, N_vec[i], 0.9, "observation")

rk.plot_kernel_approximations(0.07, m_vec, n_vec, a_vec, b_vec)

Data.plot_rBergomi_smiles_BFG_AK_()
'''
H = 0.07
T = 0.9
M = 10000
rounds = 100
N_time = 2000
#  N_vec = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64])
N_vec = np.array([1, 2, 3, 4, 6, 8, 11])
k_vec = np.array([-0.4 + 0.01*i for i in range(61)])
K_vec = np.exp(k_vec)


res = rBergomiBFG.implied_volatility_call_rBergomi_BFG(H=H, T=T, K=K_vec, m=M, N=N_time, rounds=rounds)
print("True")
print(res[0])
print(res[1])
print(res[2])


for N_ in N_vec:
    res = rBergomiAK.implied_volatility_call_rBergomi_AK(H=H, T=T, K=K_vec, M=M, N=N_, mode="observation", N_time=N_time, rounds=rounds)
    print(f"Approximation, N={N_}:")
    print(res[0])
