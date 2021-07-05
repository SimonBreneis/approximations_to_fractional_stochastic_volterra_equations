import time
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import stats
import mpmath as mp
import Data
import QuadratureRulesRoughKernel as qr
import ComputationalFinance as cf
import rBergomiBFG
import rBergomiAK
import rHeston
import rHestonAK
import fBmAK
import RoughKernel as rk


Data.fit_observations()

alpha = 1.6
T = 1.
tol = 1e-7
m_vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n_vec = np.array([[1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024],
         [0, 1, 0, 2, 3, 4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512],
         [0, 0, 1, 0, 2, 3, 4, 5, 8, 11, 15, 21, 30, 43, 60, 85, 121, 171, 241, 341],
         [0, 0, 0, 1, 0, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256],
         [0, 0, 0, 0, 1, 0, 2, 3, 5, 6, 9, 13, 18, 26, 36, 51, 72, 102, 145, 205],
         [0, 0, 0, 0, 1, 0, 2, 3, 4, 5, 7, 11, 15, 21, 30, 43, 60, 85, 121, 171],
         [0, 0, 0, 0, 0, 1, 0, 2, 3, 5, 6, 9, 13, 18, 26, 37, 52, 73, 103, 146],
         [0, 0, 0, 0, 0, 1, 0, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128],
         [0, 0, 0, 0, 0, 1, 0, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 57, 80, 114],
         [0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 4, 6, 9, 13, 18, 26, 36, 51, 72, 102]])


H = 0.35
print(f"H={H}")
A = np.sqrt(1/H + 1/(1.5-H))

m_min = 4
min_err = 1.
for i in range(13, n_vec.shape[1]-4):
    print(f"N={n_vec[0, i]} here")
    for m in m_vec:
        if m == m_min or m == m_min+1:
            n = n_vec[m-1, i]
            if n != 0:
                a = alpha/(A*(1.5-H))*np.sqrt(n*m)
                b = alpha/(A*H)*np.sqrt(n*m)
                res = rk.optimize_error_fBm(H, T, int(m), int(n), a, b, tol)
                if m == m_min:
                    min_err = res[0]
                else:
                    if res[0] < min_err:
                        m_min = m
                        min_err = res[0]
                print(f"m={m}, n={n}, N={n*m}, error={res[0]}, a={res[2]}, b={res[3]}")
    print("===============================================================================")


H = 0.05
print(f"H={H}")
A = np.sqrt(1/H + 1/(1.5-H))

m_min = 4
min_err = 1.
for i in range(17, n_vec.shape[1]):
    print(f"N={n_vec[0, i]} here")
    for m in m_vec:
        if m == m_min or m == m_min+1:
            n = n_vec[m-1, i]
            if n != 0:
                a = alpha/(A*(1.5-H))*np.sqrt(n*m)
                b = alpha/(A*H)*np.sqrt(n*m)
                res = rk.optimize_error_fBm(H, T, int(m), int(n), a, b, tol)
                if m == m_min:
                    min_err = res[0]
                else:
                    if res[0] < min_err:
                        m_min = m
                        min_err = res[0]
                print(f"m={m}, n={n}, N={n*m}, error={res[0]}, a={res[2]}, b={res[3]}")
    print("===============================================================================")


H = 0.1
m = 4
n = 181
a = 10.4
b = 143.
res = rk.optimize_error_fBm(H, 1., int(m), int(n), a, b, tol)
print(f"m={m}, n={n}, N={n*m}, error={res[0]}, a={res[2]}, b={res[3]}")

H = 0.1
m = 9
n = 57
a = 8.5
b = 126.
res = rk.optimize_error_fBm(H, 1., int(m), int(n), a, b, tol)
print(f"m={m}, n={n}, N={n*m}, error={res[0]}, a={res[2]}, b={res[3]}")

H = 0.1
m = 9
n = 81
a = 10.6
b = 152.
res = rk.optimize_error_fBm(H, 1., int(m), int(n), a, b, tol)
print(f"m={m}, n={n}, N={n*m}, error={res[0]}, a={res[2]}, b={res[3]}")




