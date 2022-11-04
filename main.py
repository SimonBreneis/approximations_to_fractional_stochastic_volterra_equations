import time
import matplotlib.pyplot as plt
import numpy as np
import Data
import RoughKernel
import rBergomi
import rHestonFourier
from functions import *
import rBergomiMarkov
import rHestonMomentMatching
from scipy.stats import norm
import scipy.special


other_prices = rHestonFourier.price_avg_vol_call(K=np.linspace(0.01, 0.03, 11), H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-04)
print((other_prices,))

other_prices = rHestonFourier.price_avg_vol_call(K=np.linspace(0.01, 0.03, 11), H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-04, N=3)
print((other_prices,))
true_prices = rHestonFourier.price_geom_asian_call(S_0=1, K=np.exp(np.linspace(-0.2, 0.1, 11)), H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-04)
print((true_prices,))

true_prices = rHestonFourier.price_geom_asian_call(S_0=1, K=np.exp(np.linspace(-0.2, 0.1, 11)), H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-04, N=3)
print((true_prices,))

tic = time.perf_counter()
true_smile = rHestonFourier.iv_eur_call(S_0=1, K=np.exp(np.linspace(-0.5, 0.3, 11)), H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, theta=0.02, V_0=0.02, T=0.5, N=3, rel_tol=1e-05, verbose=0)
print(time.perf_counter() - tic)
print((true_smile,))
tic = time.perf_counter()
true_smile = rHestonFourier.iv_eur_call(S_0=1, K=np.exp(np.linspace(-0.5, 0.3, 11)), H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-05, verbose=0)
print(time.perf_counter() - tic)
print((true_smile,))
time.sleep(360000)

K = 0.02 / 0.3 * np.exp(np.linspace(-1., 0.5, 151))
print(K)
true_smile = rHestonFourier.price_avg_vol_call(K=K, H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1., rel_tol=1e-05, verbose=10)
for i in range(1, 11):
    approx_smile = rHestonFourier.price_avg_vol_call(K=K, H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1., rel_tol=1e-05, verbose=10, N=i)
    print(np.amax(np.abs(approx_smile - true_smile) / true_smile))
print('Finished')
time.sleep(360000)

true_smile = rHestonFourier.price_geom_asian_call(S_0=10, rho=-0.7, K=10 * np.linspace(0.8, 0.95, 101), H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1, rel_tol=2e-03, verbose=10)
iv = cf.iv_geom_asian_call(S_0=1, K=np.linspace(0.8, 1.2, 101), T=1, price=true_smile)
print((true_smile,))
true_smile = rHestonFourier.price_avg_vol_call(K=np.linspace(0.001, 0.035, 101), H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1., rel_tol=2e-04, verbose=10)
print((true_smile,))
time.sleep(36000)

true_smile = Data.rHeston_prices_geom_asian
for i in range(1, 11):
    approx_smile = rHestonFourier.price_geom_asian_call(S_0=1, K=np.exp(np.linspace(-1, 0.4, 281)), H=0.1, lambda_=0.3,
                                                       rho=-0.7, nu=0.3, theta=0.02,
                                                       V_0=0.02, T=1, N=i, mode='optimized', rel_tol=1e-05)
    print(np.amax(np.abs(true_smile - approx_smile) / true_smile))

