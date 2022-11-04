import time
import matplotlib.pyplot as plt
import numpy as np
import Data
import RoughKernel
import rBergomi
import rHeston
import rHestonMarkov
import rHestonBackbone
from functions import *
import rBergomiMarkov
import rHestonMomentMatching
from scipy.stats import norm
import scipy.special

a_1 = np.array([0.30499035, 0.28386569, 0.26116807, 0.23651017, 0.20934237,
       0.17891143, 0.14470142, 0.11295608, 0.10699895, 0.11578251,
       0.12689855])
a_2 = np.array([0.30499035, 0.28386567, 0.26116804, 0.23651014, 0.20934235,
       0.17891144, 0.14470149, 0.11295606, 0.10699886, 0.11578248,
       0.12689853])
print(np.amax(np.abs(a_1 - a_2) / a_1))

a_1 = np.array([0.30495498, 0.28382563, 0.26112537, 0.23646876, 0.2093087 ,
       0.17889441, 0.14470716, 0.11296505, 0.10698657, 0.11573642,
       0.12682933])
a_2 = np.array([0.30496294, 0.28383166, 0.26112978, 0.2364719 , 0.20931106,
       0.17889678, 0.14471142, 0.11297331, 0.10698504, 0.11573126,
       0.12682308])
a_3 = np.array([0.30496339, 0.28383204, 0.26113008, 0.23647212, 0.20931119,
       0.17889683, 0.14471143, 0.11297398, 0.10698509, 0.11573127,
       0.12682317])
print(np.amax(np.abs(a_1 - a_2) / a_1))
print(np.amax(np.abs(a_2 - a_3) / a_2))


other_prices = rHestonBackbone.price_avg_vol_call(K=np.linspace(0.01, 0.03, 11), H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-04)
print((other_prices,))

other_prices = rHestonBackbone.price_avg_vol_call(K=np.linspace(0.01, 0.03, 11), H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-04, N=3)
print((other_prices,))
true_prices = rHestonBackbone.price_geom_asian_call(S_0=1, K=np.exp(np.linspace(-0.2, 0.1, 11)), H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-04)
print((true_prices,))

true_prices = rHestonBackbone.price_geom_asian_call(S_0=1, K=np.exp(np.linspace(-0.2, 0.1, 11)), H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-04, N=3)
print((true_prices,))

tic = time.perf_counter()
true_smile = rHestonBackbone.iv_eur_call(S_0=1, K=np.exp(np.linspace(-0.5, 0.3, 11)), H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, theta=0.02, V_0=0.02, T=0.5, N=3, rel_tol=1e-05, verbose=0)
print(time.perf_counter() - tic)
print((true_smile,))
tic = time.perf_counter()
true_smile = rHestonBackbone.iv_eur_call(S_0=1, K=np.exp(np.linspace(-0.5, 0.3, 11)), H=0.1, lambda_=0.3, nu=0.3, rho=-0.7, theta=0.02, V_0=0.02, T=0.5, rel_tol=1e-05, verbose=0)
print(time.perf_counter() - tic)
print((true_smile,))
time.sleep(360000)

K = 0.02 / 0.3 * np.exp(np.linspace(-1., 0.5, 151))
print(K)
true_smile = rHeston.price_avg_vol_call(K=K, H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1., rel_tol=1e-05, verbose=10)
for i in range(1, 11):
    approx_smile = rHestonMarkov.price_avg_vol_call(K=K, H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1., rel_tol=1e-05, verbose=10, N=i)
    print(np.amax(np.abs(approx_smile - true_smile) / true_smile))
print('Finished')
time.sleep(360000)

true_smile = rHeston.price_geom_asian_call(S_0=10, rho=-0.7, K=10 * np.linspace(0.8, 0.95, 101), H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1, rel_tol=2e-03, verbose=10)
iv = cf.iv_geom_asian_call(S_0=1, K=np.linspace(0.8, 1.2, 101), T=1, price=true_smile)
print((true_smile,))
true_smile = rHeston.price_avg_vol_call(K=np.linspace(0.001, 0.035, 101), H=0.1, lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1., rel_tol=2e-04, verbose=10)
print((true_smile,))
time.sleep(36000)

true_smile = Data.rHeston_prices_geom_asian
for i in range(1, 11):
    approx_smile = rHestonMarkov.price_geom_asian_call(S_0=1, K=np.exp(np.linspace(-1, 0.4, 281)), H=0.1, lambda_=0.3,
                                                       rho=-0.7, nu=0.3, theta=0.02,
                                                       V_0=0.02, T=1, N=i, mode='optimized', rel_tol=1e-05)
    print(np.amax(np.abs(true_smile - approx_smile) / true_smile))

