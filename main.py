import time
import matplotlib.pyplot as plt
import numpy as np
import Data
import RoughKernel
import rBergomi
import rHestonFourier
import rHestonMarkovSamplePaths
from functions import *
import rBergomiMarkov
import rHestonMomentMatching


S_0, H, T, lambda_, rho, nu, theta, V_0, rel_tol = 1., 0.1, 1., 0.3, -0.7, 0.3, 0.02, 0.02, 1e-05
k = np.linspace(-0.2, 0.1, 31)
N = 2
'''
true_prices = rHestonFourier.price_geom_asian_call(S_0=S_0, K=np.exp(k), lambda_=lambda_, rho=rho, nu=nu, theta=theta,
                                                   V_0=V_0, T=T, rel_tol=rel_tol, H=H, verbose=10)
print((true_prices,))
'''
true_prices = np.array([0.18274756, 0.17491193, 0.16703816, 0.15913092, 0.15119566,
       0.14323871, 0.13526743, 0.12729038, 0.11931754, 0.11136052,
       0.10343291, 0.09555062, 0.08773227, 0.07999973, 0.07237874,
       0.06489954, 0.05759773, 0.0505151 , 0.04370041, 0.03721002,
       0.03110791, 0.02546448, 0.02035312, 0.01584348, 0.01199048,
       0.00882036, 0.00631822, 0.00442462, 0.00304553, 0.00207214,
       0.0014006 ])
'''
mark_prices = rHestonFourier.price_geom_asian_call(S_0=S_0, K=np.exp(k), lambda_=lambda_, rho=rho, nu=nu, theta=theta,
                                                   V_0=V_0, T=T, rel_tol=rel_tol, H=H, verbose=10, N=N)
print((mark_prices,))
'''
mark_prices = np.array([0.18277981, 0.17494436, 0.16707057, 0.1591631 , 0.15122738,
       0.1432697 , 0.1352974 , 0.12731901, 0.11934449, 0.11138544,
       0.10345543, 0.09557033, 0.08774877, 0.08001262, 0.0723876 ,
       0.06490401, 0.05759746, 0.05050982, 0.04368994, 0.03719435,
       0.03108728, 0.02543945, 0.0203247 , 0.01581317, 0.0119602 ,
       0.00879218, 0.00629392, 0.00440521, 0.00303114, 0.00206221,
       0.00139421])
p, l, u = rHestonMarkovSamplePaths.price_geom_asian_call(K=np.exp(k), lambda_=lambda_, rho=rho, nu=nu, theta=theta,
                                                           V_0=V_0, S_0=S_0, T=T, H=H, N=N, m=10000, N_time=100)

print(np.amax(np.abs(mark_prices - true_prices) / true_prices))
plt.plot(k, true_prices, 'k-')
plt.plot(k, mark_prices, 'b-')
plt.plot(k, p, 'r-')
plt.plot(k, l, 'r--')
plt.plot(k, u, 'r--')
plt.show()

params = {'S': 1., 'K': np.exp(np.linspace(-0.25, 0.15, 71)), 'H': 0.1, 'T': 1., 'lambda': 0.3, 'rho': -0.7,
                     'nu': 0.3, 'theta': 0.02, 'V_0': 0.02, 'rel_tol': 1e-04}

compute_rHeston_samples(params=params, Ns=np.array([2]), N_times=2 ** np.arange(9), modes=['european'], vol_behaviours=['mackevicius antithetic'], m=1000000, sample_paths=True, recompute=False, vol_only=True)


params = {'S': 1., 'K': np.exp(np.linspace(-0.15, 0.1, 26)), 'H': 0.1, 'T': 1., 'lambda': 0.3, 'rho': -0.7,
                     'nu': 0.3, 'theta': 0.02, 'V_0': 0.02, 'rel_tol': 1e-04}
compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(9), modes=['european'], vol_behaviours=['mackevicius antithetic'], option='geometric asian call')
