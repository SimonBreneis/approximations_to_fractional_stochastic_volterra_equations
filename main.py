import time
import matplotlib.pyplot as plt
import numpy as np
import Data
import RoughKernel
# import rBergomi
import rHestonFourier
import rHestonMarkovSamplePaths
from functions import *
# import rBergomiMarkov
import rHestonMomentMatching


print(rk.quadrature_rule(H=0.1, N=2, T=1.), rk.quadrature_rule(H=0.1, N=3, T=1.))
m, N_time, T, r, K, S_0 = 2000000, 256, 2., 0.06, 105., 100.
H, lambda_, nu, theta, V_0, rel_tol, rho = 0.1, 0.3, 0.3, 0.02, 0.02, 1e-05, -0.7
N_dates = N_time // 2
'''
eur_price = rHestonFourier.eur_call_put(S_0=S_0, K=np.array([K]), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                        T=T, r=r, rel_tol=rel_tol, H=H, implied_vol=False, call=False)'''
nodes, weights = rk.quadrature_rule(H=H, N=3, T=T)
for i in range(1, 13):
    am_price = rHestonMarkovSamplePaths.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, r=r,
                                                 T=T, nodes=nodes, weights=weights, m=m, N_time=256,
                                                 N_dates=12, payoff='put', feature_degree=i)[:4]
    '''
    nodes, weights = rk.quadrature_rule(H=H, N=2, T=T)
    am_price_approx = rHestonMarkovSamplePaths.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0,
                                                        r=r, T=T, nodes=nodes, weights=weights, m=m, N_time=N_time,
                                                        N_dates=N_dates, payoff='put', N_features=3)[0]'''
    print(am_price)
time.sleep(36000)

params = {'K': K, 'H': H, 'T': T, 'lambda': lambda_, 'nu': nu, 'theta': theta, 'V_0': V_0, 'rel_tol': rel_tol}

compute_rHeston_samples(params=params, Ns=np.array([3]), N_times=2 ** np.arange(9), modes=['european'], m=1000000,
                        sample_paths=True, recompute=True, vol_only=True, euler=False, antithetic=True)

compute_smiles_given_stock_prices(params=params, Ns=np.array([3]), N_times=2 ** np.arange(9), modes=['european'],
                                  option='average volatility call', euler=[False], antithetic=[True])
