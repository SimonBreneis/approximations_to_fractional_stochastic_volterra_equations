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


m, N_time, T, sigma, r, K, S_0 = 100000, 50, 2., 0.4, 0.06, 1.05, 1.
BS_paths = S_0 * cf.BS_paths(sigma=sigma, T=T, n=N_time, m=m, r=r, antithetic=True)
samples = np.empty((1, m, N_time + 1))
samples[0, :, :] = BS_paths
price, models, features = cf.price_am(K=K, T=T, r=r, samples=samples, N_features=3, antithetic=True, payoff='put')
print(price)
print(cf.BS_price_eur_put(S_0=S_0, K=K, sigma=sigma, T=T, r=r))
nodes, weights = rk.quadrature_rule(H=0.1, N=1, T=T)
price = rHestonMarkovSamplePaths.price_am(K=K, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, S_0=S_0, r=r, T=T, nodes=nodes, weights=weights,
                                              m=m, N_time=50, N_dates=50, payoff='put')
print(price)
print(rHestonMarkovSamplePaths.eur(K=K, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, S_0=S_0, r=r,
                                           T=T, nodes=nodes, weights=weights, m=m, N_time=50, payoff='put', implied_vol=False))
time.sleep(36000)

N = 2
T, H, lambda_, nu, theta, V_0, rel_tol, S_0, rho, r = 0.04, 0.1, 0.3, 0.3, 0.02, 0.02, 1e-04, 1., -0.7, 0.06
k = np.linspace(-0.3, 0.15, 101) * np.sqrt(T)
K = np.exp(k)

E_V = rHestonFourier.solve_fractional_Riccati(F=lambda t, x: np.array([0.02 + 0.3 * 0.02 - 0.3 * x]), T=1.,
                                              N_Riccati=10000, H=0.1) + 0.02
avg_vol = np.trapz(np.real(E_V), dx=1 / 10000)
K = avg_vol * np.exp(np.linspace(-1., 2, 251))
k = np.log(K / avg_vol)
print(avg_vol)
# true_prices = Data.rHeston_prices_avg_vol
true_prices = rHestonFourier.price_avg_vol_call(K=K, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, H=0.1, verbose=10)
true_iv = cf.iv_eur_call(S_0=avg_vol, K=K, T=T, price=true_prices)
print(true_iv)
plt.plot(k, true_iv)
plt.show()
mark_prices = rHestonFourier.price_avg_vol_call(K=K, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, N=N, mode='european', rel_tol=rel_tol, H=H)
print(np.amax(np.abs(mark_prices - true_prices) / true_prices))
nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='european')
p, l, u = rHestonMarkovSamplePaths.price_avg_vol_call(K=K, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, m=100000,
                                                      N_time=200, nodes=nodes, weights=weights, euler=False)
plt.plot(k, true_prices, 'k-')
plt.plot(k, mark_prices, 'b-')
plt.plot(k, p, 'r-')
plt.plot(k, l, 'r--')
plt.plot(k, u, 'r--')
plt.show()

params = {'K': K, 'H': H, 'T': T, 'lambda': lambda_, 'nu': nu, 'theta': theta, 'V_0': V_0, 'rel_tol': rel_tol}

compute_rHeston_samples(params=params, Ns=np.array([3]), N_times=2 ** np.arange(9), modes=['european'], m=1000000,
                        sample_paths=True, recompute=True, vol_only=True, euler=False, antithetic=True)

compute_smiles_given_stock_prices(params=params, Ns=np.array([3]), N_times=2 ** np.arange(9), modes=['european'],
                                  option='average volatility call', euler=[False], antithetic=[True],
                                  true_smile=true_prices)
