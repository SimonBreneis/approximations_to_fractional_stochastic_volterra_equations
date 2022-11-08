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


print(rk.quadrature_rule(H=0.1, N=2, T=1.), rk.quadrature_rule(H=0.1, N=3, T=1.))
m, N_time, T, sigma, r, K, S_0 = 1000000, 100, 2., 0.4, 0.06, 105., 100.
H, lambda_, nu, theta, V_0, rel_tol, rho = 0.1, 0.3, 0.3, 0.02, 0.02, 1e-04, -0.7
nodes, weights = rk.quadrature_rule(H=H, N=2, T=T)
print(rHestonFourier.eur_call_put(S_0=S_0, K=np.array([K]), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, r=r, rel_tol=rel_tol, H=H, verbose=10, nodes=nodes, weights=weights, implied_vol=False, call=False))
print(rHestonMarkovSamplePaths.eur(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T, nodes=nodes, weights=weights, r=r, m=m, N_time=N_time, payoff='put', implied_vol=False))
nodes, weights = rk.quadrature_rule(H=0.1, N=1, T=T)
price = rHestonMarkovSamplePaths.price_am(K=K, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, S_0=S_0, r=r, T=T, nodes=nodes, weights=weights,
                                              m=m, N_time=50, N_dates=50, payoff='put')
print(price)
print(rHestonMarkovSamplePaths.eur(K=K, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, S_0=S_0, r=r,
                                           T=T, nodes=nodes, weights=weights, m=m, N_time=50, payoff='put', implied_vol=False))
time.sleep(36000)

N = 2
k = np.linspace(-0.3, 0.15, 101) * np.sqrt(T)
K = np.exp(k)

E_V = rHestonFourier.solve_fractional_Riccati(F=lambda t, x: np.array([0.02 + 0.3 * 0.02 - 0.3 * x]), T=1.,
                                              N_Riccati=10000, H=0.1) + 0.02
avg_vol = np.trapz(np.real(E_V), dx=1 / 10000)
K = avg_vol * np.exp(np.linspace(-1., 2, 251))
k = np.log(K / avg_vol)
print(avg_vol)
# true_prices = Data.rHeston_prices_avg_vol
true_prices = rHestonFourier.price_avg_vol_call_put(K=K, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, H=0.1, verbose=10)
true_iv = cf.iv_eur_call(S_0=avg_vol, K=K, T=T, price=true_prices)
print(true_iv)
plt.plot(k, true_iv)
plt.show()
mark_prices = rHestonFourier.price_avg_vol_call_put(K=K, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, N=N, mode='european', rel_tol=rel_tol, H=H)
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
