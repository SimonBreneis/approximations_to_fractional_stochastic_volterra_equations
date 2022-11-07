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


N = 2
T, H, lambda_, nu, theta, V_0, rel_tol, S_0, rho, r = 0.04, 0.1, 0.3, 0.3, 0.02, 0.02, 1e-04, 1., -0.7, 0.06
k = np.linspace(-0.3, 0.15, 101) * np.sqrt(T)
K = np.exp(k)
true_smile_r = rHestonFourier.iv_eur_call(S_0=S_0, K=K, lambda_=lambda_, H=H, T=T, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                        rel_tol=rel_tol, r=r, verbose=10)
approx_smile_r = rHestonFourier.iv_eur_call(S_0=S_0, K=K, lambda_=lambda_, H=H, T=T, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                          rel_tol=rel_tol, r=r, verbose=10, N=N)
true_smile = rHestonFourier.iv_eur_call(S_0=S_0, K=K, lambda_=lambda_, H=H, T=T, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                        rel_tol=rel_tol, r=0., verbose=10)
approx_smile = rHestonFourier.iv_eur_call(S_0=S_0, K=K, lambda_=lambda_, H=H, T=T, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                          rel_tol=rel_tol, r=0., verbose=10, N=N)
nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='european')
p, l, u = rHestonSP.iv_eur_call(K=K, lambda_=lambda_, nodes=nodes, weights=weights, T=T, rho=rho, nu=nu,
                                       theta=theta, V_0=V_0, r=r, m=1000000, N_time=200, S_0=S_0)
p_, l_, u_ = rHestonSP.iv_eur_call(K=K, lambda_=lambda_, nodes=nodes, weights=weights, T=T, rho=rho, nu=nu,
                                       theta=theta, V_0=V_0, r=r, m=1000000, N_time=200, S_0=S_0, euler=True)
print(np.amax(np.abs(true_smile - approx_smile) / true_smile))
print(np.amax(np.abs(true_smile_r - approx_smile_r) / true_smile_r))
print(np.amax(np.abs(true_smile_r - p) / true_smile_r))
print(np.amax(np.abs(true_smile_r - l) / true_smile_r))
print(np.amax(np.abs(true_smile_r - u) / true_smile_r))
print(np.amax(np.abs(true_smile_r - p_) / true_smile_r))
print(np.amax(np.abs(true_smile_r - l_) / true_smile_r))
print(np.amax(np.abs(true_smile_r - u_) / true_smile_r))
plt.plot(k, true_smile, 'k-')
plt.plot(k, approx_smile, 'brown')
plt.plot(k, true_smile_r, 'r-')
plt.plot(k, approx_smile_r, 'orange')
plt.plot(k, p, 'b-')
plt.plot(k, l, 'b--')
plt.plot(k, u, 'b--')
plt.plot(k, p_, 'g-')
plt.plot(k, l_, 'g--')
plt.plot(k, u_, 'g--')
plt.show()



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
