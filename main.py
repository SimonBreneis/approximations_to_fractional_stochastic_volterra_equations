import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
# import Data
import RoughKernel as rk
# import rBergomi
import rHestonFourier
import rHestonMarkovSamplePaths
# from functions import *
# import rBergomiMarkov
# import rHestonMomentMatching


# Parameters for the rough Heston model. Note that the volatility process is parametrized as
# V_t = V_0 + int_0^t K(t-s) (theta - lambda_ V_s) ds + int_0^t nu sqrt(V_s) dW_s
S_0, V_0, T, lambda_, theta, nu, rho, H, N, N_time, m = 1., 0.02, 1., 0.3, 0.02, 0.3, -0.7, 0.1, 3, 200, 100000
rel_tol = 1e-05
K = S_0 * np.exp(np.linspace(-0.3, 0.15, 101))

# Get good nodes and weights for the Markovian approximation. If one is interested in standard Heston, set
# nodes = np.array([0]), weights = np.array([1])
nodes, weights = rk.quadrature_rule(H=H, N=N, T=T)

# To get samples, use the following function:
# samples = rHestonMarkovSamplePaths.samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes,
# weights=weights, rho=rho, S_0=S_0, m=m, N_time=N_time)
# The result will be an array of the shape (N+2, m), where m is the number of samples. The first component of the result
# is the final stock price, the second the final volatility, and the remaining N components are the components of the
# Markovian approximation of the volatility process.
# If one wants sample paths instead of only final sample values, one can set the optional parameter sample_paths in the
# above function to True. The result will then be of shape (N+2, m, N_time+1)

# We now give an example of computing implied volatility smiles.
# Compute the implied volatility smile for a European call option with MC and Mackevicius, including a MC confidence
# interval
(iv_MC, iv_MC_lower, iv_MC_upper) = rHestonMarkovSamplePaths.eur(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta,
                                                                 V_0=V_0, S_0=S_0, T=T, nodes=nodes, weights=weights,
                                                                 m=m, N_time=N_time, payoff='call', implied_vol=True)

# Compute the implied volatility smile using Fourier inversion for the Markovian approximation (hence, no MC and
# discretization errors)
iv_Markov = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                        rel_tol=rel_tol, nodes=nodes, weights=weights, implied_vol=True, call=True)

# Compute the implied volatility smile using Fourier inversion for the non-approximated rough Heston model. Note that
# we do not supply the nodes and weights here, but rather the Hurst parameter H
iv_true = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                      rel_tol=rel_tol, H=H, implied_vol=True, call=True)

plt.plot(np.log(K / S_0), iv_true, 'k-', label='True smile')
plt.plot(np.log(K / S_0), iv_Markov, 'b-', label='True Markovian approximation')
plt.plot(np.log(K / S_0), iv_MC, 'r-', label='Markovian simulation')
plt.plot(np.log(K / S_0), iv_MC_lower, 'r--')
plt.plot(np.log(K / S_0), iv_MC_upper, 'r--')
plt.xlabel('log-moneyness')
plt.ylabel('implied volatility')
plt.legend(loc='best')
plt.show()
time.sleep(36000)


# ----------------------------------------------------------------------------------------------------




N = np.arange(1, 101)
H = np.linspace(0.0001, 0.5, 101)
beta = 0.4275
alpha = 1.06418
print(5 * 2 ** (-1.5) / 200 / 122 ** 0.25)
print(6 * beta)
print(beta ** 2 * np.exp(alpha * beta) / (np.exp(alpha * beta) - 1))
H = 0.1
plt.plot(N, 0.03 / (0.5 - H) * (122 / ((0.5 + H) * N)) ** (beta * np.sqrt((0.5 + H) * N) - 0.5), label='Small mean-reversion')
plt.plot(N, 8 * ((0.5 + H) * N) ** (0.5 - H/2), label='Intermediate mean-reversion')
plt.plot(N, 2 * np.ones(100), label='Large mean-reversion')
plt.legend(loc='best')
plt.xlabel('N')
plt.ylabel('Coefficient')
plt.show()

plt.plot(H, alpha * np.sqrt(1 / (1 / H + 1/(1.5 - H))), label='Bayer, Breneis 2022')
plt.plot(H, alpha * np.sqrt(1/2 + H), label='New result')
plt.xlabel('H')
plt.ylabel('Rate of convergence')
plt.legend(loc='best')
plt.show()
print(np.sqrt(122))
print(1 / (0.5 + 0) ** 2 * np.sqrt(5 * np.pi ** 3 / 18) * beta ** (-1-0) * 0.5 * (np.exp(alpha * beta) - 1))
plt.plot(H, 1 / (0.5 + H) ** 2 * np.sqrt(5 * np.pi ** 3 / 18) * beta ** (-1-H) * 0.5 * (np.exp(alpha * beta) - 1))
plt.show()
print(1 / (4 * 0.4275 * np.sqrt(122 * np.pi)))
print(np.exp(2 + 1.06418/0.4275) / (4 * 0.4275 ** 2))
print(1 / 3 * N ** (-0.5) * (122 / ((0.5 + H) * N)) ** (0.4275 * np.sqrt((H+0.5) * N)))
print(1 / 3 * N ** (-0.5) * np.exp(2 * 0.4275 * np.sqrt((0.5 + H) * N) * (1 + 1.06418 / (2 * 0.4275) - np.log(2 * 0.4275 * np.sqrt((0.5 + H) * N)))))
print(np.exp(1) * (0.5 + 0.5) ** 0.25 / (4 * np.sqrt(np.pi) * 0.4275 ** 1.5))
print(7/5)
print(np.sqrt(5 * np.pi ** 3 / 18) * (1 / 0.4275) ** (1 + 0.5) * (0.5 * (np.exp(0.4275 * 1.06418) - 1)))
print(1 / (4 * 0.4275 * np.sqrt(np.pi)))

H = np.linspace(0, 0.5, 101)
plt.plot(H, np.sqrt(5 * np.pi ** 3 / 18) * (1 / (0.4275 * np.sqrt(0.5 + H))) ** (1 + H) * (0.5 * (np.exp(0.4275 * 1.06418) - 1)))
plt.plot(H, np.exp(1) * (0.5 + H) ** 0.25 / (4 * np.sqrt(np.pi) * 0.4275 ** 1.5))
plt.show()



def lower_gamma(a, x):
    return scipy.special.gamma(a) * scipy.special.gammainc(a, x)


def upper_bound(a, x):
    return scipy.special.gamma(a) - (((x + 2) ** a - x ** a - 2 ** a)/ (2 * a) + scipy.special.gamma(a)) * np.exp(-x)


for a in np.linspace(2, 20, 10):
    for x in np.exp(np.linspace(0, 20, 11)):
        print(a, x, lower_gamma(a, x), upper_bound(a, x))
time.sleep(36000)

H = np.linspace(0, 0.5, 201)
paper = 1.06418 / (1/H + 1/(1.5-H)) ** 0.5
new_res = 1.06418 / (1/(0.5+H) + 1/(1.5-H)) ** 0.5
new_res_2 = 1.06418 / (1/(0.5+H) + 1/(2.5-H)) ** 0.5
new_res_4 = 1.06418 / (1/(0.5+H) + 1/(4.5-H)) ** 0.5
new_res_3 = 1.06418 / (1/(0.5+H) + 0 * 1/(2.5-H)) ** 0.5
plt.plot(H, paper, label='Bayer, Breneis 2022')
plt.plot(H, new_res, label='New theorem')
plt.plot(H, new_res_2, label='New theorem 2')
plt.plot(H, new_res_3, label='New theorem 3')
plt.plot(H, new_res_4, label='New theorem 4')
plt.legend(loc='best')
plt.show()


for i in range(10):
    print(i + 1)
    print(rk.quadrature_rule(H=0.1, N=i + 1, T=1., mode='optimized'))
    print(rk.quadrature_rule(H=0.1, N=i + 1, T=1., mode='european'))
    print(rk.quadrature_rule(H=0.1, N=i + 1, T=1., mode='integrated'))

m, N_time, T, r, K, S_0 = 10000000, 256, 1., 0.06, 105., 100.
H, lambda_, nu, theta, V_0, rel_tol, rho = 0.1, 0.3, 0.3, 0.02, 0.02, 1e-05, -0.7
N_dates = N_time // 2
'''
eur_price = rHestonFourier.eur_call_put(S_0=S_0, K=np.array([K]), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                        T=T, r=r, rel_tol=rel_tol, H=H, implied_vol=False, call=False)'''

for N_time in 12 * 2 ** np.arange(7):
    for N in range(2, 3):
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T)
        # print(N_time, N, np.amax(nodes))
        '''
        print(rHestonFourier.eur_call_put(S_0=S_0, rel_tol=rel_tol, K=np.array([K]), lambda_=lambda_, rho=rho, nu=nu,
                                          theta=theta, V_0=V_0, r=r, T=T, H=H, call=False,
                                          implied_vol=False))'''
        N_time = 240
        am_price = rHestonMarkovSamplePaths.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                     S_0=S_0, r=r, T=T, nodes=nodes, weights=weights, m=m,
                                                     N_time=N_time, N_dates=48, payoff='put', feature_degree=6)[:4]
        print(N_time, N, np.amax(nodes), am_price)
time.sleep(36000)

params = {'K': K, 'H': H, 'T': T, 'lambda': lambda_, 'nu': nu, 'theta': theta, 'V_0': V_0, 'rel_tol': rel_tol}
'''
compute_rHeston_samples(params=params, Ns=np.array([3]), N_times=2 ** np.arange(9), modes=['european'], m=1000000,
                        sample_paths=True, recompute=True, vol_only=True, euler=False, antithetic=True)

compute_smiles_given_stock_prices(params=params, Ns=np.array([3]), N_times=2 ** np.arange(9), modes=['european'],
                                  option='average volatility call', euler=[False], antithetic=[True])
'''
