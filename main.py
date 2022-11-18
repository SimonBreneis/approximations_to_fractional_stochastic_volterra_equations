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


H = 0.05
N = 10
T = 1.
tol = 1e-10
nodes, weights = rk.quadrature_rule(H=H, N=N, T=T)
real_error = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='intersections', tol=tol)[0]
print(real_error)
for tol in 10. ** (-np.arange(0, 10)):
    tic = time.perf_counter()
    res = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='intersections', tol=tol)
    print(tol, 'intersections', np.abs(res[0] - real_error) / real_error, res[1], 1000 * (time.perf_counter() - tic))
    tic = time.perf_counter()
    # res = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='exact - trapezoidal', tol=tol)
    print(tol, 'exact - trapezoidal', np.abs(res[0] - real_error) / real_error, res[1], 1000 * (time.perf_counter() - tic))
    if tol > 2e-03:
        tic = time.perf_counter()
        # res = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='trapezoidal', tol=tol)
        print(tol, 'trapezoidal', np.abs(res[0] - real_error) / real_error, res[1], 1000 * (time.perf_counter() - tic))

tic = time.perf_counter()
print(rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='intersections', tol=tol))
print(time.perf_counter() - tic)
time.sleep(36000)

N = np.linspace(0.5, 100, 1001)
H = np.linspace(0.0001, 0.5, 101)
beta = 0.4275
alpha = 1.06418
q = 10
r = 0.3
gamma = 0.86
print(np.sqrt(11))
plt.plot(N, np.sqrt(11) * 7 * N ** (-0.0009) * N ** (1 / (4 * beta * np.sqrt(N))))
plt.show()
plt.plot(H, ((0.75 - H/2) * (H + 0.5)) / (2 * beta + 1/2 - H))
plt.show()
x = np.linspace(1, 10, 101)
print(0.173 * 0.75)
plt.plot(x, np.log(x) / np.sqrt(x))
plt.show()
print(-3*beta * (1 - np.exp(-alpha*beta)) / (4 * (1 - np.exp(-alpha*beta) + 2 * gamma * beta ** 2)))
print(2 * 122 ** 0.25)
print(3 / 100 / 121 ** 0.75)
print(np.exp(2*gamma-2))
print((1-np.exp(-alpha*beta) - gamma*beta) / (2*(1-np.exp(-alpha*beta)+2*gamma*beta)))
print(3 * beta * (1 - np.exp(-alpha * beta)) / (4 * (1 - np.exp(-alpha*beta) + beta**2)))
print(3 * beta * (1 - np.exp(-alpha * beta)) / (4 * (1 - np.exp(-alpha*beta) + 2*beta**2)))
plt.plot(N, (N / 122) ** (- (3 * beta * (1 - np.exp(-alpha*beta))) / (4 * (1 - np.exp(-alpha*beta) + 2 * beta ** 2))) + (N / 122) ** (1 / np.sqrt(N)))
plt.show()
plt.plot(N, (N / 122) ** ((2 * r - 1) * beta / 2) * (N / 122) ** (1 / np.sqrt(N)))
plt.show()
plt.plot(N, (N / 122) ** ((4 * q + 3) / (8 * N)))
plt.show()
print(np.sqrt(2/122) * 5/200)
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
