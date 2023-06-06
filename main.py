import time
import matplotlib.pyplot as plt
import numpy as np
import Data
import ComputationalFinance
import RoughKernel as rk
import rBergomi
import fBmMarkov
import functions
import rHestonFourier
import rHestonMarkovSimulation
import rHestonQESimulation
from functions import *
import rBergomiMarkov
import rHestonMomentMatching
import scipy.stats, scipy.optimize, scipy.integrate, scipy.linalg, scipy.special
from scipy.special import gamma


T, rho, nu, theta, V_0, lambda_, S_0, rel_tol = 0.5, -0.7, 0.3, 0.02, 0.02, 0.3, 1., 1e-05
k = np.linspace(-1., 0.5, 201)
K = np.exp(k)
H_vec = np.array([-0.05])
N = 3
true_smiles = np.empty((len(H_vec), len(K)))
approx_smiles = np.empty((len(H_vec), len(K)))

print(rk.quadrature_rule(H=0.1, N=3, T=1.))
for i in range(len(H_vec)):
    H = H_vec[i]
    '''
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='GG')
    samples = rHestonMarkovSimulation.samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, r=0.,
                                              nodes=nodes, weights=weights, S_0=S_0, m=2 ** 20, N_time=256, verbose=2)[0]
    vol, lower, upper = cf.eur_MC(S_0=S_0, K=K, T=T, samples=samples[0, :], r=0., payoff='call', implied_vol=True)
    plt.plot(k, vol, color=color(i, len(H_vec)), label=f'H={H}')
    plt.plot(k, lower, '--', color=color(i, len(H_vec)))
    plt.plot(k, upper, '--', color=color(i, len(H_vec)))
    '''

    tic = time.perf_counter()
    true_smiles[i, :] = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T,
                                                    H=H, rho=rho, r=0., rel_tol=rel_tol, verbose=2)
    dur = time.perf_counter() - tic
    print(dur)
    plt.plot(k, true_smiles[i, :])
    plt.show()
    print('did the first one')

    '''
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='GG')
    print(H, nodes, weights)
    approx_smiles[i, :] = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T,
                                                      nodes=nodes, weights=weights, rho=rho, r=0., rel_tol=rel_tol)
    '''
    '''
    print(f'H={H}, N={N}, error={np.amax(np.abs(true_smiles[i, :] - approx_smiles[i, :]) / true_smiles[i, :])}, '
          f'{dur}, {time.perf_counter() - tic}')
    plt.plot(k, true_smiles[i, :], color=color(i, len(H_vec)), label=f'H={H}')
    '''
    # plt.plot(k, approx_smiles[i, :], '--', color=color(i, len(H_vec)), label=f'H={H}')
nodes, weights = rk.quadrature_rule(H=H_vec[0], N=N, T=T, mode='GG')
tic = time.perf_counter()
approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, H=H_vec[0],
                                           nodes=nodes, weights=weights, rho=rho, r=0., rel_tol=rel_tol, verbose=2)
print(time.perf_counter() - tic)
print((true_smiles[0, :], approx_smile))
plt.plot(k, true_smiles[0, :])
plt.plot(k, approx_smile)
plt.legend(loc='best')
plt.xlabel('log-moneyness')
plt.ylabel('implied volatility')
plt.show()



H, T, rho, nu, theta, V_0, lambda_, S_0 = 0.1, 1., -0.7, 0.3, 0.02, 0.02, 0.3, 1.
N_time = 100000
nodes, weights = rk.quadrature_rule(H=H, N=3, T=T, mode='BL2')
sample = rHestonMarkovSimulation.samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes,
                                         weights=weights, rho=rho, S_0=S_0, r=0., m=2, N_time=N_time, sample_paths=True,
                                         qmc=False)[0]
# sample = rHestonQESimulation.samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0, r=0., m=2, N_time=N_time, sample_paths=True, qmc=False)
'''
plt.plot(np.linspace(0, T, N_time + 1), sample[0, 0, :], label='stock S')
plt.plot(np.linspace(0, T, N_time + 1), np.sqrt(sample[1, 0, :]), label=r'volatility $\sqrt{V}$')
plt.legend(loc='best')
plt.xlabel('Time t')
plt.show()'''
plt.plot(np.linspace(0, T, N_time + 1), np.sqrt(sample[1, 0, :]))
plt.legend(loc='best')
plt.ylabel(r'Volatility $\sqrt{V}$')
plt.xlabel('Time t')
plt.show()

'''
Data.illustrate_bermudan_option_prices()
'''
T, H, rho, eta, S_0, V_0 = 0.9, 0.07, -0.9, 1.9, 1., 0.235 ** 2
m, N_time = 10 ** 6, 2000
k = np.linspace(-0.35, 0.4, 101)
K = np.exp(k)
rel_tol = 9e-02
true_smile, lower, upper = rBergomi.implied_volatility(H=H, T=T, eta=eta, V_0=V_0, S_0=S_0, rho=rho, K=K, rel_tol=rel_tol, verbose=2)
plt.plot(k, true_smile, 'k-', label='Non-Markovian approximation')
plt.plot(k, lower, 'k--', label='confidence interval')
plt.plot(k, upper, 'k--')
nodes, weights = rk.quadrature_rule(H=H, N=16, T=T, mode='paper')
our_smile, _, _ = rBergomiMarkov.implied_volatility(H=H, T=T, eta=eta, V_0=V_0, S_0=S_0, rho=rho, K=K, nodes=nodes,
                                                    weights=weights, rel_tol=rel_tol, verbose=2)
plt.plot(k, our_smile, 'r-', label='Our approach')
nodes, weights = rk.quadrature_rule(H=H, N=16, T=T, mode='AK')
our_smile, _, _ = rBergomiMarkov.implied_volatility(H=H, T=T, eta=eta, V_0=V_0, S_0=S_0, rho=rho, K=K, nodes=nodes,
                                                    weights=weights, rel_tol=rel_tol, verbose=2)
plt.plot(k, our_smile, 'b-', label='Alfonsi, Kebaier')
nodes, weights = rk.harms_rule(H=H, n=16, m=1)
our_smile, _, _ = rBergomiMarkov.implied_volatility(H=H, T=T, eta=eta, V_0=V_0, S_0=S_0, rho=rho, K=K, nodes=nodes,
                                                    weights=weights, rel_tol=rel_tol, verbose=2)
plt.plot(k, our_smile, 'orange', label='Harms, m=1')
nodes, weights = rk.harms_rule(H=H, n=2, m=8)
our_smile, _, _ = rBergomiMarkov.implied_volatility(H=H, T=T, eta=eta, V_0=V_0, S_0=S_0, rho=rho, K=K, nodes=nodes,
                                                    weights=weights, rel_tol=rel_tol, verbose=2)
plt.plot(k, our_smile, 'green', label='Harms, m=8')
plt.legend(loc='best')
plt.xlabel('log-moneyness')
plt.ylabel('implied volatility')
plt.show()



lambda_, rho, nu, H, T, theta, V_0, S_0 = 0.3, -0.7, 0.3, 0.1, 1., 0.02, 0.02, 1.
rel_tol = 3e-04
k = np.linspace(-1.5, 1., 251)
K = np.exp(k)
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol,
                                         lambda_=lambda_, H=H)
nodes, weights = rk.quadrature_rule(H=H, N=6, T=T, mode='paper')
our_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol,
                                        lambda_=lambda_, nodes=nodes, weights=weights)
plt.plot(k, true_smile, 'k-', label='Non-Markovian approximation')
plt.plot(k, our_smile, 'k--', label='N=6, our method')
for N in [1, 4, 16, 64, 256, 1024]:
    print(N)
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='AE')
    our_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol,
                                             lambda_=lambda_, nodes=nodes, weights=weights)
    plt.plot(k, our_smile, label=f'N={N}, Abi Jaber, El Euch')
plt.legend(loc='best')
plt.xlabel('log-moneyness')
plt.ylabel('implied volatility')
plt.show()



nodes, weights = rk.quadrature_rule(H=0.07, T=1., N=6)
tic = time.perf_counter()
samples = rBergomiMarkov.generate_samples(H=0.07, T=1., eta=1.9, rho=-0.9, S_0=100., V_0=0.09, nodes=nodes, weights=weights, M=100000, N_time=128)
print(time.perf_counter() - tic)
tic = time.perf_counter()
samples = rBergomi.generate_samples(H=0.07, T=1., eta=1.9, rho=-0.9, S_0=100., V_0=0.09, M=100000, N=128)
print(time.perf_counter() - tic)
print(cf.eur_MC(S_0=100., K=np.arange(70, 150, 10), T=1., samples=samples, payoff='put'))
time.sleep(36000)

'''
S_0, lambda_, rho, nu, theta, V_0, rel_tol, T, H = 1., 0.3, -0.7, 0.3, 0.02, 0.02, 1e-05, 1., 0.1
k = np.linspace(-1, 0.5, 51)
K = np.exp(k)
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, H=H)
for N in [1, 2, 3, 4, 5, 6]:
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='BL2')
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    err = np.amax(np.abs(true_smile - approx_smile) / true_smile)
    print(N, np.amax(nodes), err)
print('Finished')
time.sleep(3600000)
'''
'''
Data.illustrate_bermudan_option_prices()
print('Finished')
time.sleep(3600000)
'''
'''
S_0, lambda_, rho, nu, theta, V_0, T, rel_tol = 1., 0.3, -0.7, 0.3, 0.02, 0.02, 0.04, 1e-04
k_vec = np.linspace(-1., 0.5, 61) * np.sqrt(T)
K = np.exp(k_vec)
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                              H=0.001, verbose=2, rel_tol=rel_tol)
nodes, weights = rk.quadrature_rule(H=0.001, T=T, N=8, mode='paper')
l2_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                    T=T, nodes=nodes, weights=weights, verbose=2, rel_tol=rel_tol)
nodes, weights = rk.quadrature_rule(H=0.001, T=T, N=8, mode='GG')
l1_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                    T=T, nodes=nodes, weights=weights, verbose=2, rel_tol=rel_tol)
plt.plot(k_vec, true_smile, 'k-', label='True smile')
plt.plot(k_vec, l2_smile, label=r'$L^2$-quadrature rule')
plt.plot(k_vec, l1_smile, label=r'$L^1$-quadrature rule')
plt.xlabel('Log-moneyness')
plt.ylabel('Implied volatility')
plt.title('Implied volatility smiles using\n' + r'$L^1$- or $L^2$-quadrature rules for $H=0.001$ and $N=8$')
plt.legend(loc='best')
plt.show()
'''
'''
H_vec = [0.1, 0.01, 0.001]
N_vec = [1, 2, 4, 8, 16, 32]
errors = np.empty((len(H_vec), len(N_vec)))

for i in range(len(H_vec)):
    # true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
    #                                          H=H_vec[i], verbose=2, rel_tol=rel_tol)
    for j in range(len(N_vec)):
        nodes, weights = rk.quadrature_rule(H=H_vec[i], N=N_vec[j], T=1., mode='GG')
        print(H_vec[i], N_vec[j], np.amax(nodes))
        # approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
        #                                            T=T, nodes=nodes, weights=weights, verbose=2, rel_tol=rel_tol)
        # errors[i, j] = np.amax(np.abs(approx_smile - true_smile) / true_smile)
        # print(H_vec[i], N_vec[j], errors[i, j])

print(errors)

for i in range(len(H_vec)):
    plt.loglog(N_vec, errors[i, :], label=f'H={H_vec[i]}')
plt.legend(loc='best')
plt.xlabel('Dimension N')
plt.ylabel('Maximal relative error')
plt.title('Relative errors in European implied volatility smiles')
plt.show()
'''

H = 0.1
lambda_, rho, nu, theta, V_0, S_0, T, rel_tol, verbose = 0.3, -0.7, 0.3, 0.02, 0.02, 1., 1., 1e-05, 2
r = 0.06
N = 2
k = np.linspace(-0.1, 0.05, 16)
K = np.exp(k)
K = 1.05
# T = np.linspace(0, 1, 17)[1:]
# T = np.array([0.5, 1.])
# T = np.array([1.])
verbose = 1

params = {'S': 1., 'K': np.array([K]), 'H': H, 'T': T, 'lambda': lambda_, 'rho': rho, 'nu': nu, 'theta': theta, 'V_0': V_0,
          'rel_tol': rel_tol, 'r': r}
'''
print(rHestonFourier.eur_call_put(S_0=100, K=np.array([100, 105, 110]), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                  T=T, call=False, rel_tol=1e-05, implied_vol=False, r=r, H=H))

for N_time in [2048]:
    est, stat, _, _, _, _ = \
        rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T,
                                     payoff='put', r=r, m=1_000_000, N_time=N_time, N_dates=4,
                                     feature_degree=8, antithetic=False)
    print('QE, 4 dates, ', N_time, 100 * est, 100 * stat)

for N_time in [2048]:
    est, stat, _, _, _, _ = \
        rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T,
                                     payoff='put', r=r, m=1_000_000, N_time=N_time, N_dates=16,
                                     feature_degree=8, antithetic=False)
    print('QE, 16 dates, ', N_time, 100 * est, 100 * stat)

for N_time in [2048]:
    est, stat, _, _, _, _ = \
        rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T,
                                     payoff='put', r=r, m=1_000_000, N_time=N_time, N_dates=256,
                                     feature_degree=8, antithetic=False)
    print('QE, 256 dates, ', N_time, 100 * est, 100 * stat)
'''
'''
params['r'] = 0.
params['K'] = np.exp(np.linspace(-0.1, 0.05, 16))
print(compute_smiles_given_stock_prices_QMC(params=params, Ns=np.array([2, 3]), N_times=np.array([4, 8]),
                                            simulator=['Euler', 'Weak'], n_samples=2 ** 17))
print('Finished')
time.sleep(3600000)
'''
for N_time in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    if N_time >= 512:
        est, stat = rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                 S_0=S_0, T=T, payoff='put', r=r, m=2 ** 18, N_time=N_time, N_dates=4,
                                                 feature_degree=6, qmc=True, qmc_error_estimators=25, verbose=1)
        print('QE, 4 dates, ', N_time, 100 * est, 100 * stat)

    if N_time >= 512:
        est, stat = rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                 S_0=S_0, T=T, payoff='put', r=r, m=2 ** 18, N_time=N_time, N_dates=16,
                                                 feature_degree=6, qmc=True, qmc_error_estimators=25, verbose=1)
        print('QE, 16 dates, ', N_time, 100 * est, 100 * stat)
    if N_time >= 512:
        est, stat = rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                 S_0=S_0,
                                                 T=T, payoff='put', r=r, m=2 ** 17, N_time=N_time, N_dates=256,
                                                 feature_degree=6, qmc=True, qmc_error_estimators=50, verbose=1)
        print('QE, 256 dates, ', N_time, 100 * est, 100 * stat)
print('Finished')
time.sleep(360000)
verbose = 0
for N in [4]:
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='BL2')
    print(rHestonFourier.eur_call_put(S_0=100, K=np.array([100, 105, 110]), lambda_=lambda_, rho=rho, nu=nu,
                                      theta=theta, V_0=V_0,
                                      T=T, call=False, rel_tol=1e-05, implied_vol=False, r=r, nodes=nodes,
                                      weights=weights))

    for N_time in [256, 512, 1024, 2048]:
        for d in [6]:
            if N_time >= 2048:
                tic = time.perf_counter()
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=4, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=True, verbose=verbose)
                print(time.perf_counter() - tic)
                print('Euler, 4 dates, ', N, N_time, d, 100 * est, 100 * stat)
            if N_time >= 1024:
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=4, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=False, verbose=verbose)
                print('Weak, 4 dates, ', N, N_time, d, 100 * est, 100 * stat)
            if N_time >= 1024:
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=16, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=True, verbose=verbose)
                print('Euler, 16 dates, ', N, N_time, d, 100 * est, 100 * stat)
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=16, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=False, verbose=verbose)
                print('Weak, 16 dates, ', N, N_time, d, 100 * est, 100 * stat)
            if N_time >= 256:
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=256, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=True, verbose=verbose)
                print('Euler, 256 dates, ', N, N_time, d, 100 * est, 100 * stat)
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=256, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=False, verbose=verbose)
                print('Weak, 256 dates, ', N, N_time, d, 100 * est, 100 * stat)
'''
    for N_time in [1024]:
        est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                     S_0=S_0, T=T, payoff='put', r=r, m=1_000_000, N_time=N_time,
                                                     N_dates=256, feature_degree=8, qmc=True, nodes=nodes,
                                                     weights=weights, euler=True)
        print('Euler, 256 dates, ', N_time, 100 * est, 100 * stat)
        est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                     S_0=S_0, T=T, payoff='put', r=r, m=1_000_000, N_time=N_time,
                                                     N_dates=256, feature_degree=8, qmc=True, nodes=nodes,
                                                     weights=weights, euler=False)
        print('Weak, 256 dates, ', N_time, 100 * est, 100 * stat)
        '''
print('Finished')
time.sleep(360000)

# smile_errors_weak_Euler_QE(params=params, N=3, N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
# surface_errors_weak_Euler_QE(params=params, N=3, N_times=np.array([16, 32, 64, 128, 256]))
# asian_errors_weak_Euler_QE(params=params, N=3, N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]))
# print('Finished')
# time.sleep(360000)
'''
filename = 'rHeston sample paths 4 dim BL2 euler antithetic 1024 time steps, H=0.1, lambda=0.3, rho=-0.7, nu=0.3, theta=0.02, r = 0.06, V_0=0.02, T=1.0'
arr = np.empty((6, 1000000, 257))
for i in range(10):
    print(i)
    batch = np.load(filename + f', batch {i + 1}.npy')
    arr[:, i * 50000:(i + 1) * 50000, :] = batch[:, :50000, ::4]
    arr[:, 500000 + i * 50000:500000 + (i + 1) * 50000, :] = batch[:, 50000:, ::4]
np.save(filename + '.npy', arr)
print('Finished')
time.sleep(3600000)

for N_time in np.array([1024]):
    for N in np.array([4]):
        for i in range(5, 10):
            print(N_time, N, i, 'weak')
            f, r = compute_rHeston_samples(params=params, Ns=np.array([N]), N_times=np.array([N_time]),
                                    modes=['BL2'], m=100_000, euler=False, sample_paths=True)
            f = f.replace('.npy', '')
            f += f', batch {i + 1}.npy'
            np.save(f, r)
            print(N, i, 'euler')
            f, r = compute_rHeston_samples(params=params, Ns=np.array([N]), N_times=np.array([N_time]),
                                    modes=['BL2'], m=100_000, euler=True, sample_paths=True)
            f = f.replace('.npy', '')
            f += f', batch {i + 1}.npy'
            np.save(f, r)
print('Finished!')
time.sleep(360000)
'''
'''
for N_time in [2048]:
    print(N_time)
    rHestonQESimulation.samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=1., rho=rho, r=r, m=500_000,
                                N_time=N_time, sample_paths=True, verbose=2)
print('Finished')
time.sleep(3600000)

'''
nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode="BL2")
print(nodes, weights)
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                         rel_tol=rel_tol, verbose=verbose, H=H)
# markov_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
#                                            rel_tol=rel_tol, verbose=verbose, nodes=nodes, weights=weights)

samples = rHestonQESimulation.samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, r=0.,
                                      m=100000, N_time=200, sample_paths=False, verbose=2)
middle, lower, upper = cf.eur_MC(S_0=S_0, K=K, T=T, samples=samples[0, :], r=0., payoff='call', antithetic=False,
                                 implied_vol=True)
plt.plot(k, true_smile, 'k-')
plt.plot(k, middle, 'r-')
plt.plot(k, lower, 'r--')
plt.plot(k, upper, 'r--')
plt.show()

compute_rHeston_samples(params=params, Ns=np.array([2]), N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
                        modes=['BL2'], m=20000000, euler=False, sample_paths=False)

compute_rHeston_samples(params=params, Ns=np.array([2]), N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
                        modes=['BL2'], m=20000000, euler=True, sample_paths=False)

compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
                                  modes=['BL2'], antithetic=[True])
time.sleep(360000)
'''

k = np.linspace(-0.5, 0.5, 51)
K = np.exp(k)
nodes, weights = rk.quadrature_rule(H=H, N=2, T=T, mode='european')

euler_smile, euler_smile_lower, euler_smile_upper = \
    rHestonMarkovSamplePaths.eur(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, nodes=nodes,
                                 weights=weights, T=T, S_0=S_0, N_time=25, m=1000000, euler=True, implied_vol=True)
weak_smile, weak_smile_lower, weak_smile_upper = \
    rHestonMarkovSamplePaths.eur(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, nodes=nodes,
                                 weights=weights, T=T, S_0=S_0, N_time=25, m=1000000, euler=False, implied_vol=True)
'''
'''
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                         rel_tol=rel_tol, verbose=verbose, H=H)
markov_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                           rel_tol=rel_tol, verbose=verbose, nodes=nodes, weights=weights)
samples_ = np.load('rHeston samples 2 dim BL2 euler antithetic 32 time steps, H=0.1, lambda=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1.0.npy')
euler_smile, euler_smile_lower, euler_smile_upper = cf.eur_MC(S_0=S_0, K=K, T=T, r=0., samples=samples_[0, :], payoff='call', antithetic=True,
                     implied_vol=True)
samples_ = np.load('rHeston samples 2 dim BL2 mackevicius antithetic 32 time steps, H=0.1, lambda=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1.0.npy')
weak_smile, weak_smile_lower, weak_smile_upper = cf.eur_MC(S_0=S_0, K=K, T=T, r=0., samples=samples_[0, :], payoff='call', antithetic=True,
                     implied_vol=True)
samples_ = np.load('rHeston samples HQE 32 time steps, H=0.1, lambda=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1.0.npy')
QE_smile, QE_smile_lower, QE_smile_upper = cf.eur_MC(S_0=S_0, K=K, T=T, r=0., samples=samples_[0, :], payoff='call', antithetic=True,
                     implied_vol=True)

plt.plot(k, true_smile, 'k-', label='True smile')
plt.plot(k, markov_smile, 'brown', label='Markov smile')
plt.plot(k, euler_smile, 'b-', label='Euler smile')
plt.plot(k, euler_smile_lower, 'b--')
plt.plot(k, euler_smile_upper, 'b--')
plt.plot(k, weak_smile, 'r-', label='Weak smile')
plt.plot(k, weak_smile_lower, 'r--')
plt.plot(k, weak_smile_upper, 'r--')
plt.plot(k, QE_smile, 'g-', label='QE smile')
plt.plot(k, QE_smile_lower, 'g--')
plt.plot(k, QE_smile_upper, 'g--')
plt.legend(loc='best')
plt.xlabel('Log-moneyness')
plt.ylabel('Implied volatility')
plt.title('Implied volatility for European call options')
plt.show()
time.sleep(36000)
'''
'''
modes = ['GG', 'NGG', 'OL1', 'OLD', 'OL2', 'BL2', 'AE', 'AK']
N = np.arange(1, 11)
H = np.array([0.001, 0.01, 0.1])
# quadrature_rules = np.load('Optimal l1 nodes weights H in 0001 001 01 N from 1 to 10.npy')
lambda_, rho, nu, theta, V_0, S_0, T, rel_tol, verbose = 0.3, -0.7, 0.3, 0.02, 0.02, 1., 0.01, 1e-04, 1
k = np.linspace(-1, 0.5, 301) # * np.sqrt(T)
K = np.exp(k)
# T = np.linspace(0.04, 1., 25)
T = functions.log_linspace(0.004, 1., 25)
# T = 1.


nodes, weights = rk.quadrature_rule(H=0.1, T=1., N=5, mode='AE')
tic = time.perf_counter()
rHestonMarkovSamplePaths.samples(lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=1., m=100000,
                                 N_time=200, nodes=nodes, weights=weights, euler=True)
print(f'Euler, N={N}, time={time.perf_counter() - tic}')
tic = time.perf_counter()
rHestonMarkovSamplePaths.samples(lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=1., m=100000,
                                 N_time=200, nodes=nodes, weights=weights, euler=False)
print(f'Weak, N={N}, time={time.perf_counter() - tic}')
time.sleep(360000)


def plot_samples_volatility_markov():
    nodes = np.array([1., 10.])
    weights = np.array([1., 2.])
    samples = rHestonMarkovSamplePaths.samples(lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=1., m=100000, N_time=1000,
                                               nodes=nodes, weights=weights, vol_only=True, sample_paths=False,
                                               rho=-0.7, S_0=1.)[1:, :]
    # samples = weights[:, None] * (samples - np.array([0.01, 0.])[:, None])
    # samples = weights[:, None] * samples

    plt.scatter(samples[0, :].flatten(), samples[1, :].flatten())
    # plt.scatter(np.array([-0.01 * nodes[1] / np.sum(nodes)]), np.array([-0.01 * nodes[0] / np.sum(nodes)]))
    plt.plot(np.linspace(-1, 1, 100), -np.linspace(-1, 1, 100) / weights[1] * weights[0], 'k-') # - 0.01)
    # plt.plot(np.linspace(-1, 1, 100), (weights[1] - weights[0]) / weights[0] * np.linspace(-1, 1, 100))
    plt.xlabel(r'$V_1$')
    plt.ylabel(r'$V_2$')
    plt.title("Samples of the two-dimensional volatility process")
    plt.show()


def plot_smiles_eur_call_markov_samples():
    H = 0.1
    lambda_, rho, nu, theta, V_0, S_0, T, rel_tol, verbose = 0.3, -0.7, 0.3, 0.02, 0.02, 1., 1., 1e-05, 2
    N = 3
    k = np.linspace(-0.5, 0.5, 101)
    K = np.exp(k)
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode="BL2")

    print(nodes, weights)
    true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                             rel_tol=rel_tol, verbose=verbose, H=H)
    print((true_smile,))
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode="BL2")
    markov_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                               rel_tol=rel_tol, verbose=verbose, nodes=nodes, weights=weights)
    print((markov_smile,))
    print(np.amax(np.abs(true_smile - markov_smile) / true_smile))

    euler_smile, euler_smile_lower, euler_smile_upper = \
        rHestonMarkovSamplePaths.eur(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, nodes=nodes,
                                     weights=weights, T=T, S_0=S_0, N_time=25, m=1000000, euler=True, implied_vol=True)
    weak_smile, weak_smile_lower, weak_smile_upper = \
        rHestonMarkovSamplePaths.eur(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, nodes=nodes,
                                     weights=weights, T=T, S_0=S_0, N_time=25, m=1000000, euler=False, implied_vol=True)
    plt.plot(k, true_smile, 'k-', label='True smile')
    plt.plot(k, markov_smile, 'brown', label='Markov smile')
    plt.plot(k, euler_smile, 'b-', label='Euler smile')
    plt.plot(k, euler_smile_lower, 'b--')
    plt.plot(k, euler_smile_upper, 'b--')
    plt.plot(k, weak_smile, 'r-', label='Weak smile')
    plt.plot(k, weak_smile_lower, 'r--')
    plt.plot(k, weak_smile_upper, 'r--')
    plt.legend(loc='best')
    plt.xlabel('Log-moneyness')
    plt.ylabel('Implied volatility')
    plt.title('Implied volatility for European call options')
    plt.show()
    time.sleep(36000)


print(rHestonMarkovSamplePaths.price_am(K=1.0, lambda_=0., rho=-0.7, nu=0.3, theta=0.0, V_0=0.01, S_0=1., T=1.,
                                        nodes=nodes, weights=weights, payoff='put', r=0.05, m=100000, N_time=120))
time.sleep(36000)
'''
