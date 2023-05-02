import time
import matplotlib.pyplot as plt
import numpy as np
# import Data
import ComputationalFinance
import RoughKernel as rk
# import rBergomi
import fBmMarkov
import functions
import rHestonFourier
import rHestonMarkovSimulation
import rHestonQESimulation
from functions import *
# import rBergomiMarkov
import rHestonMomentMatching
import scipy.stats, scipy.optimize, scipy.integrate, scipy.linalg, scipy.special


print(rHestonFourier.eur_call_put(S_0=100, K=np.array([105, 110, 115]), lambda_=1.4, rho=-0.5711, nu=0.5, theta=0.03, V_0=0.0435, T=1, r=0.03, rel_tol=1e-05, implied_vol=False, call=False, nodes=np.array([0.]), weights=np.array([1.])))
print('Finished')
time.sleep(360000)
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
N = 2
k = np.linspace(-0.1, 0.05, 16)
K = np.exp(k)
T = np.linspace(0, 1, 17)[1:]
T = np.array([0.5, 1.])
# T = np.array([1.])

params = {'S': 1., 'K': K, 'H': H, 'T': T, 'lambda': lambda_, 'rho': rho, 'nu': nu, 'theta': theta, 'V_0': V_0,
          'rel_tol': rel_tol, 'r': 0.}

# smile_errors_weak_Euler_QE(params=params, N=3, N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
# surface_errors_weak_Euler_QE(params=params, N=3, N_times=np.array([16, 32, 64, 128, 256]))
asian_errors_weak_Euler_QE(params=params, N=3, N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]))
print('Finished')
# time.sleep(360000)

compute_rHeston_samples(params=params, Ns=np.array([3]), N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]),
                        modes=['BL2'], m=1_000_000, euler=False, sample_paths=True)
compute_rHeston_samples(params=params, Ns=np.array([3]), N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]),
                        modes=['BL2'], m=1_000_000, euler=True, sample_paths=True)
print('Finished!')
time.sleep(360000)
for N_time in [512]:
    print(N_time)
    rHestonQESamplePaths.samples_QE(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=1., rho=rho, r=0., m=1_000_000,
                                    N_time=N_time, sample_paths=True, verbose=2)
print('Finished')
time.sleep(360000)

nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode="BL2")
print(nodes, weights)
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                         rel_tol=rel_tol, verbose=verbose, H=H)
# markov_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
#                                            rel_tol=rel_tol, verbose=verbose, nodes=nodes, weights=weights)

samples = rHestonQESamplePaths.samples_QE(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, r=0.,
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


k = np.linspace(-0.5, 0.5, 51)
K = np.exp(k)
nodes, weights = rk.quadrature_rule(H=H, N=2, T=T, mode='european')
'''
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
