import sys
import time
import numpy as np
import Data
import rBergomi
from functions import *
import rBergomiMarkov


T = 1.
k = np.linspace(-0.5, 0.5, 301) * np.sqrt(T)
M = 100000
N_time = 2000

tic = time.perf_counter()
smile = rBergomi.implied_volatility(K=np.exp(k), rel_tol=1e-01, T=T, verbose=1)
print(time.perf_counter() - tic)
plt.plot(k, smile, 'k-')
plt.show()

tic = time.perf_counter()
smile, l, u = rBergomiMarkov.implied_volatility(K=np.exp(k), mode='paper', N=3, M=M, N_time=N_time)
plt.plot(k, smile, 'r-')
print(time.perf_counter() - tic)
tic = time.perf_counter()
smile, l, u = rBergomiMarkov.implied_volatility(K=np.exp(k), mode='optimized', N=3, M=M, N_time=N_time)
plt.plot(k, smile, 'g-')
print(time.perf_counter() - tic)
tic = time.perf_counter()
smile, l, u = rBergomiMarkov.implied_volatility(K=np.exp(k), mode='european', N=3, M=M, N_time=N_time)
plt.plot(k, smile, 'b-')
print(time.perf_counter() - tic)

plt.show()

'''
params = {'H': 0.05, 'lambda': 0.2, 'rho': -0.6, 'nu': 0.6,
          'theta': 0.01, 'V_0': 0.01, 'S': 1., 'K': np.exp(np.linspace(-1, 0.5, 301)),
          'T': np.linspace(0.04, 1., 25), 'rel_tol': 1e-05}
print(params)
rHeston_iv_eur_call(params=params, load=True, save=True, verbose=1)
print('Finished!')
time.sleep(3600000)

tic = time.perf_counter()
print(rHeston_iv_eur_call(params=rHeston_params('simple'), load=False, save=False, verbose=1))
print(time.perf_counter() - tic)
print('finished')
time.sleep(36000)
'''

'''tic = time.perf_counter()
params = rHeston_params('simple')
rHeston_iv_eur_call(params)
print('time', time.perf_counter()-tic)
time.sleep(36000)
'''

# rHeston_iv_eur_call(params=rHeston_params('simple'), load=False, save=False, verbose=1)

if __name__ == '__main__':
    # 'nu': log_linspace(0.2, 0.6, 2),
    # 'theta': log_linspace(0.01, 0.03, 2),
    # 'V_0': log_linspace(0.01, 0.03, 2)
    # 'lambda': np.array([0.2, 1.0])
    # 'rho': np.array([-0.6, -0.8]),
    '''
    params = {'H': np.array([0.15]), 'lambda': np.array([0.2, 1.0]), 'rho': np.array([-0.6, -0.8]),
              'nu': np.array([0.2]), 'theta': np.array([0.01, 0.03]), 'V_0': np.array([0.01, 0.03])}
    for i in range(25):
        params['T'] = (i+1)/25
        params['K'] = np.exp(np.linspace(-1., 0.5, 301) * np.sqrt(params['T']))
        # print(params)
        rHestonMarkov_iv_eur_call_parallelized(params=params, Ns=np.arange(1, 11), modes=['paper', 'optimized', 'european'], num_threads=1, verbose=1)

    '''
    params = {'H': np.array([0.05]), 'lambda': np.array([0.2, 1.0]), 'rho': np.array([-0.6, -0.8]),
              'nu': np.array([0.2]), 'theta': np.array([0.01, 0.03]), 'V_0': np.array([0.01, 0.03]),
              'T': np.linspace(0.04, 1., 25), 'K': np.exp(np.linspace(-1., 0.5, 301))}

    rHestonMarkov_iv_eur_call_parallelized(params=params, Ns=np.arange(1, 11),
                                           modes=['paper', 'optimized', 'european'], num_threads=1, verbose=1)


# compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['paper', 'optimized', 'european'], vol_behaviours=['correct ninomiya victoir'], recompute=False)
# print('Finished')
# time.sleep(360000)
# print('Finished')
# time.sleep(360000)

'''
k = np.sqrt(0.04) * np.linspace(-1.5, 0.75, 451)[220:-70]
true_smile = Data.true_iv_surface_eur_call[0, 220:-70]
params = {'K': np.exp(k), 'T': 0.04}
params = rHeston_params(params)
print(k, len(k))
# simulation_errors_depending_on_node_size(params=params, verbose=1, true_smile=true_smile, N_times=2**np.arange(2, 5), largest_nodes=np.linspace(0, 10, 21)/0.04, vol_behaviour='correct ninomiya victoir')
optimize_kernel_approximation_for_simulation_vector_inputs(N_times=2 ** np.arange(10, 11), params=params, true_smile=true_smile, plot=True, recompute=True, vol_behaviours=['correct ninomiya victoir'])
'''
# compute_strong_discretization_errors(Ns=np.array([2]), N_times=2 ** np.arange(14), N_time_ref=2 ** 14, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(16), modes=None, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=None, vol_behaviours=['sticky', 'adaptive'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european'], vol_behaviours=['ninomiya victoir', 'correct ninomiya victoir', 'sticky'], plot='vol_behaviour', true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european', 'fitted'], vol_behaviours=['hyperplane reset'], plot='mode', true_smile=true_smile)
print('Finished')
