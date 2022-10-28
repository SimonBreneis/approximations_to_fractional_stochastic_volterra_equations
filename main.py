import time
import matplotlib.pyplot as plt
import numpy as np
import Data
import rBergomi
from functions import *
import rBergomiMarkov
import rHestonMomentMatching
from scipy.stats import norm
import scipy.special


for N_times in 2 ** np.arange(11):

    tic = time.perf_counter()
    compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=N_times, modes=['european'], vol_behaviours=['sticky'], recompute=True, m=20000000)
    print(time.perf_counter() - tic)
    tic = time.perf_counter()
    compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=N_times, modes=['european'], vol_behaviours=['mackevicius random'], recompute=True, m=20000000)
    print(time.perf_counter() - tic)
    tic = time.perf_counter()
    compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=N_times, modes=['european'], vol_behaviours=['mackevicius sequential'], recompute=True, m=20000000)
    print(time.perf_counter() - tic)
    tic = time.perf_counter()
    compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=N_times, modes=['european'], vol_behaviours=['mackevicius sequential antithetic'], recompute=True, m=20000000)
    print(time.perf_counter() - tic)

print('Finished')
# time.sleep(360000)

# print(rk.quadrature_rule(0.1, 2, 1))
k = np.sqrt(1) * np.linspace(-1.5, 0.75, 451)[220:-70:5]# [280:-140:10]# [220:-70:5]
params = {'K': np.exp(k), 'T': 1.}
params = rHeston_params(params)
true_smile = Data.true_iv_surface_eur_call[-1, 220:-70:5]
print(k, len(k))

# simulation_errors_depending_on_node_size(params=params, verbose=1, true_smile=true_smile, N_times=2**np.arange(4, 10), largest_nodes=np.linspace(0, 10, 101)/0.04, vol_behaviour='sticky')
# optimize_kernel_approximation_for_simulation_vector_inputs(Ns=np.array([1]), N_times=2 ** np.arange(6, 9), params=params, true_smile=true_smile, plot=True, recompute=True, vol_behaviours=['hyperplane reset'], m=10000000)

# compute_strong_discretization_errors(Ns=np.array([2]), N_times=2 ** np.arange(14), N_time_ref=2 ** 14, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(16), modes=None, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True, true_smile=true_smile)
compute_smiles_given_stock_prices(params=params, Ns=np.array([3]), N_times=2 ** np.arange(11), modes=['european'], vol_behaviours=['sticky', 'mackevicius random', 'mackevicius sequential', 'mackevicius sequential antithetic'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european'], vol_behaviours=['ninomiya victoir', 'correct ninomiya victoir', 'sticky'], plot='vol_behaviour', true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european', 'fitted'], vol_behaviours=['hyperplane reset'], plot='mode', true_smile=true_smile)
print('Finished')

