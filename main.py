import time
import numpy as np

from functions import *

'''tic = time.perf_counter()
params = rHeston_params('simple')
rHeston_iv_eur_call(params)
print('time', time.perf_counter()-tic)
time.sleep(36000)
'''
'''
if __name__ == '__main__':
    # 'nu': log_linspace(0.2, 0.6, 2),
    # 'theta': log_linspace(0.01, 0.03, 2),
    # 'V_0': log_linspace(0.01, 0.03, 2)
    params = {'nu': np.array([0.6]), 'theta': np.array([0.03]), 'V_0': np.array([0.03])}
    print(params)
    rHeston_iv_eur_call_parallelized(params=params, num_threads=15)
'''
if __name__ == '__main__':
    kernel_errors(H=0.1, T=1., Ns=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), modes=['paper', 'optimized', 'european'], verbose=1)
    print('finished')
    time.sleep(36000)
# compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['paper', 'optimized', 'european'], vol_behaviours=['correct ninomiya victoir'], recompute=False)
# print('Finished')
# time.sleep(360000)
# print('Finished')
# time.sleep(360000)
# k = np.linspace(-1.5, 0.75, 451)[220:-70]
# true_smile = Data.true_iv_surface_eur_call[-1, 220:-70]
# params = {'K': np.exp(k)}
# optimize_kernel_approximation_for_simulation_vector_inputs(params=params, Ns=np.array([2]), N_times=None, vol_behaviours=['sticky'], true_smile=true_smile, plot=True, recompute=False)
# compute_strong_discretization_errors(Ns=np.array([2]), N_times=2 ** np.arange(14), N_time_ref=2 ** 14, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(16), modes=None, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=None, vol_behaviours=['sticky', 'adaptive'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european'], vol_behaviours=['ninomiya victoir', 'correct ninomiya victoir', 'sticky'], plot='vol_behaviour', true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european', 'fitted'], vol_behaviours=['hyperplane reset'], plot='mode', true_smile=true_smile)
