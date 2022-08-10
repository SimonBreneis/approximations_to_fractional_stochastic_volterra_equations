import time
import numpy as np

from functions import *

'''tic = time.perf_counter()
params = rHeston_params('simple')
rHeston_iv_eur_call(params)
print('time', time.perf_counter()-tic)
time.sleep(36000)
'''

array = np.random.uniform(0, 1, 120).reshape((5, 2, 3, 4))
print(np.amax(array, axis=3) == array[argmax_indices(array, axis=3)])
time.sleep(3600)
ind = np.argmax(array, axis=0)
print(ind)
print(array)
print(np.amax(array, axis=0))
indexing = (ind,)
for i in range(len(array.shape)-1):
    subindexing = ()
    for j in range(len(array.shape)-1):
        if i == j:
            subindexing = subindexing + (slice(None),)
        else:
            subindexing = subindexing + (None,)
    indi = np.arange(array.shape[i+1])
    indi = indi[subindexing]
    indexing = indexing + (indi,)
print(indexing)
print(array[indexing])
time.sleep(3600)


if __name__ == '__main__':
    # 'nu': log_linspace(0.2, 0.6, 2),
    # 'theta': log_linspace(0.01, 0.03, 2),
    # 'V_0': log_linspace(0.01, 0.03, 2)
    params = {'H': np.array([0.05]), 'lambda': np.array([1.0]), 'rho': np.array([-0.6]), 'nu': np.array([0.6]), 'theta': np.array([0.01]), 'V_0': np.array([0.01])}
    print(params)
    rHeston_iv_eur_call_parallelized(params=params, num_threads=1)
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
