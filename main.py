import Data
import time

import rHeston
from functions import *
import numpy as np
import multiprocessing as mp

'''
if __name__ == '__main__':
    num_threads = 10
    num_runs = 4949

    t = time.time()
    with mp.Pool(processes=num_threads) as pool:
        print("\nMulti threaded map.")
        # print(f"Prices = {pool.map(nasty_parallelization, range(num_runs))}")
        result = pool.map(nasty_parallelization, range(num_runs))
        print(f"Run time = {(time.time() - t) / num_runs:.2f}s per run.")
    print(result.__class__)
    result = np.asarray(result)
    print(result.__class__)
    print(result.shape)
    largest_nodes = result[:, 0, :, :]
    kernel_errors = result[:, 1, :, :]
    durations = result[:, 2, :, :]
    largest_nodes_paper = largest_nodes[:, 0, :]
    largest_nodes_optimized_old = largest_nodes[:, 1, :]
    largest_nodes_optimized = largest_nodes[:, 2, :]
    kernel_errors_paper = kernel_errors[:, 0, :]
    kernel_errors_optimized_old = kernel_errors[:, 1, :]
    kernel_errors_optimized = kernel_errors[:, 2, :]
    durations_paper = durations[:, 0, :]
    durations_optimized_old = durations[:, 1, :]
    durations_optimized = durations[:, 2, :]
    Hs =



'''
# compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['paper', 'optimized', 'european'], vol_behaviours=['adaptive'], recompute=True)
# print('Finished')
# time.sleep(360000)
rHestonMarkov.iv_eur_call(S=1., K=np.array([1.]), H=0.1, lambda_=1.0, rho=-0.7, nu=0.8, theta=0.04, V_0=0.04, T=1.0, rel_tol=1e-5, N=2)
print('Finished')
time.sleep(360000)
k = np.linspace(-1.5, 0.75, 451)[220:-70]
true_smile = Data.true_iv_surface_eur_call[-1, 220:-70]
params = {'K': np.exp(k)}
# optimize_kernel_approximation_for_simulation_vector_inputs(params=params, Ns=np.array([2]), N_times=None, vol_behaviours=['sticky'], true_smile=true_smile, plot=True, recompute=False)
compute_strong_discretization_errors(Ns=np.array([2]), N_times=2 ** np.arange(14), N_time_ref=2 ** 14, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(16), modes=None, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=None, vol_behaviours=['sticky', 'adaptive'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([1]), N_times=2 ** np.arange(14), modes=['optimized'], vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot='vol_behaviour', true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european', 'fitted'], vol_behaviours=['hyperplane reset'], plot='mode', true_smile=true_smile)
