from functions import *


if __name__ == '__main__':
    kernel_errors_parallelized_testing()

'''
compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['paper', 'optimized', 'european'], vol_behaviours=['correct ninomiya victoir'], recompute=False)
# print('Finished')
# time.sleep(360000)
print('Finished')
# time.sleep(360000)
k = np.linspace(-1.5, 0.75, 451)[220:-70]
true_smile = Data.true_iv_surface_eur_call[-1, 220:-70]
params = {'K': np.exp(k)}
# optimize_kernel_approximation_for_simulation_vector_inputs(params=params, Ns=np.array([2]), N_times=None, vol_behaviours=['sticky'], true_smile=true_smile, plot=True, recompute=False)
# compute_strong_discretization_errors(Ns=np.array([2]), N_times=2 ** np.arange(14), N_time_ref=2 ** 14, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(16), modes=None, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=None, vol_behaviours=['sticky', 'adaptive'], plot=True, true_smile=true_smile)
compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european'], vol_behaviours=['ninomiya victoir', 'correct ninomiya victoir', 'sticky'], plot='vol_behaviour', true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european', 'fitted'], vol_behaviours=['hyperplane reset'], plot='mode', true_smile=true_smile)
'''