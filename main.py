import Data
import time
from functions import *


compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=None, modes=None, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'])
print('Finished')
time.sleep(360000)

k = np.linspace(-1.5, 0.75, 451)[220:-90]
true_smile = Data.true_iv_surface_eur_call[-1, 220:-90]
params = {'K': np.exp(k)}
print(optimize_kernel_approximation_for_simulation(params=params, N_time=64, true_smile=true_smile))
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(11), modes=None, vol_behaviours=['ninomiya victoir'], plot=True, true_smile=true_smile)
