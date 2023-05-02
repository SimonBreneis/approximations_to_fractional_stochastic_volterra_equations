import time
import numpy as np
import scipy.interpolate, scipy.special
import ComputationalFinance as cf
import scipy.stats
import RoughKernel as rk
import os
import rHestonQE


def samples_QE(H, lambda_, nu, theta, V_0, T, rho=0., S_0=1., r=0., m=1000, N_time=1000, sample_paths=False,
               antithetic=False, verbose=0):
    """
    Simulates (and saves, or loads) sample paths under the QE scheme of the rough Heston model by Jim Gatheral.
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps
    :param S_0: Initial stock price
    :param r: Interest rate
    :param m: Number of samples
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility
    :param antithetic: If True, uses antithetic variates to reduce the MC error. Not implemented. Always uses False
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Numpy array of the final stock prices
    """
    kind = 'sample paths' if sample_paths else 'samples'
    mode = 'HQE'
    antith = ' antithetic' if antithetic else ''
    if isinstance(T, np.ndarray) and len(T) > 1:
        T_string = f'T=({np.amin(T):.4}, {np.amax(T):.4}, {len(T)})'
    elif isinstance(T, np.ndarray):
        T_string = f'T={T[0]:.4}'
    else:
        T_string = f'T={T:.4}'
    file_name = f'rHeston {kind} {mode}{antith} {N_time} time steps, H={H:.3}, lambda={lambda_:.3}, rho={rho:.3}, ' \
                f'nu={nu:.3}, theta={theta:.3}, V_0={V_0:.3}, {T_string}.npy'
    if os.path.exists(file_name):
        result = np.load(file_name)
    else:
        print("File " + file_name + " does not exist!")
        tic = time.perf_counter()
        kernel = rk.kernel_rheston(H=H, lam=lambda_, zeta=nu)
        # Compute the forward variance curve. For faster use in the algorithm, we pre-
        # compute and do linear interpolation.
        n_inter = 100  # number of points for interpolation
        x = np.linspace(0.0, T, n_inter)
        xi_val = kernel.xi(x, V_0, lambda_, theta / lambda_)
        xi = scipy.interpolate.interp1d(x, xi_val, kind='linear')
        if verbose >= 1:
            print(f'Computing the forward variance curve took {time.perf_counter() - tic} seconds.')

        tic_2 = time.perf_counter()
        num_batch = 100
        eps = 1e-06
        X, V = rHestonQE.hqe_batch(xi=xi, kernel=kernel, rho=rho, X0=np.log(S_0), t_final=T, N=N_time, M=m,
                                   num_batch=num_batch, eps=eps, sample_paths=sample_paths, r=r, verbose=verbose - 1)
        result = np.empty((2,) + X.shape)
        result[0, ...] = np.exp(X)
        result[1, ...] = V
        np.save(file_name, result)
        print(f'Simulating the sample paths took {time.perf_counter() - tic_2} seconds.')
        print(f'Computing the forward variance curve and simulating the samples took {time.perf_counter() - tic} '
              f'seconds.')
    return result

