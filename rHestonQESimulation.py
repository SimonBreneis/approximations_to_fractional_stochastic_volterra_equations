import time
import numpy as np
import scipy.interpolate, scipy.special
import ComputationalFinance as cf
import scipy.stats
import RoughKernel as rk
import os
import rHestonQE


def samples(H, lambda_, nu, theta, V_0, T, rho=0., S_0=1., r=0., m=1000, N_time=1000, sample_paths=False,
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


def am_features(x, degree=6, K=0.):
    n_samples = x.shape[-1]
    normalized_stock = ((x[0, :] - K) / K) if np.abs(K) > 0.01 else x[0, :]
    vol = x[1, :]
    dim = degree + degree // 2
    feat = np.empty((n_samples, dim))
    if degree >= 1:
        feat[:, 0] = normalized_stock
    if degree >= 2:
        feat[:, 1] = normalized_stock ** 2
        feat[:, degree] = vol
    if degree >= 3:
        for i in range(3, degree + 1):
            feat[:, i - 1] = feat[:, i - 2] * normalized_stock
            if i % 2 == 0:
                feat[:, degree + i // 2 - 1] = feat[:, degree + i // 2 - 2] * vol
    return feat


def price_am(K, H, lambda_, rho, nu, theta, V_0, S_0, T, payoff, r=0., m=1000000, N_time=200, N_dates=12,
             feature_degree=6, antithetic=True):
    """
    Gives the price of an American option using the QE scheme.
    :param K: Strike price
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param payoff: The payoff function, either 'call' or 'put' or a function taking as inputs S (samples) and K (strike)
    :param r: Interest rate
    :param m: Number of samples. Uses half of them for fitting the stopping rule, and half of them for pricing
    :param N_time: Number of time steps used in the simulation
    :param N_dates: Number of exercise dates. If None, N_dates = N_time
    :param feature_degree: The degree of the polynomial features used in the regression
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    return: The prices of the call option for the various strike prices in K
    """
    if payoff == 'call':
        def payoff(S):
            return cf.payoff_call(S=S, K=K)
    elif payoff == 'put':
        def payoff(S):
            return cf.payoff_put(S=S, K=K)
    if N_dates is None:
        N_dates = N_time

    def features(x):
        return am_features(x=x, degree=feature_degree, K=K)

    samples_ = samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0, r=r, m=m,
                       N_time=N_time, sample_paths=True, antithetic=antithetic)
    samples_ = samples_[:, :, ::N_time // N_dates]
    samples_[1, :, :] = samples_[1, :, :] - samples_[1, :, :1]
    samples_1 = np.empty((2, m // 2, N_dates + 1))
    samples_1[:, :m // 4, :] = samples_[:, :m // 4, :]
    samples_1[:, m // 4:, :] = samples_[:, m // 2:3 * m // 4, :]
    samples_2 = np.empty((2, m // 2, N_dates + 1))
    samples_2[:, :m // 4, :] = samples_[:, m // 4:m // 2, :]
    samples_2[:, m // 4:, :] = samples_[:, 3 * m // 4:, :]
    (biased_est, biased_stat), models = cf.price_am(T=T, r=r, samples=samples_1, antithetic=antithetic, payoff=payoff,
                                                    features=features)
    est, stat = cf.price_am_forward(T=T, r=r, samples=samples_2, payoff=payoff, models=models, features=features,
                                    antithetic=antithetic)
    return est, stat, biased_est, biased_stat, models, features
