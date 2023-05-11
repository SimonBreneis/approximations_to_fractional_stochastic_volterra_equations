import time
import numpy as np
from scipy.interpolate import interp1d
import scipy.interpolate, scipy.special
import ComputationalFinance as cf
import scipy.stats
import RoughKernel as rk
import os
import rHestonQE
import rHestonMarkovSimulation
from scipy.stats.qmc import Sobol
import psutil


def samples(H, lambda_, nu, theta, V_0, T, rho, S_0, r, m, N_time, sample_paths=False, qmc=True, rng=None,
            return_times=None, rv_shift=False, xi=None, b_star=None, beta=None, verbose=0):
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
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param verbose: Determines the number of intermediary results printed to the console
    :return: The simulated sample paths, and the rng that was used for generating the underlying random variables
    """
    if sample_paths is False:
        return_times = 1
    if return_times is None:
        return_times = N_time
    if N_time % return_times != 0:
        raise ValueError(f'The number of time steps for the simulation N_time={N_time} is not divisible by the number'
                         f'of time steps that should be returned return_times={return_times}.')
    saving_steps = N_time // return_times
    dt = T / N_time
    if rng is None:
        if qmc:
            rng = Sobol(d=3 * N_time, scramble=False)
        else:
            rng = np.random.default_rng()
    if isinstance(rv_shift, bool) and rv_shift:
        rv_shift = np.random.uniform(0, 1, 3 * N_time)
    eps = 1e-06
    kernel = rk.kernel_rheston(H=H, lam=lambda_, zeta=nu)
    if xi is None:
        # Compute the forward variance curve. For faster use in the algorithm, we precompute and do linear
        # interpolation.
        n_inter = 100  # number of points for interpolation
        x = np.linspace(0.0, T, n_inter)
        xi_val = kernel.xi(x, V_0, lambda_, theta / lambda_)
        xi = interp1d(x, xi_val, kind='linear')
    if b_star is None:
        b_star = rHestonQE.b_star(kernel, dt, N_time)
    if beta is None:
        beta = kernel.K_0(dt) / dt
    gamma = (b_star[0] ** 2 - beta ** 2) * dt
    assert gamma > 0.0, f"gamma fails positivity, gamma = {gamma}."

    m_input = m  # the original input of how many samples should be simulated
    m_return = m  # the final number of samples that we will actually return
    # m itself is the number of samples that we simulate
    # We always have m >= m_return >= m_input

    if qmc:
        m = int(2 ** np.ceil(np.log2(m)) + 0.001)
        m_return = m
        if m != m_input:
            print(f'Using QMC requires simulating a number m of samples that is a power of 2. The input m={m_input} '
                  f'is not a power of 2. Simulates m={m} samples instead.')

    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = 4 * np.sqrt(return_times + 1) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)
    if necessary_memory > available_memory:
        raise MemoryError(f'Not enough memory to store the sample paths of the rough Heston model with {return_times} '
                          f'time points where the sample paths should be '
                          f'returned and {m} sample paths. Roughly {necessary_memory}**2 bytes needed, '
                          f'while only {available_memory}**2 bytes are available.')

    available_memory_for_random_variables = available_memory / 3
    necessary_memory_for_random_variables = np.sqrt(3 * N_time) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)
    n_batches = int(np.ceil(necessary_memory_for_random_variables / available_memory_for_random_variables))
    m_batch = int(np.ceil(m_return / n_batches))
    m = m_batch * n_batches

    if sample_paths:
        result = np.empty((2, m, return_times + 1))
        result[0, :, 0] = np.log(S_0)
        result[1, :, 0] = xi(0)
    else:
        result = np.empty((2, m))

    for j in range(n_batches):
        if verbose >= 1:
            print(f'Simulating batch {j + 1} of {n_batches}.')
        current_V = np.full(m_batch, xi(0))
        current_X = np.full(m_batch, np.log(S_0))
        chi = np.zeros((m_batch, N_time))
        if qmc:
            rv = rng.random(m_batch)
        else:
            rv = rng.uniform(0, 1, (m_batch, 3 * N_time))
        if isinstance(rv_shift, np.ndarray):
            rv = (rv + rv_shift) % 1.
        index = rv == 0
        rv[index] = 0.5
        index = rv == 1
        rv[index] = 0.5

        for i in range(N_time):
            current_X, current_V, chi[:, i] = rHestonQE.hqe_step(xi=xi, kernel=kernel, rho=rho, dt=dt, eps=eps,
                                                                 b_star_=b_star, v_old=current_V, chi_old=chi[:, :i],
                                                                 X_old=current_X, rv=rv[:, 3 * i:3 * (i + 1)],
                                                                 beta=beta, gamma=gamma, r=r)
            if sample_paths and (i + 1) % saving_steps == 0:
                result[1, j * m_batch:(j + 1) * m_batch, (i + 1) // saving_steps] = current_V
                result[0, j * m_batch:(j + 1) * m_batch, (i + 1) // saving_steps] = current_X
        if not sample_paths:
            result[1, j * m_batch:(j + 1) * m_batch] = current_V
            result[0, j * m_batch:(j + 1) * m_batch] = current_X
    result = result[:, :m_return, ...]  # discard those values which were unnecessarily generated
    result[0, ...] = np.exp(result[0, ...])
    return result, rng, xi, b_star, beta


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


def am_features(x, degree=6, K=0.):
    n_samples = x.shape[-1]
    normalized_stock = ((x[0, :] - K) / K) if np.abs(K) > 0.01 else x[0, :]
    vol = x[1, :]
    feat = np.empty((n_samples, 1 if degree == 1 else 3))
    if degree >= 1:
        feat[:, 0] = normalized_stock
    if degree >= 2:
        feat[:, 1] = normalized_stock ** 2
        feat[:, 2] = vol
    current_N = 3
    current_ind = 3
    lower_N_stock = 1
    upper_N_stock = 3
    lower_N_vol = 3
    upper_N_vol = 3
    next_lower_N_vol = 1
    next_upper_N_vol = 3
    while current_N <= degree:
        feat_new = np.empty((n_samples, feat.shape[1] + upper_N_stock - lower_N_stock + upper_N_vol - lower_N_vol))
        feat_new[:, :current_ind] = feat
        feat = feat_new
        next_ind = current_ind + upper_N_stock - lower_N_stock
        feat[:, current_ind:next_ind] = normalized_stock[:, None] * feat[:, lower_N_stock:upper_N_stock]
        lower_N_stock = current_ind
        current_ind = next_ind
        next_ind = current_ind + upper_N_vol - lower_N_vol
        feat[:, current_ind:next_ind] = vol[:, None] * feat[:, lower_N_vol:upper_N_vol]
        lower_N_vol = next_lower_N_vol
        next_lower_N_vol = current_ind
        current_ind = next_ind
        upper_N_stock = current_ind
        upper_N_vol = next_upper_N_vol
        next_upper_N_vol = current_ind
        current_N = current_N + 1
    return feat


def price_am(K, H, lambda_, rho, nu, theta, V_0, S_0, T, payoff, r=0., m=1000000, N_time=200, N_dates=12,
             feature_degree=6, qmc=True, qmc_error_estimators=25, verbose=0):
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
        return rHestonMarkovSimulation.am_features(x=x, degree=feature_degree, K=K)
        # return am_features(x=x, degree=feature_degree, K=K)

    def get_samples(rng_=None, rv_shift=False, xi_=None, b_star_=None, beta_=None):
        samples_, rng_, xi_, b_star_, beta_ = \
            samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0, r=r, m=m,
                    N_time=N_time, sample_paths=True, qmc=qmc, rng=rng_, return_times=N_dates, rv_shift=rv_shift,
                    xi=xi_, b_star=b_star_, beta=beta_, verbose=2)
        samples_[1, :, :] = samples_[1, :, :] - samples_[1, :, :1]
        return samples_, rng_, xi_, b_star_, beta_

    if qmc:
        rng, xi, b_star, beta = None, None, None, None
        estimates = np.empty(qmc_error_estimators)
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            samples_1, rng, xi, b_star, beta = get_samples(rng, False if i == 0 else True, xi, b_star, beta)
            rng.reset()
            rng.fast_forward(m)
            (biased_est, biased_stat), models = cf.price_am(T=T, r=r, samples=samples_1, payoff=payoff,
                                                            features=features)
            samples_1, rng, xi, b_star, beta = get_samples(rng, False if i == 0 else True, xi, b_star, beta)
            rng.reset()
            estimates[i], _ = cf.price_am_forward(T=T, r=r, samples=samples_1, payoff=payoff, models=models,
                                                  features=features)
        est, stat = cf.MC(estimates)
    else:
        samples_1, rng, xi, b_star, beta = get_samples()
        (biased_est, biased_stat), models = cf.price_am(T=T, r=r, samples=samples_1[:, :m, :], payoff=payoff,
                                                        features=features)
        est, stat = cf.price_am_forward(T=T, r=r, samples=samples_1[:, :m, :], payoff=payoff, models=models,
                                        features=features)

    '''
    samples_ = np.load('rHeston sample paths HQE 2048 time steps, H=0.1, lambda=0.3, rho=-0.7, nu=0.3, theta=0.02, r=0.06, V_0=0.02, T=1.0, batch 1.npy')
    samples_ = samples_[:, :, ::(samples_.shape[-1] - 1) // N_dates]
    samples_[1, :, :] = samples_[1, :, :] - samples_[1, :, :1]
    (biased_est, biased_stat), models = cf.price_am(T=T, r=r, samples=samples_, antithetic=False, payoff=payoff,
                                                    features=features)
    samples_ = np.load(
        'rHeston sample paths HQE 2048 time steps, H=0.1, lambda=0.3, rho=-0.7, nu=0.3, theta=0.02, r=0.06, V_0=0.02, T=1.0, batch 2.npy')
    samples_ = samples_[:, :, ::(samples_.shape[-1] - 1) // N_dates]
    samples_[1, :, :] = samples_[1, :, :] - samples_[1, :, :1]
    est, stat = cf.price_am_forward(T=T, r=r, samples=samples_, payoff=payoff, models=models, features=features,
                                    antithetic=False)
    '''
    return est, stat
