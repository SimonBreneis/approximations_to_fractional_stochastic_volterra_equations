import numpy as np
from scipy.interpolate import interp1d
import ComputationalFinance as cf
import RoughKernel as rk
import rHestonMarkovSimulation
from scipy.stats.qmc import Sobol
import psutil
from qe import psi_QE
from scipy.special import ndtri


def discrete_convolution(b, chi):
    """
    Discrete convolution of b and chi.

    Parameters
    ----------
    b : numpy array
        Numpy array of dimension n.
    chi : numpy array
        Numpy array of dimension (m,n).

    Returns
    -------
    dc : numpy array
        numpy array of dimension m. dc[j] = sum_k b[n-1-k] * chi[j,k]

    """
    m, n = chi.shape
    if n > 0:
        b_reversed = b[::-1]  # Note that this is a view, not a copy!
        dc = np.sum(b_reversed * chi, axis=1)
    else:
        dc = np.zeros(m)
    return dc


def b_star(kernel, Delta, N):
    b2 = kernel.K_diag(Delta, N) / Delta
    return np.sqrt(np.maximum(b2, 0.0))


def bivariate_QE(xi_hat, v_bar, Delta, beta, gamma, rv):
    """
    Apply QE algorithm twice, as needed for the HQE scheme.
    :param xi_hat:
    :param v_bar:
    :param Delta:
    :param beta:
    :param gamma:
    :param rv: Uniform random variables of twice the size of v_bar
    """
    m = 0.5 * xi_hat  # the mean for both bchi_tilde and epsilon_tilde
    s2_chi = beta ** 2 * v_bar * Delta
    s2_epsilon = v_bar * gamma
    psi_chi = s2_chi / m ** 2
    psi_epsilon = s2_epsilon / m ** 2
    assert np.all(psi_chi > 0.0), \
        f"Failure of positivity. psi_chi = {psi_chi}."
    assert np.all(psi_epsilon > 0.0), \
        f"Failure of positivity. psi_epsilon = {psi_epsilon}."
    bchi_tilde = psi_QE(psi_chi, m, rv[:, 0])
    epsilon_tilde = psi_QE(psi_epsilon, m, rv[:, 1])
    return bchi_tilde, epsilon_tilde


def hqe_step(xi, kernel, rho, dt, eps, b_star_, v_old, chi_old, X_old, rv, beta, gamma, r=0.):
    """
    .
    :param xi:
    :param kernel:
    :param rho:
    :param dt:
    :param eps:
    :param b_star_:
    :param v_old:
    :param chi_old:
    :param X_old:
    :param rv: Uniform random variables of size (m_batch, 3)
    :param beta:
    :param gamma:
    :param r:
    """
    j = chi_old.shape[1] + 1
    xi_hat = np.maximum(eps, xi(j * dt) + discrete_convolution(b_star_[1:j], chi_old))
    assert np.all(xi_hat > 0.0), f"xi_hat fails positivity, xihat = {xi_hat}."
    v_bar = (xi_hat + 2 * kernel.H * v_old) / (2 * kernel.H + 1)
    assert np.all(v_bar > 0.0), f"v_bar fails positivity at {v_bar}."
    bchi_tilde, epsilon_tilde = bivariate_QE(xi_hat, v_bar, dt, beta, gamma, rv[:, :2])
    # bchi_tilde is beta*chi_tilde
    chi_new = (bchi_tilde - 0.5 * xi_hat) / beta
    v_new = bchi_tilde + epsilon_tilde
    index = rv[:, 2] != 0
    rv[index, 2] = ndtri(rv[index, 2])
    v_mean = 0.5 * (v_old + v_new)
    X_new = X_old - 0.5 * v_mean * dt + np.sqrt(1 - rho ** 2) * np.sqrt(v_mean * dt) * rv[:, 2] \
        + rho * chi_new + r * dt
    return X_new, v_new, chi_new


def get_necessary_memory(return_times, m):
    """
    Estimates the (square root of the) number of bytes needed to simulate the required number of paths.
    :param return_times: Number of time steps that should be returned
    :param m: Number of sample paths
    :return: Square root of the number of bytes that are required for storing m paths with return_times time steps
        where the volatility has N dimensions
    """
    return 2.5 * np.sqrt(return_times + 1) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)


def get_n_batches(return_times, m):
    """
    Returns the number of batches that need to be used to simulate m samples.
    :param return_times: Number of time steps that should be returned
    :param m: Number of sample paths
    :return: Number of batches and number of samples per batch
    """
    available_memory = np.sqrt(psutil.virtual_memory().available) / 2
    necessary_memory = get_necessary_memory(return_times=return_times, m=m)
    n_batches = int(np.ceil((necessary_memory / available_memory) ** 2))
    m_batch = int(np.ceil(m / n_batches))
    return n_batches, m_batch


def get_kernel_dict(H, lambda_, nu, V_0, theta, T, N_time, kernel_dict=None):
    dt = float(T / N_time)
    if kernel_dict is None:
        kernel_dict = {}
    if 'kernel' not in kernel_dict:
        kernel_dict['kernel'] = rk.kernel_rheston(H=H, lam=lambda_, zeta=nu)
    if 'xi' not in kernel_dict:
        # Compute the forward variance curve. For faster use in the algorithm, we precompute and do linear
        # interpolation.
        n_inter = 100  # number of points for interpolation
        x = np.linspace(0.0, T, n_inter)
        xi_val = kernel_dict['kernel'].xi(x, V_0, lambda_, theta / lambda_)
        kernel_dict['xi'] = interp1d(x, xi_val, kind='linear')
    if 'b_star' not in kernel_dict:
        kernel_dict['b_star'] = b_star(kernel_dict['kernel'], dt, N_time)
    if 'beta' not in kernel_dict:
        kernel_dict['beta'] = kernel_dict['kernel'].K_0(dt) / dt
    return kernel_dict


def samples(H, lambda_, nu, theta, V_0, T, rho, S_0, r, m, N_time, sample_paths=False, qmc=True, rng=None,
            return_times=None, rv_shift=False, kernel_dict=None, verbose=0):
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
    :param rng: Can specify a sampler to use for sampling the underlying random variables. If qmc is true, expects
        an instance of scipy.stats.qmc.Sobol() with the correctly specified dimension of the simulated random variables.
        If qmc is False, expects an instance of np.random.default_rng()
    :param return_times: Integer that specifies how many time steps are returned. Only relevant if sample_paths is True.
        E.g., N_time is 100 and return_times is 25, then the paths are simulated using 100 equispaced time steps, but
        only the 26 = 25 + 1 values at the times np.linspace(0, T, 26) are returned. May be used especially for storage
        saving reasons, as only these (in this case 26) values are ever stored. The number N_time must be divisible by
        return_times. If return_times is None, it is set to N_time, i.e. we return every time step that was simulated.
    :param rv_shift: Only relevant when using QMC. Can specify a shift by which the uniform random variables in [0,1)^d
        are drawn. When the random variables X are drawn from Sobol, instead uses (X + rv_shift) mod 1. If True,
        randomly generates such a random shift
    :param kernel_dict: A dictionary containing precomputed xi, b_star, and beta. If None, precomputes these quantities
    :param verbose: Determines the number of intermediary results printed to the console
    :return: The simulated sample paths, the rng that was used for generating the underlying random variables, and a
        kernel_dict with the precomputed xi, b_star, beta, and kernel
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
    kernel_dict = get_kernel_dict(H=H, lambda_=lambda_, nu=nu, V_0=V_0, T=T, N_time=N_time, theta=theta,
                                  kernel_dict=kernel_dict)
    kernel, xi, b_star_, beta = kernel_dict['kernel'], kernel_dict['xi'], kernel_dict['b_star'], kernel_dict['beta']
    gamma = (b_star_[0] ** 2 - beta ** 2) * dt
    assert gamma > 0.0, f"gamma fails positivity, gamma = {gamma}."

    m_input = m  # the original input of how many samples should be simulated
    # m itself is the number of samples that we simulate
    # We always have m >= m_input. At the end, we discard the additionally simulated paths.

    if qmc:
        if int(2 ** np.ceil(np.log2(m)) + 0.001) != m_input:
            print(f'Using QMC requires simulating a number m of samples that is a power of 2. The input m={m_input} '
                  f'is not a power of 2.')

    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = get_necessary_memory(return_times=return_times, m=m)
    if necessary_memory > available_memory:
        raise MemoryError(f'Not enough memory to store the sample paths of the rough Heston model with {return_times} '
                          f'time points where the sample paths should be '
                          f'returned and {m} sample paths. Roughly {necessary_memory}**2 bytes needed, '
                          f'while only {available_memory}**2 bytes are available.')

    available_memory_for_random_variables = available_memory / 3
    necessary_memory_for_random_variables = np.sqrt(3 * N_time) * np.sqrt(m) * np.sqrt(np.array([0.]).nbytes)
    n_batches = int(np.ceil((necessary_memory_for_random_variables / available_memory_for_random_variables) ** 2))
    m_batch = int(np.ceil(m / n_batches))
    m = m_batch * n_batches

    if sample_paths:
        result = np.empty((2, return_times + 1, m))
        result[0, 0, :] = np.log(S_0)
        result[1, 0, :] = xi(0)
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
            current_X, current_V, chi[:, i] = hqe_step(xi=xi, kernel=kernel, rho=rho, dt=dt, eps=eps, b_star_=b_star_,
                                                       v_old=current_V, chi_old=chi[:, :i], X_old=current_X,
                                                       rv=rv[:, 3 * i:3 * (i + 1)], beta=beta, gamma=gamma, r=r)
            if sample_paths and (i + 1) % saving_steps == 0:
                result[1, (i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_V
                result[0, (i + 1) // saving_steps, j * m_batch:(j + 1) * m_batch] = current_X
        if not sample_paths:
            result[1, j * m_batch:(j + 1) * m_batch] = current_V
            result[0, j * m_batch:(j + 1) * m_batch] = current_X
    result = result[..., :m_input]  # discard those values which were unnecessarily generated
    result[0, ...] = np.exp(result[0, ...])
    return result, rng, kernel_dict


def eur(H, K, lambda_, rho, nu, theta, V_0, S_0, T, r, m, N_time, qmc=True, payoff='call', n_maturities=None,
        implied_vol=False, qmc_error_estimators=25, verbose=0):
    """
    Gives the price or the implied volatility of a European option in the approximated, Markovian rough Heston model
    using MC or QMC simulation.
    :param H: Hurst parameter
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param r: Interest rate
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param payoff: The payoff function, or the string 'call' or the string 'put'
    :param n_maturities: If None, only uses the maturity T. If an integer, uses the maturity vector
        np.linspace(0, T, n_maturities + 1)[1:]. The strikes K are rescaled accordingly, where it is assumed that the
        given vector K corresponds to the largest maturity T. If n_maturities is an integer, N_time must be divisible by
        n_maturities
    :param implied_vol: If True (only for payoff 'call' or 'put') returns the implied volatility, else returns the price
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose: Determines the number of intermediary results printed to the console
    return: The prices of the call option for the various strike prices in K
    """
    is_smile = n_maturities is None
    if is_smile:
        n_maturities = 1
    T_vec = T * np.linspace(0, 1, n_maturities + 1)[1:]
    K_mat = S_0 * np.exp(np.sqrt(T_vec[:, None] / T) * np.log(K / S_0)[None, :])
    kernel_dict = get_kernel_dict(H=H, lambda_=lambda_, nu=nu, V_0=V_0, theta=theta, T=T, N_time=N_time)

    def get_samples(rv_shift=False):
        samples_, rng_, kernel_dict_ = samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0,
                                               r=r, m=m, N_time=N_time, sample_paths=True, return_times=n_maturities,
                                               qmc=qmc, rv_shift=rv_shift, kernel_dict=kernel_dict,
                                               verbose=verbose - 2)
        return samples_[0, 1:, :]

    if qmc:
        estimates = np.empty((n_maturities, len(K), qmc_error_estimators))
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            samples_1 = get_samples(False if i == 0 else True)
            estimates[:, :, i], _, _ = cf.eur_MC(S_0=S_0, K=K_mat, T=T_vec, r=r, samples=samples_1, payoff=payoff,
                                                 implied_vol=False)
        est, stat = cf.MC(estimates)
        if not implied_vol:
            if is_smile:
                est, stat = est[0, :], stat[0, :]
            return est, est - stat, est + stat
        smile, lower, upper = cf.iv_eur(S_0=S_0, K=K_mat, r=r, T=T_vec, price=est, payoff=payoff, stat=stat)
        if is_smile:
            smile, lower, upper = smile[0, :], lower[0, :], upper[0, :]
        return smile, lower, upper
    else:
        return cf.eur_MC(S_0=S_0, K=K_mat, T=T_vec, r=r, samples=get_samples()[0], payoff=payoff,
                         implied_vol=implied_vol)


def price_geom_asian_call(H, K, lambda_, rho, nu, theta, V_0, S_0, T, r, m, N_time, qmc=True, qmc_error_estimators=25,
                          verbose=0):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    :param H: Hurst parameter
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param r: Interest rate
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose: Determines the number of intermediary results printed to the console
    return: The prices of the call option for the various strike prices in K
    """
    n_batches, m_batch = get_n_batches(return_times=N_time, m=m)
    kernel_dict = get_kernel_dict(H=H, lambda_=lambda_, nu=nu, V_0=V_0, theta=theta, T=T, N_time=N_time)

    def get_sample_batch(rv_shift=False, rng=None):
        samples_, rng, kernel_dict_ = samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0,
                                              r=r, m=m_batch, N_time=N_time, sample_paths=True, qmc=qmc,
                                              rv_shift=rv_shift, kernel_dict=kernel_dict, rng=rng,
                                              verbose=verbose - 1)
        return samples_[0, :, :], rng

    def single_estimator(rv_shift=False):
        estimates = np.empty((len(K), n_batches))
        confidences = np.empty((len(K), n_batches))
        rng = None
        for j in range(n_batches):
            samples_, rng = get_sample_batch(rng=rng, rv_shift=rv_shift)
            estimates[:, j], low, upp = cf.price_geom_asian_call_MC(K=K, samples=samples_)
            confidences[:, j] = np.abs(estimates[:, j] - low)
            if qmc:
                rng.reset()
                rng.fast_forward((j + 1) * m_batch)
        estimate, confidence = cf.MC(estimates)
        if n_batches < 100:
            confidence, _ = cf.MC(confidence) / np.sqrt(n_batches)
        return estimate, estimate - confidence, estimate + confidence

    if qmc:
        estimators = np.empty((len(K), qmc_error_estimators))
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            estimators[:, i], _, _ = single_estimator(rv_shift=i != 0)
        est, stat = cf.MC(estimators)
        return est, est - stat, est + stat
    else:
        return single_estimator()


def price_avg_vol_call(H, K, lambda_, nu, theta, V_0, T, m, N_time, qmc=True, qmc_error_estimators=25, verbose=0):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    :param H: Hurst parameter
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param m: Number of samples
    :param N_time: Number of time steps used in simulation
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose: Determines the number of intermediary results printed to the console
    return: The prices of the call option for the various strike prices in K
    """
    n_batches, m_batch = get_n_batches(return_times=N_time, m=m)
    kernel_dict = get_kernel_dict(H=H, lambda_=lambda_, nu=nu, V_0=V_0, theta=theta, T=T, N_time=N_time)

    def get_samples(rv_shift=False, rng_=None):
        samples_, rng_, kernel_dict_ = samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=0., S_0=1.,
                                               r=0., m=m_batch, N_time=N_time, sample_paths=True, qmc=qmc,
                                               rv_shift=rv_shift, rng=rng_, kernel_dict=kernel_dict,
                                               verbose=verbose - 1)
        return samples_[0, :, :], rng_

    if qmc:
        estimates = np.empty((len(K), qmc_error_estimators * n_batches))
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            samples_1, rng = get_samples(False if i == 0 else True)
            estimates[:, i * n_batches], _, _ = cf.price_avg_vol_call_MC(K=K, samples=samples_1)
            for k in range(1, n_batches):
                samples_1, rng = get_samples(False if i == 0 else True, rng)
                estimates[:, i * n_batches + k], _, _ = cf.price_avg_vol_call_MC(K=K, samples=samples_1)
        est, stat = cf.MC(estimates)
        return est, est - stat, est + stat
    else:
        samples_1, _ = get_samples()
        return cf.price_geom_asian_call_MC(K=K, samples=samples_1)


def price_am(K, H, lambda_, rho, nu, theta, V_0, S_0, T, payoff, r, m=1000000, N_time=200, N_dates=12,
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
    :param qmc: If True, uses Quasi-Monte Carlo simulation with the Sobol sequence. If False, uses standard Monte Carlo
    :param qmc_error_estimators: Runs the pricing step of the Longstaff-Schwartz algorithm qmc_error_estimators times
        to get an MC estimate for the QMC error
    :param verbose: Determines the number of intermediary results printed to the console
    return: The prices of the call option for the various strike prices in K
    """
    kernel_dict = get_kernel_dict(H=H, lambda_=lambda_, nu=nu, V_0=V_0, theta=theta, T=T, N_time=N_time)
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

    def get_samples(rng_=None, rv_shift=False):
        samples_1, rng_, kernel_dict_ = \
            samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0, r=r, m=m,
                    N_time=N_time, sample_paths=True, qmc=qmc, rng=rng_, return_times=N_dates, rv_shift=rv_shift,
                    kernel_dict=kernel_dict, verbose=verbose - 1)
        samples_1 = np.transpose(samples_, (0, 2, 1))
        samples_1[1, :, :] = samples_1[1, :, :] - samples_1[1, :, :1]
        return samples_1, rng_

    if qmc:
        rng = None
        estimates = np.empty(qmc_error_estimators)
        for i in range(qmc_error_estimators):
            if verbose >= 1:
                print(f'Computing estimator {i + 1} of {qmc_error_estimators}')
            samples_, rng = get_samples(rng, False if i == 0 else True)
            rng.reset()
            rng.fast_forward(m)
            (_, _), models = cf.price_am(T=T, r=r, samples=samples_, payoff=payoff, features=features)
            samples_, rng = get_samples(rng, False if i == 0 else True)
            rng.reset()
            estimates[i], _ = cf.price_am_forward(T=T, r=r, samples=samples_, payoff=payoff, models=models,
                                                  features=features)
        est, stat = cf.MC(estimates)
    else:
        samples_, rng = get_samples()
        (_, _), models = cf.price_am(T=T, r=r, samples=samples_, payoff=payoff, features=features)
        samples_, rng = get_samples(rng_=rng, rv_shift=False)
        est, stat = cf.price_am_forward(T=T, r=r, samples=samples_, payoff=payoff, models=models, features=features)
    return est, stat
