import numpy as np
import RoughKernel as rk
import ComputationalFinance as cf
from scipy.special import gamma, hyp2f1


def sqrt_cov_matrix_non_markovian(H, T, N):
    """
    Computes the Cholesky decomposition of the covariance matrix of
    (int_0^(T/N) (T/N - s)^(H-1/2) dW_s, ..., int_0^T (T-s)^(H-1/2) dW_s).
    :return: The Cholesky decomposition of the above covariance matrix
    """
    times = np.arange(1, N + 1) * (T / N)
    minima = np.fmin(times[:, None], times[None, :])
    maxima = np.fmax(times[:, None], times[None, :])
    cov = gamma(H + 0.5) / gamma(H + 1.5) * minima ** (0.5 + H) * maxima ** (H - 0.5) \
        * hyp2f1(0.5 - H, 1, 1.5 + H, minima / maxima)
    return np.linalg.cholesky(cov)


def samples_non_markovian(H, T, m, N_time):
    """
    Computes m sample paths of fractional Brownian motion.
    :param H: Hurst parameter
    :param T: Final time
    :param N_time: Number of time discretization steps
    :param m: Number of samples
    :return: Numpy array of shape (N_time + 1, m).
    """
    sqrt_cov = sqrt_cov_matrix_non_markovian(H=H, T=T, N=N_time)
    return np.concatenate((np.zeros((1, m)), sqrt_cov @ np.random.normal(0, 1, size=(N_time, m))), axis=0)


def sqrt_cov_matrix_non_markovian_with_brownian_motion(H, T, N):
    """
    Computes the Cholesky decomposition of the covariance matrix of
    (int_0^(T/N) (T/N - s)^(H-1/2) dW_s, ..., int_0^T (T-s)^(H-1/2) dW_s, W_(T/N), ..., W_T).
    :return: The Cholesky decomposition of the above covariance matrix
    """
    cov = np.empty(shape=(2 * N, 2 * N))
    times = np.arange(1, N + 1) * (T / N)
    minima = np.fmin(times[:, None], times[None, :])
    maxima = np.fmax(times[:, None], times[None, :])
    cov[0:N, 0:N] = gamma(H + 0.5) / gamma(H + 1.5) * minima ** (0.5 + H) * maxima ** (H - 0.5) \
        * hyp2f1(0.5 - H, 1, 1.5 + H, minima / maxima)
    cov[N:(2 * N), N:(2 * N)] = minima
    cov[0:N, N:(2 * N)] = 1 / (H + 0.5) \
        * (times[:, None] ** (H + 0.5) - np.fmax(times[:, None] - times[None, :], 0) ** (H + 0.5))
    cov[N:(2 * N), 0:N] = cov[0:N, N:(2 * N)].T
    return np.linalg.cholesky(cov)


def samples_non_markovian_with_brownian_motion(H, T, m, N_time):
    """
    Computes m sample paths of fractional Brownian motion together with the corresponding Brownian motion.
    :param H: Hurst parameter
    :param T: Final time
    :param N_time: Number of time discretization steps
    :param m: Number of samples
    :return: Numpy array of shape (N_time + 1, 2, m), where result[:, 0, :] is the Brownian motion and result[:, 1, :]
        is the fractional Brownian motion.
    """
    sqrt_cov = sqrt_cov_matrix_non_markovian_with_brownian_motion(H=H, T=T, N=N_time)
    fBm_Bm = np.einsum('ij,jk->ik', sqrt_cov, np.random.normal(0, 1, size=(2 * N_time, m)))
    result = np.zeros((N_time + 1, 2, m))
    result[1:, 0, :] = fBm_Bm[N_time:, :]
    result[1:, 1, :] = fBm_Bm[:N_time, :]
    return result


def sqrt_cov_matrix(nodes, dt):
    """
    Computes the Cholesky decomposition of the covariance matrix of one step in Markovian approximation of fBm.
    More precisely, computes the Cholesky decomposition of the covariance matrix of the following Gaussian vector:
    (int_0^dt exp(-x_1(dt-s)) dW_s, ..., int_0^dt exp(-x_N(dt-s)) dW_s). Note that no weights of the
    quadrature rule are used.
    :param nodes: The nodes of the quadrature rule
    :param dt: Time step size
    :return: The Cholesky decomposition of the above covariance matrix
    """
    N = len(nodes)
    nodes = np.fmax(nodes, 1e-03)
    node_matrix = nodes[:, None] + nodes[None, :]
    cov_matrix = (1 - rk.exp_underflow(dt * node_matrix)) / node_matrix
    computed_cholesky = False
    cholesky = None
    while not computed_cholesky:
        try:
            cholesky = np.linalg.cholesky(cov_matrix)
            computed_cholesky = True
        except np.linalg.LinAlgError:
            print('We damp!')
            dampening_factor = 0.999
            for j in range(N):
                cov_matrix[:j, j] = cov_matrix[:j, j] * dampening_factor ** ((j + 1) / N)
                cov_matrix[j, :j] = cov_matrix[j, :j] * dampening_factor ** ((j + 1) / N)
            computed_cholesky = False
    return cholesky


def samples(T, nodes, weights, m, N_time, antithetic=True):
    """
    Computes m sample paths of the Markovian approximations of fractional Brownian motion.
    :param T: Final time
    :param N_time: Number of time discretization steps
    :param nodes: The nodes of the approximation
    :param weights: The weights of the approximation
    :param m: Number of samples
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :return: Numpy array of shape (len(nodes), m, N_time + 1). The fractional Brownian motion paths themselves can be
        obtained by summing over the first component.
    """
    dt = T / N_time
    N = len(nodes)
    sqrt_cov = sqrt_cov_matrix(nodes=nodes, dt=dt)
    exp_vector = rk.exp_underflow(dt * nodes)

    fBm = np.zeros((N, m, N_time + 1))
    normals = np.empty((m, N, N_time))
    if antithetic:
        normals[:m // 2, :, :] = np.random.normal(0, 1, size=(m // 2, N, N_time))
        if m % 2 == 0:
            normals[m // 2:, :, :] = -normals[:m // 2, :, :]
        else:
            normals[m // 2:-1, :, :] = -normals[:m // 2, :, :]
            normals[-1, :, :] = np.random.normal(0, 1, size=(N, N_time))
    else:
        normals = np.random.normal(0, 1, size=(m, N, N_time))
    increments = np.einsum('ij,kjl->ikl', sqrt_cov, normals)
    for i in range(N_time):
        fBm[:, :, i + 1] = exp_vector[:, None] * fBm[:, :, i] + increments[:, :, i]
    '''
    # Too unstable, although it would avoid the for loop:
    exp_vector = rk.exp_underflow(np.linspace(0, T - dt, N_time)[None, ::-1] * active_nodes[:, None])
    increments = np.concatenate((np.zeros((M, N, 1)), increments[:, -N:, :-1]), axis=-1)
    fBm = np.sum((exp_vector[None, ...] * increments).cumsum(axis=-1)
                     / (exp_vector / weights[:, None])[None, :, :], axis=-2)
    '''
    return weights[:, None, None] * fBm


def samples_Bm(T, nodes, weights, m, N_time, antithetic=True):
    """
    Computes m sample paths of the Markovian approximations of fractional Brownian motion.
    :param T: Final time
    :param N_time: Number of time discretization steps
    :param nodes: The nodes of the approximation
    :param weights: The weights of the approximation
    :param m: Number of samples
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :return: Numpy array of shape (len(nodes), m, N_time + 1). The fractional Brownian motion paths themselves can be
        obtained by summing over the first component.
    """
    result = np.zeros((2, m, N_time + 1))
    result[0, :, 1:] = np.cumsum(np.random.normal(0, np.sqrt(T / N_time), size=(m, N_time)), axis=-1)
    return result


def am_features(x, degree=6):
    n_samples = x.shape[-1]
    d = x.shape[0]
    vol = np.sum(x, axis=0)
    vol_factors = x[:-1, :].T
    if degree == 1:
        dim = 1
    else:
        dim = d + 1
    feat = np.empty((n_samples, dim))
    if degree >= 1:
        feat[:, 0] = vol
    if degree >= 2:
        feat[:, 1] = vol ** 2
        feat[:, 2:] = vol_factors
    current_N = 3
    current_ind = d + 1
    lower_N_vol = 1
    upper_N_vol = d + 1
    lower_N_vol_factors = np.arange(2, d + 1, dtype=int)
    upper_N_vol_factors = d + 1
    while current_N <= degree:
        feat_new = np.empty((n_samples, feat.shape[1] + upper_N_vol - lower_N_vol
                             + (current_N % 2 == 0)
                             * (upper_N_vol_factors * (d - 1) - np.sum(lower_N_vol_factors, dtype=int))))
        feat_new[:, :current_ind] = feat
        feat = feat_new
        next_ind = current_ind + upper_N_vol - lower_N_vol
        feat[:, current_ind:next_ind] = vol[:, None] * feat[:, lower_N_vol:upper_N_vol]
        lower_N_vol = current_ind
        current_ind = next_ind
        if current_N % 2 == 0:
            for i in range(d - 1):
                next_ind = current_ind + upper_N_vol_factors - lower_N_vol_factors[i]
                feat[:, current_ind:next_ind] = \
                    vol_factors[:, i:i + 1] * feat[:, lower_N_vol_factors[i]:upper_N_vol_factors]
                lower_N_vol_factors[i] = current_ind
                current_ind = next_ind
            upper_N_vol_factors = current_ind
        upper_N_vol = current_ind
        current_N = current_N + 1
    return feat


def price_am(T, nodes, weights, r=0., m=1000000, N_time=12, feature_degree=6, antithetic=True, unbiased=True):
    """
    Gives the price of an American option in the approximated, Markovian rough Heston model.
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param r: Interest rate
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param N_time: Number of time steps used in the simulation. Is the same as the number of exercise dates because
        the sample path simulation is exact
    :param feature_degree: The degree of the polynomial features used in the regression
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :param unbiased: If True, uses newly generated samples to compute the price of the American option
    return: The prices of the call option for the various strike prices in K
    """
    def payoff(V):
        return np.sum(V, axis=0)

    def features(x):
        return am_features(x=x, degree=feature_degree)

    samples_ = samples(T=T, nodes=nodes, weights=weights, m=m, N_time=N_time, antithetic=antithetic)
    # samples_ = samples_non_markovian_with_brownian_motion(H=0.49, T=T, m=m, N_time=N_time)[:, 1, :].T
    # samples_ = samples_[None, :, :]


    (biased_est, biased_stat), models = cf.price_am_iso(T=T, r=r, samples=samples_, antithetic=antithetic,
                                                        payoff=payoff, features=features)
    # print(biased_est, biased_stat)
    if not unbiased:
        return biased_est, biased_stat, models, features

    samples_ = samples(T=T, nodes=nodes, weights=weights, m=m, N_time=N_time, antithetic=antithetic)
    # samples_ = samples_non_markovian_with_brownian_motion(H=0.49, T=T, m=m, N_time=N_time)[:, 1, :].T
    # samples_ = samples_[None, :, :]
    est, stat = cf.price_am_iso_forward(T=T, r=r, samples=samples_, payoff=payoff, models=models,
                                        features=features, antithetic=antithetic)
    return est, stat, biased_est, biased_stat, models, features
