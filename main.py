import time

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import stats, linalg, special
from scipy.special import gamma
import mpmath as mp

c_H = (math.gamma(0.1 + 0.5) * math.gamma(0.5 - 0.1))

# noinspection PyShadowingNames
def fractional_kernel(H, t):
    return t ** (H - 0.5) / math.gamma(H + 0.5)


# noinspection PyShadowingNames
def fractional_kernel_laplace(H, t, rho):
    return np.exp(-rho * t) / (math.gamma(H + 0.5) * math.gamma(0.5 - H))


V_0 = 0.02
theta = 0.02
lambda_ = 0.3
nu = 0.3
T = 0.5
N = 5000
m = 1

# improvements

G = np.zeros(shape=(N + 1,), dtype=float)
# G[1:] = np.array([fractional_kernel(H, float(i)/N) for i in range(1, N+1)])
GK = np.zeros(shape=(N + 1,), dtype=float)


# GK[1:] = np.array([fractional_kernel_approximation(H, K, float(i)/N) for i in range(1, N+1)])


def BlackScholes(T=1., N=1000, sigma=0.2, S_0=1., m=1000):
    print("The BlackScholes function is deprecated.")
    normals = np.random.normal(0, 1, size=m)
    return np.exp(sigma * normals - sigma ** 2 / 2)


# noinspection SpellCheckingInspection,PyShadowingNames
def quadrature_rule_interval(H, m, a, b):
    """
    Returns the nodes and weights of the Gauss quadrature rule level m for the fractional weight function on [a,b]
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param a: Left end of interval
    :param b: Right end of interval
    :return: The nodes and weights, in form [[node1, node2, ...], [weight1, weight2, ...]]
    """
    frac = b / a
    c_H = 1. / (mp.gamma(mp.mpf(0.5) + H) * mp.gamma(mp.mpf(0.5) - H))
    if m == 1:
        node = (0.5 - H) / (1.5 - H) * a * (frac ** (1.5 - H) - 1.) / (frac ** (0.5 - H) - 1.)
        weight = c_H / (0.5 - H) * (a ** (0.5 - H)) * (frac ** (0.5 - H) - 1.)
        return mp.matrix([[node], [weight]])
    if m == 2:
        a00 = (0.5 - H) / (1.5 - H) * a * (frac ** (1.5 - H) - 1.) / (frac ** (0.5 - H) - 1.)
        a10 = (1. / (2.5 - H) * (a * a) * (frac ** (2.5 - H) - 1.) - a00 / (1.5 - H) * a * (frac ** (1.5 - H) - 1.)) / (
                1. / (0.5 - H) * (frac ** (0.5 - H) - 1.))
        a11 = (1. / (3.5 - H) * (a ** 3) * (frac ** (3.5 - H) - 1.) - 2. * a00 / (2.5 - H) * a * a * (
                frac ** (2.5 - H) - 1.) + a00 ** 2. / (1.5 - H) * a * (frac ** (1.5 - H) - 1.)) / (
                      1. / (2.5 - H) * a * a * (frac ** (2.5 - H) - 1.) - 2. * a00 / (1.5 - H) * a * (
                      frac ** (1.5 - H) - 1.) + a00 ** 2. / (0.5 - H) * (frac ** (0.5 - H) - 1.))
        x1 = (a11 + a00) / 2. + np.sqrt(((a11 + a00) / 2.) ** 2 + a10 - a11 * a00)
        x2 = (a11 + a00) / 2. - np.sqrt(((a11 + a00) / 2.) ** 2 + a10 - a11 * a00)
        numerator = 1. / (2.5 - H) * (a ** (2.5 - H)) * (frac ** (2.5 - H) - 1.) - 2. * a00 / (1.5 - H) * (
                a ** (1.5 - H)) * (frac ** (1.5 - H) - 1.) + a00 ** 2 / (0.5 - H) * (a ** (0.5 - H)) * (
                            frac ** (0.5 - H) - 1.)
        w1 = c_H * numerator / ((2 * x1 - a11 - a00) * (x1 - a00))
        w2 = c_H * numerator / ((2 * x2 - a11 - a00) * (x2 - a00))
        return mp.matrix([[x1, x2], [w1, w2]])


# noinspection PyShadowingNames
def quadrature_rule(H, m, partition):
    """
    Returns the quadrature rule of level m of the fractional kernel with Hurst parameter H on all the partition
    intervals.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param partition: The partition points
    :return: All the nodes and weights, in the form [[node1, node2, ...], [weight1, weight2, ...]]
    """

    number_intervals = len(partition) - 1
    rule = mp.matrix(2, m * number_intervals)
    for i in range(0, number_intervals):
        rule[:, m * i:m * (i + 1)] = quadrature_rule_interval(H, m, partition[i], partition[i + 1])
    return rule


# noinspection PyShadowingNames
def quadrature_rule_geometric(H, m, n, a=1., b=1.):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of subintervals
    :param a: Can shift the left end-point of the total interval
    :param b: Can shift the right end-point of the total interval
    :return: All the nodes and weights, in the form [[node1, node2, ...], [weight1, weight2, ...]]
    """
    r = 1. * m
    gamma = 0.5 - H
    delta = H
    xi0 = a * n ** (-r / gamma)
    xin = b * n ** (r / delta)
    partition = np.array([xi0 ** (float(n - i) / n) * xin ** (float(i) / n) for i in range(0, n + 1)])
    return quadrature_rule(H, m, partition)


# noinspection PyShadowingNames
def sqrt_cov_matrix_fBm_approximation_mpmath(H, dt, nodes):
    """
    Computes the Cholesky factorization of the joint covariance matrix of the OU-approximations of the fBm with Hurst
    parameter H on a time-step of dt, and of the fBm itself. Returns an instance of mpmath.matrix.
    :param H: Hurst parameter
    :param dt: Time step size
    :param nodes: The nodes of the approximation (mean-reversion parameters of the OU processes)
    :return: The Cholesky factorization of the covariance matrix, the last row/column is the actual fBm
    """
    n = len(nodes)
    cov_matrix = mp.matrix(n + 1, n + 1)
    for i in range(0, n):
        for j in range(0, n):
            cov_matrix[i, j] = (1 - mp.exp(-dt * (nodes[i] + nodes[j]))) / (nodes[i] + nodes[j])
    for i in range(0, n):
        entry = nodes[i] ** (-H - mp.mpf(0.5)) * mp.gammainc(H + mp.mpf(0.5), 0., dt * nodes[i]) / mp.gammainc(
            H + mp.mpf(0.5), 0.)
        cov_matrix[n, i] = entry
        cov_matrix[i, n] = entry
    cov_matrix[n, n] = dt ** (2. * H) / (2. * H * mp.gamma(H + 0.5) ** 2.)
    return mp.cholesky(cov_matrix)


# noinspection PyShadowingNames
def sqrt_cov_matrix_fBm_approximation(H, dt, nodes):
    """
    Computes the Cholesky factorization of the joint covariance matrix of the OU-approximations of the fBm with Hurst
    parameter H on a time-step of dt, and of the fBm itself. Computations are done using the mpmath library, but the
    output is a numpy array.
    :param H: Hurst parameter
    :param dt: Time step size
    :param nodes: The nodes of the approximation (mean-reversion parameters of the OU processes)
    :return: The Cholesky factorization of the covariance matrix, the last row/column is the actual fBm
    """
    cov_root_mp = sqrt_cov_matrix_fBm_approximation_mpmath(H, dt, nodes)
    n = len(nodes) + 1
    cov_root = np.empty(shape=(n, n))
    for i in range(0, n):
        for j in range(0, n):
            cov_root[i, j] = cov_root_mp[i, j]
    return cov_root


# noinspection PyShadowingNames
def fBm_true_and_approximated(H, dt, nodes, weights, samples):
    """
    Simulates jointly samples samples of a fBm (with Hurst parameter H) increment (step size dt) and the OU
    approximation with nodes (mean-reversions) nodes, and weights weights.
    :param H: Hurst parameter
    :param dt: Time step size
    :param nodes: Nodes of the approximation (mean-reversions)
    :param weights: Weights of the approximation
    :param samples: The number of samples simulated
    :return: The final value of the true fBms and the approximated fBms, i.e. [[true], [approximated]]
    """
    n = len(nodes)
    cov_root = sqrt_cov_matrix_fBm_approximation(H, dt, nodes)
    V_approx = np.empty(shape=(samples,))
    V_true = np.empty(shape=(samples,))
    for i in range(0, samples):
        V_OU = cov_root.dot(np.random.normal(0., 1., n + 1))
        V_approx[i] = np.dot(weights, V_OU[0:n])
        V_true[i] = V_OU[n]
    return np.array([V_true, V_approx])


# noinspection PyShadowingNames
def strong_error_fBm_approximation_MC(H, T, m, n, samples, a=1., b=1.):
    """
    Estimates the L^2 error of the OU-approximation of the final value at time T of a fBm with Hurst parameter H. The
    OU-approximation uses an m-point quadrature rule on n geometrically spaced subintervals. The error is estimated
    using Monte Carlo, where samples samples are used.
    :param H: Hurst parameter
    :param T: Final time
    :param m: Level of the quadrature rule
    :param n: Number of intervals used for the quadrature rule (total number of nodes: m*n)
    :param samples: Number of samples used for the Monte Carlo estimate
    :param a: Can shift left end-point of total interval of the quadrature rule
    :param b: Can shift right end-point of total interval of the quadrature rule
    :return: The estimated error and a 95% confidence interval, i.e. [error, 1.96*std/sqrt(samples)]
    """
    mp.mp.dps = int(np.maximum(20., (m * n) ** 2 / 20.))

    quad_rule = quadrature_rule_geometric(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights = quad_rule[1, :]

    V_true_and_approx = fBm_true_and_approximated(H, T, nodes, weights, samples)
    V_true = V_true_and_approx[0]
    V_approx = V_true_and_approx[1]

    V_errors = np.fabs(V_true - V_approx) ** 2

    error = np.sqrt(np.mean(V_errors))
    stat = 1 / (2. * error) * 1.96 * np.std(V_errors) / np.sqrt(samples)

    return np.array([error, stat])


def sqrt_cov_matrix_rBergomi_BFG(H=0.1, T=1., N=1000):
    """
    Computes the Cholesky decomposition of the covariance matrix of
    (int_0^(T/N) (T/N - s)^(H-1/2) dW_s, ..., int_0^T (T-s)^(H-1/2) dW_s, W_(T/N), ..., W_T).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :return: The Cholesky decomposition of the above covariance matrix
    """
    cov = np.empty(shape=(2 * N, 2 * N))

    cov[0:N, 0:N] = scipy.special.gamma(H + 0.5) / scipy.special.gamma(H + 1.5) * np.array([[(np.fmin(i,
                                                                                                      j) * T / N) ** (
                                                                                                         0.5 + H) * (
                                                                                                         np.fmax(i,
                                                                                                                 j) * T / N) ** (
                                                                                                         H - 0.5) * scipy.special.hyp2f1(
        0.5 - H, 1., 1.5 + H, np.fmin(float(i), float(j)) / np.fmax(float(i), float(j))) for i in range(1, N + 1)] for j
                                                                                            in range(1, N + 1)])
    '''
    cov[0:N, 0:N] = 1/(H+0.5) * np.array([[(np.fmax(i, j)*T/N)**(0.5+H) * (np.fmin(i, j)*T/N)**(H-0.5) * scipy.special.hyp2f1(1., 0.5-H, 1.5+H, np.fmin(float(i), float(j))/np.fmax(float(i), float(j))) for i in range(1, N+1)] for j in range(1, N+1)])
    
    for i in range(1, N+1):
        cov[i-1, i-1] = i*T/(2*H*N)
    '''
    cov[N:(2 * N), N:(2 * N)] = np.array([[np.fmin(i, j) * T / N for i in range(1, N + 1)] for j in range(1, N + 1)])
    cov[0:N, N:(2 * N)] = 1. / (H + 0.5) * np.array(
        [[(i * T / N) ** (H + 0.5) - (np.fmax(i - j, 0) * T / N) ** (H + 0.5) for j in range(1, N + 1)] for i in
         range(1, N + 1)])
    cov[N:(2 * N), 0:N] = 1. / (H + 0.5) * np.array(
        [[(i * T / N) ** (H + 0.5) - (np.fmax(i - j, 0) * T / N) ** (H + 0.5) for i in range(1, N + 1)] for j in
         range(1, N + 1)])
    return np.linalg.cholesky(cov)


def plot_rBergomi_BFG(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9):
    """
    Plots a path of the rough Bergomi model together with the variance process (approximation taken from
    Bayer, Friz, Gatheral, Pricing under rough volatility).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the variance process and the stock price
    :return: Nothing, plots a realization of the stock price and the volatility
    """
    sqrt_cov = sqrt_cov_matrix_rBergomi_BFG(H, T, N)
    W_vec = sqrt_cov.dot(np.random.normal(0., 1., 2 * N))
    V = np.empty(shape=(N + 1,))
    V[0] = V_0
    V[1:] = V_0 * np.exp(
        eta * np.sqrt(2 * H) * W_vec[0:N] - eta * eta / 2 * np.array([(i * T / N) ** (2 * H) for i in range(1, N + 1)]))

    W_diff = np.empty(shape=(N,))
    W_diff[0] = W_vec[N]
    W_diff[1:] = W_vec[(N + 1):(2 * N)] - W_vec[N:(2 * N - 1)]
    W_2 = np.random.normal(0., T / N, N)
    S = np.empty(shape=(N + 1,))
    S[0] = S_0
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp(
            np.sqrt(V[i - 1]) * (rho * W_diff[i - 1] + np.sqrt(1 - rho ** 2) * W_2[i - 1]) - V[i - 1] * T / (2 * N))

    times = np.array([i * T / N for i in range(0, N + 1)])
    plt.plot(times, V)
    plt.plot(times, S)
    plt.show()


def rBergomi_BFG(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, m=1000, rounds=1):
    """
    Computes m final stock prices of the rough Bergomi model, using the approximation from Bayer, Friz, Gatheral.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param m: Number of samples
    :param rounds: Actually computes m*rounds samples, but only m at a time to avoid excessive memory usage.
    :return: An array of the final stock prices [S_T^1, S_T^2, ..., S_T^m]
    """
    dt = T / N
    sqrt_cov = sqrt_cov_matrix_rBergomi_BFG(H, T, N)
    S = np.empty(m*rounds)
    for round in range(rounds):
        W_vec = sqrt_cov.dot(np.random.normal(0, 1, (2 * N, m)))
        V = np.empty(shape=(N + 1, m))
        V[0, :] = V_0
        V[1:, :] = V_0 * np.exp(
            eta * np.sqrt(2 * H) * W_vec[0:N, :] - eta * eta / 2 * np.array(
                [[(i * dt) ** (2 * H) for _ in range(0, m)] for i in range(1, N + 1)]))
        W_diff = np.empty(shape=(N, m))
        W_diff[0, :] = W_vec[N, :]
        W_diff[1:, :] = W_vec[(N + 1):(2 * N), :] - W_vec[N:(2 * N - 1), :]
        W_2 = np.random.normal(0, np.sqrt(dt), (N, m))
        S_ = S_0 * np.ones(shape=(m,))
        for i in range(N):
            S_ = S_ * np.exp(np.sqrt(V[i, :]) * (rho * W_diff[i, :] + np.sqrt(1 - rho ** 2) * W_2[i, :]) - V[i, :] * dt / 2)
        S[round*m:(round+1)*m] = S_
    return S


def MC(samples):
    """
    Computes an approximation of E[X], where samples~X.
    :param samples: The samples of the random variable
    :return: The expectation and a 95% confidence interval, (expectation, confidence)
    """
    return np.average(samples, axis=-1), 1.95 * np.std(samples, axis=-1) / np.sqrt(samples.shape[-1])


def BS_d1(S=1., K=1., sigma=1., r=0., T=1.):
    """
    Computes the first of the two nodes of the Black-Scholes model where the CDF is evaluated.
    :param S: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The first node
    """
    return (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def BS_d2(S=1., K=1., sigma=1., r=0., T=1.):
    """
    Computes the second of the two nodes of the Black-Scholes model where the CDF is evaluated.
    :param S: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The second node
    """
    return BS_d1(S, K, sigma, r, T) - sigma * np.sqrt(T)


def BS_call_price(S=1., K=1., sigma=1., r=0., T=1.):
    """
    Computes the price of a European call option under the Black-Scholes model
    :param S: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The price of a call option
    """
    d1 = BS_d1(S, K, sigma, r, T)
    d2 = BS_d2(S, K, sigma, r, T)
    return S * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)


def BS_put_price(S=1., K=1., sigma=1., r=0., T=1.):
    """
    Computes the price of a European put option under the Black-Scholes model
    :param S: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The price of a put option
    """
    d1 = BS_d1(S, K, sigma, r, T)
    d2 = BS_d2(S, K, sigma, r, T)
    return - S * scipy.stats.norm.cdf(-d1) + K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2)


def implied_volatility_call(S=1., K=1., r=0., T=1., price=1., tol=10. ** (-10), sl=10. ** (-10), sr=10.):
    """
    Computes the implied volatility of a European call option, assuming it is in [sl, sr].
    :param S: Initial stock price
    :param K: Strike price
    :param r: Drift
    :param T: Final time/maturity
    :param price: (Market) price of the option
    :param tol: Error tolerance in the approximation of the implied volatility
    :param sl: Left point of the search interval
    :param sr: Right point of the search interval
    :return: The implied volatility
    """
    sm = (sl + sr) / 2
    while np.amax(sr - sl) > tol:
        em = BS_call_price(S, K, sm, r, T) - price
        sl = (em < 0) * sm + (em >= 0) * sl
        sr = (em >= 0) * sm + (em < 0) * sr
        sm = (sl + sr) / 2
    return sm


def call_option_payoff(S, K):
    """
    Computes the payoff of a (European) call option.
    :param S: (Final) stock price
    :param K: Strike price
    :return: The payoff. If S and K are floats, a float is returned. If either S or K are floats, the other being a
            vector, a vector is returned. If both S and K are vectors, a matrix is returned. In the matrix, the rows
            have fixed K, the columns have fixed S.
    """
    if isinstance(K, float) or isinstance(S, float):
        return np.fmax(S - K, 0)

    S_matrix = np.repeat(np.array([S]), len(K), axis=0)
    K_matrix = np.repeat(np.array([K]), len(S), axis=0).transpose()
    return np.fmax(S_matrix - K_matrix, 0)


def put_option_payoff(S, K):
    """
    Computes the payoff of a (European) put option.
    :param S: (Final) stock price
    :param K: Strike price
    :return: The payoff. If S and K are floats, a float is returned. If either S or K are floats, the other being a
            vector, a vector is returned. If both S and K are vectors, a matrix is returned. In the matrix, the rows
            have fixed K, the columns have fixed S.
    """
    if isinstance(K, float) or isinstance(S, float):
        return np.fmax(K - S, 0)

    S_matrix = np.repeat(np.array([S]), len(K), axis=0)
    K_matrix = np.repeat(np.array([K]), len(S), axis=0).transpose()
    return np.fmax(K_matrix - S_matrix, 0)


# noinspection PyShadowingNames
def implied_volatility_call_rBergomi_BFG(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, K=1., m=1000, rounds=1):
    """
    Computes the implied volatility of the European call option under the rough Bergomi model, using the approximation
    from Bayer, Friz, Gatheral.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param K: The strike price
    :param m: Number of samples
    :param rounds: Actually uses m*rounds samples, but only m at a time to avoid excessive memory usage
    :return: The implied volatility and a 95% confidence interval, in the form (estimate, lower, upper)
    """
    tic = time.perf_counter()
    samples = rBergomi_BFG(H, T, N, eta, V_0, S_0, rho, m, rounds)
    toc = time.perf_counter()
    print(f"Generating {m*rounds} rBergomi samples with N={N} takes {np.round(toc-tic, 2)} seconds.")
    (price_estimate, price_stat) = MC(call_option_payoff(samples, K))
    implied_volatility_estimate = implied_volatility_call(S_0, K, 0, T, price_estimate)
    implied_volatility_lower = implied_volatility_call(S_0, K, 0, T, price_estimate - price_stat)
    implied_volatility_upper = implied_volatility_call(S_0, K, 0, T, price_estimate + price_stat)
    return implied_volatility_estimate, implied_volatility_lower, implied_volatility_upper


def sqrt_cov_matrix_rBergomi_AK_mpmath(dt=1., nodes=None):
    """
    Computes the Cholesky decomposition of the covariance matrix of one step in the scheme that was inspired by
    Alfonsi and Kebaier, "Approximation of stochastic Volterra equations with kernels of completely monotone type".
    More precisely, computes the Cholesky decomposition of the covariance matrix of the following Gaussian vector:
    (int_0^dt exp(-x_1(dt-s)) dW_s, ..., int_0^dt exp(-x_k(dt-s)) dW_s, W_dt),
    where k = m*n is the number of nodes, and these nodes are determined by the m-point quadrature rule approximation
    of the rough kernel on n subintervals. Note that no weights of the quadrature rule are used. Returns a numpy array,
    but the computation is done using the mpmath library.
    :param dt: Time step size
    :param nodes: The nodes of the quadrature rule approximation
    :return: The Cholesky decomposition of the above covariance matrix
    """
    k = len(nodes)
    mp.mp.dps = int(np.amax(np.array([20., k ** 2, mp.mp.dps])))
    cov_matrix = mp.matrix(k + 1, k + 1)
    for i in range(0, k):
        for j in range(0, k):
            cov_matrix[i, j] = (1 - mp.exp(-dt * (nodes[i] + nodes[j]))) / (
                    nodes[i] + nodes[j])
    for i in range(0, k):
        entry = (1 - mp.exp(-dt * nodes[i])) / nodes[i]
        cov_matrix[k, i] = entry
        cov_matrix[i, k] = entry
    cov_matrix[k, k] = dt
    return mp.cholesky(cov_matrix)


def sqrt_cov_matrix_rBergomi_AK(dt=1., nodes=None):
    """
    Computes the Cholesky decomposition of the covariance matrix of one step in the scheme that was inspired by
    Alfonsi and Kebaier, "Approximation of stochastic Volterra equations with kernels of completely monotone type".
    More precisely, computes the Cholesky decomposition of the covariance matrix of the following Gaussian vector:
    (int_0^dt exp(-x_1(dt-s)) dW_s, ..., int_0^dt exp(-x_k(dt-s)) dW_s, W_dt),
    where k = m*n is the number of nodes, and these nodes are determined by the m-point quadrature rule approximation
    of the rough kernel on n subintervals. Note that no weights of the quadrature rule are used. Returns a numpy array,
    but the computation is done using the mpmath library.
    :param dt: Time step size
    :param nodes: The nodes of the quadrature rule approximation
    :return: The Cholesky decomposition of the above covariance matrix
    """
    cov_root_mp = sqrt_cov_matrix_rBergomi_AK_mpmath(dt, nodes)
    num = len(nodes) + 1
    cov_root = np.empty(shape=(num, num))
    for i in range(0, num):
        for j in range(0, num):
            cov_root[i, j] = float(cov_root_mp[i, j])
    return cov_root


def rBergomi_AK_variance_integral(nodes, weights, t):
    """
    Computes an integral that appears in the rBergomi approximation that is inspired by Alfonsi and Kebaier. Computes
    the integral int_0^t ( sum_i w_i exp(-x_i (t-s)) )^2 ds. Takes a vector in t. May take as an input mpmath elements,
    but returns a numpy array.
    :param nodes: The nodes of the quadrature rule
    :param weights: The weights of the quadrature rule
    :param t: The current time
    :return: The vector of integrals
    """
    weight_matrix = mp.matrix([[weight_i * weight_j for weight_j in weights] for weight_i in weights])
    node_matrix = mp.matrix([[node_i + node_j for node_j in nodes] for node_i in nodes])
    result = np.empty(shape=(len(t),))
    for time_index in range(len(t)):
        expression = mp.matrix([[weight_matrix[i, j] / node_matrix[i, j] * (
                1 - mp.exp(-node_matrix[i, j] * mp.mpf(t[time_index]))) for j in range(len(nodes))] for i in
                                range(len(nodes))])
        expression_np = np.array([[expression[i, j] for j in range(len(nodes))] for i in range(len(nodes))])
        result[time_index] = np.sum(expression_np.flatten())
    return result


def plot_rBergomi_AK(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, m=1, n=10, a=1., b=1.):
    """
    Plots a path of the rough Bergomi model together with the variance process (approximation inspired by Alfonsi and
    Kebaier, with discretization points taken from Harms).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the variance process and the stock price
    :param m: Order of the quadrature rule
    :param n: Number of intervals in the quadrature rule
    :param a: Can shift the left end-point of the total interval of the quadrature rule
    :param b: Can shift the right end-point of the total interval of the quadrature rule
    :return: Nothing, plots a realization of the stock price and the volatility
    """
    dt = T / N
    quad_rule = quadrature_rule_geometric(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights_mp = quad_rule[1, :]
    sqrt_cov = sqrt_cov_matrix_rBergomi_AK(dt, nodes)
    weights = np.array([float(weight) for weight in weights_mp])
    W_1_diff = sqrt_cov.dot(np.random.normal(0., 1., size=(m * n + 1, N)))
    eta_transformed = eta * np.sqrt(2 * H) * scipy.special.gamma(
        H + 0.5)  # the sqrt is in the model, the Gamma takes care of the c_H in the weights of the quadrature rule
    exp_vector = np.exp(np.array([-float(node) * dt for node in nodes]))
    W_1_fBm = np.zeros(shape=(m * n, N + 1))
    for i in range(1, N + 1):
        W_1_fBm[:, i] = exp_vector * W_1_fBm[:, i - 1] + W_1_diff[:-1, i - 1]
    W_1_fBm = eta_transformed * np.dot(weights, W_1_fBm)
    W_1_diff = W_1_diff[-1, :]
    times = np.arange(N + 1) * dt
    variances = eta_transformed ** 2 / 2 * rBergomi_AK_variance_integral(nodes, weights, times)
    V = V_0 * np.exp(W_1_fBm - variances)
    W_2_diff = np.sqrt(dt) * np.random.normal(0, 1, size=N)
    S = np.empty(shape=(N + 1,))
    S[0] = S_0
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp(
            np.sqrt(V[i - 1]) * (rho * W_1_diff[i - 1] + np.sqrt(1 - rho ** 2) * W_2_diff[i - 1]) - V[i - 1] / 2 * dt)
    plt.plot(times, V)
    plt.plot(times, S)
    plt.show()


def rBergomi_AK(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, m=1, n=10, a=1., b=1., M=1000, rounds=1):
    """
    Computes M final stock prices of an approximation of the rough Bergomi model (approximation inspired by Alfonsi and
    Kebaier, with discretization points taken from Harms).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the variance process and the stock price
    :param m: Order of the quadrature rule
    :param n: Number of intervals in the quadrature rule
    :param a: Can shift the left end-point of the total interval of the quadrature rule
    :param b: Can shift the right end-point of the total interval of the quadrature rule
    :param M: Number of final stock prices
    :param rounds: Actually computes M*rounds samples, but only M at a time to avoid excessive memory usage.
    :return: An array containing the final stock prices
    """
    dt = T / N
    quad_rule = quadrature_rule_geometric(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights_mp = quad_rule[1, :]
    sqrt_cov = sqrt_cov_matrix_rBergomi_AK(dt, nodes)
    weights = np.array([float(weight) for weight in weights_mp])
    exp_vector = np.exp(np.array([-float(node) * dt for node in nodes]))
    eta_transformed = eta * np.sqrt(2 * H) * scipy.special.gamma(
        H + 0.5)  # the sqrt is in the model, the Gamma takes care of the c_H in the weights of the quadrature rule
    times = np.arange(N + 1) * dt
    variances = eta_transformed ** 2 / 2 * rBergomi_AK_variance_integral(nodes, weights, times)
    S = np.empty(shape=(M*rounds))
    for round in range(rounds):
        W_1_diff = np.array([sqrt_cov.dot(np.random.normal(0., 1., size=(m * n + 1, N))) for _ in range(M)])
        # W_1_diff = np.transpose(sqrt_cov.dot(np.random.normal(0., 1., size=(M, m*n+1, N))), [1, 0, 2]) is slower
        W_1_fBm = np.zeros(shape=(M, m * n, N + 1))
        for i in range(1, N + 1):
            W_1_fBm[:, :, i] = exp_vector * W_1_fBm[:, :, i - 1] + W_1_diff[:, :-1, i - 1]
        for i in range(len(weights)):
            W_1_fBm[:, i, :] = weights[i] * W_1_fBm[:, i, :]
        W_1_fBm = eta_transformed * np.sum(W_1_fBm, axis=1)
        W_1_diff = W_1_diff[:, -1, :]
        V = V_0 * np.exp(W_1_fBm - variances)
        W_2_diff = np.sqrt(dt) * np.random.normal(0, 1, size=(M, N))
        S_ = S_0 * np.ones(shape=(M,))
        for i in range(1, N + 1):
            S_ = S_ * np.exp(
                np.sqrt(V[:, i - 1]) * (rho * W_1_diff[:, i - 1] + np.sqrt(1 - rho ** 2) * W_2_diff[:, i - 1]) - V[:,
                                                                                                                 i - 1] / 2 * dt)
        S[round*M:(round+1)*M] = S_
    return S


def implied_volatility_call_rBergomi_AK(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, K=1., m=1, n=10,
                                        a=1., b=1., M=1000, rounds=1):
    """
    Computes the implied volatility of the European call option under the rough Bergomi model, using the approximation
    inspired by Alfonsi and Kebaier.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param K: The strike price
    :param m: Order of the quadrature rule
    :param n: Number of intervals in the quadrature rule
    :param a: Can be used to shift the left endpoint of the total quadrature interval
    :param b: Can be used to shift the right endpoint of the total quadrature interval
    :param M: Number of samples
    :param rounds: Actually uses M*rounds samples, but only m at a time to avoid excessive memory usage
    :return: The implied volatility and a 95% confidence interval, in the form (estimate, lower, upper)
    """
    tic = time.perf_counter()
    samples = rBergomi_AK(H, T, N, eta, V_0, S_0, rho, m, n, a, b, M, rounds)
    toc = time.perf_counter()
    print(f"Generating {M*rounds} approximate rBergomi samples with N={N} and n*m={n*m} takes {np.round(toc-tic, 2)} seconds.")
    (price_estimate, price_stat) = MC(call_option_payoff(samples, K))
    implied_volatility_estimate = implied_volatility_call(S_0, K, 0, T, price_estimate)
    implied_volatility_lower = implied_volatility_call(S_0, K, 0, T, price_estimate - price_stat)
    implied_volatility_upper = implied_volatility_call(S_0, K, 0, T, price_estimate + price_stat)
    return implied_volatility_estimate, implied_volatility_lower, implied_volatility_upper


def sqrt_cov_matrix_rHeston(H=0.1, T=1., N=1000):
    """
    Computes the Cholesky decomposition of the covariance matrix for the rough Heston model with Hurst parameter H,
    final time T and N number of time steps. This is the Cholesky decomposition of the covariance matrix of the
    Gaussian vector (int_0^(T/N) (T/N - s)^(H-1/2) dW_s, ..., int_0^(T/N) (T-s)^(H-1/2) dW_s, W_(T/N)).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :return: The Cholesky decomposition of the covariance matrix
    """
    dt = T / N
    cov_matrix = np.empty(shape=(N + 1, N + 1))
    cov_matrix[:-1, :-1] = dt ** (2 * H) / (H + 0.5) * np.array([[(np.fmax(i, j) - 0.5) ** (H - 0.5) * (
                np.fmin(i, j) ** (H + 0.5) - (np.fmin(i, j) - 1) ** (H + 0.5)) for i in range(1, N + 1)] for j in
                                                                 range(1, N + 1)])
    for i in range(N):
        cov_matrix[i, i] = dt ** (2 * H) / (2 * H) * ((i + 1) ** (2 * H) - i ** (2 * H))
    cov_matrix[:-1, -1] = 1 / (H + 0.5) * dt ** (H + 0.5) * np.array(
        [((i + 1) ** (H + 0.5) - i ** (H + 0.5)) for i in range(N)])
    cov_matrix[-1, :-1] = cov_matrix[:-1, -1]
    cov_matrix[-1, -1] = dt
    (L, D, P) = scipy.linalg.ldl(cov_matrix)
    return np.dot(L, np.sqrt(np.fmax(D, np.zeros(shape=D.shape))))


def plot_rHeston(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1.):
    """
    Plots a realization of the rough Heston model together with the variance process.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motion driving the volatility and the Brownian motion driving the stock
    :param lambda_: Mean-reversion speed
    :param theta: Mean volatility
    :param nu: Volatility of volatility
    :param V_0: Initial volatility
    :param S_0: Initial stock price
    :return: Nothing, plots a realization of the rough Heston model together with the variance process.
    """
    dt = T / N
    sqrt_cov = sqrt_cov_matrix_rHeston(H, T, N)
    gaussian_increments = sqrt_cov.dot(np.random.normal(0, 1, size=(N + 1, N)))
    # CHECK IF THE RESULT IS AN (N+1, N) ARRAY!!! If so, i is for (t_i-s)^(H-1/2) and j is for int_t_(j-1)^t_j
    V = np.zeros(shape=(N + 1,))
    V[0] = V_0
    coeffs = lambda_ / (H + 0.5) * dt ** (H + 0.5) * np.arange(N, -1, -1) ** (H + 0.5)
    for i in range(1, N + 1):
        V[i] = V_0 + 1 / gamma(H + 0.5) * np.sum(
            (theta - V[:i]) * (coeffs[-i - 1:-1] - coeffs[-i:]) + nu * np.sqrt(V[:i]) * gaussian_increments[i - 1, :i])
        V[i] = np.fmax(V[i], 0)
    S = np.zeros(shape=(N + 1,))
    S[0] = S_0
    W_2_diff = np.random.normal(0, np.sqrt(dt), size=N)
    for i in range(N):
        S[i + 1] = S[i] * np.exp(
            np.sqrt(V[i]) * (rho * gaussian_increments[-1, i] + np.sqrt(1 - rho ** 2) * W_2_diff[i]) - V[i] * dt / 2)
    times = np.arange(N + 1) * dt
    plt.plot(times, S)
    plt.plot(times, V)
    plt.show()


def rHeston(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1., m=1000, rounds=1):
    """
    Computes m final stock values of the rough Heston model.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motion driving the volatility and the Brownian motion driving the stock
    :param lambda_: Mean-reversion speed
    :param theta: Mean volatility
    :param nu: Volatility of volatility
    :param V_0: Initial volatility
    :param S_0: Initial stock price
    :param m: Number of stock prices computed
    :param rounds: Actually generates m*rounds samples, but only m at a time. This is to avoid excessive memory usage.
    :return: An array of all the final stock prices
    """
    dt = T / N
    sqrt_cov = sqrt_cov_matrix_rHeston(H, T, N)
    S = np.empty(shape=(m*rounds))
    for round in range(rounds):
        gaussian_increments = np.array([sqrt_cov.dot(np.random.normal(0, 1, size=(N + 1, N))) for _ in range(m)])
        V = np.zeros(shape=(m, N + 1))
        V[:, 0] = V_0 * np.ones(m)
        coeffs = lambda_ / (H + 0.5) * dt ** (H + 0.5) * np.arange(N, -1, -1) ** (H + 0.5)
        for i in range(1, N + 1):
            V[:, i] = V[:, 0] + 1 / gamma(H + 0.5) * ((theta - V[:, :i]).dot(coeffs[-i - 1:-1] - coeffs[-i:]) + np.sum(
                nu * np.sqrt(V[:, :i]) * gaussian_increments[:, i - 1, :i], axis=1))
            V[:, i] = np.fmax(V[:, i], np.zeros(m))
        S_ = np.zeros(shape=(m, N + 1,))
        S_[:, 0] = S_0 * np.ones(m)
        W_2_diff = np.random.normal(0, np.sqrt(dt), size=(m, N))
        for i in range(N):
            S_[:, i + 1] = S_[:, i] * np.exp(
                np.sqrt(V[:, i]) * (rho * gaussian_increments[:, -1, i] + np.sqrt(1 - rho ** 2) * W_2_diff[:, i]) - V[:,
                                                                                                                    i] * dt / 2)
        S[round*m:(round+1)*m] = S_[:, -1]
    return S


def implied_volatility_call_rHeston(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1.,
                                    K=1., m=1000, rounds=1):
    """
    Computes the implied volatility of a European option under the rough Heston model.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motion driving the volatility and the Brownian motion driving the stock
    :param lambda_: Mean-reversion speed
    :param theta: Mean volatility
    :param nu: Volatility of volatility
    :param V_0: Initial volatility
    :param S_0: Initial stock price
    :param K: Strike price
    :param m: Number of stock prices computed
    :param rounds: Actually uses m*rounds samples, but only m at a time to avoid excessive memory usage
    :return: An array of all the final stock prices
    """
    tic = time.perf_counter()
    samples = rHeston(H, T, N, rho, lambda_, theta, nu, V_0, S_0, m, rounds)
    toc = time.perf_counter()
    print(f"Generating {m*rounds} rHeston samples with N={N} takes {np.round(toc-tic, 2)} seconds.")
    (price_estimate, price_stat) = MC(call_option_payoff(samples, K))
    implied_volatility_estimate = implied_volatility_call(S_0, K, 0, T, price_estimate)
    implied_volatility_lower = implied_volatility_call(S_0, K, 0, T, price_estimate - price_stat)
    implied_volatility_upper = implied_volatility_call(S_0, K, 0, T, price_estimate + price_stat)
    return implied_volatility_estimate, implied_volatility_lower, implied_volatility_upper


def sqrt_cov_matrix_rHeston_AK_mpmath(dt=1., nodes=None):
    """
    Computes the Cholesky decomposition of the covariance matrix of the approximation of the rough Heston model that
    was inspired by Alfonsi and Kebaier. The covariance matrix is the covariance matrix of the Gaussian vector
    (int_0^Delta t exp(-(Delta t - s) node_1) dW_s, ..., int_0^Delta t exp(-(Delta t - s) node_k) dW_s, W_Delta t).
    The computation is done with the mpmath library, the result is an mpmath matrix.
    :param dt: Time step size
    :param nodes: The quadrature nodes
    :return: The Cholesky decomposition of the covariance matrix (an mpmath matrix)
    """
    return sqrt_cov_matrix_rBergomi_AK_mpmath(dt, nodes)


def sqrt_cov_matrix_rHeston_AK(dt=1., nodes=None):
    """
    Computes the Cholesky decomposition of the covariance matrix of one step in the scheme that was inspired by
    Alfonsi and Kebaier, "Approximation of stochastic Volterra equations with kernels of completely monotone type".
    More precisely, computes the Cholesky decomposition of the covariance matrix of the following Gaussian vector:
    (int_0^dt exp(-x_1(dt-s)) dW_s, ..., int_0^dt exp(-x_k(dt-s)) dW_s, W_dt),
    where k = m*n is the number of nodes, and these nodes are determined by the m-point quadrature rule approximation
    of the rough kernel on n subintervals. Note that no weights of the quadrature rule are used. Returns a numpy array,
    but the computation is done using the mpmath library.
    :param dt: Time step size
    :param nodes: The nodes of the quadrature rule approximation
    :return: The Cholesky decomposition of the above covariance matrix
    """
    return sqrt_cov_matrix_rBergomi_AK(dt, nodes)


def plot_rHeston_AK(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1., m=1, n=20, a=1., b=1.):
    """
    Plots a realization of the stock price and variance process of the approximation of the rough Heston model that
    was inspired by Alfonsi and Kebaier.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motions driving the variance process and the stock price
    :param lambda_: Mean reversion rate of the variance process
    :param theta: Average variance
    :param nu: Volatility of volatility
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param m: Level of the quadrature rule
    :param n: Number of quadrature intervals
    :param a: Can be used to shift the left endpoint of the total quadrature interval
    :param b: Can be used to shift the right endpoint of the total quadrature interval
    :return: Nothing, plots a realization of the stock price and variance process
    """
    dt = T/N
    quad_rule = quadrature_rule_geometric(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights = quad_rule[1, :]
    k = len(nodes)
    sqrt_cov_matrix = sqrt_cov_matrix_rHeston_AK(dt, nodes)
    weights = np.array([float(weight) for weight in weights])
    nodes = np.array([float(node) for node in nodes])
    total_weight = np.sum(weights)
    V = np.empty(shape=(N+1,))
    V[0] = V_0
    Y = np.empty(shape=(N+1, k))
    Y[0, :] = V_0/total_weight * np.ones(k)
    W_1_trans = sqrt_cov_matrix.dot(np.random.normal(0, 1, size=(k+1, N)))
    exp_beta = np.exp(-dt*nodes)
    for i in range(N):
        mu = V_0/total_weight + lambda_*(theta-V[i])/nodes
        mean = exp_beta * Y[i, :] + (1-exp_beta) * mu
        Y[i+1, :] = mean + nu*np.sqrt(V[i])*W_1_trans[:k, i]
        # V[i+1] = np.fmax(np.dot(weights, Y[i+1, :]), 0)
        V[i+1] = np.dot(weights, Y[i+1])
        if V[i+1] < 0:
            factor = -V[i+1]/(V_0-V[i+1])
            V[i+1] = 0.
            Y[i+1, :] = V_0/total_weight - (V_0/total_weight - Y[i+1, :]) * (1-factor)
    W_1_diff = W_1_trans[-1, :]
    W_2_diff = np.random.normal(0, np.sqrt(dt), size=(N,))
    S = np.empty(shape=(N+1,))
    S[0] = S_0
    for i in range(N):
        S[i+1] = S[i] * np.exp(np.sqrt(V[i]) * (rho*W_1_diff[i] + np.sqrt(1-rho**2)*W_2_diff[i]) - V[i] * dt/2)
    times = dt * np.arange(N+1)
    plt.plot(times, S)
    plt.plot(times, V)
    plt.show()


def rHeston_AK(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1., m=1, n=20, a=1., b=1., M=1000, rounds=1):
    """
    Computes M final values of the stock price of the approximation of the rough Heston model that
    was inspired by Alfonsi and Kebaier.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motions driving the variance process and the stock price
    :param lambda_: Mean reversion rate of the variance process
    :param theta: Average variance
    :param nu: Volatility of volatility
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param m: Level of the quadrature rule
    :param n: Number of quadrature intervals
    :param a: Can be used to shift the left endpoint of the total quadrature interval
    :param b: Can be used to shift the right endpoint of the total quadrature interval
    :param M: Number of samples
    :param rounds: Acutally generates M*rounds samples, but always only M at once. Can be used to avoid using excessive memory
    :return: A numpy array containing the final stock values
    """
    dt = T/N
    quad_rule = quadrature_rule_geometric(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights = quad_rule[1, :]
    k = len(nodes)
    sqrt_cov_matrix = sqrt_cov_matrix_rHeston_AK(dt, nodes)
    weights = np.array([float(weight) for weight in weights])
    nodes = np.array([float(node) for node in nodes])
    total_weight = np.sum(weights)
    exp_beta = np.exp(-dt*nodes)
    S = np.empty(M*rounds)
    for round in range(rounds):
        V = V_0*np.ones(shape=(M,))
        Y = V_0/total_weight * np.ones(shape=(k, M))
        W_1_trans = sqrt_cov_matrix.dot(np.random.normal(0, 1, size=(M, k+1, N)))
        W_1_diff = W_1_trans[-1, :, :]
        W_2_diff = np.random.normal(0, np.sqrt(dt), size=(M, N))
        S_ = S_0 * np.ones(M)
        for i in range(N):
            S_ = S_ * np.exp(np.sqrt(V) * (rho*W_1_diff[:, i] + np.sqrt(1-rho**2)*W_2_diff[:, i]) - V * dt/2)
            mu = V_0/total_weight + lambda_*np.array([(theta-V)/nodes[j] for j in range(k)]).transpose()
            mean = np.array([exp_beta[node] * Y[node, :] + (1-exp_beta[node])*mu[:, node] for node in range(k)])
            Y = mean + nu*np.sqrt(V)*W_1_trans[:k, :, i]
            # V[i+1] = np.fmax(np.dot(weights, Y[i+1, :]), 0)
            V = np.dot(weights, Y)
            # factor = np.fmax(-V, 0)/np.fmax(V_0-V, V_0)  # The fmax in the denominator is to avoid division by 0
            V = np.fmax(V, 0)
            # Y = V_0/total_weight - (V_0/total_weight - Y) * (1-factor)
        S[round*M:(round+1)*M] = S_
    return S


def implied_volatility_call_rHeston_AK(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1.,
                                    K=1., m=1, n=20, a=1., b=1., M=1000, rounds=1):
    """
    Computes the implied volatility of a European option under the rough Heston model using the approximation inspired
    by Alfonsi and Kebaier.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time steps
    :param rho: Correlation between the Brownian motion driving the volatility and the Brownian motion driving the stock
    :param lambda_: Mean-reversion speed
    :param theta: Mean volatility
    :param nu: Volatility of volatility
    :param V_0: Initial volatility
    :param S_0: Initial stock price
    :param K: Strike price
    :param m: Level of the quadrature rule
    :param n: Number of quadrature intervals
    :param a: Can shift the left endpoint of the total quadrature interval
    :param b: Can shift the right endpoint of the total quadrature interval
    :param m: Number of stock prices computed
    :param rounds: Actually uses m*rounds samples, but only m at a time to avoid excessive memory usage
    :return: An array of all the final stock prices
    """
    tic = time.perf_counter()
    samples = rHeston_AK(H, T, N, rho, lambda_, theta, nu, V_0, S_0, m, n, a, b, M, rounds)
    toc = time.perf_counter()
    print(f"Generating {M*rounds} approximate rHeston samples with N={N} and n*m={n*m} takes {np.round(toc-tic, 2)} seconds.")
    (price_estimate, price_stat) = MC(call_option_payoff(samples, K))
    implied_volatility_estimate = implied_volatility_call(S_0, K, 0, T, price_estimate)
    implied_volatility_lower = implied_volatility_call(S_0, K, 0, T, price_estimate - price_stat)
    implied_volatility_upper = implied_volatility_call(S_0, K, 0, T, price_estimate + price_stat)
    return implied_volatility_estimate, implied_volatility_lower, implied_volatility_upper


'''
H = 0.1
m = 1
r = m
gamma = 0.5 - H
delta = H
number_time_steps = 1000000


# Plot G and the approximation of G
time_steps = np.array([float(i+100)/number_time_steps for i in range(0, number_time_steps)])
approximations = np.empty(shape=(number_time_steps, 4))
approximations[:, 0] = fractional_kernel(H, time_steps)

for n in (4, 8, 16):
    xi0 = n**(-r/gamma)
    xin = n**(r/delta)
    partition = np.array([xi0**(float(n-i)/n) * xin**(float(i)/n) for i in range(0, n+1)])
    quad_rule = quadrature_rule(H, m, partition)
    quad_nodes = quad_rule[0, :]
    quad_weights = quad_rule[1, :]
    approximations[:, int(np.log2(n))-1] = np.array([fractional_kernel_laplace(H, t, quad_nodes) for t in time_steps]).dot(quad_weights)

plt.plot(time_steps, approximations)
plt.show()



# Plot errors for different numbers of intervals (error = L^2-norm of difference between kernel and approximation)
time_steps = np.array([float(i+1)/number_time_steps for i in range(0, number_time_steps)])
true_function = fractional_kernel(H, time_steps)
print('The L^2-norm of G is {0}.'.format(np.sqrt(np.trapz(true_function**2, dx=1./number_time_steps))))
error_vector = np.empty(shape=(7, 2))

for n in (2, 4, 8, 16, 32, 64, 128):
    xi0 = n**(-r/gamma)
    xin = n**(r/delta)
    partition = np.array([xi0**(float(n-i)/n) * xin**(float(i)/n) for i in range(0, n+1)])
    quad_rule = quadrature_rule(H, m, partition)
    quad_nodes = quad_rule[0, :]
    quad_weights = quad_rule[1, :]
    approximation = np.array([fractional_kernel_laplace(H, t, quad_nodes) for t in time_steps]).dot(quad_weights)
    errors = (approximation - true_function)**2
    error = np.sqrt(np.trapz(errors, dx=1./number_time_steps))
    error_vector[int(np.log2(n)) - 1, 0] = error
    print('The error for m={0} and n={1} is {2}.'.format(m, n, error))

error_vector[:, 1] = np.array([32**m, 16**m, 8**m, 4**m, 2**m, 1, 0.5**m])*error_vector[-2, 0]
ns = np.array([2, 4, 8, 16, 32, 64, 128])
plt.loglog(ns, error_vector)
plt.show()
'''

'''
# Computing strong approximation errors for different values of n, m
H = mp.mpf(0.1)
T = mp.mpf(1)
number_paths = 100000

for m in (1,):
    for n in (8,):
        result = strong_error_fBm_approximation_MC(H, T, m, n, number_paths)
        print('m={0}, n={1}, error={2}, std={3}'.format(m, n, result[0], result[1]))
'''

'''
# Plot loglog-plots of the L^2 errors of the OU-approximations of fBm for different values of m and n.
m1_errors = np.array(
    [0.8580795741857218, 0.5933883792763698, 0.3506089358024185, 0.17621325270440755, 0.07995320204075512,
     0.03430761416752963, 0.014829003039174117, 0.006666020657043866])
m2_errors = np.array(
    [1.0274299516166492, 0.4857599152180374, 0.29783742948598957, 0.12454282877701886, 0.03481046890585258,
     0.005911347425846188, 0.0008669864303425754])

m1_vec = np.array([2., 4., 8., 16., 32., 64., 128., 256.])
c_arr = scipy.stats.linregress(np.log(m1_vec), np.log(m1_errors))
c11 = c_arr[0]
c01 = c_arr[1]
print(c11, c01)
plt.loglog(m1_vec, m1_errors, 'b-', label="error")
plt.loglog(m1_vec, np.exp(c01) * m1_vec ** c11, 'or--', label="regression")
plt.loglog(m1_vec, np.exp(c01) * m1_vec ** (-1), 'sg--', label="order 1")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

m2_vec = np.array([4., 8., 16., 32., 64., 128., 256.])
c_arr = scipy.stats.linregress(np.log(m2_vec), np.log(m2_errors))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(m2_vec, m2_errors, 'b-', label="error")
plt.loglog(m2_vec, np.exp(c02) * m2_vec ** c12, 'or--', label="regression")
plt.loglog(m2_vec, np.exp(c02) * m2_vec ** (-2), 'sg--', label="order 2")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

plt.loglog(m2_vec, m1_errors[1:], 'b-', label='m=1')
plt.loglog(m2_vec, m2_errors, 'r-', label='m=2')
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

m2_vec = np.array([16., 32., 64., 128., 256.])
c_arr = scipy.stats.linregress(np.log(m2_vec), np.log(m2_errors[2:]))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(m2_vec, m2_errors[2:], 'b-', label="error")
plt.loglog(m2_vec, np.exp(c02) * m2_vec ** c12, 'or--', label="regression")
plt.loglog(m2_vec, np.exp(c02) * m2_vec ** (-2), 'sg--', label="order 2")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()
'''

sigma_imp = implied_volatility_call(S=100., K=120., r=0.05, T=0.5, price=2.)
print('Test implied volatility: {0}'.format(sigma_imp))
print('Test call price: {0}'.format(BS_call_price(100., 120., sigma_imp, 0.05, 0.5)))
H = 0.07
T = 0.9
N = 100
eta = 1.9
V_0 = 0.235 ** 2
S_0 = 1.
rho = -0.9
r = 0.
M = 10000
K = 1.
m = 1
n = 20
a = 1.
b = 1.
S_0 = 1.
rounds = 2
k_vec = np.array([i / 100. for i in range(-40, 21)])
K_vec = S_0 * np.exp(k_vec)
n_vec = np.array([2, 4, 8])
(mi, lo, up) = implied_volatility_call_rBergomi_BFG(H, T, N, eta, V_0, S_0, rho, K_vec, M, rounds)
print("Vector of k:")
print(k_vec)
print("Implied volatility of call options of the rBergomi model, mean estimates:")
print(mi)
#print("Implied volatility of call options of the rBergomi model, lower estimates:")
#print(lo)
#print("Implied volatility of call options of the rBergomi model, upper estimates:")
#print(up)

for n in n_vec:
    print(f"n={n}")
    (mi2, lo2, up2) = implied_volatility_call_rBergomi_AK(H, T, N, eta, V_0, S_0, rho, K_vec, m, n, a, b, M, rounds)
    print("Implied volatility of call options of the approximate rBergomi model, mean estimates:")
    print(mi2)
    #print("Implied volatility of call options of the approximate rBergomi model, lower estimates:")
    #print(lo2)
    #print("Implied volatility of call options of the approximate rBergomi model, upper estimates:")
    #print(up2)
    #print("Mean error estimate:")
    mean_error = np.fabs(mi2-mi)
    #print(mean_error)
    #print("Upper error estimate:")
    upper_error = np.fmax(np.fabs(up2-lo), np.fabs(up-lo2))
    #print(upper_error)
    print("Mean error integral:")
    print(np.average(mean_error))
    print("Upper error integral:")
    print(np.average(upper_error))
    print("Mean error for k=0.1:")
    print(mean_error[50])
    print("Upper error for k=0.1:")
    print(upper_error[50])


k_vec = np.array([i / 100. for i in range(-40, 21)])

#plt.plot(k_vec, np.array([mi, lo, up, mi2, lo2, up2]).transpose())
#plt.show()


H = 0.1
T = 1.
N = 100
rho = -0.9
lambda_ = 0.3
theta = 0.02
nu = 0.3
V_0 = 0.02
S_0 = 1
K = 1.
m = 1
n = 80
a = 1.
b = 1.
M = 10000
rounds = 2
(mi, lo, up) = implied_volatility_call_rHeston(H, T, N, rho, lambda_, theta, nu, V_0, S_0, K_vec, M, rounds)
print("Vector of k:")
print(k_vec)
print("Implied volatility of call options of the rHeston model, mean estimates:")
print(mi)
#print("Implied volatility of call options of the rHeston model, lower estimates:")
#print(lo)
#print("Implied volatility of call options of the rHeston model, upper estimates:")
#print(up)
for n in n_vec:
    print(f"n={n}")
    (mi2, lo2, up2) = implied_volatility_call_rHeston_AK(H, T, N, rho, lambda_, theta, nu, V_0, S_0, K_vec, m, n, a, b, M, rounds)
    print("Implied volatility of call options of the approximate rHeston model, mean estimates:")
    print(mi2)
    #print("Implied volatility of call options of the approximate rHeston model, lower estimates:")
    #print(lo2)
    #print("Implied volatility of call options of the approximate rHeston model, upper estimates:")
    #print(up2)
    #print("Mean error estimate:")
    mean_error = np.fabs(mi2-mi)
    #print(mean_error)
    #print("Upper error estimate:")
    upper_error = np.fmax(np.fabs(up2-lo), np.fabs(up-lo2))
    #print(upper_error)
    print("Mean error integral:")
    print(np.average(mean_error))
    print("Upper error integral:")
    print(np.average(upper_error))
    print("Mean error for k=0.1:")
    print(mean_error[50])
    print("Upper error for k=0.1:")
    print(upper_error[50])
#plt.plot(k_vec, np.array([mi, lo, up, mi2, lo2, up2]).transpose())
#plt.show()
