import time
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import scipy
from scipy import special
import ComputationalFinance as cf
import QuadratureRulesRoughKernel


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
    quad_rule = QuadratureRulesRoughKernel.quadrature_rule_geometric(H, m, n, a, b)
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


def rBergomi_AK(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, m=1, n=10, a=1., b=1., M=1000,
                rounds=1):
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
    quad_rule = QuadratureRulesRoughKernel.quadrature_rule_geometric(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights_mp = quad_rule[1, :]
    sqrt_cov = sqrt_cov_matrix_rBergomi_AK(dt, nodes)
    weights = np.array([float(weight) for weight in weights_mp])
    exp_vector = np.exp(np.array([-float(node) * dt for node in nodes]))
    eta_transformed = eta * np.sqrt(2 * H) * scipy.special.gamma(
        H + 0.5)  # the sqrt is in the model, the Gamma takes care of the c_H in the weights of the quadrature rule
    times = np.arange(N + 1) * dt
    variances = eta_transformed ** 2 / 2 * rBergomi_AK_variance_integral(nodes, weights, times)
    S = np.empty(shape=(M * rounds))
    for rd in range(rounds):
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
                np.sqrt(V[:, i - 1]) * (rho * W_1_diff[:, i - 1] + np.sqrt(1 - rho ** 2) * W_2_diff[:, i - 1])
                - V[:, i - 1] / 2 * dt)
        S[rd * M:(rd + 1) * M] = S_
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
    print(f"Generating {M * rounds} approximate rBergomi samples with N={N} and n*m={n * m} takes "
          f"{np.round(toc - tic, 2)} seconds.")
    return cf.volatility_smile_call(samples, K, T, S_0)
