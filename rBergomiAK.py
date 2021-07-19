import mpmath as mp
import numpy as np
import scipy
from scipy import special
import ComputationalFinance as cf
import RoughKernel as rk


def sqrt_cov_matrix_mpmath(dt, nodes):
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
    for i in range(k):
        for j in range(k):
            cov_matrix[i, j] = (1 - mp.exp(-dt * (nodes[i] + nodes[j]))) / (nodes[i] + nodes[j])
    for i in range(k):
        entry = (1 - mp.exp(-dt * nodes[i])) / nodes[i]
        cov_matrix[k, i] = entry
        cov_matrix[i, k] = entry
    cov_matrix[k, k] = dt
    return mp.cholesky(cov_matrix)


def sqrt_cov_matrix(dt, nodes):
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
    cov_root_mp = sqrt_cov_matrix_mpmath(dt, nodes)
    num = len(nodes) + 1
    cov_root = np.empty(shape=(num, num))
    for i in range(num):
        for j in range(num):
            cov_root[i, j] = float(cov_root_mp[i, j])
    return cov_root


def variance_integral(nodes, weights, t):
    """
    Computes an integral that appears in the rBergomi approximation that is inspired by Alfonsi and Kebaier. Computes
    the integral int_0^t ( sum_i w_i exp(-x_i (t-s)) )^2 ds. Takes a vector in t. May take as an input mpmath elements,
    but returns a numpy array.
    :param nodes: The nodes of the quadrature rule
    :param weights: The weights of the quadrature rule
    :param t: The current time
    :return: The vector of integrals
    """
    w_0 = weights[-1]
    nodes = nodes[:-1]
    weights = weights[:-1]
    weight_matrix = mp.matrix([[weight_i * weight_j for weight_j in weights] for weight_i in weights])
    node_matrix = mp.matrix([[node_i + node_j for node_j in nodes] for node_i in nodes])
    result = np.empty(shape=(len(t),))
    for t_ in range(len(t)):
        expression = mp.matrix([[weight_matrix[i, j] / node_matrix[i, j] * (
                1 - mp.exp(-node_matrix[i, j] * mp.mpf(t[t_]))) for j in range(len(nodes))] for i in
                                range(len(nodes))])
        expression_np = np.array([[expression[i, j] for j in range(len(nodes))] for i in range(len(nodes))])
        result[t_] = np.sum(expression_np.flatten())
        result[t_] = result[t_] + w_0**2 * t[t_]
        result[t_] = result[t_] + 2 * w_0 * np.sum(np.array([weights[i]/nodes[i] * (1-mp.exp(-nodes[i]*t[t_])) for i in range(len(nodes))]))
    return result


def generate_samples(H, T, eta, V_0, rho, nodes, weights, M, N_time=1000, S_0=1., rounds=1):
    """
    Computes M final stock prices of an approximation of the rough Bergomi model (approximation inspired by Alfonsi and
    Kebaier, with discretization points taken from Harms).
    :param H: Hurst parameter
    :param T: Final time
    :param N_time: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the variance process and the stock price
    :param nodes: The nodes of the approximation
    :param weights: The weights of the approximation
    :param M: Number of final stock prices
    :param rounds: Actually computes M*rounds samples, but only M at a time to avoid excessive memory usage.
    :return: An array containing the final stock prices
    """
    dt = T / N_time
    sqrt_cov = sqrt_cov_matrix(dt, nodes[:-1])
    weights = np.array([float(weight) for weight in weights])
    exp_vector = np.exp(-dt * np.array([float(node) for node in nodes]))
    eta_transformed = eta * np.sqrt(2 * H) * scipy.special.gamma(
        H + 0.5)  # the sqrt is in the model, the Gamma takes care of the c_H in the weights of the quadrature rule
    variances = eta_transformed ** 2 / 2 * variance_integral(nodes, weights, np.arange(N_time) * dt)
    S = np.empty(shape=(M * rounds))
    for rd in range(rounds):
        W_1_diff = np.array([sqrt_cov.dot(np.random.normal(0., 1., size=(len(nodes), N_time))) for _ in range(M)])
        # W_1_diff = np.transpose(sqrt_cov.dot(np.random.normal(0, 1, size=(M, len(nodes), N_time))), [1, 0, 2])
        # is slower
        W_1_fBm = np.zeros(shape=(M, len(nodes), N_time))
        for i in range(N_time-1):
            W_1_fBm[:, :, i+1] = exp_vector * W_1_fBm[:, :, i] + W_1_diff[:, :, i]
        W_1_fBm = eta_transformed * np.dot(weights, W_1_fBm)
        V = V_0 * np.exp(W_1_fBm - variances)
        W_2_diff = np.random.normal(0, np.sqrt(1 - rho ** 2) * np.sqrt(dt), size=(M, N_time))
        S[rd * M:(rd + 1) * M] = np.exp(np.log(S_0) + np.sum(np.sqrt(V) * (rho * W_1_diff[:, -1, :] + W_2_diff) - V / 2 * dt, axis=1))
    return S


def implied_volatility(H=0.1, T=1., N_time=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, K=1., N=10, M=1000,
                       rounds=1, mode="observation"):
    """
    Computes the implied volatility of the European call option under the rough Bergomi model, using the approximation
    inspired by Alfonsi and Kebaier.
    :param H: Hurst parameter
    :param T: Final time
    :param N_time: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param K: The strike price
    :param N: Total number of points in the quadrature rule, N=n*m
    :param M: Number of samples
    :param rounds: Actually uses M*rounds samples, but only m at a time to avoid excessive memory usage
    :param mode: If observation, use the values of the interpolation of the optimum. If theorem, use the values of the
    theorem.
    :return: The implied volatility and a 95% confidence interval, in the form (estimate, lower, upper)
    """
    quad_rule = rk.quadrature_rule_geometric_good(H, N, T, mode)
    nodes = quad_rule[0, :]
    weights = quad_rule[1, :]
    samples = generate_samples(H, T, eta, V_0, rho, nodes, weights, M, N_time, S_0, rounds)
    return cf.volatility_smile_call(samples, K, T, S_0)
