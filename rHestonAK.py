import time
import matplotlib.pyplot as plt
import numpy as np
import ComputationalFinance as cf
import QuadratureRulesRoughKernel
import rBergomiAK


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
    return rBergomiAK.sqrt_cov_matrix_rBergomi_AK_mpmath(dt, nodes)


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
    return rBergomiAK.sqrt_cov_matrix_rBergomi_AK(dt, nodes)


def plot_rHeston_AK(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1., m=1, n=20, a=1.,
                    b=1.):
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
    dt = T / N
    quad_rule = QuadratureRulesRoughKernel.quadrature_rule_geometric(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights = quad_rule[1, :]
    k = len(nodes)
    sqrt_cov_matrix = sqrt_cov_matrix_rHeston_AK(dt, nodes)
    weights = np.array([float(weight) for weight in weights])
    nodes = np.array([float(node) for node in nodes])
    total_weight = np.sum(weights)
    V = np.empty(shape=(N + 1,))
    V[0] = V_0
    Y = np.empty(shape=(N + 1, k))
    Y[0, :] = V_0 / total_weight * np.ones(k)
    W_1_trans = sqrt_cov_matrix.dot(np.random.normal(0, 1, size=(k + 1, N)))
    exp_beta = np.exp(-dt * nodes)
    for i in range(N):
        mu = V_0 / total_weight + lambda_ * (theta - V[i]) / nodes
        mean = exp_beta * Y[i, :] + (1 - exp_beta) * mu
        Y[i + 1, :] = mean + nu * np.sqrt(V[i]) * W_1_trans[:k, i]
        # V[i+1] = np.fmax(np.dot(weights, Y[i+1, :]), 0)
        V[i + 1] = np.dot(weights, Y[i + 1])
        if V[i + 1] < 0:
            # factor = -V[i+1]/(V_0-V[i+1])
            V[i + 1] = 0.
            # Y[i+1, :] = V_0/total_weight - (V_0/total_weight - Y[i+1, :]) * (1-factor)
    W_1_diff = W_1_trans[-1, :]
    W_2_diff = np.random.normal(0, np.sqrt(dt), size=(N,))
    S = np.empty(shape=(N + 1,))
    S[0] = S_0
    for i in range(N):
        S[i + 1] = S[i] * np.exp(
            np.sqrt(V[i]) * (rho * W_1_diff[i] + np.sqrt(1 - rho ** 2) * W_2_diff[i]) - V[i] * dt / 2)
    times = dt * np.arange(N + 1)
    plt.plot(times, S)
    plt.plot(times, V)
    plt.show()


def rHeston_AK(H=0.1, T=1., N=1000, rho=-0.9, lambda_=0.3, theta=0.02, nu=0.3, V_0=0.02, S_0=1., m=1, n=20, a=1., b=1.,
               M=1000, rounds=1):
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
    :param rounds: Actually generates M*rounds samples, but always only M at once. Can be used to avoid using excessive
                   memory
    :return: A numpy array containing the final stock values
    """
    dt = T / N
    quad_rule = QuadratureRulesRoughKernel.quadrature_rule_geometric(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights = quad_rule[1, :]
    k = len(nodes)
    sqrt_cov_matrix = sqrt_cov_matrix_rHeston_AK(dt, nodes)
    weights = np.array([float(weight) for weight in weights])
    nodes = np.array([float(node) for node in nodes])
    total_weight = np.sum(weights)
    exp_beta = np.exp(-dt * nodes)
    S = np.empty(M * rounds)
    for rd in range(rounds):
        V = V_0 * np.ones(shape=(M,))
        Y = V_0 / total_weight * np.ones(shape=(k, M))
        W_1_trans = sqrt_cov_matrix.dot(np.random.normal(0, 1, size=(M, k + 1, N)))
        W_1_diff = W_1_trans[-1, :, :]
        W_2_diff = np.random.normal(0, np.sqrt(dt), size=(M, N))
        S_ = S_0 * np.ones(M)
        for i in range(N):
            S_ = S_ * np.exp(np.sqrt(V) * (rho * W_1_diff[:, i] + np.sqrt(1 - rho ** 2) * W_2_diff[:, i]) - V * dt / 2)
            mu = V_0 / total_weight + lambda_ * np.array([(theta - V) / nodes[j] for j in range(k)]).transpose()
            mean = np.array([exp_beta[node] * Y[node, :] + (1 - exp_beta[node]) * mu[:, node] for node in range(k)])
            Y = mean + nu * np.sqrt(V) * W_1_trans[:k, :, i]
            # V[i+1] = np.fmax(np.dot(weights, Y[i+1, :]), 0)
            V = np.dot(weights, Y)
            # factor = np.fmax(-V, 0)/np.fmax(V_0-V, V_0)  # The fmax in the denominator is to avoid division by 0
            V = np.fmax(V, 0)
            # Y = V_0/total_weight - (V_0/total_weight - Y) * (1-factor)
        S[rd * M:(rd + 1) * M] = S_
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
    :param M: Number of stock prices computed
    :param rounds: Actually uses m*rounds samples, but only m at a time to avoid excessive memory usage
    :return: An array of all the final stock prices
    """
    tic = time.perf_counter()
    samples = rHeston_AK(H, T, N, rho, lambda_, theta, nu, V_0, S_0, m, n, a, b, M, rounds)
    toc = time.perf_counter()
    print(
        f"Generating {M * rounds} approximate rHeston samples with N={N} and n*m={n * m} takes {np.round(toc - tic, 2)}"
        f" seconds.")
    return cf.volatility_smile_call(samples, K, T, S_0)
