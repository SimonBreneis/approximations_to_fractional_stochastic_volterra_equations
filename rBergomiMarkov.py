import numpy as np
from scipy.special import gamma
import RoughKernel as rk
import psutil
import rBergomiBackbone


def preprocess_rule(nodes, weights):
    """
    Ensures that the nodes and weights are in a standard format. This means that the nodes are ordered in increasing
    order, the smallest node is zero, and there is only one zero node.
    :param nodes: The nodes
    :param weights: The weights
    :return: The standardized nodes and weights
    """
    nodes, weights = rk.sort(nodes, weights)
    if nodes[0] > 1e-04:
        nodes_ = np.zeros(len(nodes) + 1)
        nodes_[1:] = nodes
        weights_ = np.zeros(len(weights) + 1)
        weights_[1:] = weights
        nodes, weights = nodes_, weights_
    n_zero_nodes = np.sum(nodes < 1e-04)
    nodes[:n_zero_nodes] = 0.
    if n_zero_nodes > 1:
        nodes_ = np.zeros(len(nodes) - n_zero_nodes + 1)
        nodes_[1:] = nodes[n_zero_nodes:]
        weights_ = np.zeros(len(nodes) - n_zero_nodes + 1)
        weights_[0] = np.sum(weights[:n_zero_nodes])
        weights_[1:] = weights[n_zero_nodes:]
        nodes, weights = nodes_, weights_
    return nodes, weights


def generate_samples(H, T, eta, V_0, rho, nodes, weights, M, N_time=1000, S_0=1.):
    """
    Computes M final stock prices of the Markovian approximation of the rough Bergomi model.
    :param H: Hurst parameter
    :param T: Final time
    :param N_time: Number of time discretization steps
    :param eta: Volatility of volatility
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the variance process and the stock price
    :param nodes: The nodes of the approximation
    :param weights: The weights of the approximation
    :param M: Number of final stock prices
    :return: An array containing the final stock prices
    """

    def sqrt_cov_matrix():
        """
        Computes the Cholesky decomposition of the covariance matrix of one step in the scheme that was inspired by
        Alfonsi and Kebaier, "Approximation of stochastic Volterra equations with kernels of completely monotone type".
        More precisely, computes the Cholesky decomposition of the covariance matrix of the following Gaussian vector:
        (W_dt, int_0^dt exp(-x_1(dt-s)) dW_s, ..., int_0^dt exp(-x_N(dt-s)) dW_s). Note that no weights of the
        quadrature rule are used.
        :return: The Cholesky decomposition of the above covariance matrix
        """
        nodes_ = nodes[1:]
        cov_matrix = np.empty((N, N))
        node_matrix = nodes_[:, None] + nodes_[None, :]
        exp_matrix = np.exp(-np.fmin(dt * node_matrix, 300))
        exp_matrix = np.where(exp_matrix < 1e-299, 0, exp_matrix)
        cov_matrix[1:, 1:] = (1 - exp_matrix) / node_matrix
        exp_vec = np.exp(-np.fmin(dt * nodes_, 300))
        exp_vec = np.where(exp_vec < 1e-299, 0, exp_vec)
        entry = (1 - exp_vec) / nodes_
        cov_matrix[0, 1:] = entry
        cov_matrix[1:, 0] = entry
        cov_matrix[0, 0] = dt
        return np.linalg.cholesky(cov_matrix)

    def variance_integral():
        """
        Computes an integral that appears in the rBergomi approximation that is inspired by Alfonsi and Kebaier.
        Computes the integral int_0^t ( sum_i w_i exp(-x_i (t-s)) )^2 ds.
        :return: The vector of integrals
        """
        times = np.arange(N_time) * dt
        w_0 = weights[0]
        nodes_ = nodes[1:]
        weights_ = weights[1:]
        weight_matrix = weights_[None, :] * weights_[:, None]
        node_matrix = nodes_[None, :] + nodes_[:, None]
        exp_matrix = np.exp(-np.fmin(node_matrix[None, ...] * times[:, None, None], 300))
        exp_matrix = np.where(exp_matrix < 1e-299, 0, exp_matrix)
        expression = (weight_matrix / node_matrix)[None, ...] * (1 - exp_matrix)
        result = np.sum(expression, axis=(1, 2))
        result = result + w_0 ** 2 * times
        exp_vec = np.exp(-np.fmin(nodes_[None, :] * times[:, None], 300))
        exp_vec = np.where(exp_vec < 1e-299, 0, exp_vec)
        result = result + 2 * w_0 * np.sum((weights_ / nodes_)[None, :] * (1 - exp_vec), axis=-1)
        return result

    dt = T / N_time
    N = len(nodes)
    sqrt_cov = sqrt_cov_matrix()
    active_nodes = nodes
    active_weights = weights
    if weights[0] == 0:  # this is faster if the zero-node is not used in the volatility process
        active_nodes = nodes[1:]
        active_weights = weights[1:]
    active_N = len(active_nodes)
    exp_vector = np.exp(-np.fmin(dt * active_nodes, 300))
    exp_vector = np.where(exp_vector < 1e-299, 0, exp_vector)
    eta_transformed = eta * np.sqrt(2 * H) * gamma(H + 0.5)
    variances = eta_transformed ** 2 / 2 * variance_integral()

    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = 3 * np.sqrt(M) * np.sqrt(N) * np.sqrt(N_time) * np.sqrt(np.array([0.]).nbytes)
    rounds = int(np.ceil((necessary_memory / available_memory) ** 2))
    M_ = int(np.ceil(M / rounds))
    S = np.empty(shape=(M_ * rounds))
    for rd in range(rounds):
        W_1_diff = np.einsum('ij,kjl->kil', sqrt_cov, np.random.normal(0, 1, size=(M_, N, N_time)))
        W_1_fBm = np.zeros(shape=(M_, active_N, N_time))
        for i in range(N_time-1):
            W_1_fBm[:, :, i+1] = exp_vector * W_1_fBm[:, :, i] + W_1_diff[:, -active_N:, i]
        V = V_0 * np.exp(np.dot(eta_transformed * active_weights, W_1_fBm) - variances)
        S_BM = rho * W_1_diff[:, 0, :] + np.random.normal(0, np.sqrt(1 - rho ** 2) * np.sqrt(dt), size=(M_, N_time))
        S[rd * M_:(rd + 1) * M_] = S_0 * np.exp(np.sum(np.sqrt(V) * S_BM - V * (dt / 2), axis=1))
    return S[:M]


def implied_volatility(H=0.1, T=1., eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, K=1., N=10, mode="paper",
                       rel_tol=1e-03, verbose=0):
    """
    Computes the implied volatility of the European call option under the rough Bergomi model, using the approximation
    inspired by Alfonsi and Kebaier.
    :param H: Hurst parameter
    :param T: Final time
    :param eta: Volatility of volatility
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param K: The strike price
    :param N: Dimension of the Markovian approximation
    :param mode: Kind of Markovian approximation
    :param rel_tol: Relative error tolerance
    :param verbose: Determines the number of intermediary results printed to the console
    :return: The implied volatility and a 95% confidence interval, in the form (estimate, lower, upper)
    """
    nodes, weights = rk.quadrature_rule(H, N, T, mode)
    nodes, weights = preprocess_rule(nodes, weights)
    return rBergomiBackbone.iv_eur_call(sample_generator=lambda T_, N_time, M: generate_samples(H=H, T=T_, eta=eta,
                                                                                                V_0=V_0, rho=rho,
                                                                                                nodes=nodes,
                                                                                                weights=weights, M=M,
                                                                                                N_time=N_time, S_0=S_0),
                                        S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose)
