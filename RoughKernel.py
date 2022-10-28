import numpy as np
import mpmath as mp
from scipy.optimize import minimize
from scipy.special import gamma, gammainc
import orthopy
import quadpy


def sort(a, b):
    """
    Sorts two numpy arrays jointly according to the ordering of the first.
    :param a: First numpy array
    :param b: Second numpy array
    :return: Sorted numpy arrays
    """
    perm = np.argsort(a)
    return a[perm], b[perm]


def mp_to_np(x):
    """
    Converts a mpmath matrix to a numpy array.
    :param x: The mpmath matrix to convert
    :return: The converted numpy array.
    """
    y = np.array(x.tolist())
    shape = y.shape
    y = y.flatten()
    y = np.array([float(z) for z in y])
    if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
        return y
    return y.reshape(shape)


def exp_underflow(x):
    """
    Computes exp(-x) while avoiding underflow errors.
    :param x: Float of numpy array
    :return: exp(-x)
    """
    if isinstance(x, np.ndarray):
        if x.dtype == np.int:
            x = x.astype(np.float)
        eps = np.finfo(x.dtype).tiny
    else:
        if isinstance(x, int):
            x = float(x)
        eps = np.finfo(x.__class__).tiny
    log_eps = -np.log(eps) / 2
    result = np.exp(-np.fmin(x, log_eps))
    result = np.where(x > log_eps, 0, result)
    return result


def fractional_kernel(H, t):
    """
    The fractional kernel.
    :param H: Hurst parameter
    :param t: Time, may also be a numpy array
    :return: The value of the fractional kernel at t
    """
    return t ** (H - 0.5) / gamma(H + 0.5)


def kernel_norm(H, T):
    """
    Returns the L^2-norm of the fractional kernel.
    :param H: Hurst parameter
    :param T: Final time
    :return: The L^2-norm (square root has been taken) of the fractional kernel
    """
    return T ** H / (np.sqrt(2 * H) * gamma(H + 0.5))


def c_H(H):
    """
    Returns the constant c_H.
    :param H: Hurst parameter
    :return: c_H
    """
    return 1. / (gamma(0.5 + H) * gamma(0.5 - H))


def fractional_kernel_laplace(H, t, nodes):
    """
    The Laplace transform of the fractional kernel.
    :param H: Hurst parameter
    :param t: Time, may also be a numpy array
    :param nodes: Laplace transform argument, may also be a numpy array
    :return: The Laplace transform. May be a number, a one-dimensional or a two-dimensional numpy array depending on
        the shape of t and nodes. If both t and nodes are a numpy array, the tensor product that we take is
        nodes x time
    """
    if isinstance(t, np.ndarray) and isinstance(nodes, np.ndarray):
        return c_H(H) * np.exp(-np.tensordot(nodes, t, axes=0))
    return c_H(H) * np.exp(-nodes * t)


def fractional_kernel_approximation(H, t, nodes, weights):
    """
    Returns the Markovian approximation of the fractional kernel.
    :param H: Hurst parameter
    :param t: Time points
    :param nodes: Nodes of the quadrature rule
    :param weights: Weights of the quadrature rule
    :return: The approximated kernel using nodes and weights at times t (a numpy array)
    """
    return 1 / c_H(H) * np.tensordot(fractional_kernel_laplace(H, t, nodes), weights, axes=([0, 0]))


def AK_improved_rule(H, N, K=None, T=1):
    """
    The quadrature rule from Alfonsi and Kebaier in Table 6, left column.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param K: Cutoff point where the regime changes
    :param T: Final time
    :return: The quadrature rule in the form nodes, weights
    """
    N = N // 2

    if K is None:
        K = N ** 0.8

    def AK_initial_guess(A_):
        partition = np.empty(2 * N + 1)
        partition[:N + 1] = np.linspace(0, K, N + 1)
        partition[N + 1:] = K * A_ ** np.arange(1, N + 1)
        a = partition ** (1.5 - H)
        b = partition ** (0.5 - H)
        nodes_ = (0.5 - H) / (1.5 - H) * (a[1:] - a[:-1]) / (b[1:] - b[:-1])
        weights_ = c_H(H) / (0.5 - H) * (b[1:] - b[:-1])
        return nodes_, weights_

    def error_func(A_):
        nodes_, weights_ = AK_initial_guess(A_[0])
        return error(H, nodes_, weights_, T)

    res = minimize(fun=lambda A_: error_func(A_), x0=np.array([1.2]), bounds=((0, None),))
    A = res.x
    nodes, weights = AK_initial_guess(A[0])

    res = minimize(fun=lambda x: error(H, nodes, x * weights, T), x0=np.array([1]), bounds=((0, None),))
    return nodes, res.x * weights


def Gaussian_parameters(H, N, T, mode):
    """
    Returns the parameters m, n, a, b of the geometric quadrature rule.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem
    :return: Quadrature level m, number of intervals n, left point a, right point b
    """
    N = N - 1
    a = 0.
    b = 0.
    beta = 0.
    A = mp.sqrt(1 / H + 1 / (1.5 - H))

    if mode == "theorem":
        beta = 0.4275
        alpha = 1.06418
        gamma_ = np.exp(alpha * beta)
        exponent = 1 / (3 * gamma_ / (8 * (gamma_ - 1)) + 6 * H - 4 * H * H)
        temp_1 = ((9 - 6 * H) / (2 * H)) ** (gamma_ / (8 * (gamma_ - 1)))
        temp_2 = 5 * np.pi ** 3 * gamma_ * (gamma_ - 1) * A ** (2 - 2 * H) * float(N) ** (1 - H) / (beta ** (2 - 2 * H))
        base_0 = temp_1 * (temp_2 * (3 - 2 * H) / (768 * H)) ** (2 * H)
        a = -mp.log(T ** (-1) * base_0 ** exponent * mp.exp(-alpha / ((1.5 - H) * A) * np.sqrt(N)))
        base_n = temp_1 * (temp_2 / 1152) ** (2 * H - 3)
        b = mp.log(T ** (-1) * base_n ** exponent * mp.exp(alpha / (H * A) * np.sqrt(N)))
    elif mode == "observation":
        beta = 0.9
        alpha = 1.8
        a = -mp.log(0.65 * 1 / T * mp.exp(3.1 * H) * mp.exp(-alpha / ((1.5 - H) * A) * np.sqrt(N)))
        b = mp.log(1 / T * mp.exp(3 * H ** (-0.4)) * mp.exp(alpha / (H * A) * np.sqrt(N)))

    m = int(np.fmax(np.round(float(beta / A * np.sqrt(N))), 1))
    n = int(np.round(N / m))
    return m, n, a, b


def Gaussian_no_zero_node(H, m, n, a, b):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals. The result is two instances of mp.matrix with mp.mpf entries. Does not
    contain a node at 0.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of subintervals
    :param a: a = - log(xi_0)
    :param b: b = log(xi_n)
    :return: All the nodes and weights
    """

    def quadrature_rule_interval_Gaussian(y, z):
        """
        Returns the nodes and weights of the Gauss quadrature rule level m for the fractional weight function on [y, z].
        :param y: Left end of interval
        :param z: Right end of interval
        :return: The nodes and weights
        """
        c = mp.mpf(c_H(float(H)))
        moments = np.array(
            [mp.mpf(c / (mp.mpf(k) + mp.mpf(0.5) - H) * (
                    z ** (mp.mpf(k) + mp.mpf(0.5) - H) - y ** (mp.mpf(k) + mp.mpf(0.5) - H))) for k in
             range(2 * m)])
        alpha, beta, int_1 = orthopy.tools.chebyshev(moments)
        points, weights_ = quadpy.tools.scheme_from_rc(alpha, beta, int_1, mode="mpmath")
        return mp.matrix(points.tolist()), mp.matrix(weights_.tolist())

    mp.mp.dps = int(np.fmax(a + b + 50, 50))
    partition = np.array([mp.exp(-a + (a + b) * (mp.mpf(i) / mp.mpf(n))) for i in range(n + 1)])
    nodes = mp.matrix(m * n, 1)
    weights = mp.matrix(m * n, 1)
    for i in range(n):
        new_nodes, new_weights = quadrature_rule_interval_Gaussian(partition[i], partition[i + 1])
        nodes[m * i:m * (i + 1), 0] = new_nodes
        weights[m * i:m * (i + 1), 0] = new_weights
    return nodes, weights


def Gaussian_error_and_zero_weight(H, m, n, a, b, T):
    """
    Computes an error estimate of the L^2-norm of the difference between the rough kernel and its approximation
    on [0, T]. The approximation contains a constant/Brownian term.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of quadrature intervals
    :param a: xi_0 = e^(-a)
    :param b: xi_n = e^b
    :param T: Final time
    :return: An error estimate and the optimal weight for the constant term: error, weight
    """
    if n * m == 0:
        c = T ** (2 * H) / (2 * H * gamma(H + 0.5) ** 2)
        b = - 2 * T ** (H + 0.5) / gamma(H + 1.5)
        a = T
        w_0 = -b / (2 * a)
        return np.sqrt(np.fmax(a * w_0 * w_0 + b * w_0 + c, 0.)), w_0

    nodes, weights = Gaussian_no_zero_node(H, m, n, mp.mpf(a), mp.mpf(b))
    nodes, weights = mp_to_np(nodes), mp_to_np(weights)
    c = error(H, nodes, weights, T)
    b = np.sum(weights / nodes * (1-np.exp(-np.fmin(nodes * T, 300))))
    b -= T ** (H + 0.5) / gamma(H + 1.5)
    b *= 2
    a = T
    w_0 = -b / (2 * a)
    return np.sqrt(np.fmax(a * w_0 * w_0 + b * w_0 + c, 0.)), w_0


def Gaussian_rule(H, N, T, mode='observation', optimal_weights=False):
    """
    Returns the nodes and weights of the Gaussian rule with roughly N nodes.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem
    :param optimal_weights: If True, uses the optimal weights for the kernel error
    :return: The nodes and weights, ordered by the size of the nodes
    """
    if isinstance(T, np.ndarray):
        T = T[-1]
    if N == 1:
        w_0 = Gaussian_error_and_zero_weight(H, 0, 0, 0, 0, T)[1]
        return np.array([0]), np.array([w_0])

    m, n, a, b = Gaussian_parameters(H, N, T, mode)
    w_0 = Gaussian_error_and_zero_weight(H, m, n, a, b, T)[1]
    nodes = mp.matrix(m * n + 1, 1)
    weights = mp.matrix(m * n + 1, 1)
    nodes[0, 0] = mp.mpf(0)
    weights[0, 0] = mp.mpf(w_0)
    nodes_, weights_ = Gaussian_no_zero_node(H, m, n, a, b)
    nodes[1:, 0] = nodes_
    weights[1:, 0] = weights_
    nodes, weights = mp_to_np(nodes), mp_to_np(weights)
    if optimal_weights:
        weights = error_optimal_weights(H=H, T=T, nodes=nodes, output='error')[1]
    return nodes, weights


def error(H, nodes, weights, T, output='error'):
    """
    Computes an error estimate of the squared L^2-norm of the difference between the rough kernel and its approximation
    on [0, T].
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param weights: The weights of the approximation
    :param T: Final time, may also be a numpy array
    :param output: If error, returns the error. If gradient, returns the error and the gradient of the error
    :return: An error estimate
    """
    '''
    if np.amin(nodes) < 0 or np.amin(weights) < 0:
        return 1e+10
    '''
    nodes = np.fmin(np.fmax(nodes, 1e-08), 1e+150)
    weights = np.fmin(weights, 1e+75)
    weight_matrix = np.outer(weights, weights)
    summand = T ** (2 * H) / (2 * H * gamma(H + 0.5) ** 2)
    node_matrix = nodes[:, None] + nodes[None, :]
    if isinstance(T, np.ndarray):
        gamma_ints = gammainc(H + 0.5, np.outer(T, nodes))
        nmT = np.einsum('i,jk->ijk', T, node_matrix)
        exp_node_matrix = exp_underflow(nmT)
        sum_1 = np.sum((weight_matrix / node_matrix)[None, :, :] * (1 - exp_node_matrix), axis=(1, 2))
        sum_2 = 2 * np.sum((weights / nodes ** (H + 0.5))[None, :] * gamma_ints, axis=1)
    else:
        gamma_ints = gammainc(H + 0.5, nodes * T)
        nmT = node_matrix * T
        exp_node_matrix = exp_underflow(nmT)
        sum_1 = np.sum(weight_matrix / node_matrix * (1 - exp_node_matrix))
        sum_2 = 2 * np.sum(weights / nodes ** (H + 0.5) * gamma_ints)
    err = summand + sum_1 - sum_2
    if output == 'error' or output == 'err':
        return err

    N = len(nodes)
    if isinstance(T, np.ndarray):
        grad = np.empty((len(T), 2 * N))
        exp_node_vec = exp_underflow(np.outer(T, nodes)) / nodes[None, :]
        first_summands = (weight_matrix / (node_matrix * node_matrix))[None, :] * (1 - (1 + nmT) * exp_node_matrix)
        second_summands = weights[None, :] * ((T ** (H + 1 / 2) / gamma(H + 1 / 2))[:, None] * exp_node_vec - (
                ((H + 1 / 2) * nodes ** (-H - 3 / 2))[None, :] * gamma_ints))
        grad[:, :N] = -2 * np.sum(first_summands, axis=2) - 2 * second_summands
        third_summands = np.einsum('ijk,k->ij', ((1 - exp_node_matrix) / node_matrix[None, :, :]), weights)
        forth_summands = (nodes ** (-(H + 1 / 2)))[None, :] * gamma_ints
        grad[:, N:] = 2 * third_summands - 2 * forth_summands
    else:
        grad = np.empty(2 * N)
        exp_node_vec = np.zeros(N)
        indices = nodes * T < 300
        exp_node_vec[indices] = np.exp(- T * nodes[indices]) / nodes[indices]
        first_summands = weight_matrix / (node_matrix * node_matrix) * (1 - (1 + nmT) * exp_node_matrix)
        second_summands = weights * (T ** (H + 1 / 2) / gamma(H + 1 / 2) * exp_node_vec - (H + 1 / 2) * nodes ** (
                -H - 3 / 2) * gamma_ints)
        grad[:N] = -2 * np.sum(first_summands, axis=1) - 2 * second_summands
        third_summands = ((1 - exp_node_matrix) / node_matrix) @ weights
        forth_summands = nodes ** (-(H + 1 / 2)) * gamma_ints
        grad[N:] = 2 * third_summands - 2 * forth_summands
    return err, grad


def error_optimal_weights(H, T, nodes, output='error'):
    """
    Computes an error estimate of the squared L^2-norm of the difference between the rough kernel and its approximation
    on [0, T]. Uses the best possible weights given the nodes specified.
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param output: If error, returns the error and the optimal weights. If gradient, returns the error, the gradient
        (of the nodes only), and the optimal weights. If hessian, returns the error, the gradient, the Hessian, and
        the optimal weights
    :param T: Final time, may also be a numpy array
    :return: An error estimate
    """
    if len(nodes) == 1:
        node = np.fmax(1e-04, nodes[0])
        gamma_1 = gamma(H + 0.5)

        if isinstance(T, np.ndarray):
            nT = node * T
            gamma_ints = gammainc(H + 0.5, nT)
            exp_node_matrix = exp_underflow(2 * nT)
            exp_node_vec = exp_underflow(nT)
            A = (1 - exp_node_matrix) / (2 * node)
            b = -2 * gamma_ints / node ** (H + 0.5)
            c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
            v = b / A
            err = c - 0.25 * b * v
            opt_weights = -0.5 * v
            if len(opt_weights.shape) > 1:
                opt_weights = opt_weights[-1, ...]
            if output == 'error' or output == 'err':
                return err, opt_weights

            A_grad = (-1 + (1 + 2 * nT) * exp_node_matrix) / (2 * node) ** 2
            b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec[None, :] / gamma_1 - (H + 0.5) * gamma_ints) \
                / node ** (H + 1.5)
            grad = 0.5 * A_grad * v ** 2 - 0.5 * b_grad * v
            if output == 'gradient' or output == 'grad':
                return err, grad, opt_weights

            A_hess = 2 * (1 - (1 + 2 * nT + 2 * nT ** 2) * exp_node_matrix) / (8 * node ** 3)
            b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
                    H + 1.5) * gamma_ints) / nodes ** (H + 2.5)
            U = b_grad / A
            Y = 2 * A_grad * v
            hess = 0.5 * (2 * Y * U - Y ** 2 / A + 2 * A_hess * v ** 2 - b_hess * v - b_grad * U)
            return err, grad, hess, opt_weights

        gamma_ints = gammainc(H + 0.5, node * T)
        exp_node_matrix = exp_underflow(2 * node * T)
        exp_node_vec = exp_underflow(node * T)
        A = (1 - exp_node_matrix) / (2 * node)
        b = -2 * gamma_ints / node ** (H + 0.5)
        c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
        v = b / A
        err = c - 0.25 * b * v
        opt_weight = np.array([-0.5 * v])
        if output == 'error' or output == 'err':
            return err, opt_weight

        A_grad = (-1 + (1 + 2 * node * T) * exp_node_matrix) / (4 * node ** 2)
        b_grad = -2 * ((node * T) ** (H + 0.5) * exp_node_vec / gamma_1 - (H + 0.5) * gamma_ints) / node ** (H + 1.5)
        grad = 0.5 * (A_grad * v - b_grad) * v
        if output == 'gradient' or output == 'grad':
            return err, grad, opt_weight

        A_hess = 2 * (1 - (1 + 2 * node * T + 2 * (node * T) ** 2) * exp_node_matrix) / (8 * node ** 3)
        b_hess = -2 * (-((node * T) ** (H + 1.5) + (H + 1.5) * (node * T) ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
                H + 1.5) * gamma_ints) / node ** (H + 2.5)
        U = b_grad / A
        Y = 2 * A_grad * v
        hess = 0.5 * (2 * Y * U - Y ** 2 / A + 2 * A_hess * v ** 2 - b_hess * v - b_grad * U)
        return err, grad, hess, opt_weight

    def invert_permutation(p):
        s = np.empty_like(p)
        s[p] = np.arange(p.size)
        return s

    perm = np.argsort(nodes)
    nodes = nodes[perm]
    nodes[0] = np.fmax(1e-04, nodes[0])
    for i in range(len(nodes) - 1):
        if 1.01 * nodes[i] > nodes[i + 1]:
            nodes[i + 1] = nodes[i] * 1.01
    nodes = nodes[invert_permutation(perm)]

    node_matrix = nodes[:, None] + nodes[None, :]
    gamma_1 = gamma(H + 0.5)

    if isinstance(T, np.ndarray):
        nT = np.outer(T, nodes)
        nmT = np.einsum('i,jk->ijk', T, node_matrix)
        gamma_ints = gammainc(H + 0.5, nT)
        exp_node_matrix = exp_underflow(nmT)
        exp_node_vec = exp_underflow(nT)
        A = (1 - exp_node_matrix) / node_matrix[None, :, :]
        b = -2 * gamma_ints / nodes[None, :] ** (H + 0.5)
        c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
        try:
            v = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            v = np.empty((len(T), len(nodes)))
            for i in range(len(T)):
                try:
                    v[i, :] = np.linalg.solve(A[i, ...], b[i, ...])
                except np.linalg.LinAlgError:
                    v[i, :] = np.linalg.lstsq(A[i, ...], b[i, ...], rcond=None)[0]
        err = c - 0.25 * np.sum(b * v, axis=1)
        opt_weights = -0.5 * v
        if len(opt_weights.shape) > 1:
            opt_weights = opt_weights[-1, ...]
        if output == 'error' or output == 'err':
            return err, opt_weights

        def mvp(A_, b_):
            return np.sum(A_ * b_[:, None, :], axis=-1)

        A_grad = (-1 + (1 + nmT) * exp_node_matrix[None, :, :]) / node_matrix[None, :, :] ** 2
        b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec[None, :] / gamma_1 - (H + 0.5) * gamma_ints) \
            / nodes[None, :] ** (H + 1.5)
        grad = 0.5 * v * mvp(A_grad, v) - 0.5 * b_grad * v
        if output == 'gradient' or output == 'grad':
            return err, grad, opt_weights

        def diagonalize(x):
            new_x = np.empty((x.shape[0], x.shape[1], x.shape[1]))
            for j in range(x.shape[0]):
                new_x[j, :, :] = np.diag(x[j, :])
            return new_x

        def trans(x):
            return np.transpose(x, (0, 2, 1))

        A_hess = 2 * (1 - (1 + nmT + nmT ** 2 / 2) * exp_node_matrix[None, :, :]) / node_matrix[None, :, :] ** 3
        b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
                    H + 1.5) * gamma_ints) / nodes[None, :] ** (H + 2.5)
        try:
            U = np.linalg.solve(A, diagonalize(b_grad))
        except np.linalg.LinAlgError:
            diag_b = diagonalize(b_grad)
            U = np.empty((len(T), len(nodes), len(nodes)))
            for i in range(len(T)):
                for j in range(len(nodes)):
                    try:
                        U[i, j, :] = np.linalg.solve(A[i, ...], diag_b[i, j, :])
                    except np.linalg.LinAlgError:
                        U[i, j, :] = np.linalg.lstsq(A[i, ...], diag_b[i, j, :])[0]
        Y = diagonalize(mvp(A_grad, v)) + A_grad * v[:, None, :]
        YTU = trans(Y) @ U
        hess = 0.5 * (YTU - trans(np.linalg.solve(A, Y)) @ Y + diagonalize(v * mvp(A_hess, v))
                      + v[:, None, :] * v[:, :, None] * A_hess - diagonalize(b_hess * v) - b_grad[:, :, None] * U
                      + trans(YTU))
        return err, grad, hess, opt_weights

    nT = nodes * T
    nmT = node_matrix * T
    gamma_ints = gammainc(H + 0.5, nT)
    exp_node_matrix = exp_underflow(nmT)
    A = (1 - exp_node_matrix) / node_matrix
    b = -2 * gamma_ints / nodes ** (H + 0.5)
    c = T ** (2 * H) / (2 * H * gamma_1 ** 2)
    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        v = np.linalg.lstsq(A, b, rcond=None)[0]
    err = c - 0.25 * np.dot(b, v)
    opt_weights = -0.5 * v
    if output == 'error' or output == 'err':
        return err, opt_weights

    exp_node_vec = exp_underflow(nT)
    A_grad = (-1 + (1 + nmT) * exp_node_matrix) / node_matrix ** 2
    b_grad = -2 * (nT ** (H + 0.5) * exp_node_vec / gamma_1 - (H + 0.5) * gamma_ints) / nodes ** (H + 1.5)
    grad = 0.5 * v * (A_grad @ v) - 0.5 * b_grad * v
    if output == 'gradient' or output == 'grad':
        return err, grad, opt_weights

    A_hess = 2 * (1 - (1 + nmT + nmT ** 2 / 2) * exp_node_matrix) / node_matrix ** 3
    b_hess = -2 * (-(nT ** (H + 1.5) + (H + 1.5) * nT ** (H + 0.5)) * exp_node_vec / gamma_1 + (H + 0.5) * (
            H + 1.5) * gamma_ints) / nodes ** (H + 2.5)
    try:
        U = np.linalg.solve(A, np.diag(b_grad))
    except np.linalg.LinAlgError:
        U = np.linalg.lstsq(A, b, rcond=None)[0]
    Y = np.diag(A_grad @ v) + A_grad * v[None, :]
    YTU = Y.T @ U
    hess = 0.5 * (YTU - np.linalg.solve(A, Y).T @ Y + np.diag(v * (A_hess @ v)) + v[None, :] * v[:, None] * A_hess
                  - np.diag(b_hess * v) - b_grad[:, None] * U + YTU.T)
    return err, grad, hess, opt_weights


def optimize_error_optimal_weights(H, N, T, tol=1e-08, bound=None, method='gradient', force_order=False,
                                   post_processing=True):
    """
    Optimizes the L^2 strong approximation error with N points for fBm. Always uses the best weights and only
    numerically optimizes over the nodes.
    :param H: Hurst parameter
    :param N: Number of points
    :param T: Final time, may be a numpy array (only if grad is False and fast is True)
    :param tol: Error tolerance
    :param bound: Upper bound on the nodes. If no upper bound is desired, use None
    :param method: If error, uses only the error estimates for optimizing over the nodes, and uses the optimizer
        L-BFGS-B. If gradient, uses also the gradient of the error with respect to the nodes, and uses the optimizer
        L-BFGS-B. If hessian, uses also the gradient and the Hessian of the error with respect to the nodes, and uses
        the optimizer trust-constr
    :param force_order: Forces the nodes to stay in order, i.e. not switch places. May improve numerical stability
    :param post_processing: After optimizing the error, ensures that the results are reasonable and yield good results.
        If this is not the case, may call this function, or other processes again to potentially achieve better results
    :return: The minimal relative error together with the associated nodes and weights.
    """
    original_bound = bound

    # if T is a vector, choose a single T
    if isinstance(T, np.ndarray):
        if N <= 7:
            T = (8-N)/8 * np.amin(T) + N/8 * np.amax(T)
        else:
            T = np.amax(T)

    # get starting value and bounds for the optimization problem
    if bound is None:
        bound = 1e+100
        nodes_, w = quadrature_rule(H, N, T, mode='observation')
        if N == 2:
            bound = np.fmax(bound, np.amax(nodes_))
        if len(nodes_) < N:
            nodes = np.zeros(N)
            nodes[:len(nodes_)] = nodes_
            for i in range(len(nodes_), N):
                nodes[i] = nodes_[-1] * 10 ** (i - len(nodes_) + 1)
        else:
            nodes = nodes_[:N]
    else:
        nodes = np.exp(np.linspace(0, np.log(np.fmin(bound, 5. ** (np.fmin(140, N - 1)))), N))
    lower_bound = 1/(10*N*np.amin(T)) * ((0.5-H)/0.4)**2
    nodes = np.fmin(np.fmax(nodes, lower_bound*2), bound/2)
    bounds = ((np.log(lower_bound), np.log(bound)),) * N
    original_error, original_weights = error_optimal_weights(H=H, T=T, nodes=nodes, output='error')
    original_nodes = nodes.copy()

    # carry out the optimization
    if force_order:
        constraints = ()
        for i in range(1, N):
            def jac_here(x):
                res_ = np.zeros(N)
                res_[i] = 1
                res_[i-1] = 0
                return res_
            constraints = constraints + ({'type': 'ineq', 'fun': lambda x: x[i] - x[i-1], 'jac': jac_here},)

        if method == 'error' or method == 'err':
            def func(x):
                return error_optimal_weights(H, T, np.exp(x), output='error')[0]

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, constraints=constraints)

        else:
            def func(x):
                err_, grad, _ = error_optimal_weights(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, constraints=constraints, jac=True)

    else:
        if method == 'error' or method == 'err':
            def func(x):
                return error_optimal_weights(H, T, np.exp(x), output='error')[0]

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds)

        elif method == 'gradient' or method == 'grad':
            def func(x):
                err_, grad, _ = error_optimal_weights(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, jac=True)

        else:
            def func(x):
                err_, grad, _ = error_optimal_weights(H, T, np.exp(x), output='gradient')
                return err_, np.exp(x) * grad

            def hess(x):
                _, grad, hessian, _ = error_optimal_weights(H, T, np.exp(x), output='hessian')
                return hessian * np.exp(x[None, :] + x[:, None]) + np.diag(grad * np.exp(x))

            res = minimize(func, np.log(nodes), tol=tol ** 2, bounds=bounds, jac=True, hess=hess,
                           method='trust-constr')

    # post-processing, ensuring that the results are of good quality
    nodes = np.sort(np.exp(res.x))
    err, weights = error_optimal_weights(H=H, T=T, nodes=nodes, output='error')
    if post_processing:
        if err < 0 or np.fmax(original_error, 1e-9) < err or err > kernel_norm(H, T) ** 2 \
                or (N >= 3 and np.sqrt(nodes[-1]/nodes[-2]) > nodes[-2]/nodes[-3]) \
                or (N >= 2 and bound >= 1e+51 and (np.amin(weights) < 0 or np.amin(nodes[1:]/nodes[:-1]) < 1.01)):
            if force_order is False:
                return optimize_error_optimal_weights(H=H, N=N, T=T, tol=tol, bound=original_bound, method=method,
                                                      force_order=True, post_processing=True)
            elif bound >= 1e+24:
                return optimize_error_optimal_weights(H=H, N=N, T=T, tol=tol, bound=bound / 1e+5, method=method,
                                                      force_order=True, post_processing=True)

        factor = 3.
        if N >= 2 and np.amin(weights) < 0:
            if np.log(bound / lower_bound) < (N-1) * np.log(factor):
                nodes = np.exp(np.linspace(np.log(lower_bound), np.log(bound), N))
            elif np.log(bound / nodes[0]) < (N-1) * np.log(factor):
                nodes = bound / factor ** (N-1) * factor ** np.arange(N)
            elif np.log(nodes[-1]/lower_bound) < (N-1) * np.log(factor):
                nodes = lower_bound * factor ** np.arange(N)
            else:
                nodes = np.fmax(nodes, lower_bound * factor ** np.arange(N))
                nodes = np.fmin(nodes, bound / factor ** (N-1) * factor ** np.arange(N))
                for i in range(N-1):
                    nodes[i+1] = np.fmax(nodes[i] * factor, nodes[i+1])
            err, weights = error_optimal_weights(H=H, T=T, nodes=nodes, output='error')

        if err > 1e-9:
            paper_nodes, paper_weights = quadrature_rule(H=H, N=N, T=T, mode='paper')
            if np.amax(paper_nodes) <= bound:
                paper_nodes = np.fmax(paper_nodes, lower_bound)
                paper_error = error(H=H, nodes=paper_nodes, weights=paper_weights, T=T, output='error')
                paper_opt_error, paper_opt_weights = error_optimal_weights(H=H, T=T, nodes=paper_nodes, output='error')
                if paper_opt_error < paper_error and paper_opt_error < err:
                    nodes = paper_nodes
                    weights = paper_opt_weights
                    err = paper_opt_error
                elif paper_error < err:
                    nodes = paper_nodes
                    weights = paper_weights
                    err = paper_error

    if err > 2 * np.fmax(original_error, 1e-9):
        return np.sqrt(np.fmax(original_error, 0)) / kernel_norm(H, T), original_nodes, original_weights

    return np.sqrt(np.fmax(err, 0)) / kernel_norm(H, T), nodes, weights


def optimize_error(H, N, T, tol=1e-08, bound=None, iterative=False):
    """
    Optimizes the L^2 strong approximation error with N points for fBm. Uses the Nelder-Mead
    optimizer as implemented in scipy.
    :param H: Hurst parameter
    :param N: Number of points
    :param T: Final time, may be a numpy array (only if grad is False and fast is True)
    :param tol: Error tolerance
    :param bound: Upper bound on the nodes. If no upper bound is desired, use None
    :param iterative: If True, starts with one node and iteratively adds nodes, while always optimizing
    :return: The minimal relative error together with the associated nodes and weights.
    """
    if bound is None:
        bound = 1e+100

    def optimize_error_given_rule(nodes_1, weights_1):
        N_ = len(nodes_1)
        coefficient = 1 / kernel_norm(H, T) ** 2

        nodes_1 = np.fmin(np.fmax(nodes_1, 1e-02), bound / 2)
        bounds = (((np.log(1/(10*N_*np.amin(T))), np.log(bound)),) * N_) \
            + (((np.log(0.1), np.log(np.fmax(bound, 1e+60))),) * N_)
        rule = np.log(np.concatenate((nodes_1, weights_1)))

        if isinstance(T, np.ndarray):

            def func(x):
                return np.amax(coefficient * error(H, np.exp(x[:N_]), np.exp(x[N_:]), T, output='error'))

            res = minimize(func, rule, tol=tol ** 2, bounds=bounds)
        else:

            def func(x):
                err_, grad = error(H, np.exp(x[:N_]), np.exp(x[N_:]), T, output='gradient')
                err_ = np.amax(coefficient * err_)
                grad = coefficient * np.exp(x) * grad
                return err_, grad

            res = minimize(func, rule, tol=tol ** 2, bounds=bounds, jac=True)

        nodes_1, weights_1 = sort(np.exp(res.x[:N_]), np.exp(res.x[N_:]))
        return np.sqrt(res.fun), nodes_1, weights_1

    if not iterative:
        if isinstance(T, np.ndarray):
            nodes_, weights_ = quadrature_rule(H, N, (np.amin(T) + np.amax(T))/2, mode='optimized')
        else:
            nodes_, weights_ = quadrature_rule(H, N, T, mode='observation')
        if len(nodes_) < N:
            nodes = np.zeros(N)
            weights = np.zeros(N)
            nodes[:len(nodes_)] = nodes_
            weights[:len(weights_)] = weights_
            for i in range(len(nodes_), N):
                nodes[i] = nodes_[-1] * 10 ** (i - len(nodes_) + 1)
                weights[i] = weights_[-1]
        else:
            nodes = nodes_[:N]
            weights = weights_[:N]

        return optimize_error_given_rule(nodes, weights)

    if isinstance(T, np.ndarray):
        nodes, weights = quadrature_rule(H, 1, (np.amin(T) + np.amax(T)) / 2, mode='optimized')
    else:
        nodes, weights = quadrature_rule(H, 1, T, mode='observation')
    err, nodes, weights = optimize_error_given_rule(nodes, weights)

    while len(nodes) < N:
        if bound is None:
            nodes = np.append(nodes, 10 * nodes[-1])
            weights = np.append(weights, np.amax(weights))
        else:
            if len(nodes) == 1:
                if bound > 10 * nodes[0]:
                    nodes = np.array([nodes[0], 10 * nodes[0]])
                    weights = np.array([weights[0], weights[0]])
                elif bound >= 2 * nodes[0]:
                    nodes = np.array([nodes[0], bound])
                    weights = np.array([weights[0], weights[0] / 2])
                else:
                    nodes = np.array([nodes[0] / 3, nodes[0]])
                    weights = np.array([weights[0] / 3, weights[0] / 2])
            else:
                if bound > 10 * nodes[-1]:
                    nodes = np.append(nodes, 10 * nodes[-1])
                    weights = np.append(weights, np.amax(weights))
                elif bound > 2 * nodes[-1] or bound / nodes[-1] > nodes[-1] / nodes[-2]:
                    nodes = np.append(nodes, bound)
                    weights = np.append(weights, np.amax(weights) / 2)
                else:
                    nodes = np.append(nodes, np.sqrt(nodes[-1] * nodes[-2]))
                    weights[-1] = weights[-1] * 0.7
                    weights[-2] = weights[-2] * 0.7
                    weights = np.append(weights, np.fmin(weights[-1], weights[-2]))
                    nodes, weights = sort(nodes, weights)
        err, nodes, weights = optimize_error_given_rule(nodes, weights)

    return err, nodes, weights


def optimized_rule(H, N, T, optimal_weights=False):
    """
    Returns the optimal nodes and weights of the N-point quadrature rule for the fractional kernel with Hurst parameter
    H.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time
    :param optimal_weights: If True, uses the optimal weights for the kernel error
    :return: All the nodes and weights in increasing order, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    if optimal_weights:
        _, nodes, weights = optimize_error_optimal_weights(H=H, N=N, T=T, bound=None, method='gradient')
    else:
        _, nodes, weights = optimize_error(H=H, N=N, T=T, bound=None, iterative=False)
    return nodes, weights


def european_rule(H, N, T, optimal_weights=False):
    """
    Returns a quadrature rule that is optimized for pricing European options under the rough Heston model.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time/Maturity
    :param optimal_weights: If True, uses the optimal weights for the kernel error
    :return: Nodes and weights
    """

    def optimizing_func(N_, tol_, bound_):
        if optimal_weights:
            return optimize_error_optimal_weights(H=H, N=N_, T=T, tol=tol_, bound=bound_, method='gradient',
                                                  post_processing=False)
        return optimize_error(H=H, N=N_, T=T, tol=tol_, bound=bound_, iterative=True)

    _, nodes, weights = optimizing_func(N_=1, tol_=1e-06, bound_=None)
    if N == 1:
        return nodes, weights

    L_step = 1.05
    bound = np.amax(nodes) / L_step
    kernel_tol = 0.999
    current_N = 1

    while current_N < N:
        increase_N = 0

        while increase_N < 3:
            bound = bound * L_step
            error_1, _, _ = optimizing_func(N_=current_N, tol_=1e-07/current_N, bound_=bound)
            error_2, nodes, weights = optimizing_func(N_=current_N+1, tol_=1e-07/current_N, bound_=bound)
            nodes = np.sort(nodes)
            if np.amin(nodes[1:] / nodes[:-1]) < 2:
                increase_N = 0
            elif error_2 / error_1 < 1 - (1 - kernel_tol)/current_N:
                increase_N += 1
            else:
                increase_N = 0

        current_N = current_N + 1

    if N >= 4:
        return nodes, weights
    if N == 2:
        L_4 = bound * 2
        L_5 = bound * 3
        L_6 = bound * 4
    else:  # N == 3
        L_4 = bound
        L_5 = bound * 1.25
        L_6 = bound * 1.5
    error_4, nodes_4, weights_4 = optimizing_func(N_=N, tol_=1e-08, bound_=L_4)
    error_5, nodes_5, weights_5 = optimizing_func(N_=N, tol_=1e-08, bound_=L_5)
    error_6, nodes_6, weights_6 = optimizing_func(N_=N, tol_=1e-08, bound_=L_6)
    if error_4 <= error_5 and error_4 <= error_6:
        return nodes_4, weights_4
    if error_5 <= error_6:
        return nodes_5, weights_5
    return nodes_6, weights_6


def AbiJaberElEuch_quadrature_rule(H, N, T):
    pi_n = N ** (-0.2) / T * (np.sqrt(10) * (1 - 2 * H) / (5 - 2 * H)) ** 0.4
    eta = pi_n * np.arange(N + 1)
    c_vec = (eta[1:] ** (0.5 - H) - eta[:-1] ** (0.5 - H)) / (gamma(H + 0.5) * gamma(1.5 - H))
    gamma_vec = (eta[1:] ** (1.5 - H) - eta[:-1] ** (1.5 - H)) / ((1.5 - H) * gamma(H + 0.5) + gamma(0.5 - H)) / c_vec
    return gamma_vec, c_vec


def quadrature_rule(H, N, T, mode="optimized"):
    """
    Returns the nodes and weights of a quadrature rule for the fractional kernel with Hurst parameter H. The nodes are
    sorted in increasing order.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If european, chooses a
        rule that is especially suitable for pricing European options. Appending old leads to using suboptimal weights
    :return: All the nodes and weights, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    if isinstance(T, np.ndarray):
        if N == 1:
            T = np.amin(T) ** (3 / 5) * np.amax(T) ** (2 / 5)
        if N == 2:
            T = np.amin(T) ** (1 / 2) * np.amax(T) ** (1 / 2)
        if N == 3:
            T = np.amin(T) ** (1 / 3) * np.amax(T) ** (2 / 3)
        if N == 4:
            T = np.amin(T) ** (1 / 4) * np.amax(T) ** (3 / 4)
        if N == 5:
            T = np.amin(T) ** (1 / 6) * np.amax(T) ** (5 / 6)
        if N == 6:
            T = np.amin(T) ** (1 / 10) * np.amax(T) ** (9 / 10)
        else:
            T = np.amax(T)

    if mode == "optimized":
        return optimized_rule(H=H, N=N, T=T, optimal_weights=True)
    if mode == "european":
        return european_rule(H=H, N=N, T=T, optimal_weights=True)
    if mode == "optimized old":
        return optimized_rule(H=H, N=N, T=T, optimal_weights=False)
    if mode == "european old":
        return european_rule(H=H, N=N, T=T, optimal_weights=False)
    if mode == "observation" or mode == "theorem":
        return Gaussian_rule(H=H, N=N, T=T, mode=mode, optimal_weights=True)
    if mode == "observation old" or mode == "paper":
        return Gaussian_rule(H=H, N=N, T=T, mode="observation", optimal_weights=False)
    if mode == "abi jaber":
        return AbiJaberElEuch_quadrature_rule(H=H, N=N, T=T)
    if mode == "european alt":
        return quadrature_rule(H=H, N=N, T=2, mode="european")
    return Gaussian_rule(H=H, N=N, T=T, mode="theorem", optimal_weights=False)
