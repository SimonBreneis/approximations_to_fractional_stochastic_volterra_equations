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
        return np.sqrt(np.fmax(float(a * w_0 * w_0 + b * w_0 + c), 0.)), float(w_0)

    nodes, weights = Gaussian_no_zero_node(H, m, n, mp.mpf(a), mp.mpf(b))
    c = error(H, nodes, weights, T)
    b = mp.mpf(0)
    for i in range(n * m):
        b += weights[i] / nodes[i] * (1 - mp.exp(-nodes[i] * T))
    b -= T ** (H + 0.5) / mp.gamma(H + 1.5)
    b *= mp.mpf(2.)
    a = T
    w_0 = -b / (mp.mpf(2.) * a)
    return np.sqrt(np.fmax(float(a * w_0 * w_0 + b * w_0 + c), 0.)), float(w_0)


def Gaussian_rule(H, N, T, mode='observation'):
    """
    Returns the nodes and weights of the Gaussian rule with roughly N nodes.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem
    :return: The nodes and weights, ordered by the size of the nodes
    """
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
    return mp_to_np(nodes), mp_to_np(weights)


def error(H, nodes, weights, T):
    """
    Computes an error estimate of the L^2-norm of the difference between the rough kernel and its approximation
    on [0, T].
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param weights: The weights of the approximation
    :param T: Final time, may also be a vector
    :return: An error estimate
    """
    nodes = mp_to_np(nodes)
    weights = mp_to_np(weights)
    if np.amin(nodes) < 0 or np.amin(weights) < 0:
        return 1e+10
    nodes = np.fmin(np.fmax(nodes, 1e-08), 1e+150)
    weights = np.fmin(weights, 1e+75)
    summand = T ** (2 * H) / (2 * H * gamma(H + 0.5) ** 2)
    node_matrix = nodes[:, None] + nodes[None, :]
    if isinstance(T, np.ndarray):
        factor = 1 - np.exp(-np.fmin(np.einsum('ij,k->kij', node_matrix, T), 300))
        sum_1 = np.sum(np.repeat((np.outer(weights, weights) / node_matrix)[None, :, :], len(T), axis=0) * factor,
                       axis=(1, 2))
        sum_2 = 2 * np.sum((weights / nodes ** (H + 0.5))[:, None] * gammainc(H + 0.5, np.outer(nodes, T)), axis=0)
    else:
        sum_1 = np.sum(np.outer(weights, weights) / node_matrix * (1 - np.exp(-np.fmin(node_matrix * T, 300))))
        sum_2 = 2 * np.sum(weights / nodes ** (H + 0.5) * gammainc(H + 0.5, nodes * T))
    return summand + sum_1 - sum_2


def gradient_of_error(H, T, nodes, weights):
    """
    Computes the gradient nabla int_0^T |K(t) - hat{K}(t)|^2 dt, where nabla acts on the nodes and weights,
    assuming that there is no node at 0.
    :param H: Hurst parameter
    :param T: Final time, may be a vector
    :param nodes: Nodes of the quadrature rule
    :param weights: Weights of the quadrature rule
    :return: The gradient
    """
    N = len(nodes)
    node_matrix = nodes[:, None] + nodes[None, :]
    weight_matrix = np.outer(weights, weights)
    if isinstance(T, np.ndarray):
        grad = np.empty((len(T), 2 * N))
        gamma_ints = gammainc(H + 1 / 2, np.outer(T, nodes))
        temp = np.einsum('i,jk->ijk', T, node_matrix)
        exp_node_matrix = np.exp(-np.fmin(temp, 300))
        exp_node_matrix = np.where(temp > 300, 0, exp_node_matrix)
        exp_node_vec = np.exp(-np.fmin(np.outer(T, nodes), 300)) / nodes[None, :]
        exp_node_vec = np.where(exp_node_vec < np.exp(-299), 0, exp_node_vec)
        first_summands = (weight_matrix / (node_matrix * node_matrix))[None, :] * (1 - (1 + temp) * exp_node_matrix)
        second_summands = weights[None, :] * ((T ** (H + 1 / 2) / gamma(H + 1 / 2))[:, None] * exp_node_vec - (
                    ((H + 1 / 2) * nodes ** (-H - 3 / 2))[None, :] * gamma_ints))
        grad[:, :N] = -2 * np.sum(first_summands, axis=2) - 2 * second_summands
        third_summands = np.einsum('ijk,k->ij', ((1 - exp_node_matrix) / node_matrix[None, :, :]), weights)
        forth_summands = (nodes ** (-(H + 1 / 2)))[None, :] * gamma_ints
        grad[:, N:] = 2 * third_summands - 2 * forth_summands
    else:
        grad = np.empty(2 * N)
        gamma_ints = gammainc(H + 1 / 2, nodes * T)
        exp_node_matrix = np.exp(-np.fmin(node_matrix * T, 300))
        exp_node_matrix = np.where(exp_node_matrix < np.exp(-299), 0, exp_node_matrix)
        exp_node_vec = np.zeros(N)
        indices = nodes * T < 300
        exp_node_vec[indices] = np.exp(- T * nodes[indices]) / nodes[indices]
        first_summands = weight_matrix / (node_matrix * node_matrix) * (1 - (1 + node_matrix * T) * exp_node_matrix)
        second_summands = weights * (T ** (H + 1 / 2) / gamma(H + 1 / 2) * exp_node_vec - (H + 1 / 2) * nodes ** (
                    -H - 3 / 2) * gamma_ints)
        grad[:N] = -2 * np.sum(first_summands, axis=1) - 2 * second_summands
        third_summands = ((1 - exp_node_matrix) / node_matrix) @ weights
        forth_summands = nodes ** (-(H + 1 / 2)) * gamma_ints
        grad[N:] = 2 * third_summands - 2 * forth_summands
    return grad


def optimize_error(H, N, T, tol=1e-05, bound=None, iterative=False, l2=None):
    """
    Optimizes the L^2 strong approximation error with N points for fBm. Uses the Nelder-Mead
    optimizer as implemented in scipy.
    :param H: Hurst parameter
    :param N: Number of points
    :param T: Final time, may be a numpy array (only if grad is False and fast is True)
    :param tol: Error tolerance
    :param bound: Upper bound on the nodes. If no upper bound is desired, use None
    :param iterative: If True, starts with one node and iteratively adds nodes, while always optimizing
    :param l2: Boolean. Relevant if T is an array. If True, uses the l2-norm over T, else uses the l-infinity norm.
        Using the l-infinity norm may lead to better results, while using the l2-norm is faster because we can directly
        compute the gradient
    :return: The minimal error together with the associated nodes and weights.
    """
    if l2 is None:
        l2 = isinstance(T, np.ndarray)
    if not isinstance(T, np.ndarray):
        l2 = False
    if bound is None:
        bound = 1e+100

    def optimize_error_given_rule(nodes_1, weights_1):
        N_ = len(nodes_1)
        coeff = 1 / kernel_norm(H, T) ** 2

        nodes_1 = np.fmin(np.fmax(nodes_1, 1e-02), bound / 2)
        bounds = (((np.log(1e-08), np.log(bound)),) * N_) + (((np.log(0.1), np.log(np.fmax(bound, 1e+60))),) * N_)
        rule = np.log(np.concatenate((nodes_1, weights_1)))

        if l2:
            T_ = np.array([(np.amin(T) + np.amax(T)) / 2, np.amax(T)])
            coeff = 1 / kernel_norm(H, T_) ** 2

            def func(x):
                return np.sum(coeff * error(H=H, nodes=np.exp(x[:N_]), weights=np.exp(x[N_:]), T=T_))/len(T_)

            def jac(x):
                r = np.exp(x)
                return r * np.sum(coeff[:, None] * gradient_of_error(H=H, T=T_, nodes=r[:N_], weights=r[N_:]), axis=0)/len(T_)

        else:
            def func(x):
                return np.amax(coeff * error(H, np.exp(x[:N_]), np.exp(x[N_:]), T))

            def jac(x):
                return coeff * np.exp(x) * gradient_of_error(H=H, T=T, nodes=np.exp(x[:N_]), weights=np.exp(x[N_:]))

        if isinstance(T, np.ndarray):
            if l2:
                res = minimize(func, rule, tol=tol ** 2, bounds=bounds, jac=jac)
            else:
                res = minimize(func, rule, tol=tol ** 2, bounds=bounds)
        else:
            res = minimize(func, rule, tol=tol ** 2, bounds=bounds, jac=jac)

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
            nodes[:len(nodes)] = nodes_
            weights[:len(weights)] = weights_
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


def optimized_rule(H, N, T):
    """
    Returns the optimal nodes and weights of the N-point quadrature rule for the fractional kernel with Hurst parameter
    H.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time
    :return: All the nodes and weights in increasing order, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    _, nodes, weights = optimize_error(H=H, N=N, T=T, bound=None, iterative=False)
    return nodes, weights


def european_rule(H, N, T):
    """
    Returns a quadrature rule that is optimized for pricing European options under the rough Heston model.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time/Maturity
    :return: Nodes and weights
    """

    nodes, weights = optimized_rule(H=H, N=1, T=T)
    if N == 1:
        return nodes, weights

    L_step = 1.1
    bound = np.amax(nodes) / L_step
    L_0 = bound
    kernel_tol = 0.999
    current_N = 1

    while current_N <= N:
        L_0 = bound
        increase_N = 0

        while increase_N < 3:
            bound = bound * L_step
            error_1, _, _ = optimize_error(H=H, N=current_N, T=T, bound=bound, iterative=True)
            error_2, _, _ = optimize_error(H=H, N=current_N + 1, T=T, bound=bound, iterative=True)
            if error_2 / error_1 < kernel_tol:
                increase_N += 1
            else:
                increase_N = 0

        current_N = current_N + 1
    L_1 = bound

    if N == 2:
        L_4 = L_0 * (L_1 / L_0) ** 0.2
        L_5 = L_0 * (L_1 / L_0) ** 0.3
        L_6 = L_0 * (L_1 / L_0) ** 0.4
    else:
        L_4 = L_0 * (L_1 / L_0) ** 0.05
        L_5 = L_0 * (L_1 / L_0) ** 0.1
        L_6 = L_0 * (L_1 / L_0) ** 0.15
    error_4, nodes_4, weights_4 = optimize_error(H=H, N=N, T=T, bound=L_4, iterative=True)
    error_5, nodes_5, weights_5 = optimize_error(H=H, N=N, T=T, bound=L_5, iterative=True)
    error_6, nodes_6, weights_6 = optimize_error(H=H, N=N, T=T, bound=L_6, iterative=True)

    if error_4 <= error_5 and error_4 <= error_6:
        return nodes_4, weights_4
    if error_5 <= error_6:
        return nodes_5, weights_5
    return nodes_6, weights_6


def quadrature_rule(H, N, T, mode="optimized"):
    """
    Returns the nodes and weights of a quadrature rule for the fractional kernel with Hurst parameter H.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If european, chooses a
        rule that is especially suitable for pricing European options
    :return: All the nodes and weights, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    if mode == "optimized":
        return optimized_rule(H=H, N=N, T=T)
    if mode == "european":
        return european_rule(H=H, N=N, T=T)
    return Gaussian_rule(H=H, N=N, T=T, mode=mode)
