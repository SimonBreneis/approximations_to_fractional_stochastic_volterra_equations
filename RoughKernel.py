import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from scipy.optimize import minimize
import scipy.special
import orthopy
import quadpy


def fractional_kernel(H, t):
    return t ** (H - 0.5) / scipy.special.gamma(H + 0.5)


def fractional_kernel_laplace(H, t, rho):
    return c_H(H) * np.exp(-rho * t)


def c_H(H):
    """
    Returns the constant c_H.
    :param H: Hurst parameter
    :return: c_H
    """
    return 1. / (scipy.special.gamma(0.5 + H) * scipy.special.gamma(0.5 - H))


def plot_kernel_approximations(H, m, n_vec, a, b, left=0.0001, right=1., number_time_steps=10000):
    """
    Plots the true rough kernel and the approximations that are inspired by Alfonsi and Kebaier.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n_vec: The number of nodes for which the approximations should be plotted
    :param a: Can shift the left endpoint of the total interval of the quadrature rule
    :param b: Can shift the right endpoint of the total interval of the quadrature rule
    :param left: Left endpoint of the interval on which the kernels should be plotted
    :param right: Right endpoint of the interval on which the kernels should be plotted
    :param number_time_steps: Number of (equidistant) time steps used (-1) on which we plot the kernels
    :return: Nothing, plots the true kernel and the approximations
    """
    dt = (right-left)/number_time_steps
    time_steps = left + dt*np.arange(number_time_steps+1)
    approximations = np.empty(shape=(number_time_steps+1, len(n_vec)+1))
    approximations[:, 0] = fractional_kernel(H, time_steps)
    plt.plot(time_steps, approximations[:, 0], label=f"N=infinity")

    for i in range(len(n_vec)):
        quad_rule = quadrature_rule_geometric(H, int(m[i]), int(n_vec[i]), a[i], b[i])
        quad_nodes = quad_rule[0, :]
        quad_weights = quad_rule[1, :]
        approximations[:, i+1] = 1 / c_H(H) * np.array(
            [fractional_kernel_laplace(H, t, quad_nodes) for t in time_steps]).dot(quad_weights)
        plt.plot(time_steps, approximations[:, i+1], label=f"N={int(n_vec[i]*m[i])}")
    plt.legend(loc="upper right")
    plt.show()


def gradient_of_error(H, T, nodes, weights):
    """
    Computes the gradient nabla int_0^T |K(t) - hat{K}(t)|^2 dt, where nabla acts on the nodes and weights,
    assuming that there is no node at 0.
    :param H: Hurst parameter
    :param T: Final time
    :param nodes: Nodes of the quadrature rule
    :param weights: Weights of the quadrature rule
    :return: The gradient
    """
    N = len(nodes)
    gamma_ints = scipy.special.gammainc(H+1/2, nodes*T)
    grad = np.empty(2*N)
    node_matrix = nodes[:, None] + nodes[None, :]
    exp_node_matrix = np.exp(-node_matrix*T)
    weight_matrix = np.outer(weights, weights)
    first_summands = weight_matrix / (node_matrix * node_matrix) * (1-(1-node_matrix*T)*exp_node_matrix)
    second_summands = weights * (T**(H+1/2) / nodes * np.exp(-nodes*T) / scipy.special.gamma(H+1/2) - (H+1/2) * nodes**(-(H+3/2)) * gamma_ints)
    grad[:N] = -2 * np.sum(first_summands, axis=1) - 2 * second_summands
    third_summands = ((1-exp_node_matrix)/node_matrix) @ weights
    forth_summands = nodes ** (-(H+1/2)) * gamma_ints
    grad[N:] = 2 * third_summands - 2 * forth_summands
    return grad


def error_estimate_fBm_general(H, nodes, weights, T):
    """
    Computes an error estimate of the L^2-norm of the difference between the rough kernel and its approximation
    on [0, T]. The approximation does NOT contain a constant/Brownian term.
    :param H: Hurst parameter
    :param nodes: The nodes of the approximation. Assumed that they are all non-zero
    :param weights: The weights of the approximation
    :param T: Final time
    :return: An error estimate and the optimal weight for the constant term: [error, weight]
    """
    if np.amin(mp_to_np(nodes)) <= 0 or np.amin(mp_to_np(weights)) < 0:
        return 1e+10
    N = len(nodes)
    T = mp.mpf(T)
    H = mp.mpf(H)
    summand = T ** (2 * H) / (2 * H * mp.gamma(H + 0.5) ** 2)
    sum_1 = mp.mpf(0)
    sum_2 = mp.mpf(0)
    summands_1 = np.empty(shape=(N**2,))
    summands_2 = np.empty(shape=(N,))
    for i in range(N):
        for j in range(N):
            summands_1[i * N + j] = weights[i] * weights[j] / (nodes[i] + nodes[j]) * (
                        1 - mp.exp(-(nodes[i] + nodes[j]) * T))
        summands_2[i] = weights[i] / nodes[i] ** (H + 0.5) * mp.gammainc(H + 0.5, mp.mpf(0.), nodes[i] * T)
    summands_1.sort()
    summands_2.sort()
    for summand_1 in summands_1:
        sum_1 += summand_1
    for summand_2 in summands_2:
        sum_2 += summand_2
    sum_2 *= 2 / mp.gamma(H + 0.5)
    return summand + sum_1 - sum_2


def error_estimate_fBm(H, m, n, a, b, T):
    """
    Computes an error estimate of the L^2-norm of the difference between the rough kernel and its approximation
    on [0, T]. The approximation contains a constant/Brownian term.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of quadrature intervals
    :param a: xi_0 = e^(-a)
    :param b: xi_n = e^b
    :param T: Final time
    :return: An error estimate and the optimal weight for the constant term: [error, weight]
    """
    T = mp.mpf(T)
    H = mp.mpf(H)
    a = mp.mpf(a)
    b = mp.mpf(b)
    rule = quadrature_rule_geometric_no_zero_node(H, m, n, a, b)
    nodes = rule[0, :]
    weights = rule[1, :]
    c = error_estimate_fBm_general(H, nodes, weights, T)
    b = mp.mpf(0)
    for i in range(n*m):
        b += weights[i]/nodes[i] * (1 - mp.exp(-nodes[i]*T))
    b -= T**(H+0.5)/((H+0.5)*mp.gamma(H+0.5))
    b *= mp.mpf(2.)
    a = T
    w_0 = -b/(mp.mpf(2.)*a)
    return np.array([np.sqrt(np.fmax(float(a*w_0*w_0 + b*w_0 + c), 0.)), float(w_0)])


def optimize_error_fBm_general(H, N, T, tol=1e-04, grad=True, bound=None):
    """
    Optimizes the L^2 strong approximation error with N points for fBm. Uses the Nelder-Mead
    optimizer as implemented in scipy.
    :param H: Hurst parameter
    :param N: Number of points
    :param T: Final time
    :param tol: Error tolerance
    :param grad: If True, uses the gradient in the optimization
    :param bound: Upper bound on the nodes. If no upper bound is desired, use None
    :return: The minimal error together with the associated nodes and weights.
    """

    def func(x):
        return error_estimate_fBm_general(H, x[:N], x[N:], T)

    nodes, weights = quadrature_rule_geometric_standard(H, N+1, T, mode='observation')
    rule = np.zeros(2*N)
    if len(nodes) < N:
        rule[:len(nodes)] = nodes
        rule[N:N+len(nodes)] = weights
        for i in range(len(nodes), N):
            rule[i] = nodes[-1] * 10**(i-len(nodes)+1)
            rule[N+i] = weights[-1]
    else:
        rule[:N] = nodes[:N]
        rule[N:] = weights[:N]
    rule[0] = rule[1]/10

    kernel_l2_norm = T**(2*H) / (2*H*scipy.special.gamma(H+1/2)**2)
    bounds = ((0, bound)*N) + ((0, None)*N)
    if grad:
        res = minimize(func, rule, method="BFGS", jac=lambda x: gradient_of_error(H=H, T=T, nodes=x[:N], weights=x[N:]),
                       options={"fatol": tol**2 * kernel_l2_norm, "maxiter": 10000}, bounds=bounds)
    else:
        res = minimize(func, rule, method="nelder-mead", options={"fatol": tol**2 * kernel_l2_norm, "maxiter": 10000},
                       bounds=bounds)
    return res.fun, res.x[:N], res.x[N:]


def optimize_error_fBm(H, T, m, n, a=0., b=0., tol=1e-8):
    """
    Optimizes the L^2 strong approximation error of the AK scheme over xi_0 and xi_n for fBm. Uses the Nelder-Mead
    optimizer as implemented in scipy.
    :param H: Hurst parameter
    :param T: Final time
    :param m: Level of quadrature rule
    :param n: Number of quadrature intervals
    :param a: xi_0 = e^(-a)
    :param b: xi_n = e^b
    :param tol: Tolerance in a, b of the optimal values
    :return: The minimal error together with the weight w_0 and a and b in the form [error, w_0, a, b] as a numpy array.
    """
    def func(x):
        if x[0] + x[1] <= 0.:
            return 10.**20
        return error_estimate_fBm(H, m, n, x[0], x[1], T)[0]

    res = minimize(func, np.array([a, b]), method="nelder-mead", options={"xatol": tol, "maxiter": 10000})
    a = res.x[0]
    b = res.x[1]
    return np.array([res.fun, error_estimate_fBm(H, m, n, a, b, T)[1], a, b])


def quadrature_rule_interval(H, m, a, b):
    """
    Returns the nodes and weights of the Gauss quadrature rule level m for the fractional weight function on [a,b]
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param a: Left end of interval
    :param b: Right end of interval
    :return: The nodes and weights, in form [[node1, node2, ...], [weight1, weight2, ...]]
    """
    c = mp.mpf(c_H(float(H)))
    moments = np.array(
        [mp.mpf(c / (mp.mpf(k) + mp.mpf(0.5) - H) * (b ** (mp.mpf(k) + mp.mpf(0.5) - H) - a ** (mp.mpf(k) + mp.mpf(0.5) - H))) for k in range(2 * m)])
    alpha, beta, int_1 = orthopy.tools.chebyshev(moments)
    points, weights = quadpy.tools.scheme_from_rc(alpha, beta, int_1, mode="mpmath")
    result = mp.matrix(2, m)
    result[0, :] = mp.matrix(points.tolist()).transpose()
    result[1, :] = mp.matrix(weights.tolist()).transpose()
    return result


def quadrature_rule_mpmath(H, m, partition):
    """
    Returns the quadrature rule of level m of the fractional kernel with Hurst parameter H on all the partition
    intervals. The result is an mpmath matrix.
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


def quadrature_rule_geometric_no_zero_node(H, m, n, a, b):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals. The result is an instance of mp.matrix with mp.mpf entries. Does not
    contain a node at 0.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of subintervals
    :param a: a = - log(xi_0)
    :param b: b = log(xi_n)
    :return: All the nodes and weights, in the form [[node1, node2, ...], [weight1, weight2, ...]]
    """
    mp.mp.dps = int(np.fmax(a+b + 50, 50))
    partition = np.array([mp.exp(-a + (a + b) * (mp.mpf(i) / mp.mpf(n))) for i in range(0, n + 1)])
    return quadrature_rule_mpmath(H, m, partition)


def quadrature_rule_geometric_mpmath(H, m, n, a, b, T=1.):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals. The result is an instance of mp.matrix with mp.mpf entries.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of subintervals
    :param a: a = - log(xi_0)
    :param b: b = log(xi_n)
    :param T: Final time
    :return: All the nodes and weights, in the form [[node1, node2, ...], [weight1, weight2, ...]]
    """
    w_0 = error_estimate_fBm(H, m, n, a, b, T)[1]
    mp.mp.dps = int(np.fmax(a+b + 50, 50))
    partition = np.array([mp.exp(-a + (a + b) * (mp.mpf(i) / mp.mpf(n))) for i in range(n + 1)])
    rule = mp.matrix(2, m*n+1)
    rule[0, m*n] = mp.mpf(0)
    rule[1, m*n] = mp.mpf(w_0)
    rule[:, :-1] = quadrature_rule_mpmath(H, m, partition)
    return rule


def quadrature_rule_geometric_good(H, N, T=1., mode="best"):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals. The result is an instance of mp.matrix with mp.mpf entries. Here, n, m, xi_0
    and xi_n are chosen according to the mode.
    :param H: Hurst parameter
    :param N: Total number of nodes, N=nm
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :return: All the nodes and weights, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    if mode == 'best':
        if N <= 5:
            mode = 'optimized'
        else:
            mode = 'observation'
    if mode == "optimized":
        nodes, weights = quadrature_rule_optimized(H=H, N=N, T=T)
        nodes = mp.matrix([mp.mpf(node) for node in nodes])
        weights = mp.matrix([mp.mpf(weight) for weight in weights])
        return nodes, weights
    N = N-1
    [m, n, a, b] = get_parameters(H, N, T, mode)
    rule = quadrature_rule_geometric_mpmath(H, m, n, a, b, T)
    nodes = rule[0, :]
    weights = rule[1, :]
    return nodes, weights


def quadrature_rule_geometric_standard(H, N, T=1., mode="best"):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals. The result is an instance of a numpy array, with nodes ordered in increasing
    order. Here, n, m, xi_0 and xi_n are chosen according to the mode.
    :param H: Hurst parameter
    :param N: Total number of nodes, N=nm
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :return: All the nodes and weights, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    if mode == 'best':
        if N <= 5:
            mode = 'optimized'
        else:
            mode = 'observation'
    if mode == "optimized":
        return quadrature_rule_optimized(H=H, N=N, T=T)
    nodes, weights = quadrature_rule_geometric_good(H=H, N=N, T=T, mode=mode)
    nodes = mp_to_np(nodes)
    weights = mp_to_np(weights)
    N = len(nodes)
    nodes_ = np.empty(N)
    nodes_[0] = nodes[-1]
    nodes_[1:] = nodes[:-1]
    weights_ = np.empty(N)
    weights_[0] = weights[-1]
    weights_[1:] = weights[:-1]
    return nodes_, weights_


def quadrature_rule_geometric(H, m, n, a, b, T=1.):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals. The result is a numpy array, but the computations are done using the
    mpmath library.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of subintervals
    :param a: a = - log(xi_0)
    :param b: b = log(xi_n)
    :param T: Final time
    :return: All the nodes and weights, in the form [[node1, node2, ...], [weight1, weight2, ...]]
    """
    rule = quadrature_rule_geometric_mpmath(H, m, n, a, b, T)
    return mp_to_np(rule)


def quadrature_rule_optimized(H, N, T=1.):
    """
    Returns the optimal nodes and weights of the N-point quadrature rule for the fractional kernel with Hurst parameter
    H.
    :param H: Hurst parameter
    :param N: Number of nodes
    :param T: Final time
    :return: All the nodes and weights in increasing order, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    _, nodes, weights = optimize_error_fBm_general(H=H, N=N, T=T)
    permutation = np.argsort(nodes)
    nodes = nodes[permutation]
    weights = weights[permutation]
    return nodes, weights


def get_parameters(H, N, T, mode):
    """
    Returns the parameters m, n, a, b of the geometric quadrature rule.
    :param H: Hurst parameter
    :param N: Total number of nodes
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem
    :return: Quadrature level m, number of intervals n, left point a, right point b
    """
    a = 0.
    b = 0.
    beta = 0.
    A = mp.sqrt(1/H + 1/(1.5-H))

    if mode == "theorem":
        beta = 0.4275
        alpha = 1.06418
        gamma = np.exp(alpha*beta)
        exponent = 1/(3*gamma/(8*(gamma-1)) + 6*H - 4*H*H)
        base_0 = ((9-6*H)/(2*H))**(gamma/(8*(gamma-1))) * (5*np.pi**3/768 * gamma * (gamma-1) * A**(2-2*H) * (3-2*H) * float(N)**(1-H) / (beta**(2-2*H) * H))**(2*H)
        a = -mp.log(T**(-1) * base_0**exponent * mp.exp(-alpha/((1.5-H)*A) * np.sqrt(N)))
        base_n = ((9-6*H)/(2*H))**(gamma/(8*(gamma-1))) * (5*np.pi**3/1152 * gamma * (gamma-1) * A**(2-2*H) * float(N)**(1-H) / (beta**(2-2*H)))**(2*H-3)
        b = mp.log(T**(-1) * base_n**exponent * mp.exp(alpha/(H*A) * np.sqrt(N)))
    elif mode == "observation":
        beta = 0.9
        alpha = 1.8
        a = -mp.log(0.65 * 1/T * mp.exp(3.1*H) * mp.exp(-alpha / ((1.5-H)*A) * np.sqrt(N)))
        b = mp.log(1/T * mp.exp(3 * H**(-0.4)) * mp.exp(alpha/(H*A) * np.sqrt(N)))

    m = int(np.fmax(np.round(float(beta / A * np.sqrt(N))), 1))
    n = int(np.round(N / m))
    return m, n, a, b


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


def compare_approximations(H, N, T=1):
    """
    Compares the optimized quadrature rules with the quadrature rules from the observation.
    :param H: Hurst parameter
    :param N: Number of nodes (numpy array)
    :param T: Final time
    :return: The nodes, weights, and errors using the observation points, and the nodes, weights and errors using
        the optimized points (total of 6 numpy arrays)
    """
    nodes_observation = [np.empty(1)]*len(N)
    nodes_optimization = [np.empty(1)]*len(N)
    weights_observation = [np.empty(1)]*len(N)
    weights_optimization = [np.empty(1)]*len(N)
    errors_observation = np.zeros(len(N))
    errors_optimization = np.zeros(len(N))

    for i in range(len(N)):
        nodes_observation[i], weights_observation[i] = quadrature_rule_geometric_standard(H, N[i]-1, T)
        corrector = np.zeros(len(nodes_observation[i]))
        corrector[0] += 0.00001
        errors_observation[i] = error_estimate_fBm_general(H, nodes_observation[i] + corrector, weights_observation[i], T)
        print(f'{N[i]}, observation:')
        print(nodes_observation[i])
        print(weights_observation[i])
        print(np.sqrt(errors_observation[i]))

        errors_optimization[i], nodes_optimization[i], weights_optimization[i] = optimize_error_fBm_general(H, N[i], T)
        print(f'{N[i]}, optimization:')
        print(nodes_optimization[i])
        print(weights_optimization[i])
        print(np.sqrt(errors_optimization[i]))

    return nodes_observation, weights_observation, errors_observation, nodes_optimization, weights_optimization, \
           errors_optimization
