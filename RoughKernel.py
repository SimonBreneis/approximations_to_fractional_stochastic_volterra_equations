import math
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from scipy.optimize import minimize
import orthopy
import quadpy


def fractional_kernel(H, t):
    return t ** (H - 0.5) / math.gamma(H + 0.5)


def fractional_kernel_laplace(H, t, rho):
    return c_H(H) * np.exp(-rho * t)


def c_H(H):
    """
    Returns the constant c_H.
    :param H: Hurst parameter
    :return: c_H
    """
    return 1. / (math.gamma(0.5 + H) * math.gamma(0.5 - H))


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
    summand = T ** (2 * H) / (2 * H * mp.gamma(H + 0.5) ** 2)
    sum_1 = mp.mpf(0)
    sum_2 = mp.mpf(0)
    summands_1 = np.empty(shape=(n*m*n*m))
    summands_2 = np.empty(shape=(n*m))
    for i in range(n * m):
        for j in range(n * m):
            summands_1[i*n*m + j] = weights[i]*weights[j] / (nodes[i]+nodes[j]) * (1 - mp.exp(-(nodes[i]+nodes[j])*T))
        summands_2[i] = weights[i] / nodes[i] ** (H + 0.5) * mp.gammainc(H + 0.5, mp.mpf(0.), nodes[i] * T)
    summands_1.sort()
    summands_2.sort()
    for summand_1 in summands_1:
        sum_1 += summand_1
    for summand_2 in summands_2:
        sum_2 += summand_2
    sum_2 *= 2 / mp.gamma(H + 0.5)
    c = summand + sum_1 - sum_2
    b = mp.mpf(0)
    for i in range(n*m):
        b += weights[i]/nodes[i] * (1 - mp.exp(-nodes[i]*T))
    b -= T**(H+0.5)/((H+0.5)*mp.gamma(H+0.5))
    b *= mp.mpf(2.)
    a = T
    w_0 = -b/(mp.mpf(2.)*a)
    return np.array([np.sqrt(np.fmax(float(a*w_0*w_0 + b*w_0 + c), 0.)), float(w_0)])


def optimize_error_fBm(H, T, m, n, a=0., b=0., tol=1e-8):
    """
    Optimizes the L^2 strong approximation error of the AK scheme over xi_0 and xi_n for fBm. Uses the Nelder-Mead
    optimizer as implemented in scipy with maxiter=10000.
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


def quadrature_rule_geometric_good(H, N, T=1., mode="observation"):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals. The result is an instance of mp.matrix with mp.mpf entries. Here, n, m, xi_0
    and xi_n are chosen according to the mode.
    :param H: Hurst parameter
    :param N: Total number of nodes, N=nm
    :param T: Final time
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
    parameters from the theorem.
    :return: All the nodes and weights, in the form [node1, node2, ...], [weight1, weight2, ...]
    """
    [m, n, a, b] = get_parameters(H, N, T, mode)
    rule = quadrature_rule_geometric_mpmath(H, m, n, a, b, T)
    nodes = rule[0, :]
    weights = rule[1, :]
    return nodes, weights


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


def get_parameters(H, N, T, mode):
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
    return: The converted numpy array.
    """
    y = np.array(x.tolist())
    shape = y.shape
    y = y.flatten()
    y = np.array([float(z) for z in y])
    if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
        return y
    return y.reshape(shape)
