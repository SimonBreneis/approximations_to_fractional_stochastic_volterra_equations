import math
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
import QuadratureRulesRoughKernel as qr
from scipy.optimize import minimize


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


def plot_kernel_approximations(H, m, n_vec, a=1., b=1., left=0.0001, right=1., number_time_steps=10000):
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

    for i in range(len(n_vec)):
        quad_rule = qr.quadrature_rule_geometric(H, m, n_vec[i], a, b)
        quad_nodes = quad_rule[0, :]
        quad_weights = quad_rule[1, :]
        approximations[:, i+1] = 1 / c_H(H) * np.array(
            [fractional_kernel_laplace(H, t, quad_nodes) for t in time_steps]).dot(quad_weights)

    plt.plot(time_steps, approximations)
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
    rule = qr.quadrature_rule_geometric_exponential_mpmath(H, m, n, a, b)
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
