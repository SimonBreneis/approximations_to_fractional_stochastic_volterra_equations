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


def error_estimate(H, m, n, a=1., b=1., T=1.):
    """
    Computes an error estimate of the L^2-norm of the difference between the rough kernel and its approximation
    on [0, T].
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of quadrature intervals
    :param a: Can shift the left endpoint of the total interval of the quadrature rule
    :param b: Can shift the right endpoint of the total interval of the quadrature rule
    :param T: Final time
    :return: An error estimate
    """
    T = mp.mpf(T)
    rule = qr.quadrature_rule_geometric_mpmath(H, m, n, a, b)
    nodes = rule[0, :]
    weights = rule[1, :]
    summand = T**(mp.mpf(2*H))/(mp.mpf(2*H) * mp.gamma(mp.mpf(H+0.5))**mp.mpf(2.))
    sum_1 = 0
    sum_2 = 0
    for i in range(n*m):
        for j in range(n*m):
            sum_1 += weights[i] * weights[j] / (nodes[i] + nodes[j]) * (mp.mpf(1.) - mp.exp(-(nodes[i]+nodes[j])*T))
        sum_2 += weights[i] / nodes[i]**(mp.mpf(H+0.5)) * mp.gammainc(mp.mpf(H+0.5), mp.mpf(0.), nodes[i]*T)
    sum_2 *= mp.mpf(2)/mp.gamma(mp.mpf(H+0.5))
    return float(mp.sqrt(summand + sum_1 - sum_2))


def error_estimate_improved(H, m, n, a=1., b=1., T=1.):
    """
        Computes an error estimate of the L^2-norm of the difference between the rough kernel and its approximation
        on [0, T]. The approximation contains a constant/Brownian term.
        :param H: Hurst parameter
        :param m: Level of the quadrature rule
        :param n: Number of quadrature intervals
        :param a: Can shift the left endpoint of the total interval of the quadrature rule
        :param b: Can shift the right endpoint of the total interval of the quadrature rule
        :param T: Final time
        :return: An error estimate and the optimal weight for the constant term: [error, weight]
        """
    T = mp.mpf(T)
    H = mp.mpf(H)
    a = mp.mpf(a)
    b = mp.mpf(b)
    rule = qr.quadrature_rule_geometric_mpmath(H, m, n, a, b)
    nodes = rule[0, :]
    weights = rule[1, :]
    summand = T ** (2 * H) / (2 * H * mp.gamma(H + 0.5) ** 2)
    sum_1 = mp.mpf(0)
    sum_2 = mp.mpf(0)
    summands_1 = np.empty(shape=(n*m*n*m))
    summands_2 = np.empty(shape=(n*m))
    for i in range(n * m):
        for j in range(n * m):
            summands_1[i*n*m + j] = weights[i] * weights[j] / (nodes[i] + nodes[j]) * (1 - mp.exp(-(nodes[i] + nodes[j]) * T))
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
    return np.array([float(mp.sqrt(a*w_0*w_0 + b*w_0 + c)), float(w_0)])


def error_estimate_improved_exponential(H, m, n, a, b, T):
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
            summands_1[i*n*m + j] = weights[i] * weights[j] / (nodes[i] + nodes[j]) * (1 - mp.exp(-(nodes[i] + nodes[j]) * T))
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
    return np.array([float(mp.sqrt(a*w_0*w_0 + b*w_0 + c)), float(w_0)])


def plot_errors_sparse(H, T, m_list, n_list):
    """

    :param H:
    :param T:
    :param m_list:
    :param n_list:
    :return:
    """

    def func(x):
        if x[0] + x[1] <= 0.:
            return 10.**20
        return error_estimate_improved_exponential(H, m, n, x[0], x[1], T)[0]

    for j in range(len(m_list)):
        x0 = np.array([-1., 1.])
        m = m_list[j]
        errors = np.empty(shape=(len(n_list[j])))
        fatol = 1e-8
        for i in range(len(n_list[j])):
            n = n_list[j][i]
            res = minimize(func, x0, method="nelder-mead",
                                             options={"fatol": fatol, "disp": True, "maxiter": 10000})
            errors[i] = res.fun
            x_0 = res.x
            print(res.x)
            fatol = 1e-8 * errors[i]
        plt.loglog(np.array(n_list[j])*m+1, errors, label=f"m={m}")
        print(f"m = {m}")
        print(f"n = {np.array(n_list[j])*m+1}")
        print(f"errors = {errors}")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes")
    plt.ylabel("Error")
    plt.show()


mp.mp.dps = 1000
#print(error_estimate_improved(0.1, 5, 205, 2e+24, 2e-42, 1.))

ms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ns = [[1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024],
      [1, 2, 3, 4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512],
      [1, 2, 3, 4, 5, 8, 11, 15, 21, 30, 43, 60, 85, 121, 171, 241, 341],
      [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256],
      [1, 2, 3, 5, 6, 9, 13, 18, 26, 36, 51, 72, 102, 145, 205],
      [1, 2, 3, 4, 5, 7, 11, 15, 21, 30, 43, 60, 85, 121, 171],
      [1, 2, 3, 5, 6, 9, 13, 18, 26, 37, 52, 73, 103, 146],
      [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128],
      [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 57, 80, 114],
      [1, 2, 3, 4, 6, 9, 13, 18, 26, 36, 51, 72, 102]]
plot_errors_sparse(0.1, 1., ms, ns)
