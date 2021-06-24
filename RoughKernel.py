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
    rule = qr.quadrature_rule_geometric_mpmath(H, m, n, a, b)
    nodes = rule[0, :]
    weights = rule[1, :]
    summand = T ** (mp.mpf(2 * H)) / (mp.mpf(2 * H) * mp.gamma(mp.mpf(H + 0.5)) ** mp.mpf(2.))
    sum_1 = 0
    sum_2 = 0
    for i in range(n * m):
        for j in range(n * m):
            sum_1 += weights[i] * weights[j] / (nodes[i] + nodes[j]) * (mp.mpf(1.) - mp.exp(-(nodes[i] + nodes[j]) * T))
        sum_2 += weights[i] / nodes[i] ** (mp.mpf(H + 0.5)) * mp.gammainc(mp.mpf(H + 0.5), mp.mpf(0.), nodes[i] * T)
    sum_2 *= mp.mpf(2) / mp.gamma(mp.mpf(H + 0.5))
    c = summand + sum_1 - sum_2
    b = 0
    for i in range(n*m):
        b += weights[i]/nodes[i] * (mp.mpf(1.) - mp.exp(-nodes[i]*T))
    b -= T**(mp.mpf(H+0.5))/(mp.mpf(H+0.5)*mp.gamma(H+0.5))
    b *= mp.mpf(2.)
    a = T
    w_0 = -b/(mp.mpf(2.)*a)
    return np.array([float(mp.sqrt(a*w_0*w_0 + b*w_0 + c)), float(w_0)])


def plot_error_estimates(H, n1, n2, n3, T=1.):
    """
    Plots the errors for the approximations of the rough kernel using m=1 and n=n1, m=2 and n=n2, and m=3 and n=n3.
    Also plots the errors when using best shifts and best constant (rho=0) weight.
    :param H: Hurst parameter
    :param n1: Number of intervals for m=1 (vector)
    :param n2: Number of intervals for m=2 (vector)
    :param n3: Number of intervals for m=3 (vector)
    :param T: Final time
    :return:
    """
    m = 0
    n = 0

    error1 = np.empty(shape=(len(n1)))
    error1_shift = np.empty(shape=(len(n1)))
    error1_const = np.empty(shape=(len(n1)))
    error1_shift_const = np.empty(shape=(len(n1)))

    error2 = np.empty(shape=(len(n2)))
    error2_shift = np.empty(shape=(len(n2)))
    error2_const = np.empty(shape=(len(n2)))
    error2_shift_const = np.empty(shape=(len(n2)))

    error3 = np.empty(shape=(len(n3)))
    error3_shift = np.empty(shape=(len(n3)))
    error3_const = np.empty(shape=(len(n3)))
    error3_shift_const = np.empty(shape=(len(n3)))

    def func(x):
        if x[0] <= 0. or x[1] <= 0.:
            return 10. ** 20
        return error_estimate(H, m, n, x[0], x[1], T)

    def func2(x):
        if x[0] <= 0. or x[1] <= 0.:
            return 10. ** 20
        return error_estimate_improved(H, m, n, x[0], x[1], T)[0]

    m = 1
    for i in range(len(n1)):
        n = n1[i]
        error1[i] = error_estimate(H, m, n, 1., 1., T)
        error1_shift[i] = minimize(func, np.array([1., 1.]), method="nelder-mead",
                                   options={"xatol": 1e-7, "disp": True, "maxiter": 10000}).fun
        error1_const[i] = error_estimate_improved(H, m, n, 1., 1., T)[0]
        error1_shift_const[i] = minimize(func2, np.array([1., 1.]), method="nelder-mead",
                                         options={"xatol": 1e-7, "disp": True, "maxiter": 10000}).fun

    m = 2
    for i in range(len(n2)):
        n = n2[i]
        error2[i] = error_estimate(H, m, n, 1., 1., T)
        error2_shift[i] = minimize(func, np.array([1., 1.]), method="nelder-mead",
                                   options={"xatol": 1e-7, "disp": True, "maxiter": 10000}).fun
        error2_const[i] = error_estimate_improved(H, m, n, 1., 1., T)[0]
        error2_shift_const[i] = minimize(func2, np.array([1., 1.]), method="nelder-mead",
                                         options={"xatol": 1e-7, "disp": True, "maxiter": 10000}).fun

    m = 3
    for i in range(len(n3)):
        n = n3[i]
        error3[i] = error_estimate(H, m, n, 1., 1., T)
        error3_shift[i] = minimize(func, np.array([1., 1.]), method="nelder-mead",
                                   options={"xatol": 1e-7, "disp": True, "maxiter": 10000}).fun
        error3_const[i] = error_estimate_improved(H, m, n, 1., 1., T)[0]
        error3_shift_const[i] = minimize(func2, np.array([1., 1.]), method="nelder-mead",
                                         options={"xatol": 1e-7, "disp": True, "maxiter": 10000}).fun

    plt.loglog(n1, error1, "b-", label="m=1")
    plt.loglog(n1, error1_shift, "r-", label="m=1 + shift")
    plt.loglog(n1+1, error1_const, "g-", label="m=1 + const")
    plt.loglog(n1+1, error1_shift_const, "k-", label="m=1 + shift + const")

    plt.loglog(2*n2, error2, "b--", label="m=2")
    plt.loglog(2*n2, error2_shift, "r--", label="m=2 + shift")
    plt.loglog(2*n2 + 1, error2_const, "g--", label="m=2 + const")
    plt.loglog(2*n2 + 1, error2_shift_const, "k--", label="m=2 + shift + const")

    plt.loglog(3 * n3, error3, "ob--", label="m=3")
    plt.loglog(3 * n3, error3_shift, "or--", label="m=3 + shift")
    plt.loglog(3 * n3 + 1, error3_const, "og--", label="m=3 + const")
    plt.loglog(3 * n3 + 1, error3_shift_const, "ok--", label="m=3 + shift + const")

    plt.legend(loc="upper right")
    plt.xlabel("Number nodes")
    plt.ylabel("Error")
    plt.show()


def plot_errors_sparse(H, T, m_list, n_list):
    """

    :param H:
    :param T:
    :param m_list:
    :param n_list:
    :return:
    """

    def func(x):
        if x[0] <= 0. or x[1] <= 0.:
            return 10. ** 20
        return error_estimate_improved(H, m, n, x[0], x[1], T)[0]

    for j in range(len(m_list)):
        m = m_list[j]
        errors = np.empty(shape=(len(n_list[j])))
        for i in range(len(n_list[j])):
            n = n_list[j][i]
            errors[i] = minimize(func, np.array([1., 1.]), method="nelder-mead",
                                             options={"xatol": 1e-7, "disp": True, "maxiter": 10000}).fun
        plt.loglog(np.array(n_list[j])*m+1, errors, label=f"m={m}")
        print(f"m = {m}")
        print(f"n = {np.array(n_list[j])*m+1}")
        print(f"errors = {errors}")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes")
    plt.ylabel("Error")
    plt.show()


ms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ns = [[2, 4, 8, 16, 32, 64, 128, 256],
      [2, 4, 8, 16, 32, 64, 128],
      [3, 5, 11, 21, 43, 85],
      [2, 4, 8, 16, 32, 64],
      [2, 3, 6, 13, 26, 51],
      [2, 3, 5, 11, 21, 43],
      [2, 5, 9, 18, 37],
      [2, 4, 8, 16, 32],
      [2, 4, 7, 14, 28],
      [2, 3, 6, 13, 26]]
plot_errors_sparse(0.1, 1., ms, ns)
