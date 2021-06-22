import mpmath as mp
import numpy as np


def quadrature_rule_interval(H, m, a, b):
    """
    Returns the nodes and weights of the Gauss quadrature rule level m for the fractional weight function on [a,b]
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param a: Left end of interval
    :param b: Right end of interval
    :return: The nodes and weights, in form [[node1, node2, ...], [weight1, weight2, ...]]
    """
    fraction = b / a
    c_H = 1. / (mp.gamma(mp.mpf(0.5) + H) * mp.gamma(mp.mpf(0.5) - H))
    if m == 1:
        node = (0.5 - H) / (1.5 - H) * a * (fraction ** (1.5 - H) - 1.) / (fraction ** (0.5 - H) - 1.)
        weight = c_H / (0.5 - H) * (a ** (0.5 - H)) * (fraction ** (0.5 - H) - 1.)
        return mp.matrix([[node], [weight]])
    if m == 2:
        a00 = (0.5 - H) / (1.5 - H) * a * (fraction ** (1.5 - H) - 1.) / (fraction ** (0.5 - H) - 1.)
        a10 = (1. / (2.5 - H) * (a * a) * (fraction ** (2.5 - H) - 1.) - a00 / (1.5 - H) * a *
               (fraction ** (1.5 - H) - 1.)) / (1. / (0.5 - H) * (fraction ** (0.5 - H) - 1.))
        a11 = (1. / (3.5 - H) * (a ** 3) * (fraction ** (3.5 - H) - 1.) - 2. * a00 / (2.5 - H) * a * a * (
                fraction ** (2.5 - H) - 1.) + a00 ** 2. / (1.5 - H) * a * (fraction ** (1.5 - H) - 1.)) / (
                      1. / (2.5 - H) * a * a * (fraction ** (2.5 - H) - 1.) - 2. * a00 / (1.5 - H) * a * (
                       fraction ** (1.5 - H) - 1.) + a00 ** 2. / (0.5 - H) * (fraction ** (0.5 - H) - 1.))
        x1 = (a11 + a00) / 2. + np.sqrt(((a11 + a00) / 2.) ** 2 + a10 - a11 * a00)
        x2 = (a11 + a00) / 2. - np.sqrt(((a11 + a00) / 2.) ** 2 + a10 - a11 * a00)
        numerator = 1. / (2.5 - H) * (a ** (2.5 - H)) * (fraction ** (2.5 - H) - 1.) - 2. * a00 / (1.5 - H) * (
                a ** (1.5 - H)) * (fraction ** (1.5 - H) - 1.) + a00 ** 2 / (0.5 - H) * (a ** (0.5 - H)) * (
                            fraction ** (0.5 - H) - 1.)
        w1 = c_H * numerator / ((2 * x1 - a11 - a00) * (x1 - a00))
        w2 = c_H * numerator / ((2 * x2 - a11 - a00) * (x2 - a00))
        return mp.matrix([[x1, x2], [w1, w2]])


def quadrature_rule(H, m, partition):
    """
    Returns the quadrature rule of level m of the fractional kernel with Hurst parameter H on all the partition
    intervals.
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


def quadrature_rule_geometric(H, m, n, a=1., b=1.):
    """
    Returns the nodes and weights of the m-point quadrature rule for the fractional kernel with Hurst parameter H
    on n geometrically spaced subintervals.
    :param H: Hurst parameter
    :param m: Level of the quadrature rule
    :param n: Number of subintervals
    :param a: Can shift the left end-point of the total interval
    :param b: Can shift the right end-point of the total interval
    :return: All the nodes and weights, in the form [[node1, node2, ...], [weight1, weight2, ...]]
    """
    r = 1. * m
    gamma = 0.5 - H
    delta = H
    xi0 = a * n ** (-r / gamma)
    xin = b * n ** (r / delta)
    partition = np.array([xi0 ** (float(n - i) / n) * xin ** (float(i) / n) for i in range(0, n + 1)])
    return quadrature_rule(H, m, partition)
