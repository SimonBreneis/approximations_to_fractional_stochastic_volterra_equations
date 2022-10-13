import time

import numpy as np
import scipy
import scipy.integrate


def mean_V(nodes, weights, lambda_, theta, V_0, dt):
    """
    Returns a function f computing f(V_t) = E[V_{t + dt}], where V_t is the Markovian approximation of the volatility.
    :param nodes: The nodes of the quadrature rule
    :param weights: The weights of the quadrature rule
    :param lambda_: Mean-reversion speed
    :param theta: Mean-reversion level
    :param V_0: Vector of initial values of the process V (not V_t!)
    :param dt: Time step length
    :return: The function of the current volatility process computing the expected volatility
    """
    A = - lambda_ * weights[:, None] - np.diag(nodes)
    C = np.linalg.solve(A, scipy.linalg.expm(A * dt) - np.eye(len(nodes))).T

    def mean_func(V):
        return V + C @ ((theta - lambda_ * weights @ V) - nodes * (V - V_0))

    return mean_func


def cov_V(nodes, weights, lambda_, theta, nu, V_0, dt):
    """
    Returns a function f computing f(V_t) = Cov[V_{t + dt}], where V_t is the Markovian approximation of the volatility.
    :param nodes: The nodes of the quadrature rule
    :param weights: The weights of the quadrature rule
    :param lambda_: Mean-reversion speed
    :param theta: Mean-reversion level
    :param nu: Volatility of volatility
    :param V_0: Vector of initial values of the process V (not V_t!)
    :param dt: Time step length
    :return: The function of the current volatility process computing the covariance of the volatility
    """

    def refine():
        ddt = dt / n / 2
        exp_ddt_A = scipy.linalg.expm(ddt * A)
        new_exp_dt_A_w = exp_ddt_A @ weights
        new_one_exp_times_A = np.empty((2 * n + 1, N))
        new_one_exp_times_A[::2, :] = one_exp_times_A
        new_one_exp_times_A[1::2, :] = one_exp_times_A[:-1, :] @ exp_ddt_A
        new_one_exp_times_A_tensored = new_one_exp_times_A[:, :, None] * new_one_exp_times_A[:, None, :]
        return 2 * n, exp_ddt_A, new_exp_dt_A_w, new_one_exp_times_A, new_one_exp_times_A_tensored

    def compute_psi_ij():
        term_1 = exp_dt_A_w[None, None, :, None] * one_exp_times_A_tensored[:-1, :, None, :]
        term_2 = weights[None, None, :, None] * one_exp_times_A_tensored[1:, :, None, :]
        term_3 = dt / n / 2 * (term_1 + term_2)
        psi_current = np.zeros((N, N, N))  # 0th component corresponds to first index of cov matrix,
        # 1st component corresponds to the upcoming dot product, 2nd component corresponds to second index of cov matrix
        psi_ij_int = 0
        for i in range(n - 1):
            psi_current = exp_dt_A @ psi_current + term_3[i]
            psi_ij_int = psi_ij_int + psi_current
        psi_ij_int = psi_ij_int + 0.5 * (exp_dt_A @ psi_current + term_3[n - 1])
        psi_ij_int = psi_ij_int
        return (nu ** 2 * dt / n) * psi_ij_int

    A = - lambda_ * weights[:, None] - np.diag(nodes)
    N = len(nodes)
    n = 64
    temp = np.array([scipy.linalg.expm(dt * i / n * A) for i in range(n + 1)])
    exp_dt_A = temp[1, :, :]
    exp_dt_A_w = exp_dt_A @ weights
    one_exp_times_A = np.sum(temp, axis=1)
    one_exp_times_A_tensored = one_exp_times_A[:, :, None] * one_exp_times_A[:, None, :]
    psi_ij = compute_psi_ij()
    eps = 1e-07
    error = 10 * eps
    while error > eps:
        n, exp_dt_A, exp_dt_A_w, one_exp_times_A, one_exp_times_A_tensored = refine()
        psi_ij_old = psi_ij
        psi_ij = compute_psi_ij()
        error = np.amax(np.abs(psi_ij_old - psi_ij) / psi_ij)

    psi_psi_old = nu ** 2 * np.trapz(one_exp_times_A_tensored[::2, :, :], dx=dt / n * 2, axis=0)
    psi_psi = nu ** 2 * np.trapz(one_exp_times_A_tensored, dx=dt / n, axis=0)
    error = np.amax(np.abs(psi_psi_old - psi_psi) / psi_psi)
    while error > eps:
        n, exp_dt_A, exp_dt_A_w, one_exp_times_A, one_exp_times_A_tensored = refine()
        psi_psi_old = psi_psi
        psi_psi = nu ** 2 * np.trapz(one_exp_times_A_tensored, dx=dt / n, axis=0)
        error = np.amax(np.abs(psi_psi_old - psi_psi) / psi_psi)

    def cov_func(V):
        V_total = weights @ V
        return (theta - lambda_ * V_total - nodes * (V - V_0)) @ psi_ij + V_total * psi_psi

    return cov_func


def first_five_moments_V(nodes, weights, lambda_, theta, nu, V_0, dt, rtol=1e-07):

    def compute(n, max_degree=5):
        """
        Computes the matrices C_ij, up to i <= max_degree.
        :param n: Number of discretization intervals
        :param max_degree: Highest degree that should be computed. Is at most 5
        :return: The matrices C_ij
        """
        N = len(nodes)
        A = - lambda_ * weights[:, None] - np.diag(nodes)

        psi_1 = np.empty((N, N, n + 1))
        for i in range(n + 1):
            psi_1[:, :, i] = scipy.linalg.expm(dt * i / n * A)
        C_11_temp = np.linalg.solve(A, psi_1[:, :, -1] - np.eye(N))
        if max_degree == 1:
            return C_11_temp, C_21, C_22, C_31, C_32, C_41, C_42, C_51, C_52

        phi_1 = np.sum(psi_1, axis=0)
        phi_11 = phi_1[:, None, :] * phi_1[None, :, :]

        e_minus_At = np.empty((N, N, n + 1))
        for i in range(n + 1):
            e_minus_At[:, :, i] = scipy.linalg.expm(- dt * i / n * A)
        e_minus_At_w = np.einsum('ijk,j-> ik', e_minus_At, weights)
        psi_2 = np.zeros((N, N, N, n + 1))
        psi_2[:, :, :, 1:] = nu ** 2 * scipy.integrate.cumtrapz(e_minus_At_w[:, None, None, :] * phi_11[None, :, :, :], dx=dt / n, axis=-1)
        psi_2 = np.einsum('ijk,jlmk->ilmk', psi_1, psi_2)

        C_21_temp = np.trapz(psi_2, dx=dt / n, axis=-1)
        C_22_temp = nu ** 2 * np.trapz(phi_11, dx=dt / n, axis=-1)

        if max_degree == 2:
            return C_11_temp, C_21_temp, C_22_temp, C_31, C_32, C_41, C_42, C_51, C_52

        phi_2 = np.sum(psi_2, axis=0)
        phi_12 = phi_1[:, None, None, :] * phi_2[None, :, :, :] + phi_1[None, :, None, :] * phi_2[:, None, :, :] \
            + phi_1[None, None, :, :] * phi_2[:, :, None, :]
        psi_3 = np.zeros((N, N, N, N, n + 1))
        psi_3[:, :, :, :, 1:] = nu ** 2 * scipy.integrate.cumtrapz(e_minus_At_w[:, None, None, None, :] * phi_12[None, :, :, :, :], dx=dt / n, axis=-1)
        psi_3 = np.einsum('ijk,j...k->i...k', psi_1, psi_3)

        C_31_temp = np.trapz(psi_3, dx=dt / n, axis=-1)
        C_32_temp = nu ** 2 * np.trapz(phi_12, dx=dt / n, axis=-1)

        if max_degree == 3:
            return C_11_temp, C_21_temp, C_22_temp, C_31_temp, C_32_temp, C_41, C_42, C_51, C_52

        phi_3 = np.sum(psi_3, axis=0)
        phi_13 = phi_1[:, None, None, None, :] * phi_3[None, :, :, :, :] \
            + phi_1[None, :, None, None, :] * phi_3[:, None, :, :, :] \
            + phi_1[None, None, :, None, :] * phi_3[:, :, None, :, :] \
            + phi_1[None, None, None, :, :] * phi_3[:, :, :, None, :]
        phi_22 = phi_2[:, :, None, None, :] * phi_2[None, None, :, :, :] \
            + phi_2[:, None, :, None, :] * phi_2[None, :, None, :, :] \
            + phi_2[:, None, None, :, :] * phi_2[None, :, :, None, :]
        phi_13_22 = phi_13 + phi_22

        psi_4 = np.zeros((N, N, N, N, N, n + 1))
        psi_4[:, :, :, :, :, 1:] = nu ** 2 * scipy.integrate.cumtrapz(e_minus_At_w[:, None, None, None, None, :] * phi_13_22[None, :, :, :, :, :], dx=dt / n, axis=-1)
        psi_4 = np.einsum('ijk,j...k->i...k', psi_1, psi_4)

        C_41_temp = np.trapz(psi_4, dx=dt / n, axis=-1)
        C_42_temp = nu ** 2 * np.trapz(phi_13_22, dx=dt / n, axis=-1)

        if max_degree == 4:
            return C_11_temp, C_21_temp, C_22_temp, C_31_temp, C_32_temp, C_41_temp, C_42_temp, C_51, C_52

        phi_4 = np.sum(psi_4, axis=1)
        phi_14 = phi_1[:, None, None, None, None, :] * phi_4[None, :, :, :, :, :] \
            + phi_1[None, :, None, None, None, :] * phi_4[:, None, :, :, :, :] \
            + phi_1[None, None, :, None, None, :] * phi_4[:, :, None, :, :, :] \
            + phi_1[None, None, None, :, None, :] * phi_4[:, :, :, None, :, :] \
            + phi_1[None, None, None, None, :, :] * phi_4[:, :, :, :, None, :]
        phi_23 = phi_2[:, :, None, None, None, :] * phi_3[None, None, :, :, :, :] \
            + phi_2[:, None, :, None, None, :] * phi_3[None, :, None, :, :, :] \
            + phi_2[:, None, None, :, None, :] * phi_3[None, :, :, None, :, :] \
            + phi_2[:, None, None, None, :, :] * phi_3[None, :, :, :, None, :] \
            + phi_2[None, :, :, None, None, :] * phi_3[:, None, None, :, :, :] \
            + phi_2[None, :, None, :, None, :] * phi_3[:, None, :, None, :, :] \
            + phi_2[None, :, None, None, :, :] * phi_3[:, None, :, :, None, :] \
            + phi_2[None, None, :, :, None, :] * phi_3[:, :, None, None, :, :] \
            + phi_2[None, None, :, None, :, :] * phi_3[:, :, None, :, None, :] \
            + phi_2[None, None, None, :, :, :] * phi_3[:, :, :, None, None, :]
        phi_14_23 = phi_14 + phi_23

        psi_5 = np.zeros((N, N, N, N, N, N, n + 1))
        psi_5[:, :, :, :, :, :, 1:] = nu ** 2 * scipy.integrate.cumtrapz(
            e_minus_At_w[:, None, None, None, None, None, :] * phi_14_23[None, :, :, :, :, :, :], dx=dt / n, axis=-1)
        psi_5 = np.einsum('ijk,j...k->i...k', psi_1, psi_5)

        C_51_temp = np.trapz(psi_5, dx=dt / n, axis=-1)
        C_52_temp = nu ** 2 * np.trapz(phi_14_23, dx=dt / n, axis=-1)

        return C_11_temp, C_21_temp, C_22_temp, C_31_temp, C_32_temp, C_41_temp, C_42_temp, C_51_temp, C_52_temp

    n = 64
    C_11, C_21, C_22, C_31, C_32, C_41, C_42, C_51, C_52 = compute(n, 5)
    error = 10 * rtol * np.ones(5)

    while np.amax(error) > rtol:
        n = 2 * n
        highest_degree = (error > rtol).nonzero()[0][-1] + 1
        print(error, highest_degree, n)
        C_11_old, C_21_old, C_22_old, C_31_old, C_32_old, C_41_old, C_42_old, C_51_old, C_52_old = C_11, C_21, C_22, \
            C_31, C_32, C_41, C_42, C_51, C_52
        C_11, C_21, C_22, C_31, C_32, C_41, C_42, C_51, C_52 = compute(n, highest_degree)
        error[0] = np.amax(np.abs((C_11_old - C_11) / C_11))
        error[1] = np.fmax(np.amax(np.abs((C_21_old - C_21) / C_21)), np.amax(np.abs((C_22_old - C_22) / C_22)))
        error[2] = np.fmax(np.amax(np.abs((C_31_old - C_31) / C_31)), np.amax(np.abs((C_32_old - C_32) / C_32)))
        error[3] = np.fmax(np.amax(np.abs((C_41_old - C_41) / C_41)), np.amax(np.abs((C_42_old - C_42) / C_42)))
        error[4] = np.fmax(np.amax(np.abs((C_51_old - C_51) / C_51)), np.amax(np.abs((C_52_old - C_52) / C_52)))

    C_11 = C_11.T
    C_21 = C_21.transpose([1, 2, 0])
    C_31 = C_31.transpose([1, 2, 3, 0])
    C_41 = C_41.transpose([1, 2, 3, 4, 0])
    C_51 = C_51.transpose([1, 2, 3, 4, 5, 0])

    def moments(V):
        V_total = np.dot(weights, V)
        b = (theta - lambda_ * V_total) - nodes * (V - V_0)
        f_1 = V + C_11 @ b
        f_2 = C_21 @ b + V_total * C_22
        f_3 = C_31 @ b + V_total * C_32
        f_4 = C_41 @ b + V_total * C_42
        f_5 = C_51 @ b + V_total * C_52

        f_11 = f_1[:, None] * f_1[None, :]
        f_12 = f_1[:, None, None] * f_2[None, :, :] + f_1[None, :, None] * f_2[:, None, :] \
            + f_1[None, None, :] * f_2[:, :, None]
        f_111 = f_11[:, :, None] * f_1[None, None, :]
        f_13 = f_1[:, None, None, None] * f_3[None, :, :, :] + f_1[None, :, None, None] * f_3[:, None, :, :] \
            + f_1[None, None, :, None] * f_3[:, :, None, :] + f_1[None, None, None, :] * f_3[:, :, :, None]
        f_22 = f_2[:, :, None, None] * f_2[None, None, :, :] + f_2[:, None, :, None] * f_2[None, :, None, :] \
            + f_2[:, None, None, :] * f_2[None, :, :, None]
        f_112 = f_11[:, :, None, None] * f_2[None, None, :, :] + f_11[:, None, :, None] * f_2[None, :, None, :] \
            + f_11[:, None, None, :] * f_2[None, :, :, None] + f_11[None, :, :, None] * f_2[:, None, None, :] \
            + f_11[None, :, None, :] * f_2[:, None, :, None] + f_11[None, None, :, :] * f_2[:, :, None, None]
        f_1111 = f_111[:, :, :, None] * f_1[None, None, None, :]
        f_23 = f_2[:, :, None, None, None] * f_3[None, None, :, :, :] \
            + f_2[:, None, :, None, None] * f_3[None, :, None, :, :] \
            + f_2[:, None, None, :, None] * f_3[None, :, :, None, :] \
            + f_2[:, None, None, None, :] * f_3[None, :, :, :, None] \
            + f_2[None, :, :, None, None] * f_3[:, None, None, :, :] \
            + f_2[None, :, None, :, None] * f_3[:, None, :, None, :] \
            + f_2[None, :, None, None, :] * f_3[:, None, :, :, None] \
            + f_2[None, None, :, :, None] * f_3[:, :, None, None, :] \
            + f_2[None, None, :, None, :] * f_3[:, :, None, :, None] \
            + f_2[None, None, None, :, :] * f_3[:, :, :, None, None]
        f_14 = f_1[:, None, None, None, None] * f_4[None, :, :, :, :] \
            + f_1[None, :, None, None, None] * f_4[:, None, :, :, :] \
            + f_1[None, None, :, None, None] * f_4[:, :, None, :, :] \
            + f_1[None, None, None, :, None] * f_4[:, :, :, None, :] \
            + f_1[None, None, None, None, :] * f_4[:, :, :, :, None]
        f_122 = f_1[:, None, None, None, None] * f_22[None, :, :, :, :] \
            + f_1[None, :, None, None, None] * f_22[:, None, :, :, :] \
            + f_1[None, None, :, None, None] * f_22[:, :, None, :, :] \
            + f_1[None, None, None, :, None] * f_22[:, :, :, None, :] \
            + f_1[None, None, None, None, :] * f_22[:, :, :, :, None]
        f_113 = f_11[:, :, None, None, None] * f_3[None, None, :, :, :] \
            + f_11[:, None, :, None, None] * f_3[None, :, None, :, :] \
            + f_11[:, None, None, :, None] * f_3[None, :, :, None, :] \
            + f_11[:, None, None, None, :] * f_3[None, :, :, :, None] \
            + f_11[None, :, :, None, None] * f_3[:, None, None, :, :] \
            + f_11[None, :, None, :, None] * f_3[:, None, :, None, :] \
            + f_11[None, :, None, None, :] * f_3[:, None, :, :, None] \
            + f_11[None, None, :, :, None] * f_3[:, :, None, None, :] \
            + f_11[None, None, :, None, :] * f_3[:, :, None, :, None] \
            + f_11[None, None, None, :, :] * f_3[:, :, :, None, None]
        f_1112 = f_111[:, :, :, None, None] * f_2[None, None, None, :, :] \
            + f_111[:, :, None, :, None] * f_2[None, None, :, None, :] \
            + f_111[:, :, None, None, :] * f_2[None, None, :, :, None] \
            + f_111[:, None, :, :, None] * f_2[None, :, None, None, :] \
            + f_111[:, None, :, None, :] * f_2[None, :, None, :, None] \
            + f_111[:, None, None, :, :] * f_2[None, :, :, None, None] \
            + f_111[None, :, :, :, None] * f_2[:, None, None, None, :] \
            + f_111[None, :, :, None, :] * f_2[:, None, None, :, None] \
            + f_111[None, :, None, :, :] * f_2[:, None, :, None, None] \
            + f_111[None, None, :, :, :] * f_2[:, :, None, None, None]
        f_11111 = f_1[:, None, None, None, None] * f_1111[None, :, :, :, :] \
            + f_1[None, :, None, None, None] * f_1111[:, None, :, :, :] \
            + f_1[None, None, :, None, None] * f_1111[:, :, None, :, :] \
            + f_1[None, None, None, :, None] * f_1111[:, :, :, None, :] \
            + f_1[None, None, None, None, :] * f_1111[:, :, :, :, None]

        g_1 = f_1
        g_2 = f_2 + f_11
        g_3 = f_3 + f_12 + f_111
        g_4 = f_4 + f_13 + f_22 + f_112 + f_1111
        g_5 = f_5 + f_23 + f_14 + f_122 + f_113 + f_1112 + f_11111

        return g_1, g_2, g_3, g_4, g_5

    # return C_11, C_21, C_22, C_31, C_32, C_41, C_42, C_51, C_52
    return moments

