import time

import numpy as np
import scipy


def l2_norm(x):
    """
    Computes the l^2-norm of a vector
    :param x: Vector
    :return: L^2-norm of the vector
    """
    return np.sqrt(np.sum(x ** 2))


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

    def compute_psi_ij(n):
        times = np.linspace(0, dt, n + 1)
        times_A = times[:, None, None] * A[None, :, :]
        print(times_A.shape)
        exp_times_A = np.array([scipy.linalg.expm(times_A[i, :, :]) for i in range(len(times))])
        exp_dt_A_w = exp_times_A[1, :, :] @ weights
        one_exp_times_A = np.sum(exp_times_A, axis=1)
        one_exp_times_A_tensored = one_exp_times_A[:, :, None] * one_exp_times_A[:, None, :]
        psi_ij = np.zeros(
            (n + 1, N, N, N))  # 0th component is time, 1st component corresponds to first index of cov matrix
        # 2nd component corresponds to second index of cov matrix, 3rd component corresponds to the upcoming dot product
        for i in range(n):
            psi_ij[i + 1, :, :, :] = exp_times_A[1, :, :] @ psi_ij[i, :, :, :] \
                                     + 0.5 * dt / n * (exp_dt_A_w[None, :, None] * one_exp_times_A_tensored[i, :, None, :]
                                              + weights[None, :, None] * one_exp_times_A_tensored[i + 1, :, None, :])

        psi_ij_int = np.trapz(psi_ij, x=times, axis=0)
        return nu ** 2 * psi_ij_int

    A = - lambda_ * weights[:, None] - np.diag(nodes)
    N = len(nodes)
    n = 64
    psi_ij = compute_psi_ij(n=n)
    eps = 1e-06
    error = 10 * eps
    while error > eps:
        n = 2 * n
        psi_ij_old = psi_ij
        psi_ij = compute_psi_ij(n)
        error = np.amax(np.abs(psi_ij_old - psi_ij) / psi_ij)
        print(error)

    def compute_psi_psi(n):
        times = np.linspace(0, dt, n + 1)
        times_A = times[:, None, None] * A[None, :, :]
        exp_times_A = np.array([scipy.linalg.expm(times_A[i, :, :]) for i in range(n + 1)])
        one_exp_times_A = np.sum(exp_times_A, axis=1)
        one_exp_times_A_tensored = one_exp_times_A[:, :, None] * one_exp_times_A[:, None, :]
        return nu ** 2 * np.trapz(one_exp_times_A_tensored, x=times, axis=0)

    n = 64
    psi_psi = compute_psi_psi(n=n)
    error = 10 * eps
    while error > eps:
        n = 2 * n
        psi_psi_old = psi_psi
        psi_psi = compute_psi_psi(n)
        error = np.amax(np.abs(psi_psi_old - psi_psi) / psi_psi)
        print(error)

    '''
    A = - lambda_ * weights[:, None] - np.diag(nodes)
    Aw = A @ weights
    oneA = np.sum(A, axis=0)
    w_oneA = weights[:, None] * oneA[None, :]
    AAw = A @ Aw
    Aw_oneA = A @ w_oneA
    w_oneAA = w_oneA @ A
    oneAA = A.T @ oneA
    oneAAA = A.T @ oneAA
    temp_1 = np.zeros((len(nodes), len(nodes), len(nodes)))
    temp_1 = temp_1 + weights[None, None, :] * (dt ** 2 / 2)
    temp_1 = temp_1 + Aw[None, None, :] * (dt ** 3 / 2)
    temp_1 = temp_1 + w_oneA[None, :, :] * (dt ** 3 / 6) + w_oneA[:, None, :] * (dt ** 3 / 6)
    temp_1 = temp_1 + AAw[None, None, :] * (7 * dt ** 4 / 24)
    temp_1 = temp_1 + Aw_oneA[None, :, :] * (5 * dt ** 4 / 24) + Aw_oneA[:, None, :] * (5 * dt ** 4 / 24)
    temp_1 = temp_1 + w_oneAA[None, :, :] * (dt ** 4 / 24) + w_oneAA[:, None, :] * (dt ** 4 / 24)
    temp_1 = temp_1 + oneA[:, None, None] * oneA[None, :, None] * weights[None, None, :] * (dt ** 4 / 12)
    temp_1 = nu ** 2 * temp_1

    temp_2 = dt * np.ones((len(nodes), len(nodes)))
    temp_2 = temp_2 + (oneA[:, None] + oneA[None, :]) * (dt ** 2 / 2)
    temp_2 = temp_2 + (oneAA[:, None] + oneAA[None, :]) * (dt ** 3 / 6)
    temp_2 = temp_2 + oneA[:, None] * oneA[None, :] * (dt ** 3 / 3)
    temp_2 = temp_2 + (oneAAA[:, None] + oneAAA[None, :]) * (dt ** 4 / 24)
    temp_2 = temp_2 + (oneAA[:, None] * oneA[None, :] + oneA[:, None] * oneAA[None, :]) * (dt ** 4 / 8)
    temp_2 = nu ** 2 * temp_2
    
    def cov_func(V):
        V_total = weights @ V
        return np.einsum('ijk,i->jk', temp_1, ((theta - lambda_ * V_total) - nodes * (V - V_0))) + V_total * temp_2
    '''

    def cov_func(V):
        V_total = weights @ V
        return (theta - lambda_ * V_total - nodes * (V - V_0)) @ psi_ij + V_total * psi_psi

    return cov_func
