import numpy as np
import scipy


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
    eps = 1e-06
    error = 10 * eps
    while error > eps:
        n, exp_dt_A, exp_dt_A_w, one_exp_times_A, one_exp_times_A_tensored = refine()
        psi_ij_old = psi_ij
        psi_ij = compute_psi_ij()
        error = np.amax(np.abs(psi_ij_old - psi_ij) / psi_ij)
        print(error)

    psi_psi_old = nu ** 2 * np.trapz(one_exp_times_A_tensored[::2, :, :], dx=dt / n * 2, axis=0)
    psi_psi = nu ** 2 * np.trapz(one_exp_times_A_tensored, dx=dt / n, axis=0)
    error = np.amax(np.abs(psi_psi_old - psi_psi) / psi_psi)
    while error > eps:
        n, exp_dt_A, exp_dt_A_w, one_exp_times_A, one_exp_times_A_tensored = refine()
        psi_psi_old = psi_psi
        psi_psi = nu ** 2 * np.trapz(one_exp_times_A_tensored, dx=dt / n, axis=0)
        error = np.amax(np.abs(psi_psi_old - psi_psi) / psi_psi)
        print(error)

    def cov_func(V):
        V_total = weights @ V
        return (theta - lambda_ * V_total - nodes * (V - V_0)) @ psi_ij + V_total * psi_psi

    return cov_func
