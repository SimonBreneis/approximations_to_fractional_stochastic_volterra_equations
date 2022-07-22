import time

import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp
import scipy


'''
def get_largest_eigenvalue(A):
    """
    Computes the largest eigenvalue together with the corresponding eigenvector using power iteration.
    """
    min_iter = 10
    eps = mp.mpf(10) ** (-mp.mp.dps + 5)
    x = mp.randmatrix(A.rows, 1)
    x_old = x / mp.norm(x)
    x_new = x_old.copy()
    i = 1
    E, U = mp.eig(A)
    print(E)

    while i <= min_iter or mp.norm(x_new-x_old) > eps:
        # print(mp.norm(x_new - x_old))
        x_old = x_new.copy()
        x_new = A * x_old
        x_new = x_new / mp.norm(x_new)
        i = i+1
    print(i)
    return mp.fdot(x_new, A*x_new)/mp.norm(x_new), x_new


def get_largest_eigenvalues(A, n=2):
    """
    Computes the n=2 or 3 largest eigenvalues and the corresponding eigenvectors of A using power iteration.
    """
    eigvalues = mp.matrix([mp.mpf(0.)]*n)
    eigvectors = mp.matrix(A.rows, n)
    val, vec = get_largest_eigenvalue(A)
    print("eigvalues")
    eigvalues[0] = val
    eigvectors[:, 0] = vec
    B = A.copy() - val * vec * vec.T
    val, vec = get_largest_eigenvalue(B)
    eigvalues[1] = val
    eigvectors[:, 1] = (val - eigvalues[0])*vec + eigvalues[0]*mp.fdot(eigvectors[:, 0], vec)*eigvectors[:, 0]
    return eigvalues, eigvectors


def SMW(diag, pert):
    """
    Computes the inverse of the matrix diag(diag) + (pert, pert, ..., pert)^T
    """
    n = len(diag)
    diag_inv = mp.matrix([1/x for x in diag])
    X = mp.diag(diag_inv)
    c = mp.fdot(pert, diag_inv)
    M = mp.matrix([[pert[j]/(diag[i]*diag[j]) for j in range(n)] for i in range(n)])
    corr = M / (1+c)
    return X - corr


def diagonalize(nodes, weights, lambda_):
    """
    Diagonalizes the matrix -diag(nodes) - lambda_ (w,w,...,w)^T in a numerically stable way.
    """
    w = lambda_*weights.copy()
    n = len(nodes)
    eigvalues = mp.matrix([mp.mpf(0.)] * n)
    eigvectors = mp.matrix(n, n)
    A = mp.diag(nodes) + mp.matrix([[w[i] for i in range(n)] for _ in range(n)])
    val, vec = get_largest_eigenvalues(A)
    eigvalues[0] = val[0]
    eigvalues[1] = val[1]
    eigvectors[:, 0] = vec[:, 0]
    eigvectors[:, 1] = vec[:, 1]

    A_inv_approx = SMW(nodes + nodes[0], w)
    val, vec = get_largest_eigenvalues(A_inv_approx)
    eigvalues[n-1] = 1/val[0] - nodes[0]
    eigvalues[n-2] = 1/val[1] - nodes[0]
    eigvectors[:, n-1] = vec[:, 0]
    eigvectors[:, n-2] = vec[:, 1]

    avg_geo_dist = (eigvalues[0] / eigvalues[n-1])**(1/(n-1))
    n_found = 4

    while n_found < n:
        found_new = False
        exponent = 0.
        attempt = 2
        while not found_new:
            print(attempt)
            perturbation = nodes[n_found - 4]/avg_geo_dist**exponent
            print(eigvalues)
            print(avg_geo_dist)
            print(perturbation)
            print(mp.eig(A)[0])
            print(mp.eig(mp.inverse(A - perturbation*mp.eye(n)))[0])
            print("Here")
            A_pert_inv = SMW(nodes - perturbation, w)
            print("Here")
            val, vec = get_largest_eigenvalues(A_pert_inv)
            print("Here")
            val_1 = 1/val[1] + perturbation
            val_2 = 1/val[0] + perturbation
            print("Here")
            if mp.almosteq(val_1, eigvalues[n_found - 4]):
                exponent = exponent + 1/attempt
            elif not mp.almosteq(val_1, eigvalues[n_found-3]) and not mp.almosteq(val_2, eigvalues[n_found-3]):
                exponent = min(exponent - 1/attempt, 2**(-attempt))
            elif mp.almosteq(val_1, eigvalues[n_found-3]):
                eigvalues[n_found-2] = val_2
                eigvectors[:, n_found-2] = vec[0]
                found_new = True
            else:
                eigvalues[n_found-2] = val_1
                eigvectors[:, n_found-2] = vec[1]
                found_new = True
            attempt += 1

    return -eigvalues, eigvectors
'''


def lu_solve_system(A, B):
    """
    Solves A^{-1}B where B is a matrix (and not just a vector).
    """
    return np.linalg.solve(A, B)


def diagonalize(nodes, weights, lambda_, dt):
    """
    Diagonalizes the matrix (-diag(nodes) - lambda_ (w,w,...,w)^T)*dt.
    """
    A = -(np.diag(nodes) + lambda_ * weights[None, :])*dt
    return np.linalg.eig(A)


def exp_matrix(A):
    B = A[1] @ np.diag(np.exp(A[0]))
    return lu_solve_system(A[1].T, B.T).T


def ODE_drift(A, b):
    c = np.linalg.solve(A[1], b)
    M = np.diag([(np.exp(x)-1)/x if x**2/6 > 1e-200 else 1-x/2 for x in A[0]])
    c = M @ c
    return A[1] @ c


def ODE_S_drift(A, b, weights):
    M = np.diag([((np.exp(x) - 1)/x - 1)/x if x**2 / 24 > 1e-200 else (1/2 + x/6) for x in A[0]])
    print('A', A[1].T)
    print('w', weights)
    # C = A[1].T * (weights/2).T  # C = (w/2)^T @ A[1]
    C = (weights/2).T @ A[1]
    D = (M @ C).T
    c = np.linalg.solve(A[1], b)
    return (D @ c)


def ODE_S_mult(A, weights):
    M = np.diag(np.exp(A[0])/A[0])
    # C = A[1].T * (weights/2).T
    C = (weights / 2).T @ A[1]
    D = M @ C
    return np.linalg.solve(A[1].T, D)


def get_sample_path(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation"):
    """
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps used in the solution of the fractional Riccati equation
    :param N: Total number of points in the quadrature rule, N=n*m
    :param S_0: Initial stock price
    :param mode: If observation, use the values of the interpolation of the optimum. If theorem, use the values of the
    theorem.
    """
    dt = T/N_time
    sqrt_dt = np.sqrt(dt)
    log_S_ = np.log(S_0)
    rho_bar = np.sqrt(1-rho*rho)
    nodes, weights = rk.quadrature_rule(H, N, T, mode)
    N = len(nodes)
    weight_sum = np.sum(weights)
    V_ = V_0/(N*weights)
    rescaled_weights = weights / np.sum(weights ** 2)

    A = -(np.diag(nodes) + lambda_ * weights[None, :])*dt/2
    A_inverse = np.linalg.inv(A)
    b = (nodes * V_ + theta - nu**2 * weight_sum/4) * dt/2
    print(nodes*V_, theta, -nu**2*weight_sum/4)
    print(-A_inverse @ b)
    expA = scipy.linalg.expm(A)
    temp_1 = A_inverse @ ((expA - np.eye(N)) @ b)
    temp_2 = - dt/2 * (np.dot(weights, A_inverse @ (A_inverse @ (expA @ b - b) - b))/2 + nu*rho*weight_sum/4)
    temp_3 = - dt/2 * np.einsum('i,ij', weights, A_inverse @ (expA-np.eye(N)))/2
    temp_4 = nu * weight_sum * sqrt_dt / 2
    temp_5 = temp_4 / 2
    print('nodes', nodes)
    print('weights', weights)
    print('A', A)
    print('A_inverse', A_inverse)
    print('expA', expA)
    print('w...', temp_3)
    # time.sleep(36000)

    def solve_drift_ODE(log_S, V):
        """
        Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        V_final = expA @ V + temp_1
        log_S_final = log_S + np.dot(temp_3, V) + temp_2
        V_final = V_final + np.fmax(-np.dot(weights, V_final), 0) * 2 * rescaled_weights
        return log_S_final, V_final

    def solve_stochastic_ODE(log_S, V):
        """
        Solves the ODE corresponding to the stochastic integrals in the Ninomiya-Victoir scheme for one time step.
        Solves both vector fields (corresponding to the two Brownian motions) simultaneously, as this is possible
        in closed form.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        Z_1 = np.random.normal()
        Z_2 = np.random.normal()
        total_vol = np.sqrt(np.fmax(np.dot(weights, V), 0))
        if Z_1 < 0:
            tau = np.fmin(1, - total_vol / (temp_4*Z_1))
        else:
            tau = 1
        temp = sqrt_dt*tau*(total_vol + temp_5*Z_1*tau)
        V_final = V + nu*Z_1*temp
        S_final = log_S + temp*(rho*Z_1 + rho_bar*Z_2)
        V_final = V_final + np.fmax(-np.dot(weights, V_final), 0) * 2 * rescaled_weights
        return S_final, V_final

    log_S = np.zeros(N_time+1)
    V = np.zeros(N_time+1)
    log_S[0] = log_S_
    V[0] = V_0

    V_components = np.zeros(shape=(len(V_), N_time+1))
    V_components[:, 0] = V_

    for i in range(1, N_time+1):
        if i % 100 == 0:
            print(f'{i} of {N_time}')
        log_S_, V_ = solve_drift_ODE(log_S_, V_)
        log_S_, V_ = solve_stochastic_ODE(log_S_, V_)
        log_S_, V_ = solve_drift_ODE(log_S_, V_)
        log_S[i] = log_S_
        V[i] = np.dot(weights, V_)
        V_components[:, i] = V_

    return np.exp(log_S), V, V_components


def get_samples(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation", m=1000):
    """
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps used in the solution of the fractional Riccati equation
    :param N: Total number of points in the quadrature rule, N=n*m
    :param S_0: Initial stock price
    :param mode: If observation, use the values of the interpolation of the optimum. If theorem, use the values of the
    theorem.
    """
    dt = T/N_time
    sqrt_dt = np.sqrt(dt)
    log_S = np.log(S_0)
    rho_bar = np.sqrt(1-rho*rho)
    nodes, weights = rk.quadrature_rule(H, N, T, mode)
    N = len(nodes)
    weight_sum = mp.fsum(weights)
    V = mp.matrix([V_0/(N*weight) for weight in weights])

    A = diagonalize(nodes, weights, lambda_)  # A = [D, U], A = UDU^{-1}
    b = mp.matrix([nodes[i]*V[i] + theta - nu**2*weight_sum/4 for i in range(N)])*dt/2
    eAt = exp_matrix(A, dt/2)  # eAt = exp(A dt/2)
    Ainv_eAt_id_b = ODE_drift(A, b, dt/2)  # A^{-1} (exp(A dt/2) - Id) b
    S_drift = ODE_S_drift(A, b, dt/2, weights)  # w/2 cdot (A^{-2} exp(A dt/2) b - A^{-2} b - A^{-1} b dt/2)
    w_Ainv_eAt = ODE_S_mult(A, dt/2, weights)  # w/2 cdot A^{-1} exp(A dt/2)

    weight_sum = float(weight_sum)
    V = np.array([[float(v) for _ in range(m)] for v in V])
    weights = rk.mp_to_np(weights)
    eAt = rk.mp_to_np(eAt)
    Ainv_eAt_id_b = rk.mp_to_np(Ainv_eAt_id_b)
    print(f"The drift {Ainv_eAt_id_b}")
    S_drift = float(S_drift)
    w_Ainv_eAt = rk.mp_to_np(w_Ainv_eAt)

    def solve_drift_ODE(log_S, V):
        """
        Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        V_final = np.einsum('ij,jk', eAt, V) + Ainv_eAt_id_b.repeat(m).reshape(-1, m)
        log_S_final = log_S - dt/2 * (nu*rho*weight_sum*dt/8 + np.einsum('i,ij', w_Ainv_eAt, V) + S_drift)
        return log_S_final, V_final

    def solve_stochastic_ODE(log_S, V):
        """
        Solves the ODE corresponding to the stochastic integrals in the Ninomiya-Victoir scheme for one time step.
        Solves both vector fields (corresponding to the two Brownian motions) simultaneously, as this is possible
        in closed form.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        Z_1 = np.random.normal(size=m)
        Z_2 = np.random.normal(size=m)
        total_vol = np.sqrt(np.fmax(np.einsum('i,ij', weights, V), 0.))
        tau = - 2*total_vol / (nu*weight_sum*sqrt_dt*(Z_1 + (Z_1 == 0.)*1))
        tau = tau * (tau >= 0.) + 1. * (tau < 0.)
        tau = np.fmin(tau, 1.)
        temp = sqrt_dt*(total_vol*tau + nu*weight_sum*sqrt_dt*Z_1*tau**2/4)
        V_final = V + nu*Z_1*temp
        S_final = log_S + temp*(rho*Z_1 + rho_bar*Z_2)
        return S_final, V_final

    log_S = log_S * np.ones(m)
    for i in range(1, N_time+1):
        log_S, V = solve_drift_ODE(log_S, V)
        log_S, V = solve_stochastic_ODE(log_S, V)
        log_S, V = solve_drift_ODE(log_S, V)

    return np.exp(log_S)


def call(K, H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation", m=1000):
    samples = get_samples(H, lambda_, rho, nu, theta, V_0, T, N, S_0=S_0, N_time=N_time, mode=mode, m=m)
    print("generated samples")
    # print(samples)
    return cf.iv_eur_call_MC(samples, K, T, S_0)
