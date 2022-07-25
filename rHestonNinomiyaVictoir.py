import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp
import scipy


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


def get_sample_path(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., WB=None, N_time=1000, mode="observation"):
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
    if WB is None:
        dW = np.random.normal(0, np.sqrt(T / N_time), N_time)
        dB = np.random.normal(0, np.sqrt(T / N_time), N_time)
    else:
        m = WB.shape[1]
        N_time = WB.shape[2]
        dW = WB[0, 0, :]
        dB = WB[1, 0, :]

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
    expA = scipy.linalg.expm(A)
    temp_1 = A_inverse @ ((expA - np.eye(N)) @ b)
    temp_2 = - dt/2 * (np.dot(weights, A_inverse @ (A_inverse @ (expA @ b - b) - b))/2 + nu*rho*weight_sum/4)
    temp_3 = - dt/2 * np.einsum('i,ij', weights, A_inverse @ (expA-np.eye(N)))/2
    temp_4 = nu * weight_sum * sqrt_dt / 2
    temp_5 = temp_4 / 2

    def solve_drift_ODE(log_S, V):
        """
        Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        log_S_final = log_S + np.dot(temp_3, V) + temp_2
        V_final = expA @ V + temp_1
        V_final = V_final + np.fmax(-np.dot(weights, V_final), 0) * rescaled_weights
        return log_S_final, V_final

    def solve_stochastic_ODE(log_S, V, dW_, dB_):
        """
        Solves the ODE corresponding to the stochastic integrals in the Ninomiya-Victoir scheme for one time step.
        Solves both vector fields (corresponding to the two Brownian motions) simultaneously, as this is possible
        in closed form.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        Z_1 = dW_/sqrt_dt  # np.random.normal(size=m)
        Z_2 = dB_/sqrt_dt  # np.random.normal(size=m)
        total_vol = np.sqrt(np.fmax(np.dot(weights, V), 0))
        if Z_1 < 0:
            tau = np.fmin(1, - total_vol / (temp_4*Z_1))
        else:
            tau = 1
        temp = sqrt_dt*tau*(total_vol + temp_5*Z_1*tau)
        S_final = log_S + temp*(rho*Z_1 + rho_bar*Z_2)
        V_final = V + nu*Z_1*temp
        V_final = V_final + np.fmax(-np.dot(weights, V_final), 0) * rescaled_weights
        return S_final, V_final

    log_S = np.zeros(N_time+1)
    V = np.zeros(N_time+1)
    log_S[0] = log_S_
    V[0] = V_0

    V_components = np.zeros(shape=(len(V_), N_time+1))
    V_components[:, 0] = V_

    for i in range(1, N_time+1):
        if i % 100000 == 0:
            print(f'{i} of {N_time}')
        log_S_, V_ = solve_drift_ODE(log_S_, V_)
        log_S_, V_ = solve_stochastic_ODE(log_S_, V_, dW[i-1], dB[i-1])
        log_S_, V_ = solve_drift_ODE(log_S_, V_)
        log_S[i] = log_S_
        V[i] = np.dot(weights, V_)
        V_components[:, i] = V_

    return np.exp(log_S), np.sqrt(np.fmax(V, 0)), V_components


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


def sample_func(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., WB=None, m=1000, N_time=1000, mode="observation", nodes=None,
                weights=None, sample_paths=False):
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
    if WB is None:
        dW = np.random.normal(0, np.sqrt(T / N_time), (m, N_time))
        dB = np.random.normal(0, np.sqrt(T / N_time), (m, N_time))
    else:
        m = WB.shape[1]
        N_time = WB.shape[2]
        dW = WB[0, :, :]
        dB = WB[1, :, :]
    S_BM = rho * dW + np.sqrt(1 - rho**2) * dB
    dt = T / N_time
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode=mode)
    N = len(nodes)
    if N == 1:
        nodes = np.array([nodes[0], 1])
        weights = np.array([weights[0], 0])
        N = 2

    # dt = T/N_time
    sqrt_dt = np.sqrt(dt)
    rho_bar = np.sqrt(1-rho*rho)
    # nodes, weights = rk.quadrature_rule(H, N, T, mode)
    # N = len(nodes)
    weight_sum = np.sum(weights)
    V_comp = np.zeros(N)
    V_comp[0] = V_0/weights[0]
    V_comp = V_0/(N*weights)
    rescaled_weights = weights / np.sum(weights ** 2)

    A = -(np.diag(nodes) + lambda_ * weights[None, :])*dt/2
    A_inverse = np.linalg.inv(A)
    b = (nodes * V_comp + theta - nu**2 * weight_sum/4) * dt/2
    expA = scipy.linalg.expm(A)
    expA_T = expA.T
    temp_1 = A_inverse @ ((expA - np.eye(N)) @ b)
    temp_2 = - dt/2 * (np.dot(weights, A_inverse @ (A_inverse @ (expA @ b - b) - b))/2 + nu*rho*weight_sum/4)
    temp_3 = - dt/2 * np.einsum('i,ij->j', weights, A_inverse @ (expA-np.eye(N)))/2
    temp_4 = nu * weight_sum / 2
    temp_5 = temp_4 / 2

    log_S = np.log(S_0)
    if sample_paths:
        log_S = np.empty((m, N_time+1))
        V_comp_1 = np.empty((m, N, N_time+1))
        V_comp_1[:, :, 0] = V_comp[None, :]
        V_comp = V_comp_1
        log_S[0] = np.log(S_0)

    def solve_drift_ODE(log_S_, V_comp_):
        """
        Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
        :param log_S_: Initial log-stock price
        :param V_comp_: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        log_S_ = log_S_ + V_comp_ @ temp_3 + temp_2
        V_comp_ = V_comp_ @ expA_T + temp_1[None, :]
        V_comp_ = V_comp_ + np.fmax(-V_comp_ @ weights, 0)[:, None] * rescaled_weights[None, :]
        return log_S_, V_comp_

    def solve_stochastic_ODE(log_S_, V_comp_, dW_, S_BM_):
        """
        Solves the ODE corresponding to the stochastic integrals in the Ninomiya-Victoir scheme for one time step.
        Solves both vector fields (corresponding to the two Brownian motions) simultaneously, as this is possible
        in closed form.
        :param log_S_: Initial log-stock price
        :param V_comp_: Initial variance vector
        :param dW_: Increment of the Brownian motion W driving the volatility process
        :param S_BM_: Increment of the Brownian motion driving the stock price process
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        total_vol = np.sqrt(np.fmax(V_comp_ @ weights, 0.))
        tau = (dW_ < 0) * np.fmin(1, - total_vol / (temp_4*(dW_ + (dW_ == 0.)*1))) + (dW_ >= 0)
        temp = tau*(total_vol + temp_5*dW_*tau)
        log_S_ = log_S_ + temp*S_BM_
        V_comp_ = V_comp_ + (nu*dW_*temp)[:, None]
        V_comp_ = V_comp_ + np.fmax(-V_comp_ @ weights, 0)[:, None] * rescaled_weights[None, :]
        return log_S_, V_comp_

    if sample_paths:
        for i in range(N_time):
            if i % 100000 == 0:
                print(f'{i} of {N_time}')
            log_S[:, i+1], V_comp[:, :, i+1] = solve_drift_ODE(log_S[:, i], V_comp[:, :, i])
            log_S[:, i+1], V_comp[:, :, i+1] = solve_stochastic_ODE(log_S[:, i+1], V_comp[:, :, i+1], dW[:, i], S_BM[:, i])
            log_S[:, i+1], V_comp[:, :, i+1] = solve_drift_ODE(log_S[:, i+1], V_comp[:, :, i+1])
        V = np.fmax(np.einsum('ijk,j->ik', V_comp, weights), 0)
        return np.exp(log_S), np.sqrt(V), V_comp
    else:
        for i in range(N_time):
            if i % 100 == 0:
                print(f'{i} of {N_time}')
            log_S, V_comp = solve_drift_ODE(log_S, V_comp)
            log_S, V_comp = solve_stochastic_ODE(log_S, V_comp, dW[:, i], S_BM[:, i])
            log_S, V_comp = solve_drift_ODE(log_S, V_comp)
        return np.exp(log_S)


def call(K, H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation", m=1000):
    samples = get_samples(H, lambda_, rho, nu, theta, V_0, T, N, S_0=S_0, N_time=N_time, mode=mode, m=m)
    print("generated samples")
    # print(samples)
    return cf.iv_eur_call_MC(samples, K, T, S_0)
