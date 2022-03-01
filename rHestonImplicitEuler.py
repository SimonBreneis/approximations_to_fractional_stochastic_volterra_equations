import time
from matplotlib import pyplot as plt
import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp
from scipy.optimize import fsolve
import rHestonSplitKernel as sk

'''
def eq(p):
    x, y = p
    return (x + y**2-4, np.exp(x)+x*y-3)

def der(p):
    x, y = p
    return np.array([[1, np.exp(x) + y], [2*y, x]])


tic = time.perf_counter()
x, y = fsolve(func=eq, x0=(1, 1))
print(time.perf_counter()-tic)

tic = time.perf_counter()
x, y = fsolve(func=eq, x0=(1, 1), fprime=der, col_deriv=True)
print(time.perf_counter()-tic)

print(x, y)
print(eq((x, y)))
time.sleep(3600)
'''

def lu_solve_system(A, B):
    """
    Solves A^{-1}B where B is a matrix (and not just a vector).
    """
    n = A.rows
    result = mp.zeros(n, n)
    for i in range(n):
        result[:, i] = mp.lu_solve(A, B[:, i])
    return result


def diagonalize(nodes, weights, lambda_):
    """
    Diagonalizes the matrix -diag(nodes) - lambda_ (w,w,...,w)^T.
    """
    w = lambda_*weights.copy()
    n = len(nodes)
    A = -mp.diag(nodes) - mp.matrix([[w[i] for i in range(n)] for _ in range(n)])
    return mp.eig(A)


def exp_matrix(A, dt):
    n = len(A[0])
    B = A[1] * mp.diag([mp.exp(A[0][i]*dt) for i in range(n)])
    return lu_solve_system(A[1].T, B.T).T


def ODE_drift(A, b, dt):
    c = mp.lu_solve(A[1], b)
    M = mp.diag([(mp.exp(x*dt)-1)/x if x**2*dt**3/6 > 2*mp.eps else dt*(1-x*dt/2) for x in A[0]])
    c = M*c
    return A[1]*c


def ODE_S_drift(A, b, dt, weights):
    M = mp.diag([((mp.exp(x*dt) - 1)/x - dt)/x if x**2 * dt**4 / 24 > 2*mp.eps else (dt**2/2 + x*dt**3/6) for x in A[0]])
    C = A[1].T * (weights/2).T
    D = (M*C).T
    c = mp.lu_solve(A[1], b)
    return (D*c)[0, 0]


def ODE_S_mult(A, dt, weights):
    M = mp.diag([mp.exp(x*dt)/x for x in A[0]])
    C = A[1].T * (weights/2).T
    D = M*C
    return mp.lu_solve(A[1].T, D)


def get_sample_path(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation", vol_behaviour='sticky'):
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
    rho_bar = np.sqrt(1-rho*rho)
    nodes, weights = rk.quadrature_rule_geometric_good(H, N, T, mode)
    N = len(nodes)
    nodes = mp.matrix([nodes[N-1]] + [nodes[i] for i in range(len(nodes)-1)])  # zero node in front
    weights = mp.matrix([weights[N-1]] + [weights[i] for i in range(len(weights)-1)])

    A = mp.eye(N) + mp.diag(nodes)*dt + lambda_*mp.matrix([[weights[i] for i in range(N)] for _ in range(N)])*dt
    A_inv = rk.mp_to_np(mp.inverse(A))
    nodes_mp = nodes
    weights_mp = weights
    nodes = rk.mp_to_np(nodes)
    weights = rk.mp_to_np(weights)

    S_values = np.zeros(N_time+1)
    V_values = np.zeros(N_time+1)
    S_values[0] = S_0
    V_values[0] = V_0
    V_components = np.zeros(shape=(N, N_time+1))
    V_components[0, 0] = V_0/weights[0]
    b_comp = nodes * V_components[:, 0] * dt + theta * dt

    def b(vol, dw):
        return b_comp + vol + nu * np.sqrt(np.fmax(np.dot(weights, vol), 0)) * dw

    if vol_behaviour == 'sticky':
        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            V_components[:, i] = A_inv @ b(V_components[:, i - 1], dW)
            V_values[i] = np.fmax(np.dot(weights, V_components[:, i]), 0)
            S_values[i] = S_values[i - 1] + np.sqrt(V_values[i - 1]) * S_values[i - 1] * (rho * dW + rho_bar * dB)
    elif vol_behaviour == 'mean reversion':
        zero_vol = V_0 <= 0
        for i in range(1, N_time+1):
            dW = np.random.normal()*sqrt_dt
            dB = np.random.normal()*sqrt_dt
            V_components[:, i] = A_inv@b(V_components[:, i-1], dW)
            V_values[i] = np.fmax(np.dot(weights, V_components[:, i]), 0)
            if zero_vol:
                while V_values[i] <= 0:
                    V_components[:, i] = A_inv@b(V_components[:, i], dW)
                    V_values[i] = np.fmax(np.dot(weights, V_components[:, i]), 0)
            zero_vol = V_values[i] <= 0
            S_values[i] = S_values[i-1] + np.sqrt(V_values[i-1])*S_values[i-1]*(rho*dW + rho_bar*dB)
    elif vol_behaviour == 'hyperplane reset':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            V_components[:, i] = A_inv @ b(V_components[:, i - 1], dW)
            V_values[i] = np.dot(weights, V_components[:, i])
            if V_values[i] < 0:
                V_components[:, i] = V_components[:, i] - V_values[i] * rescaled_weights
                V_values[i] = 0
            S_values[i] = S_values[i - 1] + np.sqrt(V_values[i - 1]) * S_values[i - 1] * (rho * dW + rho_bar * dB)
    elif vol_behaviour == 'hyperplane reflection':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            V_components[:, i] = A_inv @ b(V_components[:, i - 1], dW)
            V_values[i] = np.dot(weights, V_components[:, i])
            if V_values[i] < 0:
                V_components[:, i] = V_components[:, i] - 2 * V_values[i] * rescaled_weights
                V_values[i] = -V_values[i]
            S_values[i] = S_values[i - 1] + np.sqrt(V_values[i - 1]) * S_values[i - 1] * (rho * dW + rho_bar * dB)
    elif vol_behaviour == 'adaptive':

        def nested_step(V_comp_loc, dW_loc, dt_loc):
            dW_mid = dW_loc / 2 + np.random.normal() * np.sqrt(dt_loc / 4)
            dt_loc = dt_loc / 2

            b = nodes * V_components[:, 0] * dt_loc + theta * dt_loc + V_comp_loc + nu * np.sqrt(np.fmax(np.dot(weights, V_comp_loc), 0)) * dW_mid
            A = mp.eye(len(nodes)) + mp.diag(nodes_mp) * dt_loc + lambda_ * mp.matrix(
                [[weights_mp[i] for i in range(len(nodes))] for _ in range(len(nodes))]) * dt_loc
            V_comp_loc_updated = rk.mp_to_np(mp.lu_solve(A, b))
            if np.dot(weights, V_comp_loc_updated) < 0:
                V_comp_loc_updated = nested_step(V_comp_loc_updated, dW_mid, dt_loc)
            dW_second = dW_loc - dW_mid
            b = nodes * V_components[:, 0] * dt_loc + theta * dt_loc + V_comp_loc_updated + nu * np.sqrt(
                np.fmax(np.dot(weights, V_comp_loc_updated), 0)) * dW_second
            V_comp_loc_final = rk.mp_to_np(mp.lu_solve(A, b))
            if np.dot(weights, V_comp_loc_final) < 0:
                V_comp_loc_final = nested_step(V_comp_loc_final, dW_second, dt_loc)
            return V_comp_loc_final

        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            V_components[:, i] = A_inv @ b(V_components[:, i - 1], dW)
            V_values[i] = np.dot(weights, V_components[:, i])
            if V_values[i] < 0:
                V_components[:, i] = nested_step(V_components[:, i-1], dW, dt)
                V_values[i] = np.fmax(np.dot(weights, V_components[:, i]), 0)
            S_values[i] = S_values[i - 1] + np.sqrt(V_values[i - 1]) * S_values[i - 1] * (rho * dW + rho_bar * dB)
    elif vol_behaviour == 'multiple time scales':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        eta = 4.6  # e^(-4.6) approx 0.01, so only 1% of initial condition remaining relevant
        L = int(np.ceil(5*eta))  # number of time steps per component
        n_trivial_intervals = np.sum(eta/nodes[1:] >= dt)

        # find the local time increments
        dt_loc = np.empty(N - n_trivial_intervals)
        dt_past = 0  # how much time of the interval [t_m, t_{m+1}] has already passed
        for j in range(n_trivial_intervals, N):
            if j == N - 1:
                dt_loc[j-n_trivial_intervals] = (dt - dt_past)/L
                dt_past = dt
            else:
                dt_loc[j-n_trivial_intervals] = (dt - eta / nodes[j + 1] - dt_past)/L
                dt_past = dt - eta/nodes[j+1]

        # find the inverses of different As
        A_invs = []
        for j in range(n_trivial_intervals, N-1):
            A = mp.eye(N) \
                + mp.diag(mp.matrix([nodes_mp[l] for l in range(j+1)] + [mp.mpf(0)]*(N-j-1))) * dt_loc[j-n_trivial_intervals] \
                + lambda_ * mp.matrix([[weights_mp[i] * (l <= j) for i in range(N)] for l in range(N)]) * dt_loc[j-n_trivial_intervals]
            A_invs.append(rk.mp_to_np(mp.inverse(A)))
        A_invs.append(A_inv)

        # find the different bs
        def bs(j, vol, dw):
            return np.array((nodes * V_components[:, 0] * dt_loc[j-n_trivial_intervals] + theta * dt_loc[j-n_trivial_intervals] + nu * np.sqrt(np.fmax(np.dot(weights, vol), 0)) * dw)[:(j+1)].tolist() + [0]*(N-j-1)) + vol

        for i in range(1, N_time + 1):
            dW = 0  # total increment of the Brownian motion on that time interval [t_m, t_{m+1}]
            V_components[:, i] = V_components[:, i-1]
            for j in range(n_trivial_intervals, N):
                for _ in range(L):
                    dW_loc = np.random.normal() * np.sqrt(dt_loc[j-n_trivial_intervals])
                    dW += dW_loc
                    V_components[:, i] = A_invs[j-n_trivial_intervals] @ bs(j, V_components[:, i], dW_loc)
                    V_values[i] = np.dot(weights, V_components[:, i])
                    if V_values[i] < 0:
                        V_components[:, i] = V_components[:, i] - V_values[i] * rescaled_weights
                        V_values[i] = 0
            dB = np.random.normal() * sqrt_dt
            S_values[i] = S_values[i - 1] + np.sqrt(V_values[i - 1]) * S_values[i - 1] * (rho * dW + rho_bar * dB)

    elif vol_behaviour == 'split kernel':
        N = np.sum(nodes < 2/dt)
        fast_nodes = nodes[N:]
        fast_weights = weights[N:]
        nodes = nodes[:N]
        weights = weights[:N]
        V_components = V_components[:N, :]
        mu_a, mu_b, sr = sk.smooth_root(nodes=fast_nodes, weights=fast_weights, theta=theta, lambda_=lambda_, nu=nu,
                            us=np.linspace(-5, 5, 101), ps=np.exp(np.linspace(-10, 1, 100)), q=10, N_Riccati=10,
                            adaptive=True, M=1000000)

        A = mp.eye(N) + mp.diag(nodes) * dt + mp.mpf(lambda_ * (1 + mu_a)) * mp.matrix(
            [[weights[i] for i in range(N)] for _ in range(N)]) * dt
        A_inv = rk.mp_to_np(mp.inverse(A))

        b_comp = nodes * V_components[:, 0] * dt + (theta-lambda_*mu_b) * dt

        def b(vol, dw):
            return b_comp + vol + nu * sr(np.dot(weights, vol)) * dw

        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            V_components[:, i] = A_inv @ b(V_components[:, i - 1], dW)
            V_values[i] = np.dot(weights, V_components[:, i])
            if V_values[i] < 0:
                V_components[:, i] = V_components[:, i] - V_values[i] * rescaled_weights
                V_values[i] = 0
            S_values[i] = S_values[i - 1] + sr(V_values[i - 1]) * S_values[i - 1] * (rho * dW + rho_bar * dB)
        return S_values, np.array([sr(v) for v in V_values]), V_components

    return S_values, np.sqrt(V_values), V_components


def sample_simple(A_inv, nodes, weights, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, vol_behaviour='sticky', params=None):
    dt = T/N_time
    sqrt_dt = np.sqrt(dt)
    rho_bar = np.sqrt(1-rho**2)
    S = S_0
    V = V_0
    V_components = np.zeros(len(weights))
    V_components[0] = V_0 / weights[0]
    V_original = V_components.copy()

    b_comp = nodes * V_components * dt + theta * dt

    def b(vol, dw):
        return b_comp + vol + nu * np.sqrt(np.fmax(np.dot(weights, vol), 0)) * dw

    if vol_behaviour == 'sticky':
        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            S = S + np.sqrt(V) * S * (rho * dW + rho_bar * dB)
            V_components = A_inv @ b(V_components, dW)
            V = np.fmax(np.dot(weights, V_components), 0)
    elif vol_behaviour == 'mean reversion':
        zero_vol = V_0 <= 0
        for i in range(1, N_time+1):
            dW = np.random.normal()*sqrt_dt
            dB = np.random.normal()*sqrt_dt
            S = S + np.sqrt(V)*S*(rho*dW + rho_bar*dB)
            V_components = A_inv@b(V_components, dW)
            V = np.fmax(np.dot(weights, V_components), 0)
            if zero_vol:
                while V <= 0:
                    V_components = A_inv@b(V_components, dW)
                    V = np.fmax(np.dot(weights, V_components), 0)
            zero_vol = V <= 0
    elif vol_behaviour == 'hyperplane reset':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            S = S + np.sqrt(V) * S * (rho * dW + rho_bar * dB)
            V_components = A_inv @ b(V_components, dW)
            V = np.dot(weights, V_components)
            if V < 0:
                V_components = V_components - V * rescaled_weights
                V = 0
    elif vol_behaviour == 'hyperplane reflection':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            S = S + np.sqrt(V) * S * (rho * dW + rho_bar * dB)
            V_components = A_inv @ b(V_components, dW)
            V = np.dot(weights, V_components)
            if V < 0:
                V_components = V_components - 2 * V * rescaled_weights
                V = -V
    elif vol_behaviour == 'adaptive':

        def nested_step(V_comp_loc, dW_loc, dt_loc):
            dW_mid = dW_loc / 2 + np.random.normal() * np.sqrt(dt_loc / 4)
            dt_loc = dt_loc / 2

            b = nodes * V_original * dt_loc + theta * dt_loc + V_comp_loc + nu * np.sqrt(np.fmax(np.dot(weights, V_comp_loc), 0)) * dW_mid
            A = np.eye(len(nodes)) + np.diag(nodes) * dt_loc + lambda_ * np.array(
                [[weights[i] for i in range(len(nodes))] for _ in range(len(nodes))]) * dt_loc
            V_comp_loc_updated = np.linalg.solve(A, b)
            if np.dot(weights, V_comp_loc_updated) < 0:
                V_comp_loc_updated = nested_step(V_comp_loc_updated, dW_mid, dt_loc)
            dW_second = dW_loc - dW_mid
            b = nodes * V_original * dt_loc + theta * dt_loc + V_comp_loc_updated + nu * np.sqrt(
                np.fmax(np.dot(weights, V_comp_loc_updated), 0)) * dW_second
            V_comp_loc_final = np.linalg.solve(A, b)
            if np.dot(weights, V_comp_loc_final) < 0:
                V_comp_loc_final = nested_step(V_comp_loc_final, dW_second, dt_loc)
            return V_comp_loc_final

        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            S = S + np.sqrt(V) * S * (rho * dW + rho_bar * dB)
            V_components = A_inv @ b(V_components, dW)
            V = np.dot(weights, V_components)
            if V < 0:
                V_components = nested_step(V_components, dW, dt)
                V = np.fmax(np.dot(weights, V_components), 0)

    elif vol_behaviour == 'multiple time scales':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        eta = params['eta']
        L = params['L']
        n_trivial_intervals = params['n_trivial_intervals']
        dt_loc = params['dt_loc']

        # find the different bs
        def bs(j, vol, dw):
            return np.array((nodes * V_original * dt_loc[j-n_trivial_intervals] + theta * dt_loc[j-n_trivial_intervals] + nu * np.sqrt(np.fmax(np.dot(weights, vol), 0)) * dw)[:(j+1)].tolist() + [0]*(N-j-1)) + vol

        for i in range(1, N_time + 1):
            V_old = V
            dW = 0  # total increment of the Brownian motion on that time interval [t_m, t_{m+1}]
            for j in range(n_trivial_intervals, N):
                for _ in range(L):
                    dW_loc = np.random.normal() * np.sqrt(dt_loc[j-n_trivial_intervals])
                    dW += dW_loc
                    V_components = A_inv[j-n_trivial_intervals] @ bs(j, V_components, dW_loc)
                    V = np.dot(weights, V_components)
                    if V < 0:
                        V_components = V_components - V * rescaled_weights
                        V = 0
            dB = np.random.normal() * sqrt_dt
            S = S + np.sqrt(V_old) * S * (rho * dW + rho_bar * dB)

    elif vol_behaviour == 'split kernel':
        mu_b = params['mu_b']
        sr = params['sr']
        b_comp = nodes * V_components * dt + (theta-lambda_*mu_b) * dt

        def b(vol, dw):
            return b_comp + vol + nu * sr(np.dot(weights, vol)) * dw

        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for i in range(1, N_time + 1):
            dW = np.random.normal() * sqrt_dt
            dB = np.random.normal() * sqrt_dt
            S = S + sr(V) * S * (rho * dW + rho_bar * dB)
            V_components = A_inv @ b(V_components, dW)
            V = np.dot(weights, V_components)
            if V < 0:
                V_components = V_components - V * rescaled_weights
                V = 0

    return S


def samples(H, lambda_, rho, nu, theta, V_0, T, N, m=1000, S_0=1., N_time=1000, mode="observation", vol_behaviour='sticky'):
    dt = T / N_time
    nodes, weights = rk.quadrature_rule_geometric_good(H, N, T, mode)
    N = len(nodes)
    nodes_mp = mp.matrix([nodes[N-1]] + [nodes[i] for i in range(len(nodes)-1)])  # zero node in front
    weights_mp = mp.matrix([weights[N-1]] + [weights[i] for i in range(len(weights)-1)])

    A = mp.eye(N) + mp.diag(nodes) * dt + lambda_ * mp.matrix([[weights[i] for i in range(N)] for _ in range(N)]) * dt
    A_inv = rk.mp_to_np(mp.inverse(A))
    nodes = rk.mp_to_np(nodes_mp)
    weights = rk.mp_to_np(weights_mp)
    params = None

    if vol_behaviour == 'multiple time scales':
        eta = 4.6  # e^(-4.6) approx 0.01, so only 1% of initial condition remaining relevant
        L = int(np.ceil(5*eta))  # number of time steps per component
        n_trivial_intervals = np.sum(eta/nodes[1:] >= dt)

        # find the local time increments
        dt_loc = np.empty(N - n_trivial_intervals)
        dt_past = 0  # how much time of the interval [t_m, t_{m+1}] has already passed
        for j in range(n_trivial_intervals, N):
            if j == N - 1:
                dt_loc[j - n_trivial_intervals] = (dt - dt_past) / L
                dt_past = dt
            else:
                dt_loc[j - n_trivial_intervals] = (dt - eta / nodes[j + 1] - dt_past) / L
                dt_past = dt - eta / nodes[j + 1]

        # find the inverses of different As
        A_invs = []
        for j in range(n_trivial_intervals, N - 1):
            A = mp.eye(N) \
                + mp.diag(mp.matrix([nodes_mp[l] for l in range(j + 1)] + [mp.mpf(0)] * (N - j - 1))) * dt_loc[
                    j - n_trivial_intervals] \
                + lambda_ * mp.matrix([[weights_mp[i] * (l <= j) for i in range(N)] for l in range(N)]) * dt_loc[
                    j - n_trivial_intervals]
            A_invs.append(rk.mp_to_np(mp.inverse(A)))
        A_invs.append(A_inv)
        A_inv = A_invs.copy()

        params = {'eta': eta, 'L': L, 'n_trivial_intervals': n_trivial_intervals, 'dt_loc': dt_loc}

    elif vol_behaviour == 'split kernel':
        print(nodes)
        N = np.sum(nodes < 2/dt)
        fast_nodes = nodes[N:]
        fast_weights = weights[N:]
        nodes = nodes[:N]
        weights = weights[:N]
        print(fast_nodes)
        print(nodes)
        mu_a, mu_b, sr = sk.smooth_root(nodes=fast_nodes, weights=fast_weights, theta=theta, lambda_=lambda_, nu=nu,
                            us=np.linspace(-5, 5, 101), ps=np.exp(np.linspace(-10, 1, 100)), q=10, N_Riccati=10,
                            adaptive=True, M=1000000)

        # A = mp.eye(N) + mp.diag(nodes) * dt + mp.mpf(lambda_ * (1 + mu_a)) * mp.matrix(
        #    [[weights[i] for i in range(N)] for _ in range(N)]) * dt
        A = mp.eye(N) + mp.diag(nodes) * dt + mp.mpf(lambda_) * mp.matrix(
            [[weights[i] for i in range(N)] for _ in range(N)]) * dt
        A_inv = rk.mp_to_np(mp.inverse(A))
        # params = {'mu_a': mu_a, 'mu_b': mu_b, 'sr': sr}
        params = {'mu_a': 0, 'mu_b': 0, 'sr': lambda x: np.sqrt(np.abs(x))}

    sample_vec = np.zeros(m)
    for i in range(m):
        if i % 100 == 0:
            print(f'{i} of {m} generated')
        sample_vec[i] = sample_simple(A_inv, nodes, weights, lambda_, rho, nu, theta, V_0, T, N, S_0, N_time, vol_behaviour, params)
    return sample_vec


def call(K, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, H=0.1, N=6, S_0=1., T=1., m=1000, N_time=1000, mode='observation', vol_behaviour='sticky'):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    Uses the Ninomiya-Victoir scheme.
    :param K: Strike prices, assumed to be a vector
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps used in the solution of the fractional Riccati equation
    :param nodes: The nodes used in the approximation
    :param weights: The weights used in the approximation
    return: The prices of the call option for the various strike prices in K
    """
    S = samples(H, lambda_, rho, nu, theta, V_0, T, N, m, S_0, N_time, mode, vol_behaviour)
    return cf.volatility_smile_call(S, K, T, S_0)
