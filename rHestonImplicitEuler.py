import time
from matplotlib import pyplot as plt
import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp
from scipy.optimize import fsolve

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


def get_sample_path(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation", bounce_vol=True):
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
    rule = rk.quadrature_rule_geometric_good(H, N, T, mode)
    nodes = rule[0, :]
    weights = rule[1, :]
    N = len(nodes)

    A = mp.eye(N) + mp.diag(nodes)*dt + lambda_*mp.matrix([[weights[i] for i in range(N)] for _ in range(N)])*dt
    A_inv = mp.inverse(A)
    A_inv = np.array([[float(A_inv[i, j]) for j in range(N)] for i in range(N)])
    nodes = np.array([float(node) for node in nodes])
    weights = np.array([float(weight) for weight in weights])

    S_values = np.zeros(N_time+1)
    V_values = np.zeros(N_time+1)
    S_values[0] = float(S_0)
    V_values[0] = V_0
    V_components = np.zeros(shape=(N, N_time+1))
    V_components[:, 0] = V_0/(N*weights)

    b_comp = nodes * V_components[:, 0] * dt + theta * dt

    def b(vol, dw):
        return b_comp + vol + nu * np.sqrt(np.fmax(np.dot(weights, vol), 0)) * dw

    zero_vol = V_0 <= 0

    for i in range(1, N_time+1):
        dW = np.random.normal()*sqrt_dt
        dB = np.random.normal()*sqrt_dt
        V_components[:, i] = A_inv@b(V_components[:, i-1], dW)
        V_values[i] = np.fmax(np.dot(weights, V_components[:, i]), 0)
        if bounce_vol and zero_vol:
            while V_values[i] <= 0:
                V_components[:, i] = A_inv@b(V_components[:, i], dW)
                V_values[i] = np.fmax(np.dot(weights, V_components[:, i]), 0)
        zero_vol = V_values[i] <= 0
        S_values[i] = S_values[i-1] + np.sqrt(V_values[i-1])*S_values[i-1]*(rho*dW + rho_bar*dB)

    return S_values, np.sqrt(V_values), V_components


def get_sample_path_(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation"):
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
    rule = rk.quadrature_rule_geometric_good(H, N, T, mode)
    nodes = rule[0, :]
    weights = rule[1, :]
    nodes = np.array([float(node) for node in nodes])
    weights = np.array([float(weight) for weight in weights])
    N = len(nodes)
    weight_sum = np.sum(weights)
    print(f'nodes = {nodes}')
    print(f'weights = {weights}')
    print(f'V_0 = {V_0}')
    # time.sleep(5)

    S_values = np.zeros(N_time+1)
    V_values = np.zeros(N_time+1)
    S_values[0] = S_0
    V_values[0] = V_0
    V_components = np.zeros(shape=(N, N_time+1))
    V_components[:, 0] = V_0/(N*weights)

    for i in range(1, N_time+1):
        if i == 100:
            #  time.sleep(3600)
            print(i)
        dW = np.random.normal()*sqrt_dt
        dB = np.random.normal()*sqrt_dt

        def V_eq(p):
            print(p)
            total_vol = np.dot(weights, p)
            if total_vol < 0:
                print(f'Total volatility is negative: {total_vol}')
                return np.ones(N)
            else:
                print(f'Total volatility is non-negative: {total_vol}')
            # x = p - V_components[:, i-1] + (nodes*(p-V_components[:, 0]) - (theta-lambda_*total_vol) + nu**2*weight_sum/2)*dt + nu*np.sqrt(total_vol)*dW
            x = p - V_components[:, i - 1] + (nodes * (p - V_components[:, 0]) - (
                        theta - lambda_ * total_vol)) * dt + nu * np.sqrt(np.dot(weights, V_components[:, i-1])) * dW

            print(x)
            return x

        def V_eq_prime(p):
            total_vol_sqrt = np.sqrt(np.fmax(np.dot(weights, p), 0))
            temp = lambda_*weights*dt # - nu*weights*dW/(2*total_vol_sqrt)
            return np.repeat(temp[..., None], N, axis=1) + np.diag(1 + nodes*dt)

        V_components[:, i] = fsolve(func=V_eq, x0=V_components[:, i-1], fprime=V_eq_prime, col_deriv=True)
        S_values[i] = S_values[i-1] + np.sqrt(V_values[i-1])*S_values[i-1]*(rho*dW + rho_bar*dB)
        V_values[i] = np.dot(weights, V_components[:, i])

    return S_values, np.sqrt(V_values), V_components


def sample_simple(A_inv, nodes, weights, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, bounce_vol=True):
    dt = T/N_time
    sqrt_dt = np.sqrt(dt)
    rho_bar = np.sqrt(1-rho**2)
    S = S_0
    V = V_0
    V_components = V_0 / (N * weights)

    b_comp = nodes * V_components * dt + theta * dt

    def b(vol, dw):
        return b_comp + vol + nu * np.sqrt(np.fmax(np.dot(weights, vol), 0)) * dw

    zero_vol = V <= 0

    for i in range(1, N_time + 1):
        dW = np.random.normal() * sqrt_dt
        dB = np.random.normal() * sqrt_dt
        S = S + np.sqrt(V) * S * (rho * dW + rho_bar * dB)
        V_components = A_inv @ b(V_components, dW)
        V = np.fmax(np.dot(weights, V_components), 0)
        if bounce_vol and zero_vol:
            while V <= 0:
                V_components = A_inv @ b(V_components, dW)
                V = np.fmax(np.dot(weights, V_components), 0)
        zero_vol = V <= 0
    return S


def samples(H, lambda_, rho, nu, theta, V_0, T, N, m=1000, S_0=1., N_time=1000, mode="observation", bounce_vol=True):
    dt = T / N_time
    rule = rk.quadrature_rule_geometric_good(H, N, T, mode)
    nodes = rule[0, :]
    weights = rule[1, :]
    N = len(nodes)

    A = mp.eye(N) + mp.diag(nodes) * dt + lambda_ * mp.matrix([[weights[i] for i in range(N)] for _ in range(N)]) * dt
    A_inv = mp.inverse(A)
    A_inv = np.array([[float(A_inv[i, j]) for j in range(N)] for i in range(N)])
    nodes = np.array([float(node) for node in nodes])
    weights = np.array([float(weight) for weight in weights])

    sample_vec = np.zeros(m)
    for i in range(m):
        if i % 100 == 0:
            print(f'{i} of {m} generated')
        sample_vec[i] = sample_simple(A_inv, nodes, weights, rho, nu, theta, V_0, T, N, S_0, N_time, bounce_vol)
    return sample_vec


def call(K, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, H=0.1, N=6, S_0=1., T=1., m=1000, N_time=1000, mode='observation', bounce_vol=True):
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
    S = samples(H, lambda_, rho, nu, theta, V_0, T, N, m, S_0, N_time, mode, bounce_vol)
    return cf.volatility_smile_call(S, K, T, S_0)


def implied_volatility(K, H, lambda_, rho, nu, theta, V_0, T, N, N_time=1000, mode="observation"):
    """
    Gives the implied volatility of the European call option in the rough Heston model. Uses a Markovian approximation,
    and the Ninomiya-Victoir scheme.
    :param K: Strike price, assumed to be a vector
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps used in the solution of the fractional Riccati equation
    :param N: Total number of points in the quadrature rule, N=n*m
    :param mode: If observation, use the values of the interpolation of the optimum. If theorem, use the values of the
    theorem.
    return: The price of the call option
    """
    rule = rk.quadrature_rule_geometric_good(H, N, T, mode)
    nodes = rule[0, :]
    weights = rule[1, :]
    prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N_time=N_time,
                  nodes=nodes, weights=weights)
    return cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)