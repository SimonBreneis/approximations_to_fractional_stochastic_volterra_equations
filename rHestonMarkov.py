import time

import numpy as np
import ComputationalFinance as cf
import Data
import RoughKernel as rk
import matplotlib.pyplot as plt


def solve_Riccati(z, lambda_, rho, nu, nodes, weights, dt):
    """
    Solves the Riccati equation in the exponent of the characteristic function of the approximation of the rough Heston
    model. Does not return psi but rather F(z, psi(., z)).
    :param z: Argument of the moment generating function
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param nodes: The nodes used in the approximation, assuming that they are ordered in increasing order
    :param weights: The weights used in the approximation
    :param dt: Array of time step sizes
    :return: An approximation of F applied to the solution of the Riccati equation. Is an array with N+1 values.
    """

    def F(x):
        return (z*z-z)/2 + (rho*nu*z-lambda_)*x + nu*nu*x*x/2

    N = len(nodes)
    psi_x = np.zeros(N, dtype=np.cdouble)
    F_psi = np.zeros(len(dt)+1, dtype=np.cdouble)
    F_psi[0] = F(np.cdouble(0))
    psi = np.cdouble(0)

    for i in range(len(dt)):
        exp_nodes = np.exp(-nodes * dt[i])
        div_nodes = np.zeros(len(nodes))
        div_nodes[0] = dt[i] if nodes[0] == 0 else (1 - np.exp(-nodes[0]*dt[i])) / nodes[0]
        div_nodes[1:] = (1 - np.exp(-nodes[1:]*dt[i])) / nodes[1:]
        psi_xP = F_psi[i]*div_nodes + psi_x*exp_nodes
        psi_P = (psi + np.dot(weights, psi_xP))/2
        psi_x = F(psi_P)*div_nodes + psi_x*exp_nodes
        psi = np.dot(weights, psi_x)
        F_psi[i+1] = F(psi)
    return F_psi


def call(K, lambda_, rho, nu, theta, V_0, nodes, weights, dt, R=2., L=200., N_fourier=40000):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    Uses Fourier inversion.
    :param K: Strike prices, assumed to be a vector
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param dt: Array of time step sizes
    :param R: The (dampening) shift that we use for the Fourier inversion
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals,
    but only over [-L, L]
    :param N_fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    :param nodes: The nodes used in the approximation
    :param weights: The weights used in the approximation
    return: The prices of the call option for the various strike prices in K
    """
    N = len(nodes)
    times = np.zeros(len(dt)+1)
    times[1:] = np.cumsum(dt)
    g = np.zeros(len(dt) + 1)
    for i in range(1, N):
        g += weights[i] / nodes[i] * (1 - np.exp(-nodes[i] * times))
    g = theta * (g + weights[0] * times) + V_0

    def mgf_(z):
        """
        Moment generating function of the log-price.
        :param z: Argument, assumed to be a vector
        :return: Value
        """
        res = np.zeros(shape=(len(z)), dtype=complex)
        for i in range(len(z)):
            # print(f'{i} of {len(z)}')
            Fz = solve_Riccati(z=z[i], lambda_=lambda_, rho=rho, nu=nu, nodes=nodes, weights=weights, dt=dt)
            res[i] = np.trapz(Fz * g, dx=dt)
        return np.exp(res)

    return cf.pricing_fourier_inversion(mgf_, K, R, L, N_fourier)

'''
N_time = 1048576
nodes, weights = rk.quadrature_rule_geometric_standard(0.1, 6, 1.)
adaptive = False
N = len(nodes)
T = 1.
q = 10
if adaptive:
    dt, times = rk.adaptive_time_steps(nodes, T=1., q=q, N_time=N_time)
else:
    dt = np.ones(N_time) * (T/N_time)
    times = np.zeros(len(dt)+1)
    times[1:] = np.cumsum(dt)
# F_psi, psi = solve_Riccati(2j, 0.3, -0.7, 0.3, nodes, weights, dt=dt)
true_int = call(1, 0.3, -0.7, 0.3, 0.02, 0.02, nodes, weights, dt)[0]
adaptive = False
print(true_int)
N_times = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
ints = np.empty(len(N_times), dtype=np.complex_)
for i in range(len(N_times)):
    print(i)
    if adaptive:
        dt, times = rk.adaptive_time_steps(nodes, T=1., q=q, N_time=N_times[i])
    else:
        dt = np.ones(N_times[i]) * (T / N_times[i])
        times = np.zeros(len(dt) + 1)
        times[1:] = np.cumsum(dt)
    ints[i] = call(1, 0.3, -0.7, 0.3, 0.02, 0.02, nodes, weights, dt)[0]
errors = np.abs(true_int - ints)
N_times = N_times
a, b, _, _, _ = Data.log_linear_regression(N_times, errors)
print(a, b)
plt.loglog(N_times, errors)
plt.loglog(N_times, b*N_times**a, 'k--')
plt.ylabel('Error')
plt.xlabel('Number of time steps')
plt.show()
time.sleep(360000)
'''

def implied_volatility(K, H, lambda_, rho, nu, theta, V_0, T, N, N_Riccati=200, R=-2, L=30., N_fourier=6000, q=10,
                       adaptive=False, mode="observation"):
    """
    Gives the implied volatility of the European call option in the rough Heston model
    as described in El Euch and Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme.
    Uses Fourier inversion.
    :param K: Strike price, assumed to be a vector
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param R: The (dampening) shift that we use for the Fourier inversion
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals,
    but only over [-L, L]
    :param N_fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    :param q: Solves the Riccati equation up to time q/min(nodes)
    :param N_Riccati: Uses q*N_Riccati time steps adapted to every node if adaptive, N_Riccati*q equidistant time steps
        if not adaptive
    :param adaptive: If true, adapts the time steps to the mean-reversion parameters (nodes). If false, uses equidistant
        time steps
    :param N: Total number of points in the quadrature rule, N=n*m
    :param mode: If observation, use the values of the interpolation of the optimum. If theorem, use the values of the
    theorem.
    return: The price of the call option
    """
    nodes, weights = rk.quadrature_rule_geometric_standard(H, N, T, mode)
    if adaptive:
        dt, times = rk.adaptive_time_steps(nodes=nodes, T=T, q=q, N_time=N_Riccati)
    else:
        dt = np.ones(N_Riccati) * (T/N_Riccati)
        times = np.zeros(len(dt)+1)
        times[1:] = np.cumsum(dt)
    prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, dt=dt, R=R, L=L, N_fourier=N_fourier,
                  nodes=nodes, weights=weights)
    return cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)

Rs = np.linspace(-5, 5, 101)
vols = np.zeros(len(Rs))
for i in range(len(Rs)):
    print(i)
    vols[i] = implied_volatility(np.array([1.]), 0.1, 0.3, -0.7, 0.3, 0.02, 0.02, 1., 6, R=Rs[i])
plt.plot(Rs, vols)
plt.xlabel('R')
plt.ylabel('Implied volatility')
plt.show()