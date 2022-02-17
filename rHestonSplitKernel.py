import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def solve_Riccati_high_mean_reversion(u, nodes, weights, lambda_, nu, N_Riccati=10, T=0.1, adaptive=True):
    N = len(nodes)
    if not adaptive:
        dt = T/N_Riccati*np.ones(N_Riccati)
    else:
        dt = []
        timescale = 0
        for i in range(N):
            prev_timescale = timescale
            timescale = np.fmin(10/nodes[-i-1], T)
            if timescale > prev_timescale:
                dt = dt + [(timescale - prev_timescale) / (10 * N_Riccati)] * (10 * N_Riccati)
        dt = np.array(dt)
    print(len(dt))

    psis = np.zeros((2*N, len(dt)+1))
    psi = np.zeros((2, len(dt)+1))
    u_r = u.real
    u_i = u.imag

    time_elapsed = 0.
    for i in range(len(dt)):

        def eq(p):
            p_r = p[:N]
            p_i = p[N:]
            psi[0, i+1] = np.dot(weights, p_r)
            psi[1, i+1] = np.dot(weights, p_i)
            result = p - psis[:, i]
            result[:N] += - u_r*(np.exp(-nodes*time_elapsed) - np.exp(-nodes*(time_elapsed+dt[i]))) + nodes*p_r*dt[i] + lambda_*psi[0, i+1]*dt[i] - nu**2*(psi[0, i+1]**2 - psi[1, i+1]**2)*dt[i]/2
            result[N:] += - u_i*(np.exp(-nodes*time_elapsed) - np.exp(-nodes*(time_elapsed+dt[i]))) + nodes*p_i*dt[i] + lambda_*psi[1, i+1]*dt[i] - nu**2*psi[0, i+1]*psi[1, i+1]*dt[i]
            return result

        def der(p):
            p_r = p[:N]
            p_i = p[N:]
            psi[0, i + 1] = np.dot(weights, p_r)
            psi[1, i + 1] = np.dot(weights, p_i)
            result = np.eye(2*N)
            result[:N, :N] += np.diag(nodes*dt[i])
            result[N:, N:] += np.diag(nodes*dt[i])
            temp = weights*dt[i]
            temp = np.repeat(temp[..., None], N, axis=1)
            result[:N, :N] += temp*lambda_
            result[N:, N:] += temp*lambda_
            result[:N, :N] -= nu**2*psi[0, i+1]*temp
            result[N:, :N] += nu**2*psi[1, i+1]*temp
            result[N:, N:] -= 2*nu**2*psi[0, i+1]*temp
            result[:N, N:] -= 2*nu**2*psi[1, i+1]*temp
            return result

        psis[:, i+1] = fsolve(func=eq, x0=psis[:, i], fprime=der, col_deriv=True)
        time_elapsed += dt[i]

    times = np.zeros(len(dt)+1)
    times[1:] = np.cumsum(dt)
    return times, psi


def psi_integrals(t, psi):
    real_1 = np.trapz(psi[0, :], x=t)
    imag_1 = np.trapz(psi[1, :], x=t)
    real_2 = np.trapz(psi[0, :]**2 - psi[1, :]**2, x=t)
    imag_2 = np.trapz(psi[0, :]*psi[1, :], x=t)
    return real_1 + imag_1*1j, real_2 + imag_2*1j


def multiple_psi_integrals(us, nodes, weights, lambda_, nu, T=0.1, N_Riccati=10, adaptive=True):
    integrals = np.empty((2, len(us)), dtype=np.complex_)
    for i in range(len(us)):
        print(f'{i} of {len(us)}')
        ints = psi_integrals(*solve_Riccati_high_mean_reversion(us[i], nodes, weights, lambda_, nu, N_Riccati, T, adaptive))
        integrals[0, i] = ints[0]
        integrals[1, i] = ints[1]
    return integrals


def chararacteristic_function_high_mean_reversion(us, p, nodes, weights, lambda_, nu, theta, T=0.1, N_Riccati=10, adaptive=True):
    ints = multiple_psi_integrals(us, nodes, weights, lambda_, nu, T, N_Riccati, adaptive)
    return ints[0, :]*(theta-lambda_*p) + ints[1, :]*nu**2*p/2


def regress_for_varying_p(us=np.linspace(-5, 5, 101), ps=np.array([0., 0.01, 0.1, 1.]), H=0.1, N=6, lambda_=0.3, nu=0.3, theta=0.02, T=1, N_Riccati=10, reversion_cutoff=100, adaptive=True):
    nodes, weights = rk.quadrature_rule_geometric_good(H, N)
    nodes = rk.mp_to_np(nodes)
    weights = rk.mp_to_np(weights)
    nodes = nodes[:-1]
    weights = weights[:-1]
    i = np.sum(nodes < reversion_cutoff)
    nodes = nodes[i:]
    weights = weights[i:]
    T = 10/np.amin(nodes)

    ints = multiple_psi_integrals(us*1j, nodes, weights, lambda_, nu, T, N_Riccati, adaptive)

    for i in range(len(ps)):
        cf = ints[0, :]*(theta-lambda_*ps[i]) + ints[1, :]*nu**2*ps[i]/2
        cfr = cf.real
        cfi = cf.imag
        mu_ests = np.sum(us*cfi)/np.sum(us**2)
        sigma_ests = np.sum(us**2 * cfr)/np.sum(us**4)
        plt.plot(us, cfi, 'b-', label='imaginary part')
        plt.plot(us, mu_ests*us, 'b--')
        plt.plot(us, cfr, 'r-', label='real part')
        plt.plot(us, sigma_ests*us**2, 'r--')
        plt.legend(loc='best')
        plt.xlabel(r'$u$')
        plt.title(f'Logarithm of characteristic function, p={ps[i]}')
        plt.show()
    return







def solve_Riccati(z, lambda_, rho, nu, exp_nodes, div_nodes, weights, N_Riccati=1000):
    """
    Solves the Riccati equation in the exponent of the characteristic function of the approximation of the rough Heston
    model. Does not return psi but rather F(z, psi(., z)).
    :param z: Argument of the moment generating function
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param N_Riccati: Number of time steps used
    :param exp_nodes: exp(-nodes*dt) where nodes are the nodes used in the approximation
    :param div_nodes: (1-exp_nodes)/nodes
    :param weights: The weights used in the approximation
    :return: An approximation of F applied to the solution of the Riccati equation. Is an array with N+1 values.
    """

    def F(x):
        return (z*z-z)/2 + (rho*nu*z-lambda_)*x + nu*nu*x*x/2

    psi_x = np.zeros(len(weights), dtype=np.clongdouble)
    F_psi = np.zeros(N_Riccati+1, dtype=np.clongdouble)
    F_psi[0] = F(np.clongdouble(0))
    psi = np.clongdouble(0)

    for i in range(N_Riccati):
        psi_xP = F_psi[i]*div_nodes + psi_x*exp_nodes
        psi_P = (psi + np.dot(weights, psi_xP))/2
        psi_x = F(psi_P)*div_nodes + psi_x*exp_nodes
        psi = np.dot(weights, psi_x)
        F_psi[i+1] = F(psi)
    F_psi = np.array([complex(F_psi[i]) for i in range(len(F_psi))])
    return F_psi


def call(K, lambda_, rho, nu, theta, V_0, nodes, weights, T, N_Riccati=1000, R=2., L=200., N_fourier=40000):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    Uses Fourier inversion.
    :param K: Strike prices, assumed to be a vector
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_Riccati: Number of time steps used in the solution of the fractional Riccati equation
    :param R: The (dampening) shift that we use for the Fourier inversion
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals,
    but only over [-L, L]
    :param N_fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    :param nodes: The nodes used in the approximation
    :param weights: The weights used in the approximation
    return: The prices of the call option for the various strike prices in K
    """
    dt = T/N_Riccati
    times = dt * mp.matrix([N_Riccati - i for i in range(N_Riccati + 1)])
    g = np.zeros(N_Riccati + 1)
    for i in range(len(nodes) - 1):
        for t in range(len(times)):
            g[t] += weights[i] / nodes[i] * (1 - mp.exp(-nodes[i] * times[t]))
    g = theta * (g + weights[len(weights)-1] * times) + V_0

    exp_nodes = mp.matrix([mp.exp(-nodes[j] * dt) for j in range(len(nodes))])
    div_nodes = np.zeros(len(nodes))
    div_nodes[:-1] = np.array([np.longdouble((1 - exp_nodes[j]) / nodes[j]) for j in range(len(nodes) - 1)])
    div_nodes[-1] = np.longdouble(dt)
    exp_nodes = np.array([np.longdouble(x) for x in exp_nodes])
    weights = np.array([np.longdouble(x) for x in weights])

    def mgf_(z):
        """
        Moment generating function of the log-price.
        :param z: Argument, assumed to be a vector
        :return: Value
        """
        res = np.zeros(shape=(len(z)), dtype=complex)
        for i in range(len(z)):
            Fz = solve_Riccati(z=z[i], lambda_=lambda_, rho=rho, nu=nu, N_Riccati=N_Riccati, exp_nodes=exp_nodes,
                               div_nodes=div_nodes, weights=weights)
            res[i] = np.trapz(Fz * g, dx=dt)
        return np.exp(res)

    return cf.pricing_fourier_inversion(mgf_, K, R, L, N_fourier)


def implied_volatility(K, H, lambda_, rho, nu, theta, V_0, T, N, N_Riccati=1000, R=2., L=200., N_fourier=40000,
                       mode="observation"):
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
    :param N_Riccati: Number of time steps used in the solution of the fractional Riccati equation
    :param R: The (dampening) shift that we use for the Fourier inversion
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals,
    but only over [-L, L]
    :param N_fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    :param N: Total number of points in the quadrature rule, N=n*m
    :param mode: If observation, use the values of the interpolation of the optimum. If theorem, use the values of the
    theorem.
    return: The price of the call option
    """
    nodes, weights = rk.quadrature_rule_geometric_good(H, N, T, mode)
    prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N_Riccati=N_Riccati, R=R, L=L,
                  N_fourier=N_fourier, nodes=nodes, weights=weights)
    return cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
