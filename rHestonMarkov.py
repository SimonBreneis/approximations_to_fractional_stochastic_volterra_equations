import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp


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
