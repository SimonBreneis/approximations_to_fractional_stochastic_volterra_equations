import time
import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk


def solve_Riccati(z, lambda_, rho, nu, nodes, weights, exp_nodes, div_nodes, N_Riccati=200):
    """
    Solves the Riccati equation in the exponent of the characteristic function of the approximation of the rough Heston
    model. Does not return psi but rather F(z, psi(., z)).
    :param z: Argument of the moment generating function
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param nodes: The nodes used in the approximation, assuming that they are ordered in increasing order
    :param weights: The weights used in the approximation
    :param exp_nodes: np.exp(-nodes * (T/N_Riccati))
    :param div_nodes: (1 - exp_nodes) / nodes
    :param N_Riccati: Number of time steps for solving the Riccati equation
    :return: An approximation of F applied to the solution of the Riccati equation. Is an array with N+1 values.
    """

    a = nu*nu/2
    b = rho*nu*z - lambda_
    c = (z*z-z)/2

    def F(x):
        return a*x*x + b*x + c

    psi_x = np.zeros((len(z), len(nodes)), dtype=np.cdouble)
    F_psi = np.zeros((len(z), N_Riccati+1), dtype=np.cdouble)
    F_psi[:, 0] = c
    psi = np.zeros(len(z), dtype=np.cdouble)

    new_div = np.dot(div_nodes, weights)
    new_exp = exp_nodes * weights

    for i in range(N_Riccati):
        psi_P = (psi + F_psi[:, i] * new_div + psi_x @ new_exp)/2
        psi_x = np.outer(F(psi_P), div_nodes) + psi_x*exp_nodes[None, :]
        psi = psi_x @ weights
        F_psi[:, i+1] = F(psi)
    return F_psi


def call(K, lambda_, rho, nu, theta, V_0, nodes, weights, T=1, N_Riccati=200, R=2., L=50., N_Fourier=300):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    Uses Fourier inversion.
    :param K: Strike prices, assumed to be a vector
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps for solving the Riccati equation
    :param R: The (dampening) shift that we use for the Fourier inversion
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals, but only over
        [0, L]
    :param N_Fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    :param nodes: The nodes used in the approximation
    :param weights: The weights used in the approximation
    return: The prices of the call option for the various strike prices in K
    """
    N = len(nodes)
    times = np.linspace(T, 0, N_Riccati+1)
    g = np.zeros(N_Riccati + 1)
    for i in range(1, N):
        g += weights[i] / nodes[i] * (1 - np.exp(-np.fmin(nodes[i] * times, 100)))
    if nodes[0] == 0:
        g = theta * (g + weights[0] * times) + V_0
    else:
        g = theta * (g + weights[0] / nodes[0] * (1 - np.exp(-nodes[0] * times))) + V_0

    exp_nodes = np.exp(-np.fmin(nodes * (T/N_Riccati), np.fmax(100, np.log(weights)*30)))
    div_nodes = np.zeros(len(nodes))
    div_nodes[0] = T/N_Riccati if nodes[0] == 0 else (1 - exp_nodes[0]) / nodes[0]
    div_nodes[1:] = (1 - exp_nodes[1:]) / nodes[1:]

    def mgf_(z):
        """
        Moment generating function of the log-price.
        :param z: Argument, assumed to be a vector
        :return: Value
        """
        Fz = solve_Riccati(z=z, lambda_=lambda_, rho=rho, nu=nu, nodes=nodes, weights=weights, N_Riccati=N_Riccati,
                           exp_nodes=exp_nodes, div_nodes=div_nodes)
        res = np.trapz(Fz * g, dx=T/N_Riccati)
        return np.exp(res)

    return cf.pricing_fourier_inversion(mgf_, K, R, L, N_Fourier)


def implied_volatility(K, H, lambda_, rho, nu, theta, V_0, T, N, N_Riccati=None, R=2, L=None, N_Fourier=None,
                       mode="best", rel_tol=1e-02, smoothing=False, nodes=None, weights=None):
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
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals, but only over
        [0, L]
    :param N_Fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    :param N_Riccati: Uses N_Riccati time steps to solve the Riccati equations
    :param N: Total number of points in the quadrature rule
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param rel_tol: Required maximal relative error in the implied volatility
    return: The price of the call option
    """
    if N_Riccati is None:
        N_Riccati = int(200 / T**0.8)
    if L is None:
        L = 50/T
    if N_Fourier is None:
        N_Fourier = int(8*L / np.sqrt(T))
    result = None
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    else:
        N = len(nodes)
    np.seterr(all='warn')
    K_result = K
    if smoothing:
        K_fine = np.empty((len(K)+1)*10)
        K_fine[:11] = np.exp(np.linspace(2*np.log(K[0]) - np.log(K[1]), np.log(K[0]), 11))
        for i in range(len(K)-1):
            K_fine[10*i:10*(i+1)+1] = np.exp(np.linspace(np.log(K[i]), np.log(K[i+1]), 11))
        K_fine[-11:] = np.exp(np.linspace(np.log(K[-1]), 2*np.log(K[-1]) - np.log(K[-2]), 11))
        K = K_fine

    tic = time.perf_counter()
    prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, R=R, L=L, N_Fourier=N_Fourier,
                  nodes=nodes, weights=weights, T=T, N_Riccati=N_Riccati)
    iv = cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
    zero_ind = np.where(np.logical_or(np.logical_or(iv == 0, iv == np.nan), iv == np.inf))
    iv[zero_ind] = 1e-16 * np.exp(np.random.uniform(-5, 10)) * np.ones(len(zero_ind))
    duration = time.perf_counter() - tic
    prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, R=R, L=L/1.2, N_Fourier=N_Fourier // 2,
                  nodes=nodes, weights=weights, T=T, N_Riccati=int(N_Riccati/1.5))
    iv_approx = cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
    zero_ind = np.where(np.logical_or(np.logical_or(iv_approx == 0, iv_approx == np.nan), iv_approx == np.inf))
    iv_approx[zero_ind] = 1e-16 * np.exp(np.random.uniform(-5, 10)) * np.ones(len(zero_ind))
    error = np.amax(np.abs(iv_approx-iv)/iv)
    
    while error > rel_tol or np.amin(iv) < 1e-08:

        prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, R=R, L=L, N_Fourier=N_Fourier,
                      nodes=nodes, weights=weights, T=T, N_Riccati=N_Riccati//2)
        iv_approx = cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
        error_Riccati = np.amax((np.abs(iv_approx - iv) / iv))
        # print(f'error Riccati: {error_Riccati}')
        if error_Riccati < rel_tol/10 and np.amax(iv_approx) > 1e-08:
            N_Riccati = N_Riccati//2

        prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, R=R, L=L, N_Fourier=N_Fourier//3,
                      nodes=nodes, weights=weights, T=T, N_Riccati=N_Riccati)
        iv_approx = cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
        error_Fourier = np.amax((np.abs(iv_approx - iv) / iv))
        # print(f'error Fourier: {error_Fourier}')
        if error_Fourier < rel_tol / 10 and np.amax(iv_approx) > 1e-08:
            N_Fourier = N_Fourier // 3

        iv_approx = iv
        L = L * 1.2
        N_Fourier = N_Fourier * 2
        N_Riccati = int(N_Riccati * 1.5)
        # print('Markov', N, error, L, N_Fourier, N_Riccati, duration, time.strftime("%H:%M:%S", time.localtime()))
        tic = time.perf_counter()
        with np.errstate(all='raise'):
            try:
                prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, R=R, L=L, N_Fourier=N_Fourier,
                              nodes=nodes, weights=weights, T=T, N_Riccati=N_Riccati)
                iv = cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
                result = iv
                zero_ind = np.where(np.logical_or(np.logical_or(iv == 0, iv == np.nan), iv == np.inf))
                iv[zero_ind] = 1e-16 * np.exp(np.random.uniform(-5, 10)) * np.ones(len(zero_ind))
            except:
                L = L/1.3
                N_Fourier = int(N_Fourier / 1.5)
                N_Riccati = int(N_Riccati/1.2)
                if result is not None and error_Fourier < rel_tol and error_Riccati < rel_tol:
                    if smoothing:
                        return cf.smoothen(np.log(K), result, np.log(K_result))
                    else:
                        return result
                iv = 1e-16 * np.exp(np.random.uniform(-5, 10)) * np.ones(len(K))
        duration = time.perf_counter() - tic
        error = np.amax((np.abs(iv_approx - iv)/iv))
        # print(error)
    return iv
