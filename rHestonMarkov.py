import time
import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk


def characteristic_function(z, lambda_, rho, nu, theta, V_0, T, N_Riccati, nodes, weights):
    """
    Gives the characteristic function of the log-price in the Markovian approximation of the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a vector)
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :return: The characteristic function
    """

    nodes, weights = rk.sort(nodes, weights)
    n_zero_nodes = np.sum(nodes < 1e-08)
    dt = T/N_Riccati
    N = len(nodes)
    times = np.linspace(T, 0, N_Riccati + 1)
    g = np.sum(weights[:n_zero_nodes]) * times
    for j in range(n_zero_nodes, N):
        exponent = nodes[j] * times
        factor = np.ones(shape=exponent.shape)
        factor[exponent < 300] = 1-np.exp(-exponent[exponent < 300])
        g = g + weights[j]/nodes[j] * factor
    g = theta * g + V_0

    a = nu * nu / 2
    b = rho * nu * np.complex(0, 1) * z - lambda_
    c = -(z + np.complex(0, 1)) * z / 2
    exp_nodes = np.zeros(N)
    temp = nodes * dt
    exp_nodes[temp < 300] = np.exp(-temp[temp < 300])
    div_nodes = np.zeros(N)
    div_nodes[:n_zero_nodes] = dt
    div_nodes[n_zero_nodes:] = (1 - exp_nodes[n_zero_nodes:]) / nodes[n_zero_nodes:]
    new_div = np.dot(div_nodes, weights)
    new_exp = exp_nodes * weights

    def F(x):
        return (a * x + b) * x + c

    psi_x = np.zeros((len(z), N), dtype=np.cdouble)
    F_psi = np.zeros((len(z), N_Riccati + 1), dtype=np.cdouble)
    F_psi[:, 0] = c
    psi = np.zeros(len(z), dtype=np.cdouble)

    for i in range(N_Riccati):
        psi_P = (psi + F_psi[:, i] * new_div + psi_x @ new_exp) / 2
        psi_x = np.outer(F(psi_P), div_nodes) + psi_x * exp_nodes[None, :]
        psi = psi_x @ weights
        F_psi[:, i + 1] = F(psi)

    return np.exp(np.trapz(F_psi * g, dx=dt))


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
    return cf.price_eur_call_fourier(mgf=lambda z: characteristic_function(z=np.complex(0, -1) * z, lambda_=lambda_,
                                                                           rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                                           T=T, N_Riccati=N_Riccati, nodes=nodes,
                                                                           weights=weights),
                                     K=K, R=R, L=L, N=N_Fourier)


def implied_volatility_smile(K, H, lambda_, rho, nu, theta, V_0, T, N, N_Riccati=None, R=2, L=None, N_Fourier=None,
                             mode="european", rel_tol=1e-02, nodes=None, weights=None):
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
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    return: The price of the call option
    """
    if N_Riccati is None:
        N_Riccati = int(200 / T**0.8)
    if L is None:
        L = 70/T
    if N_Fourier is None:
        N_Fourier = int(8*L / np.sqrt(T))
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    else:
        N = len(nodes)
    np.seterr(all='warn')

    def compute_iv(N_Riccati_, L_, N_Fourier_):
        return cf.iv_eur_call_fourier(mgf=lambda u: characteristic_function(np.complex(0, -1) * u, lambda_, rho,
                                                                            nu, theta, V_0, T, N_Riccati_, nodes,
                                                                            weights),
                                      S=1., K=K, T=T, r=0., R=R, L=L_, N=N_Fourier_)

    tic = time.perf_counter()
    iv = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
    duration = time.perf_counter() - tic
    iv_approx = compute_iv(N_Riccati_=int(N_Riccati/1.5), L_=L/1.2, N_Fourier_=N_Fourier//2)
    error = np.amax(np.abs(iv_approx-iv)/iv)
    
    while np.isnan(error) or error > rel_tol or np.sum(np.isnan(iv)) > 0:
        iv_approx = compute_iv(N_Riccati_=N_Riccati//2, L_=L, N_Fourier_=N_Fourier)
        error_Riccati = np.amax(np.abs(iv_approx - iv) / iv)
        print(f'error Riccati: {error_Riccati}')
        if not np.isnan(error_Riccati) and error_Riccati < rel_tol / 10 and np.sum(np.isnan(iv_approx)) == 0:
            N_Riccati = N_Riccati//2

        iv_approx = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier//3)
        error_Fourier = np.amax(np.abs(iv_approx - iv) / iv)
        print(f'error Fourier: {error_Fourier}')
        if not np.isnan(error_Fourier) and error_Fourier < rel_tol / 10 and np.sum(np.isnan(iv_approx)) == 0:
            N_Fourier = N_Fourier // 3

        iv_approx = iv
        L = L * 1.2
        N_Fourier = N_Fourier * 2
        N_Riccati = int(N_Riccati * 1.5)
        print('Markov', N, error, L, N_Fourier, N_Riccati, duration, time.strftime("%H:%M:%S", time.localtime()))
        tic = time.perf_counter()
        with np.errstate(all='raise'):
            try:
                iv = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
            except:
                L = L/1.3
                N_Fourier = int(N_Fourier / 1.5)
                N_Riccati = int(N_Riccati/1.2)
                if error_Fourier < rel_tol and error_Riccati < rel_tol:
                    return iv
        duration = time.perf_counter() - tic
        error = np.amax((np.abs(iv_approx - iv)/iv))
        print(error)
    return iv
