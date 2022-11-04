import numpy as np
import RoughKernel as rk
import rHestonBackbone
import psutil


def predictor_scheme(F, nodes, weights, T, N_Riccati):
    """
    Applies a predictor-corrector scheme to solve h(t) = int_0^t K(t - s) F(s, h(s)) ds, where K is the kernel
    determined by nodes and weights.
    :param F: Right-hand side (time-dependent!)
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :return: The solution h
    """
    dim = len(F(0, 0))
    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = np.sqrt(4 * dim) * np.sqrt(N_Riccati) * np.sqrt(np.array([0.], dtype=np.cdouble).nbytes)
    if necessary_memory > available_memory:
        raise MemoryError(f'Not enough memory to compute the characteristic function of the rough Heston model with'
                          f'{dim} inputs and {N_Riccati} time steps. Roughly {necessary_memory}**2 bytes needed, '
                          f'while only {available_memory}**2 bytes are available.')

    nodes, weights = rk.sort(nodes, weights)
    n_zero_nodes = np.sum(nodes < 1e-08)
    dt = T / N_Riccati
    N = len(nodes)

    exp_nodes = np.zeros(N)
    temp = nodes * dt
    exp_nodes[temp < 300] = np.exp(-temp[temp < 300])
    div_nodes = np.zeros(N)
    div_nodes[:n_zero_nodes] = dt
    div_nodes[n_zero_nodes:] = (1 - exp_nodes[n_zero_nodes:]) / nodes[n_zero_nodes:]
    div_weights = div_nodes * weights
    new_div = np.sum(div_weights)

    psi_x = np.zeros((dim, N), dtype=np.cdouble)
    psi = np.zeros((dim, N_Riccati + 1), dtype=np.cdouble)

    for i in range(N_Riccati):
        psi_P = new_div * F(i / N_Riccati, psi[:, i]) + psi_x @ exp_nodes
        psi_x = div_weights[None, :] * F((i + 0.5) / N_Riccati, 0.5 * (psi[:, i] + psi_P))[:, None] \
            + psi_x * exp_nodes[None, :]
        psi[:, i + 1] = np.sum(psi_x, axis=1)

    return psi


def cf_log_price(z, S_0, lambda_, rho, nu, theta, V_0, T, N_Riccati, nodes, weights):
    """
    Gives the characteristic function of the log-price in the Markovian approximation of the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param S_0: Initial stock price
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
    z = complex(0, 1) * z
    a = nu * nu / 2
    b = rho * nu * z - lambda_
    c = (z - 1) * z / 2

    def F(t, x):
        return c + (b + a * x) * x

    psi = predictor_scheme(F=F, nodes=nodes, weights=weights, T=T, N_Riccati=N_Riccati)
    integral = np.trapz(psi, dx=T / N_Riccati)
    integral_sq = np.trapz(psi ** 2, dx=T / N_Riccati)

    return np.exp(z * np.log(S_0) + V_0 * T * c + (theta + V_0 * b) * integral + V_0 * a * integral_sq)


def cf_avg_log_price(z, S_0, lambda_, rho, nu, theta, V_0, T, N_Riccati, nodes, weights):
    """
    Gives the characteristic function of the average (on [0, T]) log-price in the Markovian approximation of the
    rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param S_0: Initial stock price
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
    z = complex(0, 1) * z
    z_sq = z * z
    dt = T / N_Riccati

    def F(t, x):
        return 0.5 * t ** 2 * z_sq - 0.5 * t * z + (rho * nu * t * z - lambda_ + 0.5 * nu ** 2 * x) * x

    psi = predictor_scheme(F=F, nodes=nodes, weights=weights, T=T, N_Riccati=N_Riccati)

    integral = np.trapz(psi, dx=dt)
    integral_sq = np.trapz(psi ** 2, dx=dt)
    integral_time = np.trapz(psi * np.linspace(0, 1, N_Riccati + 1), dx=dt)

    return np.exp(z * np.log(S_0) + (z / 6 - 0.25) * (V_0 * T) * z + (theta - lambda_ * V_0) * integral
                  + 0.5 * V_0 * nu ** 2 * integral_sq + V_0 * nu * rho * z * integral_time)


def cf_avg_vol(z, lambda_, nu, theta, V_0, T, N_Riccati, nodes, weights):
    """
    Gives the characteristic function of the average (on [0, T]) volatility in the Markovian approximation of the
    rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :return: The characteristic function
    """
    z = complex(0, 1) * z
    dt = T / N_Riccati

    def F(t, x):
        return z / T + (-lambda_ + 0.5 * nu ** 2 * x) * x

    psi = predictor_scheme(F=F, nodes=nodes, weights=weights, T=T, N_Riccati=N_Riccati)

    integral = np.trapz(psi, dx=dt)
    integral_sq = np.trapz(psi ** 2, dx=dt)

    return np.exp(z * V_0 + (theta - lambda_ * V_0) * integral + 0.5 * V_0 * nu ** 2 * integral_sq)


def iv_eur_call(S_0, K, H, lambda_, rho, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None,
                weights=None, verbose=0):
    """
    Gives the implied volatility of the European call option in the rough Heston model. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param N: Total number of points in the quadrature rule
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param rel_tol: Required maximal relative error in the implied volatility
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: cf_log_price(u, S_0, lambda_, rho, nu, theta, V_0, T_, N_,
                                                                        nodes, weights),
                                S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose)


def skew_eur_call(S_0, H, lambda_, rho, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None,
                  weights=None, verbose=0):
    """
    Gives the skew of the European call option in the rough Heston model. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param N: Total number of points in the quadrature rule
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param rel_tol: Required maximal relative error in the implied volatility
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    :param verbose: Determines how many intermediate results are printed to the console
    return: The skew of the call option
    """
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    return rHestonBackbone.skew_eur_call(char_fun=lambda u, T_, N_: cf_log_price(u, S_0, lambda_, rho, nu, theta, V_0,
                                                                                 T_, N_, nodes, weights),
                                         T=T, rel_tol=rel_tol, verbose=verbose)


def price_geom_asian_call(S_0, K, H, lambda_, rho, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None,
                          weights=None, verbose=0):
    """
    Gives the price of the geometric Asian call option in the rough Heston model. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param N: Total number of points in the quadrature rule
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param rel_tol: Required maximal relative error in the implied volatility
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, np.array([T / 2, T]), mode)
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: cf_avg_log_price(u, S_0, lambda_, rho, nu, theta, V_0, T_,
                                                                            N_, nodes, weights),
                                S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='geometric asian',
                                output='price')


def price_avg_vol_call(K, H, lambda_, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None, weights=None,
                       verbose=0):
    """
    Gives the price of the European call option on the average volatility in the rough Heston model.
    Uses Fourier inversion.
    :param H: Hurst parameter
    :param K: Strike price, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param N: Total number of points in the quadrature rule
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param rel_tol: Required maximal relative error in the implied volatility
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: cf_avg_vol(u, lambda_, nu, theta, V_0, T_, N_, nodes,
                                                                      weights),
                                S_0=V_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='average volatility',
                                output='price')
