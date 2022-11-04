import numpy as np
import RoughKernel as rk
import rHestonBackbone
import psutil


def characteristic_function(z, S_0, lambda_, rho, nu, theta, V_0, T, N_Riccati, nodes, weights):
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
    psi = np.zeros(len(z), dtype=np.cdouble)
    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = 2 * np.sqrt(len(z)) * np.sqrt(N_Riccati) * np.sqrt(np.array([0.], dtype=np.cdouble).nbytes)
    if available_memory > necessary_memory:
        F_psi = np.zeros((len(z), N_Riccati + 1), dtype=np.cdouble)
        F_psi[:, 0] = c

        for i in range(N_Riccati):
            psi_P = (psi + F_psi[:, i] * new_div + psi_x @ new_exp) / 2
            psi_x = np.outer(F(psi_P), div_nodes) + psi_x * exp_nodes[None, :]
            psi = psi_x @ weights
            F_psi[:, i + 1] = F(psi)

        return np.exp(complex(0, 1) * z * np.log(S_0) + np.trapz(F_psi * g, dx=dt))
    else:
        F_psi = c * np.ones(len(z), dtype=np.cdouble)
        integral = F_psi * g[0] * dt / 2

        for i in range(N_Riccati):
            psi_P = (psi + F_psi * new_div + psi_x @ new_exp) / 2
            psi_x = np.outer(F(psi_P), div_nodes) + psi_x * exp_nodes[None, :]
            psi = psi_x @ weights
            F_psi = F(psi)
            integral += F_psi * g[i + 1] * dt

        integral -= F_psi * g[-1] * dt / 2

        return np.exp(complex(0, 1) * z * np.log(S_0) + integral)


def characteristic_function_geom_asian(z, S_0, lambda_, rho, nu, theta, V_0, T, N_Riccati, nodes, weights):
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

    nodes, weights = rk.sort(nodes, weights)
    n_zero_nodes = np.sum(nodes < 1e-08)
    dt = T / N_Riccati
    N = len(nodes)

    z = complex(0, 1) * z
    z_sq = z * z
    exp_nodes = np.zeros(N)
    temp = nodes * dt
    exp_nodes[temp < 300] = np.exp(-temp[temp < 300])
    div_nodes = np.zeros(N)
    div_nodes[:n_zero_nodes] = dt
    div_nodes[n_zero_nodes:] = (1 - exp_nodes[n_zero_nodes:]) / nodes[n_zero_nodes:]
    div_weights = div_nodes * weights
    new_div = np.sum(div_weights)

    def F(t, x):
        return 0.5 * t ** 2 * z_sq - 0.5 * t * z + (rho * nu * t * z - lambda_ + 0.5 * nu ** 2 * x) * x

    psi_x = np.zeros((len(z), N), dtype=np.cdouble)
    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = 2 * np.sqrt(len(z)) * np.sqrt(N_Riccati) * np.sqrt(np.array([0.], dtype=np.cdouble).nbytes)

    if available_memory > necessary_memory:
        psi = np.zeros((N_Riccati + 1, len(z)), dtype=np.cdouble)

        for i in range(N_Riccati):
            psi_P = new_div * F(i / N_Riccati, psi[i, :]) + psi_x @ exp_nodes
            psi_x = div_weights[None, :] * F((i + 0.5) / N_Riccati, 0.5 * (psi[i, :] + psi_P))[:, None] \
                + psi_x * exp_nodes[None, :]
            psi[i + 1, :] = np.sum(psi_x, axis=1)

        integral = np.trapz(psi, dx=dt, axis=0)
        integral_sq = np.trapz(psi ** 2, dx=dt, axis=0)
        integral_time = np.trapz(psi * np.linspace(0, 1, N_Riccati + 1)[:, None], dx=dt, axis=0)
    else:
        psi = np.zeros((len(z),), dtype=np.cdouble)
        integral = 0
        integral_sq = 0
        integral_time = 0

        for i in range(N_Riccati):
            psi_P = new_div * F(i / N_Riccati, psi) + psi_x @ exp_nodes
            psi_x = div_weights[None, :] * F((i + 0.5) / N_Riccati, 0.5 * (psi + psi_P))[:, None] \
                + psi_x * exp_nodes[None, :]
            psi = np.sum(psi_x, axis=1)
            integral += psi * dt
            integral_sq += psi ** 2 * dt
            integral_time += psi * (i + 1) * dt * dt

        integral -= psi * dt / 2
        integral_sq -= psi ** 2 * dt / 2
        integral_time -= psi * T * dt / 2

    return np.exp(z * np.log(S_0) + (z / 6 - 0.25) * (V_0 * T) * z + (theta - lambda_ * V_0) * integral
                  + 0.5 * V_0 * nu ** 2 * integral_sq + V_0 * nu * rho * z * integral_time)


def characteristic_function_avg_vol(z, lambda_, nu, theta, V_0, T, N_Riccati, nodes, weights):
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

    nodes, weights = rk.sort(nodes, weights)
    n_zero_nodes = np.sum(nodes < 1e-08)
    dt = T / N_Riccati
    N = len(nodes)

    z = complex(0, 1) * z
    exp_nodes = np.zeros(N)
    temp = nodes * dt
    exp_nodes[temp < 300] = np.exp(-temp[temp < 300])
    div_nodes = np.zeros(N)
    div_nodes[:n_zero_nodes] = dt
    div_nodes[n_zero_nodes:] = (1 - exp_nodes[n_zero_nodes:]) / nodes[n_zero_nodes:]
    div_weights = div_nodes * weights
    new_div = np.sum(div_weights)

    def F(x):
        return z / T + (-lambda_ + 0.5 * nu ** 2 * x) * x

    psi_x = np.zeros((len(z), N), dtype=np.cdouble)
    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = 2 * np.sqrt(len(z)) * np.sqrt(N_Riccati) * np.sqrt(np.array([0.], dtype=np.cdouble).nbytes)

    if available_memory > necessary_memory:
        psi = np.zeros((N_Riccati + 1, len(z)), dtype=np.cdouble)

        for i in range(N_Riccati):
            psi_P = new_div * F(psi[i, :]) + psi_x @ exp_nodes
            psi_x = div_weights[None, :] * F(0.5 * (psi[i, :] + psi_P))[:, None] \
                + psi_x * exp_nodes[None, :]
            psi[i + 1, :] = np.sum(psi_x, axis=1)

        integral = np.trapz(psi, dx=dt, axis=0)
        integral_sq = np.trapz(psi ** 2, dx=dt, axis=0)
    else:
        psi = np.zeros((len(z),), dtype=np.cdouble)
        integral = 0
        integral_sq = 0

        for i in range(N_Riccati):
            psi_P = new_div * F(psi) + psi_x @ exp_nodes
            psi_x = div_weights[None, :] * F(0.5 * (psi + psi_P))[:, None] + psi_x * exp_nodes[None, :]
            psi = np.sum(psi_x, axis=1)
            integral += psi * dt
            integral_sq += psi ** 2 * dt

        integral -= psi * dt / 2
        integral_sq -= psi ** 2 * dt / 2

    return np.exp(z * V_0 + (theta - lambda_ * V_0) * integral + 0.5 * V_0 * nu ** 2 * integral_sq)


def iv_eur_call(S_0, K, H, lambda_, rho, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None,
                weights=None, verbose=0):
    """
    Gives the implied volatility of the European call option in the rough Heston model
    as described in El Euch and Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme.
    Uses Fourier inversion.
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
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: characteristic_function(u, S_0, lambda_, rho, nu, theta, V_0,
                                                                                   T_, N_, nodes, weights),
                                S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose)


def skew_eur_call(S_0, H, lambda_, rho, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None,
                  weights=None, verbose=0):
    """
    Gives the skew of the European call option in the rough Heston model.
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
    return rHestonBackbone.skew_eur_call(char_fun=lambda u, T_, N_: characteristic_function(u, S_0, lambda_, rho, nu,
                                                                                            theta, V_0, T_, N_, nodes,
                                                                                            weights),
                                         T=T, rel_tol=rel_tol, verbose=verbose)


def price_geom_asian_call(S_0, K, H, lambda_, rho, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None,
                          weights=None, verbose=0):
    """
    Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
    Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, np.array([T / 2, T]), mode)
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: characteristic_function_geom_asian(u, S_0, lambda_, rho,
                                                                                              nu, theta, V_0, T_, N_,
                                                                                              nodes, weights),
                                S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='geometric asian',
                                output='price')


def price_avg_vol_call(K, H, lambda_, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None,
                          weights=None, verbose=0):
    """
    Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
    Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    return rHestonBackbone.call(char_fun=lambda u, T_, N_: characteristic_function_avg_vol(u, lambda_, nu, theta, V_0,
                                                                                           T_, N_, nodes, weights),
                                S_0=V_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='average volatility',
                                output='price')

