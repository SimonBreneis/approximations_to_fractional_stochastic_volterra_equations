import time
import numpy as np
import ComputationalFinance as cf
import psutil
import RoughKernel as rk
from scipy.special import gamma


def solve_fractional_Riccati(F, T, N_Riccati, H=None, nodes=None, weights=None):
    """
    Solves psi(t) = int_0^t K(t - s) F(s, psi(s)) ds.
    :param F: Right-hand side (time-dependent!)
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :param H: Hurst parameter
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :return: The solution psi
    """
    dim = len(F(0, 0))
    dt = T / N_Riccati
    available_memory = np.sqrt(psutil.virtual_memory().available)
    necessary_memory = np.sqrt(5 * dim) * np.sqrt(N_Riccati) * np.sqrt(np.array([0.], dtype=np.cdouble).nbytes)
    if necessary_memory > available_memory:
        raise MemoryError(f'Not enough memory to compute the characteristic function of the rough Heston model with'
                          f'{dim} inputs and {N_Riccati} time steps. Roughly {necessary_memory}**2 bytes needed, '
                          f'while only {available_memory}**2 bytes are available.')
    
    if nodes is None or weights is None:
        coefficient = dt ** (H + 0.5) / gamma(H + 2.5)
        v_1 = np.arange(N_Riccati + 1) ** (H + 1.5)
        v_2 = np.arange(N_Riccati + 1) ** (H + 0.5)
        v_3 = coefficient * (v_1[N_Riccati:1:-1] + v_1[N_Riccati - 2::-1] - 2 * v_1[N_Riccati - 1:0:-1])
        v_4 = coefficient * (v_1[:-1] - (np.arange(N_Riccati) - H - 0.5) * v_2[1:])
        v_5 = dt ** (H + 0.5) / gamma(H + 1.5) * (v_2[N_Riccati:0:-1] - v_2[N_Riccati - 1::-1])

        psi = np.zeros(shape=(dim, N_Riccati + 1), dtype=np.cdouble)
        F_vec = np.zeros(shape=(dim, N_Riccati), dtype=np.cdouble)
        F_vec[:, 0] = F(0, psi[:, 0])
        psi[:, 1] = F_vec[:, 0] * v_4[0] + coefficient * F(0, F_vec[:, 0] * v_5[-1])
        for k in range(2, N_Riccati + 1):
            F_vec[:, k - 1] = F((k - 1) / N_Riccati, psi[:, k - 1])
            psi[:, k] = F_vec[:, 0] * v_4[k - 1] + F_vec[:, 1:k] @ v_3[-k + 1:] \
                + coefficient * F(k / N_Riccati, F_vec[:, :k] @ v_5[-k:])
    else:
        nodes, weights = rk.sort(nodes, weights)
        n_zero_nodes = np.sum(nodes < 1e-08)
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


def cf_log_price(z, S_0, lambda_, rho, nu, theta, V_0, T, N_Riccati, r=0., H=None, nodes=None, weights=None):
    """
    Gives the characteristic function of the log-price of the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param S_0: Initial stock price
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :param r: Interest rate
    :param H: Hurst parameter
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

    psi = solve_fractional_Riccati(F=F, T=T, N_Riccati=N_Riccati, H=H, nodes=nodes, weights=weights)
    integral = np.trapz(psi, dx=T / N_Riccati)
    integral_sq = np.trapz(psi ** 2, dx=T / N_Riccati)

    return np.exp((np.log(S_0) + r * T) * z + V_0 * T * c + (theta + V_0 * b) * integral + V_0 * a * integral_sq)


def cf_avg_log_price(z, S_0, lambda_, rho, nu, theta, V_0, T, N_Riccati, H=None, nodes=None, weights=None):
    """
    Gives the characteristic function of the average (on [0, T]) log-price of the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param S_0: Initial stock price
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :param H: Hurst parameter
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :return: The characteristic function
    """
    z = complex(0, 1) * z
    z_sq = z * z
    dt = T / N_Riccati

    def F(t, x):
        return 0.5 * t ** 2 * z_sq - 0.5 * t * z + (rho * nu * t * z - lambda_ + 0.5 * nu ** 2 * x) * x

    psi = solve_fractional_Riccati(F=F, T=T, N_Riccati=N_Riccati, H=H, nodes=nodes, weights=weights)
    integral = np.trapz(psi, dx=dt)
    integral_sq = np.trapz(psi ** 2, dx=dt)
    integral_time = np.trapz(psi * np.linspace(0, 1, N_Riccati + 1), dx=dt)

    return np.exp(z * np.log(S_0) + (z / 6 - 0.25) * (V_0 * T) * z + (theta - lambda_ * V_0) * integral
                  + 0.5 * V_0 * nu ** 2 * integral_sq + V_0 * nu * rho * z * integral_time)


def cf_avg_vol(z, lambda_, nu, theta, V_0, T, N_Riccati, H=None, nodes=None, weights=None):
    """
    Gives the characteristic function of the average (on [0, T]) volatility of the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a numpy array)
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :param H: Hurst parameter
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :return: The characteristic function
    """
    z = complex(0, 1) * z
    dt = T / N_Riccati

    def F(t, x):
        return z / T + (-lambda_ + 0.5 * nu ** 2 * x) * x

    psi = solve_fractional_Riccati(F=F, T=T, N_Riccati=N_Riccati, H=H, nodes=nodes, weights=weights)
    integral = np.trapz(psi, dx=dt)
    integral_sq = np.trapz(psi ** 2, dx=dt)

    return np.exp(z * V_0 + (theta - lambda_ * V_0) * integral + 0.5 * V_0 * nu ** 2 * integral_sq)


def call(S_0, K, T, char_fun, r=0., rel_tol=1e-03, verbose=0, return_error=False, option='european', output='iv'):
    """
    Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
    Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array (1d or 2d)
    :param T: Numpy array of maturities
    :param char_fun: Characteristic function of the log-price. Is a function of the argument of the characteristic
        function, the maturity, and the number of steps used for the Riccati equation
    :param r: Interest rate
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    :param return_error: If True, also returns a relative error estimate
    :param option: Either 'european' or 'geometric asian'
    :param output: Either 'iv' or 'price'
    return: The implied volatility of the call option
    """

    def single_smile(K_, T_, char_fun_, eps):
        """
        Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
        Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
        :param K_: Strike price, assumed to be a numpy array
        :param char_fun_: Characteristic function of the log-price. Is a function of the argument of the characteristic
            function and the number of steps used for the Riccati equation
        :param T_: Maturity
        :param eps: Relative error tolerance
        return: The implied volatility of the call option
        """
        N_Riccati = 250  # Number of time steps used in the solution of the fractional Riccati
        # equation
        L = 120 / T_ ** 0.4  # The value at which we cut off the Fourier integral, so we do not integrate over the
        # reals, but only over [0, L]
        N_Fourier = int(8 * L)  # The number of points used in the trapezoidal rule for the
        # approximation of the Fourier integral
        R = 2.  # The (dampening) shift that we use for the Fourier inversion
        np.seterr(all='warn')
        if output == 'iv':
            if option == 'european':
                def compute(N_Riccati_, L_, N_Fourier_):
                    return cf.iv_eur_call_fourier(mgf=lambda u: char_fun_(np.complex(0, -1) * u, N_Riccati_),
                                                  S_0=S_0, K=K_, T=T_, r=r, R=R, L=L_, N=N_Fourier_)
            elif option == 'geometric asian':
                def compute(N_Riccati_, L_, N_Fourier_):
                    return cf.iv_geom_asian_call_fourier(mgf=lambda u: char_fun_(np.complex(0, -1) * u, N_Riccati_),
                                                         S_0=S_0, K=K_, T=T_, R=R, L=L_, N=N_Fourier_)
            else:
                raise NotImplementedError(f'Option {option} with output {output} is not implemented.')
        elif output == 'price':
            if option == 'geometric asian':
                def compute(N_Riccati_, L_, N_Fourier_):
                    return cf.price_call_fourier(mgf=lambda u: char_fun_(np.complex(0, -1) * u, N_Riccati_),
                                                 K=K_, R=R, L=L_, N=N_Fourier_, log_price=True)
            elif option == 'average volatility':
                def compute(N_Riccati_, L_, N_Fourier_):
                    return cf.price_call_fourier(mgf=lambda u: char_fun_(np.complex(0, -1) * u, N_Riccati_),
                                                 K=K_, R=R, L=L_, N=N_Fourier_, log_price=False)
            else:
                raise NotImplementedError(f'Option {option} with output {output} is not implemented.')
        else:
            raise NotImplementedError(f'Output {output} is not implemented.')

        tic = time.perf_counter()
        iv = compute(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
        duration = time.perf_counter() - tic
        iv_approx = compute(N_Riccati_=int(N_Riccati / 1.6), L_=L / 1.2, N_Fourier_=N_Fourier // 2)
        error = np.amax(np.abs(iv_approx - iv) / iv)
        if verbose >= 1:
            print(np.amax(np.abs(iv_approx - iv) / iv))

        while np.isnan(error) or error > eps or np.sum(np.isnan(iv)) > 0:
            if time.perf_counter() - tic > 3600:
                raise RuntimeError('Smile was not computed in given time.')
            if np.sum(np.isnan(iv)) == 0:
                iv_approx = compute(N_Riccati_=N_Riccati // 2, L_=L, N_Fourier_=N_Fourier)
                error_Riccati = np.amax(np.abs(iv_approx - iv) / iv)
                if not np.isnan(error_Riccati) and error_Riccati < eps / 5 and np.sum(np.isnan(iv_approx)) == 0:
                    N_Riccati = int(N_Riccati / 1.8)

                iv_approx = compute(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier // 2)
                error_Fourier = np.amax(np.abs(iv_approx - iv) / iv)
                if not np.isnan(error_Fourier) and error_Fourier < eps / 5 and np.sum(np.isnan(iv_approx)) == 0:
                    N_Fourier = N_Fourier // 2
            else:
                error_Fourier, error_Riccati = np.nan, np.nan

            iv_approx = iv
            if np.sum(np.isnan(iv)) > 0:
                L = L * 1.6
                N_Fourier = int(N_Fourier * 1.7)
                N_Riccati = int(N_Riccati * 1.7)
            else:
                L = L * 1.4
                N_Fourier = int(N_Fourier * 1.4) if error_Fourier < eps / 2 else N_Fourier * 2
                N_Riccati = int(N_Riccati * 1.4) if error_Riccati < eps / 2 else N_Riccati * 2
            if verbose >= 1:
                print(error, error_Fourier, error_Riccati, L, N_Fourier, N_Riccati, duration,
                      time.strftime("%H:%M:%S", time.localtime()))

            tic = time.perf_counter()
            iv = compute(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
            duration = time.perf_counter() - tic
            error = np.amax(np.abs(iv_approx - iv) / iv)
            if verbose >= 1:
                # print(np.abs(iv_approx - iv) / iv)
                print(error)

        return iv, error

    T_is_float = False
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        T_is_float = True
    if len(K.shape) == 1:
        _, K = cf.maturity_tensor_strike(S_0=S_0, K=K, T=T)

    iv_surface = np.empty_like(K)
    errors = np.empty(len(T))
    for i in range(len(T)):
        if verbose >= 1 and len(T) > 1:
            print(f'Now simulating maturity {i+1} of {len(T)}')
        iv_surface[i, :], errors[i] = single_smile(K_=K[i, :], T_=T[i], char_fun_=lambda u, N: char_fun(u, T[i], N),
                                                   eps=rel_tol[i] if isinstance(rel_tol, np.ndarray) else rel_tol)
    if T_is_float:
        iv_surface = iv_surface[0, :]
        errors = errors[0]
    if return_error:
        return iv_surface, errors
    return iv_surface


def skew_eur_call_comp(T, char_fun, rel_tol=1e-03, verbose=0):
    """
    Gives the skew of the European call option in the rough Heston model.
    :param T: Numpy array of maturities
    :param char_fun: Characteristic function of the log-price. Is a function of the argument of the characteristic
        function, the maturity, and the number of steps used for the Riccati equation
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    return: The skew of the call option
    """

    def single_skew(T_):

        def compute_smile(eps_, h_):
            K = np.exp(np.linspace(-200 * h_, 200 * h_, 401))
            smile_, error_ = call(S_0=1., K=K, T=T_, char_fun=char_fun, rel_tol=eps_, verbose=verbose - 1,
                                  return_error=True, option='european', output='iv')
            return np.array([smile_[198], smile_[199], smile_[201], smile_[202]]), error_

        eps = rel_tol / 5
        h = 0.0005 * np.sqrt(T_)
        smile, eps = compute_smile(eps_=eps, h_=h)
        skew_ = np.abs(smile[2] - smile[1]) / (2 * h)
        skew_h = np.abs(smile[3] - smile[0]) / (4 * h)
        old_eps = eps
        smile, eps = compute_smile(eps_=0.9 * eps, h_=h)
        skew_eps = skew_
        skew_eps_h = skew_h
        skew_ = np.abs(smile[2] - smile[1]) / (2 * h)
        skew_h = np.abs(smile[3] - smile[0]) / (4 * h)
        error = np.abs(skew_eps_h - skew_) / skew_
        error_eps = np.abs(skew_eps - skew_) / skew_
        error_h = np.abs(skew_h - skew_) / skew_
        print(error, error_eps, error_h)

        while error > rel_tol:
            if error_eps > rel_tol * 0.8:
                old_eps = eps
                smile, eps = compute_smile(eps_=0.9 * eps, h_=h)
                skew_eps = skew_
                skew_eps_h = skew_h
                skew_ = np.abs(smile[2] - smile[1]) / (2 * h)
                skew_h = np.abs(smile[3] - smile[0]) / (4 * h)
                error = np.abs(skew_eps_h - skew_) / skew_
                error_eps = np.abs(skew_eps - skew_) / skew_
                error_h = np.abs(skew_h - skew_) / skew_

                print(error, error_eps, error_h)
            else:
                h = h / 2
                smile, old_eps = compute_smile(eps_=1.1 * old_eps, h_=h)
                skew_eps = np.abs(smile[2] - smile[1]) / (2 * h)
                skew_eps_h = np.abs(smile[3] - smile[0]) / (4 * h)
                smile, eps = compute_smile(eps_=np.fmin(1.1 * eps, 0.99 * old_eps), h_=h)
                skew_ = np.abs(smile[2] - smile[1]) / (2 * h)
                skew_h = np.abs(smile[3] - smile[0]) / (4 * h)
                error = np.abs(skew_eps_h - skew_) / skew_
                error_eps = np.abs(skew_eps - skew_) / skew_
                error_h = np.abs(skew_h - skew_) / skew_

                print(error, error_eps, error_h)
        return skew_

    T_is_float = False
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        T_is_float = True

    skew = np.empty(len(T))
    for i in range(len(T)):
        if verbose >= 1:
            print(f'Now computing skew {i + 1} of {len(T)}')
        skew[i] = single_skew(T_=T[i])
    if T_is_float:
        skew = skew[0]
    return skew


def iv_eur_call(S_0, K, lambda_, rho, nu, theta, V_0, T, r=0., N=0, mode="european", rel_tol=1e-03, H=None, nodes=None,
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
    :param r: Interest rate
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
    if N >= 1 and (nodes is None or weights is None):
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode=mode)
    return call(char_fun=lambda u, T_, N_: cf_log_price(z=u, S_0=S_0, lambda_=lambda_, rho=rho, nu=nu, theta=theta, r=r,
                                                        V_0=V_0, T=T_, N_Riccati=N_, H=H, nodes=nodes, weights=weights),
                S_0=S_0, K=K, T=T, r=r, rel_tol=rel_tol, verbose=verbose)


def skew_eur_call(S_0, lambda_, rho, nu, theta, V_0, T, N=0, mode="european", rel_tol=1e-03, H=None, nodes=None,
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
    if N >= 1 and (nodes is None or weights is None):
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    return skew_eur_call_comp(char_fun=lambda u, T_, N_: cf_log_price(z=u, S_0=S_0, lambda_=lambda_, rho=rho, nu=nu,
                                                                      theta=theta, V_0=V_0, T=T_, N_Riccati=N_, H=H,
                                                                      nodes=nodes, weights=weights),
                              T=T, rel_tol=rel_tol, verbose=verbose)


def price_geom_asian_call(S_0, K, lambda_, rho, nu, theta, V_0, T, N=0, mode="european", rel_tol=1e-03, H=None,
                          nodes=None, weights=None, verbose=0):
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
    if N >= 1 and (nodes is None or weights is None):
        nodes, weights = rk.quadrature_rule(H, N, np.array([T / 2, T]), mode)
    return call(char_fun=lambda u, T_, N_: cf_avg_log_price(z=u, S_0=S_0, lambda_=lambda_, rho=rho, nu=nu, theta=theta,
                                                            V_0=V_0, T=T_, N_Riccati=N_, H=H, nodes=nodes,
                                                            weights=weights),
                S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='geometric asian', output='price')


def price_avg_vol_call(K, lambda_, nu, theta, V_0, T, N=0, mode="european", rel_tol=1e-03, H=None, nodes=None,
                       weights=None, verbose=0):
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
    if N >= 1 and (nodes is None or weights is None):
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    return call(char_fun=lambda u, T_, N_: cf_avg_vol(z=u, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T_,
                                                      N_Riccati=N_, H=H, nodes=nodes, weights=weights),
                S_0=V_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose, option='average volatility', output='price')
