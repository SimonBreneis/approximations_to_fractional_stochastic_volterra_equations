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


def compute_Fourier_inversion(S_0, K, T, fun, rel_tol=1e-03, verbose=0, return_error=False):
    """
    Computes the Fourier inversion given a relative error tolerance by finding the appropriate parameters for
    solving the Riccati equations, and computing the inverse Fourier integral.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array (1d or 2d)
    :param T: Numpy array of maturities
    :param fun: Function that computes the Fourier inversion given the numerical parameters. Is a function of
        T_ (single maturity), K_ (array of strikes), N_Riccati (number of Riccati intervals),
        L (cutoff for the Fourier integral), N_Fourier (number of Fourier intervals),
        R (dampening shift that makes the Fourier inversion integrable)
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    :param return_error: If True, also returns a relative error estimate
    return: The implied volatility of the call option
    """
    R = 2.  # The (dampening) shift that we use for the Fourier inversion

    def single_maturity(T_, compute, eps):
        """
        Computes the Fourier inversion for a single maturity given a relative error tolerance by finding the appropriate
        parameters for solving the Riccati equations, and computing the inverse Fourier integral.
        :param T_: Single maturity
        :param compute: Function that computes the Fourier inversion given the numerical parameters for a single
            maturity. Is a function of
            N_Riccati (number of Riccati intervals), L (cutoff for the Fourier integral),
            N_Fourier (number of Fourier intervals)
        :param eps: Relative error tolerance
        return: The implied volatility of the call option
        """
        N_Riccati = 250  # Number of time steps used in the solution of the fractional Riccati
        # equation
        L = 120 / T_ ** 0.4  # The value at which we cut off the Fourier integral, so we do not integrate over the
        # reals, but only over [0, L]
        N_Fourier = int(8 * L)  # The number of points used in the trapezoidal rule for the
        # approximation of the Fourier integral
        np.seterr(all='warn')
        tic = time.perf_counter()
        smile = compute(N_Riccati=N_Riccati, L=L, N_Fourier=N_Fourier)
        duration = time.perf_counter() - tic
        smile_approx = compute(N_Riccati=int(N_Riccati / 1.6), L=L / 1.2, N_Fourier=N_Fourier // 2)
        error = np.amax(np.abs(smile_approx - smile) / smile)
        if verbose >= 3:
            print(np.abs(smile_approx - smile) / smile)
        if verbose >= 2:
            print(error)

        while np.isnan(error) or error > eps or np.sum(np.isnan(smile)) > 0:
            if time.perf_counter() - tic > 3600:
                raise RuntimeError('Smile was not computed in given time.')
            if np.sum(np.isnan(smile)) == 0:
                smile_approx = compute(N_Riccati=N_Riccati // 2, L=L, N_Fourier=N_Fourier)
                error_Riccati = np.amax(np.abs(smile_approx - smile) / smile)
                if not np.isnan(error_Riccati) and error_Riccati < eps / 5 and np.sum(np.isnan(smile_approx)) == 0:
                    N_Riccati = int(N_Riccati / 1.8)

                smile_approx = compute(N_Riccati=N_Riccati, L=L, N_Fourier=N_Fourier // 2)
                error_Fourier = np.amax(np.abs(smile_approx - smile) / smile)
                if not np.isnan(error_Fourier) and error_Fourier < eps / 5 and np.sum(np.isnan(smile_approx)) == 0:
                    N_Fourier = N_Fourier // 2
            else:
                error_Fourier, error_Riccati = np.nan, np.nan

            smile_approx = smile
            if np.sum(np.isnan(smile)) > 0:
                L = L * 1.6
                N_Fourier = int(N_Fourier * 1.7)
                N_Riccati = int(N_Riccati * 1.7)
            else:
                L = L * 1.4
                N_Fourier = int(N_Fourier * 1.4) if error_Fourier < eps / 2 else N_Fourier * 2
                N_Riccati = int(N_Riccati * 1.4) if error_Riccati < eps / 2 else N_Riccati * 2
            if verbose >= 2:
                print(error, error_Fourier, error_Riccati, L, N_Fourier, N_Riccati, duration,
                      time.strftime("%H:%M:%S", time.localtime()))

            tic = time.perf_counter()
            smile = compute(N_Riccati=N_Riccati, L=L, N_Fourier=N_Fourier)
            duration = time.perf_counter() - tic
            error = np.amax(np.abs(smile_approx - smile) / smile)
            if verbose >= 3:
                print(np.abs(smile_approx - smile) / smile)
            if verbose >= 2:
                print(error)

        return smile, error

    T_is_float = False
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        T_is_float = True
    if len(K.shape) == 1:
        _, K = cf.maturity_tensor_strike(S_0=S_0, K=K, T=T)

    surface = np.empty_like(K)
    errors = np.empty(len(T))
    for i in range(len(T)):
        if verbose >= 1 and len(T) > 1:
            print(f'Now simulating maturity {i+1} of {len(T)}')

        def fun_maturity(N_Riccati, L, N_Fourier):
            return fun(T_=T[i], K_=K[i, :], N_Riccati=N_Riccati, L=L, N_Fourier=N_Fourier, R=R)
        surface[i, :], errors[i] = single_maturity(compute=fun_maturity, T_=T[i],
                                                   eps=rel_tol[i] if isinstance(rel_tol, np.ndarray) else rel_tol)
    if T_is_float:
        surface = surface[0, :]
        errors = errors[0]
    if return_error:
        return surface, errors
    return surface


def eur_call_put(S_0, K, lambda_, rho, nu, theta, V_0, T, r=0., N=0, mode="european", rel_tol=1e-03, H=None, nodes=None,
                 weights=None, implied_vol=True, call=True, return_error=False, verbose=0):
    """
    Computes the implied volatility or price of the European call or put option in the rough Heston model.
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
    :param r: Interest rate
    :param N: Total number of points in the quadrature rule
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param rel_tol: Required maximal relative error in the implied volatility
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    :param implied_vol: If True, computes the implied volatility, else computes the price
    :param call: If True, uses the call option, else uses the put option
    :param return_error: If True, also returns a relative maximal error estimate
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    if N >= 1 and (nodes is None or weights is None):
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode=mode)

    def mgf(z, T_, N_):
        return cf_log_price(z=np.complex(0, -1) * z, S_0=S_0, lambda_=lambda_, rho=rho, nu=nu, theta=theta, r=r,
                            V_0=V_0, T=T_, N_Riccati=N_, H=H, nodes=nodes, weights=weights)

    if implied_vol:
        def compute(T_, K_, N_Riccati, L, N_Fourier, R):
            return cf.iv_eur_call_put_fourier(mgf=lambda u: mgf(z=u, T_=T_, N_=N_Riccati), S_0=S_0, K=K_, T=T_, r=r,
                                              R=R, L=L, N=N_Fourier)
    else:
        def compute(T_, K_, N_Riccati, L, N_Fourier, R):
            return cf.price_eur_call_put_fourier(mgf=lambda u: mgf(z=u, T_=T_, N_=N_Riccati), K=K_, T=T_, r=r, R=R, L=L,
                                                 N=N_Fourier, log_price=True, call=call)

    return compute_Fourier_inversion(fun=compute, S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose,
                                     return_error=return_error)


def skew_eur_call_put(lambda_, rho, nu, theta, V_0, T, r=0., N=0, mode="european", rel_tol=1e-03, H=None, nodes=None,
                      weights=None, verbose=0):
    """
    Gives the skew of the European call or put option (skew is the same in both cases) in the rough Heston model.
    Uses Fourier inversion.
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
    return: The skew of the call option
    """
    if N >= 1 and (nodes is None or weights is None):
        nodes, weights = rk.quadrature_rule(H, N, T, mode)

    def single_skew(T_):

        def compute_smile(eps_, h_):
            K = np.exp(np.linspace(-200 * h_, 200 * h_, 401))
            smile_, error_ = eur_call_put(S_0=1., K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T_, r=r,
                                          N=N, mode=mode, rel_tol=eps_, H=H, nodes=nodes, weights=weights,
                                          implied_vol=True, call=True, return_error=True, verbose=verbose - 2)
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
        if verbose >= 2:
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

            if verbose >= 2:
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


def geom_asian_call_put(S_0, K, lambda_, rho, nu, theta, V_0, T, N=0, mode="european", rel_tol=1e-03, H=None,
                        nodes=None, weights=None, implied_vol=False, call=True, return_error=False, verbose=0):
    """
    Gives the price or implied volatility of the geometric Asian call or put option in the rough Heston model.
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
    :param implied_vol: If True, computes the implied volatility, else computes the price
    :param call: If True, uses the call option, else uses the put option
    :param return_error: If True, also returns a relative maximal error estimate
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    if N >= 1 and (nodes is None or weights is None):
        nodes, weights = rk.quadrature_rule(H, N, np.array([T / 2, T]), mode)

    def mgf(z, T_, N_):
        return cf_avg_log_price(z=np.complex(0, -1) * z, S_0=S_0, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                T=T_, N_Riccati=N_, H=H, nodes=nodes, weights=weights)

    if implied_vol:
        def compute(T_, K_, N_Riccati, L, N_Fourier, R):
            return cf.iv_geom_asian_call_fourier(mgf=lambda u: mgf(z=u, T_=T_, N_=N_Riccati), S_0=S_0, K=K_, T=T_,
                                                 R=R, L=L, N=N_Fourier)
    else:
        def compute(T_, K_, N_Riccati, L, N_Fourier, R):
            return cf.price_eur_call_put_fourier(mgf=lambda u: mgf(z=u, T_=T_, N_=N_Riccati), K=K_, T=T_, r=0., R=R,
                                                 L=L, N=N_Fourier, log_price=True, call=call)

    return compute_Fourier_inversion(fun=compute, S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose,
                                     return_error=return_error)


def price_avg_vol_call_put(K, lambda_, nu, theta, V_0, T, N=0, mode="european", rel_tol=1e-03, H=None, nodes=None,
                           weights=None, call=True, return_error=False, verbose=0):
    """
    Gives the price of the European call or put option on the average volatility in the rough Heston model.
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
    :param call: If True, uses the call option, else uses the put option
    :param return_error: If True, also returns a relative maximal error estimate
    :param verbose: Determines how many intermediate results are printed to the console
    return: The implied volatility of the call option
    """
    if N >= 1 and (nodes is None or weights is None):
        nodes, weights = rk.quadrature_rule(H, N, T, mode)

    def mgf(z, T_, N_):
        return cf_avg_vol(z=np.complex(0, -1) * z, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T_, N_Riccati=N_,
                          H=H, nodes=nodes, weights=weights)

    def compute(T_, K_, N_Riccati, L, N_Fourier, R):
        return cf.price_eur_call_put_fourier(mgf=lambda u: mgf(z=u, T_=T_, N_=N_Riccati), K=K_, T=T_, r=0., R=R,
                                             L=L, N=N_Fourier, log_price=True, call=call)

    return compute_Fourier_inversion(fun=compute, S_0=1., K=K, T=T, rel_tol=rel_tol, verbose=verbose,
                                     return_error=return_error)
