import time
import numpy as np
import ComputationalFinance as cf


def iv_eur_call(S_0, K, T, char_fun, rel_tol=1e-03, verbose=0, return_error=False):
    """
    Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
    Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
    :param S_0: Initial stock price
    :param K: Strike price, assumed to be a numpy array (1d or 2d)
    :param T: Numpy array of maturities
    :param char_fun: Characteristic function of the log-price. Is a function of the argument of the characteristic
        function, the maturity, and the number of steps used for the Riccati equation
    :param rel_tol: Required maximal relative error in the implied volatility
    :param verbose: Determines how many intermediate results are printed to the console
    :param return_error: If True, also returns a relative error estimate
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

        def compute_iv(N_Riccati_, L_, N_Fourier_):
            return cf.iv_eur_call_fourier(mgf=lambda u: char_fun_(np.complex(0, -1) * u, N_Riccati_),
                                          S_0=S_0, K=K_, T=T_, r=0., R=R, L=L_, N=N_Fourier_)

        tic = time.perf_counter()
        iv = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
        duration = time.perf_counter() - tic
        iv_approx = compute_iv(N_Riccati_=int(N_Riccati / 1.6), L_=L / 1.2, N_Fourier_=N_Fourier // 2)
        error = np.amax(np.abs(iv_approx - iv) / iv)
        if verbose >= 1:
            print(np.amax(np.abs(iv_approx - iv) / iv))

        while np.isnan(error) or error > eps or np.sum(np.isnan(iv)) > 0:
            if time.perf_counter() - tic > 3600:
                raise RuntimeError('Smile was not computed in given time.')
            if np.sum(np.isnan(iv)) == 0:
                iv_approx = compute_iv(N_Riccati_=N_Riccati // 2, L_=L, N_Fourier_=N_Fourier)
                error_Riccati = np.amax(np.abs(iv_approx - iv) / iv)
                if not np.isnan(error_Riccati) and error_Riccati < eps / 10 and np.sum(np.isnan(iv_approx)) == 0:
                    N_Riccati = int(N_Riccati / 1.8)

                iv_approx = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier // 2)
                error_Fourier = np.amax(np.abs(iv_approx - iv) / iv)
                if not np.isnan(error_Fourier) and error_Fourier < eps / 10 and np.sum(np.isnan(iv_approx)) == 0:
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
            iv = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
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


def skew_eur_call(T, char_fun, rel_tol=1e-03, verbose=0):
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
            smile_, error_ = iv_eur_call(S_0=1., K=K, T=T_, char_fun=char_fun, rel_tol=eps_, verbose=verbose - 1,
                                         return_error=True)
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
