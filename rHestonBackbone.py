import time
import numpy as np
import ComputationalFinance as cf


def iv_eur_call(S, K, T, char_fun, rel_tol=1e-03):
    """
    Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
    Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
    :param S: Initial stock price
    :param K: Strike price, assumed to be a vector or a matrix
    :param T: Vector of maturities
    :param char_fun: Characteristic function of the log-price. Is a function of the argument of the characteristic
        function, the maturity, and the number of steps used for the Riccati equation
    :param rel_tol: Required maximal relative error in the implied volatility
    return: The implied volatility of the call option
    """

    def single_smile(K_, T_, char_fun_):
        """
        Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
        Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
        :param K_: Strike price, assumed to be a vector
        :param char_fun_: Characteristic function of the log-price. Is a function of the argument of the characteristic
            function and the number of steps used for the Riccati equation
        :param T_: Final time/Time of maturity
        return: The price of the call option
        """
        N_Riccati = int(200 / T_ ** 0.8)  # Number of time steps used in the solution of the fractional Riccati equation
        L = 70 / T_  # The value at which we cut off the Fourier integral, so we do not integrate over the reals, but
        # only over [0, L]
        N_Fourier = int(
            8 * L / np.sqrt(T_))  # The number of points used in the trapezoidal rule for the approximation of
        # the Fourier integral
        R = 2.  # The (dampening) shift that we use for the Fourier inversion
        np.seterr(all='warn')

        def compute_iv(N_Riccati_, L_, N_Fourier_):
            return cf.iv_eur_call_fourier(mgf=lambda u: char_fun_(np.complex(0, -1) * u, N_Riccati_),
                                          S=S, K=K_, T=T_, r=0., R=R, L=L_, N=N_Fourier_)

        tic = time.perf_counter()
        iv = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
        duration = time.perf_counter() - tic
        iv_approx = compute_iv(N_Riccati_=int(N_Riccati / 1.5), L_=L / 1.2, N_Fourier_=N_Fourier // 2)
        error = np.amax(np.abs(iv_approx - iv) / iv)
        # print(np.abs(iv_approx - iv) / iv)

        while np.isnan(error) or error > rel_tol or np.sum(np.isnan(iv)) > 0:
            iv_approx = compute_iv(N_Riccati_=N_Riccati // 2, L_=L, N_Fourier_=N_Fourier)
            error_Riccati = np.amax(np.abs(iv_approx - iv) / iv)
            print(f'error Riccati: {error_Riccati}')
            if not np.isnan(error_Riccati) and error_Riccati < rel_tol / 10 and np.sum(np.isnan(iv_approx)) == 0:
                N_Riccati = N_Riccati // 2

            iv_approx = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier // 3)
            error_Fourier = np.amax(np.abs(iv_approx - iv) / iv)
            print(f'error Fourier: {error_Fourier}')
            if not np.isnan(error_Fourier) and error_Fourier < rel_tol / 10 and np.sum(np.isnan(iv_approx)) == 0:
                N_Fourier = N_Fourier // 3

            iv_approx = iv
            L = L * 1.2
            N_Fourier = N_Fourier * 2
            N_Riccati = int(N_Riccati * 1.5)
            print(error, L, N_Fourier, N_Riccati, duration, time.strftime("%H:%M:%S", time.localtime()))
            tic = time.perf_counter()
            with np.errstate(all='raise'):
                try:
                    iv = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
                except:
                    L = L / 1.3
                    N_Fourier = int(N_Fourier / 1.5)
                    N_Riccati = int(N_Riccati / 1.2)
                    if error_Fourier < rel_tol and error_Riccati < rel_tol:
                        return iv
            duration = time.perf_counter() - tic
            error = np.amax(np.abs(iv_approx - iv) / iv)
            # print(np.abs(iv_approx - iv) / iv)
            print(error)

        return iv

    T_is_float = False
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        T_is_float = True
    if len(K.shape) == 1:
        _, K = cf.maturity_tensor_strike(S=S, K=K, T=T)

    iv_surface = np.empty_like(K)
    for i in range(len(T)):
        iv_surface[i, :] = single_smile(K_=K[i, :], T_=T[i], char_fun_=lambda u, N: char_fun(u, T[i], N))
    if T_is_float:
        iv_surface = iv_surface[0, :]
    return iv_surface
