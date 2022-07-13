import time
import numpy as np
import ComputationalFinance as cf
from scipy.special import gamma


def characteristic_function(a, H, lambda_, rho, nu, theta, V_0, T, N_Riccati=200):
    """
    Gives the characteristic function of the log-price in the rough Heston model as described in El Euch and Rosenbaum,
    The characteristic function of rough Heston models. Uses the Adams scheme.
    :param a: Argument of the characteristic function (assumed to be a vector)
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :return: The characteristic function
    """
    dt = T / N_Riccati
    a_ = -0.5 * (a + np.complex(0, 1)) * a
    b_ = np.complex(0, 1) * rho * nu * a - lambda_
    c_ = nu * nu / 2
    coefficient = dt ** (H + 0.5) / gamma(H + 2.5)
    v_1 = np.arange(N_Riccati+1) ** (H + 1.5)
    v_2 = np.arange(N_Riccati+1) ** (H + 0.5)
    v_3 = coefficient * (v_1[N_Riccati:1:-1] + v_1[N_Riccati-2::-1] - 2 * v_1[N_Riccati-1:0:-1])
    v_4 = coefficient * (v_1[:-1] - (np.arange(N_Riccati) - H - 0.5) * v_2[1:])
    v_5 = dt ** (H + 0.5) / gamma(H + 1.5) * (v_2[N_Riccati:0:-1] - v_2[N_Riccati-1::-1])

    def F(x):
        return a_ + (b_ + c_ * x) * x

    h = np.zeros(shape=(len(a), N_Riccati + 1), dtype=np.cdouble)
    F_vec = np.zeros(shape=(len(a), N_Riccati), dtype=np.cdouble)
    F_vec[:, 0] = F(h[:, 0])
    h[:, 1] = F_vec[:, 0] * v_4[0] + coefficient * F(F_vec[:, 0] * v_5[-1])
    for k in range(2, N_Riccati + 1):
        F_vec[:, k-1] = F(h[:, k-1])
        h[:, k] = F_vec[:, 0] * v_4[k-1] + F_vec[:, 1:k] @ v_3[-k+1:] + coefficient * F(F_vec[:, :k] @ v_5[-k:])

    integral = np.trapz(h, dx=dt)
    fractional_integral = -np.trapz(h, x=np.linspace(T, 0, N_Riccati+1) ** (0.5 - H) / gamma(1.5-H), axis=1)
    return np.exp(theta * integral + V_0 * fractional_integral)


def call(K, H, lambda_, rho, nu, theta, V_0, T, N_Riccati=200, R=2., L=50., N_Fourier=300):
    """
    Gives the price of a European call option in the rough Heston model as described in El Euch and Rosenbaum, The
    characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
    :param K: Strike prices, assumed to be a vector
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_Riccati: Number of time steps used in the solution of the fractional Riccati equation
    :param R: The (dampening) shift that we use for the Fourier inversion
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals, but only over
        [0, L]
    :param N_Fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    return: The price of the call option
    """
    return cf.price_eur_call_fourier(mgf=lambda u: characteristic_function(np.complex(0, -1) * u, H, lambda_, rho,
                                                                           nu, theta, V_0, T, N_Riccati),
                                     K=K, R=R, L=L, N=N_Fourier)


def implied_volatility_smile(K, H, lambda_, rho, nu, theta, V_0, T, N_Riccati=None, R=2., L=None, N_Fourier=None,
                             rel_tol=1e-02):
    """
    Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
    Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
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
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals, but only over
        [0, L]
    :param N_Fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    :param rel_tol: Required maximal relative error in the implied volatility
    return: The price of the call option
    """
    if N_Riccati is None:
        N_Riccati = int(200 / T**0.8)
    if L is None:
        L = 70 / T
    if N_Fourier is None:
        N_Fourier = int(8 * L / np.sqrt(T))
    np.seterr(all='warn')

    def compute_iv(N_Riccati_, L_, N_Fourier_):
        return cf.iv_eur_call_fourier(mgf=lambda u: characteristic_function(np.complex(0, -1) * u, H, lambda_, rho,
                                                                            nu, theta, V_0, T, N_Riccati_),
                                      S=1., K=K, T=T, r=0., R=R, L=L_, N=N_Fourier_)

    tic = time.perf_counter()
    iv = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
    duration = time.perf_counter() - tic
    iv_approx = compute_iv(N_Riccati_=int(N_Riccati/1.5), L_=L/1.2, N_Fourier_=N_Fourier//2)
    error = np.amax(np.abs(iv_approx - iv) / iv)
    # print(np.abs(iv_approx - iv) / iv)

    while np.isnan(error) or error > rel_tol or np.sum(np.isnan(iv)) > 0:
        iv_approx = compute_iv(N_Riccati_=N_Riccati//2, L_=L, N_Fourier_=N_Fourier)
        error_Riccati = np.amax(np.abs(iv_approx - iv) / iv)
        print(f'error Riccati: {error_Riccati}')
        if not np.isnan(error_Riccati) and error_Riccati < rel_tol / 10 and np.sum(np.isnan(iv_approx)) == 0:
            N_Riccati = N_Riccati // 2

        iv_approx = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier//3)
        error_Fourier = np.amax(np.abs(iv_approx - iv) / iv)
        print(f'error Fourier: {error_Fourier}')
        if not np.isnan(error_Fourier) and error_Fourier < rel_tol / 10 and np.sum(np.isnan(iv_approx)) == 0:
            N_Fourier = N_Fourier // 3

        iv_approx = iv
        L = L * 1.2
        N_Fourier = N_Fourier * 2
        N_Riccati = int(N_Riccati * 1.5)
        print('True', error, L, N_Fourier, N_Riccati, duration, time.strftime("%H:%M:%S", time.localtime()))
        tic = time.perf_counter()
        with np.errstate(all='raise'):
            try:
                iv = compute_iv(N_Riccati_=N_Riccati, L_=L, N_Fourier_=N_Fourier)
            except:
                L = L / 1.3
                N_Fourier = int(N_Fourier/1.5)
                N_Riccati = int(N_Riccati/1.2)
                if error_Fourier < rel_tol and error_Riccati < rel_tol:
                    return iv
        duration = time.perf_counter() - tic
        error = np.amax(np.abs(iv_approx - iv) / iv)
        # print(np.abs(iv_approx - iv) / iv)
        print(error)

    return iv


def implied_volatility_surface(K, H, lambda_, rho, nu, theta, V_0, T, rel_tol=1e-02):
    """
    Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
    Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
    :param K: Strike price, assumed to be a vector or a matrix
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Vector of maturities
    :param rel_tol: Required maximal relative error in the implied volatility
    return: The price of the call option
    """
    if len(K.shape) == 1:
        T = np.sort(T)
        K_ = np.sort(K)
        K = np.empty((len(T), len(K)))
        K[-1, :] = K_
        k_lower = np.log(K_[0]) / np.sqrt(T[-1])
        k_upper = np.log(K_[-1]) / np.sqrt(T[-1])
        for i in range(len(T)-1):
            K[i, :] = np.linspace(k_lower * np.sqrt(T[i]), k_upper * np.sqrt(T[i]), len(K_))

    iv_surface = np.empty_like(K)
    for i in range(len(T)):
        iv_surface[i, :] = implied_volatility_smile(K=K[i, :], H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta,
                                                    V_0=V_0, T=T[i], rel_tol=rel_tol)

    return iv_surface
