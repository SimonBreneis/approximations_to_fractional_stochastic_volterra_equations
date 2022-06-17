import numpy as np
import ComputationalFinance as cf
import scipy.special


def solve_fractional_Riccati(a, lambda_, rho, nu, A, B, N_Riccati=200):
    """
    Solves the fractional Riccati equation in the exponent of the characteristic function of the rough Heston model
    as described in El Euch and Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme.
    :param a: Argument of the characteristic function
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param A: Precomputed matrix of coefficients
    :param B: Precomputed matrix of coefficients
    :param N_Riccati: Number of time steps used
    :return: An approximation of the solution of the fractional Riccati equation. Is an array with N+1 values.
    """
    a_ = 0.5 * (-a * a - np.complex(0, 1) * a)
    b_ = (np.complex(0, 1) * a * rho * nu - lambda_)
    c_ = nu*nu/2

    def F(x):
        return a_ + b_ * x + c_*x*x

    h = np.zeros(shape=(len(a), N_Riccati + 1), dtype=np.cdouble)
    F_vec = np.zeros(shape=(len(a), N_Riccati + 1), dtype=np.cdouble)
    for k in range(1, N_Riccati + 1):
        h_p = F_vec[:, :k] @ B[:k, k]
        h[:, k] = F_vec[:, :k] @ A[:k, k] + A[k, k] * F(h_p)
        F_vec[:, k] = F(h[:, k])
    return h


def characteristic_function(a, H, lambda_, rho, nu, theta, V_0, T=1., N_Riccati=200):
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
    coefficient = dt ** (H + 0.5) / scipy.special.gamma(H + 2.5)
    A = np.zeros(shape=(N_Riccati + 1, N_Riccati + 1))
    A[0, 1:] = coefficient * (
                np.arange(N_Riccati) ** (H + 1.5) - (np.arange(N_Riccati) - H - 0.5) * np.arange(1, N_Riccati + 1) ** (
                    H + 0.5))
    for k in range(N_Riccati):
        A[1:(k + 1), k + 1] = coefficient * (
                (k - np.arange(1, k + 1) + 2) ** (H + 1.5) + (k - np.arange(1, k + 1)) ** (H + 1.5) - 2 * (
                k - np.arange(1, k + 1) + 1) ** (H + 1.5))
        A[k + 1, k + 1] = coefficient

    B = np.zeros(shape=(N_Riccati + 1, N_Riccati + 1))
    for k in range(N_Riccati):
        B[:(k + 1), k + 1] = (k - np.arange(k + 1) + 1) ** (H + 0.5) - (k - np.arange(k + 1)) ** (H + 0.5)
    B = dt ** (H + 0.5) / scipy.special.gamma(H + 1.5) * B

    temporary_weights = 1/scipy.special.gamma(1.5-H) * ((T - dt * np.arange(N_Riccati)) ** (0.5 - H) - (T - dt * np.arange(1, N_Riccati + 1)) ** (0.5 - H))
    fractional_weights = np.zeros(shape=(N_Riccati + 1))
    fractional_weights[0] = 0.5*temporary_weights[0]
    fractional_weights[1:-1] = 0.5*(temporary_weights[1:] + temporary_weights[:-1])
    fractional_weights[-1] = 0.5*temporary_weights[-1]
    h = solve_fractional_Riccati(a, lambda_, rho, nu, A, B, N_Riccati)
    fractional_integral = h @ fractional_weights
    integral = np.trapz(h, dx=dt)
    res = np.exp(theta*integral + V_0*fractional_integral)
    return res


def call(K, H, lambda_, rho, nu, theta, V_0, T=1., N_Riccati=200, R=2., L=50., N_Fourier=300):
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
    def mgf(u):
        """
        Moment generating function of the log-price.
        :param u: Argument
        :return: Value
        """
        return characteristic_function(np.complex(0, -1) * u, H, lambda_, rho, nu, theta, V_0, T, N_Riccati)

    return cf.pricing_fourier_inversion(mgf, K, R, L, N_Fourier)


def implied_volatility(K, H, lambda_, rho, nu, theta, V_0, T=1., N_Riccati=200, R=2., L=50., N_Fourier=300,
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
    prices = call(K, H, lambda_, rho, nu, theta, V_0, T, N_Riccati, R, L, N_Fourier)
    iv = cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
    prices = call(K, H, lambda_, rho, nu, theta, V_0, T, N_Riccati // 2, R, L / 1.2, N_Fourier // 2)
    iv_approx = cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
    error = np.amax(np.abs(iv_approx-iv)/iv)

    while error > rel_tol or np.amin(iv) < 1e-08:
        iv_approx = iv
        L = L * 1.2
        N_Fourier = N_Fourier * 2
        N_Riccati = N_Riccati * 2
        print('True', error, L, N_Fourier, N_Riccati)
        prices = call(K, H, lambda_, rho, nu, theta, V_0, T, N_Riccati, R, L, N_Fourier)
        iv = cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
        error = np.amax(np.abs(iv_approx - iv)/iv)

    return iv
