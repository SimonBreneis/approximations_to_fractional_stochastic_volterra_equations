import math
import numpy as np
import ComputationalFinance as cf


def solve_fractional_Riccati(a, H, lambda_, rho, nu, T=1., N_Riccati=1000):
    """
    Solves the fractional Riccati equation in the exponent of the characteristic function of the rough Heston model
    as described in El Euch and Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme.
    :param a: Argument of the characteristic function
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param T: Final time
    :param N_Riccati: Number of time steps used
    :return: An approximation of the solution of the fractional Riccati equation. Is an array with N+1 values.
    """

    def F(x):
        return 0.5 * (-a * a - complex(0, 1) * a) + lambda_ * (complex(0, 1) * a * rho * nu - 1) * x + (
                    lambda_ * nu) ** 2 / 2 * x ** 2

    dt = T / N_Riccati
    coefficient = dt ** (H + 0.5) / math.gamma(H + 2.5)
    A = np.zeros(shape=(N_Riccati + 1, N_Riccati + 1))
    A[0, 1:] = coefficient * (np.arange(N_Riccati) ** (H + 1.5) - (np.arange(N_Riccati) - H - 0.5) * np.arange(1, N_Riccati + 1) ** (H + 0.5))
    for k in range(N_Riccati):
        A[1:(k + 1), k + 1] = coefficient * (
                    (k - np.arange(1, k + 1) + 2) ** (H + 1.5) + (k - np.arange(1, k + 1)) ** (H + 1.5) - 2 * (
                        k - np.arange(1, k + 1) + 1) ** (H + 1.5))
        A[k + 1, k + 1] = coefficient

    B = np.zeros(shape=(N_Riccati + 1, N_Riccati + 1))
    for k in range(N_Riccati):
        B[:(k + 1), k + 1] = dt ** (H + 0.5) / math.gamma(H + 1.5) * (
                    (k - np.arange(k + 1) + 1) ** (H + 0.5) - (k - np.arange(k + 1)) ** (H + 0.5))

    h = np.zeros(shape=(N_Riccati + 1), dtype=complex)
    F_vec = np.zeros(shape=(N_Riccati + 1), dtype=complex)
    for k in range(1, N_Riccati + 1):
        h_p = np.dot(B[:k, k], F_vec[:k])
        h[k] = np.dot(A[:k, k], F_vec[:k]) + A[k, k] * F(h_p)
        F_vec[k] = F(h[k])
    return h


def characteristic_function(a, H, lambda_, rho, nu, theta, V_0, T=1., N_Riccati=1000):
    """
    Gives the characteristic function of the log-price in the rough Heston model
    as described in El Euch and Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme.
    :param a: Argument of the characteristic function (assumed to be a vector)
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used
    :return: The characteristic function
    """
    res = np.zeros(shape=(len(a)), dtype=complex)
    for i in range(len(a)):
        dt = T / N_Riccati
        h = solve_fractional_Riccati(a[i], H, lambda_, rho, nu, T, N_Riccati)
        temporary_weights = 1/math.gamma(1.5-H) * ((T - dt * np.arange(N_Riccati)) ** (0.5 - H) - (T - dt * np.arange(1, N_Riccati + 1)) ** (0.5 - H))
        fractional_weights = np.zeros(shape=(N_Riccati + 1))
        fractional_weights[0] = 0.5*temporary_weights[0]
        fractional_weights[1:-1] = 0.5*(temporary_weights[1:] + temporary_weights[:-1])
        fractional_weights[-1] = 0.5*temporary_weights[-1]
        fractional_integral = np.dot(fractional_weights, h)
        integral = np.trapz(h, dx=dt)
        res[i] = np.exp(theta*lambda_*integral + V_0*fractional_integral)
    return res


def call(K, H, lambda_, rho, nu, theta, V_0, T=1., N_Riccati=1000, R=2., L=200., N_fourier=40000):
    """
    Gives the price of a European call option in the rough Heston model
    as described in El Euch and Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme.
    Uses Fourier inversion.
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
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals,
    but only over [-L, L]
    :param N_fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    return: The price of the call option
    """
    def mgf(u):
        """
        Moment generating function of the log-price.
        :param u: Argument
        :return: Value
        """
        return characteristic_function(complex(0, -1) * u, H, lambda_, rho, nu, theta, V_0, T, N_Riccati)

    return cf.pricing_fourier_inversion(mgf, K, R, L, N_fourier)


def implied_volatility(K, H, lambda_, rho, nu, theta, V_0, T=1., N_Riccati=1000, R=2., L=200., N_fourier=40000):
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
    :param N_Riccati: Number of time steps used in the solution of the fractional Riccati equation
    :param R: The (dampening) shift that we use for the Fourier inversion
    :param L: The value at which we cut off the Fourier integral, so we do not integrate over the reals,
    but only over [-L, L]
    :param N_fourier: The number of points used in the trapezoidal rule for the approximation of the Fourier integral
    return: The price of the call option
    """
    prices = call(K, H, lambda_, rho, nu, theta, V_0, T, N_Riccati, R, L, N_fourier)
    return cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
