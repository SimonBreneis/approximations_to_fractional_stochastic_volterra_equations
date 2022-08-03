import numpy as np
from scipy.special import gamma
import rHestonBackbone as backbone


def characteristic_function(a, S, H, lambda_, rho, nu, theta, V_0, T, N_Riccati=200):
    """
    Gives the characteristic function of the log-price in the rough Heston model as described in El Euch and Rosenbaum,
    The characteristic function of rough Heston models. Uses the Adams scheme.
    :param a: Argument of the characteristic function (assumed to be a vector)
    :param S: Initial stock price
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
    return np.exp(complex(0, 1) * a * np.log(S) + theta * integral + V_0 * fractional_integral)


def iv_eur_call(S, K, H, lambda_, rho, nu, theta, V_0, T, rel_tol=1e-03):
    """
    Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
    Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
    :param S: Initial stock price
    :param K: Strike price, assumed to be a vector
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param rel_tol: Required maximal relative error in the implied volatility
    return: The price of the call option
    """
    return backbone.iv_eur_call(char_fun=lambda u, T_, N_: characteristic_function(u, S, H, lambda_, rho, nu, theta,
                                                                                   V_0, T_, N_),
                                S=S, K=K, T=T, rel_tol=rel_tol)
