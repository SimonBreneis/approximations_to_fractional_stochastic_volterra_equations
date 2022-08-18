import numpy as np
from scipy.special import gamma, hyp2f1
import ComputationalFinance as cf


def sqrt_cov_matrix(H, T, N=1000):
    """
    Computes the Cholesky decomposition of the covariance matrix of
    (int_0^(T/N) (T/N - s)^(H-1/2) dW_s, ..., int_0^T (T-s)^(H-1/2) dW_s, W_(T/N), ..., W_T).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :return: The Cholesky decomposition of the above covariance matrix
    """
    cov = np.empty(shape=(2 * N, 2 * N))
    times = np.arange(1, N+1) * (T / N)
    minima = np.fmin(times[:, None], times[None, :])
    maxima = np.fmax(times[:, None], times[None, :])
    cov[0:N, 0:N] = gamma(H + 0.5) / gamma(H + 1.5) * minima ** (0.5 + H) * maxima ** (H - 0.5) \
        * hyp2f1(0.5 - H, 1, 1.5 + H, minima / maxima)
    cov[N:(2 * N), N:(2 * N)] = minima
    cov[0:N, N:(2 * N)] = 1 / (H + 0.5) \
        * (times[:, None] ** (H + 0.5) - np.fmax(times[:, None] - times[None, :], 0) ** (H + 0.5))
    cov[N:(2 * N), 0:N] = cov[0:N, N:(2 * N)].T
    return np.linalg.cholesky(cov)


def generate_samples(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, M=1000, rounds=1):
    """
    Computes m final stock prices of the rough Bergomi model, using the approximation from Bayer, Friz, Gatheral.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param M: Number of samples
    :param rounds: Actually computes m*rounds samples, but only m at a time to avoid excessive memory usage.
    :return: An array of the final stock prices [S_T^1, S_T^2, ..., S_T^m]
    """
    dt = T / N
    sqrt_cov = sqrt_cov_matrix(H, T, N)
    S = np.empty(M * rounds)
    for rd in range(rounds):
        W_vec = sqrt_cov @ np.random.normal(0, 1, (2 * N, M))
        V = np.empty(shape=(N, M))  # actual V is of shape (N+1, M), but we do not need the last one for S
        V[0, :] = V_0
        V[1:, :] = V_0 * np.exp(eta * np.sqrt(2 * H) * W_vec[:N - 1, :]
                                - eta ** 2 / 2 * (np.arange(1, N) * dt)[:, None] ** (2 * H))
        W_diff = np.empty(shape=(N, M))
        W_diff[0, :] = W_vec[N, :]
        W_diff[1:, :] = W_vec[N + 1:, :] - W_vec[N:2 * N - 1, :]
        S_BM = rho * W_diff + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), (N, M))
        S[rd * M:(rd + 1) * M] = S_0 * np.exp(np.sum(np.sqrt(V) * S_BM - V * dt / 2, axis=0))
    return S


def implied_volatility(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, K=1., M=1000, rounds=1):
    """
    Computes the implied volatility of the European call option under the rough Bergomi model, using the approximation
    from Bayer, Friz, Gatheral.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param K: The strike price
    :param M: Number of samples
    :param rounds: Actually uses m*rounds samples, but only m at a time to avoid excessive memory usage
    :return: The implied volatility and a 95% confidence interval, in the form (estimate, lower, upper)
    """
    samples = generate_samples(H, T, N, eta, V_0, S_0, rho, M, rounds)
    return cf.iv_eur_call_MC(S_0, K, T, samples)
