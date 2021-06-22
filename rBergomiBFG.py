import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, hyp2f1
import ComputationalFinance as cf


def sqrt_cov_matrix_rBergomi_BFG(H=0.1, T=1., N=1000):
    """
    Computes the Cholesky decomposition of the covariance matrix of
    (int_0^(T/N) (T/N - s)^(H-1/2) dW_s, ..., int_0^T (T-s)^(H-1/2) dW_s, W_(T/N), ..., W_T).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :return: The Cholesky decomposition of the above covariance matrix
    """
    cov = np.empty(shape=(2 * N, 2 * N))

    cov[0:N, 0:N] = gamma(H + 0.5) / gamma(H + 1.5) * \
                    np.array([[(np.fmin(i, j) * T / N) ** (0.5 + H) * (np.fmax(i, j) * T / N) ** (H - 0.5) *
                            hyp2f1(0.5 - H, 1., 1.5 + H, float(np.fmin(i, j)) / np.fmax(i, j))
                                for i in range(1, N + 1)] for j in range(1, N + 1)])
    cov[N:(2 * N), N:(2 * N)] = np.array([[np.fmin(i, j) * T / N for i in range(1, N + 1)] for j in range(1, N + 1)])
    cov[0:N, N:(2 * N)] = 1. / (H + 0.5) * np.array(
        [[(i * T / N) ** (H + 0.5) - (np.fmax(i - j, 0) * T / N) ** (H + 0.5) for j in range(1, N + 1)] for i in
         range(1, N + 1)])
    cov[N:(2 * N), 0:N] = 1. / (H + 0.5) * np.array(
        [[(i * T / N) ** (H + 0.5) - (np.fmax(i - j, 0) * T / N) ** (H + 0.5) for i in range(1, N + 1)] for j in
         range(1, N + 1)])
    return np.linalg.cholesky(cov)


def plot_rBergomi_BFG(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9):
    """
    Plots a path of the rough Bergomi model together with the variance process (approximation taken from
    Bayer, Friz, Gatheral, Pricing under rough volatility).
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the variance process and the stock price
    :return: Nothing, plots a realization of the stock price and the volatility
    """
    sqrt_cov = sqrt_cov_matrix_rBergomi_BFG(H, T, N)
    W_vec = sqrt_cov.dot(np.random.normal(0., 1., 2 * N))
    V = np.empty(shape=(N + 1,))
    V[0] = V_0
    V[1:] = V_0 * np.exp(
        eta * np.sqrt(2 * H) * W_vec[0:N] - eta * eta / 2 * np.array([(i * T / N) ** (2 * H) for i in range(1, N + 1)]))

    W_diff = np.empty(shape=(N,))
    W_diff[0] = W_vec[N]
    W_diff[1:] = W_vec[(N + 1):(2 * N)] - W_vec[N:(2 * N - 1)]
    W_2 = np.random.normal(0., T / N, N)
    S = np.empty(shape=(N + 1,))
    S[0] = S_0
    for i in range(1, N + 1):
        S[i] = S[i - 1] * np.exp(
            np.sqrt(V[i - 1]) * (rho * W_diff[i - 1] + np.sqrt(1 - rho ** 2) * W_2[i - 1]) - V[i - 1] * T / (2 * N))

    times = np.array([i * T / N for i in range(0, N + 1)])
    plt.plot(times, V)
    plt.plot(times, S)
    plt.show()


def rBergomi_BFG(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, m=1000, rounds=1):
    """
    Computes m final stock prices of the rough Bergomi model, using the approximation from Bayer, Friz, Gatheral.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta:
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param m: Number of samples
    :param rounds: Actually computes m*rounds samples, but only m at a time to avoid excessive memory usage.
    :return: An array of the final stock prices [S_T^1, S_T^2, ..., S_T^m]
    """
    dt = T / N
    sqrt_cov = sqrt_cov_matrix_rBergomi_BFG(H, T, N)
    S = np.empty(m * rounds)
    for rd in range(rounds):
        W_vec = sqrt_cov.dot(np.random.normal(0, 1, (2 * N, m)))
        V = np.empty(shape=(N + 1, m))
        V[0, :] = V_0
        V[1:, :] = V_0 * np.exp(
            eta * np.sqrt(2 * H) * W_vec[0:N, :] - eta * eta / 2 * np.array(
                [[(i * dt) ** (2 * H) for _ in range(0, m)] for i in range(1, N + 1)]))
        W_diff = np.empty(shape=(N, m))
        W_diff[0, :] = W_vec[N, :]
        W_diff[1:, :] = W_vec[(N + 1):(2 * N), :] - W_vec[N:(2 * N - 1), :]
        W_2 = np.random.normal(0, np.sqrt(dt), (N, m))
        S_ = S_0 * np.ones(shape=(m,))
        for i in range(N):
            S_ = S_ * np.exp(
                np.sqrt(V[i, :]) * (rho * W_diff[i, :] + np.sqrt(1 - rho ** 2) * W_2[i, :]) - V[i, :] * dt / 2)
        S[rd * m:(rd + 1) * m] = S_
    return S


def implied_volatility_call_rBergomi_BFG(H=0.1, T=1., N=1000, eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, K=1., m=1000,
                                         rounds=1):
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
    :param m: Number of samples
    :param rounds: Actually uses m*rounds samples, but only m at a time to avoid excessive memory usage
    :return: The implied volatility and a 95% confidence interval, in the form (estimate, lower, upper)
    """
    tic = time.perf_counter()
    samples = rBergomi_BFG(H, T, N, eta, V_0, S_0, rho, m, rounds)
    toc = time.perf_counter()
    print(f"Generating {m * rounds} rBergomi samples with N={N} takes {np.round(toc - tic, 2)} seconds.")
    return cf.volatility_smile_call(samples, K, T, S_0)
