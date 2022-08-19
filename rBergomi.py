import numpy as np
from scipy.special import gamma, hyp2f1
import rBergomiBackbone
import psutil


def generate_samples(H, T, N, eta, V_0, S_0, rho, M):
    """
    Computes m final stock prices of the rough Bergomi model, using the approximation from Bayer, Friz, Gatheral.
    :param H: Hurst parameter
    :param T: Final time
    :param N: Number of time discretization steps
    :param eta: Volatility of volatility
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param M: Number of samples
    :return: An array of the final stock prices [S_T^1, S_T^2, ..., S_T^m]
    """

    def sqrt_cov_matrix():
        """
        Computes the Cholesky decomposition of the covariance matrix of
        (int_0^(T/N) (T/N - s)^(H-1/2) dW_s, ..., int_0^T (T-s)^(H-1/2) dW_s, W_(T/N), ..., W_T).
        :return: The Cholesky decomposition of the above covariance matrix
        """
        cov = np.empty(shape=(2 * N, 2 * N))
        times = np.arange(1, N + 1) * (T / N)
        minima = np.fmin(times[:, None], times[None, :])
        maxima = np.fmax(times[:, None], times[None, :])
        cov[0:N, 0:N] = gamma(H + 0.5) / gamma(H + 1.5) * minima ** (0.5 + H) * maxima ** (H - 0.5) \
            * hyp2f1(0.5 - H, 1, 1.5 + H, minima / maxima)
        cov[N:(2 * N), N:(2 * N)] = minima
        cov[0:N, N:(2 * N)] = 1 / (H + 0.5) \
            * (times[:, None] ** (H + 0.5) - np.fmax(times[:, None] - times[None, :], 0) ** (H + 0.5))
        cov[N:(2 * N), 0:N] = cov[0:N, N:(2 * N)].T
        return np.linalg.cholesky(cov)

    dt = T / N
    sqrt_cov = sqrt_cov_matrix()

    available_memory = np.sqrt(psutil.virtual_memory().available)  # sqrt to avoid overflows
    necessary_memory = np.sqrt(20) * np.sqrt(M) * np.sqrt(N) * np.sqrt(np.array([0.]).nbytes)
    rounds = int(np.ceil((necessary_memory / available_memory)**2))
    M_ = int(np.ceil(M / rounds))
    S = np.empty(shape=(M_ * rounds))
    for rd in range(rounds):
        W_vec = sqrt_cov @ np.random.normal(0, 1, (2 * N, M_))
        V = np.empty(shape=(N, M_))  # actual V is of shape (N+1, M), but we do not need the last one for S
        V[0, :] = V_0
        V[1:, :] = V_0 * np.exp(eta * np.sqrt(2 * H) * W_vec[:N - 1, :]
                                - eta ** 2 / 2 * (np.arange(1, N) * dt)[:, None] ** (2 * H))
        W_diff = np.empty(shape=(N, M_))
        W_diff[0, :] = W_vec[N, :]
        W_diff[1:, :] = W_vec[N + 1:, :] - W_vec[N:2 * N - 1, :]
        S_BM = rho * W_diff + np.random.normal(0, np.sqrt(1 - rho ** 2) * np.sqrt(dt), (N, M_))
        S[rd * M_:(rd + 1) * M_] = S_0 * np.exp(np.sum(np.sqrt(V) * S_BM - V * dt / 2, axis=0))
    return S[:M]


def implied_volatility(H=0.1, T=1., eta=1.9, V_0=0.235 ** 2, S_0=1., rho=-0.9, K=1., rel_tol=1e-03, verbose=0):
    """
    Computes the implied volatility of the European call option under the rough Bergomi model, using the approximation
    from Bayer, Friz, Gatheral.
    :param H: Hurst parameter
    :param T: Final time
    :param eta: Volatility of volatility
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param rho: Correlation between the Brownian motions driving the volatility and the stock
    :param K: The strike price
    :param rel_tol: Relative error tolerance
    :param verbose: Determines the number of intermediary results printed to the console
    :return: The implied volatility and a 95% confidence interval, in the form (estimate, lower, upper)
    """
    return rBergomiBackbone.iv_eur_call(sample_generator=lambda T_, N, M: generate_samples(H=H, T=T_, N=N, eta=eta,
                                                                                           V_0=V_0, S_0=S_0, rho=rho,
                                                                                           M=M),
                                        S_0=S_0, K=K, T=T, rel_tol=rel_tol, verbose=verbose)
