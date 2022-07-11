import numpy as np
import scipy
from scipy import stats


def MC(samples):
    """
    Computes an approximation of E[X], where samples~X.
    :param samples: The samples of the random variable
    :return: The expectation and a 95% confidence interval, (expectation, confidence)
    """
    return np.average(samples, axis=-1), 1.95 * np.std(samples, axis=-1) / np.sqrt(samples.shape[-1])


def BS(sigma, T, N):
    """
    Simulates N samples of a Black-Scholes price at time T.
    :param sigma: Volatility
    :param T: Final time
    :param N: Number of samples
    :return: Array of final stock values
    """
    return np.exp(np.random.normal(-sigma*sigma/2*T, sigma*np.sqrt(T), N))


def BS_d1(S, K, sigma, T, r=0.):
    """
    Computes the first of the two nodes of the Black-Scholes model where the CDF is evaluated.
    :param S: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The first node
    """
    return (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def BS_d2(S, K, sigma, T, r=0.):
    """
    Computes the second of the two nodes of the Black-Scholes model where the CDF is evaluated.
    :param S: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The second node
    """
    return (np.log(S / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def BS_call_price(S, K, sigma, T, r=0.):
    """
    Computes the price of a European call option under the Black-Scholes model
    :param S: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The price of a call option
    """
    d1 = np.fmax(BS_d1(S=S, K=K, sigma=sigma, r=r, T=T), -6)  # to avoid overflow/underflow errors
    d2 = np.fmax(BS_d2(S=S, K=K, sigma=sigma, r=r, T=T), -6)
    return S * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)


def BS_put_price(S, K, sigma, T, r=0.):
    """
    Computes the price of a European put option under the Black-Scholes model
    :param S: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The price of a put option
    """
    d1 = BS_d1(S=S, K=K, sigma=sigma, r=r, T=T)
    d2 = BS_d2(S=S, K=K, sigma=sigma, r=r, T=T)
    return - S * scipy.stats.norm.cdf(-d1) + K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2)


def implied_volatility_call(S, K, T, price, r=0., tol=1e-10, sl=1e-10, sr=10.):
    """
    Computes the implied volatility of a European call option, assuming it is in [sl, sr].
    :param S: Initial stock price
    :param K: Strike price
    :param r: Drift
    :param T: Final time/maturity
    :param price: (Market) price of the option
    :param tol: Error tolerance in the approximation of the implied volatility
    :param sl: Left point of the search interval
    :param sr: Right point of the search interval
    :return: The implied volatility
    """
    sm = (sl + sr) / 2
    while np.amax(sr - sl) > tol:
        em = BS_call_price(S=S, K=K, sigma=sm, r=r, T=T) - price
        sl = (em < 0) * sm + (em >= 0) * sl
        sr = (em >= 0) * sm + (em < 0) * sr
        sm = (sl + sr) / 2
    return sm


def call_option_payoff(S, K):
    """
    Computes the payoff of a (European) call option.
    :param S: (Final) stock price
    :param K: Strike price
    :return: The payoff. If S and K are floats, a float is returned. If either S or K are floats, the other being a
            vector, a vector is returned. If both S and K are vectors, a matrix is returned. In the matrix, the rows
            have fixed K, the columns have fixed S.
    """
    if isinstance(K, float) or isinstance(S, float):
        return np.fmax(S - K, 0)

    S_matrix = np.repeat(np.array([S]), len(K), axis=0)
    K_matrix = np.repeat(np.array([K]), len(S), axis=0).transpose()
    return np.fmax(S_matrix - K_matrix, 0)


def put_option_payoff(S, K):
    """
    Computes the payoff of a (European) put option.
    :param S: (Final) stock price
    :param K: Strike price
    :return: The payoff. If S and K are floats, a float is returned. If either S or K are floats, the other being a
            vector, a vector is returned. If both S and K are vectors, a matrix is returned. In the matrix, the rows
            have fixed K, the columns have fixed S.
    """
    if isinstance(K, float) or isinstance(S, float):
        return np.fmax(K - S, 0)

    S_matrix = np.repeat(np.array([S]), len(K), axis=0)
    K_matrix = np.repeat(np.array([K]), len(S), axis=0).transpose()
    return np.fmax(K_matrix - S_matrix, 0)


def volatility_smile_call(samples, K, T, S):
    """
    Computes the volatility smile for a European call option.
    :param samples: The final stock prices
    :param K: The strike prices for which the implied volatilities should be calculated
    :param T: The final time
    :param S: The initial stock price
    :return: Three vectors: The implied volatility smile, and a lower and an upper bound on the volatility smile,
             so as to get 95% confidence intervals
    """
    (price_estimate, price_stat) = MC(call_option_payoff(S=samples, K=K))
    implied_volatility_estimate = implied_volatility_call(S=S, K=K, r=0, T=T, price=price_estimate)
    implied_volatility_lower = implied_volatility_call(S=S, K=K, r=0, T=T, price=price_estimate - price_stat)
    implied_volatility_upper = implied_volatility_call(S=S, K=K, r=0, T=T, price=price_estimate + price_stat)
    return implied_volatility_estimate, implied_volatility_lower, implied_volatility_upper


def fourier_transform_payoff(K, u):
    """
    Returns the value of the Fourier transform of the payoff of a European put or call option (same Fourier transform).
    Is a complex number.
    :param K: Strike price
    :param u: Argument of the Fourier transform
    :return: hat(f)(u)
    """
    u = complex(0, 1) * u
    return np.exp(np.log(K)*(1+u))/(u*(1+u))


def pricing_fourier_inversion(mgf, K, R=2., L=50., N=300):
    """
    Computes the option price using Fourier inversion.
    :param mgf: The moment generating function of the final log-price
    :param R: The (dampening) shift that we use
    :param K: The strike prices, assumed to be a vector
    :param L: The value at which we cut off the integral, so we do not integrate over the reals, but only over [-L, L]
    :param N: The number of points used in the trapezoidal rule for the approximation of the integral
    :return: The estimate of the option price
    """
    u = np.linspace(0, L, N+1)
    mgf_values = mgf(R-complex(0, 1)*u)
    prices = np.zeros(len(K))
    for i in range(len(K)):
        values = mgf_values*fourier_transform_payoff(K[i], u + complex(0, 1)*R)
        prices[i] = np.real(1/np.pi * np.trapz(values, dx=L/N))
    return prices
