import numpy as np
from scipy.stats import norm
from scipy import integrate


def maturity_tensor_strike(S_0, K, T):
    """
    Given a numpy array of strikes K for maturity T[-1], computes appropriate array of strikes for all other
    maturities in T for the computation of an implied volatility surface.
    :param S_0: Initial stock price
    :param K: Strikes for maturity T[-1]
    :param T: Array of maturities
    :return: Two 2-dim arrays of maturities and and strikes of the form T=T(T, K) and K=K(T, K)
    """
    return np.tile(T, (len(K), 1)).T, S_0 * np.exp(np.multiply.outer(np.sqrt(T / T[-1]), np.log(K / S_0)))


def MC(samples):
    """
    Computes an approximation of E[X], where samples~X.
    :param samples: The samples of the random variable
    :return: The expectation and a 95% confidence interval, (expectation, confidence)
    """
    return np.average(samples, axis=-1), 1.95 * np.std(samples, axis=-1) / np.sqrt(samples.shape[-1])


def BS_samples(sigma, T, N):
    """
    Simulates N samples of a Black-Scholes price at time T.
    :param sigma: Volatility
    :param T: Final time
    :param N: Number of samples
    :return: Array of final stock values
    """
    return np.exp(np.random.normal(-sigma*sigma/2*T, sigma*np.sqrt(T), N))


def BS_nodes(S_0, K, sigma, T, r=0., regularize=True):
    """
    Computes the two nodes of the Black-Scholes model where the CDF is evaluated.
    :param S_0: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :param regularize: Ensures that the results are in the interval [-30, 30].
    :return: The first node
    """
    d1 = (np.log(S_0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    if regularize:
        d1 = np.fmax(np.fmin(d1, 30), -30)
        d2 = np.fmax(np.fmin(d2, 30), -30)
    return d1, d2


def BS_price_eur_call(S_0, K, sigma, T, r=0.):
    """
    Computes the price of a European call option under the Black-Scholes model.
    :param S_0: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The price of a call option
    """
    d1, d2 = BS_nodes(S_0=S_0, K=K, sigma=sigma, T=T, r=r, regularize=True)
    return S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BS_price_eur_put(S_0, K, sigma, T, r=0.):
    """
    Computes the price of a European put option under the Black-Scholes model.
    :param S_0: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :return: The price of a put option
    """
    d1, d2 = BS_nodes(S_0=S_0, K=K, sigma=sigma, T=T, r=r, regularize=True)
    return - S_0 * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2)


def iv(BS_price_fun, price, tol=1e-10, sl=1e-10, sr=10.):
    """
    Computes the implied volatility of an option given its price, assuming the volatility is in [sl, sr].
    :param BS_price_fun: A function of the volatility sigma that returns the corresponding Black-Scholes price
    :param price: (Market) price of the option
    :param tol: Error tolerance in the approximation of the implied volatility
    :param sl: Left point of the search interval
    :param sr: Right point of the search interval
    :return: The implied volatility
    """
    threshold = sl + tol
    sm = (sl + sr) / 2
    while np.amax(sr - sl) > tol:
        em = BS_price_fun(sm) - price
        sl = (em < 0) * sm + (em >= 0) * sl
        sr = (em >= 0) * sm + (em < 0) * sr
        sm = (sl + sr) / 2
    return np.where(sm < threshold, np.nan, sm)


def iv_eur_call(S_0, K, T, price, r=0.):
    """
    Computes the implied volatility of a European call option given its price.
    :param S_0: Initial stock price
    :param K: Strike price
    :param r: Drift
    :param T: Final time/maturity
    :param price: (Market) price of the option
    :return: The implied volatility
    """
    return iv(BS_price_fun=lambda s: BS_price_eur_call(S_0=S_0, K=K, sigma=s, r=r, T=T), price=price)


def iv_eur_put(S_0, K, T, price, r=0.):
    """
    Computes the implied volatility of a European put option given its price.
    :param S_0: Initial stock price
    :param K: Strike price
    :param r: Drift
    :param T: Final time/maturity
    :param price: (Market) price of the option
    :return: The implied volatility
    """
    return iv(BS_price_fun=lambda s: BS_price_eur_put(S_0=S_0, K=K, sigma=s, r=r, T=T), price=price)


def payoff_call(S, K):
    """
    Computes the payoff of a (European) call option.
    If S or K are a float, simply computes the payoff.
    If both S and K are 1-dim arrays, computes the payoff matrix payoff(K, S), a 2-dim array. This may be the case when
    computing payoffs for various strikes using MC, as in iv smiles.
    If both S=S(T, .) and K=K(T, .) are 2-dim arrays, computes the payoff matrix payoff(T, K, S), a 3-dim array. This
    may be the case when computing payoffs for various strikes and maturities using MC, as in iv surfaces.
    :param S: (Final) stock price
    :param K: Strike price
    :return: The payoff.
    """
    if not isinstance(K, np.ndarray) or not isinstance(S, np.ndarray):
        return np.fmax(S - K, 0)
    return np.fmax(S[..., None, :] - K[..., None], 0)


def payoff_put(S, K):
    """
    Computes the payoff of a (European) put option.
    If S or K are a float, simply computes the payoff.
    If both S and K are 1-dim arrays, computes the payoff matrix payoff(K, S), a 2-dim array. This may be the case when
    computing payoffs for various strikes using MC, as in iv smiles.
    If both S=S(T, .) and K=K(T, .) are 2-dim arrays, computes the payoff matrix payoff(T, K, S), a 3-dim array. This
    may be the case when computing payoffs for various strikes and maturities using MC, as in iv surfaces.
    :param S: (Final) stock price
    :param K: Strike price
    :return: The payoff.
    """
    if isinstance(K, float) or isinstance(S, float):
        return np.fmax(K - S, 0)
    if len(K.shape) == 1 and len(S.shape) == 1:
        return np.fmax(K[:, None] - S[None, :], 0)
    return np.fmax(K[:, :, None] - S[:, None, :], 0)


def iv_eur_call_MC(S_0, K, T, samples):
    """
    Computes the volatility smile for a European call option given samples of final stock prices.
    :param S_0: The initial stock price
    :param K: The strike prices for which the implied volatilities should be calculated
    :param T: The final time
    :param samples: The final stock prices
    :return: Three numpy arrays: The implied volatility smile, and a lower and an upper bound on the volatility smile,
             so as to get 95% confidence intervals
    """
    price_estimate, price_stat = MC(payoff_call(S=samples, K=K))
    implied_volatility_estimate = iv_eur_call(S_0=S_0, K=K, r=0, T=T, price=price_estimate)
    implied_volatility_lower = iv_eur_call(S_0=S_0, K=K, r=0, T=T, price=price_estimate - price_stat)
    implied_volatility_upper = iv_eur_call(S_0=S_0, K=K, r=0, T=T, price=price_estimate + price_stat)
    return implied_volatility_estimate, implied_volatility_lower, implied_volatility_upper


def iv_eur_put_MC(S_0, K, T, samples):
    """
    Computes the volatility smile for a European call option given samples of final stock prices.
    :param samples: The final stock prices
    :param K: The strike prices for which the implied volatilities should be calculated
    :param T: The final time
    :param S_0: The initial stock price
    :return: Three numpy arrays: The implied volatility smile, and a lower and an upper bound on the volatility smile,
             so as to get 95% confidence intervals
    """
    price_estimate, price_stat = MC(payoff_put(S=samples, K=K))
    implied_volatility_estimate = iv_eur_put(S_0=S_0, K=K, r=0, T=T, price=price_estimate)
    implied_volatility_lower = iv_eur_put(S_0=S_0, K=K, r=0, T=T, price=price_estimate - price_stat)
    implied_volatility_upper = iv_eur_put(S_0=S_0, K=K, r=0, T=T, price=price_estimate + price_stat)
    return implied_volatility_estimate, implied_volatility_lower, implied_volatility_upper


def fourier_payoff_call_put(K, u):
    """
    Returns the value of the Fourier transform of the payoff of a put or call option (same Fourier transform).
    Is a complex number.
    :param K: Strike price, may also be a numpy array
    :param u: Argument of the Fourier transform
    :return: hat(f)(u) (or hat(f)(K, u) if K is a numpy array)
    """
    u = complex(0, 1) * u
    if isinstance(K, np.ndarray):
        return np.exp(np.multiply.outer(np.log(K), 1+u)) / (u * (1+u))
    return np.exp(np.log(K)*(1+u))/(u*(1+u))


def price_eur_call_fourier(mgf, K, R=2., L=50., N=300):
    """
    Computes the option price of an European call option using Fourier inversion.
    :param mgf: The moment generating function of the final log-price, a function of the Fourier argument only
    :param R: The (dampening) shift that we use
    :param K: The strike prices, assumed to be a numpy array
    :param L: The value at which we cut off the integral, so we do not integrate over the reals, but only over [-L, L]
    :param N: The number of points used in the trapezoidal rule for the approximation of the integral
    :return: The estimate of the option price
    """
    x = np.linspace(0, L, N + 1)
    y = np.zeros(N+1)
    y[:N] = x[1:] - x[:-1]
    y[1:] += x[1:] - x[:-1]
    y /= 2
    mgf_output = np.empty(len(x), dtype=np.cdouble)
    total_rounds = 1
    current_round = 0
    while current_round < total_rounds:
        n_inputs = int(np.ceil(len(x) / total_rounds))
        try:
            mgf_output[current_round * n_inputs:(current_round + 1) * n_inputs] = \
                mgf(R - complex(0, 1) * x[current_round * n_inputs:(current_round + 1) * n_inputs])
        except MemoryError:
            if total_rounds < len(x):
                total_rounds = total_rounds * 2
                current_round = current_round * 2 - 1
            else:
                raise MemoryError('Not enough memory to carry out Fourier inversion.')
        current_round = current_round + 1
    return np.real(fourier_payoff_call_put(K, x + complex(0, 1) * R) @ (mgf_output * y)) / np.pi


def iv_eur_call_fourier(mgf, S_0, K, T, r=0., R=2., L=50., N=300):
    """
    Computes the implied volatility of an European call option using Fourier inversion.
    :param mgf: The moment generating function of the final log-price, a function of the Fourier argument only
    :param S_0: Initial stock price
    :param T: Maturity
    :param r: Drift
    :param R: The (dampening) shift that we use
    :param K: The strike prices, assumed to be a numpy array
    :param L: The value at which we cut off the integral, so we do not integrate over the reals, but only over [-L, L]
    :param N: The number of points used in the trapezoidal rule for the approximation of the integral
    :return: The estimate of the option price
    """
    return iv_eur_call(S_0=S_0, K=K, T=T, price=price_eur_call_fourier(mgf=mgf, K=K, R=R, L=L, N=N), r=r)
