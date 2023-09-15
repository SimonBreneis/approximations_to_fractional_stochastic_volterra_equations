import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


def MC(samples):
    """
    Computes an approximation of E[X], where samples~X.
    :param samples: The samples of the random variable
    :return: The expectation and a 95% confidence interval, (expectation, confidence)
    """
    return np.average(samples, axis=-1), 1.95 * np.std(samples, axis=-1) / np.sqrt(samples.shape[-1])


def rand_normal(loc=0, scale=1, size=1, antithetic=False):
    if antithetic:
        rv = np.empty(size)
        rv[:size // 2] = np.random.normal(loc, scale, size // 2)
        if size % 2 == 0:
            rv[size // 2:] = -rv[:size // 2]
        else:
            rv[size // 2:-1] = -rv[:size // 2]
            rv[-1] = np.random.normal(loc, scale, 1)
    else:
        rv = np.random.normal(loc, scale, size)
    return rv


def rand_uniform(size=1, antithetic=False):
    if antithetic:
        rv = np.empty(size)
        rv[:size // 2] = np.random.uniform(0, 1, size // 2)
        if size % 2 == 0:
            rv[size // 2:] = 1 - rv[:size // 2]
        else:
            rv[size // 2:-1] = 1 - rv[:size // 2]
            rv[-1] = np.random.uniform(0, 1, 1)
    else:
        rv = np.random.uniform(size)
    return rv


def BS_samples(sigma, T, N, r=0., antithetic=True):
    """
    Simulates N samples of a Black-Scholes price at time T.
    :param sigma: Volatility
    :param T: Final time
    :param N: Number of samples
    :param r: Interest rate
    :param antithetic: If True, uses antithetic variates
    :return: Array of final stock values
    """
    return np.exp(rand_normal(loc=(r - sigma ** 2 / 2) * T, scale=sigma * np.sqrt(T), size=N, antithetic=antithetic))


def BS_paths(sigma, T, n, m, r=0., antithetic=True):
    """
    Simulates m samples of Black-Scholes paths at n + 1 equidistant times.
    :param sigma: Volatility
    :param T: Final time
    :param n: Number of time steps
    :param m: Number of samples
    :param r: Interest rate
    :param antithetic: If True, uses antithetic variates
    :return: Array of samples of shape (m, n + 1)
    """
    if antithetic:
        BM_increments = np.empty((m, n))
        BM_increments[:m // 2, :] = np.random.normal(0, sigma * np.sqrt(T / n), (m // 2, n))
        BM_increments[m // 2:, :] = - BM_increments[:m // 2, :]
    else:
        BM_increments = np.random.normal(0, sigma * np.sqrt(T / n), (m, n))
    BM_samples = np.zeros((m, n + 1))
    BM_samples[:, 1:] = np.cumsum(BM_increments, axis=1)
    return np.exp(BM_samples + (r - 0.5 * sigma ** 2) * np.linspace(0, T, n + 1)[None, :])


def BS_price_eur_call_put(S_0, K, sigma, T, r=0., call=True, digital=False):
    """
    Computes the price of a (digital or standard) European call or put option under the Black-Scholes model.
    :param S_0: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param r: Drift
    :param T: Final time
    :param call: If True, prices a call option. Else prices a put option
    :param digital: If True, prices a digital option. Else prices a standard option
    :return: The price of a call option
    """
    T = T[..., None] if (isinstance(T, np.ndarray) and len(T.shape) == 1) else T

    def BS_nodes(regularize=True):
        """
        Computes the two nodes of the Black-Scholes model where the CDF is evaluated.
        :param regularize: Ensures that the results are in the interval [-30, 30] to avoid overflow/underflow errors.
        :return: The nodes
        """
        d1_ = (np.log(S_0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2_ = (np.log(S_0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        if regularize:
            d1_ = np.fmax(np.fmin(d1_, 30), -30)
            d2_ = np.fmax(np.fmin(d2_, 30), -30)
        return d1_, d2_

    if digital:
        put_price = norm.cdf((np.log(K / S_0) + sigma ** 2 * T / 2 - r * T) / (sigma * np.sqrt(T)))
        if call:
            return 1 - put_price
        else:
            return put_price
    d1, d2 = BS_nodes()
    if call:
        return S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return - S_0 * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2)


def BS_price_geom_asian_call(S_0, K, sigma, T):
    """
    Computes the price of a geometric Asian call option under the Black-Scholes model.
    :param S_0: Initial stock price
    :param K: Strike price
    :param sigma: Volatility
    :param T: Final time
    :return: The price of a call option
    """
    return BS_price_eur_call_put(S_0=S_0 * np.exp(-sigma ** 2 * T / 12), K=K, sigma=sigma / np.sqrt(3), T=T, r=0.,
                                 call=True, digital=False)


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


def iv_eur(S_0, K, T, price, payoff, r=0., stat=None):
    """
    Computes the implied volatility of a European call or put option given its price.
    :param S_0: Initial stock price
    :param K: Strike price
    :param r: Drift
    :param T: Final time/maturity
    :param price: (Market) price of the option
    :param payoff: Either 'call' or 'put' or a payoff function taking as input S (final stock price samples) and K
        (strike or other parameters), and returning the payoffs for all the stock prices S
    :param stat: Can specify a MC error (statistical variance interval). In this case, returns the implied volatility as
        well as the corresponding MC interval
    :return: The implied volatility
    """
    T = T[..., None] if (isinstance(T, np.ndarray) and len(T.shape) == 1) else T

    if payoff == 'call' or payoff == payoff_call:
        def price_fun(s):
            return BS_price_eur_call_put(S_0=S_0, K=K, sigma=s, r=r, T=T, call=True, digital=False)
    elif payoff == 'digital call':
        def price_fun(s):
            return BS_price_eur_call_put(S_0=S_0, K=K, sigma=s, r=r, T=T, call=True, digital=True)
    elif payoff == 'put' or payoff == payoff_put:
        def price_fun(s):
            return BS_price_eur_call_put(S_0=S_0, K=K, sigma=s, r=r, T=T, call=False, digital=False)
    elif payoff == 'digital put':
        def price_fun(s):
            return BS_price_eur_call_put(S_0=S_0, K=K, sigma=s, r=r, T=T, call=False, digital=True)
    else:
        if isinstance(T, np.ndarray):
            normal_rv = np.sqrt(T) * np.random.normal(0, 1, (T.shape[0], 1000000))
        else:
            normal_rv = np.sqrt(T) * np.random.normal(0, 1, 1000000)

        def price_fun(s):
            return np.average(np.exp(-r * T) * payoff(S=np.exp(s * normal_rv + (r - 0.5 * s ** 2) * T), K=K), axis=-1)
    if stat is None:
        return iv(BS_price_fun=price_fun, price=price)
    return iv(BS_price_fun=price_fun, price=price), iv(BS_price_fun=price_fun, price=price - stat), \
        iv(BS_price_fun=price_fun, price=price + stat)


def iv_geom_asian_call(S_0, K, T, price):
    """
    Computes the implied volatility of a geometric Asian call option given its price.
    :param S_0: Initial stock price
    :param K: Strike price
    :param T: Final time/maturity
    :param price: (Market) price of the option
    :return: The implied volatility
    """
    return iv(BS_price_fun=lambda s: BS_price_geom_asian_call(S_0=S_0, K=K, sigma=s, T=T), price=price)


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
    if not isinstance(K, np.ndarray) or not isinstance(S, np.ndarray):
        return np.fmax(K - S, 0)
    return np.fmax(K[..., None] - S[..., None, :], 0)


def eur_MC(S_0, K, T, samples, r=0., payoff="call", antithetic=False, implied_vol=False):
    """
    Computes the price or the implied volatility for a European option given samples of final stock prices.
    :param S_0: The initial stock price
    :param K: The strike prices for which the implied volatilities should be calculated. 1d numpy array if T is float,
        2d numpy array if T is 1d numpy array (of shape (len(T), n))
    :param T: The final time. Float or 1d numpy array
    :param samples: The final stock prices. Numpy array of shape (K.shape, m)
    :param r: Interest rate
    :param payoff: Either a payoff function with two parameters S (for the samples) and K (for the strike or additional
        parameters), or one of the strings 'put' and 'call'
    :param antithetic: If True, the samples are antithetic, with the first half corresponding to the second half
    :param implied_vol: If True, returns the implied volatility, else returns the prices
    :return: Three numpy arrays: The implied volatility smile, and a lower and an upper bound on the volatility smile,
             so as to get 95% confidence intervals
    """
    if payoff == 'call':
        payoff = payoff_call
    elif payoff == 'put':
        payoff = payoff_put
    if antithetic:
        payoffs = 0.5 * (payoff(S=samples[..., :len(samples) // 2], K=K)
                         + payoff(S=samples[..., len(samples) // 2:], K=K))
    else:
        payoffs = payoff(S=samples, K=K)
    price_estimate, price_stat = MC(np.exp(-r * T)[..., None, None] * payoffs)
    if implied_vol:
        return iv_eur(S_0=S_0, K=K, r=r, T=T, price=price_estimate, payoff=payoff, stat=price_stat)
    return price_estimate, price_estimate - price_stat, price_estimate + price_stat


def price_geom_asian_call_MC(K, samples, antithetic=False):
    """
    Computes the prices for geometric Asian call options given samples of the stock price processes.
    :param K: The strike prices of the call options
    :param samples: The stock price paths
    :param antithetic: If True, the samples are antithetic, with the first half corresponding to the second half
    :return: Three numpy arrays: The prices, and lower and upper confidence interval bounds
    """
    geom_avg_prices = np.exp(np.trapz(np.log(samples), dx=1 / (samples.shape[-2] - 1), axis=-2))
    if antithetic:
        payoffs = 0.5 * (payoff_call(S=geom_avg_prices[..., :len(geom_avg_prices) // 2], K=K)
                         + payoff_call(S=geom_avg_prices[..., len(geom_avg_prices) // 2:], K=K))
    else:
        payoffs = payoff_call(S=geom_avg_prices, K=K)
    price_estimate, price_stat = MC(payoffs)
    return price_estimate, price_estimate - price_stat, price_estimate + price_stat


def price_avg_vol_call_MC(K, samples, antithetic=False):
    """
    Computes the prices for call options on the average volatility given samples of the stock price processes.
    :param K: The strike prices of the call options
    :param samples: The volatility paths
    :param antithetic: If True, the samples are antithetic, with the first half corresponding to the second half
    :return: Three numpy arrays: The prices, and lower and upper confidence interval bounds
    """
    avg_vol = np.trapz(samples, dx=1 / (samples.shape[-2] - 1), axis=-2)
    if antithetic:
        payoffs = 0.5 * (payoff_call(S=avg_vol[..., :len(avg_vol) // 2], K=K)
                         + payoff_call(S=avg_vol[..., len(avg_vol) // 2:], K=K))
    else:
        payoffs = payoff_call(S=avg_vol, K=K)
    price_estimate, price_stat = MC(payoffs)
    return price_estimate, price_estimate - price_stat, price_estimate + price_stat


def fourier_payoff_call_put(K, u, call=True, digital=False, logarithmic=True):
    """
    Returns the value of the Fourier transform of the payoff of a (digital or standard) put or call option.
    Is a complex number.
    :param K: Strike price, may also be a numpy array
    :param u: Argument of the Fourier transform
    :param call: If True, returns the Fourier transform of a call option. If False, of a put option
    :param digital: If True, returns the Fourier transform of a digital option. If False, of a standard option
    :param logarithmic: If True, assumes that the payoff is a function of the log-stock price. If False, assumes it is
        a function of the (normal) stock price
    :return: hat(f)(u) (or hat(f)(K, u) if K is a numpy array)
    """
    u = complex(0, 1) * u
    if digital:
        sign = -1 if call else 1
        if logarithmic:
            if isinstance(K, np.ndarray):
                return sign * np.exp(np.multiply.outer(np.log(K), u)) / u
            return sign * np.exp(np.log(K) * u) / u
        if isinstance(K, np.ndarray):
            return sign * np.exp(np.multiply.outer(K, u)) / u
        return sign * np.exp(K * u) / u
    if logarithmic:
        if isinstance(K, np.ndarray):
            return np.exp(np.multiply.outer(np.log(K), 1 + u)) / (u * (1 + u))
        return np.exp(np.log(K) * (1 + u)) / (u * (1 + u))
    if isinstance(K, np.ndarray):
        return np.exp(np.multiply.outer(K, u)) / u ** 2
    return np.exp(K * u) / u ** 2


def price_eur_call_put_fourier(mgf, K, r=0., T=1., R=2., L=50., N=300, log_price=True, call=True, digital=False):
    """
    Computes the option price of a (digital or standard) European call or put option using Fourier inversion.
    :param mgf: The moment generating function of the log-variable that should be priced (e.g. the final log-price, or
        the final log-average), a function of the Fourier argument only
    :param R: The (dampening) shift that we use
    :param K: The strike prices, assumed to be a numpy array
    :param r: Interest rate
    :param T: Maturity (only needed for discounting, i.e. if r != 0)
    :param L: The value at which we cut off the integral, so we do not integrate over the reals, but only over [-L, L]
    :param N: The number of points used in the trapezoidal rule for the approximation of the integral
    :param log_price: If True, assumes that the mgf is the mgf of the log-price. If False, assumes it is the mgf of the
        price (without the logarithm)
    :param call: If True, returns the call price. Else, returns the put price
    :param digital: If True, computes the price of a digital option. Else, of a standard option
    :return: The estimate of the option price
    """
    if call:
        R = np.abs(R)
    else:
        R = - np.abs(R)
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
    fourier_payoff = fourier_payoff_call_put(K=K, u=x + complex(0, 1) * R, call=call, digital=digital,
                                             logarithmic=log_price)
    return np.exp(-r * T) / np.pi * np.real(fourier_payoff @ (mgf_output * y))


def iv_eur_call_put_fourier(mgf, S_0, K, T, r=0., R=2., L=50., N=300, call=True, digital=False):
    """
    Computes the implied volatility of a (digital or standard) European call or put option using Fourier inversion.
    :param mgf: The moment generating function of the final log-price, a function of the Fourier argument only
    :param S_0: Initial stock price
    :param T: Maturity
    :param r: Drift
    :param R: The (dampening) shift that we use
    :param K: The strike prices, assumed to be a numpy array
    :param L: The value at which we cut off the integral, so we do not integrate over the reals, but only over [-L, L]
    :param N: The number of points used in the trapezoidal rule for the approximation of the integral
    :param call: If True, computes the implied volatility of a call option. Else, of a put option
    :param digital: If True, computes the implied volatility of a digital option. Else, of a standard option
    :return: The estimate of the implied volatility
    """
    payoff = 'call'
    if digital:
        if call:
            payoff = 'digital call'
        else:
            payoff = 'digital put'
    return iv_eur(S_0=S_0, K=K, T=T, price=price_eur_call_put_fourier(mgf=mgf, K=K, r=r, T=T, R=R, L=L, N=N, call=True),
                  r=r, payoff=payoff)


def iv_geom_asian_call_fourier(mgf, S_0, K, T, R=2., L=50., N=300):
    """
    Computes the implied volatility of a geometric Asian call option using Fourier inversion.
    :param mgf: The moment generating function of the final log-price, a function of the Fourier argument only
    :param S_0: Initial stock price
    :param T: Maturity
    :param R: The (dampening) shift that we use
    :param K: The strike prices, assumed to be a numpy array
    :param L: The value at which we cut off the integral, so we do not integrate over the reals, but only over [-L, L]
    :param N: The number of points used in the trapezoidal rule for the approximation of the integral
    :return: The estimate of the option price
    """
    return iv_geom_asian_call(S_0=S_0, K=K, T=T, price=price_eur_call_put_fourier(mgf=mgf, K=K, R=R, L=L, N=N))


def price_am(T, r, samples, payoff, features, antithetic=False, varying_initial_conditions=False):
    """
    Prices American options.
    :param T: Maturity
    :param r: Interest rate
    :param samples: Numpy array of shape (d, m, N + 1), where N is the number of time steps, m the number of samples,
        and d is the dimension of the Markovian process. It is assumed that the first dimension is the stock price
        process
    :param payoff: The payoff function, a function of S (sample values) only
    :param features: A function turning samples at a specific time step into features
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :param varying_initial_conditions: If True, assumes that the initial condition varies with samples. Otherwise,
        assumes that the initial condition is the same for all samples
    :return: The price of the American option and a list of the models for each exercise time that approximate the
        future payoff
    """
    N_time = samples.shape[-1] - 1
    m = samples.shape[1]
    dt = T / N_time
    d = samples.shape[0]

    discount = np.exp(-r * dt)
    models = [LinearRegression() for _ in range(N_time)]
    discounted_future_payoffs = payoff(S=samples[0, :, -1])
    for i in range(N_time - 1, -1 if varying_initial_conditions else 0, -1):
        discounted_future_payoffs = discount * discounted_future_payoffs
        current_payoff = payoff(S=samples[0, :, i])
        active_indices = current_payoff > 0
        if np.sum(active_indices) > 1:
            ft = features(samples[:, active_indices, i])
            models[i] = models[i].fit(ft, discounted_future_payoffs[active_indices])
            predicted_future_payoffs = np.ones(m)  # the indices where the current payoff is 0 do not matter and are
            # certainly excluded in the following code if we just set the predicted future payoff 1
            predicted_future_payoffs[active_indices] = models[i].predict(ft)
            execute_now = current_payoff > predicted_future_payoffs
            discounted_future_payoffs[execute_now] = current_payoff[execute_now]
        else:
            ft = features(np.zeros((d, 1)))
            models[i] = models[i].fit(ft, np.array([0]))
    if not varying_initial_conditions:
        discounted_future_payoffs = discount * discounted_future_payoffs
        current_payoff = payoff(S=samples[0, :, 0])
        average_discounted_future_payoffs = np.average(discounted_future_payoffs)
        average_current_payoff = np.average(current_payoff)
        ft = features(samples[:, :1, 0])
        models[0].fit(ft, np.array([average_discounted_future_payoffs]))
        if average_current_payoff >= average_discounted_future_payoffs:
            discounted_future_payoffs = average_current_payoff * np.ones(m)
    if antithetic:
        discounted_future_payoffs = 0.5 * (discounted_future_payoffs[m // 2:] + discounted_future_payoffs[:m // 2])
    return MC(discounted_future_payoffs), models


def price_am_forward(T, r, samples, payoff, models, features, antithetic=False):
    """
    Prices American options.
    :param T: Maturity
    :param r: Interest rate
    :param samples: Numpy array of shape (d, m, N + 1), where N is the number of time steps, m the number of samples,
        and d is the dimension of the Markovian process. It is assumed that the first dimension is the stock price
        process
    :param payoff: The payoff function, a function of S (sample values) only
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :return: The price of the American option, a list of the models for each exercise time that approximate the
        future payoff, and the function computing the features of the samples
    """
    N_time = samples.shape[-1] - 1
    m = samples.shape[1]
    dt = T / N_time

    discount = np.exp(-r * dt)
    discounted_future_payoffs = payoff(S=samples[0, :, -1])
    for i in range(N_time - 1, -1, -1):
        discounted_future_payoffs = discount * discounted_future_payoffs
        current_payoff = payoff(S=samples[0, :, i])
        active_indices = current_payoff > 0
        if np.sum(active_indices) > 0:
            ft = features(samples[:, active_indices, i])
            predicted_future_payoffs = np.ones(m)  # the indices where the current payoff is 0 do not matter and are
            # certainly excluded in the following code if we just set the predicted future payoff 1
            predicted_future_payoffs[active_indices] = models[i].predict(ft)
            execute_now = current_payoff > predicted_future_payoffs
            discounted_future_payoffs[execute_now] = current_payoff[execute_now]
    if antithetic:
        discounted_future_payoffs = 0.5 * (discounted_future_payoffs[m // 2:] + discounted_future_payoffs[:m // 2])
    return MC(discounted_future_payoffs)


def price_am_BS(S_0, sigma, T, r, m, N, K, feature_degree, antithetic=False):
    def features(x):
        feat = np.empty((x.shape[1], feature_degree))
        feat[:, 0] = x
        for i in range(1, feature_degree):
            feat[:, i] = feat[:, i-1] * x
        return feat

    samples = S_0 * BS_paths(sigma=sigma, T=T, n=N, m=m, r=r, antithetic=antithetic)
    price, models = price_am(T=T, r=r, samples=samples[None, :, :], payoff=lambda S: payoff_put(S, K),
                             features=features, antithetic=antithetic, varying_initial_conditions=False)
    return price


def price_am_iso(T, r, samples, payoff, features, antithetic=False, varying_initial_conditions=False):
    """
    Prices American options.
    :param T: Maturity
    :param r: Interest rate
    :param samples: Numpy array of shape (d, m, N + 1), where N is the number of time steps, m the number of samples,
        and d is the dimension of the Markovian process
    :param payoff: The vectorized payoff function taking as input an arbitrary numpy array of shape (d, ...) (is a
        vectorized function of d variables)
    :param features: A function turning samples at a specific time step into features
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :param varying_initial_conditions: If True, assumes that the initial condition varies with samples. Otherwise,
        assumes that the initial condition is the same for all samples
    :return: The price of the American option and a list of the models for each exercise time that approximate the
        future payoff
    """
    N_time = samples.shape[-1] - 1
    m = samples.shape[1]
    dt = T / N_time

    discount = np.exp(-r * dt)
    models = [LinearRegression() for _ in range(N_time)]
    discounted_future_payoffs = payoff(samples[:, :, -1])
    for i in range(N_time - 1, -1 if varying_initial_conditions else 0, -1):
        discounted_future_payoffs = discount * discounted_future_payoffs
        current_payoff = payoff(samples[:, :, i])
        ft = features(samples[:, :, i])
        models[i] = models[i].fit(ft, discounted_future_payoffs)
        predicted_future_payoffs = models[i].predict(ft)
        execute_now = current_payoff > predicted_future_payoffs
        discounted_future_payoffs[execute_now] = current_payoff[execute_now]
    if not varying_initial_conditions:
        discounted_future_payoffs = discount * discounted_future_payoffs
        current_payoff = payoff(samples[:, :, 0])
        average_discounted_future_payoffs = np.average(discounted_future_payoffs)
        average_current_payoff = np.average(current_payoff)
        ft = features(samples[:, :1, 0])
        models[0].fit(ft, np.array([average_discounted_future_payoffs]))
        if average_current_payoff >= average_discounted_future_payoffs:
            discounted_future_payoffs = average_current_payoff * np.ones(m)
    if antithetic:
        discounted_future_payoffs = 0.5 * (discounted_future_payoffs[m // 2:2 * (m // 2)]
                                           + discounted_future_payoffs[:m // 2])
    return MC(discounted_future_payoffs), models


def price_am_iso_forward(T, r, samples, payoff, models, features, antithetic=False):
    """
    Prices American options.
    :param T: Maturity
    :param r: Interest rate
    :param samples: Numpy array of shape (d, m, N + 1), where N is the number of time steps, m the number of samples,
        and d is the dimension of the Markovian process
    :param payoff: The vectorized payoff function taking as input an arbitrary numpy array of shape (d, ...) (is a
        vectorized function of d variables)
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :return: The price of the American option, a list of the models for each exercise time that approximate the
        future payoff, and the function computing the features of the samples
    """
    N_time = samples.shape[-1] - 1
    m = samples.shape[1]
    dt = T / N_time

    discount = np.exp(-r * dt)
    discounted_future_payoffs = payoff(samples[:, :, -1])
    for i in range(N_time - 1, -1, -1):
        discounted_future_payoffs = discount * discounted_future_payoffs
        current_payoff = payoff(samples[:, :, i])
        ft = features(samples[:, :, i])
        predicted_future_payoffs = models[i].predict(ft)
        execute_now = current_payoff > predicted_future_payoffs
        discounted_future_payoffs[execute_now] = current_payoff[execute_now]
    if antithetic:
        discounted_future_payoffs = 0.5 * (discounted_future_payoffs[m // 2:2 * (m // 2)]
                                           + discounted_future_payoffs[:m // 2])
    return MC(discounted_future_payoffs)
