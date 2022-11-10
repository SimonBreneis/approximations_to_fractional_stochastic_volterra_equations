import numpy as np
import scipy.interpolate
import ComputationalFinance as cf
import scipy.stats


def samples(lambda_, nu, theta, V_0, T, nodes, weights, rho=0., S_0=1., r=0., m=1000, N_time=1000, sample_paths=False,
            return_times=None, vol_only=False, euler=False, antithetic=True):
    """
    Simulates sample paths under the Markovian approximation of the rough Heston model.
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps
    :param S_0: Initial stock price
    :param r: Interest rate
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: May specify an array of times at which the sample path values should be returned. If None and
        sample_paths is True, this is equivalent to return_times = np.linspace(0, T, N_time+1)
    :param vol_only: If True, simulates only the volatility process, not the stock price process
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :return: Numpy array of the final stock prices
    """
    if return_times is not None:
        sample_paths = True
        return_times = np.fmax(return_times, 0)
        T = np.amax(return_times)
    dt = T / N_time
    N = len(nodes)
    if N == 1:
        nodes = np.array([nodes[0], 2 * nodes[0] + 1])
        weights = np.array([weights[0], 0])
        N = 2
        one_node = True
    else:
        one_node = False

    V_init = np.zeros(N)
    V_init[0] = V_0 / weights[0]

    if euler:
        A = np.eye(N) + np.diag(nodes) * dt + lambda_ * weights[None, :] * dt
        A_inv = np.linalg.inv(A)
        b = theta * dt + (nodes * V_init)[:, None] * dt

        if vol_only:
            def step_SV(V_comp_):
                sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))
                dW = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                return A_inv @ (V_comp_ + nu * (sq_V * dW)[None, :] + b)
        else:
            def step_SV(log_S_, V_comp_):
                sq_V = np.sqrt(np.fmax(weights @ V_comp_, 0))
                dW = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                dB = cf.rand_normal(loc=0, scale=np.sqrt(dt), size=m, antithetic=antithetic)
                log_S_ = log_S_ + r * dt + sq_V * (rho * dW + np.sqrt(1 - rho ** 2) * dB) - 0.5 * sq_V ** 2 * dt
                V_comp_ = A_inv @ (V_comp_ + nu * (sq_V * dW)[None, :] + b)
                return log_S_, V_comp_

    else:
        weight_sum = np.sum(weights)
        A = -(np.diag(nodes) + lambda_ * weights[None, :]) * dt / 2
        exp_A = scipy.linalg.expm(A)
        b = (nodes * V_init + theta) * dt / 2
        ODE_b = np.linalg.solve(A, (exp_A - np.eye(N)) @ b)[:, None]
        z = weight_sum ** 2 * nu ** 2 * dt
        rho_bar_sq = 1 - rho ** 2
        rho_bar = np.sqrt(rho_bar_sq)

        def ODE_step_V(V_):
            return exp_A @ V_ + ODE_b

        B = (6 + np.sqrt(3)) / 4
        A = B - 0.75

        def SDE_step_V(V_):
            x = weights @ V_
            rv = cf.rand_uniform(size=m, antithetic=antithetic)
            temp = np.sqrt((3 * z) * x + (B * z) ** 2)
            p_1 = (z / 2) * x * ((A * B - A - B + 1.5) * z + (np.sqrt(3) - 1) / 4 * temp + x) / (
                    (x + B * z - temp) * temp * (temp - (B - A) * z))
            p_2 = x / (1.5 * x + A * (B - A / 2) * z)
            test_1 = rv < p_1
            test_2 = p_1 + p_2 <= rv
            x_step = A * z * np.ones(len(temp))
            x_step[test_1] = B * z - temp[test_1]
            x_step[test_2] = B * z + temp[test_2]
            return V_ + (x_step / weight_sum)[None, :]

        def step_V(V_):
            return ODE_step_V(SDE_step_V(ODE_step_V(V_)))

        def SDE_step_B(log_S_, V_):
            dB = cf.rand_normal(loc=0, scale=np.sqrt(dt / 2), size=m, antithetic=antithetic)
            x = weights @ V_
            return log_S_ + np.sqrt(x) * rho_bar * dB - (0.5 * rho_bar_sq * dt / 2) * x, V_

        drift_SDE_step_W = - (nodes[0] * V_init[0] + theta) * dt
        fact_1 = dt / 2 * (lambda_ - 0.5 * rho * nu)

        def SDE_step_W(log_S_, V_):
            V_new = step_V(V_)
            dY = V_ + V_new
            log_S_new = log_S_ + r * dt + rho / nu * (drift_SDE_step_W + (dt / 2 * nodes[0]) * dY[0, :]
                                                      + fact_1 * (weights @ dY) + (V_new[0, :] - V_[0, :]))
            return log_S_new, V_new

        if vol_only:
            def step_SV(V_):
                return step_V(V_)
        else:
            def step_SV(S_, V_):
                return SDE_step_B(*SDE_step_W(*SDE_step_B(S_, V_)))

    if vol_only:
        if sample_paths:
            V_comp = np.empty((N, m, N_time + 1))
            V_comp[:, :, 0] = V_init[:, None]
            for i in range(N_time):
                print(f'Step {i} of {N_time}')
                V_comp[:, :, i + 1] = step_SV(V_comp[:, :, i])
            V = np.fmax(np.einsum('i,ijk->jk', weights, V_comp), 0)
            if return_times is not None:
                times = np.linspace(0, T, N_time + 1)
                V = scipy.interpolate.interp1d(x=times, y=V)(return_times)
                V_comp = scipy.interpolate.interp1d(x=times, y=V_comp)(return_times)
            result = np.empty((N + 1, m, V.shape[-1]))
            result[0, :, :] = V
            result[1:, :, :] = V_comp
        else:
            V_comp = np.zeros((N, m))
            V_comp[:, :] = V_init[:, None]
            for i in range(N_time):
                print(f'Step {i} of {N_time}')
                V_comp = step_SV(V_comp)
            V = np.fmax(weights @ V_comp, 0)
            result = np.empty((N + 1, m))
            result[0, :] = V
            result[1:, :] = V_comp
    else:
        if sample_paths:
            V_comp = np.empty((N, m, N_time + 1))
            V_comp[:, :, 0] = V_init[:, None]
            log_S = np.empty((m, N_time + 1))
            log_S[:, 0] = np.log(S_0)
            for i in range(N_time):
                print(f'Step {i} of {N_time}')
                log_S[:, i + 1], V_comp[:, :, i + 1] = step_SV(log_S[:, i], V_comp[:, :, i])
            V = np.fmax(np.einsum('i,ijk->jk', weights, V_comp), 0)
            if return_times is not None:
                times = np.linspace(0, T, N_time + 1)
                log_S = scipy.interpolate.interp1d(x=times, y=log_S)(return_times)
                V = scipy.interpolate.interp1d(x=times, y=V)(return_times)
                V_comp = scipy.interpolate.interp1d(x=times, y=V_comp)(return_times)
            result = np.empty((N + 2, m, V.shape[-1]))
            result[0, :, :] = np.exp(log_S)
            result[1, :, :] = V
            result[2:, :, :] = V_comp
        else:
            V_comp = np.zeros((N, m))
            V_comp[:, :] = V_init[:, None]
            log_S = np.ones(m) * np.log(S_0)
            for i in range(N_time):
                # print(f'Step {i} of {N_time}')
                log_S, V_comp = step_SV(log_S, V_comp)
            V = np.fmax(weights @ V_comp, 0)
            result = np.empty((N + 2, m))
            result[0, :] = np.exp(log_S)
            result[1, :] = V
            result[2:, :] = V_comp

    if one_node:
        result = result[:-1, ...]
    return result


def eur(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, r=0., m=1000, N_time=1000, euler=False,
        antithetic=True, payoff='call', implied_vol=False):
    """
    Gives the price or the implied volatility of a European option in the approximated, Markovian rough Heston model.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param r: Interest rate
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param N_time: Number of time steps used in simulation
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :param payoff: The payoff function, or the string 'call' or the string 'put'
    :param implied_vol: If True (only for payoff 'call' or 'put') returns the implied volatility, else returns the price
    return: The prices of the call option for the various strike prices in K
    """
    samples_ = samples(lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, m=m, S_0=S_0, r=r, N_time=N_time,
                       nodes=nodes, weights=weights, sample_paths=False, euler=euler, antithetic=antithetic)[0, :]
    return cf.eur_MC(S_0=S_0, K=K, T=T, r=r, samples=samples_, payoff=payoff, antithetic=antithetic,
                     implied_vol=implied_vol)


def price_geom_asian_call(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, m=1000, N_time=1000, euler=False,
                          antithetic=True):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param N_time: Number of time steps used in simulation
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    return: The prices of the call option for the various strike prices in K
    """
    samples_ = samples(lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, m=m, S_0=S_0, N_time=N_time,
                       nodes=nodes, weights=weights, sample_paths=True, euler=euler, antithetic=antithetic)[0, :, :]
    return cf.price_geom_asian_call_MC(K=K, samples=samples_, antithetic=antithetic)


def price_avg_vol_call(K, lambda_, nu, theta, V_0, T, nodes, weights, m=1000, N_time=1000, euler=False,
                       antithetic=True):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param N_time: Number of time steps used in simulation
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    return: The prices of the call option for the various strike prices in K
    """
    samples_ = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, m=m, N_time=N_time, nodes=nodes,
                       weights=weights, sample_paths=True, vol_only=True, euler=euler, antithetic=antithetic)[0, :, :]
    return cf.price_avg_vol_call_MC(K=K, samples=samples_, antithetic=antithetic)


def am_features(x, degree=6, K=0.):
    n_samples = x.shape[-1]
    d = x.shape[0]
    normalized_stock = ((x[0, :] - K) / K) if np.abs(K) > 0.01 else x[0, :]
    vol = (np.sum(x[1:, :], axis=0))
    vol_factors = x[1:-1, :].T
    if degree == 1:
        dim = 1
    elif degree == 2:
        dim = 3
    else:
        dim = d + 3
    feat = np.empty((n_samples, dim))
    if degree >= 1:
        feat[:, 0] = normalized_stock
    if degree >= 2:
        feat[:, 1] = normalized_stock ** 2
        feat[:, 2] = vol
    if degree >= 3:
        feat[:, 3:5] = normalized_stock[:, None] * feat[:, 1:3]
        feat[:, 5:d + 3] = vol_factors
    current_N = 4
    current_ind = d + 3
    lower_N_stock = 3
    upper_N_stock = d + 3
    lower_N_vol = 2
    upper_N_vol = 3
    next_lower_N_vol = 5
    next_upper_N_vol = d + 3
    lower_N_vol_factors = np.arange(5, d + 3, dtype=int)
    upper_N_vol_factors = d + 3
    while current_N <= degree:
        feat_new = np.empty((n_samples, feat.shape[1] + upper_N_stock - lower_N_stock + upper_N_vol - lower_N_vol
                             + (current_N % 3 == 0)
                             * (upper_N_vol_factors * (d - 2) - np.sum(lower_N_vol_factors, dtype=int))))
        # print(feat_new.shape[1])
        feat_new[:, :current_ind] = feat
        feat = feat_new
        next_ind = current_ind + upper_N_stock - lower_N_stock
        feat[:, current_ind:next_ind] = normalized_stock[:, None] * feat[:, lower_N_stock:upper_N_stock]
        lower_N_stock = current_ind
        current_ind = next_ind
        next_ind = current_ind + upper_N_vol - lower_N_vol
        feat[:, current_ind:next_ind] = vol[:, None] * feat[:, lower_N_vol:upper_N_vol]
        lower_N_vol = next_lower_N_vol
        next_lower_N_vol = current_ind
        current_ind = next_ind
        if current_N % 3 == 0:
            for i in range(d - 2):
                next_ind = current_ind + upper_N_vol_factors - lower_N_vol_factors[i]
                feat[:, current_ind:next_ind] = \
                    vol_factors[:, i:i + 1] * feat[:, lower_N_vol_factors[i]:upper_N_vol_factors]
                lower_N_vol_factors[i] = current_ind
                current_ind = next_ind
            upper_N_vol_factors = current_ind
        upper_N_stock = current_ind
        upper_N_vol = next_upper_N_vol
        next_upper_N_vol = current_ind
        current_N = current_N + 1
    return feat


def price_am(K, lambda_, rho, nu, theta, V_0, S_0, T, nodes, weights, payoff, r=0., m=1000000, N_time=200, N_dates=12,
             feature_degree=6, euler=False, antithetic=True, unbiased=True):
    """
    Gives the price of an American option in the approximated, Markovian rough Heston model.
    :param K: Strike price
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param S_0: Initial stock price
    :param T: Final time/Time of maturity
    :param nodes: The nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param payoff: The payoff function, either 'call' or 'put' or a function taking as inputs S (samples) and K (strike)
    :param r: Interest rate
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param N_time: Number of time steps used in the simulation
    :param N_dates: Number of exercise dates. If None, N_dates = N_time
    :param feature_degree: The degree of the polynomial features used in the regression
    :param euler: If True, uses an Euler scheme. If False, uses moment matching
    :param antithetic: If True, uses antithetic variates to reduce the MC error
    :param unbiased: If True, uses newly generated samples to compute the price of the American option
    return: The prices of the call option for the various strike prices in K
    """
    if payoff == 'call':
        def payoff(S):
            return cf.payoff_call(S=S, K=K)
    elif payoff == 'put':
        def payoff(S):
            return cf.payoff_put(S=S, K=K)
    if N_dates is None:
        N_dates = N_time
    ex_times = np.linspace(0, T, N_dates + 1)
    samples_ = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights, rho=rho,
                       S_0=S_0, r=r, m=m, N_time=N_time, sample_paths=True, return_times=ex_times, vol_only=False,
                       euler=euler, antithetic=antithetic)
    preprocessed_samples = np.empty((samples_.shape[0] - 1, samples_.shape[1], samples_.shape[2]))
    preprocessed_samples[0, :, :] = samples_[0, :, :]
    preprocessed_samples[1:, :, :] = weights[:, None, None] * samples_[2:, :, :]
    preprocessed_samples[1:, :, :] = preprocessed_samples[1:, :, :] - preprocessed_samples[1:, :, :1]
    (biased_est, biased_stat), models, features = cf.price_am(T=T, r=r, samples=preprocessed_samples,
                                                              antithetic=antithetic, payoff=payoff,
                                                              features=lambda x: am_features(x=x, degree=feature_degree,
                                                                                             K=K))
    print(biased_est, biased_stat)
    if not unbiased:
        return biased_est, biased_stat, models, features
    samples_ = samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights, rho=rho,
                       S_0=S_0, r=r, m=m, N_time=N_time, sample_paths=True, return_times=ex_times, vol_only=False,
                       euler=euler, antithetic=antithetic)
    preprocessed_samples = np.empty((samples_.shape[0] - 1, samples_.shape[1], samples_.shape[2]))
    preprocessed_samples[0, :, :] = samples_[0, :, :]
    preprocessed_samples[1:, :, :] = weights[:, None, None] * samples_[2:, :, :]
    preprocessed_samples[1:, :, :] = preprocessed_samples[1:, :, :] - preprocessed_samples[1:, :, :1]
    est, stat = cf.price_am_forward(T=T, r=r, samples=preprocessed_samples, payoff=payoff, models=models,
                                    features=features, antithetic=antithetic)
    return est, stat, biased_est, biased_stat, models, features
