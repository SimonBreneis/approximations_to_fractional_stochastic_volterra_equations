import time
import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp
import rHestonSplitKernel as sk


def get_sample_paths(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, WB=None, m=1, mode="best",
                     vol_behaviour='sticky'):
    """
    Simulates paths under the Markovian approximation of the rough Heston model.
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps
    :param N: Total number of points in the quadrature rule
    :param S_0: Initial stock price
    :param WB: Brownian increments. Has shape (2, m, N_time). First components are the dW, second the dB
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param vol_behaviour: The behaviour of the volatility at 0. The following options are possible:
        - sticky: If V becomes negative, resets V to 0 without resetting the components of V
        - constant: Uses constant volatility V_0, as in the Black-Scholes model
        - mean reversion: If V becomes negative, inserts as much fictitious time as needed until the components have
            reverted enough to ensure positive V. The fictitious time is only for the dt component, as the dW and dB
            components always have a factor containing V in front, which is interpreted to be 0 on these fictitious time
            intervals
        - hyperplane reset: If V becomes negative, resets the components just enough to ensure V=0. The components are
            reset by adding the correct multiple of the weights vector
        - hyperplane reflection: If V becomes negative, resets the components by twice as much as in hyperplane reset.
            This ensures that V changes sign and does not stay 0
        - adaptive: If V becomes negative, subdivides the last interval into two intervals. This is continued
            recursively until V is non-negative, but at most 5 consecutive times (i.e. an interval is divided into at
            most 32 intervals)
        - multiple time scales: Solves for the components on different time scales corresponding to their mean
            reversions. If V becomes negative, applies the hyperplane reset method
        - split kernel: Uses the invariant measure of the high mean-reversion components instead of actually simulating
            them. In practice, this is essentially equivalent to a certain smoothing of the square root. If V becomes
            negative, applies the hyperplane reset method
    :return: Three numpy arrays, the stock sample paths, the square roots of the volatility sample paths, and the
        sample paths of the components of the volatility. If vol_behaviour is adaptive, also returns a list of the
        values of the components of the volatility where the volatility became negative
    """
    dt = T / N_time
    nodes, weights = rk.quadrature_rule(H, N, T, mode)
    N = len(nodes)

    A = mp.eye(N) + mp.diag(nodes) * dt + lambda_ * mp.matrix([[weights[i] for i in range(N)] for _ in range(N)]) * dt
    A_inv = rk.mp_to_np(mp.inverse(A))

    if WB is None:
        dW = np.random.normal(0, np.sqrt(dt), (m, N_time))
        dB = np.random.normal(0, np.sqrt(dt), (m, N_time))
    else:
        dW = WB[0, :, :]
        dB = WB[1, :, :]
        N_time = len(dW[0, :])
        m = len(dW[:, 0])
    S_BM = rho * dW + np.sqrt(1 - rho ** 2) * dB

    S_values = np.zeros((m, N_time + 1))
    V_values = np.zeros((m, N_time + 1))
    S_values[:, 0] = S_0 * np.ones(m)
    V_values[:, 0] = V_0 * np.ones(m)
    V_components = np.zeros(shape=(m, N, N_time + 1))
    V_components[:, 0, 0] = V_0 / weights[0] * np.ones(m)
    b_comp = nodes * V_components[0, :, 0] * dt + theta * dt

    def b(vol, dw):
        return b_comp + vol + nu * np.sqrt(np.fmax(np.dot(weights, vol), 0)) * dw

    if vol_behaviour == 'sticky':
        for j in range(m):
            print(j)
            for i in range(1, N_time + 1):
                V_components[j, :, i] = A_inv @ b(V_components[j, :, i - 1], dW[j, i - 1])
                V_values[j, i] = np.fmax(np.dot(weights, V_components[j, :, i]), 0)
                S_values[j, i] = S_values[j, i - 1] + np.sqrt(V_values[j, i - 1]) * S_values[j, i - 1] * S_BM[j, i - 1]
    elif vol_behaviour == 'constant':
        for j in range(m):
            print(j)
            for i in range(1, N_time + 1):
                S_values[j, i] = S_values[j, i - 1] + np.sqrt(V_0) * S_values[j, i - 1] * S_BM[j, i - 1]
    elif vol_behaviour == 'mean reversion':
        for j in range(m):
            zero_vol = V_0 <= 0
            for i in range(1, N_time + 1):
                V_components[j, :, i] = A_inv @ b(V_components[j, :, i - 1], dW[j, i - 1])
                V_values[j, i] = np.fmax(np.dot(weights, V_components[j, :, i]), 0)
                if zero_vol:
                    while V_values[j, i] <= 0:
                        V_components[j, :, i] = A_inv @ b(V_components[j, :, i], dW[j, i - 1])
                        V_values[j, i] = np.fmax(np.dot(weights, V_components[j, :, i]), 0)
                zero_vol = V_values[j, i] <= 0
                S_values[j, i] = S_values[j, i - 1] + np.sqrt(V_values[j, i - 1]) * S_values[j, i - 1] * S_BM[j, i - 1]
    elif vol_behaviour == 'hyperplane reset':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for j in range(m):
            print(j)
            for i in range(1, N_time + 1):
                if i % 1000 == 0:
                    print(f'{i} of {N_time}')
                V_components[j, :, i] = A_inv @ b(V_components[j, :, i - 1], dW[j, i - 1])
                V_values[j, i] = np.dot(weights, V_components[j, :, i])
                if V_values[j, i] < 0:
                    V_components[j, :, i] = V_components[j, :, i] - V_values[j, i] * rescaled_weights
                    V_values[j, i] = 0
                S_values[j, i] = S_values[j, i - 1] + np.sqrt(V_values[j, i - 1]) * S_values[j, i - 1] * S_BM[j, i - 1]
    elif vol_behaviour == 'hyperplane reflection':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for j in range(m):
            for i in range(1, N_time + 1):
                V_components[j, :, i] = A_inv @ b(V_components[j, :, i - 1], dW[j, i - 1])
                V_values[j, i] = np.dot(weights, V_components[j, :, i])
                if V_values[j, i] < 0:
                    V_components[j, :, i] = V_components[j, :, i] - 2 * V_values[j, i] * rescaled_weights
                    V_values[j, i] = -V_values[j, i]
                S_values[j, i] = S_values[j, i - 1] + np.sqrt(V_values[j, i - 1]) * S_values[j, i - 1] * S_BM[j, i - 1]
    elif vol_behaviour == 'adaptive':
        max_iter = 5
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm

        def nested_step(V_comp_loc, dW_loc_, dt_loc_, iteration):
            dW_mid = dW_loc_ / 2 + np.random.normal() * np.sqrt(dt_loc_ / 4)
            dt_loc_ = dt_loc_ / 2

            b_ = nodes * V_components[0, :, 0] * dt_loc_ + theta * dt_loc_ + V_comp_loc + nu * np.sqrt(
                np.fmax(np.dot(weights, V_comp_loc), 0)) * dW_mid
            A_ = mp.eye(len(nodes)) + mp.diag(nodes) * dt_loc_ + lambda_ * mp.matrix(
                [[weights[i] for i in range(len(nodes))] for _ in range(len(nodes))]) * dt_loc_
            V_comp_loc_updated = rk.mp_to_np(mp.lu_solve(A_, b_))
            V_loc_updated = np.dot(weights, V_comp_loc_updated)
            if V_loc_updated < 0:
                if iteration >= max_iter:
                    V_comp_loc_updated = V_comp_loc_updated - V_loc_updated * rescaled_weights
                else:
                    V_comp_loc_updated = nested_step(V_comp_loc, dW_mid, dt_loc_, iteration + 1)
            dW_second = dW_loc_ - dW_mid
            b_ = nodes * V_components[0, :, 0] * dt_loc_ + theta * dt_loc_ + V_comp_loc_updated + nu * np.sqrt(
                np.fmax(np.dot(weights, V_comp_loc_updated), 0)) * dW_second
            V_comp_loc_final = rk.mp_to_np(mp.lu_solve(A_, b_))
            V_loc_final = np.dot(weights, V_comp_loc_final)
            if V_loc_final < 0:
                if iteration >= max_iter:
                    V_comp_loc_final = V_comp_loc_final - V_loc_final * rescaled_weights
                else:
                    V_comp_loc_final = nested_step(V_comp_loc_updated, dW_second, dt_loc_, iteration + 1)
            return V_comp_loc_final

        bad_values = []

        for j in range(m):
            for i in range(1, N_time + 1):
                print(f'{j} of {m} samples, {i} of {N_time} time steps')
                V_components[j, :, i] = A_inv @ b(V_components[j, :, i - 1], dW[j, i - 1])
                V_values[j, i] = np.dot(weights, V_components[j, :, i])
                if V_values[j, i] < 0:
                    print('BAD!')
                    bad_values.append(V_components[j, :, i - 1])
                    V_components[j, :, i] = nested_step(V_components[j, :, i - 1], dW[j, i - 1], dt, 1)
                    V_values[j, i] = np.fmax(np.dot(weights, V_components[j, :, i]), 0)
                S_values[j, i] = S_values[j, i - 1] + np.sqrt(V_values[j, i - 1]) * S_values[j, i - 1] * S_BM[j, i - 1]
    elif vol_behaviour == 'multiple time scales':
        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        eta = 4.6  # e^(-4.6) approx 0.01, so only 1% of initial condition remaining relevant
        L = int(np.ceil(5 * eta))  # number of time steps per component
        n_trivial_intervals = np.sum(eta / nodes[1:] >= dt)

        # find the local time increments
        dt_loc = np.empty(N - n_trivial_intervals)
        dt_past = 0  # how much time of the interval [t_m, t_{m+1}] has already passed
        for j in range(n_trivial_intervals, N):
            if j == N - 1:
                dt_loc[j - n_trivial_intervals] = (dt - dt_past) / L
                dt_past = dt
            else:
                dt_loc[j - n_trivial_intervals] = (dt - eta / nodes[j + 1] - dt_past) / L
                dt_past = dt - eta / nodes[j + 1]

        # find the inverses of different As
        A_invs = []
        for j in range(n_trivial_intervals, N - 1):
            A = mp.eye(N) \
                + mp.diag(mp.matrix([nodes[k] for k in range(j + 1)] + [mp.mpf(0)] * (N - j - 1))) * dt_loc[
                    j - n_trivial_intervals] \
                + lambda_ * mp.matrix([[weights[i] * (k <= j) for i in range(N)] for k in range(N)]) * dt_loc[
                    j - n_trivial_intervals]
            A_invs.append(rk.mp_to_np(mp.inverse(A)))
        A_invs.append(A_inv)

        # find the different bs
        def bs(j_, vol, dw):
            return np.array((nodes * V_components[0, :, 0] * dt_loc[j_ - n_trivial_intervals] + theta * dt_loc[
                j_ - n_trivial_intervals] + nu * np.sqrt(np.fmax(np.dot(weights, vol), 0)) * dw)[:(j_ + 1)].tolist() + [
                                0] * (N - j_ - 1)) + vol

        for k in range(m):
            for i in range(1, N_time + 1):
                dW_so_far = 0
                dt_so_far = 0
                V_components[k, :, i] = V_components[k, :, i - 1]
                for j in range(n_trivial_intervals, N):
                    for _ in range(L):
                        dW_loc = np.random.normal(
                            (dW[k, i - 1] - dW_so_far) * dt_loc[j - n_trivial_intervals] / (dt - dt_so_far),
                            np.sqrt(dt_loc[j - n_trivial_intervals]))
                        dW_so_far += dW_loc
                        dt_so_far += dt_loc[j - n_trivial_intervals]
                        V_components[k, :, i] = A_invs[j - n_trivial_intervals] @ bs(j, V_components[k, :, i], dW_loc)
                        V_values[k, i] = np.dot(weights, V_components[k, :, i])
                        if V_values[k, i] < 0:
                            V_components[k, :, i] = V_components[k, :, i] - V_values[k, i] * rescaled_weights
                            V_values[k, i] = 0
                S_values[k, i] = S_values[k, i - 1] + np.sqrt(V_values[k, i - 1]) * S_values[k, i - 1] * S_BM[k, i - 1]

    elif vol_behaviour == 'split kernel':
        N = np.sum(nodes < 2 / dt)
        fast_nodes = nodes[N:]
        fast_weights = weights[N:]
        nodes = nodes[:N]
        weights = weights[:N]
        V_components = V_components[:, :N, :]
        mu_a, mu_b, sr = sk.smooth_root(nodes=fast_nodes, weights=fast_weights, theta=theta, lambda_=lambda_, nu=nu,
                                        us=np.linspace(-5, 5, 101), ps=np.exp(np.linspace(-10, 1, 100)), q=10,
                                        N_Riccati=10, adaptive=True, M=1000000)

        A = mp.eye(N) + mp.diag(nodes) * dt + mp.mpf(lambda_ * (1 + mu_a)) * mp.matrix(
            [[weights[i] for i in range(N)] for _ in range(N)]) * dt
        A_inv = rk.mp_to_np(mp.inverse(A))

        b_comp = nodes * V_components[0, :, 0] * dt + (theta - lambda_ * mu_b) * dt

        def b(vol, dw):
            return b_comp + vol + nu * sr(np.dot(weights, vol)) * dw

        weights_norm = np.dot(weights, weights)
        rescaled_weights = weights / weights_norm
        for j in range(m):
            print(j)
            for i in range(1, N_time + 1):
                V_components[j, :, i] = A_inv @ b(V_components[j, :, i - 1], dW[j, i - 1])
                V_values[j, i] = np.dot(weights, V_components[j, :, i])
                if V_values[j, i] < 0:
                    V_components[j, :, i] = V_components[j, :, i] - V_values[j, i] * rescaled_weights
                    V_values[j, i] = 0
                S_values[j, i] = S_values[j, i - 1] + sr(V_values[j, i - 1]) * S_values[j, i - 1] * S_BM[j, i - 1]
        return S_values, np.array([[sr(V_values[i, j]) for j in range(len(V_values[0, :]))] for i in
                                   range(len(V_values[:, 0]))]), V_components

    if vol_behaviour == 'adaptive':
        return S_values, np.sqrt(V_values), V_components, bad_values
    return S_values, np.sqrt(V_values), V_components


def samples(H, lambda_, rho, nu, theta, V_0, T, N, S, m=1000, N_time=1000, WB=None, mode="best", vol_behaviour='sticky',
            nodes=None, weights=None):
    """
    Simulates (the final stock prices) of sample paths under the Markovian approximation of the rough Heston model.
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps
    :param N: Total number of points in the quadrature rule
    :param S: Initial stock price
    :param WB: Brownian increments. Has shape (2, m, N_time). First components are the dW, second the dB
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param vol_behaviour: The behaviour of the volatility at 0. The following options are possible:
        - sticky: If V becomes negative, resets V to 0 without resetting the components of V
        - constant: Uses constant volatility V_0, as in the Black-Scholes model
        - hyperplane reset: If V becomes negative, resets the components just enough to ensure V=0. The components are
            reset by adding the correct multiple of the weights vector
        - hyperplane reflection: If V becomes negative, resets the components by twice as much as in hyperplane reset.
            This ensures that V changes sign and does not stay 0
        - adaptive: If V becomes negative, subdivides the last interval into two intervals. This is continued
            recursively until V is non-negative, but at most 5 consecutive times (i.e. an interval is divided into at
            most 32 intervals)
        - multiple time scales: Solves for the components on different time scales corresponding to their mean
            reversions. If V becomes negative, applies the hyperplane reset method
        - split kernel: Uses the invariant measure of the high mean-reversion components instead of actually simulating
            them. In practice, this is essentially equivalent to a certain smoothing of the square root. If V becomes
            negative, applies the hyperplane reset method
    :return: Numpy array of the final stock prices
    """
    if WB is None:
        dW = np.random.normal(0, np.sqrt(T / N_time), (m, N_time))
        dB = np.random.normal(0, np.sqrt(T / N_time), (m, N_time))
    else:
        m = WB.shape[1]
        N_time = WB.shape[2]
        dW = WB[0, :, :]
        dB = WB[1, :, :]
    dt = T / N_time
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode=mode)
    N = len(nodes)
    if N == 1:
        nodes = np.array([nodes[0], 1])
        weights = np.array([weights[0], 0])
        N = 2

    A = np.eye(N) + np.diag(nodes) * dt + lambda_ * weights[None, :] * dt
    A_inv = np.linalg.inv(A)
    A_inv_T = A_inv.T

    tic = time.perf_counter()
    V = V_0

    V_components = np.zeros((m, len(weights)))
    V_components[:, 0] = V_0 / weights[0]
    V_original = V_components[0, :]
    S_BM = rho * dW + np.sqrt(1 - rho ** 2) * dB
    rescaled_weights = weights / np.sum(weights ** 2)

    b_comp = theta * dt + (nodes * V_original)[None, :] * dt

    def b(V_comp_, sq_V_, dW_):
        return V_comp_ + nu * (sq_V_ * dW_)[:, None] + b_comp

    if vol_behaviour == 'sticky':
        sq_V = np.sqrt(V)
        for i in range(N_time):
            S = S + sq_V * S * S_BM[:, i]
            V_components = b(V_components, sq_V, dW[:, i]) @ A_inv_T
            sq_V = np.sqrt(np.fmax(V_components @ weights, 0))

    elif vol_behaviour == 'hyperplane reset':
        # log_S = 0
        for i in range(N_time):
            '''
            if i % 100 == 0:
                print(f'{i} of {N_time}')
            '''
            sq_V = np.sqrt(V)
            S = S + sq_V * S * S_BM[:, i]
            # log_S = log_S + np.sqrt(V) * S_BM[:, i] - V * (dt/2)
            V_components = b(V_components, sq_V, dW[:, i]) @ A_inv_T
            V = V_components @ weights
            V_components = V_components + np.fmax(-V, 0)[:, None] * rescaled_weights[None, :]
            V = np.fmax(V, 0)
        # S = S * np.exp(log_S)

    elif vol_behaviour == 'hyperplane reflection':
        for i in range(N_time):
            sq_V = np.sqrt(V)
            S = S + sq_V * S * S_BM[:, i]
            V_components = b(V_components, sq_V, dW[:, i]) @ A_inv_T
            V = V_components @ weights
            V_components = V_components + 2 * np.fmax(-V, 0)[:, None] * rescaled_weights[None, :]
            V = np.abs(V)

    elif vol_behaviour == 'adaptive':
        max_iter = 5
        A_list = []
        for i in range(1, max_iter + 1):
            A = np.diag(1 + nodes * (dt / 2 ** i)) + lambda_ * weights[None, :] * (dt / 2 ** i)
            A_list.append(np.linalg.inv(A).T)

        def nested_step(V_comp_loc, dW_loc_, iteration):
            n = len(V_comp_loc)
            if n == 0:
                return V_comp_loc
            dW_mid = np.random.normal(dW_loc_ / 2, np.sqrt(dt / 2 ** (iteration + 1)), n)

            b_ = b_comp / 2 ** iteration + V_comp_loc + (nu * np.sqrt(np.fmax(V_comp_loc @ weights, 0)) * dW_mid)[:,
                                                        None]
            V_comp_loc_updated = b_ @ A_list[iteration - 1]
            V_loc_updated = V_comp_loc_updated @ weights
            if iteration >= max_iter:
                V_comp_loc_updated = V_comp_loc_updated + np.fmax(-V_loc_updated, 0)[:, None] * rescaled_weights[None,
                                                                                                :]
            else:
                crit_ind = V_loc_updated < 0
                V_comp_loc_updated[crit_ind] = nested_step(V_comp_loc[crit_ind], dW_mid[crit_ind], iteration + 1)

            dW_second = dW_loc_ - dW_mid
            b_ = b_comp / 2 ** iteration + V_comp_loc_updated + (nu * np.sqrt(
                np.fmax(V_comp_loc_updated @ weights, 0)) * dW_second)[:, None]
            V_comp_loc_final = b_ @ A_list[iteration - 1]
            V_loc_final = V_comp_loc_final @ weights
            if iteration >= max_iter:
                V_comp_loc_final = V_comp_loc_final + np.fmax(-V_loc_final, 0)[:, None] * rescaled_weights[None, :]
            else:
                crit_ind = V_loc_final < 0
                V_comp_loc_final[crit_ind] = nested_step(V_comp_loc_updated[crit_ind], dW_second[crit_ind],
                                                         iteration + 1)
            return V_comp_loc_final

        sq_V = np.sqrt(V)
        for i in range(N_time):
            S = S + sq_V * S * S_BM[:, i]
            V_components_ = b(V_components, sq_V, dW[:, i]) @ A_inv_T
            critical_ind = V_components_ @ weights < 0
            V_components_[critical_ind] = nested_step(V_components[critical_ind], dW[critical_ind, i], 1)
            V_components = V_components_
            sq_V = np.sqrt(np.fmax(V_components @ weights, 0))

    elif vol_behaviour == 'multiple time scales':
        eta = 4.6  # e^(-4.6) approx 0.01, so only 1% of initial condition remaining relevant
        L = int(np.ceil(5 * eta))  # number of time steps per component
        n_triv_int = np.sum(eta / nodes[1:] >= dt)

        # find the local time increments
        dt_loc = np.empty(N - n_triv_int)
        dt_past = 0  # how much time of the interval [t_m, t_{m+1}] has already passed
        for j in range(n_triv_int, N):
            if j == N - 1:
                dt_loc[j - n_triv_int] = (dt - dt_past) / L
                dt_past = dt
            else:
                dt_loc[j - n_triv_int] = (dt - eta / nodes[j + 1] - dt_past) / L
                dt_past = dt - eta / nodes[j + 1]

        A_invs = []
        for j in range(n_triv_int, N - 1):
            A = (np.diag(nodes) + lambda_ * weights[None, :]) * dt_loc[j - n_triv_int]
            A[j + 1:, :] = 0
            A = A + np.eye(N)
            A_invs.append(np.linalg.inv(A).T)
        A_invs.append(A_inv)
        A_inv = A_invs.copy()

        def bs(k, vol, dw):
            res = vol
            res[:, :k + 1] = res[:, :k + 1] + ((nodes * V_original)[None, :k + 1] + theta) * dt_loc[k - n_triv_int] \
                + (nu * np.sqrt(np.fmax(vol @ weights, 0)) * dw)[:, None]
            return res

        for i in range(N_time):
            S = S + np.sqrt(V) * S * S_BM[:, i]
            dW_so_far = 0
            dt_so_far = 0
            for j in range(n_triv_int, N):
                for _ in range(L):
                    dW_loc = np.random.normal((dW[:, i] - dW_so_far) * dt_loc[j - n_triv_int] / (dt - dt_so_far),
                                              np.sqrt(dt_loc[j - n_triv_int]), m)
                    dW_so_far += dW_loc
                    dt_so_far += dt_loc[j - n_triv_int]
                    V_components = bs(j, V_components, dW_loc) @ A_inv[j - n_triv_int]
                    V = V_components @ weights
                    V_components = V_components + np.fmax(-V, 0)[:, None] * rescaled_weights[None, :]
                    V = np.fmax(V, 0)

    elif vol_behaviour == 'split kernel' or vol_behaviour == 'split throw':

        if vol_behaviour == 'split kernel':
            N = np.sum(nodes < 2 / dt)
            fast_nodes = nodes[N:]
            fast_weights = weights[N:]
            if len(fast_nodes) == 0:
                mu_a, mu_b, sr = 0, 0, np.sqrt
            else:
                nodes = nodes[:N]
                weights = weights[:N]
                mu_a, mu_b, sr = sk.smooth_root(nodes=fast_nodes, weights=fast_weights, theta=theta, lambda_=lambda_,
                                                nu=nu,
                                                us=np.linspace(-5, 5, 101), ps=np.exp(np.linspace(-10, 1, 100)), q=10,
                                                N_Riccati=10, adaptive=True, M=1000000)

                A = np.eye(N) + np.diag(nodes) * dt + lambda_ * (1 + mu_a) * weights[None, :] * dt
                A_inv = np.linalg.inv(A)
                A_inv_T = A_inv.T
        else:
            N = np.sum(nodes < 2 / dt)
            nodes = nodes[:N]
            weights = weights[:N]
            mu_a, mu_b, sr = 0, 0, np.sqrt

        b_comp = nodes[None, :] * V_components * dt + (theta - lambda_ * mu_b) * dt

        def b(vol, sr_V_, dw):
            return b_comp + vol + (nu * sr_V_ * dw)[:, None]

        sr_V = sr(V)
        for i in range(N_time):
            S = S + sr_V * S * S_BM[:, i]
            V_components = b(V_components, sr_V, dW[:, i]) @ A_inv_T
            V = V_components @ weights
            V_components = V_components + np.fmax(-V, 0)[:, None] * rescaled_weights[None, :]
            sr_V = sr(np.fmax(V, 0))

    print(time.perf_counter() - tic)
    return S


def call(K, lambda_, rho, nu, theta, V_0, H, N, S, T, m=1000, N_time=1000, WB=None, mode='best',
         vol_behaviour='sticky'):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    Uses the implicit Euler scheme.
    :param K: Strike prices, assumed to be a vector
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param H: Hurst parameter
    :param N: Number of quadrature points
    :param S: Initial stock price
    :param T: Final time/Time of maturity
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param N_time: Number of time steps used in simulation
    :param WB: Brownian increments. Has shape (2, m, N_time). First components are the dW, second the dB
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param vol_behaviour: The behaviour of the volatility at 0. The following options are possible:
        - sticky: If V becomes negative, resets V to 0 without resetting the components of V
        - constant: Uses constant volatility V_0, as in the Black-Scholes model
        - mean reversion: If V becomes negative, inserts as much fictitious time as needed until the components have
            reverted enough to ensure positive V. The fictitious time is only for the dt component, as the dW and dB
            components always have a factor containing V in front, which is interpreted to be 0 on these fictitious time
            intervals
        - hyperplane reset: If V becomes negative, resets the components just enough to ensure V=0. The components are
            reset by adding the correct multiple of the weights vector
        - hyperplane reflection: If V becomes negative, resets the components by twice as much as in hyperplane reset.
            This ensures that V changes sign and does not stay 0
        - adaptive: If V becomes negative, subdivides the last interval into two intervals. This is continued
            recursively until V is non-negative, but at most 5 consecutive times (i.e. an interval is divided into at
            most 32 intervals)
        - multiple time scales: Solves for the components on different time scales corresponding to their mean
            reversions. If V becomes negative, applies the hyperplane reset method
        - split kernel: Uses the invariant measure of the high mean-reversion components instead of actually simulating
            them. In practice, this is essentially equivalent to a certain smoothing of the square root. If V becomes
            negative, applies the hyperplane reset method
    return: The prices of the call option for the various strike prices in K
    """
    samples_ = samples(H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, m=m, S=S, N_time=N_time,
                       WB=WB, mode=mode, vol_behaviour=vol_behaviour)
    return cf.iv_eur_call_MC(S=S, K=K, T=T, samples=samples_)
