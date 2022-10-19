import numpy as np
import scipy.interpolate
import ComputationalFinance as cf
import RoughKernel as rk
import rHestonSplitKernel as sk


def sample_values(H, lambda_, rho, nu, theta, V_0, T, S_0, N=None, m=1000, N_time=1000, WB=None, mode="best",
                  vol_behaviour='sticky', nodes=None, weights=None, sample_paths=False, return_times=None):
    """
    Simulates sample paths under the Markovian approximation of the rough Heston model.
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
            reset by adding the correct multiple of the weights array
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
        - ninomiya victoir: This is actually not an implicit Euler method, but the Ninomiya-Victoir method. If the
            volatility becomes negative, applies the hyperplane reset method
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: May specify an array of times at which the sample path values should be returned. If None and
        sample_paths is True, this is equivalent to return_times = np.linspace(0, T, N_time+1)
    :return: Numpy array of the final stock prices
    """
    if vol_behaviour == 'mackevicius':
        return sample_values_mackevicius(H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, S_0=S_0, N=N,
                                         m=m, N_time=N_time, mode=mode, nodes=nodes, weights=weights,
                                         sample_paths=sample_paths, return_times=return_times)
    if return_times is not None:
        sample_paths = True
        return_times = np.fmax(return_times, 0)
        T = np.amax(return_times)
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

    S_BM = rho * dW + np.sqrt(1 - rho ** 2) * dB
    rescaled_weights = weights / np.sum(weights ** 2)
    V_comp = np.zeros((m, N))
    V_comp[:, 0] = V_0 / weights[0]
    V = V_0
    V_orig = V_comp[0, :]
    sq_V = np.sqrt(V_0)

    if vol_behaviour == 'ninomiya victoir':
        weight_sum = np.sum(weights)
        A = -(np.diag(nodes) + lambda_ * weights[None, :]) * dt / 2
        A_inverse = np.linalg.inv(A)
        b = (nodes * V_orig + theta - nu ** 2 * weight_sum / 4) * dt / 2
        expA = scipy.linalg.expm(A)
        expA_T = expA.T
        temp_1 = A_inverse @ ((expA - np.eye(N)) @ b)
        temp_2 = - dt / 2 * (
                    np.dot(weights, A_inverse @ (A_inverse @ (expA @ b - b) - b)) / 2 + nu * rho * weight_sum / 4)
        temp_3 = - dt / 2 * np.einsum('i,ij->j', weights, A_inverse @ (expA - np.eye(N))) / 2
        temp_4 = nu * weight_sum / 2
        temp_5 = temp_4 / 2

        log_S = np.log(S_0)
        if sample_paths:
            log_S = np.empty((m, N_time + 1))
            V_comp_1 = np.empty((m, N, N_time + 1))
            V_comp_1[:, :, 0] = V_comp
            V_comp = V_comp_1
            log_S[0] = np.log(S_0)

        def solve_drift_ODE(log_S_, V_comp_):
            """
            Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
            :param log_S_: Initial log-stock price
            :param V_comp_: Initial variance array
            return: Final log-stock price, final variance array, in the form log_S, V.
            """
            log_S_ = log_S_ + V_comp_ @ temp_3 + temp_2
            V_comp_ = V_comp_ @ expA_T + temp_1[None, :]
            V_comp_ = V_comp_ + np.fmax(-V_comp_ @ weights, 0)[:, None] * rescaled_weights[None, :]
            return log_S_, V_comp_

        def solve_stochastic_ODE(log_S_, V_comp_, dW_, S_BM_):
            """
            Solves the ODE corresponding to the stochastic integrals in the Ninomiya-Victoir scheme for one time step.
            Solves both vector fields (corresponding to the two Brownian motions) simultaneously, as this is possible
            in closed form.
            :param log_S_: Initial log-stock price
            :param V_comp_: Initial variance array
            :param dW_: Increment of the Brownian motion W driving the volatility process
            :param S_BM_: Increment of the Brownian motion driving the stock price process
            return: Final log-stock price, final variance array, in the form log_S, V.
            """
            total_vol = np.sqrt(np.fmax(V_comp_ @ weights, 0.))
            tau = (dW_ < 0) * np.fmin(1, - total_vol / (temp_4 * (dW_ + (dW_ == 0.) * 1))) + (dW_ >= 0)
            temp = tau * (total_vol + temp_5 * dW_ * tau)
            log_S_ = log_S_ + temp * S_BM_
            V_comp_ = V_comp_ + (nu * dW_ * temp)[:, None]
            V_comp_ = V_comp_ + np.fmax(-V_comp_ @ weights, 0)[:, None] * rescaled_weights[None, :]
            return log_S_, V_comp_

        if sample_paths:
            for i in range(N_time):
                log_S[:, i + 1], V_comp[:, :, i + 1] = solve_drift_ODE(log_S[:, i], V_comp[:, :, i])
                log_S[:, i + 1], V_comp[:, :, i + 1] = solve_stochastic_ODE(log_S[:, i + 1], V_comp[:, :, i + 1],
                                                                            dW[:, i], S_BM[:, i])
                log_S[:, i + 1], V_comp[:, :, i + 1] = solve_drift_ODE(log_S[:, i + 1], V_comp[:, :, i + 1])
            V = np.fmax(np.einsum('ijk,j->ik', V_comp, weights), 0)
            S = np.exp(log_S)
        else:
            for i in range(N_time):
                log_S, V_comp = solve_drift_ODE(log_S, V_comp)
                log_S, V_comp = solve_stochastic_ODE(log_S, V_comp, dW[:, i], S_BM[:, i])
                log_S, V_comp = solve_drift_ODE(log_S, V_comp)
            S = np.exp(log_S)

    elif vol_behaviour == 'correct ninomiya victoir':
        weight_sum = np.sum(weights)
        A = -(np.diag(nodes) + lambda_ * weights[None, :]) * dt / 2
        A_inverse = np.linalg.inv(A)
        b = (nodes * V_orig + theta - nu ** 2 * weight_sum / 4) * dt / 2
        expA = scipy.linalg.expm(A)
        expA_T = expA.T
        temp_1 = A_inverse @ ((expA - np.eye(N)) @ b)
        temp_2 = - dt / 2 * (
                    np.dot(weights, A_inverse @ (A_inverse @ (expA @ b - b) - b)) / 2 + nu * rho * weight_sum / 4)
        temp_3 = - dt / 2 * np.einsum('i,ij->j', weights, A_inverse @ (expA - np.eye(N))) / 2
        temp_4 = nu * weight_sum / 2
        temp_5 = temp_4 / 2

        log_S = np.log(S_0)
        if sample_paths:
            log_S = np.empty((m, N_time + 1))
            V_comp_1 = np.empty((m, N, N_time + 1))
            V_comp_1[:, :, 0] = V_comp
            V_comp = V_comp_1
            log_S[0] = np.log(S_0)

        def solve_drift_ODE(log_S_, V_comp_):
            """
            Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
            :param log_S_: Initial log-stock price
            :param V_comp_: Initial variance array
            return: Final log-stock price, final variance array, in the form log_S, V.
            """
            log_S_ = log_S_ + V_comp_ @ temp_3 + temp_2
            V_comp_ = V_comp_ @ expA_T + temp_1[None, :]
            V_comp_ = V_comp_ + np.fmax(-V_comp_ @ weights, 0)[:, None] * rescaled_weights[None, :]
            return log_S_, V_comp_

        def solve_dB_ODE(log_S_, V_comp_, dB_):
            """
            Solves the ODE corresponding to the dB integral in the Ninomiya-Victoir scheme for one time step.
            :param log_S_: Initial log-stock price
            :param V_comp_: Initial variance array
            :param dB_: Increment of the Brownian motion W driving the stock price process
            return: Final log-stock price, final variance array, in the form log_S, V.
            """
            total_vol = np.sqrt(np.fmax(V_comp_ @ weights, 0.))
            return log_S_ + total_vol * np.sqrt(1-rho**2) * dB_, V_comp_

        def solve_dW_ODE(log_S_, V_comp_, dW_):
            """
            Solves the ODE corresponding to the dW integral in the Ninomiya-Victoir scheme for one time step.
            :param log_S_: Initial log-stock price
            :param V_comp_: Initial variance array
            :param dW_: Increment of the Brownian motion W driving the volatility process
            return: Final log-stock price, final variance array, in the form log_S, V.
            """
            total_vol = np.sqrt(np.fmax(V_comp_ @ weights, 0.))
            tau = (dW_ < 0) * np.fmin(1, - total_vol / (temp_4 * (dW_ + (dW_ == 0.) * 1))) + (dW_ >= 0)
            temp = tau * (total_vol + temp_5 * dW_ * tau)
            log_S_ = log_S_ + temp * rho * dW_
            V_comp_ = V_comp_ + (nu * dW_ * temp)[:, None]
            V_comp_ = V_comp_ + np.fmax(-V_comp_ @ weights, 0)[:, None] * rescaled_weights[None, :]
            return log_S_, V_comp_

        if sample_paths:
            for i in range(N_time):
                log_S[:, i + 1], V_comp[:, :, i + 1] = solve_drift_ODE(log_S[:, i], V_comp[:, :, i])
                indices = np.random.binomial(1, 0.5, len(log_S)) == 0
                log_S[indices, i + 1], V_comp[indices, :, i + 1] = solve_dB_ODE(log_S[indices, i + 1],
                                                                                V_comp[indices, :, i + 1],
                                                                                dB[indices, i])
                log_S[:, i + 1], V_comp[:, :, i + 1] = solve_dW_ODE(log_S[:, i + 1], V_comp[:, :, i + 1], dW[:, i])
                log_S[~indices, i + 1], V_comp[~indices, :, i + 1] = solve_dB_ODE(log_S[~indices, i + 1],
                                                                                  V_comp[~indices, :, i + 1],
                                                                                  dB[~indices, i])
                log_S[:, i + 1], V_comp[:, :, i + 1] = solve_drift_ODE(log_S[:, i + 1], V_comp[:, :, i + 1])
            V = np.fmax(np.einsum('ijk,j->ik', V_comp, weights), 0)
            S = np.exp(log_S)
        else:
            for i in range(N_time):
                log_S, V_comp = solve_drift_ODE(log_S, V_comp)
                indices = np.random.binomial(1, 0.5, len(log_S)) == 0
                log_S[indices], V_comp[indices, :] = solve_dB_ODE(log_S[indices], V_comp[indices, :], dB[indices, i])
                log_S, V_comp = solve_dW_ODE(log_S, V_comp, dW[:, i])
                log_S[~indices], V_comp[~indices, :] = solve_dB_ODE(log_S[~indices], V_comp[~indices, :],
                                                                    dB[~indices, i])
                # log_S, V_comp = solve_stochastic_ODE(log_S, V_comp, dW[:, i], S_BM[:, i])
                log_S, V_comp = solve_drift_ODE(log_S, V_comp)
            S = np.exp(log_S)

    elif vol_behaviour == 'mackevicius':
        weight_sum = np.sum(weights)
        A = -(np.diag(nodes) + lambda_ * weights[None, :]) * dt / 2
        expA = scipy.linalg.expm(A)
        b = (nodes * V_orig + theta) * dt / 2
        ODE_b = np.linalg.solve(A, (expA - np.eye(N)) @ b)
        z = weight_sum ** 2 * nu ** 2 * dt

        def ODE_step_V(V):
            return expA @ V + ODE_b[:, None]

        def SDE_step_V(V):
            x = weights @ V
            rv = np.random.uniform(0, 1, len(x))
            m_1 = x
            m_2 = x * (x + z)
            m_3 = x * (x * (x + 3 * z) + 1.5 * z ** 2)
            B = (6 + np.sqrt(3)) / 4
            temp = np.sqrt((3 * x + B ** 2 * z) * z)
            x_1 = x + B * z - temp
            x_3 = x + B * z + temp
            x_2 = x + (B - 0.75) * z
            p_1 = (m_1 * x_2 * x_3 - m_2 * (x_2 + x_3) + m_3) / (x_1 * (x_3 - x_1) * (x_2 - x_1))
            p_2 = (m_1 * x_1 * x_3 - m_2 * (x_1 + x_3) + m_3) / (x_2 * (x_3 - x_2) * (x_1 - x_2))
            # p_3 = (m_1 * x_1 * x_2 - m_2 * (x_1 + x_2) + m_3) / (x_3 * (x_2 - x_3) * (x_1 - x_3))
            test_1 = rv < p_1
            test_2 = p_1 + p_2 <= rv
            x_after = test_1 * x_1 + np.logical_and(np.logical_not(test_1), np.logical_not(test_2)) * x_2 + test_2 * x_3
            y = (x_after - x) / weight_sum
            return V + y[None, :]

        def step_V(V):
            return ODE_step_V(SDE_step_V(ODE_step_V(V)))

        def SDE_step_B(S, V):
            x = weights @ V
            return S + np.sqrt(x) * np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), len(x)) - 0.5 * x * (1 - rho ** 2) * dt, V

        drift_SDE_step_W = - (nodes[0] * V_orig[0] + theta) * dt

        def SDE_step_W(S, V):
            V_new = step_V(V)
            dY = (V + V_new) * (dt / 2)
            S_new = S + rho / nu * (drift_SDE_step_W + nodes[0] * dY[0, :] + (lambda_ - 0.5 * rho * nu) * (weights @ dY)
                                    + (V_new[0, :] - V[0, :]))
            return S_new, V_new

        def step_SV(S, V):
            rv = np.random.binomial(1, 0.5, len(S))
            ind_1 = np.where(rv == 0)[0]
            ind_2 = np.where(rv == 1)[0]
            S[ind_1], V[:, ind_1] = SDE_step_B(*SDE_step_W(S[ind_1], V[:, ind_1]))
            S[ind_2], V[:, ind_2] = SDE_step_W(*SDE_step_B(S[ind_2], V[:, ind_2]))
            return S, V

        V_comp = V_comp.T
        if sample_paths:
            log_S = np.empty((m, N_time + 1))
            V_comp_1 = np.empty((N, m, N_time + 1))
            V_comp_1[:, :, 0] = V_comp
            V_comp = V_comp_1
            log_S[0] = np.log(S_0)
            for i in range(N_time):
                log_S[:, i + 1], V_comp[:, :, i + 1] = step_SV(log_S[:, i], V_comp[:, :, i])
            V = np.fmax(weights @ V_comp, 0)
            S = np.exp(log_S)
        else:
            log_S = np.ones(m) * np.log(S_0)
            for i in range(N_time):
                log_S, V_comp = step_SV(log_S, V_comp)
            S = np.exp(log_S)

    else:  # Some implicit Euler scheme
        A = np.eye(N) + np.diag(nodes) * dt + lambda_ * weights[None, :] * dt
        A_inv_T = np.linalg.inv(A).T

        b_comp = theta * dt + (nodes * V_orig)[None, :] * dt

        def S_euler_step(S_, sq_V_, S_BM_):
            return S_ + sq_V_ * S_ * S_BM_

        def V_comp_euler_step(V_comp_, sq_V_, dW_):
            return (V_comp_ + nu * (sq_V_ * dW_)[:, None] + b_comp) @ A_inv_T

        S = S_0
        if sample_paths:
            S = np.zeros((m, N_time + 1))
            V = np.zeros((m, N_time + 1))
            S[:, 0] = S_0
            V[:, 0] = V_0
            V_comp = np.zeros(shape=(m, N, N_time + 1))
            V_comp[:, 0, 0] = V_0 / weights[0]

        if vol_behaviour == 'sticky':

            def sticky(S_, sq_V_, S_BM_, V_comp_, dW_):
                S_ = S_euler_step(S_, sq_V_, S_BM_)
                V_comp_ = V_comp_euler_step(V_comp_, sq_V_, dW_)
                sq_V_ = np.sqrt(np.fmax(V_comp_ @ weights, 0))
                return S_, V_comp_, sq_V_

            if sample_paths:
                V[:, 0] = np.sqrt(V[:, 0])
                for i in range(N_time):
                    S[:, i + 1], V_comp[:, :, i + 1], V[:, i + 1] = sticky(S_=S[:, i], sq_V_=V[:, i], S_BM_=S_BM[:, i],
                                                                           V_comp_=V_comp[:, :, i], dW_=dW[:, i])
                V = V**2
            else:
                for i in range(N_time):
                    S, V_comp, sq_V = sticky(S_=S, sq_V_=sq_V, S_BM_=S_BM[:, i], V_comp_=V_comp, dW_=dW[:, i])

        elif vol_behaviour == 'constant':
            if sample_paths:
                for i in range(N_time):
                    S[:, i + 1] = S_euler_step(S[:, i], sq_V, S_BM[:, i])
                V_comp[:, :, :] = V_comp[:, :, 0][:, :, None]
                V[:, :] = V_0
            else:
                for i in range(N_time):
                    S = S_euler_step(S, sq_V, S_BM[:, i])

        elif vol_behaviour == 'mean reversion':

            def mean_reversion(S_, V_, S_BM_, V_comp_, dW_, zero_vol_):
                sq_V_ = np.sqrt(V_)
                S_ = S_euler_step(S_, sq_V_, S_BM_)
                V_comp_ = V_comp_euler_step(V_comp_, sq_V_, dW_)
                V_ = np.fmax(V_comp_ @ weights, 0)
                sq_V_ = np.sqrt(V_)
                if np.sum(zero_vol_) > 0:
                    relevant_ind = np.logical_and(zero_vol_, V_ <= 0)
                    while np.sum(relevant_ind) > 0:
                        V_comp_[relevant_ind, :] = V_comp_euler_step(V_comp_[relevant_ind, :], sq_V_[relevant_ind],
                                                                     dW_[relevant_ind])
                        V_[relevant_ind] = np.fmax(V_comp_[relevant_ind, :] @ weights, 0)
                        sq_V_[relevant_ind] = np.sqrt(V_[relevant_ind])
                        relevant_ind = np.logical_and(zero_vol_, V_ <= 0)
                return S_, V_comp_, V_, zero_vol_

            zero_vol = V_0 <= 0
            if sample_paths:
                for i in range(N_time):
                    S[:, i + 1], V_comp[:, :, i + 1], V[:, i + 1], zero_vol = mean_reversion(S_=S[:, i], V_=V[:, i],
                                                                                             S_BM_=S_BM[:, i],
                                                                                             V_comp_=V_comp[:, :, i],
                                                                                             dW_=dW[:, i],
                                                                                             zero_vol_=zero_vol)
            else:
                V = V_0
                for i in range(N_time):
                    S, V_comp, V, zero_vol = mean_reversion(S_=S, V_=V, S_BM_=S_BM[:, i], V_comp_=V_comp, dW_=dW[:, i],
                                                            zero_vol_=zero_vol)

        elif vol_behaviour == 'hyperplane reset':

            def hyperplane_reset(S_, V_, S_BM_, V_comp_, dW_):
                sq_V_ = np.sqrt(V_)
                S_ = S_euler_step(S_, sq_V_, S_BM_)
                V_comp_ = V_comp_euler_step(V_comp_, sq_V_, dW_)
                V_ = V_comp_ @ weights
                V_comp_ = V_comp_ + np.fmax(-V_, 0)[:, None] * rescaled_weights[None, :]
                V_ = np.fmax(V_, 0)
                return S_, V_comp_, V_

            if sample_paths:
                for i in range(N_time):
                    S[:, i + 1], V_comp[:, :, i + 1], V[:, i + 1] = hyperplane_reset(S_=S[:, i], V_=V[:, i],
                                                                                     S_BM_=S_BM[:, i],
                                                                                     V_comp_=V_comp[:, :, i],
                                                                                     dW_=dW[:, i])
            else:
                for i in range(N_time):
                    S, V_comp, V = hyperplane_reset(S_=S, V_=V, S_BM_=S_BM[:, i], V_comp_=V_comp, dW_=dW[:, i])

        elif vol_behaviour == 'hyperplane reflection':

            def hyperplane_reflection(S_, V_, S_BM_, V_comp_, dW_):
                sq_V_ = np.sqrt(V_)
                S_ = S_euler_step(S_, sq_V_, S_BM_)
                V_comp_ = V_comp_euler_step(V_comp_, sq_V_, dW_)
                V_ = V_comp_ @ weights
                V_comp_ = V_comp_ + 2 * np.fmax(-V_, 0)[:, None] * rescaled_weights[None, :]
                V_ = np.abs(V_)
                return S_, V_comp_, V_

            if sample_paths:
                for i in range(N_time):
                    S[:, i + 1], V_comp[:, :, i + 1], V[:, i + 1] = hyperplane_reflection(S_=S[:, i], V_=V[:, i],
                                                                                          S_BM_=S_BM[:, i],
                                                                                          V_comp_=V_comp[:, :, i],
                                                                                          dW_=dW[:, i])
            else:
                for i in range(N_time):
                    S, V_comp, V = hyperplane_reflection(S_=S, V_=V, S_BM_=S_BM[:, i], V_comp_=V_comp, dW_=dW[:, i])

        elif vol_behaviour == 'adaptive':
            max_iter = 5
            A_list = [A_inv_T]
            for i in range(1, max_iter + 1):
                A = np.diag(1 + nodes * (dt / 2 ** i)) + lambda_ * weights[None, :] * (dt / 2 ** i)
                A_list.append(np.linalg.inv(A).T)

            def nested_step(V_comp_loc, dW_loc_, iteration):
                n = len(V_comp_loc)
                if n == 0:
                    return V_comp_loc
                dW_first = np.random.normal(dW_loc_ / 2, np.sqrt(dt / 2 ** (iteration + 1)), n)

                b_ = b_comp / 2 ** iteration + V_comp_loc \
                    + (nu * np.sqrt(np.fmax(V_comp_loc @ weights, 0)) * dW_first)[:, None]
                V_comp_loc_upd = b_ @ A_list[iteration]
                V_loc_upd = V_comp_loc_upd @ weights
                if iteration >= max_iter:
                    # V_comp_loc_upd = V_comp_loc_upd + np.fmax(-V_loc_upd, 0)[:, None] * rescaled_weights[None, :]
                    pass
                else:
                    crit_ind = V_loc_upd < 0
                    V_comp_loc_upd[crit_ind, :] = nested_step(V_comp_loc[crit_ind, :], dW_first[crit_ind],
                                                              iteration + 1)

                dW_second = dW_loc_ - dW_first
                b_ = b_comp / 2 ** iteration + V_comp_loc_upd + (nu * np.sqrt(
                    np.fmax(V_comp_loc_upd @ weights, 0)) * dW_second)[:, None]
                V_comp_loc_final = b_ @ A_list[iteration]
                V_loc_final = V_comp_loc_final @ weights
                if iteration >= max_iter:
                    # V_comp_loc_final = V_comp_loc_final + np.fmax(-V_loc_final, 0)[:, None] * rescaled_weights[None, :]
                    pass
                else:
                    crit_ind = V_loc_final < 0
                    V_comp_loc_final[crit_ind, :] = nested_step(V_comp_loc_upd[crit_ind, :], dW_second[crit_ind],
                                                                iteration + 1)
                return V_comp_loc_final

            def adaptive(S_, V_, S_BM_, V_comp_, dW_):
                sq_V_ = np.sqrt(V_)
                S_ = S_euler_step(S_, sq_V_, S_BM_)
                V_comp_2 = V_comp_euler_step(V_comp_, sq_V_, dW_)
                critical_ind = V_comp_2 @ weights < 0
                V_comp_2[critical_ind, :] = nested_step(V_comp_[critical_ind, :], dW_[critical_ind], 1)
                V_comp_ = V_comp_2
                V_ = np.fmax(V_comp @ weights, 0)
                return S_, V_comp_, V_

            if sample_paths:
                for i in range(N_time):
                    S[:, i + 1], V_comp[:, :, i + 1], V[:, i + 1] = adaptive(S_=S[:, i], V_=V[:, i], S_BM_=S_BM[:, i],
                                                                             V_comp_=V_comp[:, :, i+1], dW_=dW[:, i])
            else:
                for i in range(N_time):
                    S, V_comp, V = adaptive(S_=S, V_=V, S_BM_=S_BM[:, i], V_comp_=V_comp, dW_=dW[:, i])

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
            A_invs.append(A_inv_T)
            A_inv = A_invs.copy()

            def bs(k, vol, dw):
                res = vol
                res[:, :k + 1] = res[:, :k + 1] + ((nodes * V_orig)[None, :k + 1] + theta) * dt_loc[k - n_triv_int] \
                    + (nu * np.sqrt(np.fmax(vol @ weights, 0)) * dw)[:, None]
                return res

            def multiple_time_scales(S_, V_, S_BM_, V_comp_, dW_):
                S_ = S_euler_step(S_, np.sqrt(V_), S_BM_)
                dW_so_far = 0
                dt_so_far = 0
                for k in range(n_triv_int, N):
                    for _ in range(L):
                        dW_loc = np.random.normal((dW_ - dW_so_far) * dt_loc[k - n_triv_int] / (dt - dt_so_far),
                                                  np.sqrt(dt_loc[k - n_triv_int]), m)
                        dW_so_far += dW_loc
                        dt_so_far += dt_loc[k - n_triv_int]
                        V_comp_ = bs(k, V_comp_, dW_loc) @ A_inv[k - n_triv_int]
                        V_ = V_comp_ @ weights
                        V_comp_ = V_comp_ + np.fmax(-V_, 0)[:, None] * rescaled_weights[None, :]
                        V_ = np.fmax(V_, 0)
                return S_, V_comp_, V_

            if sample_paths:
                for i in range(N_time):
                    S[:, i + 1], V_comp[:, i + 1], V[:, i + 1] = multiple_time_scales(S_=S[:, i], V_=V[:, i],
                                                                                      S_BM_=S_BM[:, i],
                                                                                      V_comp_=V_comp[:, :, i],
                                                                                      dW_=dW[:, i])
            else:
                for i in range(N_time):
                    S, V_comp, V = multiple_time_scales(S_=S, V_=V, S_BM_=S_BM[:, i], V_comp_=V_comp, dW_=dW[:, i])

        elif vol_behaviour == 'split kernel' or vol_behaviour == 'split throw':

            if vol_behaviour == 'split kernel':
                N = int(np.sum(nodes < 2 / dt))
                fast_nodes = nodes[N:]
                fast_weights = weights[N:]
                if len(fast_nodes) == 0:
                    mu_a, mu_b, sr = 0, 0, np.sqrt
                else:
                    nodes = nodes[:N]
                    weights = weights[:N]
                    mu_a, mu_b, sr = sk.smooth_root(nodes=fast_nodes, weights=fast_weights, theta=theta,
                                                    lambda_=lambda_, nu=nu, us=np.linspace(-5, 5, 101),
                                                    ps=np.exp(np.linspace(-10, 1, 100)), q=10, N_Riccati=10,
                                                    adaptive=True, M=1000000)

                    A = np.eye(N) + np.diag(nodes) * dt + lambda_ * (1 + mu_a) * weights[None, :] * dt
                    A_inv_T = np.linalg.inv(A).T
            else:
                N = np.sum(nodes < 2 / dt)
                nodes = nodes[:N]
                weights = weights[:N]
                mu_a, mu_b, sr = 0, 0, np.sqrt

            b_comp = nodes[None, :] * V_comp * dt + (theta - lambda_ * mu_b) * dt

            def V_comp_euler_step(vol, sr_V_, dw):
                return (b_comp + vol + (nu * sr_V_ * dw)[:, None]) @ A_inv_T

            def split_whatever(S_, V_, S_BM_, V_comp_, dW_):
                sr_V = sr(V_)
                S_ = S_euler_step(S_, sr_V, S_BM_)
                V_comp_ = V_comp_euler_step(V_comp_, sr_V, dW_)
                V_ = V_comp_ @ weights
                V_comp_ = V_comp_ + np.fmax(-V_, 0)[:, None] * rescaled_weights[None, :]
                return S_, V_comp_, V_

            if sample_paths:
                for i in range(N_time):
                    S[:, i + 1], V_comp[:, i + 1], V[:, i + 1] = split_whatever(S_=S[:, i], V_=V[:, i],
                                                                                S_BM_=S_BM[:, i],
                                                                                V_comp_=V_comp[:, :, i], dW_=dW[:, i])
            for i in range(N_time):
                S, V_comp, V = split_whatever(S_=S, V_=V, S_BM_=S_BM[:, i], V_comp_=V_comp, dW_=dW[:, i])

    if sample_paths:
        if return_times is not None:
            times = np.linspace(0, T, N_time+1)
            S = scipy.interpolate.interp1d(x=times, y=S)(return_times)
            V = scipy.interpolate.interp1d(x=times, y=V)(return_times)
            V_comp = scipy.interpolate.interp1d(x=times, y=V_comp)(return_times)
        return S, np.sqrt(V), V_comp
    return S


def sample_values_mackevicius(H, lambda_, rho, nu, theta, V_0, T, S_0, N=None, m=1000, N_time=1000, mode="best",
                              nodes=None, weights=None, sample_paths=False, return_times=None):
    """
    Simulates sample paths under the Markovian approximation of the rough Heston model, using a combination of
    Mackevicius and Alfonsi.
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
    :param m: Number of samples. If WB is specified, uses as many samples as WB contains, regardless of the parameter m
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: May specify an array of times at which the sample path values should be returned. If None and
        sample_paths is True, this is equivalent to return_times = np.linspace(0, T, N_time+1)
    :return: Numpy array of the final stock prices
    """
    if return_times is not None:
        sample_paths = True
        return_times = np.fmax(return_times, 0)
        T = np.amax(return_times)
    dt = T / N_time
    if nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode=mode)
    N = len(nodes)
    if N == 1:
        nodes = np.array([nodes[0], 1])
        weights = np.array([weights[0], 0])
        N = 2

    V_comp = np.zeros((N, m))
    V_comp[0, :] = V_0 / weights[0]
    V = V_0
    V_orig = V_comp[:, 0]

    weight_sum = np.sum(weights)
    A = -(np.diag(nodes) + lambda_ * weights[None, :]) * dt / 2
    exp_A = scipy.linalg.expm(A)
    b = (nodes * V_orig + theta) * dt / 2
    ODE_b = np.linalg.solve(A, (exp_A - np.eye(N)) @ b)
    z = weight_sum ** 2 * nu ** 2 * dt
    rho_bar_sq = 1 - rho ** 2
    rho_bar = np.sqrt(rho_bar_sq)

    def ODE_step_V(V):
        return exp_A @ V + ODE_b[:, None]

    def SDE_step_V(V):
        x = weights @ V
        rv = np.random.uniform(0, 1, len(x))
        m_1 = x
        m_2 = x * (x + z)
        m_3 = x * (x * (x + 3 * z) + 1.5 * z ** 2)
        B = (6 + np.sqrt(3)) / 4
        temp = np.sqrt((3 * x + B ** 2 * z) * z)
        x_1 = x + B * z - temp
        x_3 = x + B * z + temp
        x_2 = x + (B - 0.75) * z
        p_1 = (m_1 * x_2 * x_3 - m_2 * (x_2 + x_3) + m_3) / (x_1 * (x_3 - x_1) * (x_2 - x_1))
        p_2 = (m_1 * x_1 * x_3 - m_2 * (x_1 + x_3) + m_3) / (x_2 * (x_3 - x_2) * (x_1 - x_2))
        # p_3 = (m_1 * x_1 * x_2 - m_2 * (x_1 + x_2) + m_3) / (x_3 * (x_2 - x_3) * (x_1 - x_3))
        test_1 = rv < p_1
        test_2 = p_1 + p_2 <= rv
        x_after = test_1 * x_1 + np.logical_and(np.logical_not(test_1), np.logical_not(test_2)) * x_2 + test_2 * x_3
        y = (x_after - x) / weight_sum
        return V + y[None, :]

    def step_V(V):
        return ODE_step_V(SDE_step_V(ODE_step_V(V)))

    def SDE_step_B(S, V):
        x = weights @ V
        return S + np.sqrt(x) * rho_bar * np.random.normal(0, np.sqrt(dt), len(x)) - 0.5 * x * rho_bar_sq * dt, V

    drift_SDE_step_W = - (nodes[0] * V_orig[0] + theta) * dt
    fact_1 = lambda_ - 0.5 * rho * nu

    def SDE_step_W(S, V):
        V_new = step_V(V)
        dY = (V + V_new) * (dt / 2)
        S_new = S + rho / nu * (drift_SDE_step_W + nodes[0] * dY[0, :] + fact_1 * (weights @ dY)
                                + (V_new[0, :] - V[0, :]))
        return S_new, V_new

    def step_SV(S, V):
        rv = np.random.binomial(1, 0.5, len(S))
        ind_1 = np.where(rv == 0)[0]
        ind_2 = np.where(rv == 1)[0]
        S[ind_1], V[:, ind_1] = SDE_step_B(*SDE_step_W(S[ind_1], V[:, ind_1]))
        S[ind_2], V[:, ind_2] = SDE_step_W(*SDE_step_B(S[ind_2], V[:, ind_2]))
        return S, V

    if sample_paths:
        log_S = np.empty((m, N_time + 1))
        V_comp_1 = np.empty((N, m, N_time + 1))
        V_comp_1[:, :, 0] = V_comp
        V_comp = V_comp_1
        log_S[0] = np.log(S_0)
        for i in range(N_time):
            log_S[:, i + 1], V_comp[:, :, i + 1] = step_SV(log_S[:, i], V_comp[:, :, i])
        V = np.fmax(weights @ V_comp, 0)
        S = np.exp(log_S)
    else:
        log_S = np.ones(m) * np.log(S_0)
        for i in range(N_time):
            log_S, V_comp = step_SV(log_S, V_comp)
        S = np.exp(log_S)

    if sample_paths:
        if return_times is not None:
            times = np.linspace(0, T, N_time+1)
            S = scipy.interpolate.interp1d(x=times, y=S)(return_times)
            V = scipy.interpolate.interp1d(x=times, y=V)(return_times)
            V_comp = scipy.interpolate.interp1d(x=times, y=V_comp)(return_times)
        return S, np.sqrt(V), V_comp
    return S


def call(K, lambda_, rho, nu, theta, V_0, H, N, S_0, T, m=1000, N_time=1000, WB=None, mode='best',
         vol_behaviour='sticky'):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    Uses the implicit Euler scheme.
    :param K: Strike prices, assumed to be a numpy array
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param H: Hurst parameter
    :param N: Number of quadrature points
    :param S_0: Initial stock price
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
            reset by adding the correct multiple of the weights array
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
        - ninomiya victoir: This is actually not an implicit Euler method, but the Ninomiya-Victoir method. If the
            volatility becomes negative, applies the hyperplane reset method
    return: The prices of the call option for the various strike prices in K
    """
    samples_ = sample_values(H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, m=m, S_0=S_0,
                             N_time=N_time, WB=WB, mode=mode, vol_behaviour=vol_behaviour)
    return cf.iv_eur_call_MC(S_0=S_0, K=K, T=T, samples=samples_)
