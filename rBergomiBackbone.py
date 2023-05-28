import numpy as np
import ComputationalFinance as cf
import time


def iv_eur_call(sample_generator, S_0, K, T, rel_tol, verbose=0):
    """
    Computes the implied volatility of the European call option under the rough Bergomi model.
    :param sample_generator: Function that generates samples of the rough Bergomi model at the final time. Is a
        function of the maturity, the number of time discretization steps and the number of samples that should be
        generated
    :param S_0: Initial stock price
    :param T: Final time
    :param K: The strike price
    :param rel_tol: Relative error tolerance
    :param verbose: Determines the number of intermediary results printed to the console
    :return: The implied volatility and a 95% confidence interval, in the form (estimate, lower, upper)
    """

    def single_smile(K_, T_, sample_generator_):
        """
        Gives the implied volatility of the European call option in the rough Heston model as described in El Euch and
        Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme. Uses Fourier inversion.
        :param K_: Strike price, assumed to be a numpy array
        :param sample_generator_: Function that generates samples of the rough Bergomi model at the final time. Is a
            only function of the number of time discretization steps and the number of samples that should be generated
        :param T_: Maturity
        return: The price of the call option
        """
        N = 100
        M = 200000
        np.seterr(all='warn')

        def compute_iv(N_):
            samples = sample_generator_(N=N_, M=M)
            return cf.eur_MC(S_0=S_0, K=K_, T=T_, samples=samples, payoff='call', implied_vol=True)

        tic = time.perf_counter()
        iv, l, u = compute_iv(N_=N)
        duration = time.perf_counter() - tic
        iv_approx, l_approx, u_approx = compute_iv(N_=N // 2)
        mc_error = np.fmax(np.amax((u - iv) / iv), np.amax((iv - l) / l))
        mc_error_approx = np.fmax(np.amax((u_approx - iv_approx) / iv_approx),
                                  np.amax((iv_approx - l_approx) / l_approx))
        discr_error = np.amax(np.abs(iv - iv_approx)/iv)
        total_error = (1 + mc_error) * (1 + mc_error_approx) * (1 + discr_error) - 1
        # print(np.abs(iv_approx - iv) / iv)

        while np.isnan(total_error) or np.sum(np.isnan(iv)) > 0 or np.sum(iv < 1e-06) > 0 or total_error > rel_tol:
            if np.isnan(total_error) or np.sum(np.isnan(iv)) > 0 or np.sum(iv < 1e-06) > 0:
                N = N * 3 // 2
                M = 2 * M
            else:
                if mc_error > discr_error / 3:
                    M = 2 * M
                elif mc_error < rel_tol / 6:
                    M = np.fmax(M // 2, 10)
                if discr_error > mc_error / 2:
                    N = N * 3 // 2
                else:
                    N = np.fmax(N // 3 * 2, 10)

            if verbose >= 1:
                print(total_error, discr_error, mc_error, mc_error_approx, N, M, duration,
                      time.strftime("%H:%M:%S", time.localtime()))
            iv_approx, l_approx, u_approx, mc_error_approx = iv, l, u, mc_error

            tic = time.perf_counter()
            iv, l, u = compute_iv(N_=N)
            duration = time.perf_counter() - tic
            mc_error = np.fmax(np.amax((u - iv) / iv), np.amax((iv - l) / l))
            discr_error = np.amax(np.abs(iv - iv_approx)/iv)
            total_error = (1 + mc_error) * (1 + mc_error_approx) * (1 + discr_error) - 1
            # print(np.abs(iv_approx - iv) / iv)

        if verbose >= 1:
            print(total_error, discr_error, mc_error, mc_error_approx, N, M, duration,
                  time.strftime("%H:%M:%S", time.localtime()))

        return iv, l, u

    T_is_float = False
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        T_is_float = True
    if len(K.shape) == 1:
        _, K = cf.maturity_tensor_strike(S_0=S_0, K=K, T=T)

    iv_surface = np.empty_like(K)
    lower_surface = np.empty_like(K)
    upper_surface = np.empty_like(K)
    for i in range(len(T)):
        if verbose >= 1:
            print(f'Now simulating maturity {i+1} of {len(T)}')
        iv_surface[i, :], lower_surface[i, :], upper_surface[i, :] = \
            single_smile(K_=K[i, :], T_=T[i], sample_generator_=lambda N, M: sample_generator(T[i], N, M))
    if T_is_float:
        iv_surface, lower_surface, upper_surface = iv_surface[0, :], lower_surface[0, :], upper_surface[0, :]
    return iv_surface, lower_surface, upper_surface
