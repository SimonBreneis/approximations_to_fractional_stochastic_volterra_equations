import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import ComputationalFinance as cf
import Data
import rBergomi
import rBergomiMarkov
import rHeston
import rHestonMarkov
import RoughKernel as rk
import rHestonMarkovSamplePaths as rHestonSP
import rHestonSplitKernel as sk
from os.path import exists

c = ['r', 'C1', 'y', 'g', 'b', 'purple']
c_ = ['darkred', 'r', 'C1', 'y', 'lime', 'g', 'deepskyblue', 'b', 'purple', 'deeppink']
c_short = ['r', 'g', 'b']
simple_params = {'S': 1., 'K': np.exp(np.linspace(-1.5, 0.75, 451)), 'H': 0.1, 'T': 1., 'lambda': 0.3, 'rho': -0.7,
                 'nu': 0.3, 'theta': 0.02, 'V_0': 0.02, 'rel_tol': 1e-05}
difficult_params = {'S': 1., 'K': np.exp(np.linspace(-0.5, 0.1, 181)), 'H': 0.07, 'T': 0.04, 'lambda': 0.6, 'rho': -0.8,
                    'nu': 0.5, 'theta': 0.01, 'V_0': 0.01, 'rel_tol': 1e-05}


def rHeston_params(params):
    if params is None or params == 'simple':
        return simple_params
    if params == 'difficult':
        return difficult_params
    for key in simple_params:
        params[key] = params.get(key, simple_params[key])
    return params


def color(i, N):
    if N <= 3:
        return c_short[i % 3]
    if N <= 6 or N == 11 or N == 12:
        return c[i % 6]
    return c_[i % 10]


def label(mode):
    if mode == 'observation':
        return 'Paper'
    return mode[0].upper() + mode[1:]


def load_WB(i):
    dW = np.load(f'dW{i}.npy')
    dB = np.load(f'dB{i}.npy')
    WB = np.empty((2, dW.shape[0], dW.shape[1]))
    WB[0, :, :] = dW
    WB[1, :, :] = dB
    return WB


def coarsen_WB(WB):
    return WB[:, :, ::2] + WB[:, :, 1::2]


def refine_WB(WB):
    WB_ = np.empty((WB.shape[0], WB.shape[1], 2*WB.shape[2]))
    incr = np.random.normal(0, 1 / np.sqrt(4*WB.shape[2]), WB.shape)
    WB_[:, :, ::2] = WB / 2 + incr
    WB_[:, :, 1::2] = WB / 2 - incr
    return WB_


def resize_WB(WB, N_time):
    while WB.shape[2] > N_time:
        WB = coarsen_WB(WB)
    while WB.shape[2] < N_time:
        WB = refine_WB(WB)
    return WB


def set_array_default(x, default):
    if x is None:
        x = default
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    return x


def set_list_default(x, default):
    if x is None:
        x = default
    if not isinstance(x, list):
        x = [x]
    return x


def get_filename(N, mode, vol_behaviour, N_time, sample_paths):
    if sample_paths:
        return f'rHeston sample paths {N} dim {mode} {vol_behaviour} {N_time} time steps.npy'
    return f'rHeston samples {N} dim {mode} {vol_behaviour} {N_time} time steps.npy'


def log_linear_regression(x, y):
    """
    Applies log-linear regression of y against x.
    :param x: The argument
    :param y: The function value
    :return: Exponent, constant, R-value, p-value, empirical standard deviation
    """
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(x), np.log(y))
    return slope, np.exp(intercept), r_value, p_value, std_err


def plot_log_linear_regression(x, y, col, lab, kind='--', offset=0):
    power, constant, r_value, p_value, std_err = log_linear_regression(x[offset:], y[offset:])
    plt.loglog(x, y, color=col, label=lab)
    x_alt = np.exp(np.linspace(np.amin(x), np.amax(x), 5))  # could use just x, but this is to avoid excessively many
    # dots or crosses, etc., if kind is not '--'
    plt.loglog(x_alt, constant * x_alt ** power, kind, color=col)
    return power, constant


def kernel_errors(H, T, Ns=None, modes=None):
    """
    Prints the largest nodes, the relative errors and the computational times for the strong L^2-error approximation
    of the fractional kernel.
    :param H: Hurst parameter
    :param T: Final time, may also be a vector
    :param Ns: Vector of values for N. Default is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024]
    :param modes: List of modes of the quadrature rule. Default is ['paper', 'optimized old', 'optimized']
    :return: Arrays containing the largest nodes, relative errors and computational times, all of
        shape (len(modes), len(Ns))
    """
    if Ns is None:
        Ns = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024])
    if modes is None:
        modes = ['paper', 'optimized old', 'optimized']

    kernel_errs = np.empty((len(modes), len(Ns)))
    largest_nodes = np.empty((len(modes), len(Ns)))
    duration = np.empty((len(modes), len(Ns)))
    ker_norm = rk.kernel_norm(H=H, T=T)

    for i in range(len(Ns)):
        for j in range(len(modes)):
            tic = time.perf_counter()
            nodes, weights = rk.quadrature_rule(H=H, N=Ns[i], T=T, mode=modes[j])
            duration[j, i] = time.perf_counter() - tic
            largest_nodes[j, i] = np.amax(nodes)
            kernel_errs[j, i] = np.amax(np.sqrt(rk.error(H=H, nodes=nodes, weights=weights, T=T, output='error'))
                                        / ker_norm)
            print(f'N={Ns[i]}, mode={modes[j]}, node={largest_nodes[j, i]:.3}, error={100*kernel_errs[j, i]:.4}%, time={duration[j, i]:.3}sec')
    return largest_nodes, kernel_errs, duration


def smile_errors(params=None, Ns=None, modes=None, true_smile=None, plot=False):
    """
    Prints the largest nodes, the relative errors and the computational times for the strong L^2-error approximation
    of the fractional kernel.
    :param params: Parameters of the rough Heston model
    :param Ns: Vector of values for N. Default is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    :param modes: List of modes of the quadrature rule. Default is ['paper', 'optimized', 'european old', 'european']
    :param true_smile: May specify the actual, non-approximated smile. If None, computes the smile and prints the
        computational time
    :param plot: If True, plots all sorts of plots
    :return: Arrays containing the true smile, approximate smiles, largest nodes, relative kernel errors,
        relative smile errors, computational time of the true smile, and computational times of the approximate smiles
    """
    params = rHeston_params(params)
    if Ns is None:
        Ns = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if modes is None:
        modes = ['paper', 'optimized', 'european old', 'european']

    kernel_errs = np.empty((len(modes), len(Ns)))
    largest_nodes = np.empty((len(modes), len(Ns)))
    duration = np.empty((len(modes), len(Ns)))
    smile_errs = np.empty((len(modes), len(Ns)))

    if true_smile is None:
        tic = time.perf_counter()
        true_smile = rHeston.iv_eur_call(S=params['S'], K=params['K'], H=params['H'], lambda_=params['lambda'],
                                         rho=params['rho'], nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                         T=params['T'], rel_tol=params['rel_tol'])
        true_duration = time.perf_counter() - tic
        print(f'The computation of the true smile took {true_duration:.3} seconds.')
    else:
        true_duration = 0.

    if isinstance(params['T'], np.ndarray):
        approx_smiles = np.empty((len(modes), len(Ns), len(params['T']), len(params['K'])))
    else:
        approx_smiles = np.empty((len(modes), len(Ns), len(params['K'])))

    ker_norm = rk.kernel_norm(H=params['H'], T=params['T'])

    for i in range(len(Ns)):
        for j in range(len(modes)):
            tic = time.perf_counter()
            nodes, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
            approx_smiles[j, i] = rHestonMarkov.iv_eur_call(S=params['S'], K=params['K'], H=params['H'],
                                                            lambda_=params['lambda'], rho=params['rho'],
                                                            nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                                            T=params['T'], N=Ns[i], mode=modes[j],
                                                            rel_tol=params['rel_tol'], nodes=nodes, weights=weights)
            duration[j, i] = time.perf_counter() - tic
            largest_nodes[j, i] = np.amax(nodes)
            kernel_errs[j, i] = np.amax(np.sqrt(rk.error(H=params['H'], nodes=nodes, weights=weights, T=params['T'],
                                                         output='error')) / ker_norm)
            smile_errs[j, i] = np.amax(np.abs(true_smile - approx_smiles[j, i]) / true_smile)
            print(f'N={Ns[i]}, mode={modes[j]}, node={largest_nodes[j, i]:.3}, kernel error={100 * kernel_errs[j, i]:.4}%, smile error={100 * smile_errs[j, i]:.4}%, time={duration[j, i]:.3}sec')

    k = np.log(params['K'] / params['S'])
    if plot:
        if isinstance(params['T'], np.ndarray):
            T, K = np.meshgrid(params['T'], params['K'])
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(T, K, true_smile, rstride=1, cstride=1, cmap='rainbow', edgecolor=None)
            ax.set_title('Implied volatility surface for European call option')
            ax.set_xlabel('Maturity')
            ax.set_ylabel('Log-moneyness')
            ax.set_zlabel('Implied volatility')
            plt.show()
            for j in range(len(modes)):
                for i in range(len(Ns)):
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')
                    rel_err = np.abs(true_smile - approx_smiles[j, i]) / true_smile
                    ax.plot_surface(T, K, rel_err, rstride=1, cstride=1, cmap='rainbow', edgecolor=None)
                    ax.set_title(f'Relative error of the implied volatility\nwith {label(modes[j])} quadrature and {Ns[i]} nodes')
                    ax.set_xlabel('Maturity')
                    ax.set_ylabel('Log-moneyness')
                    ax.set_zlabel('Relative error')
                    plt.show()
        else:
            for j in range(len(modes)):
                for i in range(len(Ns)):
                    plt.plot(k, approx_smiles[j, i], color=color(i, len(Ns)), label=f'N={Ns[i]}')
                plt.plot(k, true_smile, color='k', label=f'Exact smile')
                plt.legend(loc='best')
                plt.xlabel('Log-moneyness')
                plt.ylabel('Implied volatility')
                plt.title(f'Implied volatility for European call option\nwhere {label(modes[j])} quadrature is used')
                plt.show()
            for i in range(len(Ns)):
                for j in range(len(modes)):
                    plt.plot(k, approx_smiles[j, i], color=color(j, len(modes)), label=label(modes[j]))
                plt.plot(k, true_smile, color='k', label=f'Exact smile')
                plt.legend(loc='best')
                plt.xlabel('Log-moneyness')
                plt.ylabel('Implied volatility')
                plt.title(f'Implied volatility for European call option\nwith {Ns[i]} nodes')
                plt.show()
    return true_smile, approx_smiles, largest_nodes, kernel_errs, smile_errs, true_duration, duration


def plot_rHeston_sample_path(params, N=2, N_time=1024, mode='european', vol_behaviour='hyperplane reset',
                             plot_vol=True):
    """
    Plots a sample path of the stock price under the Markovian approximation of the rough Heston model. If one of
    N, N_steps or modes is a vector, plots all these sample paths in the same plot.
    :param params: Parameters of the rough Heston model
    :param N: Integer or vector of values for the number of dimensions of the Markovian approximation
    :param N_time: Integer or vector of values for the number of time steps
    :param mode: Mode or list of modes of the quadrature rule
    :param vol_behaviour: Behaviour of the volatility when it hits zero
    :param plot_vol: If True, plots the volatility in the same plot
    :return: The stock price processes, the volatility processes, the processes of the volatility components and the
        computational times
    """
    params = rHeston_params(params)
    input_is_array = True
    if isinstance(N, np.ndarray) and not isinstance(N_time, np.ndarray) and not isinstance(mode, list)\
            and not isinstance(vol_behaviour, list):
        N_time = np.array([N_time for _ in range(len(N))])
        mode = [mode for _ in range(len(N))]
        vol_behaviour = [vol_behaviour for _ in range(len(N))]
        labels = N
    elif isinstance(N_time, np.ndarray) and not isinstance(N, np.ndarray) and not isinstance(mode, list)\
            and not isinstance(vol_behaviour, list):
        N = np.array([N for _ in range(len(N_time))])
        mode = [mode for _ in range(len(N_time))]
        vol_behaviour = [vol_behaviour for _ in range(len(N_time))]
        labels = N_time
    elif isinstance(mode, list) and not isinstance(N_time, np.ndarray) and not isinstance(N, np.ndarray)\
            and not isinstance(vol_behaviour, list):
        N_time = np.array([N_time for _ in range(len(mode))])
        N = np.array([N for _ in range(len(mode))])
        vol_behaviour = [vol_behaviour for _ in range(len(mode))]
        labels = mode
    elif isinstance(vol_behaviour, list) and not isinstance(N, np.ndarray) and not isinstance(N_time, np.ndarray)\
            and not isinstance(mode, list):
        N = np.array([N for _ in range(len(vol_behaviour))])
        N_time = np.array([N_time for _ in range(len(vol_behaviour))])
        mode = np.array([mode for _ in range(len(vol_behaviour))])
        labels = vol_behaviour
    elif not isinstance(N, np.ndarray) and not isinstance(N_time, np.ndarray) and not isinstance(mode, list)\
            and not isinstance(vol_behaviour, list):
        N = np.array([N])
        N_time = np.array([N_time])
        mode = [mode]
        vol_behaviour = [vol_behaviour]
        input_is_array = False
        labels = ['stock price']
    else:
        raise ValueError('Bad input')

    S = []
    V = []
    V_components = []
    durations = np.empty(len(N))
    for i in range(len(N)):
        tic = time.perf_counter()
        S_, V_, V_comp = rHestonSP.sample_values(H=params['H'], lambda_=params['lambda'], rho=params['rho'],
                                                 nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                                 T=params['T'], S_0=params['S'], N=N[i], m=1, N_time=N_time[i],
                                                 mode=mode[i], vol_behaviour=vol_behaviour[i], sample_paths=True)
        durations[i] = time.perf_counter() - tic
        S.append(S_)
        V.append(V_)
        V_components.append(V_comp)

        plt.plot(np.linspace(0, params['T'], N_time[i]+1), S_, color=color(i, len(N)), label=str(labels[i]))
        if plot_vol:
            plt.plot(np.linspace(0, params['T'], N_time[i]+1), V_,
                     color=color(i, len(N)) if input_is_array else color(3, 3),
                     label=None if input_is_array else 'volatility')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.title('Stock price processes')
    plt.show()

    if not input_is_array:
        S = S[0]
        V = V[0]
        V_components = V_components[0]
        durations = durations[0]
    return S, V, V_components, durations


def compute_final_rHeston_stock_prices(params, Ns=None, N_times=None, modes=None, vol_behaviours=None,
                                       recompute=False, sample_paths=False, return_times=None):
    """
    Computes the final stock prices of 1000000 sample paths of the Markovian approximation of the rough Heston model.
    :param params: Parameters of the rough Heston model
    :param Ns: Vector of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Vector of values for the number of time steps. Default is 2**np.arange(18)
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param vol_behaviours: Behaviour of the volatility when it hits zero. Default is
        ['hyperplane reset', 'ninomiya victoir', 'sticky', 'hyperplane reflection', 'adaptive']
    :param recompute: If False, checks first whether the file in which the results will be saved already exist. If so,
        does not recompute the stock prices
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: May specify an array of times at which the sample path values should be returned. If None and
        sample_paths is True, this is equivalent to return_times = np.linspace(0, T, N_time+1)
    :return: None, the stock prices are saved in a file
    """
    params = rHeston_params(params)
    Ns = set_array_default(Ns, np.array([1, 2, 6]))
    N_times = set_array_default(N_times, np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                                                   32768, 65536, 131072]))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])

    for N in Ns:
        for N_time in N_times:
            for vol_behaviour in vol_behaviours:
                for mode in modes:
                    filename = get_filename(N=N, mode=mode, vol_behaviour=vol_behaviour, N_time=N_time,
                                            sample_paths=sample_paths)
                    print(f'Now simulating {filename}')
                    if not recompute and exists(filename):
                        pass
                    else:
                        n_rounds = int(np.ceil(N_time / 2048))
                        samples_per_round = int(np.ceil(100000 / n_rounds))
                        if not sample_paths:
                            final_S = np.empty(1000000)
                        else:
                            final_S = np.empty(1)
                        nodes, weights = rk.quadrature_rule(H=params['H'], N=N, T=params['T'], mode=mode)
                        for i in range(10):
                            WB_1 = load_WB(i) * np.sqrt(params['T'])
                            for j in range(n_rounds):
                                print(f'{i} of 10, {j} of {n_rounds}')
                                if j == n_rounds - 1:
                                    WB = WB_1[:, samples_per_round*j:, :]
                                else:
                                    WB = WB_1[:, samples_per_round*j:samples_per_round*(j+1), :]
                                WB = resize_WB(WB, N_time)
                                if not sample_paths:
                                    final_S[
                                        i * 100000 + j * samples_per_round:i * 100000 + (j + 1) * samples_per_round] = \
                                        rHestonSP.sample_values(H=params['H'], lambda_=params['lambda'],
                                                                rho=params['rho'], nu=params['nu'],
                                                                theta=params['theta'], V_0=params['V_0'], T=params['T'],
                                                                N=N, S_0=params['S'], N_time=N_time, WB=WB,
                                                                m=WB.shape[1], mode=mode, vol_behaviour=vol_behaviour,
                                                                nodes=nodes, weights=weights)
                                if sample_paths:
                                    S, V, V_comp = rHestonSP.sample_values(H=params['H'], lambda_=params['lambda'],
                                                                           rho=params['rho'], nu=params['nu'],
                                                                           theta=params['theta'], V_0=params['V_0'],
                                                                           T=params['T'], S_0=params['S'], N=N,
                                                                           m=WB.shape[1], N_time=N_time, WB=WB,
                                                                           mode=mode, vol_behaviour=vol_behaviour,
                                                                           nodes=nodes, weights=weights,
                                                                           sample_paths=sample_paths,
                                                                           return_times=return_times)
                                    np.save(filename, S)
                                    np.save(filename, V)
                                    np.save(filename, V_comp)
                        if not sample_paths:
                            np.save(filename, final_S)


def compute_smiles_given_stock_prices(params, Ns=None, N_times=None, modes=None, vol_behaviours=None, plot='N_time',
                                      true_smile=None):
    """
    Computes the implied volatility smiles given the final stock prices
    :param params: Parameters of the rough Heston model
    :param Ns: Vector of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Vector of values for the number of time steps. Default is 2**np.arange(18)
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param vol_behaviours: Behaviour of the volatility when it hits zero. Default is
        ['hyperplane reset', 'ninomiya victoir', 'sticky', 'hyperplane reflection', 'adaptive']
    :param plot: Creates plots depending on that parameter
    :param true_smile: The smile under the true rough Heston model can be specified here. Otherwise it is computed
    :return: The true smile, the Markovian smiles, the approximated smiles, the lower MC approximated smiles,
        the upper MC approximated smiles, the Markovian errors, the total errors of the approximated smiles, and
        the discretization errors of the approximated smiles
    """
    params = rHeston_params(params)
    if not isinstance(params['T'], np.ndarray):
        params['T'] = np.array([params['T']])
    Ns = set_array_default(Ns, np.array([1, 2, 6]))
    N_times = set_array_default(N_times, np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                                                   32768, 65536, 131072]))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])
    if true_smile is None:
        true_smile = rHeston.iv_eur_call(S=params['S'], K=params['K'], H=params['H'], lambda_=params['lambda'],
                                         rho=params['rho'], nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                         T=params['T'], rel_tol=params['rel_tol'])

    total_errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    discretization_errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    markov_errors = np.empty((len(Ns), len(modes)))
    markov_smiles = np.empty((len(Ns), len(modes), len(params['T']), len(params['K'])))
    approx_smiles = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times), len(params['T']),
                              len(params['K'])))
    lower_smiles = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times), len(params['T']),
                              len(params['K'])))
    upper_smiles = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times), len(params['T']),
                              len(params['K'])))

    for i in range(len(Ns)):
        for j in range(len(modes)):
            nodes, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
            markov_smiles[i, j, :, :] = rHestonMarkov.iv_eur_call(S=params['S'], K=params['K'], H=params['H'],
                                                      lambda_=params['lambda'], rho=params['rho'], nu=params['nu'],
                                                      theta=params['theta'],  V_0=params['V_0'], T=params['T'], N=Ns[i],
                                                      mode=modes[j], rel_tol=params['rel_tol'], nodes=nodes,
                                                      weights=weights)
            markov_errors[i, j] = np.amax(np.abs(markov_smiles[i, j, :, :] - true_smile)/true_smile)
            for k in range(len(vol_behaviours)):
                for m in range(len(N_times)):
                    final_S = np.load(f'rHeston samples {Ns[i]} dim {modes[j]} {vol_behaviours[k]} {N_times[m]} time steps.npy')
                    vol, low, upp = cf.iv_eur_call_MC(S=params['S'], K=params['K'], T=params['T'], samples=final_S)
                    approx_smiles[i, j, k, m, :, :] = vol
                    lower_smiles[i, j, k, m, :, :] = low
                    upper_smiles[i, j, k, m, :, :] = upp
                    total_errors[i, j, k, m] = np.amax(np.abs(vol - true_smile)/true_smile)
                    discretization_errors[i, j, k, m] = np.amax(np.abs(vol - markov_smiles[i, j, :, :])/markov_smiles[i, j, :, :])

    k_vec = np.log(params['K'])

    def finalize_plot():
        plt.plot(k_vec, true_smile[-1, :], color='k', label='Exact smile')
        plt.legend(loc='upper right')
        plt.xlabel('Log-moneyness')
        plt.ylabel('Implied volatility')
        plt.show()

    if plot == 'N' or (isinstance(plot, bool) and plot):
        for j in range(len(modes)):
            for k in range(len(vol_behaviours)):
                for m in range(len(N_times)):
                    for i in range(len(Ns)):
                        plt.plot(k_vec, approx_smiles[i, j, k, m, -1, :], color=color(i, len(Ns)), label=f'N={Ns[i]}')
                        if i == len(Ns) - 1:
                            plt.plot(k_vec, upper_smiles[i, j, k, m, -1, :], '--', color=color(i, len(Ns)))
                            plt.plot(k_vec, lower_smiles[i, j, k, m, -1, :], '--', color=color(i, len(Ns)))
                    plt.title(f'Rough Heston with {modes[j]} quadrature rule,\n{vol_behaviours[k]} and {N_times[m]} time steps')
                    finalize_plot()
    if plot == 'mode' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for k in range(len(vol_behaviours)):
                for m in range(len(N_times)):
                    for j in range(len(modes)):
                        plt.plot(k_vec, approx_smiles[i, j, k, m, -1, :], color=color(j, len(modes)), label=modes[j])
                        if j == len(modes) - 1:
                            plt.plot(k_vec, upper_smiles[i, j, k, m, -1, :], '--', color=color(j, len(modes)))
                            plt.plot(k_vec, lower_smiles[i, j, k, m, -1, :], '--', color=color(j, len(modes)))
                    plt.title(f'Rough Heston with {Ns[i]} nodes,\n{vol_behaviours[k]} and {N_times[m]} time steps')
                    finalize_plot()
    if plot == 'vol_behaviour' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(modes)):
                for m in range(len(N_times)):
                    for k in range(len(vol_behaviours)):
                        plt.plot(k_vec, approx_smiles[i, j, k, m, -1, :], color=color(k, len(vol_behaviours)),
                                 label=vol_behaviours[k])
                        if k == len(vol_behaviours) - 1:
                            plt.plot(k_vec, upper_smiles[i, j, k, m, -1, :], '--', color=color(k, len(vol_behaviours)))
                            plt.plot(k_vec, lower_smiles[i, j, k, m, -1, :], '--', color=color(k, len(vol_behaviours)))
                    plt.plot(k_vec, markov_smiles[i, j], color='brown', label='Exact Markovian smile')
                    plt.title(f'Rough Heston with {Ns[i]} nodes,\n{modes[j]} quadrature rule and {N_times[m]} time steps')
                    finalize_plot()
    if plot == 'N_time' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(modes)):
                for k in range(len(vol_behaviours)):
                    for m in range(len(N_times)):
                        plt.plot(k_vec, approx_smiles[i, j, k, m, -1, :], color=color(m, len(N_times)),
                                 label=f'{N_times[m]} time steps')
                        if m == len(N_times) - 1:
                            plt.plot(k_vec, upper_smiles[i, j, k, m, -1, :], '--', color=color(m, len(N_times)))
                            plt.plot(k_vec, lower_smiles[i, j, k, m, -1, :], '--', color=color(m, len(N_times)))
                    plt.plot(k_vec, markov_smiles[i, j], color='brown', label='Exact Markovian smile')
                    plt.title(f'Rough Heston with {Ns[i]} nodes,\n{modes[j]} quadrature rule and {vol_behaviours[k]}')
                    finalize_plot()

    return true_smile, markov_smiles, approx_smiles, lower_smiles, upper_smiles, markov_errors, total_errors, \
        discretization_errors


def compute_strong_discretization_errors(Ns=None, N_times=None, N_time_ref=131072, modes=None, vol_behaviours=None,
                                         plot='N_time'):
    """
    Computes the strong discretization error for the path simulation of the rough Heston model.
    :param Ns: Vector of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Vector of values for the number of time steps. Default is 2**np.arange(17)
    :param N_time_ref: Reference number of time steps. These solutions are taken as exact
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param vol_behaviours: Behaviour of the volatility when it hits zero. Default is
        ['hyperplane reset', 'ninomiya victoir', 'sticky', 'hyperplane reflection', 'adaptive']
    :param plot: Creates plots depending on that parameter
    :return: The errors, the lower MC errors, the upper MC errors, and the approximate convergence rates
    """
    Ns = set_array_default(Ns, np.array([1, 2, 6]))
    N_times = set_array_default(N_times, np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                                                   32768, 65536]))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])

    errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    stats = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))

    for i in range(len(Ns)):
        for j in range(len(modes)):
            for k in range(len(vol_behaviours)):
                filename = get_filename(N=Ns[i], mode=modes[j], vol_behaviour=vol_behaviours[k], N_time=N_time_ref,
                                        sample_paths=False)
                S = np.load(filename)
                for m in range(len(N_times)):
                    filename = get_filename(N=Ns[i], mode=modes[j], vol_behaviour=vol_behaviours[k], N_time=N_times[m],
                                            sample_paths=False)
                    S_approx = np.load(filename)
                    errs = np.abs(S - S_approx)**2
                    errors[i, j, k, m], stats[i, j, k, m] = cf.MC(errs)
    lower = np.sqrt(np.fmax(errors - stats, 0))
    upper = np.sqrt(errors + stats)
    errors = np.sqrt(errors)

    conv_rates = np.empty((len(Ns), len(modes), len(vol_behaviours)))
    if plot == 'N' or (isinstance(plot, bool) and plot):
        for j in range(len(modes)):
            for k in range(len(vol_behaviours)):
                subtext = ''
                for i in range(len(Ns)):
                    col = color(i, len(Ns))
                    conv_rates[i, j, k], _ = plot_log_linear_regression(x=N_times, y=errors[i, j, k, :], col=col,
                                                                        lab=f'N={Ns[i]}', kind='o-', offset=4)
                    plt.loglog(N_times, lower[i, j, k, :], '--', col=col)
                    plt.loglog(N_times, upper[i, j, k, :], '--', col=col)
                    if i != 0:
                        subtext += ', '
                    subtext += f'{Ns[i]}: {conv_rates[i, j, k]}'
                plt.title(f'Rough Heston strong discretization errors\nwith {modes[j]} quadrature rule and {vol_behaviours[k]}')
                plt.legend(loc='upper right')
                plt.xlabel('Number of time steps')
                plt.ylabel(r'$L^2$' + '-error')
                plt.annotate(subtext, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                plt.show()
    if plot == 'mode' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for k in range(len(vol_behaviours)):
                subtext = ''
                for j in range(len(modes)):
                    col = color(j, len(modes))
                    conv_rates[i, j, k], _ = plot_log_linear_regression(x=N_times, y=errors[i, j, k, :], col=col,
                                                                        lab=modes[j], kind='o-', offset=4)
                    plt.loglog(N_times, lower[i, j, k, :], '--', col=col)
                    plt.loglog(N_times, upper[i, j, k, :], '--', col=col)
                    if i != 0:
                        subtext += ', '
                    subtext += f'{modes[j]}: {conv_rates[i, j, k]}'
                plt.title(f'Rough Heston strong discretization errors\nwith {Ns[i]} nodes and {vol_behaviours[k]}')
                plt.legend(loc='upper right')
                plt.xlabel('Number of time steps')
                plt.ylabel(r'$L^2$' + '-error')
                plt.annotate(subtext, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                plt.show()
    if plot == 'vol_behaviour' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(modes)):
                subtext = ''
                for k in range(len(vol_behaviours)):
                    col = color(k, len(vol_behaviours))
                    conv_rates[i, j, k], _ = plot_log_linear_regression(x=N_times, y=errors[i, j, k, :], col=col,
                                                                        lab=vol_behaviours[k], kind='o-', offset=4)
                    plt.loglog(N_times, lower[i, j, k, :], '--', col=col)
                    plt.loglog(N_times, upper[i, j, k, :], '--', col=col)
                    if i != 0:
                        subtext += ', '
                    subtext += f'{vol_behaviours[k]}: {conv_rates[i, j, k]}'
                plt.title(f'Rough Heston strong discretization errors\nwith {Ns[i]} nodes and {modes[j]} quadrature rule')
                plt.legend(loc='upper right')
                plt.xlabel('Number of time steps')
                plt.ylabel(r'$L^2$' + '-error')
                plt.annotate(subtext, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                plt.show()

    return errors, lower, upper, conv_rates
