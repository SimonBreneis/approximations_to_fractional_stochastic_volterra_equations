import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats, optimize
import ComputationalFinance as cf
import rHeston
import rHestonMarkov
import RoughKernel as rk
import rHestonMarkovSamplePaths as rHestonSP
from os.path import exists
import multiprocessing as mp
import itertools


def log_linspace(a, b, n):
    return np.exp(np.linspace(np.log(a), np.log(b), n))


def rHeston_params(params):
    """
    Ensures that the parameters of the rHeston model are all specified.
    :param params: Dictionary of already specified parameters of the rough Heston model. If 'simple' or None, is set to
        simple_params. If 'difficult', is set to difficult_params. Else, every key of the dictionary that is not
        specified is given the corresponding value in simple_params
    :return: The modified and completed params dictionary
    """
    simple_params = {'S': 1., 'K': np.exp(np.linspace(-1.5, 0.75, 451)), 'H': 0.1, 'T': 1., 'lambda': 0.3, 'rho': -0.7,
                     'nu': 0.3, 'theta': 0.02, 'V_0': 0.02, 'rel_tol': 1e-05}
    difficult_params = {'S': 1., 'K': np.exp(np.linspace(-0.5, 0.1, 181)), 'H': 0.07, 'T': 0.04, 'lambda': 0.6,
                        'rho': -0.8, 'nu': 0.5, 'theta': 0.01, 'V_0': 0.01, 'rel_tol': 1e-05}
    if params is None or params == 'simple':
        return simple_params
    if params == 'difficult':
        return difficult_params
    for key in simple_params:
        params[key] = params.get(key, simple_params[key])
    return params


def rHeston_params_grid(params):
    simple_params = {'S': 1., 'K': np.exp(np.linspace(-1, 0.5, 301)), 'H': log_linspace(0.05, 0.1, 2),
                     'T': np.linspace(0.04, 1., 25), 'lambda': log_linspace(0.2, 1., 2),
                     'rho': np.linspace(-0.6, -0.8, 2), 'nu': log_linspace(0.2, 0.6, 2),
                     'theta': log_linspace(0.01, 0.03, 2), 'V_0': log_linspace(0.01, 0.03, 2), 'rel_tol': 1e-05}
    difficult_params = {'S': 1., 'K': np.exp(np.linspace(-1.5, 0.75, 451)), 'H': log_linspace(0.01, 0.2, 15),
                        'T': np.linspace(0.003, 10., 50), 'lambda': log_linspace(0.05, 5., 15),
                        'rho': np.linspace(-0.3, -0.95, 15), 'nu': log_linspace(0.05, 5., 15),
                        'theta': log_linspace(0.005, 0.3, 15), 'V_0': log_linspace(0.005, 0.3, 15), 'rel_tol': 1e-05}
    if params is None or params == 'simple':
        return simple_params
    if params == 'difficult':
        return difficult_params
    for key in simple_params:
        params[key] = set_array_default(params.get(key, simple_params[key]), simple_params[key])
    return params


def color(i, N):
    """
    Returns a color for plotting.
    :param i: Index of the line. This is the color of the ith line that is being plotted
    :param N: Total number of lines to plot. Depending on how many colors are needed, returns a different color to
        ensure that the colors are well distinguishable.
    :return: Some color for plotting (a string)
    """
    c = ['r', 'C1', 'y', 'g', 'b', 'purple']
    c_ = ['darkred', 'r', 'C1', 'y', 'lime', 'g', 'deepskyblue', 'b', 'purple', 'deeppink']
    c_short = ['r', 'g', 'b']
    c_intermediate = ['r', 'C1', 'g', 'b']
    if N <= 3:
        return c_short[i % 3]
    if N <= 4:
        return c_intermediate[i % 4]
    if N <= 6 or N == 11 or N == 12:
        return c[i % 6]
    return c_[i % 10]


def label(mode):
    """
    Returns the label for plotting.
    :param mode: A string denoting how the quadrature rule was chosen.
    :return: A string that can be used as a label for plotting
    """
    if mode == 'observation':
        return 'Paper'
    return mode[0].upper() + mode[1:]


def load_WB(i):
    """
    Loads the ith batch (of 10) of the precomputed Brownian motions.
    :param i: The index of the batch
    :return: An array of shape (2, 100000, 2048). These are 2 * 100000 * 2048 iid normally distributed random variables
        with mean 0 and standard deviation 1/sqrt(2048)
    """
    dW = np.load(f'dW{i}.npy')
    dB = np.load(f'dB{i}.npy')
    WB = np.empty((2, dW.shape[0], dW.shape[1]))
    WB[0, :, :] = dW
    WB[1, :, :] = dB
    return WB


def coarsen_WB(WB):
    """
    Given the increments of a Brownian motion of shape (..., N_time), returns the same increments on a coarser time
    grid.
    :param WB: The Brownian motion with N_time increments (time steps)
    :return: The same Brownian motion with ceil(N_time/2) increments (time steps)
    """
    return WB[..., ::2] + WB[..., 1::2]


def refine_WB(WB):
    """
    Given the increments of a Brownian motion of shape (..., N_time), returns the same increments on a finer time grid.
    :param WB: The Brownian motion with N_time increments (time steps)
    :return: The same Brownian motion with 2*N_time increments (time steps)
    """
    WB_ = np.empty((WB.shape[:-1],) + (2*WB.shape[2],))
    increments = np.random.normal(0, 1 / np.sqrt(4*WB.shape[2]), WB.shape)
    WB_[..., ::2] = WB / 2 + increments
    WB_[..., 1::2] = WB / 2 - increments
    return WB_


def resize_WB(WB, N_time):
    """
    Given the increments of a Brownian motion of shape (..., c), where c is the number of time steps, returns the
    same increments on a time grid with roughly (!) N_time time steps.
    :param WB: The Brownian motion with c increments (time steps)
    :param N_time: Number of desired time steps
    :return: The same Brownian motion with roughly (!) N_time increments (time steps)
    """
    while WB.shape[2] > N_time:
        WB = coarsen_WB(WB)
    while WB.shape[2] < N_time:
        WB = refine_WB(WB)
    return WB


def set_array_default(x, default):
    """
    If x is None, returns default. If x is not a numpy array, returns np.array([x]). Else, returns x.
    :param x: The object which should be an array
    :param default: What the default value of x should be if it is None
    :return: Either x, or default, or np.array([x])
    """
    if x is None:
        x = default
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    return x


def set_list_default(x, default):
    """
    If x is None, returns default. If x is not a list, returns [x]. Else, returns x.
    :param x: The object which should be a list
    :param default: What the default value of x should be if it is None
    :return: Either x, or default, or [x]
    """
    if x is None:
        x = default
    if not isinstance(x, list):
        x = [x]
    return x


def flat_mesh(*xi):
    """
    Given a sequence of iterable objects x0, ..., xn, creates the corresponding n-dimensional mesh, and then flattens
    all the components.
    :param xi: The iterable objects the mesh consists of
    :return: The flattened mesh
    """
    mesh = np.meshgrid(*xi, indexing='ij')
    return [x.flatten() for x in mesh]


def argmax_indices(x, axis=0):
    """
    If one applies np.argmax(x, axis=i), one gets a single (n-1)-dimensional array denoting the indices of the i-th
    dimension where the maximum is attained. Conversely, this function returns an n-tuple of (n-1)-dimensional arrays
    such that np.amax(x, axis=i) == x[argmax_indices(x, axis=i)] holds true.
    :param x: A numpy array
    :param axis: An integer, the axis over which the maximum should be taken
    :return: An n-tuple of (n-1)-dimensional arrays such that np.amax(x, axis=i) == x[argmax_indices(x, axis=i)] holds
        true.
    """
    ind = np.argmax(x, axis=axis)
    indexing = ()
    for k in range(len(x.shape)):
        if k == axis:
            k_th_index = (ind,)
        else:
            temp_indices = ()
            for j in range(len(x.shape)):
                if j != axis:
                    if k == j:
                        temp_indices = temp_indices + (slice(None),)
                    else:
                        temp_indices = temp_indices + (None,)
            k_th_index = np.arange(x.shape[k])
            k_th_index = k_th_index[temp_indices]
        indexing = indexing + (k_th_index,)
    return indexing


def get_filename(kind, N=None, mode=None, vol_behaviour=None, N_time=None, params=None, truth=False, markov=False):
    """
    Filename under which we save our simulated data.
    :param N: Number of nodes
    :param mode: The kind of quadrature rule we use
    :param vol_behaviour: The behaviour of the volatility when it hits zero
    :param N_time: The number of time steps
    :param kind: The kind of object we are saving (i.e. 'samples' or 'sample paths' or 'nodes', etc.)
    :param params: If None, assumes params is simple_params and does not include it in the file name
    :param truth: True if the smile is a true smile, False if not, i.e. if the smile is approximated
    :param markov: True if the smile is an exact Markovian smile. False if not, i.e. if it is a true smile or an
        approximated Markovian smile
    :return: A string representing the file name
    """
    if params is None:
        if truth:
            return f'rHeston {kind}.npy'
        else:
            if markov:
                return f'rHeston {kind} {N} dim {mode}.npy'
            else:
                return f'rHeston {kind} {N} dim {mode} {vol_behaviour} {N_time} time steps.npy'

    H = params['H']
    lambda_ = params['lambda']
    rho = params['rho']
    nu = params['nu']
    theta = params['theta']
    V_0 = params['V_0']
    T = params['T']
    K = (np.log(np.amin(params['K'])), np.log(np.amax(params['K'])), len(params['K']))
    K_string = f'K=({K[0]:.4}, {K[1]:.4}, {K[2]})'
    if isinstance(T, np.ndarray):
        T_string = f'T=({np.amin(T):.4}, {np.amax(T):.4}, {len(T)})'
    else:
        T_string = f'T={T:.4}'
    if truth:
        return f'rHeston {kind}, H={H:.3}, lambda={lambda_:.3}, rho={rho:.3}, nu={nu:.3}, theta={theta:.3}, ' \
               f'V_0={V_0:.3}, {T_string}, {K_string}.npy'
    else:
        if markov:
            return f'rHeston {kind} {N} dim {mode}, H={H:.3}, lambda={lambda_:.3}, rho={rho:.3}, nu={nu:.3}, ' \
                   f'theta={theta:.3}, V_0={V_0:.3}, {T_string}, {K_string}.npy'
        else:
            return f'rHeston {kind} {N} dim {mode} {vol_behaviour} {N_time} time steps, H={H:.3}, ' \
                   f'lambda={lambda_:.3}, rho={rho:.3}, nu={nu:.3}, theta={theta:.3}, V_0={V_0:.3}, {T_string}, ' \
                   f'{K_string}.npy'


def log_linear_regression(x, y):
    """
    Applies log-linear regression of y against x.
    :param x: The argument
    :param y: The function value
    :return: Exponent, constant, R-value, p-value, empirical standard deviation
    """
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(x), np.log(y))
    return slope, np.exp(intercept), r_value, p_value, std_err


def plot_log_linear_regression(x, y, col, lab, line_style='--', offset=0):
    """
    Plots x and y, as well as the corresponding log-linear regression as a log-log-plot.
    :param x: Independent variable
    :param y: Dependent variable
    :param col: The color with which the lines should be plotted
    :param lab: The label for the plot of x and y
    :param line_style: The line style of the regression
    :param offset: The regression is only done on elements with index offset or higher
    :return: The power of the log-linear regression, and the multiplicative constant
    """
    power, constant, r_value, p_value, std_err = log_linear_regression(x[offset:], y[offset:])
    plt.loglog(x, y, color=col, label=lab)
    x_alt = np.exp(np.linspace(np.log(np.amin(x)), np.log(np.amax(x)), 5))  # could use just x, but this is to avoid
    # excessively many dots or crosses, etc., if line_style is not '--'
    plt.loglog(x_alt, constant * x_alt ** power, line_style, color=col)
    return power, constant


def rHeston_samples(params, N=-1, N_time=-1, WB=None, m=1, mode=None, vol_behaviour=None, nodes=None, weights=None,
                    sample_paths=False, return_times=None):
    """
    Shorthand for rHestonSP.sample_values.
    """
    params = rHeston_params(params)
    return rHestonSP.sample_values(H=params['H'], lambda_=params['lambda'], rho=params['rho'], nu=params['nu'],
                                   theta=params['theta'], V_0=params['V_0'], T=params['T'], N=N, S_0=params['S'],
                                   N_time=N_time, WB=WB, m=m, mode=mode, vol_behaviour=vol_behaviour, nodes=nodes,
                                   weights=weights, sample_paths=sample_paths, return_times=return_times)


def rHeston_iv_eur_call(params, load=True, save=False, verbose=0):
    """
    Shorthand for rHeston.iv_eur_call.
    :param params: Parameters of the rough Heston model
    :param load: If True, loads the smile instead of computing it, if it is saved
    :param save: If True, saves the smile after computing it
    :param verbose: Determines how many intermediate results are printed to the console
    :return: The smile
    """
    filename = get_filename(kind='true', params=params, truth=True, markov=False)
    if load:
        if exists(filename):
            return np.load(filename)
        if not isinstance(params['T'], np.ndarray):
            T_vec = np.linspace(0.04, 1., 25)
            if np.amin(np.abs(params['T'] - T_vec)) < 1e-06:
                min_ind = np.argmin(np.abs(params['T'] - T_vec))
                alt_params = params.copy()
                alt_params['T'] = T_vec
                alt_filename = get_filename(kind='true', params=alt_params, truth=True, markov=False)
                if exists(alt_filename):
                    true_smile = np.load(alt_filename)
                    true_smile = true_smile[min_ind, :]
                    if save:
                        np.save(filename, true_smile)
                    return true_smile
    print('Actually compute it')
    try:
        result = rHeston.iv_eur_call(S_0=params['S'], K=params['K'], H=params['H'], lambda_=params['lambda'],
                                     rho=params['rho'], nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                     T=params['T'], rel_tol=params['rel_tol'], verbose=verbose)
    except RuntimeError:
        print('Did not converge in given time')
        if isinstance(params['T'], np.ndarray):
            return np.empty((len(params['T']), len(params['K'])))
        else:
            return np.empty(len(params['K']))
    if save:
        np.save(filename, result)
    return result


def rHeston_iv_eur_call_parallelized(params, num_threads=5, verbose=0):
    """
    Parallelized version of rHeston_iv_eur_call.
    :param params: Parameters of the rough Heston model
    :param num_threads: Number of parallel processes
    :param verbose: Determines how many intermediate results are printed to the console
    :return: The smiles
    """
    params = rHeston_params_grid(params)
    H, lambda_, rho, nu, theta, V_0 = flat_mesh(params['H'], params['lambda'], params['rho'], params['nu'],
                                                params['theta'], params['V_0'])
    dictionaries = [{'S': params['S'], 'K': params['K'], 'H': H[i], 'T': params['T'], 'lambda': lambda_[i],
                     'rho': rho[i], 'nu': nu[i], 'theta': theta[i], 'V_0': V_0[i], 'rel_tol': params['rel_tol']}
                    for i in range(len(H))]
    with mp.Pool(processes=num_threads) as pool:
        result = pool.starmap(rHeston_iv_eur_call, zip(dictionaries, itertools.repeat(True), itertools.repeat(True),
                                                       itertools.repeat(verbose)))
    return result


def rHestonMarkov_iv_eur_call(params, N=-1, mode=None, nodes=None, weights=None, load=True, save=False, verbose=0):
    """
    Shorthand for rHestonMarkov.iv_eur_call.
    :param params: Parameters of the rough Heston model
    :param N: Number of nodes
    :param mode: The mode of the quadrature rule
    :param nodes: The nodes of the quadrature rule
    :param weights: The weights of the quadrature rule
    :param load: If True, loads the smile instead of computing it, if it is saved
    :param save: If True, saves the smile after computing it
    :param verbose: Determines how many intermediate results are printed to the console
    :return: The smile
    """
    filename = get_filename(kind='Markov', params=params, truth=False, markov=True, N=N, mode=mode)
    print(filename)
    if load and exists(filename):
        return np.load(filename)
    try:
        result = rHestonMarkov.iv_eur_call(S_0=params['S'], K=params['K'], H=params['H'], lambda_=params['lambda'],
                                           rho=params['rho'], nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                           T=params['T'], rel_tol=params['rel_tol'], N=N, mode=mode, nodes=nodes,
                                           weights=weights, verbose=verbose)
    except RuntimeError:
        print('Did not converge in given time')
        if isinstance(params['T'], np.ndarray):
            return np.empty((len(params['T']), len(params['K'])))
        else:
            return np.empty(len(params['K']))
    if save:
        np.save(filename, result)
    return result


def rHestonMarkov_iv_eur_call_parallelized(params, Ns, modes, num_threads=15, verbose=0):
    """
    Parallelized version of rHestonMarkov_iv_eur_call.
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of N values
    :param modes: List of modes of the quadrature rule
    :param num_threads: Number of parallel processes
    :return: The smiles
    """
    params = rHeston_params_grid(params)
    H, lambda_, rho, nu, theta, V_0, N, mode = flat_mesh(params['H'], params['lambda'], params['rho'], params['nu'],
                                                         params['theta'], params['V_0'], Ns, modes)
    dictionaries = [{'S': params['S'], 'K': params['K'], 'H': H[i], 'T': params['T'], 'lambda': lambda_[i],
                     'rho': rho[i], 'nu': nu[i], 'theta': theta[i], 'V_0': V_0[i], 'rel_tol': params['rel_tol']}
                    for i in range(len(H))]
    with mp.Pool(processes=num_threads) as pool:
        result = pool.starmap(rHestonMarkov_iv_eur_call, zip(dictionaries, N, mode, itertools.repeat(None),
                                                             itertools.repeat(None), itertools.repeat(True),
                                                             itertools.repeat(True), itertools.repeat(verbose)))
    return result


def max_errors_MC(truth, estimate, lower, upper):
    """
    Computes the maximal (l^infinity) relative error, as well as the maximal lower and upper MC bounds.
    :param truth: The exact values
    :param estimate: The approximated values (involving among other things MC simulation)
    :param lower: The lower MC confidence bound for the approximated values
    :param upper: The upper MC confidence bound for the approximated values
    :return: The maximal relative error of estimate, as well as corresponding MC lower and upper confidence bounds
    """
    t, e, l, u = truth.flatten(), estimate.flatten(), lower.flatten(), upper.flatten()
    e_err, l_err, u_err = np.abs(e-t)/t, np.abs(l-t)/t, np.abs(u-t)/t
    positive_err_ind = (t-l) * (u-t) < 0
    l_err_vec = np.zeros(len(t))
    l_err_vec[positive_err_ind] = np.fmin(l_err, u_err)[positive_err_ind]
    u_err_vec = np.fmax(l_err, u_err)
    return np.amax(e_err), np.amax(l_err_vec), np.amax(u_err_vec)


def geometric_average(x):
    """
    Computes the geometric average of the samples in x.
    :param x: Numpy array of samples
    :return: The geometric average of x
    """
    return np.exp(np.average(np.log(x)))


def arr_argmin(x):
    """
    Returns the argmin of x given as an index that can be used on x.
    :param x: Array
    :return: Index with the argmin.
    """
    return np.unravel_index(np.argmin(x), x.shape)


def arr_argmax(x):
    """
    Returns the argmax of x given as an index that can be used on x.
    :param x: Array
    :return: Index with the argmax.
    """
    return np.unravel_index(np.argmax(x), x.shape)


def kernel_errors(H, T, Ns=None, modes=None, verbose=0):
    """
    Prints the largest nodes, the relative errors and the computational times for the strong L^2-error approximation
    of the fractional kernel.
    :param H: Hurst parameter
    :param T: Final time, may also be a numpy array
    :param Ns: Numpy array of values for N. Default is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024]
    :param modes: List of modes of the quadrature rule. Default is ['paper', 'optimized old', 'optimized']
    :param verbose: Determines how many intermediate results are printed to the console
    :return: Arrays containing the largest nodes, relative errors and computational times, all of
        shape (len(modes), len(Ns))
    """
    Ns = set_array_default(Ns, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024]))
    modes = set_list_default(modes, ['paper', 'optimized old', 'optimized'])

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
            kernel_errs[j, i] = np.amax(np.sqrt(np.fmax(
                rk.error(H=H, nodes=nodes, weights=weights, T=T, output='error'), 0)) / ker_norm)
            if verbose >= 1:
                print(f'N={Ns[i]}, mode={modes[j]}, node={largest_nodes[j, i]:.3}, error={100*kernel_errs[j, i]:.4}%, '
                      + f'time={duration[j, i]:.3}sec')
    return largest_nodes, kernel_errs, duration


def kernel_errors_parallelized_testing(H=None, T=None, N=None, mode=None, num_threads=5):
    """
    Parallelization of the function kernel_errors for testing purposes.
    :param H: Numpy array of Hurst parameters. Default is np.exp(np.linspace(np.log(0.01), np.log(0.49), 300))
    :param T: Numpy array of final times. Default is np.exp(np.linspace(np.log(0.0001), np.log(10), 300))
    :param N: Numpy array of values for N. Default is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    :param mode: List of modes of the quadrature rule. Default is ['paper', 'optimized old', 'optimized']
    :param num_threads: Number of parallel threads
    :return: Numpy arrays containing the largest nodes, relative errors and computational times, of
        shape (len(H), len(T), 3, len(mode), len(N)). The three represents the three kinds of information (nodes,
        errors, times) present
    """
    H = set_array_default(H, log_linspace(0.01, 0.2, 300))
    T = set_array_default(T, log_linspace(0.001, 2., 300))
    H_grid, T_grid = flat_mesh(H, T)
    N = set_array_default(N, np.arange(1, 11))
    mode = set_list_default(mode, ['paper', 'optimized'])

    with mp.Pool(processes=num_threads) as pool:
        result = pool.starmap(kernel_errors, zip(H_grid, T_grid, itertools.repeat(N), itertools.repeat(mode)))
    result = np.asarray(result)
    result = result.reshape((len(H), len(T)) + result.shape[1:])
    largest_nodes = result[:, :, 0, :, :]
    errors = result[:, :, 1, :, :]
    largest_nodes_paper = largest_nodes[:, :, 0, :]
    largest_nodes_optimized = largest_nodes[:, :, 1, :]
    errors_paper = errors[:, :, 0, :]
    errors_optimized = errors[:, :, 1, :]
    for i in range(len(N)):
        print(f'Results for N={N[i]}')
        print(f'The highest error in the kernel using paper quadrature is {100 * np.amax(errors_paper[..., i]):.4}%, '
              f'which is attained at H={H[arr_argmax(errors_paper[..., i])[0]]:.3} and '
              f'T={T[arr_argmax(errors_paper[..., i])[1]]:.3}.')
        print(f'The lowest error in the kernel using paper quadrature is {100 * np.amin(errors_paper[..., i]):.4}%, '
              f'which is attained at H={H[arr_argmin(errors_paper[..., i])[0]]:.3} and '
              f'T={T[arr_argmin(errors_paper[..., i])[1]]:.3}.')
        print(f'The geometric average error in the kernel using paper quadrature is '
              f'{100 * geometric_average(errors_paper[..., i]):.4}%.')
        print(f'The median error in the kernel using paper quadrature is {100 * np.median(errors_paper[..., i]):.4}%.')
        print(f'The highest error in the kernel using optimized quadrature is '
              f'{100 * np.amax(errors_optimized[..., i]):.4}%, which is attained at '
              f'H={H[arr_argmax(errors_optimized[..., i])[0]]:.3} and '
              f'T={T[arr_argmax(errors_optimized[..., i])[1]]:.3}.')
        print(f'The lowest error in the kernel using optimized quadrature is '
              f'{100 * np.amin(errors_optimized[..., i]):.4}%, which is attained at '
              f'H={H[arr_argmin(errors_optimized[..., i])[0]]:.3} and '
              f'T={T[arr_argmin(errors_optimized[..., i])[1]]:.3}.')
        print(f'The geometric average error in the kernel using optimized quadrature is '
              + f'{100 * geometric_average(errors_optimized[..., i]):.4}%.')
        print(f'The median error in the kernel using optimized quadrature is '
              + f'{100 * np.median(errors_optimized[..., i]):.4}%.')
        keo_flat = errors_optimized[..., i].flatten()
        kep_flat = errors_paper[..., i].flatten()
        faulty_ind = np.logical_or(kep_flat < 1e-6, keo_flat < 1e-6)
        keo_flat, kep_flat = keo_flat[~faulty_ind], kep_flat[~faulty_ind]
        error_improvements = keo_flat / kep_flat
        print(f'The smallest improvement of the error in the kernel using optimized quadrature instead of paper '
              f'quadrature is attaining an error equal to {100 * np.amax(error_improvements):.4}% of the original '
              f'error.')
        print(f'The biggest improvement of the error in the kernel using optimized quadrature instead of paper '
              f'quadrature is attaining an error equal to {100 * np.amin(error_improvements):.4}% of the original '
              f'error.')
        print(f'The geometric average improvement of the error in the kernel using optimized quadrature instead of '
              f'paper quadrature is attaining an error equal to {100 * geometric_average(error_improvements):.4}% of '
              f'the original error.')
        print(f'The median improvement of the error in the kernel using optimized quadrature instead of '
              f'paper quadrature is attaining an error equal to {100 * np.median(error_improvements):.4}% of the '
              f'original error.')
        if N[i] >= 2:
            node_improvements = largest_nodes_optimized[..., i] / largest_nodes_paper[..., i]
            print(f'The largest increase of the largest node when using optimized quadrature instead of paper '
                  f'quadrature is a node equal to {np.amax(node_improvements):.4} times of the original node, which '
                  f'is attained at H={H[arr_argmax(node_improvements)[0]]:.3} and '
                  f'T={T[arr_argmax(node_improvements)[1]]:.3}.')
            print(f'The largest decrease of the largest node when using optimized quadrature instead of paper '
                  f'quadrature is a node equal to {np.amin(node_improvements):.4} times of the original node, which '
                  f'is attained at H={H[arr_argmin(node_improvements)[0]]:.3} and '
                  f'T={T[arr_argmin(node_improvements)[1]]:.3}.')
            print(f'The geometric average increase of the largest node when using optimized quadrature instead of '
                  f'paper quadrature is a node equal to {geometric_average(node_improvements):.4} times of the'
                  f'original node.')
            print(f'The median increase of the largest node when using optimized quadrature instead of paper '
                  f'quadrature is a node equal to {np.median(node_improvements):.4} times of the original node.')
        print('-----------------------------------------------------------------------')
    return result


def smile_errors(params=None, Ns=None, modes=None, true_smile=None, plot=False, load=True, save=False, verbose=0,
                 parallelizable_output=False):
    """
    Prints the largest nodes, the relative errors and the computational times for the strong L^2-error approximation
    of the fractional kernel.
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of values for N. Default is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    :param modes: List of modes of the quadrature rule. Default is ['paper', 'optimized', 'european old', 'european']
    :param true_smile: May specify the actual, non-approximated smile. If None, computes the smile and prints the
        computational time
    :param plot: If True, plots all sorts of plots
    :param load: If True, loads the smile instead of computing it, if it is saved
    :param save: If True, saves the smile after computing it
    :param verbose: Determines how many intermediate results are printed to the console
    :param parallelizable_output: If True, returns the largest nodes, kernel errors and smile errors (all of the same
        shape). If False, returns the true smiles, Markovian smiles, largest nodes, kernel errors and smile errors
    :return: Arrays containing the true smile, approximate smiles, largest nodes, relative kernel errors, and
        relative smile errors
    """
    params = rHeston_params(params)
    Ns = set_array_default(Ns, np.arange(1, 11))
    modes = set_list_default(modes, ['paper', 'optimized', 'european old', 'european'])

    kernel_errs = np.empty((len(modes), len(Ns)))
    largest_nodes = np.empty((len(modes), len(Ns)))
    smile_errs = np.empty((len(modes), len(Ns)))

    if true_smile is None:
        true_smile = rHeston_iv_eur_call(params=params, load=load, save=save, verbose=verbose)

    if isinstance(params['T'], np.ndarray):
        approx_smiles = np.empty((len(modes), len(Ns), len(params['T']), len(params['K'])))
    else:
        approx_smiles = np.empty((len(modes), len(Ns), len(params['K'])))

    ker_norm = rk.kernel_norm(H=params['H'], T=params['T'])

    for i in range(len(Ns)):
        for j in range(len(modes)):
            nodes, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
            approx_smiles[j, i] = rHestonMarkov_iv_eur_call(params=params, N=Ns[i], mode=modes[j], nodes=nodes,
                                                            weights=weights, verbose=verbose, load=load, save=save)
            largest_nodes[j, i] = np.amax(nodes)
            kernel_errs[j, i] = np.amax(np.sqrt(rk.error(H=params['H'], nodes=nodes, weights=weights, T=params['T'],
                                                         output='error')) / ker_norm)
            smile_errs[j, i] = np.amax(np.abs(true_smile - approx_smiles[j, i]) / true_smile)
            print(f'N={Ns[i]}, mode={modes[j]}, node={largest_nodes[j, i]:.3}, kernel error='
                  + f'{100 * kernel_errs[j, i]:.4}%, smile error={100 * smile_errs[j, i]:.4}%')

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
                    ax.set_title(f'Relative error of the implied volatility\nwith {label(modes[j])} quadrature and '
                                 + f'{Ns[i]} nodes')
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

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        for j in range(len(modes)):
            ax1.loglog(Ns, smile_errs[j, :], color=color(j, len(modes)), label=modes[j])
            ax1.loglog(Ns, kernel_errs[j, :], '--', color=color(j, len(modes)))
            ax2.loglog(Ns, largest_nodes[j, :], ':', color=color(j, len(modes)))
        ax1.loglog(Ns, 2 * params['rel_tol'] * np.ones(len(Ns)), '--', color='k', label='discretization error')
        ax1.set_xlabel('Number of nodes')
        ax1.set_ylabel('Relative error')
        ax2.set_ylabel('Largest node')
        ax1.legend(loc='best')
        plt.title(f'Relative errors and largest nodes of the implied volatility')
        plt.show()

    if not parallelizable_output:
        return true_smile, approx_smiles, largest_nodes, kernel_errs, smile_errs
    else:
        return largest_nodes, kernel_errs, smile_errs


def smile_errors_parallelized_testing(params=None, N=None, mode=None, surface=False, num_threads=5):
    """
    Prints the largest nodes, the relative errors and the computational times for the strong L^2-error approximation
    of the fractional kernel.
    :param params: Parameters of the rough Heston model
    :param N: Numpy array of values for N. Default is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    :param mode: List of modes of the quadrature rule. Default is ['paper', 'optimized', 'european']
    :param surface: Since T is a vector here, it is not clear whether smiles or surfaces have to be compared. If
        True, compares surfaces, else compares smiles
    :param num_threads: Number of parallel processes
    :return: Arrays containing the largest nodes, relative kernel errors and relative smile errors
    """
    params = rHeston_params_grid(params)
    if surface:
        H, lambda_, rho, nu, theta, V_0 = flat_mesh(params['H'], params['lambda'], params['rho'], params['nu'],
                                                    params['theta'], params['V_0'])
        dictionaries = [{'S': params['S'], 'K': params['K'], 'H': H[i], 'T': params['T'], 'lambda': lambda_[i],
                         'rho': rho[i], 'nu': nu[i], 'theta': theta[i], 'V_0': V_0[i], 'rel_tol': params['rel_tol']}
                        for i in range(len(H))]
    else:
        H, lambda_, rho, nu, theta, V_0, T = flat_mesh(params['H'], params['lambda'], params['rho'], params['nu'],
                                                       params['theta'], params['V_0'], params['T'])
        dictionaries = [{'S': params['S'], 'K': np.exp(np.log(params['K'] * np.sqrt(T[i]))), 'H': H[i], 'T': T[i],
                         'lambda': lambda_[i], 'rho': rho[i], 'nu': nu[i], 'theta': theta[i], 'V_0': V_0[i],
                         'rel_tol': params['rel_tol']} for i in range(len(H))]
    N = set_array_default(N, np.arange(1, 11))
    mode = set_list_default(mode, ['paper', 'optimized', 'european'])
    with mp.Pool(processes=num_threads) as pool:
        result = pool.starmap(smile_errors, zip(dictionaries, itertools.repeat(N), itertools.repeat(mode),
                                                itertools.repeat(None), itertools.repeat(False), itertools.repeat(True),
                                                itertools.repeat(True), itertools.repeat(0), itertools.repeat(True)))
    result = np.asarray(result)
    largest_nodes = result[:, 0, :, :]
    kernel_errs = result[:, 1, :, :]
    smile_errs = result[:, 2, :, :]
    for i in range(len(N)):
        for j in range(len(mode)):
            print(f'For N={N[i]} and mode={mode[j]}, the largest node is {largest_nodes[0, j, i]:.3}.')
            print(f'For N={N[i]} and mode={mode[j]}, the kernel error is {100*kernel_errs[0, j, i]:.4}%.')
            print(f'For N={N[i]} and mode={mode[j]}, the largest smile error is '
                  f'{100*np.amax(smile_errs[:, j, i]):.4}%.')
            print(f'For N={N[i]} and mode={mode[j]}, the smallest smile error is '
                  f'{100*np.amin(smile_errs[:, j, i]):.4}%.')
            print(f'For N={N[i]} and mode={mode[j]}, the geometric average smile error is '
                  f'{100*geometric_average(smile_errs[:, j, i]):.4}%.')
            print(f'For N={N[i]} and mode={mode[j]}, the median smile error is '
                  f'{100*np.median(smile_errs[:, j, i]):.4}%.')
        print(f'For N={N[i]}, the smile error when using {mode[1]} instead of {mode[0]} is at most '
              f'{100 * np.amax(smile_errs[:, 1, i] / smile_errs[:, 0, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[1]} instead of {mode[0]} is at least '
              f'{100 * np.amin(smile_errs[:, 1, i] / smile_errs[:, 0, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[1]} instead of {mode[0]} is on average (geometrically) '
              f'{100 * geometric_average(smile_errs[:, 1, i] / smile_errs[:, 0, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[1]} instead of {mode[0]} is on average (median) '
              f'{100 * np.median(smile_errs[:, 1, i] / smile_errs[:, 0, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[2]} instead of {mode[1]} is at most '
              f'{100 * np.amax(smile_errs[:, 2, i] / smile_errs[:, 1, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[2]} instead of {mode[1]} is at least '
              f'{100 * np.amin(smile_errs[:, 2, i] / smile_errs[:, 1, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[2]} instead of {mode[1]} is on average (geometrically) '
              f'{100 * geometric_average(smile_errs[:, 2, i] / smile_errs[:, 1, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[2]} instead of {mode[1]} is on average (median) '
              f'{100 * np.median(smile_errs[:, 2, i] / smile_errs[:, 1, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[2]} instead of {mode[0]} is at most '
              f'{100 * np.amax(smile_errs[:, 2, i] / smile_errs[:, 0, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[2]} instead of {mode[0]} is at least '
              f'{100 * np.amin(smile_errs[:, 2, i] / smile_errs[:, 0, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[2]} instead of {mode[0]} is on average (geometrically) '
              f'{100 * geometric_average(smile_errs[:, 2, i] / smile_errs[:, 0, i]):.4}% the size.')
        print(f'For N={N[i]}, the smile error when using {mode[2]} instead of {mode[0]} is on average (median) '
              f'{100 * np.median(smile_errs[:, 2, i] / smile_errs[:, 0, i]):.4}% the size.')
    return largest_nodes, kernel_errs, smile_errs


def plot_rHeston_sample_path(params, N=2, N_time=1024, mode='european', vol_behaviour='hyperplane reset',
                             plot_vol=True):
    """
    Plots a sample path of the stock price under the Markovian approximation of the rough Heston model. If one of
    N, N_steps or modes is a numpy array, plots all these sample paths in the same plot.
    :param params: Parameters of the rough Heston model
    :param N: Integer or numpy array of values for the number of dimensions of the Markovian approximation
    :param N_time: Integer or numpy array of values for the number of time steps
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
        S_, V_, V_comp = rHeston_samples(params=params, N=N[i], m=1, N_time=N_time[i], mode=mode[i],
                                         vol_behaviour=vol_behaviour[i], sample_paths=True, return_times=None)
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
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(18)
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
    N_times = set_array_default(N_times, 2 ** np.arange(18))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])

    for N in Ns:
        for N_time in N_times:
            for vol_behaviour in vol_behaviours:
                for mode in modes:
                    filename = get_filename(N=N, mode=mode, vol_behaviour=vol_behaviour, N_time=N_time,
                                            kind='sample paths' if sample_paths else 'samples')
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
                                        rHeston_samples(params=params, N=N, N_time=N_time, WB=WB, m=WB.shape[1],
                                                        mode=mode, vol_behaviour=vol_behaviour, nodes=nodes,
                                                        weights=weights)
                                if sample_paths:
                                    S, V, V_comp = rHeston_samples(params=params, N=N, m=WB.shape[1], N_time=N_time,
                                                                   WB=WB, mode=mode, vol_behaviour=vol_behaviour,
                                                                   nodes=nodes, weights=weights,
                                                                   sample_paths=sample_paths, return_times=return_times)
                                    np.save(filename, S)
                                    np.save(filename, V)
                                    np.save(filename, V_comp)
                        if not sample_paths:
                            np.save(filename, final_S)


def compute_final_rHeston_stock_prices_parallelized(params, Ns=None, N_times=None, modes=None, vol_behaviours=None,
                                                    recompute=False, sample_paths=False, return_times=None,
                                                    num_threads=5):
    """
    Computes the final stock prices of 1000000 sample paths of the Markovian approximation of the rough Heston model.
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(18)
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param vol_behaviours: Behaviour of the volatility when it hits zero. Default is
        ['hyperplane reset', 'ninomiya victoir', 'sticky', 'hyperplane reflection', 'adaptive']
    :param recompute: If False, checks first whether the file in which the results will be saved already exist. If so,
        does not recompute the stock prices
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: May specify an array of times at which the sample path values should be returned. If None and
        sample_paths is True, this is equivalent to return_times = np.linspace(0, T, N_time+1)
    :param num_threads: Number of parallel processes
    :return: None, the stock prices are saved in a file
    """
    params = rHeston_params_grid(params)
    Ns = set_array_default(Ns, np.array([1, 2, 6]))
    N_times = set_array_default(N_times, 2 ** np.arange(18))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])
    H, lambda_, rho, nu, theta, V_0, T, N, N_time, mode, vol_behaviour = flat_mesh(params['H'], params['lambda'],
                                                                                   params['rho'], params['nu'],
                                                                                   params['theta'], params['V_0'],
                                                                                   params['T'], Ns, N_times, modes,
                                                                                   vol_behaviours)
    dictionaries = [{'S': params['S'], 'K': np.exp(np.log(params['K'] * np.sqrt(T[i]))), 'H': H[i], 'T': T[i],
                     'lambda': lambda_[i], 'rho': rho[i], 'nu': nu[i], 'theta': theta[i], 'V_0': V_0[i],
                     'rel_tol': params['rel_tol']} for i in range(len(H))]
    with mp.Pool(processes=num_threads) as pool:
        pool.starmap(compute_final_rHeston_stock_prices, zip(dictionaries, N, N_time, mode, vol_behaviour,
                                                             itertools.repeat(recompute),
                                                             itertools.repeat(sample_paths),
                                                             itertools.repeat(return_times)))


def plot_smile_errors_given_stock_prices(Ns, N_times, modes, vol_behaviours, nodes, markov_errors, total_errors,
                                         lower_total_errors, upper_total_errors, discretization_errors, plot=True):
    """
    Produces some plots for smile_errors_given_stock_prices and smile_errors_given_stock_prices_parallelized.
    :param Ns: Numpy array of dimensions N
    :param N_times: Numpy array of number of time steps
    :param modes: List of quadrature modes for approximating the kernel
    :param vol_behaviours: List of behaviours if the volatility hits zero
    :param nodes: The nodes associated to the modes and Ns
    :param markov_errors: The errors of the exact Markovian approximations
    :param total_errors: The errors of the simulated Markovian approximations
    :param lower_total_errors: The lower MC bounds of the errors of the simulated Markovian approximations
    :param upper_total_errors: The upper MC bounds of the errors of the simulated Markovian approximations
    :param discretization_errors: The discretization errors of the simulated Markovian approximations
    :param plot: Plots various plots depending on this parameter
    :return: None
    """

    def finalize_plot_2():
        plt.xlabel('Number of time steps')
        plt.ylabel('Maximal relative error')
        plt.legend(loc='best')
        plt.show()

    def node_line_plots(col):
        nodes_1 = nodes[i][j, :]
        min_val = np.fmin(np.fmin(np.amin(lower_total_errors[i, j, k, :]),
                                  np.amin(discretization_errors[i, j, k, :])), markov_errors[i, j])
        max_val = np.fmax(np.fmax(np.amax(upper_total_errors[i, j, k, :]),
                                  np.amax(discretization_errors[i, j, k, :])), markov_errors[i, j])
        for node in nodes_1:
            if node >= 1:
                plt.loglog(np.array([node, node]), np.array([min_val, max_val]), color=col)

    if plot == 'N' or (isinstance(plot, bool) and plot):
        for j in range(len(modes)):
            for k in range(len(vol_behaviours)):
                for i in range(len(Ns)):
                    plt.loglog(N_times, total_errors[i, j, k, :], color=color(i, len(Ns)), label=f'N={Ns[i]}')
                    plt.loglog(N_times, lower_total_errors[i, j, k, :], '--', color=color(i, len(Ns)))
                    plt.loglog(N_times, upper_total_errors[i, j, k, :], '--', color=color(i, len(Ns)))
                    plt.loglog(N_times, discretization_errors[i, j, k, :], 'o-', color=color(i, len(Ns)))
                    plt.loglog(N_times, markov_errors[i, j] * np.ones(len(N_times)), 'x-', color=color(i, len(Ns)))
                    node_line_plots(col=color(i, len(Ns)))
                plt.title(f'Rough Heston with {modes[j]} quadrature rule\nand {vol_behaviours[k]}')
                finalize_plot_2()
    if plot == 'mode' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for k in range(len(vol_behaviours)):
                for j in range(len(modes)):
                    plt.loglog(N_times, total_errors[i, j, k, :], color=color(j, len(modes)), label=modes[j])
                    plt.loglog(N_times, lower_total_errors[i, j, k, :], '--', color=color(j, len(modes)))
                    plt.loglog(N_times, upper_total_errors[i, j, k, :], '--', color=color(j, len(modes)))
                    plt.loglog(N_times, discretization_errors[i, j, k, :], 'o-', color=color(j, len(modes)))
                    plt.loglog(N_times, markov_errors[i, j] * np.ones(len(N_times)), 'x-', color=color(j, len(modes)))
                    node_line_plots(col=color(j, len(modes)))
                plt.title(f'Rough Heston with {Ns[i]} nodes\nand {vol_behaviours[k]}')
                finalize_plot_2()
    if plot == 'vol_behaviour' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(modes)):
                for k in range(len(vol_behaviours)):
                    plt.loglog(N_times, total_errors[i, j, k, :], color=color(k, len(vol_behaviours)),
                               label=vol_behaviours[k])
                    plt.loglog(N_times, lower_total_errors[i, j, k, :], '--', color=color(k, len(vol_behaviours)))
                    plt.loglog(N_times, upper_total_errors[i, j, k, :], '--', color=color(k, len(vol_behaviours)))
                    plt.loglog(N_times, discretization_errors[i, j, k, :], 'o-', color=color(k, len(vol_behaviours)))
                    node_line_plots(col=color(k, len(vol_behaviours)))
                plt.loglog(N_times, markov_errors[i, j] * np.ones(len(N_times)), 'k-', label='Markovian error')
                plt.title(f'Rough Heston with {Ns[i]} nodes\nand {modes[j]} quadrature rule')
                finalize_plot_2()


def compute_smiles_given_stock_prices(params, Ns=None, N_times=None, modes=None, vol_behaviours=None, plot=True,
                                      true_smile=None, parallelizable_output=False):
    """
    Computes the implied volatility smiles given the final stock prices
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(18)
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param vol_behaviours: Behaviour of the volatility when it hits zero. Default is
        ['hyperplane reset', 'ninomiya victoir', 'sticky', 'hyperplane reflection', 'adaptive']
    :param plot: Creates plots depending on that parameter
    :param true_smile: The smile under the true rough Heston model can be specified here. Otherwise it is computed
    :param parallelizable_output: If True, returns only the Markovian errors, the total errors of the approximated
        smiles, the lower MC errors, the upper MC errors, the discretization errors of the approximated smiles, the
        lower MC discretization errors and the upper MC discretization errors, all of shape
        (len(Ns), len(modes), len(vol_behaviours), len(N_times)). If False, returns what is written below
    :return: The true smile, the Markovian smiles, the approximated smiles, the lower MC approximated smiles,
        the upper MC approximated smiles, the Markovian errors, the total errors of the approximated smiles, the lower
        MC errors, the upper MC errors, the discretization errors of the approximated smiles, the lower MC
        discretization errors and the upper MC discretization errors
    """
    params = rHeston_params(params)
    if not isinstance(params['T'], np.ndarray):
        params['T'] = np.array([params['T']])
    Ns = set_array_default(Ns, np.array([1, 2, 6]))
    N_times = set_array_default(N_times, 2 ** np.arange(18))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])
    if true_smile is None:
        true_smile = rHeston_iv_eur_call(params)

    total_errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    discretization_errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    lower_total_errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    lower_discretization_errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    upper_total_errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    upper_discretization_errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    markov_errors = np.empty((len(Ns), len(modes)))
    markov_smiles = np.empty((len(Ns), len(modes), len(params['T']), len(params['K'])))
    approx_smiles = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times), len(params['T']),
                              len(params['K'])))
    lower_smiles = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times), len(params['T']),
                             len(params['K'])))
    upper_smiles = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times), len(params['T']),
                             len(params['K'])))
    nodes = [np.empty((len(modes), N)) for N in Ns]

    for i in range(len(Ns)):
        for j in range(len(modes)):
            nodes_, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
            nodes[i][j, :] = nodes_
            markov_smiles[i, j, :, :] = rHestonMarkov_iv_eur_call(params=params, N=Ns[i], mode=modes[j], nodes=nodes_,
                                                                  weights=weights)
            markov_errors[i, j] = np.amax(np.abs(markov_smiles[i, j, :, :] - true_smile)/true_smile)
            print(f'N={Ns[i]}, {modes[j]}: Markovian error={100*markov_errors[i, j]:.4}%')
            for k in range(len(vol_behaviours)):
                for m in range(len(N_times)):
                    final_S = np.load(f'rHeston samples {Ns[i]} dim {modes[j]} {vol_behaviours[k]} {N_times[m]} time '
                                      + f'steps.npy')
                    vol, low, upp = cf.iv_eur_call_MC(S_0=params['S'], K=params['K'], T=params['T'], samples=final_S)
                    approx_smiles[i, j, k, m, :, :] = vol
                    lower_smiles[i, j, k, m, :, :] = low
                    upper_smiles[i, j, k, m, :, :] = upp
                    total_errors[i, j, k, m], lower_total_errors[i, j, k, m], upper_total_errors[i, j, k, m] = \
                        max_errors_MC(truth=true_smile, estimate=vol, lower=low, upper=upp)
                    discretization_errors[i, j, k, m], lower_discretization_errors[i, j, k, m], \
                        upper_discretization_errors[i, j, k, m] = \
                        max_errors_MC(truth=markov_smiles[i, j, :, :], estimate=vol, lower=low, upper=upp)
                    print(f'N={Ns[i]}, {modes[j]}, {vol_behaviours[k]}, N_time={N_times[m]}: total error='
                          + f'{100*total_errors[i, j, k, m]:.4}%, discretization error='
                          + f'{100*discretization_errors[i, j, k, m]:.4}%')

    k_vec = np.log(params['K'])

    def finalize_plot():
        plt.plot(k_vec, true_smile, color='k', label='Exact smile')
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
                    plt.title(f'Rough Heston with {modes[j]} quadrature rule,\n{vol_behaviours[k]} and {N_times[m]} '
                              + f'time steps')
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
                    plt.plot(k_vec, markov_smiles[i, j, -1, :], color='brown', label='Exact Markovian smile')
                    plt.title(f'Rough Heston with {Ns[i]} nodes,\n{modes[j]} quadrature rule and {N_times[m]} time '
                              + f'steps')
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
                    plt.plot(k_vec, markov_smiles[i, j, -1, :], color='brown', label='Exact Markovian smile')
                    plt.title(f'Rough Heston with {Ns[i]} nodes,\n{modes[j]} quadrature rule and {vol_behaviours[k]}')
                    finalize_plot()

    plot_smile_errors_given_stock_prices(Ns=Ns, N_times=N_times, modes=modes, vol_behaviours=vol_behaviours,
                                         nodes=nodes, markov_errors=markov_errors, total_errors=total_errors,
                                         lower_total_errors=lower_total_errors, upper_total_errors=upper_total_errors,
                                         discretization_errors=discretization_errors, plot=plot)

    if parallelizable_output:
        return markov_errors[:, :, None, None], total_errors, lower_total_errors, upper_total_errors, \
               discretization_errors, lower_discretization_errors, upper_discretization_errors
    return true_smile, markov_smiles, approx_smiles, lower_smiles, upper_smiles, markov_errors, total_errors, \
        lower_total_errors, upper_total_errors, discretization_errors, lower_discretization_errors, \
        upper_discretization_errors


def compute_smiles_given_stock_prices_parallelized(params, Ns=None, N_times=None, modes=None, vol_behaviours=None,
                                                   surface=False, plot=True, num_threads=5):
    """
    Parallelized version of compute_smiles_given_stock_prices.
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(18)
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param vol_behaviours: Behaviour of the volatility when it hits zero. Default is
        ['hyperplane reset', 'ninomiya victoir', 'sticky', 'hyperplane reflection', 'adaptive']
    :param plot: Creates plots depending on that parameter
    :param surface: If True, compares the surfaces. If False, compares the individual smiles for each maturity
    :param num_threads: Number of parallel processes
    :return: The Markovian errors, the total errors of the approximated smiles, the lower MC errors, the upper MC
        errors, the discretization errors of the approximated smiles, the lower MC discretization errors and the upper
        MC discretization errors
    """
    params = rHeston_params_grid(params)
    if not isinstance(params['T'], np.ndarray):
        params['T'] = np.array([params['T']])
    Ns = set_array_default(Ns, np.array([1, 2, 6]))
    N_times = set_array_default(N_times, 2 ** np.arange(18))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])
    if surface:
        H, lambda_, rho, nu, theta, V_0, N, N_time, mode, vol_behaviour = flat_mesh(params['H'], params['lambda'],
                                                                                    params['rho'], params['nu'],
                                                                                    params['theta'], params['V_0'], Ns,
                                                                                    N_times, modes, vol_behaviours)
        dictionaries = [{'S': params['S'], 'K': params['K'], 'H': H[i], 'T': params['T'], 'lambda': lambda_[i],
                         'rho': rho[i], 'nu': nu[i], 'theta': theta[i], 'V_0': V_0[i], 'rel_tol': params['rel_tol']}
                        for i in range(len(H))]
    else:
        H, lambda_, rho, nu, theta, V_0, T, N, N_time, mode, vol_behaviour = flat_mesh(params['H'], params['lambda'],
                                                                                       params['rho'], params['nu'],
                                                                                       params['theta'], params['V_0'],
                                                                                       params['T'], Ns, N_times, modes,
                                                                                       vol_behaviours)
        dictionaries = [{'S': params['S'], 'K': np.exp(np.log(params['K'] * np.sqrt(T[i]))), 'H': H[i], 'T': T[i],
                         'lambda': lambda_[i], 'rho': rho[i], 'nu': nu[i], 'theta': theta[i], 'V_0': V_0[i],
                         'rel_tol': params['rel_tol']} for i in range(len(H))]
    with mp.Pool(processes=num_threads) as pool:
        result = pool.starmap(compute_smiles_given_stock_prices, zip(dictionaries, N, N_time, mode, vol_behaviour,
                                                                     itertools.repeat(False), itertools.repeat(None),
                                                                     itertools.repeat(True)))
    markov_errors = result[:, 0, ...]
    total_errors = result[:, 1, ...]
    lower_total_errors = result[:, 2, ...]
    upper_total_errors = result[:, 3, ...]
    discretization_errors = result[:, 4, ...]
    lower_discretization_errors = result[:, 5, ...]
    upper_discretization_errors = result[:, 6, ...]

    markov_errors_ = np.amax(markov_errors, axis=0)
    ind_1 = argmax_indices(total_errors, axis=0)
    total_errors_ = total_errors[ind_1]
    lower_total_errors_ = lower_total_errors[ind_1]
    upper_total_errors_ = upper_total_errors[ind_1]
    ind_2 = argmax_indices(discretization_errors, axis=0)
    discretization_errors_ = discretization_errors[ind_2]
    lower_discretization_errors_ = lower_discretization_errors[ind_2]
    upper_discretization_errors_ = upper_discretization_errors[ind_2]

    nodes = [np.empty((len(modes), N)) for N in Ns]
    for i in range(len(Ns)):
        for j in range(len(modes)):
            nodes_, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
            nodes[i][j, :] = nodes_

    plot_smile_errors_given_stock_prices(Ns=Ns, N_times=N_times, modes=modes, vol_behaviours=vol_behaviours,
                                         nodes=nodes, markov_errors=markov_errors_, total_errors=total_errors_,
                                         lower_total_errors=lower_total_errors_, upper_total_errors=upper_total_errors_,
                                         discretization_errors=discretization_errors_, plot=plot)
    return markov_errors, total_errors, lower_total_errors, upper_total_errors, discretization_errors, lower_discretization_errors, upper_discretization_errors


def compute_strong_discretization_errors(Ns=None, N_times=None, N_time_ref=131072, modes=None, vol_behaviours=None,
                                         plot='N_time'):
    """
    Computes the strong discretization error for the path simulation of the rough Heston model.
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(17)
    :param N_time_ref: Reference number of time steps. These solutions are taken as exact
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param vol_behaviours: Behaviour of the volatility when it hits zero. Default is
        ['hyperplane reset', 'ninomiya victoir', 'sticky', 'hyperplane reflection', 'adaptive']
    :param plot: Creates plots depending on that parameter
    :return: The errors, the lower MC errors, the upper MC errors, and the approximate convergence rates
    """
    H, T = 0.1, 1.
    Ns = set_array_default(Ns, np.array([1, 2, 6]))
    N_times = set_array_default(N_times, 2 ** np.arange(17))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])

    errors = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    stds = np.empty((len(Ns), len(modes), len(vol_behaviours), len(N_times)))
    nodes = [np.empty((len(modes), N)) for N in Ns]

    for i in range(len(Ns)):
        for j in range(len(modes)):
            nodes[i][j, :], _ = rk.quadrature_rule(H=H, N=Ns[i], T=T, mode=modes[j])
            for k in range(len(vol_behaviours)):
                filename = get_filename(N=Ns[i], mode=modes[j], vol_behaviour=vol_behaviours[k], N_time=N_time_ref,
                                        kind='samples')
                S = np.load(filename)
                for m in range(len(N_times)):
                    filename = get_filename(N=Ns[i], mode=modes[j], vol_behaviour=vol_behaviours[k], N_time=N_times[m],
                                            kind='samples')
                    S_approx = np.load(filename)
                    errs = np.abs(S - S_approx)**2
                    errors[i, j, k, m], stds[i, j, k, m] = cf.MC(errs)
    lower = np.sqrt(np.fmax(errors - stds, 0))
    upper = np.sqrt(errors + stds)
    errors = np.sqrt(errors)

    def node_line_plots(col_):
        nodes_1 = nodes[i][j, :]
        min_val = np.amin(lower[i, j, k, :])
        max_val = np.amax(upper[i, j, k, :])
        for node in nodes_1:
            if node >= 1:
                plt.loglog(np.array([node, node]), np.array([min_val, max_val]), color=col_)

    conv_rates = np.empty((len(Ns), len(modes), len(vol_behaviours)))
    if plot == 'N' or (isinstance(plot, bool) and plot):
        for j in range(len(modes)):
            for k in range(len(vol_behaviours)):
                subtext = ''
                for i in range(len(Ns)):
                    col = color(i, len(Ns))
                    conv_rates[i, j, k], _ = plot_log_linear_regression(x=N_times, y=errors[i, j, k, :], col=col,
                                                                        lab=f'N={Ns[i]}', line_style='o-', offset=4)
                    plt.loglog(N_times, lower[i, j, k, :], '--', color=col)
                    plt.loglog(N_times, upper[i, j, k, :], '--', color=col)
                    node_line_plots(col_=col)
                    if i != 0:
                        subtext += ', '
                    subtext += f'{Ns[i]}: {conv_rates[i, j, k]}'
                plt.title(f'Rough Heston strong discretization errors\nwith {modes[j]} quadrature rule and '
                          + f'{vol_behaviours[k]}')
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
                                                                        lab=modes[j], line_style='o-', offset=4)
                    plt.loglog(N_times, lower[i, j, k, :], '--', color=col)
                    plt.loglog(N_times, upper[i, j, k, :], '--', color=col)
                    node_line_plots(col_=col)
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
                                                                        lab=vol_behaviours[k], line_style='o-',
                                                                        offset=4)
                    plt.loglog(N_times, lower[i, j, k, :], '--', color=col)
                    plt.loglog(N_times, upper[i, j, k, :], '--', color=col)
                    node_line_plots(col_=col)
                    if i != 0:
                        subtext += ', '
                    subtext += f'{vol_behaviours[k]}: {conv_rates[i, j, k]}'
                plt.title(f'Rough Heston strong discretization errors\nwith {Ns[i]} nodes and {modes[j]} '
                          + f'quadrature rule')
                plt.legend(loc='upper right')
                plt.xlabel('Number of time steps')
                plt.ylabel(r'$L^2$' + '-error')
                plt.annotate(subtext, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                plt.show()

    return errors, lower, upper, conv_rates


def optimize_kernel_approximation_for_simulation(params, N=2, N_time=128, vol_behaviour='hyperplane reset',
                                                 true_smile=None, test=True):
    """
    Optimizes the nodes such that the error in the MC simulation of the implied volatility smile is as small as
    possible.
    :param params: Parameters of the rough Heston model
    :param N: Number of dimensions of the Markovian approximation
    :param N_time: Number of time steps
    :param vol_behaviour: Behaviour of the volatility when it hits zero
    :param true_smile: May specify the smile of the actual rough Heston model. If not, it is computed
    :param test: If True, uses 500000 samples for the optimization and then 500000 samples for testing to get the error.
        If False, uses 1000000 samples for the optimization and the same samples for testing to get the error.
    :return: The optimal nodes and weights, the true smile, the Markovian true smile, the approximated smile, the
        lower MC smile, the upper MC smile, the total error, the lower MC total error, the upper MC total error,
        the Markovian error, the discretization error, the lower MC discretization error, the upper MC
        discretization error, and the final sample values
    """
    params = rHeston_params(params)
    if true_smile is None:
        true_smile = rHeston_iv_eur_call(params)
    if test:
        WB = np.empty((2, 500000, N_time))
        for i in range(5):
            WB[:, 100000*i:100000*(i+1), :] = resize_WB(WB=load_WB(i), N_time=N_time)
    else:
        WB = np.empty((2, 1000000, N_time))
        for i in range(10):
            WB[:, 100000 * i:100000 * (i + 1), :] = resize_WB(WB=load_WB(i), N_time=N_time)
    WB = WB * np.sqrt(params['T'])

    def func(nodes_, full_output=False):
        weights_ = rk.error_optimal_weights(H=params['H'], T=params['T'], nodes=nodes_, output='error')[1]
        S_ = rHeston_samples(params=params, N=N, N_time=N_time, WB=WB, vol_behaviour=vol_behaviour, nodes=nodes_,
                             weights=weights_, sample_paths=False)
        approx_smile_, l_, u_ = cf.iv_eur_call_MC(S_0=params['S'], K=params['K'], T=params['T'], samples=S_)
        error = np.amax(np.abs(approx_smile_-true_smile)/true_smile)
        print(f'The current error is {100*error:.4}%, where we have the nodes {nodes_}.')
        if full_output:
            return error, approx_smile_, l_, u_, S_
        return error

    nodes = np.exp(np.arange(N))
    res = scipy.optimize.minimize(lambda x: func(x, full_output=False), x0=nodes, bounds=((1e-06, None),) * N,
                                  method='Powell')
    nodes = res.x
    weights = rk.error_optimal_weights(H=params['H'], T=params['T'], nodes=nodes, output='error')[1]
    if test:
        for i in range(5):
            WB[:, 100000*i:100000*(i+1), :] = resize_WB(WB=load_WB(i+5), N_time=N_time)
        WB = WB * np.sqrt(params['T'])
    total_error, approx_smile, lower_smile, upper_smile, S = func(nodes, full_output=True)
    total_error, lower_error, upper_error = max_errors_MC(truth=true_smile, estimate=approx_smile, lower=lower_smile,
                                                          upper=upper_smile)

    markov_smile = rHestonMarkov_iv_eur_call(params=params, nodes=nodes, weights=weights)
    discretization_error, lower_disc_error, upper_disc_error = max_errors_MC(truth=markov_smile, estimate=approx_smile,
                                                                             lower=lower_smile, upper=upper_smile)
    markov_error = np.amax(np.abs(markov_smile - true_smile)/true_smile)

    print(f'Total error = {100*total_error:.4}%, Markovian error = {100*markov_error:.4}%, discretization error = '
          + f'{100*discretization_error:.4}%,\nnodes={nodes}')

    return nodes, weights, true_smile, markov_smile, approx_smile, lower_smile, upper_smile, total_error, lower_error, \
        upper_error, markov_error, discretization_error, lower_disc_error, upper_disc_error, S


def optimize_kernel_approximation_for_simulation_vector_inputs(params, Ns=None, N_times=None, vol_behaviours=None,
                                                               true_smile=None, plot='N_time', recompute=False):
    """
    Optimizes the nodes such that the error in the MC simulation of the implied volatility smile is as small as
    possible.
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of numbers of dimensions of the Markovian approximation. Default is [1, 2, 3]
    :param N_times: Numpy array of numbers of time steps. Default is 2 ** np.arange(1, 12)
    :param vol_behaviours: List of behaviours of the volatility when it hits zero. Default is
        ['hyperplane reset', 'ninomiya victoir', 'sticky', 'hyperplane reflection', 'adaptive']
    :param true_smile: May specify the smile of the actual rough Heston model. If not, it is computed
    :param plot: Creates plots depending on that parameter
    :param recompute: If False, checks first whether the file in which the results will be saved already exist. If so,
        does not recompute the stock prices
    :return: The largest nodes, the true smile, the Markovian true smiles, the approximated smiles, the
        lower MC smiles, the upper MC smiles, the total errors, the lower MC total errors, the upper MC total errors,
        the Markovian errors, the discretization errors, the lower MC discretization errors, and the upper MC
        discretization errors
    """
    params = rHeston_params(params)
    if not isinstance(params['T'], np.ndarray):
        params['T'] = np.array([params['T']])
    Ns = set_array_default(Ns, np.array([1, 2, 3]))
    N_times = set_array_default(N_times, 2 ** np.arange(1, 12))
    vol_behaviours = set_list_default(vol_behaviours, ['hyperplane reset', 'ninomiya victoir', 'sticky',
                                                       'hyperplane reflection', 'adaptive'])
    largest_nodes = np.empty((len(Ns), len(vol_behaviours), len(N_times)))
    if true_smile is None:
        true_smile = rHeston_iv_eur_call(params)
    elif len(true_smile.shape) == 1:
        true_smile = true_smile[None, :]
    markov_smiles = np.empty((len(Ns), len(params['T']), len(params['K'])))
    approx_smiles = np.empty((len(Ns), len(vol_behaviours), len(N_times), len(params['T']), len(params['K'])))
    lower_smiles = np.empty((len(Ns), len(vol_behaviours), len(N_times), len(params['T']), len(params['K'])))
    upper_smiles = np.empty((len(Ns), len(vol_behaviours), len(N_times), len(params['T']), len(params['K'])))
    total_errors = np.empty((len(Ns), len(vol_behaviours), len(N_times)))
    lower_total_errors = np.empty((len(Ns), len(vol_behaviours), len(N_times)))
    upper_total_errors = np.empty((len(Ns), len(vol_behaviours), len(N_times)))
    markov_errors = np.empty(len(Ns))
    discretization_errors = np.empty((len(Ns), len(vol_behaviours), len(N_times)))
    lower_discretization_errors = np.empty((len(Ns), len(vol_behaviours), len(N_times)))
    upper_discretization_errors = np.empty((len(Ns), len(vol_behaviours), len(N_times)))

    for i in range(len(Ns)):
        for j in range(len(vol_behaviours)):
            for k in range(len(N_times)):
                filename = get_filename(N=Ns[i], mode='fitted', vol_behaviour=vol_behaviours[j], N_time=N_times[k],
                                        kind='samples')
                if not recompute and exists(filename):
                    pass
                else:
                    print(f'Now optimizing N={Ns[i]}, vol_behaviour={vol_behaviours[j]}, N_time={N_times[k]}')
                    nodes, _, _, markov_smiles[i, :, :], approx_smiles[i, j, k, :, :], lower_smiles[i, j, k],\
                        upper_smiles[i, j, k], total_errors[i, j, k], lower_total_errors[i, j, k], \
                        upper_total_errors[i, j, k], markov_errors[i], discretization_errors[i, j, k], \
                        lower_discretization_errors[i, j, k], upper_discretization_errors[i, j, k], S = \
                        optimize_kernel_approximation_for_simulation(params=params, N=Ns[i], N_time=N_times[k],
                                                                     vol_behaviour=vol_behaviours[j],
                                                                     true_smile=true_smile, test=True)
                    np.save(filename, S)
                    filename = get_filename(N=Ns[i], mode='fitted', vol_behaviour=vol_behaviours[j], N_time=N_times[k],
                                            kind='nodes')
                    np.save(filename, nodes)
                    largest_nodes[i, j, k] = np.amax(nodes)
                    print(f'For N={Ns[i]}, vol_behaviours{vol_behaviours[j]}, N_time={N_times[k]}, we have the largest '
                          + f'node {largest_nodes[i, j, k]:.4}, a total error of {100*total_errors[i, j, k]:.4}%, '
                          + f'a Markovian error of {100*markov_errors[i]:.4}% and a discretization error of '
                          + f'{100*discretization_errors[i, j, k]:.4}%')

    k_vec = np.log(params['K'])

    def finalize_plot():
        plt.plot(k_vec, true_smile, color='k', label='Exact smile')
        plt.legend(loc='upper right')
        plt.xlabel('Log-moneyness')
        plt.ylabel('Implied volatility')
        plt.show()

    def finalize_plot_2():
        ax1.xlabel('Number of time steps')
        ax1.ylabel('Maximal relative error')
        ax2.ylabel('Largest node')
        ax1.legend(loc='best')
        plt.show()

    if plot == 'N' or (isinstance(plot, bool) and plot):
        for j in range(len(vol_behaviours)):
            for k in range(len(N_times)):
                for i in range(len(Ns)):
                    plt.plot(k_vec, approx_smiles[i, j, k, -1, :], color=color(i, len(Ns)), label=f'N={Ns[i]}')
                    if i == len(Ns) - 1:
                        plt.plot(k_vec, upper_smiles[i, j, k, -1, :], '--', color=color(i, len(Ns)))
                        plt.plot(k_vec, lower_smiles[i, j, k, -1, :], '--', color=color(i, len(Ns)))
                plt.title(f'Rough Heston with {vol_behaviours[j]}\nand {N_times[k]} time steps')
                finalize_plot()
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            for i in range(len(Ns)):
                ax1.loglog(N_times, total_errors[i, j, :], color=color(i, len(Ns)), label=f'N={Ns[i]}')
                ax1.loglog(N_times, lower_total_errors[i, j, :], '--', color=color(i, len(Ns)))
                ax1.loglog(N_times, upper_total_errors[i, j, :], '--', color=color(i, len(Ns)))
                ax1.loglog(N_times, discretization_errors[i, j, :], 'o-', color=color(i, len(Ns)))
                ax1.loglog(N_times, markov_errors[i, j] * np.ones(len(N_times)), 'x-', color=color(i, len(Ns)))
                ax2.loglog(N_times, largest_nodes[i, j, :], ':', color=color(i, len(Ns)))
            plt.title(f'Rough Heston with {vol_behaviours[j]}')
            finalize_plot_2()
    if plot == 'vol_behaviour' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for k in range(len(N_times)):
                for j in range(len(vol_behaviours)):
                    plt.plot(k_vec, approx_smiles[i, j, k, -1, :], color=color(j, len(vol_behaviours)),
                             label=vol_behaviours[j])
                    if j == len(vol_behaviours) - 1:
                        plt.plot(k_vec, upper_smiles[i, j, k, -1, :], '--', color=color(j, len(vol_behaviours)))
                        plt.plot(k_vec, lower_smiles[i, j, k, -1, :], '--', color=color(j, len(vol_behaviours)))
                plt.plot(k_vec, markov_smiles[i, -1, :], color='brown', label='Exact Markovian smile')
                plt.title(f'Rough Heston with {Ns[i]} nodes\nand {N_times[k]} time steps')
                finalize_plot()
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            for j in range(len(vol_behaviours)):
                ax1.loglog(N_times, total_errors[i, j, :], color=color(j, len(vol_behaviours)), label=vol_behaviours[j])
                ax1.loglog(N_times, lower_total_errors[i, j, :], '--', color=color(j, len(vol_behaviours)))
                ax1.loglog(N_times, upper_total_errors[i, j, :], '--', color=color(j, len(vol_behaviours)))
                ax1.loglog(N_times, discretization_errors[i, j, :], 'o-', color=color(j, len(vol_behaviours)))
                ax2.loglog(N_times, largest_nodes[i, j, :], ':', color=color(j, len(vol_behaviours)))
            ax1.loglog(N_times, markov_errors[i] * np.ones(len(N_times)), 'k-', label='Markovian error')
            plt.title(f'Rough Heston with {Ns[i]} nodes')
            finalize_plot_2()
    if plot == 'N_time' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(vol_behaviours)):
                for k in range(len(N_times)):
                    plt.plot(k_vec, approx_smiles[i, j, k, -1, :], color=color(k, len(N_times)),
                             label=f'{N_times[k]} time steps')
                    if k == len(N_times) - 1:
                        plt.plot(k_vec, upper_smiles[i, j, k, -1, :], '--', color=color(k, len(N_times)))
                        plt.plot(k_vec, lower_smiles[i, j, k, -1, :], '--', color=color(k, len(N_times)))
                plt.plot(k_vec, markov_smiles[i, -1, :], color='brown', label='Exact Markovian smile')
                plt.title(f'Rough Heston with {Ns[i]} nodes\nand {vol_behaviours[j]}')
                finalize_plot()
    return largest_nodes, true_smile, markov_smiles, approx_smiles, lower_smiles, upper_smiles, total_errors, \
        lower_total_errors, upper_total_errors, markov_errors, discretization_errors, lower_discretization_errors, \
        upper_discretization_errors


def simulation_errors_depending_on_node_size(params, N=1, N_times=None, vol_behaviour='sticky', true_smile=None,
                                             largest_nodes=None, plot=True, verbose=0):
    """
    Plots the errors in the implied volatility smile using simulation for different numbers of time steps and
    different largest nodes.
    :param params: Parameters of the rough Heston model
    :param N: Number of nodes
    :param N_times: Numpy array of number of time steps in the simulation. Default is 2 ** np.arange(12)
    :param vol_behaviour: Behaviour of the volatility when it hits 0
    :param true_smile: Can specify the true smile of the option. If left unspecified, it is computed
    :param largest_nodes: Numpy array of the largest nodes that should be used. Default is np.linspace(0, 10, 101)
    :param plot: If True, plots the results
    :param verbose: Determines how many intermediary results are printed to the console
    :return: The Markovian smiles, the Markovian errors, the total simulation errors, the lower MC simulation error
        bounds, the upper MC simulation error bounds, and the discretization errors (discretization + MC)
    """
    params = rHeston_params(params)
    if true_smile is None:
        true_smile = rHeston_iv_eur_call(params)
    N_times = set_array_default(N_times, 2 ** np.arange(12))
    largest_nodes = set_array_default(largest_nodes, np.linspace(0, 10, 101))

    markov_errors = np.empty(len(largest_nodes))
    total_errors = np.empty((len(N_times), len(largest_nodes)))
    lower_total_errors = np.empty((len(N_times), len(largest_nodes)))
    upper_total_errors = np.empty((len(N_times), len(largest_nodes)))
    discretization_errors = np.empty((len(N_times), len(largest_nodes)))
    markov_smiles = np.empty((len(largest_nodes), len(params['K'])))

    for i in range(len(N_times)):
        WB = np.empty((2, 1000000, N_times[i]))
        for j in range(10):
            WB[:, 100000 * j:100000 * (j + 1), :] = resize_WB(WB=load_WB(j), N_time=N_times[i])
        for j in range(len(largest_nodes)):
            if N == 1:
                nodes = np.array([largest_nodes[j]])
            else:
                nodes = np.linspace(0, largest_nodes[j], N)
            weights = rk.error_optimal_weights(H=params['H'], T=params['T'], nodes=nodes, output='error')[1]
            if i == 0:
                if verbose >= 1:
                    print(f'Simulating Markovian smile with N={N} and largest node={largest_nodes[j]}.')
                markov_smiles[j, :] = rHestonMarkov_iv_eur_call(params=params, N=N, mode=None, nodes=nodes,
                                                                weights=weights, load=False, save=False,
                                                                verbose=verbose-1)
                markov_errors[j] = np.amax(np.abs(true_smile - markov_smiles[j, :]) / true_smile)
            if verbose >= 1:
                print(f'Simulating paths with N={N}, N_times={N_times[i]}, largest node={largest_nodes[j]} '
                      f'and vol behaviour={vol_behaviour}.')
            S_ = rHeston_samples(params=params, N=N, N_time=N_times[i], WB=WB, m=1000000, mode=None,
                                 vol_behaviour=vol_behaviour, nodes=nodes, weights=weights, sample_paths=False,
                                 return_times=None)
            approx_smile, l, u = cf.iv_eur_call_MC(S_0=params['S'], K=params['K'], T=params['T'], samples=S_)
            total_errors[i, j], lower_total_errors[i, j], upper_total_errors[i, j] = max_errors_MC(truth=true_smile,
                                                                                                   estimate=
                                                                                                   approx_smile,
                                                                                                   lower=l, upper=u)
            discretization_errors[i, j], _, _ = max_errors_MC(truth=markov_smiles[j, :], estimate=approx_smile,
                                                              lower=l, upper=u)

    if plot:
        for i in range(len(N_times)):
            plt.plot(largest_nodes, total_errors[i, :], color=color(i, len(N_times)), label=f'{N_times[i]} time steps')
            plt.plot(largest_nodes, lower_total_errors[i, :], '--', color=color(i, len(N_times)))
            plt.plot(largest_nodes, upper_total_errors[i, :], '--', color=color(i, len(N_times)))
            plt.plot(largest_nodes, discretization_errors[i, :], ':', color=color(i, len(N_times)))
        plt.plot(largest_nodes, markov_errors, 'k-', label='Markovian error')
        plt.xlabel('Largest node')
        plt.ylabel('Relative error')
        plt.yscale('log')
        plt.title(f'Relative error of {vol_behaviour} volatility behaviour depending on\nnumber of time steps and '
                  f'largest node')
        plt.legend(loc='best')
        plt.show()

    return markov_smiles, markov_errors, total_errors, lower_total_errors, upper_total_errors, discretization_errors
