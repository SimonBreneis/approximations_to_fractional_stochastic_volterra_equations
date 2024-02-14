import time
import numpy as np
import matplotlib.pyplot as plt
import ComputationalFinance as cf
import rHestonFourier
import RoughKernel as rk
import rHestonMarkovSimulation as rHestonSP
from os.path import exists
import multiprocessing as mp
import itertools
import rHestonQESimulation
from scipy.stats import linregress


def profile(statement):
    """
    Profiles statement. Only call in main file under if __name__ == '__main__': clause.
    :param statement: The statement (function call) that should be profiled. Is a string
    :return: Nothing, but prints the results
    """
    import cProfile
    import pstats
    cProfile.run(statement, "{}.profile".format(__file__))
    stats_ = pstats.Stats("{}.profile".format(__file__))
    stats_.strip_dirs()
    stats_.sort_stats("cumtime").print_stats(100)
    stats_.sort_stats("tottime").print_stats(100)


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
                     'nu': 0.3, 'theta': 0.02, 'V_0': 0.02, 'rel_tol': 1e-05, 'r': 0.}
    difficult_params = {'S': 1., 'K': np.exp(np.linspace(-0.5, 0.1, 181)), 'H': 0.07, 'T': 0.04, 'lambda': 0.6,
                        'rho': -0.8, 'nu': 0.5, 'theta': 0.01, 'V_0': 0.01, 'rel_tol': 1e-05, 'r': 0.}
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


def get_filename(kind, N=None, mode=None, euler=False, antithetic=True, N_time=None, params=None,
                 truth=False, markov=False):
    """
    Filename under which we save our simulated data.
    :param N: Number of nodes
    :param mode: The kind of quadrature rule we use
    :param N_time: The number of time steps
    :param kind: The kind of object we are saving (i.e. 'samples' or 'sample paths' or 'iv european call', etc.)
    :param params: If None, assumes params is simple_params and does not include it in the file name
    :param truth: True if the smile is a true smile, False if not, i.e. if the smile is approximated
    :param markov: True if the smile is an exact Markovian smile. False if not, i.e. if it is a true smile or an
        approximated Markovian smile
    :param euler: If True, uses the Euler method, else, uses moment matching
    :param antithetic: If True uses antithetic variates
    :return: A string representing the file name
    """
    H = params['H']
    lambda_ = params['lambda']
    rho = params['rho']
    nu = params['nu']
    theta = params['theta']
    V_0 = params['V_0']
    T = params['T']
    r = params['r']
    K = (np.log(np.amin(params['K'])), np.log(np.amax(params['K'])), len(params['K']))
    K_string = f'K=({K[0]:.4}, {K[1]:.4}, {K[2]})'
    if isinstance(T, np.ndarray) and len(T) > 1:
        T_string = f'T=({np.amin(T):.4}, {np.amax(T):.4}, {len(T)})'
    elif isinstance(T, np.ndarray):
        T_string = f'T={T[0]:.4}'
    else:
        T_string = f'T={T:.4}'
    if r == 0.:
        r_string = ''
    else:
        r_string = f'r = {r:.3}, '

    if truth:
        return f'rHeston {kind}, H={H:.3}, lambda={lambda_:.3}, rho={rho:.3}, nu={nu:.3}, theta={theta:.3}, ' \
               f'{r_string}V_0={V_0:.3}, {T_string}, {K_string}.npy'
    else:
        if markov:
            return f'rHeston {kind} {N} dim {mode}, H={H:.3}, lambda={lambda_:.3}, rho={rho:.3}, nu={nu:.3}, ' \
                   f'theta={theta:.3}, {r_string}V_0={V_0:.3}, {T_string}, {K_string}.npy'
        else:
            vol_simulation = 'euler' if euler else 'mackevicius'
            antith = ' antithetic' if antithetic else ''
            return f'rHeston {kind} {N} dim {mode} {vol_simulation}{antith} {N_time} time steps, H={H:.3}, ' \
                   f'lambda={lambda_:.3}, rho={rho:.3}, nu={nu:.3}, theta={theta:.3}, {r_string}V_0={V_0:.3}, ' \
                   f'{T_string}.npy'


def log_linear_regression(x, y):
    """
    Applies log-linear regression of y against x.
    :param x: The argument
    :param y: The function value
    :return: Exponent, constant, R-value, p-value, empirical standard deviation
    """
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x), np.log(y))
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


def rHeston_samples(params, N_time=-1, m=1, nodes=None, weights=None, sample_paths=False, return_times=None,
                    vol_only=False, euler=False):
    """
    Shorthand for rHestonSP.sample_values.
    """
    params = rHeston_params(params)
    return rHestonSP.samples(lambda_=params['lambda'], rho=params['rho'], nu=params['nu'], theta=params['theta'],
                             V_0=params['V_0'], T=params['T'], S_0=params['S'], r=params['r'], N_time=N_time, m=m,
                             nodes=nodes, weights=weights, sample_paths=sample_paths, return_times=return_times,
                             euler=euler, vol_only=vol_only)


def rHestonFourier_iv_eur_call(params, N=0, mode=None, nodes=None, weights=None, load=True, save=False, verbose=0):
    """
    Shorthand for rHestonFourier.iv_eur_call.
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
    is_true = N == 0 and (nodes is None or weights is None)
    filename = get_filename(kind='true' if is_true else 'Markov', params=params, truth=is_true, markov=not is_true, N=N,
                            mode=mode)
    print(filename)
    if load and exists(filename):
        return np.load(filename)
    try:
        result = rHestonFourier.eur_call_put(S_0=params['S'], K=params['K'], H=params['H'], lambda_=params['lambda'],
                                             rho=params['rho'], nu=params['nu'], theta=params['theta'],
                                             V_0=params['V_0'], T=params['T'], r=params['r'], rel_tol=params['rel_tol'],
                                             N=N, mode=mode, nodes=nodes, weights=weights, verbose=verbose,
                                             implied_vol=True)
    except RuntimeError:
        print('Did not converge in given time')
        if isinstance(params['T'], np.ndarray):
            return np.empty((len(params['T']), len(params['K'])))
        else:
            return np.empty(len(params['K']))
    if save:
        np.save(filename, result)
    return result


def rHestonFourier_price_geom_asian_call(params, N=0, mode=None, nodes=None, weights=None, load=True, save=False,
                                         verbose=0):
    """
    Shorthand for rHestonFourier.price_geom_asian_call.
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
    is_true = N == 0 and (nodes is None or weights is None)
    filename = get_filename(kind='true' if is_true else 'Markov', params=params, truth=is_true, markov=not is_true, N=N,
                            mode=mode)
    print(filename)
    if load and exists(filename):
        return np.load(filename)
    try:
        result = rHestonFourier.geom_asian_call_put(S_0=params['S'], K=params['K'], H=params['H'],
                                                    lambda_=params['lambda'], rho=params['rho'], nu=params['nu'],
                                                    theta=params['theta'], V_0=params['V_0'], T=params['T'],
                                                    rel_tol=params['rel_tol'], N=N, mode=mode, nodes=nodes,
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


def rHestonFourier_price_avg_vol_call(params, N=0, mode=None, nodes=None, weights=None, load=True, save=False,
                                      verbose=0):
    """
    Shorthand for rHestonFourier.price_avg_vol_call.
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
    is_true = N == 0 and (nodes is None or weights is None)
    filename = get_filename(kind='true vol' if is_true else 'Markov vol', params=params, truth=is_true,
                            markov=not is_true, N=N, mode=mode)
    print(filename)
    if load and exists(filename):
        return np.load(filename)
    try:
        result = rHestonFourier.price_avg_vol_call_put(K=params['K'], H=params['H'], lambda_=params['lambda'],
                                                       nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                                       T=params['T'], rel_tol=params['rel_tol'], N=N, mode=mode,
                                                       nodes=nodes, weights=weights, verbose=verbose)
    except RuntimeError:
        print('Did not converge in given time')
        if isinstance(params['T'], np.ndarray):
            return np.empty((len(params['T']), len(params['K'])))
        else:
            return np.empty(len(params['K']))
    if save:
        np.save(filename, result)
    return result


def max_errors_MC(truth, estimate, lower, upper):
    """
    Computes the maximal (l^infinity) relative error, as well as the maximal lower and upper MC bounds.
    :param truth: The exact values
    :param estimate: The approximated values (involving among other things MC simulation)
    :param lower: The lower MC confidence bound for the approximated values
    :param upper: The upper MC confidence bound for the approximated values
    :return: The maximal relative error of estimate, as well as corresponding MC lower and upper confidence bounds and
    a bound on the MC error
    """
    t, e, l, u = truth.flatten(), estimate.flatten(), lower.flatten(), upper.flatten()
    e_err, l_err, u_err = np.abs(e-t)/t, np.abs(l-t)/t, np.abs(u-t)/t
    positive_err_ind = (t-l) * (u-t) < 0
    l_err_vec = np.zeros(len(t))
    l_err_vec[positive_err_ind] = np.fmin(l_err, u_err)[positive_err_ind]
    u_err_vec = np.fmax(l_err, u_err)
    MC_err = np.fmax(u - e, e - l) / np.abs(e)
    return np.amax(e_err), np.amax(l_err_vec), np.amax(u_err_vec), np.amax(MC_err)


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
                rk.error_l2(H=H, nodes=nodes, weights=weights, T=T, output='error'), 0)) / ker_norm)
            if verbose >= 1:
                print(f'N={Ns[i]}, mode={modes[j]}, node={largest_nodes[j, i]:.3}, error={100*kernel_errs[j, i]:.4}%, '
                      + f'time={duration[j, i]:.3}sec')
    return largest_nodes, kernel_errs, duration


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
    modes = set_list_default(modes, ['paper', 'optimized', 'european'])

    kernel_errs = np.empty((len(modes), len(Ns)))
    largest_nodes = np.empty((len(modes), len(Ns)))
    smile_errs = np.empty((len(modes), len(Ns)))

    if true_smile is None:
        true_smile = rHestonFourier_iv_eur_call(params=params, load=load, save=save, verbose=verbose)

    if isinstance(params['T'], np.ndarray):
        approx_smiles = np.empty((len(modes), len(Ns), len(params['T']), len(params['K'])))
    else:
        approx_smiles = np.empty((len(modes), len(Ns), len(params['K'])))

    ker_norm = rk.kernel_norm(H=params['H'], T=params['T'])

    for i in range(len(Ns)):
        for j in range(len(modes)):
            nodes, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
            approx_smiles[j, i] = rHestonFourier_iv_eur_call(params=params, N=Ns[i], mode=modes[j], nodes=nodes,
                                                             weights=weights, verbose=verbose, load=load, save=save)
            largest_nodes[j, i] = np.amax(nodes)
            kernel_errs[j, i] = np.amax(np.sqrt(rk.error_l2(H=params['H'], nodes=nodes, weights=weights, T=params['T'],
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


def compute_rHeston_samples(params, Ns=None, N_times=None, modes=None, euler=False, antithetic=True, recompute=False,
                            sample_paths=False, return_times=None, m=1000000, vol_only=False):
    """
    Computes samples of the Markovian approximation of the rough Heston model.
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 6]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(18)
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param euler: If True uses the Euler scheme, else uses moment matching
    :param antithetic: If True uses antithetic variates
    :param recompute: If False, checks first whether the file in which the results will be saved already exist. If so,
        does not recompute the stock prices
    :param sample_paths: If True, returns the entire sample paths, not just the final values. Also returns the sample
        paths of the square root of the volatility and the components of the volatility
    :param return_times: May specify an array of times at which the sample path values should be returned. If None and
        sample_paths is True, this is equivalent to return_times = np.linspace(0, T, N_time+1)
    :param m: Number of samples - only relevant for vol_behaviour = mackevicius
    :param vol_only: If True, simulates only the volatility process
    :return: None, the stock prices are saved in a file
    """
    params = rHeston_params(params)
    Ns = set_array_default(Ns, np.array([1, 2, 6]))
    N_times = set_array_default(N_times, 2 ** np.arange(18))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])

    filename, result = None, None
    for N in Ns:
        for N_time in N_times:
            for mode in modes:
                kind = 'volatility ' if vol_only else ''
                kind = kind + 'sample paths' if sample_paths else 'samples'
                filename = get_filename(N=N, mode=mode, euler=euler, antithetic=antithetic, N_time=N_time,
                                        kind=kind, params=params, truth=False, markov=False)
                print(f'Now simulating {filename}')
                if recompute or not exists(filename):
                    nodes, weights = rk.quadrature_rule(H=params['H'], N=N, T=params['T'], mode=mode)
                    params_ = params.copy()
                    if isinstance(params['T'], np.ndarray):
                        params_['T'] = params['T'][-1]
                    result = rHeston_samples(params=params_, m=m, N_time=N_time, nodes=nodes, weights=weights,
                                             sample_paths=sample_paths, return_times=return_times,
                                             vol_only=vol_only, euler=euler)
                    # np.save(filename, result)
    return filename, result


def plot_smile_errors_given_stock_prices(Ns, N_times, modes, euler, antithetic, nodes, markov_errors, total_errors,
                                         lower_total_errors, upper_total_errors, discretization_errors, plot=True):
    """
    Produces some plots for smile_errors_given_stock_prices and smile_errors_given_stock_prices_parallelized.
    :param Ns: Numpy array of dimensions N
    :param N_times: Numpy array of number of time steps
    :param modes: List of quadrature modes for approximating the kernel
    :param euler: If True uses the Euler scheme, else uses moment matching
    :param antithetic: If True uses antithetic variates
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
            for k in range(len(euler)):
                for m in range(len(antithetic)):
                    kind = 'Euler' if euler[k] else 'Mackevicius'
                    kind = kind + (' antithetic' if antithetic[m] else '')
                    for i in range(len(Ns)):
                        plt.loglog(N_times, total_errors[i, j, k, m, :], color=color(i, len(Ns)), label=f'N={Ns[i]}')
                        plt.loglog(N_times, lower_total_errors[i, j, k, m, :], '--', color=color(i, len(Ns)))
                        plt.loglog(N_times, upper_total_errors[i, j, k, m, :], '--', color=color(i, len(Ns)))
                        plt.loglog(N_times, discretization_errors[i, j, k, m, :], 'o-', color=color(i, len(Ns)))
                        plt.loglog(N_times, markov_errors[i, j] * np.ones(len(N_times)), 'x-', color=color(i, len(Ns)))
                        node_line_plots(col=color(i, len(Ns)))
                    plt.title(f'Rough Heston with {modes[j]} quadrature rule\nand {kind}')
                    finalize_plot_2()
    if plot == 'mode' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for k in range(len(euler)):
                for m in range(len(antithetic)):
                    kind = 'Euler' if euler[k] else 'Mackevicius'
                    kind = kind + (' antithetic' if antithetic[m] else '')
                    for j in range(len(modes)):
                        plt.loglog(N_times, total_errors[i, j, k, m, :], color=color(j, len(modes)), label=modes[j])
                        plt.loglog(N_times, lower_total_errors[i, j, k, m, :], '--', color=color(j, len(modes)))
                        plt.loglog(N_times, upper_total_errors[i, j, k, m, :], '--', color=color(j, len(modes)))
                        plt.loglog(N_times, discretization_errors[i, j, k, m, :], 'o-', color=color(j, len(modes)))
                        plt.loglog(N_times, markov_errors[i, j] * np.ones(len(N_times)), 'x-',
                                   color=color(j, len(modes)))
                        node_line_plots(col=color(j, len(modes)))
                    plt.title(f'Rough Heston with {Ns[i]} nodes\nand {kind}')
                    finalize_plot_2()
    if plot == 'vol_behaviour' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(modes)):
                for k in range(len(euler)):
                    for m in range(len(antithetic)):
                        kind = 'Euler' if euler[k] else 'Mackevicius'
                        kind = kind + (' antithetic' if antithetic[m] else '')
                        plt.loglog(N_times, total_errors[i, j, k, m, :], color=color(k, 4),
                                   label=kind)
                        plt.loglog(N_times, lower_total_errors[i, j, k, m, :], '--', color=color(k, 4))
                        plt.loglog(N_times, upper_total_errors[i, j, k, m, :], '--', color=color(k, 4))
                        plt.loglog(N_times, discretization_errors[i, j, k, m, :], 'o-', color=color(k, 4))
                        node_line_plots(col=color(k, 4))
                plt.loglog(N_times, markov_errors[i, j] * np.ones(len(N_times)), 'k-', label='Markovian error')
                plt.title(f'Rough Heston with {Ns[i]} nodes\nand {modes[j]} quadrature rule')
                finalize_plot_2()


def compute_smiles_given_stock_prices(params, Ns=None, N_times=None, modes=None, euler=None, antithetic=None, plot=True,
                                      true_smile=None, option='european call'):
    """
    Computes the implied volatility smiles given the final stock prices
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 3]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(11)
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param euler: List of simulation schemes. Default is [True, False]
    :param antithetic: List of simulation schemes. Default is [False, True]
    :param plot: Creates plots depending on that parameter
    :param true_smile: The smile under the true rough Heston model can be specified here. Otherwise it is computed
    :param option: The kind of option that is used
    :return: The true smile, the Markovian smiles, the approximated smiles, the lower MC approximated smiles,
        the upper MC approximated smiles, the Markovian errors, the total errors of the approximated smiles, the lower
        MC errors, the upper MC errors, the discretization errors of the approximated smiles, the lower MC
        discretization errors and the upper MC discretization errors
    """
    params = rHeston_params(params)
    if not isinstance(params['T'], np.ndarray):
        params['T'] = np.array([params['T']])
    Ns = set_array_default(Ns, np.array([1, 2, 3]))
    N_times = set_array_default(N_times, 2 ** np.arange(11))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    euler = set_list_default(euler, [True, False])
    antithetic = set_list_default(antithetic, [False, True])
    if true_smile is None:
        if option == 'european call':
            true_smile = rHestonFourier_iv_eur_call(params)
        elif option == 'geometric asian call':
            true_smile = rHestonFourier_price_geom_asian_call(params)
        elif option == 'average volatility call':
            true_smile = rHestonFourier_price_avg_vol_call(params)
        else:
            raise NotImplementedError

    total_errors = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times)))
    discretization_errors = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times)))
    lower_total_errors = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times)))
    lower_discretization_errors = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times)))
    upper_total_errors = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times)))
    upper_discretization_errors = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times)))
    MC_errors = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times)))
    markov_errors = np.empty((len(Ns), len(modes)))
    markov_smiles = np.empty((len(Ns), len(modes), len(params['T']), len(params['K'])))
    approx_smiles = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times), len(params['T']),
                              len(params['K'])))
    lower_smiles = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times), len(params['T']),
                             len(params['K'])))
    upper_smiles = np.empty((len(Ns), len(modes), len(euler), len(antithetic), len(N_times), len(params['T']),
                             len(params['K'])))
    nodes = [np.empty((len(modes), N)) for N in Ns]

    for i in range(len(Ns)):
        for j in range(len(modes)):
            if option == 'european call':
                nodes_, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
                nodes[i][j, :] = nodes_
                markov_smiles[i, j, :, :] = rHestonFourier_iv_eur_call(params=params, N=Ns[i], mode=modes[j],
                                                                       nodes=nodes_, weights=weights)
            elif option == 'geometric asian call':
                nodes_, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=np.array([params['T'] / 2, params['T']]),
                                                     mode=modes[j])
                nodes[i][j, :] = nodes_
                markov_smiles[i, j, :, :] = rHestonFourier_price_geom_asian_call(params=params, N=Ns[i], mode=modes[j],
                                                                                 nodes=nodes_, weights=weights)

            elif option == 'average volatility call':
                nodes_, weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
                nodes[i][j, :] = nodes_
                markov_smiles[i, j, :, :] = rHestonFourier_price_avg_vol_call(params=params, N=Ns[i], mode=modes[j],
                                                                              nodes=nodes_, weights=weights)
            else:
                raise NotImplementedError
            markov_errors[i, j] = np.amax(np.abs(markov_smiles[i, j, :, :] - true_smile)/true_smile)
            print(f'N={Ns[i]}, {modes[j]}: Markovian error={100*markov_errors[i, j]:.4}%')
            for k_1 in range(len(euler)):
                for k_2 in range(len(antithetic)):
                    for m in range(len(N_times)):
                        if option == 'european call':
                            kind = 'samples'
                            samples = np.load(get_filename(kind=kind, N=Ns[i], mode=modes[j], euler=euler[k_1],
                                                           antithetic=antithetic[k_2], N_time=N_times[m], params=params,
                                                           truth=False, markov=False))
                            vol, low, upp = cf.eur_MC(S_0=params['S'], K=params['K'], T=params['T'],
                                                      samples=samples[0, :], antithetic=antithetic, r=params['r'],
                                                      implied_vol=True)
                        elif option == 'geometric asian call':
                            kind = 'sample paths'
                            samples = np.load(get_filename(kind=kind, N=Ns[i], mode=modes[j], euler=euler[k_1],
                                                           antithetic=antithetic[k_2], N_time=N_times[m], params=params,
                                                           truth=False, markov=False))
                            vol, low, upp = cf.price_geom_asian_call_MC(K=params['K'], samples=samples[0, :, :],
                                                                        antithetic=antithetic)
                        elif option == 'average volatility call':
                            kind = 'volatility sample paths'
                            samples = np.load(get_filename(kind=kind, N=Ns[i], mode=modes[j], euler=euler[k_1],
                                                           antithetic=antithetic[k_2], N_time=N_times[m], params=params,
                                                           truth=False, markov=False))
                            vol, low, upp = cf.price_avg_vol_call_MC(K=params['K'], samples=samples[0, :, :],
                                                                     antithetic=antithetic)
                        else:
                            raise NotImplementedError
                        approx_smiles[i, j, k_1, k_2, m, :, :] = vol
                        lower_smiles[i, j, k_1, k_2, m, :, :] = low
                        upper_smiles[i, j, k_1, k_2, m, :, :] = upp
                        total_errors[i, j, k_1, k_2, m], lower_total_errors[i, j, k_1, k_2, m], \
                            upper_total_errors[i, j, k_1, k_2, m], MC_errors[i, j, k_1, k_2, m] = \
                            max_errors_MC(truth=true_smile, estimate=vol, lower=low, upper=upp)
                        discretization_errors[i, j, k_1, k_2, m], lower_discretization_errors[i, j, k_1, k_2, m], \
                            upper_discretization_errors[i, j, k_1, k_2, m], MC_errors[i, j, k_1, k_2, m] = \
                            max_errors_MC(truth=markov_smiles[i, j, :, :], estimate=vol, lower=low, upper=upp)
                        kind_here = 'Euler' if euler[k_1] else 'Mackevicius'
                        kind_here = kind_here + (' antithetic' if antithetic[k_2] else '')
                        print(f'N={Ns[i]}, {modes[j]}, {kind_here}, N_time={N_times[m]}: total error='
                              f'{100 * total_errors[i, j, k_1, k_2, m]:.4}%, discretization error='
                              f'{100 * discretization_errors[i, j, k_1, k_2, m]:.4}%, MC error='
                              f'{100 * MC_errors[i, j, k_1, k_2, m]:.4}%, excess error='
                              f'{100 * lower_discretization_errors[i, j, k_1, k_2, m]:.4}%'
                              + ('' if m == 0 else
                                 f' improvement factor='
                                 f'{discretization_errors[i, j, k_1, k_2, m - 1] / discretization_errors[i, j, k_1, k_2, m]:.3}, '
                                 f'{upper_discretization_errors[i, j, k_1, k_2, m - 1] / lower_discretization_errors[i, j, k_1, k_2, m]:.3}, '
                                 f'{lower_discretization_errors[i, j, k_1, k_2, m - 1] / upper_discretization_errors[i, j, k_1, k_2, m]:.3}'))

    k_vec = np.log(params['K'])

    def finalize_plot():
        plt.plot(k_vec, true_smile[0, :], color='k', label='Exact smile')
        plt.legend(loc='upper right')
        plt.xlabel('Log-moneyness')
        plt.ylabel('Implied volatility')
        plt.show()

    if plot == 'N' or (isinstance(plot, bool) and plot):
        for j in range(len(modes)):
            for k_1 in range(len(euler)):
                for k_2 in range(len(antithetic)):
                    for m in range(len(N_times)):
                        for i in range(len(Ns)):
                            plt.plot(k_vec, approx_smiles[i, j, k_1, k_2, m, -1, :], color=color(i, len(Ns)),
                                     label=f'N={Ns[i]}')
                            if i == len(Ns) - 1:
                                plt.plot(k_vec, upper_smiles[i, j, k_1, k_2, m, -1, :], '--', color=color(i, len(Ns)))
                                plt.plot(k_vec, lower_smiles[i, j, k_1, k_2, m, -1, :], '--', color=color(i, len(Ns)))
                        kind_here = 'Euler' if euler[k_1] else 'Mackevicius'
                        kind_here = kind_here + (' antithetic' if antithetic[k_2] else '')
                        plt.title(f'Rough Heston with {modes[j]} quadrature rule,\n{kind_here} and {N_times[m]} '
                                  + f'time steps')
                        finalize_plot()
    if plot == 'mode' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for k_1 in range(len(euler)):
                for k_2 in range(len(antithetic)):
                    for m in range(len(N_times)):
                        for j in range(len(modes)):
                            plt.plot(k_vec, approx_smiles[i, j, k_1, k_2, m, -1, :], color=color(j, len(modes)),
                                     label=modes[j])
                            if j == len(modes) - 1:
                                plt.plot(k_vec, upper_smiles[i, j, k_1, k_2, m, -1, :], '--',
                                         color=color(j, len(modes)))
                                plt.plot(k_vec, lower_smiles[i, j, k_1, k_2, m, -1, :], '--',
                                         color=color(j, len(modes)))
                        kind_here = 'Euler' if euler[k_1] else 'Mackevicius'
                        kind_here = kind_here + (' antithetic' if antithetic[k_2] else '')
                        plt.title(f'Rough Heston with {Ns[i]} nodes,\n{kind_here} and {N_times[m]} time steps')
                        finalize_plot()
    if plot == 'vol_behaviour' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(modes)):
                for m in range(len(N_times)):
                    for k_1 in range(len(euler)):
                        for k_2 in range(len(antithetic)):
                            kind_here = 'Euler' if euler[k_1] else 'Mackevicius'
                            kind_here = kind_here + (' antithetic' if antithetic[k_2] else '')
                            plt.plot(k_vec, approx_smiles[i, j, k_1, k_2, m, -1, :], color=color(2 * k_1 + k_2, 4),
                                     label=kind_here)
                            if k_1 == len(euler) - 1 and k_2 == len(antithetic) - 1:
                                plt.plot(k_vec, upper_smiles[i, j, k_1, k_2, m, -1, :], '--',
                                         color=color(2 * k_1 + k_2, 4))
                                plt.plot(k_vec, lower_smiles[i, j, k_1, k_2, m, -1, :], '--',
                                         color=color(2 * k_1 + k_2, 4))
                    plt.plot(k_vec, markov_smiles[i, j, -1, :], color='brown', label='Exact Markovian smile')
                    plt.title(f'Rough Heston with {Ns[i]} nodes,\n{modes[j]} quadrature rule and {N_times[m]} time '
                              + f'steps')
                    finalize_plot()
    if plot == 'N_time' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(modes)):
                for k_1 in range(len(euler)):
                    for k_2 in range(len(antithetic)):
                        kind_here = 'Euler' if euler[k_1] else 'Mackevicius'
                        kind_here = kind_here + (' antithetic' if antithetic[k_2] else '')
                        for m in range(len(N_times)):
                            plt.plot(k_vec, approx_smiles[i, j, k_1, k_2, m, -1, :], color=color(m, len(N_times)),
                                     label=f'{N_times[m]} time steps')
                            if m == len(N_times) - 1:
                                plt.plot(k_vec, upper_smiles[i, j, k_1, k_2, m, -1, :], '--',
                                         color=color(m, len(N_times)))
                                plt.plot(k_vec, lower_smiles[i, j, k_1, k_2, m, -1, :], '--',
                                         color=color(m, len(N_times)))
                        plt.plot(k_vec, markov_smiles[i, j, -1, :], color='brown', label='Exact Markovian smile')
                        plt.title(f'Rough Heston with {Ns[i]} nodes,\n{modes[j]} quadrature rule and {kind_here}')
                        finalize_plot()

    plot_smile_errors_given_stock_prices(Ns=Ns, N_times=N_times, modes=modes, euler=euler, antithetic=antithetic,
                                         nodes=nodes, markov_errors=markov_errors, total_errors=total_errors,
                                         lower_total_errors=lower_total_errors, upper_total_errors=upper_total_errors,
                                         discretization_errors=discretization_errors, plot=plot)

    return true_smile, markov_smiles, approx_smiles, lower_smiles, upper_smiles, markov_errors, total_errors, \
        lower_total_errors, upper_total_errors, discretization_errors, lower_discretization_errors, \
        upper_discretization_errors


def plot_smile_errors_given_stock_prices_QE(Ns, N_times, simulator, nodes, markov_errors, total_errors,
                                            lower_total_errors, upper_total_errors, discretization_errors):
    """
    Produces some plots for smile_errors_given_stock_prices and smile_errors_given_stock_prices_parallelized.
    :param Ns: Numpy array of dimensions N
    :param N_times: Numpy array of number of time steps
    :param simulator: If True uses the Euler scheme, else uses moment matching
    :param nodes: The nodes associated to the modes and Ns
    :param markov_errors: The errors of the exact Markovian approximations
    :param total_errors: The errors of the simulated Markovian approximations
    :param lower_total_errors: The lower MC bounds of the errors of the simulated Markovian approximations
    :param upper_total_errors: The upper MC bounds of the errors of the simulated Markovian approximations
    :param discretization_errors: The discretization errors of the simulated Markovian approximations
    :return: None
    """

    def finalize_plot_2():
        plt.xlabel('Number of time steps')
        plt.ylabel('Maximal relative error')
        plt.legend(loc='best')
        plt.show()

    def node_line_plots(col):
        nodes_1 = nodes[i]
        min_val = np.fmin(np.fmin(np.amin(lower_total_errors[i, k, :]),
                                  np.amin(discretization_errors[i, k, :])), markov_errors[i])
        max_val = np.fmax(np.fmax(np.amax(upper_total_errors[i, k, :]),
                                  np.amax(discretization_errors[i, k, :])), markov_errors[i])
        for node in nodes_1:
            if node >= 1:
                plt.loglog(np.array([node, node]), np.array([min_val, max_val]), color=col)

    for k in range(len(simulator)):
        for i in range(len(Ns)):
            plt.loglog(N_times, total_errors[i, k, :], color=color(i, len(Ns)), label=f'N={Ns[i]}')
            plt.loglog(N_times, lower_total_errors[i, k, :], '--', color=color(i, len(Ns)))
            plt.loglog(N_times, upper_total_errors[i, k, :], '--', color=color(i, len(Ns)))
            plt.loglog(N_times, discretization_errors[i, k, :], 'o-', color=color(i, len(Ns)))
            plt.loglog(N_times, markov_errors[i] * np.ones(len(N_times)), 'x-', color=color(i, len(Ns)))
            node_line_plots(col=color(i, len(Ns)))
        plt.title(f'Rough Heston with and {simulator[k]}')
        finalize_plot_2()
    for i in range(len(Ns)):
        for k in range(len(simulator)):
            plt.loglog(N_times, total_errors[i, k, :], color=color(k, 3), label=simulator[k])
            plt.loglog(N_times, lower_total_errors[i, k, :], '--', color=color(k, 3))
            plt.loglog(N_times, upper_total_errors[i, k, :], '--', color=color(k, 3))
            plt.loglog(N_times, discretization_errors[i, k, :], 'o-', color=color(k, 3))
            node_line_plots(col=color(k, 3))
        plt.loglog(N_times, markov_errors[i] * np.ones(len(N_times)), 'k-', label='Markovian error')
        plt.title(f'Rough Heston with {Ns[i]} nodes')
        finalize_plot_2()


def compute_smiles_given_stock_prices_QMC(params, Ns=None, N_times=None, simulator=None, true_smile=None,
                                          option='european call', n_samples=2 ** 20, verbose=0):
    """
    Computes the implied volatility smiles given the final stock prices
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 3]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(11)
    :param simulator: List of simulation schemes. Default is [True, False]
    :param true_smile: The smile under the true rough Heston model can be specified here. Otherwise it is computed
    :param option: The kind of option that is used
    :param n_samples: Number of (Q)MC samples in the simulation
    :param verbose: Determines the number of intermediary results printed to the console
    :return: The true smile, the Markovian smiles, the approximated smiles, the lower MC approximated smiles,
        the upper MC approximated smiles, the Markovian errors, the total errors of the approximated smiles, the lower
        MC errors, the upper MC errors, the discretization errors of the approximated smiles, the lower MC
        discretization errors and the upper MC discretization errors
    """
    params = rHeston_params(params)
    if not isinstance(params['T'], np.ndarray):
        params['T'] = np.array([params['T']])
    Ns = set_array_default(Ns, np.array([1, 2, 3]))
    N_times = set_array_default(N_times, 2 ** np.arange(11))
    simulator = set_list_default(simulator, ['Euler', 'Weak', 'QE'])
    mode = 'BL2'
    if true_smile is None:
        if option == 'european call' or option == 'surface':
            true_smile = rHestonFourier_iv_eur_call(params)
        elif option == 'geometric asian call':
            true_smile = rHestonFourier_price_geom_asian_call(params)
        else:
            raise NotImplementedError

    total_errors = np.empty((len(Ns), len(simulator), len(N_times)))
    discretization_errors = np.empty((len(Ns), len(simulator), len(N_times)))
    lower_total_errors = np.empty((len(Ns), len(simulator), len(N_times)))
    lower_discretization_errors = np.empty((len(Ns), len(simulator), len(N_times)))
    upper_total_errors = np.empty((len(Ns), len(simulator), len(N_times)))
    upper_discretization_errors = np.empty((len(Ns), len(simulator), len(N_times)))
    MC_errors = np.empty((len(Ns), len(simulator), len(N_times)))
    markov_errors = np.empty(len(Ns))
    markov_smiles = np.empty((len(Ns), len(params['T']), len(params['K'])))
    approx_smiles = np.empty((len(Ns), len(simulator), len(N_times), len(params['T']), len(params['K'])))
    lower_smiles = np.empty((len(Ns), len(simulator), len(N_times), len(params['T']), len(params['K'])))
    upper_smiles = np.empty((len(Ns), len(simulator), len(N_times), len(params['T']), len(params['K'])))
    nodes = [np.empty(N) for N in Ns]

    for i in range(len(Ns)):
        if option == 'european call' or option == 'surface':
            nodes[i], weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=mode)
            markov_smiles[i, :, :] = rHestonFourier_iv_eur_call(params=params, N=Ns[i], mode=mode, nodes=nodes[i],
                                                                weights=weights, verbose=verbose - 1)
        elif option == 'geometric asian call':
            # originally (or better) T = np.array([params['T'] / 2, params['T']]) below
            nodes[i], weights = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=mode)
            markov_smiles[i, :, :] = \
                rHestonFourier_price_geom_asian_call(params=params, N=Ns[i], mode=mode, nodes=nodes[i], weights=weights,
                                                     verbose=verbose - 1)
        else:
            raise NotImplementedError
        markov_errors[i] = np.amax(np.abs(markov_smiles[i, :, :] - true_smile)/true_smile)
        print(f'Quadrature rule: nodes={nodes[i]}, weights={weights}')
        print(f'N={Ns[i]}: Markovian error={100*markov_errors[i]:.4}%, largest node={np.amax(nodes[i]):.4}')
        for k in range(len(simulator)):
            for m in range(len(N_times)):
                if simulator[k] == 'QE':
                    markovian = False
                    is_euler = True
                else:
                    markovian = True
                    if simulator[k] == 'euler' or simulator[k] == 'Euler':
                        is_euler = True
                    else:
                        is_euler = False
                if option == 'european call' or option == 'surface':
                    if markovian:
                        vol, low, upp = rHestonSP.eur(K=params['K'], lambda_=params['lambda'], rho=params['rho'],
                                                      nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                                      S_0=params['S'], T=float(np.amax(params['T'])), nodes=nodes[i],
                                                      weights=weights, r=params['r'], m=n_samples, N_time=N_times[m],
                                                      euler=is_euler, implied_vol=True,
                                                      n_maturities=len(params['T']) if option == 'surface' else None,
                                                      verbose=verbose - 1)
                    elif i == 0:
                        vol, low, upp = \
                            rHestonQESimulation.eur(H=params['H'], K=params['K'], lambda_=params['lambda'],
                                                    rho=params['rho'], nu=params['nu'], theta=params['theta'],
                                                    V_0=params['V_0'], S_0=params['S'], T=float(np.amax(params['T'])),
                                                    r=params['r'], m=n_samples, N_time=N_times[m], implied_vol=True,
                                                    n_maturities=len(params['T']) if option == 'surface' else None,
                                                    verbose=verbose - 1)
                    else:
                        vol, low, upp = approx_smiles[0, k, m, :, :], lower_smiles[0, k, m, :, :], \
                            upper_smiles[0, k, m, :, :]
                elif option == 'geometric asian call':
                    if markovian:
                        vol, low, upp = \
                            rHestonSP.price_geom_asian_call(K=params['K'], lambda_=params['lambda'], rho=params['rho'],
                                                            nu=params['nu'], theta=params['theta'], V_0=params['V_0'],
                                                            S_0=params['S'], T=float(params['T']), nodes=nodes[i],
                                                            weights=weights, r=params['r'], m=n_samples,
                                                            N_time=N_times[m], euler=is_euler, verbose=verbose - 1)
                    else:
                        vol, low, upp = \
                            rHestonQESimulation.price_geom_asian_call(H=params['H'], K=params['K'],
                                                                      lambda_=params['lambda'], rho=params['rho'],
                                                                      nu=params['nu'], theta=params['theta'],
                                                                      V_0=params['V_0'], S_0=params['S'],
                                                                      T=float(params['T']), r=params['r'], m=n_samples,
                                                                      N_time=N_times[m], verbose=verbose - 1)
                else:
                    raise NotImplementedError
                approx_smiles[i, k, m, :, :] = vol
                lower_smiles[i, k, m, :, :] = low
                upper_smiles[i, k, m, :, :] = upp
                print(true_smile.shape, vol.shape, low.shape, upp.shape)
                total_errors[i, k, m], lower_total_errors[i, k, m], upper_total_errors[i, k, m], MC_errors[i, k, m] = \
                    max_errors_MC(truth=true_smile, estimate=vol, lower=low, upper=upp)
                discretization_errors[i, k, m], lower_discretization_errors[i, k, m], \
                    upper_discretization_errors[i, k, m], MC_errors[i, k, m] = \
                    max_errors_MC(truth=markov_smiles[i, :, :], estimate=vol, lower=low, upper=upp)
                print(f'N={Ns[i]}, {simulator[k]}, N_time={N_times[m]}: total error='
                      f'{100 * total_errors[i, k, m]:.4}%, discretization error='
                      f'{100 * discretization_errors[i, k, m]:.4}%, MC error='
                      f'{100 * MC_errors[i, k, m]:.4}%, excess error='
                      f'{100 * lower_discretization_errors[i, k, m]:.4}%'
                      + ('' if m == 0 else
                         f' improvement factor='
                         f'{discretization_errors[i, k, m - 1] / discretization_errors[i, k, m]:.3}, '
                         f'{upper_discretization_errors[i, k, m - 1] / lower_discretization_errors[i, k, m]:.3}, '
                         f'{lower_discretization_errors[i, k, m - 1] / upper_discretization_errors[i, k, m]:.3}')
                      + f', Time: {time.strftime("%H:%M:%S", time.localtime())}')

    k_vec = np.log(params['K'])

    def finalize_plot():
        plt.plot(k_vec, true_smile[0, :], color='k', label='Exact smile')
        plt.legend(loc='upper right')
        plt.xlabel('Log-moneyness')
        plt.ylabel('Implied volatility')
        plt.show()
    '''
    for k in range(len(simulator)):
        for m in range(len(N_times)):
            for i in range(len(Ns)):
                plt.plot(k_vec, approx_smiles[i, k, m, -1, :], color=color(i, len(Ns)),
                         label=f'N={Ns[i]}')
                if i == len(Ns) - 1:
                    plt.plot(k_vec, upper_smiles[i, k, m, -1, :], '--', color=color(i, len(Ns)))
                    plt.plot(k_vec, lower_smiles[i, k, m, -1, :], '--', color=color(i, len(Ns)))
            plt.title(f'Rough Heston with {simulator[k]} and {N_times[m]} time steps')
            finalize_plot()
    for i in range(len(Ns)):
        for m in range(len(N_times)):
            for k in range(len(simulator)):
                plt.plot(k_vec, approx_smiles[i, k, m, -1, :], color=color(k, 3), label=simulator[k])
                if k == len(simulator) - 1:
                    plt.plot(k_vec, upper_smiles[i, k, m, -1, :], '--', color=color(k, 3))
                    plt.plot(k_vec, lower_smiles[i, k, m, -1, :], '--', color=color(k, 3))
            plt.plot(k_vec, markov_smiles[i, -1, :], color='brown', label='Exact Markovian smile')
            plt.title(f'Rough Heston with {Ns[i]} nodes and {N_times[m]} time steps')
            finalize_plot()
    for i in range(len(Ns)):
        for k in range(len(simulator)):
            for m in range(len(N_times)):
                plt.plot(k_vec, approx_smiles[i, k, m, -1, :], color=color(m, len(N_times)),
                         label=f'{N_times[m]} time steps')
                if m == len(N_times) - 1:
                    plt.plot(k_vec, upper_smiles[i, k, m, -1, :], '--', color=color(m, len(N_times)))
                    plt.plot(k_vec, lower_smiles[i, k, m, -1, :], '--', color=color(m, len(N_times)))
            plt.plot(k_vec, markov_smiles[i, -1, :], color='brown', label='Exact Markovian smile')
            plt.title(f'Rough Heston with {Ns[i]} nodes and {simulator[k]}')
            finalize_plot()

    plot_smile_errors_given_stock_prices_QE(Ns=Ns, N_times=N_times, simulator=simulator, nodes=nodes,
                                            markov_errors=markov_errors, total_errors=total_errors,
                                            lower_total_errors=lower_total_errors,
                                            upper_total_errors=upper_total_errors,
                                            discretization_errors=discretization_errors)
    '''
    return true_smile, markov_smiles, approx_smiles, lower_smiles, upper_smiles, markov_errors, total_errors, \
        lower_total_errors, upper_total_errors, discretization_errors, lower_discretization_errors, \
        upper_discretization_errors


def smile_errors_weak_Euler_QE(params, N, N_times):
    """
    Computes the implied volatility smiles given the final stock prices
    :param params: Parameters of the rough Heston model
    :param N: Number of dimensions of the Markovian approximation
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(11)
    :return: The true smile, the Markovian smiles, the approximated smiles, the lower MC approximated smiles,
        the upper MC approximated smiles, the Markovian errors, the total errors of the approximated smiles, the lower
        MC errors, the upper MC errors, the discretization errors of the approximated smiles, the lower MC
        discretization errors and the upper MC discretization errors
    """
    params = rHeston_params(params)
    if isinstance(params['T'], np.ndarray):
        params['T'] = params['T'][0]
    true_smile = rHestonFourier_iv_eur_call(params)

    total_errors = np.empty((3, len(N_times)))  # first component is 0 for weak scheme, 1 for Euler scheme and 2 for QE
    discretization_errors = np.empty((2, len(N_times)))
    lower_total_errors = np.empty((3, len(N_times)))
    lower_discretization_errors = np.empty((2, len(N_times)))
    upper_total_errors = np.empty((3, len(N_times)))
    upper_discretization_errors = np.empty((2, len(N_times)))
    MC_errors = np.empty((3, len(N_times)))
    approx_smiles = np.empty((3, len(N_times), len(params['K'])))
    lower_smiles = np.empty((3, len(N_times), len(params['K'])))
    upper_smiles = np.empty((3, len(N_times), len(params['K'])))

    nodes, weights = rk.quadrature_rule(H=params['H'], N=N, T=params['T'], mode='european')
    markov_smiles = rHestonFourier_iv_eur_call(params=params, nodes=nodes, weights=weights)
    markov_error = np.amax(np.abs(markov_smiles - true_smile)/true_smile)
    print(f'Markovian error={100 * markov_error:.4}%')

    for m in range(len(N_times)):
        for i in range(2):
            kind = 'samples'
            samples = np.load(get_filename(kind=kind, N=N, mode='BL2', euler=i == 1, antithetic=True,
                                           N_time=N_times[m], params=params, truth=False, markov=False))
            vol, low, upp = cf.eur_MC(S_0=params['S'], K=params['K'], T=params['T'], samples=samples[0, :],
                                      antithetic=True, r=params['r'], implied_vol=True)
            approx_smiles[i, m, :] = vol
            lower_smiles[i, m, :] = low
            upper_smiles[i, m, :] = upp
            total_errors[i, m], lower_total_errors[i, m], upper_total_errors[i, m], MC_errors[i, m] = \
                max_errors_MC(truth=true_smile, estimate=vol, lower=low, upper=upp)
            discretization_errors[i, m], lower_discretization_errors[i, m], \
                upper_discretization_errors[i, m], MC_errors[i, m] = \
                max_errors_MC(truth=markov_smiles, estimate=vol, lower=low, upper=upp)
            kind_here = 'Euler' if i == 1 else 'Weak'
            kind_here = kind_here + ' antithetic'
            print(f'N={N}, {kind_here}, N_time={N_times[m]}: total error={100 * total_errors[i, m]:.4}%, '
                  f'discretization error={100 * discretization_errors[i, m]:.4}%, '
                  f'MC error={100 * MC_errors[i, m]:.4}%, excess error={100 * lower_discretization_errors[i, m]:.4}%'
                  + ('' if m == 0 else
                     f' improvement factor={discretization_errors[i, m - 1] / discretization_errors[i, m]:.3}, '
                     f'{upper_discretization_errors[i, m - 1] / lower_discretization_errors[i, m]:.3}, '
                     f'{lower_discretization_errors[i, m - 1] / upper_discretization_errors[i, m]:.3}'))

        H, lambda_, rho, nu, theta, V_0, T_string = params['H'], params['lambda'], params['rho'], params['nu'], \
            params['theta'], params['V_0'], params['T']
        samples = np.load(f'rHeston samples HQE {N_times[m]} time steps, H={H:.3}, lambda={lambda_:.3}, '
                          f'rho={rho:.3}, nu={nu:.3}, theta={theta:.3}, V_0={V_0:.3}, T={T_string:.3}.npy')
        vol, low, upp = cf.eur_MC(S_0=params['S'], K=params['K'], T=params['T'], samples=samples[0, :],
                                  antithetic=True, r=params['r'], implied_vol=True)
        approx_smiles[2, m, :] = vol
        lower_smiles[2, m, :] = low
        upper_smiles[2, m, :] = upp
        total_errors[2, m], lower_total_errors[2, m], upper_total_errors[2, m], MC_errors[2, m] = \
            max_errors_MC(truth=true_smile, estimate=vol, lower=low, upper=upp)
        kind_here = 'QE'
        print(f'{kind_here}, N_time={N_times[m]}: total error={100 * total_errors[2, m]:.4}%, MC error='
              f'{100 * MC_errors[2, m]:.4}%, excess error={100 * lower_total_errors[2, m]:.4}%'
              + ('' if m == 0 else f' improvement factor={total_errors[2, m - 1] / total_errors[2, m]:.3}, '
                 f'{upper_total_errors[2, m - 1] / lower_total_errors[2, m]:.3}, '
                 f'{lower_total_errors[2, m - 1] / upper_total_errors[2, m]:.3}'))

    plt.loglog(N_times, total_errors[1, :], 'b-', label=f'Euler')
    plt.loglog(N_times, lower_total_errors[1, :], 'b--')
    plt.loglog(N_times, upper_total_errors[1, :], 'b--')
    plt.loglog(N_times, discretization_errors[1, :], 'o-', color='b')
    plt.loglog(N_times, total_errors[0, :], 'r-', label=f'Weak')
    plt.loglog(N_times, lower_total_errors[0, :], 'r--')
    plt.loglog(N_times, upper_total_errors[0, :], 'r--')
    plt.loglog(N_times, discretization_errors[0, :], 'o-', color='r')
    plt.loglog(N_times, total_errors[2, :], 'g-', label=f'QE')
    plt.loglog(N_times, lower_total_errors[2, :], 'g--')
    plt.loglog(N_times, upper_total_errors[2, :], 'g--')
    plt.loglog(N_times, markov_error * np.ones(len(N_times)), 'k-', label='Markov')
    min_val = np.fmin(np.fmin(np.amin(lower_total_errors), np.amin(discretization_errors)), markov_error)
    max_val = np.fmax(np.fmax(np.amax(upper_total_errors), np.amax(discretization_errors)), markov_error)
    for node in nodes:
        if node >= 1:
            plt.loglog(np.array([node, node]), np.array([min_val, max_val]), 'k-')
    plt.title(f'Maximal relative errors for European call smiles with N={N}')
    plt.xlabel('Number of time steps')
    plt.ylabel('Maximal relative error')
    plt.legend(loc='best')
    plt.show()

    return true_smile, markov_smiles, approx_smiles, lower_smiles, upper_smiles, markov_error, total_errors, \
        lower_total_errors, upper_total_errors, discretization_errors, lower_discretization_errors, \
        upper_discretization_errors


def surface_errors_weak_Euler_QE(params, N, N_times):
    """
    Computes the implied volatility smiles given the final stock prices
    :param params: Parameters of the rough Heston model
    :param N: Number of dimensions of the Markovian approximation
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(11)
    :return: The true smile, the Markovian smiles, the approximated smiles, the lower MC approximated smiles,
        the upper MC approximated smiles, the Markovian errors, the total errors of the approximated smiles, the lower
        MC errors, the upper MC errors, the discretization errors of the approximated smiles, the lower MC
        discretization errors and the upper MC discretization errors
    """
    params = rHeston_params(params)
    true_smile = rHestonFourier_iv_eur_call(params)
    T = params['T']

    total_errors = np.empty((3, len(N_times)))  # first component is 0 for weak scheme, 1 for Euler scheme and 2 for QE
    discretization_errors = np.empty((2, len(N_times)))
    lower_total_errors = np.empty((3, len(N_times)))
    lower_discretization_errors = np.empty((2, len(N_times)))
    upper_total_errors = np.empty((3, len(N_times)))
    upper_discretization_errors = np.empty((2, len(N_times)))
    MC_errors = np.empty((3, len(N_times)))
    approx_smiles = np.empty((3, len(N_times), len(T), len(params['K'])))
    lower_smiles = np.empty((3, len(N_times), len(T), len(params['K'])))
    upper_smiles = np.empty((3, len(N_times), len(T), len(params['K'])))

    nodes, weights = rk.quadrature_rule(H=params['H'], N=N, T=params['T'], mode='european')
    print(nodes, weights)
    markov_smiles = rHestonFourier_iv_eur_call(params=params, nodes=nodes, weights=weights)
    markov_error = np.amax(np.abs(markov_smiles - true_smile)/true_smile)
    print(f'Markovian error={100 * markov_error:.4}%')

    for m in range(len(N_times)):
        for i in range(2):
            kind = 'sample paths'
            samples = np.load(get_filename(kind=kind, N=N, mode='BL2', euler=i == 1, antithetic=True,
                                           N_time=N_times[m], params=params, truth=False, markov=False))
            samples = samples[0, ...]
            samples = samples[:, ::samples.shape[-1] // len(T)]
            samples = samples[:, 1:]
            for j in range(len(T)):
                vol, low, upp = cf.eur_MC(S_0=params['S'], K=np.exp(np.log(params['K']) * np.sqrt(params['T'][j])),
                                          T=params['T'][j], samples=samples[:, j], antithetic=True, r=params['r'],
                                          implied_vol=True)
                approx_smiles[i, m, j, :] = vol
                lower_smiles[i, m, j, :] = low
                upper_smiles[i, m, j, :] = upp
            total_errors[i, m], lower_total_errors[i, m], upper_total_errors[i, m], MC_errors[i, m] = \
                max_errors_MC(truth=true_smile, estimate=approx_smiles[i, m, :, :], lower=lower_smiles[i, m, :, :],
                              upper=upper_smiles[i, m, :, :])
            discretization_errors[i, m], lower_discretization_errors[i, m], \
                upper_discretization_errors[i, m], MC_errors[i, m] = \
                max_errors_MC(truth=markov_smiles, estimate=approx_smiles[i, m, :, :], lower=lower_smiles[i, m, :, :],
                              upper=upper_smiles[i, m, :, :])
            kind_here = 'Euler' if i == 1 else 'Weak'
            kind_here = kind_here + ' antithetic'
            print(f'N={N}, {kind_here}, N_time={N_times[m]}: total error={100 * total_errors[i, m]:.4}%, '
                  f'discretization error={100 * discretization_errors[i, m]:.4}%, '
                  f'MC error={100 * MC_errors[i, m]:.4}%, excess error={100 * lower_discretization_errors[i, m]:.4}%'
                  + ('' if m == 0 else
                     f' improvement factor={discretization_errors[i, m - 1] / discretization_errors[i, m]:.3}, '
                     f'{upper_discretization_errors[i, m - 1] / lower_discretization_errors[i, m]:.3}, '
                     f'{lower_discretization_errors[i, m - 1] / upper_discretization_errors[i, m]:.3}'))

        H, lambda_, rho, nu, theta, V_0, T_string = params['H'], params['lambda'], params['rho'], params['nu'], \
            params['theta'], params['V_0'], params['T'][-1]
        sam = np.load(f'rHeston sample paths HQE {N_times[m]} time steps, H={H:.3}, lambda={lambda_:.3}, '
                      f'rho={rho:.3}, nu={nu:.3}, theta={theta:.3}, V_0={V_0:.3}, T={T_string:.3}.npy')
        samples = np.empty((sam.shape[1], sam.shape[2] + 1))
        samples[:, 1:] = sam[0, :, :]
        samples = samples[:, ::samples.shape[-1] // len(T)]
        samples = samples[:, 1:]
        for j in range(len(T)):
            vol, low, upp = cf.eur_MC(S_0=params['S'], K=np.exp(np.log(params['K']) * np.sqrt(params['T'][j])),
                                      T=params['T'][j], samples=samples[:, j], antithetic=True, r=params['r'],
                                      implied_vol=True)
            approx_smiles[2, m, j, :] = vol
            lower_smiles[2, m, j, :] = low
            upper_smiles[2, m, j, :] = upp
        total_errors[2, m], lower_total_errors[2, m], upper_total_errors[2, m], MC_errors[2, m] = \
            max_errors_MC(truth=true_smile, estimate=approx_smiles[2, m, :, :], lower=lower_smiles[2, m, :, :],
                          upper=upper_smiles[2, m, :, :])
        kind_here = 'QE'
        print(f'{kind_here}, N_time={N_times[m]}: total error={100 * total_errors[2, m]:.4}%, MC error='
              f'{100 * MC_errors[2, m]:.4}%, excess error={100 * lower_total_errors[2, m]:.4}%'
              + ('' if m == 0 else f' improvement factor={total_errors[2, m - 1] / total_errors[2, m]:.3}, '
                 f'{upper_total_errors[2, m - 1] / lower_total_errors[2, m]:.3}, '
                 f'{lower_total_errors[2, m - 1] / upper_total_errors[2, m]:.3}'))

    plt.loglog(N_times, total_errors[1, :], 'b-', label=f'Euler')
    plt.loglog(N_times, lower_total_errors[1, :], 'b--')
    plt.loglog(N_times, upper_total_errors[1, :], 'b--')
    plt.loglog(N_times, discretization_errors[1, :], 'o-', color='b')
    plt.loglog(N_times, total_errors[0, :], 'r-', label=f'Weak')
    plt.loglog(N_times, lower_total_errors[0, :], 'r--')
    plt.loglog(N_times, upper_total_errors[0, :], 'r--')
    plt.loglog(N_times, discretization_errors[0, :], 'o-', color='r')
    plt.loglog(N_times, total_errors[2, :], 'g-', label=f'QE')
    plt.loglog(N_times, lower_total_errors[2, :], 'g--')
    plt.loglog(N_times, upper_total_errors[2, :], 'g--')
    plt.loglog(N_times, markov_error * np.ones(len(N_times)), 'k-', label='Markov')
    min_val = np.fmin(np.fmin(np.amin(lower_total_errors), np.amin(discretization_errors)), markov_error)
    max_val = np.fmax(np.fmax(np.amax(upper_total_errors), np.amax(discretization_errors)), markov_error)
    for node in nodes:
        if node >= 1:
            plt.loglog(np.array([node, node]), np.array([min_val, max_val]), 'k-')
    plt.title(f'Maximal relative errors for European call surfaces with N={N}')
    plt.xlabel('Number of time steps')
    plt.ylabel('Maximal relative error')
    plt.legend(loc='best')
    plt.show()

    return true_smile, markov_smiles, approx_smiles, lower_smiles, upper_smiles, markov_error, total_errors, \
        lower_total_errors, upper_total_errors, discretization_errors, lower_discretization_errors, \
        upper_discretization_errors


def asian_errors_weak_Euler_QE(params, N, N_times):
    """
    Computes the implied volatility smiles given the final stock prices
    :param params: Parameters of the rough Heston model
    :param N: Number of dimensions of the Markovian approximation
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(11)
    :return: The true smile, the Markovian smiles, the approximated smiles, the lower MC approximated smiles,
        the upper MC approximated smiles, the Markovian errors, the total errors of the approximated smiles, the lower
        MC errors, the upper MC errors, the discretization errors of the approximated smiles, the lower MC
        discretization errors and the upper MC discretization errors
    """
    params = rHeston_params(params)
    params_ = params.copy()
    params_['T'] = params['T'][-1]
    true_smile = rHestonFourier_price_geom_asian_call(params_)

    total_errors = np.empty((3, len(N_times)))  # first component is 0 for weak scheme, 1 for Euler scheme and 2 for QE
    discretization_errors = np.empty((2, len(N_times)))
    lower_total_errors = np.empty((3, len(N_times)))
    lower_discretization_errors = np.empty((2, len(N_times)))
    upper_total_errors = np.empty((3, len(N_times)))
    upper_discretization_errors = np.empty((2, len(N_times)))
    MC_errors = np.empty((3, len(N_times)))
    approx_smiles = np.empty((3, len(N_times), len(params['K'])))
    lower_smiles = np.empty((3, len(N_times), len(params['K'])))
    upper_smiles = np.empty((3, len(N_times), len(params['K'])))

    nodes, weights = rk.quadrature_rule(H=params['H'], N=N, T=params['T'], mode='european')
    print(nodes, weights)
    markov_smiles = rHestonFourier_price_geom_asian_call(params=params_, nodes=nodes, weights=weights)
    markov_error = np.amax(np.abs(markov_smiles - true_smile)/true_smile)
    print(f'Markovian error={100 * markov_error:.4}%')

    for m in range(len(N_times)):
        for i in range(2):
            kind = 'sample paths'
            samples = np.load(get_filename(kind=kind, N=N, mode='BL2', euler=i == 1, antithetic=True,
                                           N_time=N_times[m], params=params, truth=False, markov=False))
            vol, low, upp = cf.price_geom_asian_call_MC(K=params['K'], samples=samples[0, ...], antithetic=True)
            approx_smiles[i, m, :] = vol
            lower_smiles[i, m, :] = low
            upper_smiles[i, m, :] = upp
            total_errors[i, m], lower_total_errors[i, m], upper_total_errors[i, m], MC_errors[i, m] = \
                max_errors_MC(truth=true_smile, estimate=vol, lower=low, upper=upp)
            discretization_errors[i, m], lower_discretization_errors[i, m], \
                upper_discretization_errors[i, m], MC_errors[i, m] = \
                max_errors_MC(truth=markov_smiles, estimate=vol, lower=low, upper=upp)
            kind_here = 'Euler' if i == 1 else 'Weak'
            kind_here = kind_here + ' antithetic'
            print(f'N={N}, {kind_here}, N_time={N_times[m]}: total error={100 * total_errors[i, m]:.4}%, '
                  f'discretization error={100 * discretization_errors[i, m]:.4}%, '
                  f'MC error={100 * MC_errors[i, m]:.4}%, excess error={100 * lower_discretization_errors[i, m]:.4}%'
                  + ('' if m == 0 else
                     f' improvement factor={discretization_errors[i, m - 1] / discretization_errors[i, m]:.3}, '
                     f'{upper_discretization_errors[i, m - 1] / lower_discretization_errors[i, m]:.3}, '
                     f'{lower_discretization_errors[i, m - 1] / upper_discretization_errors[i, m]:.3}'))

        H, lambda_, rho, nu, theta, V_0, T_string = params['H'], params['lambda'], params['rho'], params['nu'], \
            params['theta'], params['V_0'], params['T'][-1]
        samples = np.load(f'rHeston sample paths HQE {N_times[m]} time steps, H={H:.3}, lambda={lambda_:.3}, '
                          f'rho={rho:.3}, nu={nu:.3}, theta={theta:.3}, V_0={V_0:.3}, T={T_string:.3}.npy')
        vol, low, upp = cf.price_geom_asian_call_MC(K=params['K'], samples=samples[0, ...], antithetic=True)
        approx_smiles[2, m, :] = vol
        lower_smiles[2, m, :] = low
        upper_smiles[2, m, :] = upp
        total_errors[2, m], lower_total_errors[2, m], upper_total_errors[2, m], MC_errors[2, m] = \
            max_errors_MC(truth=true_smile, estimate=vol, lower=low, upper=upp)
        kind_here = 'QE'
        print(f'{kind_here}, N_time={N_times[m]}: total error={100 * total_errors[2, m]:.4}%, MC error='
              f'{100 * MC_errors[2, m]:.4}%, excess error={100 * lower_total_errors[2, m]:.4}%'
              + ('' if m == 0 else f' improvement factor={total_errors[2, m - 1] / total_errors[2, m]:.3}, '
                 f'{upper_total_errors[2, m - 1] / lower_total_errors[2, m]:.3}, '
                 f'{lower_total_errors[2, m - 1] / upper_total_errors[2, m]:.3}'))

    plt.loglog(N_times, total_errors[1, :], 'b-', label=f'Euler')
    plt.loglog(N_times, lower_total_errors[1, :], 'b--')
    plt.loglog(N_times, upper_total_errors[1, :], 'b--')
    plt.loglog(N_times, discretization_errors[1, :], 'o-', color='b')
    plt.loglog(N_times, total_errors[0, :], 'r-', label=f'Weak')
    plt.loglog(N_times, lower_total_errors[0, :], 'r--')
    plt.loglog(N_times, upper_total_errors[0, :], 'r--')
    plt.loglog(N_times, discretization_errors[0, :], 'o-', color='r')
    plt.loglog(N_times, total_errors[2, :], 'g-', label=f'QE')
    plt.loglog(N_times, lower_total_errors[2, :], 'g--')
    plt.loglog(N_times, upper_total_errors[2, :], 'g--')
    plt.loglog(N_times, markov_error * np.ones(len(N_times)), 'k-', label='Markov')
    min_val = np.fmin(np.fmin(np.amin(lower_total_errors), np.amin(discretization_errors)), markov_error)
    max_val = np.fmax(np.fmax(np.amax(upper_total_errors), np.amax(discretization_errors)), markov_error)
    for node in nodes:
        if node >= 1:
            plt.loglog(np.array([node, node]), np.array([min_val, max_val]), 'k-')
    plt.title(f'Maximal relative errors for European call smiles with N={N}')
    plt.xlabel('Number of time steps')
    plt.ylabel('Maximal relative error')
    plt.legend(loc='best')
    plt.show()

    return true_smile, markov_smiles, approx_smiles, lower_smiles, upper_smiles, markov_error, total_errors, \
        lower_total_errors, upper_total_errors, discretization_errors, lower_discretization_errors, \
        upper_discretization_errors


def compute_strong_discretization_errors(params, Ns=None, N_times=None, N_time_ref=2048, modes=None, euler=None,
                                         plot='N_time'):
    """
    Computes the strong discretization error for the path simulation of the rough Heston model.
    :param params: Parameters of the rough Heston model
    :param Ns: Numpy array of values for the number of dimensions of the Markovian approximation. Default is [1, 2, 3]
    :param N_times: Numpy array of values for the number of time steps. Default is 2**np.arange(11)
    :param N_time_ref: Reference number of time steps. These solutions are taken as exact
    :param modes: List of modes of the quadrature rule. Default is ['european', 'optimized', 'paper']
    :param euler: List of simulation schemes. Default is [True, False]
    :param plot: Creates plots depending on that parameter
    :return: The errors, the lower MC errors, the upper MC errors, and the approximate convergence rates
    """
    Ns = set_array_default(Ns, np.array([1, 2, 3]))
    N_times = set_array_default(N_times, 2 ** np.arange(11))
    modes = set_list_default(modes, ['european', 'optimized', 'paper'])
    euler = set_list_default(euler, [True, False])

    errors = np.empty((len(Ns), len(modes), len(euler), len(N_times)))
    stds = np.empty((len(Ns), len(modes), len(euler), len(N_times)))
    nodes = [np.empty((len(modes), N)) for N in Ns]

    for i in range(len(Ns)):
        for j in range(len(modes)):
            nodes[i][j, :], _ = rk.quadrature_rule(H=params['H'], N=Ns[i], T=params['T'], mode=modes[j])
            for k in range(len(euler)):
                filename = get_filename(N=Ns[i], mode=modes[j], euler=euler[k], antithetic=False, N_time=N_time_ref,
                                        kind='samples', params=params, truth=False, markov=False)
                S = np.load(filename)
                for m in range(len(N_times)):
                    filename = get_filename(N=Ns[i], mode=modes[j], euler=euler[k], antithetic=False, N_time=N_times[m],
                                            kind='samples', params=params, truth=False, markov=False)
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

    conv_rates = np.empty((len(Ns), len(modes), len(euler)))
    if plot == 'N' or (isinstance(plot, bool) and plot):
        for j in range(len(modes)):
            for k in range(len(euler)):
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
                kind_here = 'Euler' if euler[k] else 'Mackevicius'
                plt.title(f'Rough Heston strong discretization errors\nwith {modes[j]} quadrature rule and '
                          + f'{kind_here}')
                plt.legend(loc='upper right')
                plt.xlabel('Number of time steps')
                plt.ylabel(r'$L^2$' + '-error')
                plt.annotate(subtext, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                plt.show()
    if plot == 'mode' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for k in range(len(euler)):
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
                kind_here = 'Euler' if euler[k] else 'Mackevicius'
                plt.title(f'Rough Heston strong discretization errors\nwith {Ns[i]} nodes and {kind_here}')
                plt.legend(loc='upper right')
                plt.xlabel('Number of time steps')
                plt.ylabel(r'$L^2$' + '-error')
                plt.annotate(subtext, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                plt.show()
    if plot == 'vol_behaviour' or (isinstance(plot, bool) and plot):
        for i in range(len(Ns)):
            for j in range(len(modes)):
                subtext = ''
                for k in range(len(euler)):
                    col = color(k, len(euler))
                    conv_rates[i, j, k], _ = plot_log_linear_regression(x=N_times, y=errors[i, j, k, :], col=col,
                                                                        lab=euler[k], line_style='o-',
                                                                        offset=4)
                    plt.loglog(N_times, lower[i, j, k, :], '--', color=col)
                    plt.loglog(N_times, upper[i, j, k, :], '--', color=col)
                    node_line_plots(col_=col)
                    if i != 0:
                        subtext += ', '
                    kind_here = 'Euler' if euler[k] else 'Mackevicius'
                    subtext += f'{kind_here}: {conv_rates[i, j, k]}'
                plt.title(f'Rough Heston strong discretization errors\nwith {Ns[i]} nodes and {modes[j]} '
                          + f'quadrature rule')
                plt.legend(loc='upper right')
                plt.xlabel('Number of time steps')
                plt.ylabel(r'$L^2$' + '-error')
                plt.annotate(subtext, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                plt.show()

    return errors, lower, upper, conv_rates


def illustrate_Markovian_approximation(H=0.2, T=1., n=10000):
    """
    Illustrates how the Markovian approximation works by plotting a sample path of fBm and an associated 2-dim
    Markovian approximation.
    :param H: Hurst parameter
    :param T: Final time
    :param n: Number of time steps
    :return: None
    """
    nodes = np.array([1., 400])
    weights = np.array([1.6, 8])
    exact_nodes, exact_weights = rk.quadrature_rule(H=H, N=6, T=T, mode="optimized")
    print(exact_nodes, exact_weights)
    dt = T / n

    B = np.random.normal(0, np.sqrt(dt), n)
    exp_nodes = np.exp(-nodes * dt)
    div_nodes = (1 - np.exp(-2 * nodes * dt)) / (2 * nodes * dt)
    OU_1 = np.zeros((2, n + 1))
    for i in range(n):
        OU_1[:, i + 1] = exp_nodes * OU_1[:, i] + div_nodes * B[i]

    exp_nodes = np.exp(-exact_nodes * dt)
    div_nodes = (1 - np.exp(-2 * exact_nodes * dt)) / (2 * exact_nodes * dt)
    OU_2 = np.zeros((6, n + 1))
    for i in range(n):
        OU_2[:, i + 1] = exp_nodes * OU_2[:, i] + div_nodes * B[i]

    exact_fBm = np.einsum('i,ij->j', exact_weights, OU_2)
    approx_fBm = np.einsum('i,ij->j', weights, OU_1)

    time_vec = np.linspace(0, T, n + 1)
    BM = np.zeros(n + 1)
    BM[1:] = np.cumsum(B)
    plt.plot(time_vec, exact_fBm, 'k-', label='Fractional Brownian motion')
    plt.plot(time_vec, approx_fBm, 'g-', label='Markovian approximation')
    plt.plot(time_vec, weights[0] * OU_1[0, :], 'b-', label='Slow component')
    plt.plot(time_vec, weights[-1] * OU_1[-1, :], 'r-', label='Fast component')
    # plt.plot(time_vec, BM, color='grey', label='Underlying Brownian motion')
    plt.legend(loc='best')
    plt.show()


def comp_times_largest_nodes(H=None, N=None, T=1., mode=None):
    H = set_array_default(H, np.array([0.001, 0.01, 0.1]))
    N = set_array_default(N, np.arange(1, 11))
    mode = set_list_default(mode, ["new geometric theorem l1", "non-geometric l1", "optimized l1", "paper",
                                   "optimized l2", "european", "abi jaber", "alfonsi"])

    for i in range(len(H)):
        for j in range(len(mode)):
            for k in range(len(N)):
                tic = time.perf_counter()
                nodes, weights = rk.quadrature_rule(H=H[i], N=N[k], T=T, mode=mode[j])
                print(f'H={H[i]}, mode={mode[j]}, N={N[k]}, time={np.log10(time.perf_counter() - tic)}, '
                      f'largest node={np.log10(np.amax(nodes))}')


def compute_and_save_optimal_l1_rules(H=None, N=10):
    H = set_array_default(H, np.array([0.001, 0.01, 0.1]))
    quadrature_rules = np.zeros((2, len(H), N, N))
    for j in range(len(H)):
        nodes, weights = rk.quadrature_rule(H=H[j], N=1, T=1., mode='optimized l1')
        quadrature_rules[0, j, 0, :1] = nodes
        quadrature_rules[1, j, 0, :1] = weights
        for i in range(1, N):
            print(j, i)
            err, nodes, weights = rk.optimize_error_l1(H=H[j], N=i + 1, T=1., iterative=True,
                                                       init_nodes=quadrature_rules[0, j, i - 1, :i],
                                                       init_weights=quadrature_rules[1, j, i - 1, :i])
            quadrature_rules[0, j, i, :i + 1] = nodes
            quadrature_rules[1, j, i, :i + 1] = weights
    np.save(f'Optimal l1 nodes weights H in {H} N from 1 to {N}.npy', quadrature_rules)


def plot_GG_NGG_and_error_bounds():
    N = 173
    H = 0.1
    T = 1.

    errors_GG = np.empty(N)
    errors_NGG = np.empty(N)
    error_bounds_GG = 2 * (np.sqrt(2) + 1) ** (-2 * np.sqrt((H + 0.5) * np.arange(1, N + 1)))
    error_bounds_NGG = 60 * np.exp(-2.38 * np.sqrt((H + 0.5) * np.arange(1, N + 1)))

    for i in range(N):
        print(f"Computing {i + 1} of {N}.")
        nodes, weights = rk.quadrature_rule(H=H, N=i + 1, T=T, mode='GG')
        errors_GG[i] = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='gaussian')
        nodes, weights = rk.quadrature_rule(H=H, N=i + 1, T=T, mode='NGG')
        errors_NGG[i] = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='gaussian')

    errors_GG = errors_GG / rk.kernel_norm(H=H, T=T, p=1.)
    errors_NGG = errors_NGG / rk.kernel_norm(H=H, T=T, p=1.)

    plt.loglog(np.arange(1, N + 1), errors_GG, label='GG')
    plt.loglog(np.arange(1, N + 1), errors_NGG, label='NGG')
    plt.loglog(np.arange(1, N + 1), error_bounds_GG, label='GG rate')
    plt.loglog(np.arange(1, N + 1), error_bounds_NGG, label='NGG rate')
    plt.xlabel('Number of nodes N')
    plt.ylabel('Relative error')
    plt.title('Relative ' + r'$L^1$' + f'-error for GG and NGG,\nwith H={H} and T={T}, together with convergence rates')
    plt.legend(loc='best')
    plt.show()


def plot_GG_varying_H():
    N = 183
    H = np.array([-0.4, -0.3, -0.2, -0.1, 0., 0.1])
    T = 1.

    errors_GG = np.empty((len(H), N))

    for j in range(len(H)):
        for i in range(N):
            print(f"Computing {i + 1} of {N} for H={H[j]}.")
            nodes, weights = rk.quadrature_rule(H=H[j], N=i + 1, T=T, mode='GG')
            errors_GG[j, i] = rk.error_l1(H=H[j], nodes=nodes, weights=weights, T=T, method='gaussian')
            print(errors_GG[j, i])

    errors_GG = errors_GG / np.array([rk.kernel_norm(H=H[j], T=T, p=1.) for j in range(len(H))])[:, None]

    for j in range(len(H)):
        plt.loglog(np.arange(1, N + 1), errors_GG[j, :], color=color(j, len(H)), label=f'H={H[j]}')
    plt.xlabel('Number of nodes N')
    plt.ylabel('Relative error')
    plt.title('Relative ' + r'$L^1$' + f'-errors for GG')
    plt.legend(loc='best')
    plt.show()


def plot_GG_OL1_varying_H():
    N = 10
    H = np.array([-0.4, -0.3, -0.2, -0.1, 0., 0.1])
    T = 1.

    errors_GG = np.empty((len(H), N))
    errors_OL1 = np.empty((len(H), N))

    for j in range(len(H)):
        for i in range(N):
            print(f"Computing {i + 1} of {N} for H={H[j]}.")
            nodes, weights = rk.quadrature_rule(H=H[j], N=i + 1, T=T, mode='GG')
            errors_GG[j, i] = rk.error_l1(H=H[j], nodes=nodes, weights=weights, T=T, method='gaussian')
            nodes, weights = rk.quadrature_rule(H=H[j], N=i + 1, T=T, mode='OL1')
            errors_OL1[j, i] = rk.error_l1(H=H[j], nodes=nodes, weights=weights, T=T, method='intersections')[0]

    errors_GG = errors_GG / np.array([rk.kernel_norm(H=H[j], T=T, p=1.) for j in range(len(H))])[:, None]
    errors_OL1 = errors_OL1 / np.array([rk.kernel_norm(H=H[j], T=T, p=1.) for j in range(len(H))])[:, None]

    for j in range(len(H)):
        plt.loglog(np.arange(1, N + 1), errors_OL1[j, :], color=color(j, len(H)), label=f'H={H[j]}')
        plt.loglog(np.arange(1, N + 1), errors_GG[j, :], '--', color=color(j, len(H)))
    plt.xlabel('Number of nodes N')
    plt.ylabel('Relative error')
    plt.title('Relative ' + r'$L^1$' + f'-errors for GG and OL1')
    plt.legend(loc='best')
    plt.show()


def generate_and_safe_samples():
    H, T, S_0, V_0, theta, rho, nu, lambda_, r = 0.1, 1., 1., 0.02, 0.02, -0.7, 0.3, 0.3, 0.
    rng, kernel_dict = None, None
    m = 2 ** 22
    N_time = 1024
    n_batches, m_batch = rHestonQESimulation.get_n_batches(return_times=N_time, m=m)  # 220, 19066

    for i in range(25):
        if i >= 1:
            rv_shift = np.random.uniform(0, 1, 3 * N_time)
        else:
            rv_shift = None
        for j in range(n_batches):
            print(f'Batch {i + 1} {j + 1} of 25 {n_batches}')
            filename = f'rHeston HQE sample paths {i + 1} {j + 1}.npy'
            samples, rng, kernel_dict = rHestonQESimulation.samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0,
                                                                    T=T,
                                                                    rho=rho, S_0=S_0, r=r, m=m_batch, N_time=N_time,
                                                                    sample_paths=True, rng=rng, kernel_dict=kernel_dict,
                                                                    rv_shift=rv_shift)
            np.save(filename, samples)
            if i == 0 and j == 0:
                samples_2 = np.load(filename)
                print(np.array_equal(samples, samples_2))
            rng = rng.reset()
            rng = rng.fast_forward((j + 1) * m_batch)


def generate_and_safe_samples_2():
    H, T, S_0, V_0, theta, rho, nu, lambda_, r = 0.1, 1., 1., 0.02, 0.02, -0.7, 0.3, 0.3, 0.
    rng, kernel_dict = None, None
    m = 2 ** 22
    N_time = 1024
    n_batches, m_batch = rHestonQESimulation.get_n_batches(return_times=N_time, m=m)  # 220, 19066

    final_vals = np.empty(m)
    avg_vals = np.empty(m)
    for i in range(5, 25):
        rv_shift = np.random.uniform(0, 1, 3 * N_time)
        rng = None
        for j in range(n_batches):
            print(f'Batch {i + 1} {j + 1} of 25 {n_batches}')
            samples, rng, kernel_dict = rHestonQESimulation.samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0,
                                                                    T=T, rho=rho, S_0=S_0, r=r, m=m_batch,
                                                                    N_time=N_time, sample_paths=True, rng=rng,
                                                                    kernel_dict=kernel_dict, rv_shift=rv_shift)
            if j == n_batches - 1:
                final_vals[j * m_batch:] = samples[0, -1, :m - j * m_batch]
                avg_vals[j * m_batch:] = \
                    np.exp(np.trapz(np.log(samples[0, :, :m - j * m_batch]), dx=1 / N_time, axis=0))
            else:
                final_vals[j * m_batch:(j + 1) * m_batch] = samples[0, -1, :]
                avg_vals[j * m_batch:(j + 1) * m_batch] = \
                    np.exp(np.trapz(np.log(samples[0, :, :]), dx=1 / N_time, axis=0))
            rng = rng.reset()
            rng = rng.fast_forward((j + 1) * m_batch)
        np.save(f'rHeston HQE samples 1024 {i + 1}.npy', final_vals)
        np.save(f'rHeston HQE geom avg samples 1024 {i + 1}.npy', avg_vals)
