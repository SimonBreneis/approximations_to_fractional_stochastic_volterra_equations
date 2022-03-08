import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp
from scipy.optimize import fsolve


def solve_Riccati_high_mean_reversion(u, nodes, weights, lambda_, nu, dt):
    """
    Solves the Riccati equation for the high mean-reversion components.
    :param u: Complex number, parameter in the Laplace transform
    :param nodes: The high mean-reversion nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param lambda_:
    :param nu:
    :param dt: Vector of step sizes taken in the numerical approximation
    return: The solution psi as a vector of shape (2, len(dt)+1). First component is the real part of psi, second
        component is the imaginary part of psi
    """
    N = len(nodes)
    psis = np.zeros(2*N)
    psi = np.zeros((2, len(dt)+1))
    u_r = u.real
    u_i = u.imag

    time_elapsed = 0.
    for i in range(len(dt)):

        def eq(p):
            p_r = p[:N]
            p_i = p[N:]
            psi[0, i+1] = np.dot(weights, p_r)
            psi[1, i+1] = np.dot(weights, p_i)
            result = p - psis
            temp = np.exp(-nodes*time_elapsed)*(1 - np.exp(-nodes*dt[i]))
            result[:N] += (nodes*p_r + lambda_*psi[0, i+1] - nu**2*(psi[0, i+1]**2 - psi[1, i+1]**2)/2)*dt[i] - u_r*temp
            result[N:] += (nodes*p_i + lambda_*psi[1, i+1] - nu**2*psi[0, i+1]*psi[1, i+1])*dt[i] - u_i*temp
            return result

        def der(p):
            p_r = p[:N]
            p_i = p[N:]
            psi[0, i + 1] = np.dot(weights, p_r)
            psi[1, i + 1] = np.dot(weights, p_i)
            temp = weights*dt[i]
            temp = np.repeat(temp[..., None], N, axis=1)
            result = np.eye(2*N)
            result[:N, :N] += np.diag(nodes*dt[i]) + temp*(lambda_ - nu**2*psi[0, i+1])
            result[N:, N:] += np.diag(nodes*dt[i]) + temp*(lambda_ - 2*nu**2*psi[0, i+1])
            result[N:, :N] += nu**2*psi[1, i+1]*temp
            result[:N, N:] -= 2*nu**2*psi[1, i+1]*temp
            return result

        psis = fsolve(func=eq, x0=psis, fprime=der, col_deriv=True)
        time_elapsed += dt[i]

    return psi


def psi_moments(us, nodes, weights, lambda_, nu, q=10, N_Riccati=10, adaptive=True):
    """
    Computes the first and second moments of psi for varying u.
    :param us: The parameters u for which the psis should be computed (complex numbers!)
    :param nodes: The (high mean-reversion) nodes in the Markovian approximation. Assumed to be ordered in increasing
        order
    :param weights: The (high mean-reversion) weights in the Markovian approximation
    :param lambda_:
    :param nu:
    :param q: Solves the Riccati equation up to time q/min(nodes)
    :param N_Riccati: Uses q*N_Riccati time steps adapted to every node if adaptive, N_Riccati*q equidistant time steps
        if not adaptive
    :param adaptive: If true, adapts the time steps to the mean-reversion parameters (nodes). If false, uses equidistant
        time steps
    return: A complex array of the first and second moments
    """
    T = q / np.amin(nodes)
    N = len(nodes)
    if not adaptive:
        dt = T / (N_Riccati*q) * np.ones(N_Riccati*q)
    else:
        dt = []
        timescale = 0
        for i in range(N):
            prev_timescale = timescale
            timescale = q / nodes[-i - 1]
            dt = dt + [(timescale - prev_timescale) / (q * N_Riccati)] * (q * N_Riccati)
        dt = np.array(dt)
    t = np.zeros(len(dt) + 1)
    t[1:] = np.cumsum(dt)

    integrals = np.empty((2, len(us)), dtype=np.complex_)
    for i in range(len(us)):
        psi = solve_Riccati_high_mean_reversion(us[i], nodes, weights, lambda_, nu, dt)
        real_1 = np.trapz(psi[0, :], x=t)
        imag_1 = np.trapz(psi[1, :], x=t)
        real_2 = np.trapz(psi[0, :] ** 2 - psi[1, :] ** 2, x=t)
        imag_2 = np.trapz(psi[0, :] * psi[1, :], x=t)
        integrals[0, i] = real_1 + imag_1*1j
        integrals[1, i] = real_2 + imag_2*1j
    return integrals


def normal_approximation_of_high_mean_reversion(us, psi_moments, p, theta, lambda_, nu):
    """
    If the moments of psi were already calculated, computes the mean and the standard deviation of the normal
    approximation of the stationary distribution of the high mean-reversion component.
    :param us: The parameters for which the moments of psi were computed. Must be real numbers (i.e. parameters of
        the characteristic function)
    :param psi_moments: Complex numpy array. The element with index (i, j) is the (i+1)st moment (i=0,1) using us[j]
    :param p: The slow mean-reversion component (either float or numpy array of floats)
    :param theta:
    :param lambda_:
    :param nu:
    return: The mean and standard deviation of the normal approximation (either floats or numpy arrays depending on p)
    """
    is_float = False
    if not isinstance(p, np.ndarray):
        p = np.array([p])
        is_float = True
    mus = np.empty(len(p))
    sigmas = np.empty(len(p))
    for i in range(len(p)):
        cf = psi_moments[0, :]*(theta-lambda_*p[i]) + psi_moments[1, :]*nu**2*p[i]/2
        cfr = cf.real
        cfi = cf.imag
        mus[i] = np.sum(us * cfi) / np.sum(us ** 2)
        sigmas[i] = np.sqrt(2*np.abs(np.sum(us ** 2 * cfr) / np.sum(us ** 4)))
    if is_float:
        mus = mus[0]
        sigmas = sigmas[0]
    return mus, sigmas


def normal_approximation_parameter_regression(us, ps, nodes, weights, theta, lambda_, nu, q=10, N_Riccati=10, adaptive=True):
    """
    Approximates the high mean-reversion components by a normal distribution. Estimates the parameters of the normal
    distribution by regression on p.
    :param us: The parameters for which the moments of psi were computed. Must be real numbers (i.e. parameters of
        the characteristic function)
    :param ps: The slow mean-reversion components (numpy array)
    :param nodes: The high mean-reversion nodes of the Markovian approximation
    :param weights: The weights of the Markovian approximation
    :param theta:
    :param lambda_:
    :param nu:
    :param q: Solves the Riccati equation up to time q/min(nodes)
    :param N_Riccati: Uses q*N_Riccati time steps adapted to every node if adaptive, N_Riccati*q equidistant time steps
        if not adaptive
    :param adaptive: If true, adapts the time steps to the mean-reversion parameters (nodes). If false, uses equidistant
        time steps
    return: mu_a, mu_b, sigma_a, sigma_b, where mu = mu_a * p + mu_b and sigma = sqrt(abs(sigma_a*p + sigma_b))
    """
    psis = psi_moments(us*1j, nodes, weights, lambda_, nu, q, N_Riccati, adaptive)
    mus, sigmas = normal_approximation_of_high_mean_reversion(us, psis, ps, theta, lambda_, nu)
    N = len(ps)
    mu_a = (np.sum(mus*ps) - np.sum(mus)*np.sum(ps)/N)/(np.sum(ps**2) - np.sum(ps)**2/N)
    mu_b = (np.sum(mus) - mu_a*np.sum(ps))/N
    sigma_sq = sigmas**2
    sigma_a = (np.sum(sigma_sq*ps) - np.sum(sigma_sq)*np.sum(ps)/N)/(np.sum(ps**2) - np.sum(ps)**2/N)
    sigma_b = (np.sum(sigma_sq) - sigma_a*np.sum(ps))/N
    return mu_a, mu_b, sigma_a, sigma_b


def root_normal_correction(p, mu, sigma, M=1000000):
    samples = np.sqrt(np.fmax(np.repeat((p + mu(p))[..., None], M, axis=1) + np.outer(sigma(p), np.random.normal(0, 1, M)), 0))
    return cf.MC(samples)


def smooth_root(nodes, weights, theta, lambda_, nu, us=np.linspace(-5, 5, 101), ps=np.exp(np.linspace(-10, 1, 200)),
                q=10, N_Riccati=10, adaptive=True, M=1000000):
    mu_a, mu_b, sigma_a, sigma_b = normal_approximation_parameter_regression(us=us, ps=ps, nodes=nodes, weights=weights,
                                                                             theta=theta, lambda_=lambda_, nu=nu, q=q,
                                                                             N_Riccati=N_Riccati, adaptive=adaptive)
    corrected_ps, _ = root_normal_correction(ps, lambda p: mu_a * p + mu_b,
                                             lambda p: np.sqrt(np.fmax(sigma_a * p + sigma_b, 0)), M=M)

    def sr(p):
        if p > ps[-1]:
            return np.sqrt(p) + (sr(ps[-1]) - np.sqrt(ps[-1]))
        elif p < ps[0]:
            return ps[0]
        return np.interp(p, ps, corrected_ps)

    return mu_a, mu_b, sr
