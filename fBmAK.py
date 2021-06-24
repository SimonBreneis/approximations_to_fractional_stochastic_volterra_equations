import mpmath as mp
import numpy as np
import QuadratureRulesRoughKernel


def sqrt_cov_matrix_fBm_approximation_mpmath(H, dt, nodes):
    """
    Computes the Cholesky factorization of the joint covariance matrix of the OU-approximations of the fBm with Hurst
    parameter H on a time-step of dt, and of the fBm itself. Returns an instance of mpmath.matrix.
    :param H: Hurst parameter
    :param dt: Time step size
    :param nodes: The nodes of the approximation (mean-reversion parameters of the OU processes)
    :return: The Cholesky factorization of the covariance matrix, the last row/column is the actual fBm
    """
    n = len(nodes)
    cov_matrix = mp.matrix(n + 1, n + 1)
    for i in range(0, n):
        for j in range(0, n):
            cov_matrix[i, j] = (1 - mp.exp(-dt * (nodes[i] + nodes[j]))) / (nodes[i] + nodes[j])
    for i in range(0, n):
        entry = nodes[i] ** (-H - mp.mpf(0.5)) * mp.gammainc(H + mp.mpf(0.5), 0., dt * nodes[i]) / mp.gammainc(
            H + mp.mpf(0.5), 0.)
        cov_matrix[n, i] = entry
        cov_matrix[i, n] = entry
    cov_matrix[n, n] = dt ** (2. * H) / (2. * H * mp.gamma(H + 0.5) ** 2.)
    return mp.cholesky(cov_matrix)


def sqrt_cov_matrix_fBm_approximation(H, dt, nodes):
    """
    Computes the Cholesky factorization of the joint covariance matrix of the OU-approximations of the fBm with Hurst
    parameter H on a time-step of dt, and of the fBm itself. Computations are done using the mpmath library, but the
    output is a numpy array.
    :param H: Hurst parameter
    :param dt: Time step size
    :param nodes: The nodes of the approximation (mean-reversion parameters of the OU processes)
    :return: The Cholesky factorization of the covariance matrix, the last row/column is the actual fBm
    """
    cov_root_mp = sqrt_cov_matrix_fBm_approximation_mpmath(H, dt, nodes)
    n = len(nodes) + 1
    cov_root = np.empty(shape=(n, n))
    for i in range(0, n):
        for j in range(0, n):
            cov_root[i, j] = cov_root_mp[i, j]
    return cov_root


def fBm_true_and_approximated(H, dt, nodes, weights, samples):
    """
    Simulates jointly samples samples of a fBm (with Hurst parameter H) increment (step size dt) and the OU
    approximation with nodes (mean-reversions) nodes, and weights weights.
    :param H: Hurst parameter
    :param dt: Time step size
    :param nodes: Nodes of the approximation (mean-reversions)
    :param weights: Weights of the approximation
    :param samples: The number of samples simulated
    :return: The final value of the true fBms and the approximated fBms, i.e. [[true], [approximated]]
    """
    n = len(nodes)
    cov_root = sqrt_cov_matrix_fBm_approximation(H, dt, nodes)
    V_approx = np.empty(shape=(samples,))
    V_true = np.empty(shape=(samples,))
    for i in range(0, samples):
        V_OU = cov_root.dot(np.random.normal(0., 1., n + 1))
        V_approx[i] = np.dot(weights, V_OU[0:n])
        V_true[i] = V_OU[n]
    return np.array([V_true, V_approx])


def strong_error_fBm_approximation_MC(H, T, m, n, samples, a=1., b=1.):
    """
    Estimates the L^2 error of the OU-approximation of the final value at time T of a fBm with Hurst parameter H. The
    OU-approximation uses an m-point quadrature rule on n geometrically spaced subintervals. The error is estimated
    using Monte Carlo, where samples samples are used.
    :param H: Hurst parameter
    :param T: Final time
    :param m: Level of the quadrature rule
    :param n: Number of intervals used for the quadrature rule (total number of nodes: m*n)
    :param samples: Number of samples used for the Monte Carlo estimate
    :param a: Can shift left end-point of total interval of the quadrature rule
    :param b: Can shift right end-point of total interval of the quadrature rule
    :return: The estimated error and a 95% confidence interval, i.e. [error, 1.96*std/sqrt(samples)]
    """
    mp.mp.dps = int(np.maximum(20., (m * n) ** 2 / 20.))

    quad_rule = QuadratureRulesRoughKernel.quadrature_rule_geometric_mpmath(H, m, n, a, b)
    nodes = quad_rule[0, :]
    weights = quad_rule[1, :]

    V_true_and_approx = fBm_true_and_approximated(H, T, nodes, weights, samples)
    V_true = V_true_and_approx[0]
    V_approx = V_true_and_approx[1]

    V_errors = np.fabs(V_true - V_approx) ** 2

    error = np.sqrt(np.mean(V_errors))
    stat = 1 / (2. * error) * 1.96 * np.std(V_errors) / np.sqrt(samples)

    return np.array([error, stat])
