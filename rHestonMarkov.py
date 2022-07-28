import numpy as np
import RoughKernel as rk
import rHestonBackbone as backbone


def characteristic_function(z, lambda_, rho, nu, theta, V_0, T, N_Riccati, nodes, weights):
    """
    Gives the characteristic function of the log-price in the Markovian approximation of the rough Heston model.
    :param z: Argument of the characteristic function (assumed to be a vector)
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time
    :param N_Riccati: Number of time steps used for solving the fractional Riccati equation
    :param nodes: Nodes of the Markovian approximation
    :param weights: Weights of the Markovian approximation
    :return: The characteristic function
    """

    nodes, weights = rk.sort(nodes, weights)
    n_zero_nodes = np.sum(nodes < 1e-08)
    dt = T/N_Riccati
    N = len(nodes)
    times = np.linspace(T, 0, N_Riccati + 1)
    g = np.sum(weights[:n_zero_nodes]) * times
    for j in range(n_zero_nodes, N):
        exponent = nodes[j] * times
        factor = np.ones(shape=exponent.shape)
        factor[exponent < 300] = 1-np.exp(-exponent[exponent < 300])
        g = g + weights[j]/nodes[j] * factor
    g = theta * g + V_0

    a = nu * nu / 2
    b = rho * nu * np.complex(0, 1) * z - lambda_
    c = -(z + np.complex(0, 1)) * z / 2
    exp_nodes = np.zeros(N)
    temp = nodes * dt
    exp_nodes[temp < 300] = np.exp(-temp[temp < 300])
    div_nodes = np.zeros(N)
    div_nodes[:n_zero_nodes] = dt
    div_nodes[n_zero_nodes:] = (1 - exp_nodes[n_zero_nodes:]) / nodes[n_zero_nodes:]
    new_div = np.dot(div_nodes, weights)
    new_exp = exp_nodes * weights

    def F(x):
        return (a * x + b) * x + c

    psi_x = np.zeros((len(z), N), dtype=np.cdouble)
    F_psi = np.zeros((len(z), N_Riccati + 1), dtype=np.cdouble)
    F_psi[:, 0] = c
    psi = np.zeros(len(z), dtype=np.cdouble)

    for i in range(N_Riccati):
        psi_P = (psi + F_psi[:, i] * new_div + psi_x @ new_exp) / 2
        psi_x = np.outer(F(psi_P), div_nodes) + psi_x * exp_nodes[None, :]
        psi = psi_x @ weights
        F_psi[:, i + 1] = F(psi)

    return np.exp(np.trapz(F_psi * g, dx=dt))


def iv_eur_call(S, K, H, lambda_, rho, nu, theta, V_0, T, N, mode="european", rel_tol=1e-03, nodes=None,
                weights=None):
    """
    Gives the implied volatility of the European call option in the rough Heston model
    as described in El Euch and Rosenbaum, The characteristic function of rough Heston models. Uses the Adams scheme.
    Uses Fourier inversion.
    :param S: Initial stock price
    :param K: Strike price, assumed to be a vector
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Maturity
    :param N: Total number of points in the quadrature rule
    :param mode: If observation, uses the parameters from the interpolated numerical optimum. If theorem, uses the
        parameters from the theorem. If optimized, optimizes over the nodes and weights directly. If best, chooses any
        of these three options that seems most suitable
    :param rel_tol: Required maximal relative error in the implied volatility
    :param nodes: Can specify the nodes directly
    :param weights: Can specify the weights directly
    return: The price of the call option
    """
    if mode == 'euro new':
        nodes, weights = rk.european_rule(H, N, T, optimal_weights=True)
        print(nodes, weights)
    elif nodes is None or weights is None:
        nodes, weights = rk.quadrature_rule(H, N, T, mode)
    print(np.amax(nodes))
    print(np.sqrt(rk.error(H, nodes, weights, T))/rk.kernel_norm(H, T))
    return backbone.iv_eur_call(char_fun=lambda u, T_, N_: characteristic_function(u, lambda_, rho, nu, theta, V_0, T_,
                                                                                   N_, nodes, weights),
                                S=S, K=K, T=T, rel_tol=rel_tol)
