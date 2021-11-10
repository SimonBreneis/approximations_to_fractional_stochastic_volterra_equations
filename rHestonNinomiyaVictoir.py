import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp


def get_largest_eigenvalue(A):
    """
    Computes the largest eigenvalue together with the corresponding eigenvector using power iteration.
    """
    min_iter = 10
    eps = mp.mpf(1.) ** (-mp.mp.dps + 5)
    x = mp.randmatrix(A.rows, 1)
    x_old = x / mp.norm(x)
    x_new = x_old.copy()
    i = 1

    while i <= min_iter or mp.norm(x_new-x_old) > eps:
        x_old = x_new.copy()
        x_new = A * x_old
        x_new = x_new / mp.norm(x_new)

    return x_new * (A*x_new)/mp.norm(x_new), x_new


def get_largest_eigenvalues(A, n=2):
    """
    Computes the n=2 or 3 largest eigenvalues and the corresponding eigenvectors of A using power iteration.
    """
    eigvalues = mp.matrix([mp.mpf(0.)]*n)
    eigvectors = mp.matrix(A.rows, n)
    val, vec = get_largest_eigenvalue(A)
    eigvalues[0] = val
    eigvectors[:, 0] = vec
    B = A.copy() - val * vec * vec.T
    val, vec = get_largest_eigenvalue(B)
    eigvalues[1] = val
    eigvectors[:, 1] = (val - eigvalues[0])*vec + eigvalues[0]*(eigvectors[:, 0].T*vec)*eigvectors[:, 0]
    return eigvalues, eigvectors


def SMW(diag, pert):
    """
    Computes the inverse of the matrix diag(diag) + (pert, pert, ..., pert)^T
    """
    diag_inv = mp.matrix([1/x for x in diag])
    X = mp.diag(diag_inv)
    c = pert.T*diag_inv
    M = mp.matrix([[c/x for _ in diag] for x in diag])
    corr = M / (1+c*len(diag))
    return X - corr


def diagonalize(nodes, weights, lambda_):
    """
    Diagonalizes the matrix -diag(nodes) - lambda_ (w,w,...,w)^T in a numerically stable way.
    """
    w = lambda_*weights.copy()
    n = len(nodes)
    eigvalues = mp.matrix([mp.mpf(0.)] * n)
    eigvectors = mp.matrix(n, n)
    A = mp.diag(nodes) + mp.matrix([[w[i] for i in range(n)] for j in range(n)])

    val, vec = get_largest_eigenvalues(A)
    eigvalues[0] = val[0]
    eigvalues[1] = val[1]
    eigvectors[:, 0] = vec[:, 0]
    eigvectors[:, 1] = vec[:, 1]

    A_inv_approx = SMW(nodes + nodes[0], w)
    val, vec = get_largest_eigenvalues(A_inv_approx)
    eigvalues[n-1] = 1/val[0] - nodes[0]
    eigvalues[n-2] = 1/val[1] - nodes[0]
    eigvectors[:, n-1] = vec[:, 0]
    eigvectors[:, n-2] = vec[:, 1]

    avg_geo_dist = (eigvalues[0] / eigvalues[n-1])**(1/(n-1))
    n_found = 4

    while n_found < n:
        found_new = False
        exponent = 1/2
        attempt = 2
        while not found_new:
            perturbation = eigvalues[n_found - 3]/avg_geo_dist**exponent
            A_pert_inv = SMW(nodes - perturbation, w)
            val, vec = get_largest_eigenvalues(A_pert_inv)
            val_1 = 1/val[1] + perturbation
            val_2 = 1/val[0] + perturbation
            if mp.almosteq(val_1, eigvalues[n_found - 4]):
                exponent = exponent + 1/attempt
            elif not mp.almosteq(val_1, eigvalues[n_found-3]) and not mp.almosteq(val_2, eigvalues[n_found-3]):
                exponent = min(exponent - 1/attempt, 2**(-attempt))
            elif mp.almosteq(val_1, eigvalues[n_found-3]):
                eigvalues[n_found-2] = val_2
                eigvectors[:, n_found-2] = vec[0]
                found_new = True
            else:
                eigvalues[n_found-2] = val_1
                eigvectors[:, n_found-2] = vec[1]
                found_new = True
            attempt += 1

    return eigvalues, eigvectors


def exp_matrix(A, dt):
    B = A[1] * mp.diag([mp.exp(A[0][i]*dt) for i in range(len(A[0]))])
    return mp.lu_solve(A[1].T, B.T).T


def ODE_drift(A, b, dt):
    c = mp.lu_solve(A[1], b)
    M = mp.diag([(mp.exp(x*dt)-1)/x if x**2*dt**3/6 > 2*mp.eps else dt*(1-x*dt/2) for x in A[0]])
    c = M*c
    return A[1]*c


def ODE_S_drift(A, b, dt, weights):
    M = mp.diag([((mp.exp(x*dt) - 1)/x - dt)/x if x**2 * dt**4 / 24 > 2*mp.eps else (dt**2/2 + x*dt**3/6) for x in A[0]])
    C = (weights/2).T * A[1]
    D = C * M
    c = mp.lu_solve(A[1], b)
    return (D*c)[0, 0]


def ODE_S_mult(A, dt, weights):
    M = mp.diag([mp.exp(x*dt)/x for x in A[0]])
    C = (weights/2).T * A[1]
    D = C * M
    return mp.lu_solve(A[1].T, D.T)


def get_sample_path(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation"):
    """
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps used in the solution of the fractional Riccati equation
    :param N: Total number of points in the quadrature rule, N=n*m
    :param S_0: Initial stock price
    :param mode: If observation, use the values of the interpolation of the optimum. If theorem, use the values of the
    theorem.
    """
    dt = T/N_time
    sqrt_dt = mp.sqrt(dt)
    log_S = mp.log(S_0)
    rho_bar = mp.sqrt(1-rho*rho)
    rule = rk.quadrature_rule_geometric_good(H, N, T, mode)
    nodes = rule[0, :]
    weights = rule[1, :]
    weight_sum = mp.fsum(weights)
    V = mp.matrix([V_0/(N*weight) for weight in weights])

    A = diagonalize(nodes, weights, lambda_)  # A = [D, U], A = UDU^{-1}
    b = mp.matrix([nodes[i]*V[i] + theta - nu**2*weight_sum/4 for i in range(len(nodes))])*dt/2
    eAt = exp_matrix(A, dt/2)  # eAt = exp(A dt/2)
    Ainv_eAt_id_b = ODE_drift(A, b, dt/2)  # A^{-1} (exp(A dt/2) - Id) b
    S_drift = ODE_S_drift(A, b, dt/2, weights)  # w/2 cdot (A^{-2} exp(A dt/2) b - A^{-1} b dt/2)
    w_Ainv_eAt = ODE_S_mult(A, dt/2, weights)  # w/2 cdot A^{-1} exp(A dt/2)

    def solve_drift_ODE(log_S, V):
        """
        Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        V_final = eAt*V + Ainv_eAt_id_b
        log_S_final = log_S - dt/2 * (nu*rho*weight_sum*dt/8 + w_Ainv_eAt*V + S_drift)
        return V_final, log_S_final

    def solve_stochastic_ODE(log_S, V):
        """
        Solves the ODE corresponding to the stochastic integrals in the Ninomiya-Victoir scheme for one time step.
        Solves both vector fields (corresponding to the two Brownian motions) simultaneously, as this is possible
        in closed form.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        Z_1 = mp.mpf(np.random.normal())
        Z_2 = mp.mpf(np.random.normal())
        total_vol = mp.sqrt(weights*V)
        tau = dt
        if Z_1 < 0:
            temp = - 2*total_vol / (nu*weight_sum*sqrt_dt*Z_1)
            if temp < tau:
                tau = temp

        temp = sqrt_dt*(total_vol*tau + nu*weight_sum*sqrt_dt*Z_1*tau**2/4)
        V_final = V + nu*Z_1*temp
        S_final = log_S + temp*(rho*Z_1 + rho_bar*Z_2)
        return V_final, S_final

    S_values = np.zeros(N_time+1)
    V_values = np.zeros(N_time+1)
    S_values[0] = float(log_S)
    V_values[0] = V_0

    for i in range(1, N_time+1):
        log_S, V = solve_drift_ODE(log_S, V)
        log_S, V = solve_stochastic_ODE(log_S, V)
        log_S, V = solve_drift_ODE(log_S, V)
        S_values[i] = float(log_S)
        V_values[i] = float(weights*V)

    return S_values, V_values


def call(K, lambda_, rho, nu, theta, V_0, nodes, weights, T, N_time=1000):
    """
    Gives the price of a European call option in the approximated, Markovian rough Heston model.
    Uses the Ninomiya-Victoir scheme.
    :param K: Strike prices, assumed to be a vector
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps used in the solution of the fractional Riccati equation
    :param nodes: The nodes used in the approximation
    :param weights: The weights used in the approximation
    return: The prices of the call option for the various strike prices in K
    """
    dt = T / N_time
    times = dt * mp.matrix([N_time - i for i in range(N_time + 1)])
    g = np.zeros(N_time + 1)
    for i in range(len(nodes) - 1):
        for t in range(len(times)):
            g[t] += weights[i] / nodes[i] * (1 - mp.exp(-nodes[i] * times[t]))
    g = theta * (g + weights[len(weights)-1] * times) + V_0

    exp_nodes = mp.matrix([mp.exp(-nodes[j] * dt) for j in range(len(nodes))])
    div_nodes = np.zeros(len(nodes))
    div_nodes[:-1] = np.array([np.longdouble((1 - exp_nodes[j]) / nodes[j]) for j in range(len(nodes) - 1)])
    div_nodes[-1] = np.longdouble(dt)
    exp_nodes = np.array([np.longdouble(x) for x in exp_nodes])
    weights = np.array([np.longdouble(x) for x in weights])

    def mgf_(z):
        """
        Moment generating function of the log-price.
        :param z: Argument, assumed to be a vector
        :return: Value
        """
        res = np.zeros(shape=(len(z)), dtype=complex)
        for i in range(len(z)):
            Fz = solve_Riccati(z=z[i], lambda_=lambda_, rho=rho, nu=nu, N_Riccati=N_time, exp_nodes=exp_nodes,
                               div_nodes=div_nodes, weights=weights)
            res[i] = np.trapz(Fz * g, dx=dt)
        return np.exp(res)

    return cf.pricing_fourier_inversion(mgf_, K, R, L, N_fourier)


def implied_volatility(K, H, lambda_, rho, nu, theta, V_0, T, N, N_time=1000, mode="observation"):
    """
    Gives the implied volatility of the European call option in the rough Heston model. Uses a Markovian approximation,
    and the Ninomiya-Victoir scheme.
    :param K: Strike price, assumed to be a vector
    :param H: Hurst parameter
    :param lambda_: Mean-reversion speed
    :param rho: Correlation between Brownian motions
    :param nu: Volatility of volatility
    :param theta: Mean variance
    :param V_0: Initial variance
    :param T: Final time/Time of maturity
    :param N_time: Number of time steps used in the solution of the fractional Riccati equation
    :param N: Total number of points in the quadrature rule, N=n*m
    :param mode: If observation, use the values of the interpolation of the optimum. If theorem, use the values of the
    theorem.
    return: The price of the call option
    """
    rule = rk.quadrature_rule_geometric_good(H, N, T, mode)
    nodes = rule[0, :]
    weights = rule[1, :]
    prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N_time=N_time,
                  nodes=nodes, weights=weights)
    return cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)