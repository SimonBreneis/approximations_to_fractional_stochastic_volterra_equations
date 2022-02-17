import numpy as np
import ComputationalFinance as cf
import RoughKernel as rk
import mpmath as mp


def lu_solve_system(A, B):
    """
    Solves A^{-1}B where B is a matrix (and not just a vector).
    """
    n = A.rows
    result = mp.zeros(n, n)
    for i in range(n):
        result[:, i] = mp.lu_solve(A, B[:, i])
    return result


def diagonalize(nodes, weights, lambda_):
    """
    Diagonalizes the matrix -diag(nodes) - lambda_ (w,w,...,w)^T.
    """
    w = lambda_*weights.copy()
    n = len(nodes)
    A = (-mp.diag(nodes) - mp.matrix([[w[i] for i in range(n)] for _ in range(n)]))
    return mp.eig(A)


def exp_matrix(A, dt):
    n = len(A[0])
    B = A[1] * mp.diag([mp.exp(A[0][i]*dt) for i in range(n)])
    return lu_solve_system(A[1].T, B.T).T


def ODE_drift(A, b, dt):
    c = mp.lu_solve(A[1], b)
    M = mp.diag([(mp.exp(x*dt)-1)/x if x**2*dt**3/6 > 2*mp.eps else dt-x*dt**2/2 for x in A[0]])
    c = M*c
    return A[1]*c


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
    log_S = mp.log(S_0)
    nodes, weights = rk.quadrature_rule_geometric_good(H, N, T, mode)
    N = len(nodes)
    weight_sum = mp.fsum(weights)
    # print(f"weight_sum: {weight_sum}")
    V = mp.matrix([V_0/(N*weight) for weight in weights])

    A = diagonalize(nodes, weights, lambda_)  # A = [D, U], A = UDU^{-1}
    b = mp.matrix([nodes[i]*V[i] + theta for i in range(N)])
    print(f"b: {b}")
    print(f"eigvalues: {A[0]}")
    print(f"eigvectors: {A[1]}")
    print(f"A: ")
    print(-mp.diag(nodes) - lambda_ * mp.matrix([[w for w in weights] for _ in weights]))
    expA = exp_matrix(A, dt/2)  # exp(A*dt/2)
    Ainv_expA_id_b = ODE_drift(A, b, dt/2)  # A^{-1} (exp(A*dt/2) - Id) b
    print(Ainv_expA_id_b)

    def solve_drift_ODE(V):
        """
        Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        # print("Incoming")
        # print(mp.fdot(weights, V))
        # print(V)
        V_final = expA*V + Ainv_expA_id_b
        # print("Outgoing")
        # print(mp.fdot(weights, V_final))
        # print(V_final)
        return V_final

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
        total_vol = mp.fdot(weights, V)
        if total_vol <= mp.mpf(1e-8):
            print(f"Warning, total volatility is small: {total_vol}")
        total_vol_sqrt = mp.sqrt(max(total_vol, mp.mpf(0)))

        S_final = log_S + total_vol_sqrt*mp.sqrt(dt)*(rho*Z_1 + mp.sqrt(1-rho*rho)*Z_2)
        vol_step = nu*total_vol_sqrt*mp.sqrt(dt)*Z_1
        '''
        vol_step_alt = -total_vol/weight_sum
        print(Z_1)
        if vol_step_alt > vol_step:
            print("Needed to adjust Euler step in volatility.")
            vol_step = vol_step_alt
        '''
        V_final = V + vol_step
        #print(f"New total volatility: {mp.fdot(weights, V_final)}")
        return S_final, V_final

    S_values = np.zeros(N_time+1)
    V_values = np.zeros(N_time+1)
    S_values[0] = float(log_S)
    V_values[0] = V_0

    V_components = np.zeros(shape=(len(V), N_time+1))
    V_components[:, 0] = V

    for i in range(1, N_time+1):
        print(i)
        V = solve_drift_ODE(V)
        log_S, V = solve_stochastic_ODE(log_S, V)
        V = solve_drift_ODE(V)
        S_values[i] = float(log_S)
        V_values[i] = float(mp.fdot(weights, V))
        V_components[:, i] = rk.mp_to_np(V)

    return np.exp(S_values), V_values, V_components


def get_samples(H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation", m=1000):
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
    log_S = np.log(S_0)
    nodes, weights = rk.quadrature_rule_geometric_good(H, N, T, mode)
    N = len(nodes)
    weight_sum = mp.fsum(weights)
    V = mp.matrix([V_0/(N*weight) for weight in weights])

    A = diagonalize(nodes, weights, lambda_)  # A = [D, U], A = UDU^{-1}
    b = mp.matrix([nodes[i] * V[i] + theta for i in range(N)])
    eAt = exp_matrix(A, dt / 2)  # exp(A*dt/2)
    Ainv_eAt_id_b = ODE_drift(A, b, dt / 2)  # A^{-1} (exp(A*dt/2) - Id) b

    weight_sum = float(weight_sum)
    V = np.array([[float(v) for _ in range(m)] for v in V])
    weights = rk.mp_to_np(weights)
    eAt = rk.mp_to_np(eAt)
    Ainv_eAt_id_b = rk.mp_to_np(Ainv_eAt_id_b)

    def solve_drift_ODE(V):
        """
        Solves the ODE corresponding to the drift in the Ninomiya-Victoir scheme for one (half) time step.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        V_final = np.einsum('ij,jk', eAt, V) + Ainv_eAt_id_b.repeat(m).reshape(-1, m)
        return V_final

    def solve_stochastic_ODE(log_S, V):
        """
        Solves the ODE corresponding to the stochastic integrals in the Ninomiya-Victoir scheme for one time step.
        Solves both vector fields (corresponding to the two Brownian motions) simultaneously, as this is possible
        in closed form.
        :param log_S: Initial log-stock price
        :param V: Initial variance vector
        return: Final log-stock price, final variance vector, in the form log_S, V.
        """
        Z_1 = np.random.normal(size=m)
        Z_2 = np.random.normal(size=m)
        total_vol = np.sqrt(np.fmax(np.einsum('i,ij', weights, V), 0.))
        S_final = log_S + total_vol*np.sqrt(dt)*(rho*Z_1 + np.sqrt(1-rho*rho)*Z_2)
        vol_step = nu*total_vol*np.sqrt(dt)*Z_1
        V_final = V + vol_step
        return S_final, V_final

    log_S = log_S * np.ones(m)
    for i in range(1, N_time+1):
        print(i)
        V = solve_drift_ODE(V)
        log_S, V = solve_stochastic_ODE(log_S, V)
        V = solve_drift_ODE(V)

    return np.exp(log_S)


def calll(K, H, lambda_, rho, nu, theta, V_0, T, N, S_0=1., N_time=1000, mode="observation", m=1000):
    samples = get_samples(H, lambda_, rho, nu, theta, V_0, T, N, S_0=S_0, N_time=N_time, mode=mode, m=m)
    print("generated samples")
    # print(samples)
    return cf.volatility_smile_call(samples, K, T, S_0)


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
    nodes, weights = rk.quadrature_rule_geometric_good(H, N, T, mode)
    prices = call(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N_time=N_time,
                  nodes=nodes, weights=weights)
    return cf.implied_volatility_call(S=1., K=K, r=0., T=T, price=prices)
