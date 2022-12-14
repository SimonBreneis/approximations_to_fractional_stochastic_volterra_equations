import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
# import Data
import ComputationalFinance
import RoughKernel as rk
# import rBergomi
import functions
import rHestonFourier
import rHestonMarkovSamplePaths
# from functions import *
# import rBergomiMarkov
import rHestonMomentMatching
import scipy.stats, scipy.optimize, scipy.integrate


H, T = 0.1, 1.
for N in range(1, 11):
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='paper')
    err_paper = np.sqrt(rk.error_l2(H=H, nodes=nodes, weights=weights, T=T, output='error')) / rk.kernel_norm(H=H, T=T, p=2)
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='optimized l2')
    err_optimized = np.sqrt(rk.error_l2(H=H, nodes=nodes, weights=weights, T=T, output='error')) / rk.kernel_norm(H=H, T=T, p=2)
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='european')
    err_european = np.sqrt(rk.error_l2(H=H, nodes=nodes, weights=weights, T=T, output='error')) / rk.kernel_norm(H=H, T=T, p=2)
    print(N, err_paper, err_optimized, err_european)
for N in 2 ** np.arange(9):
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='paper')
    err_paper = np.sqrt(rk.error_l2(H=H, nodes=nodes, weights=weights, T=T, output='error')) / rk.kernel_norm(H=H, T=T, p=2)
    print(N, err_paper)
time.sleep(36000)


alpha = 1.76274717
L = np.exp(alpha)
a = 1/3
print(17 ** 2)
val = (L-1) / np.sqrt(a * L) * ((4*np.sqrt(L)-1) / (0.18*4*(L+2*np.sqrt(L)))) ** (-1.5) * ((L+2*np.sqrt(L))/(L-1)) ** 2 / (((L+2*np.sqrt(L))/(L-1)) ** 2 - 1)
print(val * 72 / 35, 13/3)
x = np.linspace(0, 1, 1001)
H = np.linspace(0, 0.5, 1001)
plt.plot(H, (a*L) ** (-0.5-H) * (0.18 * 4 * (L + 2 * np.sqrt(L)) / (4 * np.sqrt(L) - 1)) ** (H + 1.5))
plt.show()
plt.plot(H, (0.5-H)/(0.5+H) * a ** (-1/2-H))
plt.show()
plt.plot(x, 4 * (L + 2 * np.sqrt(L) + 1 - x) / (4 * np.sqrt(L) - x))
plt.show()
x = np.linspace(2, 10, 1001)
print(0.5 * np.exp(1 + 1.5 * alpha))
plt.plot(x, (a * L) ** 0.5 / (4 * np.sqrt(np.pi)) * 1/x * np.exp(2 * x * (1 + 1.5 * alpha + np.log(a) - np.log(2 * x))))
plt.plot(x, np.exp(5-x))
# plt.plot(x, 50000 * x - 90000)
# plt.plot(x, -50000 * x + 550000)
plt.yscale('log')
plt.show()

print(scipy.optimize.minimize(lambda x: -x[0], x0=np.array([0.5, 0.5]), constraints={'type': 'eq', 'fun': lambda x: x[0] - np.log((np.exp(x[0] * x[1]) + 2 * np.exp(x[0] * x[1] / 2) + 1) / (np.exp(x[0] * x[1]) - 1)) * 2 * x[1]}))

L = 2.
T = 1.
a = 20.
H = 0.1
m = 3
t_steps = 1000
t = np.linspace(0, T * a, t_steps + 1)
x_steps = 1000
x = np.linspace(1, L, x_steps + 1)
integrals = rk.c_H(H) * np.trapz(rk.exp_underflow(t[:, None] * x[None, :]) * x[None, :] ** (-H - 0.5), dx=(L - 1) / x_steps, axis=-1)
nodes, weights = rk.Gaussian_interval(H=H, m=m, a=1, b=L, mp_output=False)
print(nodes, weights)
eps = 0.3
sums = rk.exp_underflow(nodes[None, :] * t[:, None]) @ weights
plt.plot(integrals - sums)
plt.plot(np.pi ** 2 * rk.c_H(H) / (3 * 2 ** (2 * m) * scipy.special.gamma(2 * m + 1)) * t ** (2 * m) * rk.exp_underflow(t) * (L - 1) ** (2 * m + 1))
plt.plot(32/15 * rk.c_H(H) * (L-1) * ((4*eps*np.sqrt(L)-eps**2)/(4*(L+2*np.sqrt(L)+1-eps))) ** (-H-0.5) * ((L+2*np.sqrt(L)+1-eps)/(L-1)) ** 2 / (((L+2*np.sqrt(L)+1-eps)/(L-1))**2-1) * np.exp(t * (-4*eps*np.sqrt(L)+eps**2)/(4*(L+2*np.sqrt(L)+1-eps))) * ((L+2*np.sqrt(L)+1-eps)/(L-1))**(-2*m))
# plt.plot((integrals - sums) * x ** (0.5 - H))
plt.show()


def ODE_solution(c, beta):
    def vec_field(t, x):
        return np.log(1 + 2 * c * np.exp(x / (2 * beta ** 2)))

    res = scipy.integrate.solve_ivp(fun=vec_field, t_span=(0, 1), y0=np.array([0.]))
    print(c, beta, 2 * beta * np.log(c), - res.y[0, -1] / beta)
    return res.y[0, -1]


def constr(x):
    return -2 * np.log(x[0]) * x[1] ** 2 - ODE_solution(x[0], x[1])

res = scipy.optimize.minimize(lambda x: np.fmax(2 * x[1] * np.log(x[0]), -ODE_solution(x[0], x[1]) / x[1]), x0=np.array([0.5, 0.8]), bounds=((0.01, 1), (0, None)))
#, constraints={'type': 'eq', 'fun': constr})
print(res.x, res.fun, constr(res.x))
print(2 * np.log(res.x[0]) * res.x[1])
print(1.320950 - 0.917609 * 0.5 ** (0.160122))
time.sleep(360000)
H = 0.1
N = 10
T = 1.
tol = 1e-8
m, N_time, T, r, K, S_0 = 10000000, 256, 1., 0.06, 1.05, 1.
H, lambda_, nu, theta, V_0, rel_tol, rho = 0.1, 0.3, 0.3, 0.02, 0.02, 1e-05, -0.7
N_dates = N_time // 2
K = np.exp(np.linspace(-1, 0.5, 151))

N = np.arange(1, 11)
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, H=H)
# a_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, H=0.49)
# print(np.amax(rk.rel_err(true_smile, a_smile)))
for i in range(6):
    nodes, weights = rk.quadrature_rule(H=H, N=i + 1, T=T, mode='european')
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(i + 1, np.amax(rk.rel_err(true_smile, approx_smile)), np.amax(nodes))
    print(nodes, weights)
    nodes, weights = rk.quadrature_rule(H=H, N=i + 1, T=T, mode='optimized')
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(i + 1, np.amax(rk.rel_err(true_smile, approx_smile)), np.amax(nodes))
    nodes, weights = rk.quadrature_rule(H=H, N=i + 1, T=T, mode='theorem l1')
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(i + 1, np.amax(rk.rel_err(true_smile, approx_smile)), np.amax(nodes))
    nodes, weights = rk.quadrature_rule(H=H, N=i + 1, T=T, mode='new theorem l1')
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(i + 1, np.amax(rk.rel_err(true_smile, approx_smile)), np.amax(nodes))
    _, nodes, weights = rk.optimize_error_l1(H=H, N=i + 1, T=T)
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(i + 1, np.amax(rk.rel_err(true_smile, approx_smile)), np.amax(nodes))
    print(nodes, weights)
time.sleep(36000)


time.sleep(36000)

H = np.linspace(0.05, 0.45, 9)
N = np.arange(1, 201)
errors = np.empty((len(H), len(N)))
for i in range(len(H)):
    errors[i, :] = rk.optimize_error_l2(H=H[i], N=200, T=1., iterative=True)[0]
    '''
    for j in range(len(N)):
        nodes, weights = rk.quadrature_rule(H=H[i], N=N[j], T=1., mode='optimized')
        errors[i, j] = np.sqrt(rk.error_optimal_weights(H=H[i], nodes=nodes, T=1.)[0]) / rk.kernel_norm(H=H[i], T=1., l2=True)
        print(H[i], N[j], errors[i, j])'''
print((errors,))
errors = - np.log(errors)
print((errors,))


B = np.empty(len(H))
C = np.empty(len(H))
for i in range(len(H)):
    print(scipy.stats.linregress(np.log(N), np.log(errors[i, :])))
    B[i], C[i], _, _, _ = functions.log_linear_regression(N, errors[i, :])

for j in range(len(N)):
    plt.loglog(H, errors[:, j], color=functions.color(j, len(N)), label=f'N={N[j]}')
plt.legend(loc='best')
plt.xlabel('Hurst parameter H')
plt.ylabel('Negative log-relative error')
plt.show()
for i in range(len(H)):
    plt.loglog(N, (errors[i, :] - 0.5 * np.log(H[i]) + 0.5 * np.log(0.5 - H[i])) / np.sqrt(H[i] * N), color=functions.color(i, len(H)), label=f'H={H[i]:.3}')
    # plt.loglog(N, C[i] * N ** B[i], '--', color=functions.color(i, len(H)))
plt.legend(loc='best')
plt.xlabel('Number of dimensions N')
plt.ylabel('Negative log-relative error')
plt.show()


def fun_B(x):
    a = x[0]
    b = x[1]
    c = x[2]
    summands = a + b * H ** c - B
    return np.sum(summands ** 2), np.array([2 * np.sum(summands), 2 * np.sum(H ** c * summands), 2 * np.sum(b * H ** c * np.log(H) * summands)])


def fun_C(x):
    summands = x * H - C
    return np.sum(summands ** 2), np.array([2 * np.sum(H * summands)])


def fun_C2(x):
    a = x[0]
    b = x[1]
    c = x[2]
    summands = a + b * H ** c - C
    lam = 0.001
    return np.sum(summands ** 2) + lam * (a ** 2 + b ** 2 + c ** 2), np.array([2 * np.sum(summands) + 2 * lam * a, 2 * np.sum(H ** c * summands) + 2 * lam * b, 2 * np.sum(b * H ** c * np.log(H) * summands) + 2 * lam * c])


B = np.empty(len(H))
C = np.empty(len(H))
for i in range(len(H)):
    print(scipy.stats.linregress(np.log(N), np.log(errors[i, :])))
    B[i], C[i], _, _, _ = functions.log_linear_regression(N, errors[i, :])
print(scipy.stats.linregress(np.log(H), np.log(B)))
print(scipy.stats.linregress(H, C))
a, b, _, _, _ = scipy.stats.linregress(H, C)
res = scipy.optimize.minimize(fun_B, x0=np.array([1., 0, 0]), jac=True)
a_B, b_B, c_B = res.x[0], res.x[1], res.x[2]
print(a_B, b_B, c_B)
res = scipy.optimize.minimize(fun_C, x0=np.array([0.]), jac=True)
print(res.x)
plt.plot(H, B, label='B(H)')
plt.xlabel('Hurst parameter H')
plt.legend(loc='best')
plt.show()
plt.plot(H, B, label='B(H)')
plt.plot(H, B, label='Approximation')
plt.xlabel('Hurst parameter H')
plt.legend(loc='best')
plt.show()
plt.plot(H, C, label='C(H)')
plt.xlabel('Hurst parameter H')
plt.legend(loc='best')
plt.show()
plt.plot(H, C, label='C(H)')
# plt.plot(H, a * H + b, label='Approximation')
plt.plot(H, res.x * H, label='Approximation')
plt.xlabel('Hurst parameter H')
plt.legend(loc='best')
plt.show()
plt.loglog(H, B)
plt.xlabel('Hurst parameter H')
plt.ylabel('B(H)')
plt.show()
plt.loglog(H, C)
plt.xlabel('Hurst parameter H')
plt.ylabel('C(H)')
plt.show()

# V_T = V_0 + int_0^T K(T-s) (theta - lambda V_s) ds + int_0^T nu sqrt(V_s) dW_s
nodes, weights = rk.quadrature_rule(H=H, N=3, T=T, mode='european')
print(nodes, weights)
print(np.amax(nodes))
# nodes = np.array([0])
# weights = np.array([1])
# moments = rHestonMomentMatching.first_five_moments_V(nodes=nodes, weights=weights, lambda_=lambda_, theta=theta, nu=nu, V_0=V_0, dt=1/50)[0](V_0)
# print(moments)

true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, H=H)
# a_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, H=0.49)
# print(np.amax(rk.rel_err(true_smile, a_smile)))
for i in range(6):
    nodes, weights = rk.quadrature_rule(H=H, N=i + 1, T=T, mode='european')
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(i + 1, np.amax(rk.rel_err(true_smile, approx_smile)), np.amax(nodes))
    nodes, weights = rk.quadrature_rule(H=H, N=i + 1, T=T, mode='optimized')
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(i + 1, np.amax(rk.rel_err(true_smile, approx_smile)), np.amax(nodes))
    _, nodes, weights = rk.optimize_error_l1(H=H, N=i + 1, T=T)
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(i + 1, np.amax(rk.rel_err(true_smile, approx_smile)), np.amax(nodes))
time.sleep(36000)

real_errors = np.empty(8)
better_errors = np.empty(8)
optimal_errors = np.empty(8)
bound = np.empty(8)
N_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])

for i in range(8):
    N = N_1[i]
    nodes_1, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='theorem l1')
    real_errors[i] = rk.error_l1(H=H, nodes=nodes_1, weights=weights, T=T, method='intersections', tol=tol)[0]
    nodes_2, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='observation l1')
    better_errors[i] = rk.error_l1(H=H, nodes=nodes_2, weights=weights, T=T, method='intersections', tol=tol)[0]
    # optimal_errors[i], nodes_3, weights = rk.optimize_error_l1(H=H, N=N, T=T)
    N_ = (H + 0.5) * N
    bound[i] = rk.c_H(H) * T ** (0.5 + H) * (0.03 / (0.5 - H) * (122 / N_) ** (0.4275 * np.sqrt(N_) - 0.5) + 8 * N_ ** ((1 - H) / 2) + 2) * np.exp(-1.06418 * np.sqrt(N_))
    real_errors[i] = real_errors[i] * scipy.special.gamma(H + 0.5) * (H + 0.5) / T ** (H + 0.5)
    better_errors[i] = better_errors[i] * scipy.special.gamma(H + 0.5) * (H + 0.5) / T ** (H + 0.5)
    bound[i] = bound[i] * scipy.special.gamma(H + 0.5) * (H + 0.5) / T ** (H + 0.5)
    print(N, np.amax(nodes_1), np.amax(nodes_2), bound[i], real_errors[i], better_errors[i], optimal_errors[i])

optimal_errors = np.array([0.16354500061182706, 0.05735152942398583, 0.023932298883270593, 0.011125617233409565,
                           0.0055713939772209006, 0.002948648444794344, 0.0016297948110116681, 0.0009332303490519119])
plt.loglog(N_1, real_errors, label='Theorem')
plt.loglog(N_1, better_errors, label='Improved parameters')
plt.loglog(N_1, optimal_errors, label='Optimal rule')
plt.loglog(N_1, bound, label='Theorem bound')
plt.xlabel('Number of dimensions N')
plt.ylabel('Relative error')
plt.legend(loc='best')
plt.show()


nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='theorem l1')
print(nodes, weights)

nodes, weights = rk.quadrature_rule(H=H, N=N, T=T)
print(nodes, weights)
for N in np.arange(1, 15):
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='theorem l1')
    print('theorem l1', N, np.amax(nodes), rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='intersections', tol=tol)[0])
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T)
    print('european', N, np.amax(nodes), rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='intersections', tol=tol)[0])


time.sleep(36000)
real_error = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='intersections', tol=tol)[0]
print(real_error)
for tol in 10. ** (-np.arange(0, 10)):
    tic = time.perf_counter()
    res = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='intersections', tol=tol)
    print(tol, 'intersections', np.abs(res[0] - real_error) / real_error, res[1], 1000 * (time.perf_counter() - tic))
    tic = time.perf_counter()
    # res = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='exact - trapezoidal', tol=tol)
    print(tol, 'exact - trapezoidal', np.abs(res[0] - real_error) / real_error, res[1], 1000 * (time.perf_counter() - tic))
    if tol > 2e-03:
        tic = time.perf_counter()
        # res = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='trapezoidal', tol=tol)
        print(tol, 'trapezoidal', np.abs(res[0] - real_error) / real_error, res[1], 1000 * (time.perf_counter() - tic))

tic = time.perf_counter()
print(rk.error_l1(H=H, nodes=nodes, weights=weights, T=T, method='intersections', tol=tol))
print(time.perf_counter() - tic)
time.sleep(36000)

N = np.linspace(0.5, 100, 1001)
H = np.linspace(0.0001, 0.5, 101)
beta = 0.4275
alpha = 1.06418
q = 10
r = 0.3
gamma = 0.86
print(np.sqrt(11))
plt.plot(N, np.sqrt(11) * 7 * N ** (-0.0009) * N ** (1 / (4 * beta * np.sqrt(N))))
plt.show()
plt.plot(H, ((0.75 - H/2) * (H + 0.5)) / (2 * beta + 1/2 - H))
plt.show()
x = np.linspace(1, 10, 101)
print(0.173 * 0.75)
plt.plot(x, np.log(x) / np.sqrt(x))
plt.show()
print(-3*beta * (1 - np.exp(-alpha*beta)) / (4 * (1 - np.exp(-alpha*beta) + 2 * gamma * beta ** 2)))
print(2 * 122 ** 0.25)
print(3 / 100 / 121 ** 0.75)
print(np.exp(2*gamma-2))
print((1-np.exp(-alpha*beta) - gamma*beta) / (2*(1-np.exp(-alpha*beta)+2*gamma*beta)))
print(3 * beta * (1 - np.exp(-alpha * beta)) / (4 * (1 - np.exp(-alpha*beta) + beta**2)))
print(3 * beta * (1 - np.exp(-alpha * beta)) / (4 * (1 - np.exp(-alpha*beta) + 2*beta**2)))
plt.plot(N, (N / 122) ** (- (3 * beta * (1 - np.exp(-alpha*beta))) / (4 * (1 - np.exp(-alpha*beta) + 2 * beta ** 2))) + (N / 122) ** (1 / np.sqrt(N)))
plt.show()
plt.plot(N, (N / 122) ** ((2 * r - 1) * beta / 2) * (N / 122) ** (1 / np.sqrt(N)))
plt.show()
plt.plot(N, (N / 122) ** ((4 * q + 3) / (8 * N)))
plt.show()
print(np.sqrt(2/122) * 5/200)
print(5 * 2 ** (-1.5) / 200 / 122 ** 0.25)
print(6 * beta)
print(beta ** 2 * np.exp(alpha * beta) / (np.exp(alpha * beta) - 1))
H = 0.1
plt.plot(N, 0.03 / (0.5 - H) * (122 / ((0.5 + H) * N)) ** (beta * np.sqrt((0.5 + H) * N) - 0.5), label='Small mean-reversion')
plt.plot(N, 8 * ((0.5 + H) * N) ** (0.5 - H/2), label='Intermediate mean-reversion')
plt.plot(N, 2 * np.ones(100), label='Large mean-reversion')
plt.legend(loc='best')
plt.xlabel('N')
plt.ylabel('Coefficient')
plt.show()

plt.plot(H, alpha * np.sqrt(1 / (1 / H + 1/(1.5 - H))), label='Bayer, Breneis 2022')
plt.plot(H, alpha * np.sqrt(1/2 + H), label='New result')
plt.xlabel('H')
plt.ylabel('Rate of convergence')
plt.legend(loc='best')
plt.show()
print(np.sqrt(122))
print(1 / (0.5 + 0) ** 2 * np.sqrt(5 * np.pi ** 3 / 18) * beta ** (-1-0) * 0.5 * (np.exp(alpha * beta) - 1))
plt.plot(H, 1 / (0.5 + H) ** 2 * np.sqrt(5 * np.pi ** 3 / 18) * beta ** (-1-H) * 0.5 * (np.exp(alpha * beta) - 1))
plt.show()
print(1 / (4 * 0.4275 * np.sqrt(122 * np.pi)))
print(np.exp(2 + 1.06418/0.4275) / (4 * 0.4275 ** 2))
print(1 / 3 * N ** (-0.5) * (122 / ((0.5 + H) * N)) ** (0.4275 * np.sqrt((H+0.5) * N)))
print(1 / 3 * N ** (-0.5) * np.exp(2 * 0.4275 * np.sqrt((0.5 + H) * N) * (1 + 1.06418 / (2 * 0.4275) - np.log(2 * 0.4275 * np.sqrt((0.5 + H) * N)))))
print(np.exp(1) * (0.5 + 0.5) ** 0.25 / (4 * np.sqrt(np.pi) * 0.4275 ** 1.5))
print(7/5)
print(np.sqrt(5 * np.pi ** 3 / 18) * (1 / 0.4275) ** (1 + 0.5) * (0.5 * (np.exp(0.4275 * 1.06418) - 1)))
print(1 / (4 * 0.4275 * np.sqrt(np.pi)))

H = np.linspace(0, 0.5, 101)
plt.plot(H, np.sqrt(5 * np.pi ** 3 / 18) * (1 / (0.4275 * np.sqrt(0.5 + H))) ** (1 + H) * (0.5 * (np.exp(0.4275 * 1.06418) - 1)))
plt.plot(H, np.exp(1) * (0.5 + H) ** 0.25 / (4 * np.sqrt(np.pi) * 0.4275 ** 1.5))
plt.show()



def lower_gamma(a, x):
    return scipy.special.gamma(a) * scipy.special.gammainc(a, x)


def upper_bound(a, x):
    return scipy.special.gamma(a) - (((x + 2) ** a - x ** a - 2 ** a)/ (2 * a) + scipy.special.gamma(a)) * np.exp(-x)


for a in np.linspace(2, 20, 10):
    for x in np.exp(np.linspace(0, 20, 11)):
        print(a, x, lower_gamma(a, x), upper_bound(a, x))
time.sleep(36000)

H = np.linspace(0, 0.5, 201)
paper = 1.06418 / (1/H + 1/(1.5-H)) ** 0.5
new_res = 1.06418 / (1/(0.5+H) + 1/(1.5-H)) ** 0.5
new_res_2 = 1.06418 / (1/(0.5+H) + 1/(2.5-H)) ** 0.5
new_res_4 = 1.06418 / (1/(0.5+H) + 1/(4.5-H)) ** 0.5
new_res_3 = 1.06418 / (1/(0.5+H) + 0 * 1/(2.5-H)) ** 0.5
plt.plot(H, paper, label='Bayer, Breneis 2022')
plt.plot(H, new_res, label='New theorem')
plt.plot(H, new_res_2, label='New theorem 2')
plt.plot(H, new_res_3, label='New theorem 3')
plt.plot(H, new_res_4, label='New theorem 4')
plt.legend(loc='best')
plt.show()


for i in range(10):
    print(i + 1)
    print(rk.quadrature_rule(H=0.1, N=i + 1, T=1., mode='optimized'))
    print(rk.quadrature_rule(H=0.1, N=i + 1, T=1., mode='european'))
    print(rk.quadrature_rule(H=0.1, N=i + 1, T=1., mode='integrated'))

m, N_time, T, r, K, S_0 = 10000000, 256, 1., 0.06, 105., 100.
H, lambda_, nu, theta, V_0, rel_tol, rho = 0.1, 0.3, 0.3, 0.02, 0.02, 1e-05, -0.7
N_dates = N_time // 2
'''
eur_price = rHestonFourier.eur_call_put(S_0=S_0, K=np.array([K]), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                        T=T, r=r, rel_tol=rel_tol, H=H, implied_vol=False, call=False)'''

for N_time in 12 * 2 ** np.arange(7):
    for N in range(2, 3):
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T)
        # print(N_time, N, np.amax(nodes))
        '''
        print(rHestonFourier.eur_call_put(S_0=S_0, rel_tol=rel_tol, K=np.array([K]), lambda_=lambda_, rho=rho, nu=nu,
                                          theta=theta, V_0=V_0, r=r, T=T, H=H, call=False,
                                          implied_vol=False))'''
        N_time = 240
        am_price = rHestonMarkovSamplePaths.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                     S_0=S_0, r=r, T=T, nodes=nodes, weights=weights, m=m,
                                                     N_time=N_time, N_dates=48, payoff='put', feature_degree=6)[:4]
        print(N_time, N, np.amax(nodes), am_price)
time.sleep(36000)

params = {'K': K, 'H': H, 'T': T, 'lambda': lambda_, 'nu': nu, 'theta': theta, 'V_0': V_0, 'rel_tol': rel_tol}
'''
compute_rHeston_samples(params=params, Ns=np.array([3]), N_times=2 ** np.arange(9), modes=['european'], m=1000000,
                        sample_paths=True, recompute=True, vol_only=True, euler=False, antithetic=True)

compute_smiles_given_stock_prices(params=params, Ns=np.array([3]), N_times=2 ** np.arange(9), modes=['european'],
                                  option='average volatility call', euler=[False], antithetic=[True])
'''
