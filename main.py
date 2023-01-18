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
import scipy.stats, scipy.optimize, scipy.integrate, scipy.linalg
import orthopy, quadpy


A = np.array([[-100., 1], [1, -100]])
D, T = np.linalg.eig(A)
y_0 = np.array([1., 0])
solution = T @ np.diag(np.exp(D)) @ np.linalg.inv(T) @ y_0
print(solution)
print(D, T)
print(T @ np.diag(D) @ np.linalg.inv(T))
time.sleep(36000)


print(rHestonMarkovSamplePaths.price_am(euler=True, feature_degree=2, K=105, lambda_=1.22136, rho=-0.9, nu=0.559,
                                        theta=0.03 * 1.22136, V_0=0.02497, S_0=100, T=1., nodes=np.array([0.]),
                                        weights=np.array([1.]), payoff='put', N_time=500, N_dates=100, m=50000, r=0.))
print('Finished')
time.sleep(36000)

Ns = np.arange(1, 10)
errors_geo = np.empty(len(Ns))
errors_non = np.empty(len(Ns))
bounds_geo = np.empty(len(Ns))
bounds_non = np.empty(len(Ns))
errors_opt = np.empty(len(Ns))
opt_nodes = None
opt_weights = None
for i in range(len(Ns)):
    print(i)
    nodes, weights = rk.quadrature_rule(H=0.1, N=Ns[i], T=1., mode="new geometric theorem l1")
    errors_geo[i] = 1 / (scipy.special.gamma(0.1 + 1.5)) - np.sum(weights / nodes * (1 - rk.exp_underflow(nodes)))
    nodes, weights = rk.quadrature_rule(H=0.1, N=Ns[i], T=1., mode="new non-geometric theorem l1")
    errors_non[i] = 1 / (scipy.special.gamma(0.1 + 1.5)) - np.sum(weights / nodes * (1 - rk.exp_underflow(nodes)))
    errors_opt[i], opt_nodes, opt_weights = rk.optimize_error_l1(H=0.1, N=Ns[i], T=1., init_nodes=opt_nodes, init_weights=opt_weights)
bounds_geo = 2 * (np.sqrt(2) + 1) ** (- 2 * np.sqrt(0.6 * Ns))
bounds_non = 60 * np.exp(-2.38 * np.sqrt(0.6 * Ns))
plt.loglog(Ns, errors_geo, label='Geometric errors')
plt.loglog(Ns, errors_non, label='Non-geometric errors')
plt.loglog(Ns, errors_opt, label='Optimal quadrature errors')
plt.legend(loc='best')
plt.xlabel('N')
plt.ylabel('Error')
plt.show()

Ns = np.array([int(np.sqrt(2) ** i) for i in range(14)])
errors = np.empty(len(Ns))
for i in range(len(Ns)):
    nodes, weights = rk.quadrature_rule(H=0.1, N=Ns[i], T=1., mode="new non-geometric theorem l1")
    errors[i] = 1 / (scipy.special.gamma(0.1 + 1.5)) - np.sum(weights / nodes * (1 - rk.exp_underflow(nodes)))
print((errors,))
errors_1_1 = np.array([4.60864759e-01, 4.60864759e-01, 3.40893566e-01, 3.40893566e-01,
       3.28144740e-01, 3.59973336e-01, 6.59228008e-02, 3.55824132e-02,
       2.13324227e-02, 4.43721617e-03, 1.63726548e-03, 4.30894586e-04,
       5.77925537e-05, 9.01113110e-06, 1.24147708e-06, 5.11103804e-08])
errors_2_1 = np.array([3.58468510e-01, 3.58468510e-01, 2.50239258e-01, 2.50239258e-01,
       2.08132959e-01, 2.35728955e-01, 4.60841877e-02, 2.46998733e-02,
       1.51212797e-02, 3.05477336e-03, 1.09153762e-03, 2.76302842e-04,
       3.72850480e-05, 5.85390290e-06, 7.57287183e-07, 3.63682444e-08])
errors_3_1 = np.array([3.30436480e-01, 3.30436480e-01, 2.32770286e-01, 2.32770286e-01,
       1.50956700e-01, 1.74828610e-01, 3.64169874e-02, 1.87643782e-02,
       1.16915995e-02, 2.49319421e-03, 8.90318859e-04, 2.23825884e-04,
       2.99053926e-05, 4.57323737e-06, 5.67975116e-07, 2.91767996e-08])
errors_4_1 = np.array([3.33875654e-01, 3.33875654e-01, 2.45743949e-01, 2.45743949e-01,
       1.19927275e-01, 1.40512199e-01, 3.24382211e-02, 1.51255313e-02,
       9.43598863e-03, 2.12222949e-03, 7.64133485e-04, 1.94306023e-04,
       2.63018567e-05, 3.96673669e-06, 4.78798675e-07, 2.42935481e-08])
errors_4_05 = np.array([3.33875654e-01, 3.33875654e-01, 2.45978892e-01, 2.45978892e-01,
       9.78804728e-02, 1.14165270e-01, 3.07000372e-02, 1.36185831e-02,
       7.80693291e-03, 2.01321775e-03, 6.83698887e-04, 1.69677927e-04,
       2.47162684e-05, 3.60330489e-06, 3.91943483e-07, 2.07377773e-08])
errors_1 = np.array([4.60864759e-01, 4.60864759e-01, 2.84558024e-01, 2.84558024e-01,
       1.32644594e-01, 2.54742658e-01, 5.38771441e-02, 1.91457969e-02,
       1.71078526e-02, 3.75735419e-03, 8.55031572e-04, 2.22031441e-04,
       2.21979806e-05, 2.10863960e-06])
errors_2 = np.array([3.58468510e-01, 3.58468510e-01, 2.20847147e-01, 2.20847147e-01,
       1.34647825e-01, 1.49011977e-01, 3.48726341e-02, 1.64494798e-02,
       9.21328216e-03, 2.80211996e-03, 5.24311340e-04, 1.09897884e-04,
       1.06782030e-05, 9.92150660e-07])
errors_3 = np.array([3.30436480e-01, 3.30436480e-01, 2.22403691e-01, 2.22403691e-01,
       1.59058268e-01, 1.11811697e-01, 3.01913934e-02, 1.65298916e-02,
       6.95753979e-03, 2.54859886e-03, 4.39290160e-04, 8.28010500e-05,
       8.10437004e-06, 7.59055243e-07])
'''
plt.loglog(Ns, errors_1_1[:14], label='1 1')
plt.loglog(Ns, errors_2_1[:14], label='2 1')
plt.loglog(Ns, errors_3_1[:14], label='3 1')
plt.loglog(Ns, errors_4_1[:14], label='4 1')
plt.loglog(Ns, errors_4_05[:14], label='4 05')
'''
plt.loglog(Ns, errors_1, label='1')
plt.loglog(Ns, errors_2, label='2')
plt.loglog(Ns, errors, label='current')
plt.legend(loc='best')
plt.show()

H = np.linspace(0.0001, 0.4999, 200)
plt.plot(H, 1.06418 * (1 / H + 1 / (1.5 - H)) ** (-0.5), label='Bayer, Breneis, 2022, ' + r'$L^2$')
plt.plot(H, 1.06418 * (1 / (0.5 + H) + 1 / (1.5 - H)) ** (-0.5), label='Bayer, Breneis, 2022, ' + r'$L^1$')
plt.plot(H, 2 * np.log(1 + np.sqrt(2)) * np.sqrt(H + 0.5), label='Geometric Gaussian rule')
plt.plot(H, 2.3853845446404978 * np.sqrt(H + 0.5), label='Non-geometric Gaussian rule')
plt.legend(loc='best')
plt.xlabel('H')
plt.ylabel(r'$\alpha(H)$')
plt.show()

ODE_sol = 1.7683671330767412
c = 4.92433436
beta = 0.7447342
print(np.exp(ODE_sol / (2 * beta ** 2)))

alpha = 2.3744943150926243 # 2.375130263185753
L = np.exp(alpha)
M = (L+1)/(L-1)
rho = M + np.sqrt(M**2-1)
rho_1 = (rho + 1/rho) / 2
rho_2 = (rho - 1/rho) / 2
print(L, M, rho, rho_1, rho_2, (np.sqrt(2) + 1) / 2)

H = 0.1
T = 1.
N = np.arange(1, 194)
errors_old = np.empty(len(N))
errors = np.empty(len(N))
bounds = np.empty(len(N))
for i in range(len(N)):
    print(N[i])
    nodes, weights = rk.quadrature_rule(H=H, N=N[i], T=T, mode='new geometric theorem l1')
    errors_old[i] = T ** (H + 1/2) / scipy.special.gamma(H + 3/2) - np.sum(weights / nodes * (1 - rk.exp_underflow(nodes * T)))
    nodes, weights = rk.quadrature_rule(H=H, N=N[i], T=T, mode='new non-geometric theorem l1')
    errors[i] = T ** (H + 1/2) / scipy.special.gamma(H + 3/2) - np.sum(weights / nodes * (1 - rk.exp_underflow(nodes * T)))
    bounds[i] = 100 * rk.exp_underflow(alpha * np.sqrt((H + 1/2) * N[i]))
plt.loglog(N, errors_old)
plt.loglog(N, errors)
plt.loglog(N, bounds)
plt.show()


def ODE_solution(c, beta):
    def vec_field(t, x):
        return 2 * np.log(1 + 2 * np.exp(x / (2 * beta ** 2)) / np.fmax(c - np.exp(x / (2 * beta ** 2)), 1e-10))

    res = scipy.integrate.solve_ivp(fun=vec_field, t_span=(0, 1), y0=np.array([0.]))
    # print(c, beta, 2 * beta * np.log(c), - res.y[0, -1] / beta, res.y[0, -1])
    return res.y[0, -1]


def constr(x):
    return 2 * np.log(x[0]) * x[1] - ODE_solution(x[0], x[1]) / x[1]


def min_fun(x):
    c = x[0]
    beta = x[1]
    ode_sol = ODE_solution(c, beta)
    if not np.exp(ode_sol / (2 * beta ** 2) > c):
        return 100000
    err_1 = - 2 * beta * np.log(c)
    err_2 = - ode_sol / beta
    return np.fmax(err_1, err_2)

for var_1 in np.linspace(1.1, 10, 10):
    for var_2 in np.linspace(0.3, 3., 10):
        res = scipy.optimize.minimize(min_fun, x0=np.array([3.6, 0.9]), bounds=((1.1, 100000), (0, None)))
        #, constraints={'type': 'eq', 'fun': constr})
        print(2 * np.log(res.x[0]) * res.x[1], res.x, res.fun, constr(res.x), var_1, var_2)
time.sleep(360000)

print(
26 + 7 + 1/2 + 4
)
H = np.linspace(0, 0.5, 1001)
summand_1 = (np.exp(2) / (H+1/2)) ** (H+1/2) * (np.sqrt(2)+1) / (1 - (3+2*np.sqrt(2)) ** (-H-1/2)) * 2 / np.pi * (1 + (np.pi / 2) ** (H + 1.5) / (H + 0.5))
summand_2 = 1 / (H + 0.5) * (np.exp(1) / ((10*np.sqrt(2) - 14) * np.sqrt(H + 0.5))) ** (H + 0.5)
plt.plot(H, summand_1)
plt.plot(H, summand_2)
plt.plot(H, summand_1 + summand_2)
plt.show()


factor = rk.c_H(H) * (np.exp(1) * (np.sqrt(2) - 1)) ** (H + 0.5) / ((3 + 2*np.sqrt(2)) ** (H+0.5)-1) * (1 + np.pi ** (H+1.5) / 8 ** (H/2+3/4) / (H+0.5))
summand = rk.c_H(H) * (np.exp(1) * (np.sqrt(2) - 1)) ** (H + 0.5) / ((3 + 2*np.sqrt(2)) ** (H+0.5)-1) * 16/15 * 3 * (3 * (2 + np.sqrt(2))**(H+1.5) - np.sqrt(2) ** (H+1.5) / (H+0.5))
summand_2 = 8 * ((np.sqrt(17) - np.sqrt(2)) / 15) ** (0.5 - H) * 16/15 * 4/np.pi
summand_3 = 4/np.pi * (np.sqrt(2) + 1) * ((H+0.5) / 2) ** (-H-1/2) * (5/3*(H+0.5)**2/np.sqrt(2) + 2/15 + 5/3*(H+0.5)**2/np.sqrt(2)*1/15) * 2**(H-0.5)
middle_part = summand + summand_2 * factor + summand_3*factor
after_frac = ((3 + 2*np.sqrt(2)) * np.exp(1) / (2*np.sqrt(2) - 2)) ** (H + 1/2)
middle_part_after = middle_part * after_frac
high_part = rk.c_H(H) / (0.5+H) * after_frac
low_part = rk.c_H(H) /(np.sqrt(np.pi) * 2 ** (2.5 + H))
ultimator = 4/np.pi * (np.sqrt(2) + 1) * ((H+0.5) / 2) ** (-H-1/2)
ultimator_after = ultimator * after_frac
plt.plot(H, 9.82 - 13.2*H - ultimator * factor / (0.5-H))
plt.plot(H, ((185 + 997 * H) * (0.5-H) - (middle_part_after + high_part + low_part) * np.sqrt(H+0.5) ** (-H-0.5)) / (0.5-H))
plt.plot(H, 0 * H)
plt.show()

H = np.linspace(0, 0.4, 5)
N = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128])
errors = np.empty((len(H), len(N)))
bounds = np.empty((len(H), len(N)))
for i in range(len(H)):
    for j in range(len(N)):
        print(H[i], N[j])
        nodes, weights = rk.quadrature_rule(H=H[i], N=N[j], T=1., mode="new geometric theorem l1")
        alpha = np.log(3 + 2 * np.sqrt(2))  # 1.762747
        errors[i, j] = 1 / scipy.special.gamma(H[i] + 1.5) - np.sum(weights / nodes * (1 - rk.exp_underflow(nodes)))
        bounds[i, j] = (41 * np.sqrt(N[j]) + 206) * np.exp(-alpha * np.sqrt((H[i] + 0.5) * N[j])) / 206 / scipy.special.gamma(H[i] + 1.5)
    plt.loglog(N, errors[i, :], color=functions.color(i, 6))
    plt.loglog(N, bounds[i, :], '--', color=functions.color(i, 6))
plt.show()

H = np.linspace(0, 0.4999, 101)
factor = rk.c_H(H) * 288 / 35 * (1 + np.sqrt(2)) * np.exp(1) * (0.5 + H) ** ((0.5 + H) * (-0.5-H)) / (1.5 + H) ** ((1.5 + H) * (0.5 - H))
fac_1 = (3 + 2 * np.sqrt(2)) ** (0.5 - H) / ((3 + 2 * np.sqrt(2)) ** (0.5 - H) - 1)
fac_2 = (3 + 2 * np.sqrt(2)) ** (0.5 + H) / ((3 + 2 * np.sqrt(2)) ** (0.5 + H) - 1)
plt.plot(H, fac_1 * factor)
plt.plot(H, fac_2 * factor)
plt.plot(H, factor * (fac_1 + fac_2))
plt.plot(H, factor * (2 * fac_1 + 6 * fac_2))
plt.show()

alpha = 1.76274717
rho = 1 + np.sqrt(2)
print((rho - 1/rho) /2)
print((rho + 1/rho) /2)
print(np.exp(alpha))
print(3 + 2 * np.sqrt(2))
print(np.log(3 + 2 * np.sqrt(2)))
gamma = np.linspace(0.0001, 10, 1001)
plt.plot(gamma, ((np.exp(gamma) - 1) / (np.exp(gamma) + 2 * np.exp(gamma/2) + 1)) ** gamma)
plt.show()

alpha = 1.76274717
print(3 / 4 / np.exp(1))
L = np.exp(alpha)
print(L)
M = (L + 1) / (L - 1)
H = np.linspace(0, 0.5, 101)
plt.plot(H, (0.5+H) ** (0.5 + H) / (1.5 + H) ** (1.5 + H))
plt.plot(H, 4 / (4*np.exp(1)) * np.ones(len(H)))
plt.show()
fraction = (0.5 + H) ** (0.5 + H) / (1.5 + H) ** (1.5 + H)
summand_1 = 256 / 15 * 15 / 64 * 144 / 35 * np.exp(1) / np.sqrt(M ** 2 - 1) * rk.c_H(H) * fraction ** (0.5 - H) / (1 - np.exp(-alpha * (0.5 - H))) * 4 / 3 * (1.53 / (0.5 + H)) ** (0.5 + H)
summand_2 = 256 / 15 * 15 / 64 * 144 / 35 * np.exp(1) / np.sqrt(M ** 2 - 1) * rk.c_H(H) * fraction ** (-0.5 -H) / (1 - np.exp(-alpha * (0.5 + H))) * 1.28 * (1.25 / (1.5 + H)) ** (1.5 + H)
plt.plot(H, summand_1)
plt.plot(H, summand_2)
plt.plot(H, summand_1 + summand_2)
plt.show()

alpha = 1.76274717
L = np.exp(alpha)
print(L)
M = (L + 1) / (L - 1)
print(1 / (M - 1), 2 * M /np.sqrt(M**2-1))
mu = np.linspace(3, 100, 1001)
delta = (np.sqrt(mu ** 2 * (M ** - 1) + 1) - M) / (mu ** 2 - 1)
rho = M - delta + np.sqrt((M - delta) ** 2 - 1)
plt.plot(mu, rho ** 2 / (rho ** 2 - 1))
plt.show()
plt.plot(mu, np.sqrt(M**2 - 1) * (mu**2-1)/(mu*(np.sqrt(mu**2*(M**2-1)+1)-M)))
plt.show()


rho = (L + 2*np.sqrt(L) + 1) / (L-1)
print(1/2 * (rho - 1/rho))
time.sleep(360000)

H = np.linspace(0, 0.5, 101)
maxima = np.empty(len(H))
for i in range(len(H)):
    print(i)
    res = scipy.optimize.minimize(lambda x: - (1 - np.exp(-x)) * x ** (- H[i] - 0.5), x0=np.array([1.]), bounds=((0.00001, np.inf),))
    maxima[i] = - res.fun
plt.plot(H, maxima)
plt.plot(H, np.fmax(2/3, 0.5 + H))
plt.show()

t = np.linspace(0.01, 100, 10000)
integral = (1 - np.exp(-2 * t)) / t
plt.loglog(integral, label='integral')
for m in range(1, 11):
    print((1 - (-1) ** (1 + np.arange(2 * m))) / (1 + np.arange(2 * m)))
    alpha, beta, int_1 = orthopy.tools.chebyshev(moments=(1 - (-1) ** (1 + np.arange(2 * m))) / (1 + np.arange(2 * m)))
    nodes, weights = quadpy.tools.scheme_from_rc(alpha, beta, int_1)
    approx_integral = np.sum(weights[None, :] * np.exp(- t[:, None] * (nodes[None, :] + 1)), axis=-1)
    plt.loglog(integral - approx_integral, label=f'{m}')
plt.legend(loc='best')
plt.show()

print(100 * np.sqrt(1/500), np.exp(-0.02 / 500))
# time.sleep(36000)
print(100 * np.exp(0.02 * 1 / 500))

print(rHestonMarkovSamplePaths.price_am(euler=True, feature_degree=2, K=105, lambda_=1.22136, rho=-0.9, nu=0.559, theta=0.03 / 1.22136, V_0=0.02497, S_0=100, T=1., nodes=np.array([0.]), weights=np.array([1.]), payoff='put', N_time=500, N_dates=100, m=50000))
print('Finished')
time.sleep(36000)

alpha = 1.76274717
L = np.exp(alpha)
a = 1/3
print(17 ** 2)
val = np.exp(0.18 * np.sqrt(0.5)) / 0.5 ** 0.75 * 0.5 * 32 / 15 * (L-1) / np.sqrt(a * L) * ((4*np.sqrt(L)-1) / (4*(L+2*np.sqrt(L)))) ** (-1.5) * ((L+2*np.sqrt(L))/(L-1)) ** 2 / (((L+2*np.sqrt(L))/(L-1)) ** 2 - 1)
print(val, 64 * 15 * 144 / 35)
N = np.arange(1, 31)
plt.plot(N, np.exp(0.18 * np.sqrt(0.5 * N)) / (0.5 * N) ** 0.75)
plt.show()
x = np.linspace(0, 1, 1001)
H = np.linspace(0, 0.5, 1001)
plt.plot(H, (0.5 - H) * (a*L) ** (-0.5-H) * (0.18 * 4 * (L + 2 * np.sqrt(L)) / (4 * np.sqrt(L) - 1) / (H + 1/2)) ** (H + 1.5) * np.exp(H + 0.5))
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


H, T = 0.1, 1.
Ns = np.arange(1, 91)
errors_thm = np.empty(len(Ns))
errors_bound = np.empty(len(Ns))
errors_opt = np.empty(len(Ns))
print(rk.kernel_norm(H=H, T=T, p=1))
for i in range(len(Ns)):
    nodes, weights = rk.quadrature_rule(H=H, N=Ns[i], T=T, mode='new geometric theorem l1')
    errors_thm[i] = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T)[0] / rk.kernel_norm(H=H, T=T, p=1)
    nodes, weights = rk.quadrature_rule(H=H, N=Ns[i], T=T, mode='old geometric observation l1')
    errors_opt[i] = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T)[0] / rk.kernel_norm(H=H, T=T, p=1)
    N_ = np.sqrt((H+0.5)*Ns[i])
    errors_bound[i] = rk.c_H(H) * T ** (H+0.5) / (0.5-H) * (19 * N_ ** (H + 3/2) if Ns[i] >= 31 else (np.exp(5 - N_) + 100 * N_ ** (H + 3/2))) * np.exp(-alpha * N_) / rk.kernel_norm(H=H, T=T, p=1)
    print(Ns[i], errors_thm[i], errors_opt[i], errors_bound[i])
plt.loglog(Ns, errors_thm, label='Theorem')
plt.loglog(Ns, errors_bound, label='Theorem bound')
plt.loglog(Ns, errors_opt, label='Optimized parameters')
plt.legend(loc='upper right')
plt.xlabel('N')
plt.ylabel('Relative error')
plt.show()


for N in 2 ** np.arange(9):
    for mode in ['old geometric theorem l1', 'old geometric observation l1', 'old non-geometric theorem l1',
                 'new geometric theorem l1']:
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode=mode)
        print(N, len(nodes), mode, np.amax(nodes),
              rk.error_l1(H=H, nodes=nodes, weights=weights, T=T)[0] / rk.kernel_norm(H=H, T=T, p=1))
time.sleep(360000)


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
