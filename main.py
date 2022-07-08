c = ['r', 'C1', 'y', 'g', 'b', 'purple']

import time
import numpy as np
import matplotlib.pyplot as plt
import ComputationalFinance as cf
import Data
import rBergomi
import rBergomiMarkov
import rHeston
import rHestonMarkov
import RoughKernel as rk
import rHestonNinomiyaVictoir as nv
import rHestonNV2nd as nv2
import mpmath as mp
import rHestonImplicitEuler as ie
import rHestonSplitKernel as sk
import scipy
from scipy import special


def interval_error_fun(x, eps=0):
    nodes = np.exp(x[:len(x)//2])
    weights = np.exp(x[len(x)//2:])
    err_1 = rk.error_estimate_fBm_general(0.1, nodes, weights, 1, True)
    if eps > 0:
        err_2 = rk.error_estimate_fBm_general(0.1, nodes, weights, eps, True)
        return err_1 - err_2
    return err_1


nodes, weights = rk.quadrature_rule_geometric_standard(0.1, 1, 1, 'optimized', True, False)
rule = np.empty(2*len(nodes))
rule[:len(nodes)] = nodes
rule[len(nodes):] = weights

k_vec = np.linspace(-1.5, 0.75, 451)
with open('true surface.npy', 'rb') as f:
    true_surface = np.load(f)
true_surface = true_surface[-1, :]

bound_vec = np.exp(np.linspace(np.log(1), np.log(1000000), 200))
largest_nodes = np.empty(len(bound_vec))
kernel_errors = np.empty(len(bound_vec))
smile_errors = np.empty(len(bound_vec))
N_vec = np.array([1, 2, 3, 4, 5, 6])
for j in range(len(N_vec)):
    for i in range(len(bound_vec)):
        res = scipy.optimize.minimize(lambda x: interval_error_fun(x, 0), x0=np.log(np.fmin(rule, bound_vec[i])), bounds=((None, np.log(bound_vec[i])), (None, None),))
        x = res.x
        nodes = np.exp(x[:len(x)//2])
        weights = np.exp(x[len(x)//2:])
        perm = np.argsort(nodes)
        nodes = nodes[perm]
        weights = weights[perm]
        while len(nodes) < N_vec[j]:
            if len(nodes) == 1:
                if bound_vec[i] > 10 * nodes[0]:
                    nodes = np.array([nodes[0], 10*nodes[0]])
                    weights = np.array([weights[0], weights[0]])
                elif bound_vec[i] >= 2*nodes[0]:
                    nodes = np.array([nodes[0], bound_vec[i]])
                    weights = np.array([weights[0], weights[0]])
                else:
                    nodes = np.array([nodes[0]/3, nodes[0]])
                    weights = np.array([weights[0]/3, weights[0]])
            else:
                if bound_vec[i] > 10 * nodes[-1]:
                    nodes = np.append(nodes, np.array([10*nodes[-1]]))
                    weights = np.append(weights, np.array([weights[-1]]))
                elif bound_vec[i] > 2 * nodes[-1] or bound_vec[i]/nodes[-1] > nodes[-1]/nodes[-2]:
                    nodes = np.append(nodes, np.array([bound_vec[i]]))
                    weights = np.append(weights, np.array([weights[-1]]))
                else:
                    nodes = np.append(nodes, np.array([np.sqrt(nodes[-1] * nodes[-2])]))
                    weights = np.append(weights, np.array([np.fmin(weights[-1], weights[-2])/2]))
                    perm = np.argsort(nodes)
                    nodes = nodes[perm]
                    weights = weights[perm]

            rule_ = np.empty(2*len(nodes))
            rule_[:len(nodes)] = nodes
            rule_[len(nodes):] = weights
            res = scipy.optimize.minimize(lambda x: interval_error_fun(x, 0), x0=np.log(rule_),
                                          bounds=((np.log(1e-08), np.log(bound_vec[i])),) * (len(rule_)//2) + ((np.log(1e-08), np.log(1e+08)),) * (len(
                                              rule_)//2))
            x = res.x
            nodes = np.exp(x[:len(x) // 2])
            weights = np.exp(x[len(x) // 2:])
            perm = np.argsort(nodes)
            nodes = nodes[perm]
            weights = weights[perm]
            print(nodes, weights)
        largest_nodes[i] = np.amax(nodes)
        kernel_errors[i] = np.sqrt(rk.error_estimate_fBm_general(0.1, nodes, weights, 1, True) / rk.error_estimate_fBm_general(0.1, np.array([1]),
                                                                                                np.array([0]), 1, True))
        smile = rHestonMarkov.implied_volatility(K=np.exp(k_vec), H=0.1, lambda_=0.3,
                                                 rho=-0.7, nu=0.3, theta=0.02,
                                                 V_0=0.02, T=1, rel_tol=1e-04,
                                                 nodes=nodes, weights=weights, N=-1)
        smile_errors[i] = np.amax(np.abs(true_surface - smile) / true_surface)
        print(N_vec[j], i, bound_vec[i], largest_nodes[i], kernel_errors[i], smile_errors[i])

    print((largest_nodes,))
    print((kernel_errors,))
    print((smile_errors,))

plt.loglog(bound_vec, kernel_errors, label='Kernel errors')
plt.loglog(bound_vec, smile_errors, label='Smile errors')
plt.legend(loc='best')
plt.title('Relative errors depending on cutoff for maturity T=1 and 1 dimension')
plt.show()


time.sleep(36000)

nodes = np.array([  0.70243884 , 20.38828784, 457.20375887])
weights = np.array([1.29031826, 3.55651535, 7.41731548])

print(np.sqrt(rk.error_estimate_fBm_general(0.1, nodes, weights, 1, True))/np.sqrt(rk.error_estimate_fBm_general(0.1, np.array([0.1]), np.array([0]), 1, True)))
time.sleep(36000)

print(rk.error_estimate_fBm_general(0.1, np.array([  1.62476626, 359.99601421]), np.array([ 2.13513308, 16.81331999]), 1, True))
print(rk.error_estimate_fBm_general(0.1, np.array([0.85282142 , 28.9285625]), np.array([1.40727002, 4.79326392]), 1, True))
a = np.array([[0, 1], [2, 3]])
b = np.array([[0, 3], [7, 9]])
frac = np.abs(b-a)[b != 0]/b[b != 0]
print(frac)
print(np.average(frac))

np.printoptions(threshold=np.inf)


def error_func(H, nodes, weights, T):
    coefficient = (2 * H * scipy.special.gamma(H + 0.5) ** 2) * T ** (-2 * H)
    return np.sqrt(np.amax(coefficient * rk.error_estimate_fBm_general(H, nodes, weights, T, True)))


T = np.linspace(0.04, 1., 25)
# norm = rk.error_estimate_fBm_general(0.1, np.array([0.0001]), np.array([0]), 1, fast=True)

k_vec = np.linspace(-1.5, 0.75, 451)
true_surface = np.empty((len(T), len(k_vec)))

'''
tic = time.perf_counter()
for i in range(len(T)):
    print(f'Time {T[i]}')
    indices = slice(int((1-np.sqrt(T[i])) * 300), -int((1-np.sqrt(T[i])) * 150))
    if i == len(T)-1:
        indices = slice(0, len(k_vec))
    k_loc = k_vec[indices]
    res_loc = rHeston.implied_volatility(K=np.exp(k_loc), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02,
                                                    V_0=0.02, T=T[i], rel_tol=5e-04)
    res_loc_ = np.zeros(len(k_vec))
    res_loc_[indices] = res_loc
    true_surface[i, :] = res_loc_
print(f'Finished rough Heston, time: {time.perf_counter()-tic}')
with open('true surface.npy', 'wb') as f:
    np.save(f, true_surface)
'''
with open('true surface.npy', 'rb') as f:
    true_surface = np.load(f)

T = 1
true_surface = true_surface[-1, :]

def err_func(x):
    nodes = np.exp(x[:len(x)//2])
    weights = x[len(x)//2:]
    # smile = np.zeros((len(T), len(k_vec)))
    '''
    errors = np.zeros(len(T))
    for i in range(len(T)):
        indices = slice(int((1-np.sqrt(T[i])) * 300), -int((1-np.sqrt(T[i])) * 150))
        if i == len(T)-1:
            indices = slice(0, len(k_vec))
        k_loc = k_vec[indices]
        res_loc = rHestonMarkov.implied_volatility(K=np.exp(k_loc), H=0.1, lambda_=0.3,
                                                   rho=-0.7, nu=0.3, theta=0.02,
                                                   V_0=0.02, T=T[i], rel_tol=2e-04,
                                                   nodes=nodes, weights=weights, N=-1)
        res_loc_ = np.zeros(len(k_vec))
        res_loc_[indices] = res_loc
        smile[i, :] = res_loc_
        errors[i] = np.amax(np.abs(true_surface[i, indices] - smile[i, indices]) / true_surface[i, indices])
    '''
    smile = rHestonMarkov.implied_volatility(K=np.exp(k_vec), H=0.1, lambda_=0.3,
                                               rho=-0.7, nu=0.3, theta=0.02,
                                               V_0=0.02, T=1, rel_tol=1e-05,
                                               nodes=nodes, weights=weights, N=-1)
    error = 100 * np.amax(np.abs(true_surface - smile) / true_surface)
    print(f'nodes: {nodes}')
    print(f'weights: {weights}')
    print(f'error: {error}')
    return error

import numpy as np
from scipy.optimize import minimize, rosen
import time
import warnings

class TookTooLong(Warning):
    pass

class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
        self.count = 0

    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            # you might want to report other stuff here
            self.count += 1
            print("Elapsed: %.3f sec" % elapsed)
            print(self.count)

# example usage
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead', callback=MinimizeStopper(1E-3))
print('still running!')
print(res.fun)
print(res.x)

N_vec = np.array([3, 4, 5, 6, 7, 8, 9, 10, 16, 32])
for i in range(len(N_vec)):
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule_geometric_standard(0.1, N_vec[i], T, 'optimized', fast=True, grad=False)
    rule = np.empty(2*len(nodes))
    nodes = np.array([0.666, 22.3, 280])
    weights = np.array([1.27, 3.63, 7])
    rule[:len(nodes)] = np.log(nodes)
    rule[len(nodes):] = weights
    res = minimize(err_func, rule, method='Nelder-Mead', callback=MinimizeStopper(7200))
    print(f'Finished optimization for N={N_vec[i]}')
    print(f'original nodes: {nodes}')
    print(f'original weights: {weights}')
    rule = res.x
    nodes = rule[:len(nodes)]
    weights = rule[len(nodes):]
    print(f'new nodes: {nodes}')
    print(f'new weights: {weights}')
    print(f'new error: {res.fun}')
    print(f'time: {time.perf_counter()-tic}')

N_vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32])

approx_surface_obs_02 = np.empty((len(N_vec)-1, len(T), len(k_vec)))
approx_surface_opt_02 = np.empty((len(N_vec), len(T), len(k_vec)))
approx_surface_opt_all = np.empty((len(N_vec), len(T), len(k_vec)))
time_obs_02 = 0
time_opt_02 = 0
time_opt_all = 0
max_errors_obs_02 = np.empty((len(N_vec)-1, len(T)))
max_errors_opt_02 = np.empty((len(N_vec), len(T)))
max_errors_opt_all = np.empty((len(N_vec), len(T)))
avg_errors_obs_02 = np.empty((len(N_vec)-1, len(T)))
avg_errors_opt_02 = np.empty((len(N_vec), len(T)))
avg_errors_opt_all = np.empty((len(N_vec), len(T)))
max_max_errors_obs_02 = np.empty(len(N_vec)-1)
max_max_errors_opt_02 = np.empty(len(N_vec))
max_max_errors_opt_all = np.empty(len(N_vec))
avg_avg_errors_obs_02 = np.empty(len(N_vec)-1)
avg_avg_errors_opt_02 = np.empty(len(N_vec))
avg_avg_errors_opt_all = np.empty(len(N_vec))

for i in range(len(N_vec)):
    if i != 0:
        tic = time.perf_counter()
        for j in range(len(T)):
            print(N_vec[i], T[j])
            indices = slice(int((1 - np.sqrt(T[j])) * 300), -int((1 - np.sqrt(T[j])) * 150))
            if j == len(T)-1:
                indices = slice(0, len(k_vec))
            k_loc = k_vec[indices]
            nodes, weights = rk.quadrature_rule_geometric_standard(0.1, N_vec[i], 0.2, 'observation')
            res_loc = rHestonMarkov.implied_volatility(K=np.exp(k_loc), H=0.1, lambda_=0.3,
                                                                                rho=-0.7, nu=0.3, theta=0.02,
                                                                                V_0=0.02, T=T[j], rel_tol=5e-04,
                                                                                nodes=nodes, weights=weights, N=-1)
            res_loc_ = np.zeros(len(k_vec))
            res_loc_[indices] = res_loc
            approx_surface_obs_02[i-1, j, :] = res_loc_
            max_errors_obs_02[i - 1, j] = np.amax(
                np.abs(true_surface[j, indices] - approx_surface_obs_02[i - 1, j, indices]) / true_surface[j, indices])
            avg_errors_obs_02[i - 1, j] = np.average(
                np.abs(true_surface[j, indices] - approx_surface_obs_02[i - 1, j, indices]) / true_surface[j, indices])
        time_obs_02 += time.perf_counter() - tic
        print(f'current time: {time_obs_02}')
        print(max_errors_obs_02[i - 1, :])
        print(avg_errors_obs_02[i - 1, :])
        max_max_errors_obs_02[i - 1] = np.amax(max_errors_obs_02[i - 1, :])
        print(max_max_errors_obs_02[i - 1])
        avg_avg_errors_obs_02[i - 1] = np.average(avg_errors_obs_02[i - 1, :])
        print(avg_avg_errors_obs_02[i - 1])


    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule_geometric_standard(0.1, N_vec[i], 0.2, 'optimized', fast=True, grad=True)
    for j in range(len(T)):
        print(N_vec[i], T[j])
        indices = slice(int((1-np.sqrt(T[j])) * 300), -int((1-np.sqrt(T[j])) * 150))
        if j == len(T)-1:
            indices = slice(0, len(k_vec))
        k_loc = k_vec[indices]
        res_loc = rHestonMarkov.implied_volatility(K=np.exp(k_loc), H=0.1, lambda_=0.3,
                                                                          rho=-0.7, nu=0.3, theta=0.02,
                                                                          V_0=0.02, T=T[j], rel_tol=5e-04,
                                                                          nodes=nodes, weights=weights, N=-1)
        res_loc_ = np.zeros(len(k_vec))
        res_loc_[indices] = res_loc
        approx_surface_opt_02[i, j, :] = res_loc_
        max_errors_opt_02[i, j] = np.amax(np.abs(true_surface[j, indices] - approx_surface_opt_02[i, j, indices]) / true_surface[j, indices])
        avg_errors_opt_02[i, j] = np.average(np.abs(true_surface[j, indices] - approx_surface_opt_02[i, j, indices]) / true_surface[j, indices])
    time_opt_02 += time.perf_counter() - tic
    print(f'current time: {time_opt_02}')
    print(max_errors_opt_02[i, :])
    print(avg_errors_opt_02[i, :])
    max_max_errors_opt_02[i] = np.amax(max_errors_opt_02[i, :])
    print(max_max_errors_opt_02[i])
    avg_avg_errors_opt_02[i] = np.average(avg_errors_opt_02[i, :])
    print(avg_avg_errors_opt_02[i])

    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule_geometric_standard(0.1, N_vec[i], T, 'optimized', fast=True, grad=False)
    for j in range(len(T)):
        print(N_vec[i], T[j])
        indices = slice(int((1-np.sqrt(T[j])) * 300), -int((1-np.sqrt(T[j])) * 150))
        if j == len(T)-1:
            indices = slice(0, len(k_vec))
        k_loc = k_vec[indices]
        res_loc = rHestonMarkov.implied_volatility(K=np.exp(k_loc), H=0.1, lambda_=0.3,
                                                                           rho=-0.7, nu=0.3, theta=0.02,
                                                                           V_0=0.02, T=T[j], rel_tol=5e-04,
                                                                           nodes=nodes, weights=weights, N=-1)
        res_loc_ = np.zeros(len(k_vec))
        res_loc_[indices] = res_loc
        approx_surface_opt_all[i, j, :] = res_loc_
        max_errors_opt_all[i, j] = np.amax(np.abs(true_surface[j, indices] - approx_surface_opt_all[i, j, indices]) / true_surface[j, indices])
        avg_errors_opt_all[i, j] = np.average(np.abs(true_surface[j, indices] - approx_surface_opt_all[i, j, indices]) / true_surface[j, indices])
    time_opt_all += time.perf_counter() - tic
    print(f'current time: {time_opt_all}')
    print(max_errors_opt_all[i, :])
    print(avg_errors_opt_all[i, :])
    max_max_errors_opt_all[i] = np.amax(max_errors_opt_all[i, :])
    print(max_max_errors_opt_all[i])
    avg_avg_errors_opt_all[i] = np.average(avg_errors_opt_all[i, :])
    print(avg_avg_errors_opt_all[i])

with open('approximate surface observation 02.npy', 'wb') as f:
    np.save(f, approx_surface_obs_02)
with open('approximate surface optimization 02.npy', 'wb') as f:
    np.save(f, approx_surface_opt_02)
with open('approximate surface optimization all.npy', 'wb') as f:
    np.save(f, approx_surface_opt_all)
print('Finished!!!!!')
time.sleep(3600000)

'''
for N in np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024]):
    print(N)
    if N >= 2:
        tic = time.perf_counter()
        nodes, weights = rk.quadrature_rule_geometric_standard(0.1, N, 0.2, 'observation')
        duration = time.perf_counter() - tic
        print(np.amax(nodes))
        print(error_func(0.1, nodes+1e-08, weights, T))
        print(duration)
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule_geometric_standard(0.1, N, 0.2, 'optimized', fast=True, grad=True)
    duration = time.perf_counter() - tic
    print(np.amax(nodes))
    print(error_func(0.1, nodes, weights, T))
    print(duration)
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule_geometric_standard(0.1, N, 0.2, 'optimized', fast=True, grad=False)
    duration = time.perf_counter() - tic
    print(np.amax(nodes))
    print(error_func(0.1, nodes, weights, T))
    print(duration)
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule_geometric_standard(0.1, N, T, 'optimized', fast=True, grad=False)
    duration = time.perf_counter() - tic
    print(np.amax(nodes))
    print(error_func(0.1, nodes, weights, T))
    print(duration)
'''
time.sleep(3600000)


H = 0.2
a = 1.06418
b = 0.4275

c_H = rk.c_H(H)
A = (1/H + 1/(1.5-H))**(1/2)
frac = np.exp(a*b) / (np.exp(a*b)-1)
gamma = 1/(3/8*frac + 6*H - 6*H*H)

const = c_H**2 * (1/(2*H) + 8/frac + 1/(3-2*H)) * ((3/H)**(frac*(3-2*H)/8) * (5*np.pi**3/384 * np.exp(a*b) * (np.exp(a*b)-1) * A**(2-2*H) / b**(2-2*H) / H)**(6*H-4*H*H) * (1/(1.5-H))**(frac*H/4))**gamma
print(np.sqrt(const))
time.sleep(360000)


for H in np.array([0.2]):
    for N in range(130, 1000):
        m, n, a, b = rk.get_parameters(H, N+1, T=1., mode='theorem')
        print(N+1, 'theorem', rk.error_estimate_fBm(H, m, n, a, b, 1.))

        m, n, a, b = rk.get_parameters(H, N+1, T=1., mode='observation')
        print(N+1, 'observation', rk.error_estimate_fBm(H, m, n, a, b, 1.))

Data.plot_rHeston_optimized_smiles()
'''
k = np.linspace(-0.8, 0.2, 201)
N_Riccati = 200
L_true = 700
L_vec = np.array([150, 200, 250, 300, 350, 450])
N_Fourier = 10
true = rHestonMarkov.implied_volatility(K=np.exp(k), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=0.1,
                                        rel_tol=1e+30, N=3, L=600, N_Fourier=10000, N_Riccati=500)
approx = rHestonMarkov.implied_volatility(K=np.exp(k), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=0.1,
                                        rel_tol=1e+30, N=3, L=400, N_Fourier=2800, N_Riccati=200)

error = np.abs(approx-true)/true
plt.plot(k, error)
print(error)
print(np.amax(error))
print(np.average(error))
plt.yscale('log')
plt.xlabel('log-moneyness')
plt.ylabel('Relative error')
plt.show()

for i in range(len(L_vec)):
    approx = rHestonMarkov.implied_volatility(K=np.exp(k), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=0.1,
                                        rel_tol=1e+30, N=3, L=L_vec[i], N_Fourier=N_Fourier*L_vec[i], N_Riccati=N_Riccati)
    error = np.abs(approx-true)/true
    plt.plot(k, error, color=c[i], label=r'$L=$' + f'{L_vec[i]}')
    print(error)
    print(L_vec[i])
    print(np.amax(error))
    print(np.average(error))
plt.yscale('log')
plt.xlabel('log-moneyness')
plt.ylabel('Relative error')
plt.legend(loc='upper right')
plt.show()

Data.plot_rHeston_optimized_smiles()
k_02 = np.linspace(-0.44, 0.22, 201)

k = np.linspace(-0.4, 0.18, 201)
rHestonMarkov.implied_volatility(K=np.exp(k), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02,
                                                 V_0=0.02, T=0.1, rel_tol=2e-03, N=3)
'''
k = Data.k_001
tic = time.perf_counter()
# iv_true = rHeston.implied_volatility(K=np.exp(k), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=0.01,
#                                      rel_tol=2e-04, smoothing=False)
iv_true = Data.rHeston_001

print(iv_true)
print(f'True rough Heston: {time.perf_counter() - tic}')
plt.plot(k, iv_true, 'k-', label='rough Heston')
N_vec = np.array([2, 4, 8, 16, 32])
for i in range(len(N_vec)):
    tic = time.perf_counter()
    iv_approx = rHestonMarkov.implied_volatility(K=np.exp(k), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02,
                                                 V_0=0.02, T=0.01, rel_tol=2e-04, N=N_vec[i], smoothing=False, mode='observation')
    print((iv_approx,))
    print(f'Markovian approximation, N={N_vec[i]}: {time.perf_counter()-tic}')
    plt.plot(k, iv_approx, color=c[i], label=f'{N_vec[i]}-dimensional approximation')
    print(f'N={N_vec[i]}, error: {np.amax(np.abs(iv_approx-iv_true)/iv_true)}')
# plt.plot(Data.k_rrHeston, Data.rrHeston)
plt.xlabel('log-moneyness')
plt.ylabel('implied volatility')
plt.legend(loc='upper right')
plt.show()

print(np.amax(np.abs(iv_true-iv_approx)/iv_true))

plt.plot(k, iv_true)
plt.plot(k, iv_approx)
plt.plot(Data.k_rHeston, Data.rHeston)
plt.show()


N = 2
N_Riccati_1 = 200
N_Riccati_2 = 1000
N_Fourier_1 = 300
N_Fourier_2 = 3000
L_1 = 50
L_2 = 100
lambda_ = np.array([0.1, 0.3, 0.8])
rho = np.array([-0.2, -0.6, -0.9])
nu = np.array([0.1, 0.3, 0.8])
V_0 = np.array([0.01, 0.03, 0.1])
theta = np.array([0.01, 0.03, 0.1])
errors_1 = np.empty((3, 3, 3, 3, 3))
errors_2 = np.empty((3, 3, 3, 3, 3))
k_vec = np.linspace(-0.8, 0.4, 121)

for a in range(3):
    for b in range(3):
        for c in range(3):
            for d in range(3):
                for e in range(3):
                    print(a, b, c, d, e)
                    N_Riccati_1, L_1, N_Fourier_1 = rHestonMarkov.find_parameters(0.1, lambda_[a], rho[b], nu[c], theta[d], V_0[e], 1., 2, 0, 2, 0, 0)
                    N_Riccati_2 = 2*N_Riccati_1
                    L_2 = 1.5 * L_1
                    N_Fourier_2 = 3 * N_Fourier_1
                    print(N_Riccati_1, L_1, N_Fourier_1)
                    true = rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(k_vec), H=0.1, lambda_=lambda_[a], rho=rho[b], nu=nu[c], theta=theta[d], V_0=V_0[e], T=1., N_Riccati=N_Riccati_2, R=2, N_Fourier=N_Fourier_2, L=L_2, q=1, adaptive=False)
                    approx = rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(k_vec), H=0.1, lambda_=lambda_[a], rho=rho[b], nu=nu[c], theta=theta[d], V_0=V_0[e], T=1., N_Riccati=N_Riccati_1, R=2, N_Fourier=N_Fourier_1, L=L_1, q=1, adaptive=False)
                    errors_1[a, b, c, d, e] = np.average(np.abs(approx-true)/true)
                    errors_2[a, b, c, d, e] = np.amax(np.abs(approx-true)/true)
                    print(errors_1[a, b, c, d, e])
                    print(errors_2[a, b, c, d, e])
print(errors_1)
print(errors_2)
print(np.amax(errors_1))
print(np.amax(errors_2))
time.sleep(3600000)

true = rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(k_vec), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=500, R=2, N_Fourier=1000, L=70, q=1, adaptive=False)
approx = rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(k_vec), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=200, R=2, N_Fourier=300, L=50, q=1, adaptive=False)

print(np.average(np.abs(approx-true)/true))
print(np.amax(np.abs(approx-true)/true))
plt.plot(k_vec, approx, label=r'$\sigma_1$')
plt.plot(k_vec, true, label=r'$\sigma_2$')
plt.xlabel('log-moneyness ' + r'$k$')
plt.ylabel('implied volatility ' + r'$\sigma(k)$')
plt.legend(loc='upper right')
plt.show()
plt.plot(Data.k_rHeston, Data.rHeston, label='True rough Heston')
plt.plot(k_vec, approx, label=r'$\sigma_1$')
plt.plot(k_vec, true, label=r'$\sigma_2$')
plt.xlabel('log-moneyness ' + r'$k$')
plt.ylabel('implied volatility ' + r'$\sigma(k)$')
plt.legend(loc='upper right')
plt.show()

# print(rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(k_vec), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1, N_Riccati=200, R=2, N_fourier=10000, L=50))

for i in range(len(m_vec)):
    true_smile = rHestonMarkov.implied_volatility(mode='optimized', N=N_vec[i], K=np.exp(k_vec), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=N_Riccati, R=2, N_Fourier=N_Fourier_true, L=L, q=1, adaptive=False)
    for j in range(len(N_Fourier_vec)):
        approx_smile = rHestonMarkov.implied_volatility(mode='optimized', N=N_vec[i], K=np.exp(k_vec), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=N_Riccati, R=2, N_Fourier=N_Fourier_vec[j], L=L, q=1, adaptive=False)
        errors[i, j] = np.average(np.abs(approx_smile-true_smile)/true_smile)
        print(errors)

c = ['r', 'C1', 'y', 'g', 'b']
for i in range(len(N_vec)):
    plt.loglog(N_Fourier_vec, errors[i, :], c[i] + '-', label=f'{N_vec[i]} nodes')
plt.legend(loc='upper right')
plt.xlabel('Number of quadrature intervals in the Fourier inversion formula')
plt.ylabel('Error')
plt.show()
print(rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(Data.k_rHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=200, R=2, N_Fourier=10000, L=50))




print(rk.compare_approximations(0.1, np.array([1])))
print(rk.quadrature_rule_geometric_standard(0.1, 1, 1, 'optimized'))


approx = np.array([0.40453267, 0.40464352, 0.40445746, 0.40374146, 0.40240433, 0.40048715,
 0.39814312, 0.39560692, 0.393148  , 0.39100713, 0.38933052, 0.38812937,
 0.38728688, 0.38660761, 0.38588205, 0.38493957, 0.38367906, 0.382079,
 0.38019219, 0.37812914, 0.37603158, 0.37403909, 0.37225494, 0.3707212,
 0.36941189, 0.36824643, 0.36711673, 0.36591728, 0.36456939, 0.36303524,
 0.36132138, 0.35947304, 0.357561  , 0.3556638 , 0.35384865, 0.35215579,
 0.35059048, 0.34912469, 0.34770736, 0.34627929, 0.34478815, 0.34319979,
 0.3415041 , 0.33971489, 0.33786463, 0.33599542, 0.33414837, 0.33235388,
 0.33062509, 0.32895612, 0.32732518, 0.32570135, 0.32405283, 0.32235448,
 0.32059302, 0.31876897, 0.31689523, 0.31499301, 0.31308613, 0.31119515,
 0.30933269, 0.30750101, 0.30569232, 0.30389152, 0.30208041, 0.30024224,
 0.29836553, 0.29644622, 0.29448796, 0.29250056, 0.29049711, 0.28849059,
 0.28649074, 0.28450194, 0.28252255, 0.28054577, 0.27856161, 0.2765595,
 0.2745308 , 0.27247064, 0.27037868, 0.26825883, 0.26611786, 0.26396356,
 0.26180273, 0.25963957, 0.2574748 , 0.25530567, 0.25312685, 0.2509318,
 0.24871436, 0.24647011, 0.24419727, 0.24189678, 0.23957189, 0.23722707,
 0.23486683, 0.23249453, 0.23011154, 0.227717  , 0.22530805, 0.22288057,
 0.22043016, 0.21795314, 0.21544729, 0.21291224, 0.21034933, 0.20776115,
 0.20515073, 0.20252071, 0.19987268, 0.19720672, 0.19452141, 0.1918142,
 0.18908205, 0.18632217, 0.18353272, 0.18071324, 0.17786487, 0.17499008,
 0.17209226, 0.16917509, 0.16624191, 0.16329536, 0.16033726, 0.15736882,
 0.15439127, 0.15140668, 0.14841887, 0.14543424, 0.14246242, 0.13951647,
 0.13661281, 0.13377065, 0.13101123, 0.1283569 , 0.12583013, 0.12345262,
 0.12124463, 0.11922439, 0.1174077 , 0.1158075 , 0.11443348, 0.11329138,
 0.11238229, 0.11170178, 0.11123917, 0.11097729, 0.11089314, 0.1109595,
 0.11114769, 0.11143101, 0.11178819, 0.11220622, 0.1126816 , 0.11321962,
 0.11383131, 0.11452826, 0.11531621, 0.1161892 , 0.11712638])

something = np.array([0.40292376, 0.40320749, 0.40312071, 0.40242589, 0.40104162, 0.39902783,
 0.39656316, 0.39391226, 0.39137456, 0.38921284, 0.38757875, 0.38646947,
 0.38574101, 0.385169  , 0.38452228, 0.38361947, 0.38235893, 0.3807265,
 0.37878791, 0.37666965, 0.37452958, 0.37251993, 0.37074973, 0.36925818,
 0.36800917, 0.36690842, 0.36583498, 0.36467451, 0.36334473, 0.36180936,
 0.36008053, 0.35821153, 0.35628189, 0.35437764, 0.35257046, 0.35090104,
 0.34937117, 0.34794672, 0.34656978, 0.34517545, 0.34370799, 0.34213261,
 0.34044112, 0.33865109, 0.33679958, 0.33493304, 0.33309569, 0.3313191,
 0.32961554, 0.32797655, 0.32637694, 0.32478243, 0.32315877, 0.32147972,
 0.31973238, 0.3179188 , 0.31605422, 0.31416227, 0.31226879, 0.31039545,
 0.30855489, 0.30674841, 0.30496659, 0.30319246, 0.30140624, 0.29959023,
 0.29773272, 0.29583022, 0.29388747, 0.29191559, 0.2899289 , 0.28794124,
 0.28596263, 0.28399716, 0.28204246, 0.28009071, 0.27813095, 0.27615188,
 0.27414454, 0.27210414, 0.27003082, 0.26792917, 0.26580669, 0.26367177,
 0.26153151, 0.25939008, 0.25724787, 0.25510161, 0.25294537, 0.25077207,
 0.24857522, 0.24635034, 0.24409576, 0.2418128 , 0.23950509, 0.23717751,
 0.23483481, 0.23248041, 0.23011556, 0.22773909, 0.22534775, 0.22293703,
 0.22050223, 0.21803949, 0.21554659, 0.21302329, 0.21047116, 0.20789302,
 0.20529211, 0.20267114, 0.20003163, 0.19737348, 0.19469498, 0.19199325,
 0.18926496, 0.18650708, 0.18371766, 0.18089628, 0.17804417, 0.17516398,
 0.17225925, 0.16933375, 0.16639082, 0.16343297, 0.16046179, 0.15747824,
 0.15448328, 0.15147882, 0.14846869, 0.14545951, 0.14246138, 0.13948806,
 0.13655685, 0.133688  , 0.13090384, 0.12822778, 0.12568325, 0.12329282,
 0.12107741, 0.11905573, 0.11724387, 0.11565484, 0.11429811, 0.11317895,
 0.11229766, 0.11164864, 0.11121979, 0.11099232, 0.11094159, 0.11103902,
 0.11125505, 0.11156275, 0.11194137, 0.11237895, 0.11287342, 0.11343143,
 0.11406486, 0.11478529, 0.11559736, 0.1164931 , 0.11744917])

plt.plot(Data.k_rHeston, Data.rHeston, 'k-', label='True rough Heston')
plt.plot(Data.k_rHeston, something, label='One-point approximation')
plt.legend(loc='upper right')
plt.show()

print(rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(Data.k_rHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=200, R=2, N_Fourier=10000, L=50))

# print(rHestonMarkov.implied_volatility(K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, adaptive=True, mode='optimized'))
# print(rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=200, adaptive=True, R=2, N_fourier=10000, L=50))
# print(rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50, R=2))
# print(rHestonMarkov.implied_volatility(N=1, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50, R=2))
time.sleep(360000)

'''
def smooth_point_distribution(x, eps):
    dim = len(x[0])
    return lambda y: np.sum(np.array([np.exp(-1/2 * np.sum((x[i]-y)**2)/eps**2)/np.sqrt((2*np.pi*eps**2)**dim) for i in range(len(x))]))


x = [np.array([0, 0]), np.array([1, 2])]
x_left, x_right = -1, 3
y_left, y_right = -3, 3
dx = 0.05
distribution = smooth_point_distribution(x, dx)

# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(np.arange(y_left, y_right, dx), np.arange(x_left, x_right, dx))

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = np.array([[distribution(np.array([x[i, j], y[i, j]])) for j in range(x.shape[1])] for i in range(x.shape[0])])
z = z[:-1, :-1]
z_min, z_max = z.min(), z.max()

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='Reds', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()

with open('dW0.npy', 'rb') as f:
    dW = np.load(f)
with open('dB0.npy', 'rb') as f:
    dB = np.load(f)

WB = np.empty((2, 1, 2048))
WB[0, 0, :] = dW[0, :]
WB[1, 0, :] = dB[0, :]
S_1, V_1, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, WB=WB, vol_behaviour='adaptive')
plt.plot(np.linspace(0, 1, 2049), S_1[0, :], label='stock price, N=2048')
plt.plot(np.linspace(0, 1, 2049), V_1[0, :], label='volatility, N=2048')
plt.plot(np.linspace(0, 1, 1025), S_2[0, :], label='stock price, N=1024')
plt.plot(np.linspace(0, 1, 1025), V_2[0, :], label='volatility, N=1024')
plt.legend(loc='best')
plt.xlabel('t')
plt.title(f'Sample paths of {vb} implementation')
plt.show()



time.sleep(3600)
'''
''''
S, V, _, bad = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1, N=6, vol_behaviour='adaptive')
print(bad)
plt.plot(np.linspace(0, 1, 1001), S[0, :])
plt.plot(np.linspace(0, 1, 1001), V[0, :])
plt.show()
ie.call(K=np.array([1.]), vol_behaviour='adaptive')
# methods = ['mean reversion', 'hyperplane reflection', 'split throw', 'multiple time scales']
# methods = ['sticky', 'hyperplane reset', 'split kernel']
'''
methods = ['adaptive']

with open('dW0.npy', 'rb') as f:
    dW = np.load(f)
with open('dB0.npy', 'rb') as f:
    dB = np.load(f)
'''
WB_1 = np.empty((2, 1, 2048))
WB_2 = np.empty((2, 1, 1024))
WB_1[0, 0, :] = dW[0, :]
WB_1[1, 0, :] = dB[0, :]
for i in range(1024):
    WB_2[:, :, i] = WB_1[:, :, 2*i] + WB_1[:, :, 2*i+1]
for vb in methods:
    S_1, V_1, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, WB=WB_1, vol_behaviour=vb)
    S_2, V_2, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, WB=WB_2, vol_behaviour=vb)
    plt.plot(np.linspace(0, 1, 2049), S_1[0, :], label='stock price, N=2048')
    plt.plot(np.linspace(0, 1, 2049), V_1[0, :], label='volatility, N=2048')
    plt.plot(np.linspace(0, 1, 1025), S_2[0, :], label='stock price, N=1024')
    plt.plot(np.linspace(0, 1, 1025), V_2[0, :], label='volatility, N=1024')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.title(f'Sample paths of {vb} implementation')
    plt.show()
'''
'''
for vb in methods:
    samples = 10000
    S_errors = np.empty(samples)
    V_errors = np.empty(samples)
    WB_1 = np.empty((2, samples, 2048))
    WB_2 = np.empty((2, samples, 1024))
    WB_1[0, :, :] = dW[:samples, :]
    WB_1[1, :, :] = dB[:samples, :]
    for j in range(1024):
        WB_2[:, :, j] = WB_1[:, :, 2 * j] + WB_1[:, :, 2 * j + 1]
    S_1_, V_1_, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6,
                                     WB=WB_1, vol_behaviour=vb)
    S_2, V_2, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6,
                                     WB=WB_2, vol_behaviour=vb)
    S_1 = np.empty((samples, 1025))
    V_1 = np.empty((samples, 1025))
    for i in range(1025):
        S_1[:, i] = S_1_[:, 2*i]
        V_1[:, i] = V_1_[:, 2*i]
    S_errors = np.amax(np.abs(S_1-S_2), axis=-1)
    V_errors = np.amax(np.abs(V_1-V_2), axis=-1)
    print(len(S_errors))
    S_avg_error, S_std_error = cf.MC(S_errors)
    V_avg_error, V_std_error = cf.MC(V_errors)
    print(f'The strong error for S is roughly {S_avg_error} +/- {S_std_error}.')
    print(f'The strong error for V is roughly {V_avg_error} +/- {V_std_error}.')
'''
N_time = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
samples = np.empty(1000000)
k_vec = Data.k_rHeston
S = np.empty((len(N_time), 1000000))

print("Here")

for vb in methods:
    for N_t in N_time:
        WB = np.empty((2, 100000, N_t))
        factor = 2048/N_t
        for i in range(10):
            with open(f'dW{i}.npy', 'rb') as f:
                dW = np.load(f)
            with open(f'dB{i}.npy', 'rb') as f:
                dB = np.load(f)
            for j in range(N_t):
                WB[0, :, j] = np.sum(dW[:, int(j*factor):int((j+1)*factor)], axis=-1)
                WB[1, :, j] = np.sum(dB[:, int(j*factor):int((j+1)*factor)], axis=-1)
            print(vb, N_t, i)
            samples[i*100000:(i+1)*100000] = ie.samples(H=0.49, N=1, N_time=N_t, WB=WB, vol_behaviour=vb)
        np.save(f'samples of {vb} mode with N_time={N_t} and H=0.49 and N=1', samples)
time.sleep(3600000)
for vb in methods:
    for i in range(len(N_time)):
        print(N_time[i])
        with open(f'samples of {vb} mode with N_time={N_time[i]}.npy', 'rb') as f:
            S[i, :] = np.load(f)
        est, lower, upper = cf.volatility_smile_call(S[i, :], np.exp(k_vec), 1., 1.)
        plt.plot(k_vec, est, label=f'N_time={N_time[i]}')
    plt.plot(k_vec, lower, 'k--')
    plt.plot(k_vec, upper, 'k--')
    plt.plot(Data.k_rHeston, Data.rHeston_6, 'k-', label=f'Fourier inversion')
    plt.title(vb)
    plt.legend(loc='best')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.show()

strikes = np.array([-0.3, 0., 0.15])
true_iv = np.array([np.interp(strike, Data.k_rHeston, Data.rHeston_6) for strike in strikes])
all_errors = np.empty((len(methods), len(N_time), len(strikes)))
for k in range(len(methods)):
    conv_rate = 1.
    for i in range(len(N_time)):
        print(N_time[i])
        with open(f'samples of {methods[k]} mode with N_time={N_time[i]}.npy', 'rb') as f:
            S[i, :] = np.load(f)
    S_Rich = (S[1:, :] - 2 ** (-conv_rate) * S[:-1, :]) / (1 - 2 ** (-conv_rate))
    iv = np.empty((3, len(N_time), len(strikes)))
    iv_Rich = np.empty((3, len(N_time)-1, len(strikes)))
    errors = np.empty((len(N_time), len(strikes)))
    errors_Rich = np.empty((len(N_time)-1, len(strikes)))
    errors_alt = np.empty((len(N_time)-1, len(strikes)))
    confidence = np.empty(len(strikes))
    confidence_Rich = np.empty(len(strikes))
    for i in range(len(N_time)):
        print(N_time[i])
        iv[0, i, :], iv[1, i, :], iv[2, i, :] = cf.volatility_smile_call(S[i, :], np.exp(strikes), 1., 1.)
        errors[i, :] = np.abs(iv[0, i, :] - true_iv)
        all_errors[k, :, :] = errors
    for i in range(len(N_time)-1):
        errors_alt[i, :] = np.abs(iv[0, i, :] - iv[0, -1, :])
        print(N_time[i+1])
        iv_Rich[0, i, :], iv_Rich[1, i, :], iv_Rich[2, i, :] = cf.volatility_smile_call(S_Rich[i, :], np.exp(strikes), 1., 1.)
        # iv_Rich[0, :, :] = (iv[0, 1:, :] - 2**conv_rate * iv[0, :-1, :])/(1-2**conv_rate)
        errors_Rich[i, :] = np.abs(iv_Rich[0, i, :] - true_iv)
    confidence = np.fmax(iv[2, -1, :]-iv[0, -1, :], iv[0, -1, :] - iv[1, -1, :])
    confidence_Rich = np.fmax(iv_Rich[2, -1, :] - iv_Rich[0, -1, :], iv_Rich[0, -1, :] - iv_Rich[1, -1, :])
    constants = np.empty(len(strikes))
    rates = np.empty(len(strikes))
    constants_alt = np.empty(len(strikes))
    rates_alt = np.empty(len(strikes))
    constants_Rich = np.empty(len(strikes))
    rates_Rich = np.empty(len(strikes))
    for i in range(len(strikes)):
        rates[i], constants[i], _, _, _ = Data.log_linear_regression(N_time[3:-1], errors[3:-1, i])
        rates_alt[i], constants_alt[i], _, _, _ = Data.log_linear_regression(N_time[3:-2], errors_alt[3:-1, i])
        rates_Rich[i], constants_Rich[i], _, _, _ = Data.log_linear_regression(N_time[4:], errors_Rich[3:, i])
    for i in range(len(strikes)):
        plt.loglog(N_time, errors[:, i], 'b-', label='error')
        plt.loglog(N_time, constants[i]*N_time**rates[i], 'b--')
        plt.loglog(N_time[1:], errors_Rich[:, i], 'r-', label='error Richardson')
        plt.loglog(N_time, constants_Rich[i]*N_time**rates_Rich[i], 'r--')
        plt.loglog(N_time, confidence[i]*np.ones(len(N_time)), 'k--', label='Monte Carlo error')
        plt.legend(loc='best')
        plt.title(f'{methods[k]}, log-moneyness={strikes[i]},\ncomparison with Fourier inversion')
        x_label = 'number of time steps'
        x_label += '\n\nerror ' + r'$\approx$' + f' {constants[i]:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(rates[i]))
        plt.xlabel(x_label)
        plt.ylabel('error')
        plt.show()

        plt.loglog(N_time, errors[:, i], 'b-', label='total error')
        plt.loglog(N_time, constants[i] * N_time ** rates[i], 'b--')
        plt.loglog(N_time, confidence[i] * np.ones(len(N_time)), 'k--', label='Monte Carlo error')
        plt.legend(loc='best')
        plt.title(f'{methods[k]}, log-moneyness={strikes[i]},\ncomparison with Fourier inversion')
        x_label = 'number of time steps'
        x_label += '\n\nerror ' + r'$\approx$' + f' {constants[i]:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(rates[i]))
        plt.xlabel(x_label)
        plt.ylabel('error')
        plt.show()

        plt.loglog(N_time[:-1], errors_alt[:, i], 'b-', label='error')
        plt.loglog(N_time[:-1], constants_alt[i] * N_time[:-1] ** rates_alt[i], 'b--', label='regression')
        plt.loglog(N_time[:-1], confidence[i]*np.ones(len(N_time[:-1])), 'k--', label='Monte Carlo error')
        plt.legend(loc='best')
        plt.title(f'{methods[k]}, log-moneyness={strikes[i]},\ncomparison with {N_time[-1]} time points')
        x_label = 'number of time steps'
        x_label += '\n\nerror ' + r'$\approx$' + f' {constants_alt[i]:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(rates_alt[i]))
        plt.xlabel(x_label)
        plt.ylabel('error')
        plt.show()

for i in range(len(strikes)):
    for k in range(len(methods)):
        plt.loglog(N_time, all_errors[k, :, i], label=methods[k])
    plt.loglog(N_time, confidence[i] * np.ones(len(N_time)), 'k--', label='Monte Carlo error')
    plt.legend(loc='best')
    plt.title(f'log-moneyness={strikes[i]},\ncomparison with Fourier inversion')
    plt.xlabel('number of time steps')
    plt.ylabel('error')
    plt.show()


time.sleep(3600000)

for vb in methods:
    for N_t in N_time:
        WB = np.empty((2, 100000, N_t))
        factor = 2048/N_t
        for i in range(10):
            with open(f'dW{i}.npy', 'rb') as f:
                dW = np.load(f)
            with open(f'dB{i}.npy', 'rb') as f:
                dB = np.load(f)
            for j in range(N_t):
                WB[0, :, j] = np.sum(dW[:, int(j*factor):int((j+1)*factor)], axis=-1)
                WB[1, :, j] = np.sum(dB[:, int(j*factor):int((j+1)*factor)], axis=-1)
            print(vb, N_t, i)
            samples[i*100000:(i+1)*100000] = ie.samples(N_time=N_t, WB=WB, vol_behaviour=vb)
        np.save(f'samples of {vb} mode with N_time={N_t}', samples)

time.sleep(3600000)

Data.plot_rHeston_IE_smiles()

K = np.exp(Data.k_rHeston)
N = 6
tic = time.perf_counter()
vol, lower, upper = ie.call(K, N=N, N_time=1000, m=200000, vol_behaviour='split kernel')
toc = time.perf_counter()
print(toc-tic)
print(vol)
print(lower)
print(upper)
time.sleep(360000)

S, V, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, vol_behaviour='split kernel')
plt.plot(np.linspace(0, 1, 1001), S, label='stock price')
plt.plot(np.linspace(0, 1, 1001), V, label='volatility')
plt.xlabel('t')
plt.title('Sample path of split kernel implementation')
plt.show()

time.sleep(360000)

'''
S, V, _ = ie.get_sample_path(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, vol_behaviour='multiple time scales')
plt.plot(np.linspace(0, 1, 1001), S, label='stock price')
plt.plot(np.linspace(0, 1, 1001), V, label='volatility')
plt.xlabel('t')
plt.title('Sample path of multiple time scales implementation')
plt.show()
'''

K = np.exp(Data.k_rHeston)
N = 6
tic = time.perf_counter()
vol, lower, upper = ie.call(K, N=N, N_time=1000, m=200000, vol_behaviour='multiple time scales')
toc = time.perf_counter()
print(toc-tic)
print(vol)
print(lower)
print(upper)
np.savetxt(f'rHestonIE mutiple time scales, vol.txt', vol, delimiter=',', header=f'time: {toc - tic}')
np.savetxt(f'rHestonIE multiple time scales, lower.txt', lower, delimiter=',', header=f'time: {toc - tic}')
np.savetxt(f'rHestonIE multiple time scales, upper.txt', upper, delimiter=',', header=f'time: {toc - tic}')
time.sleep(360000)

for N_time in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    K = np.exp(-1.1 + 0.01 * np.arange(171))
    N = 12
    tic = time.perf_counter()
    vol, lower, upper = ie.call(K, N=N, N_time=N_time, m=1000000, bounce_vol=False)
    toc = time.perf_counter()
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, vol.txt', vol, delimiter=',', header=f'time: {toc - tic}')
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, lower.txt', lower, delimiter=',', header=f'time: {toc - tic}')
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, upper.txt', upper, delimiter=',', header=f'time: {toc - tic}')

'''
S, V, V_comp = ie.get_sample_path(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6,
                                  bounce_vol=True)
plt.plot(np.linspace(0, 1, 1001), S, label="stock price")
plt.plot(np.linspace(0, 1, 1001), V, label="volatility")
plt.plot(np.linspace(0, 1, 1001), V_comp[-1, :], label="low mean reversion")
plt.plot(np.linspace(0, 1, 1001), V_comp[-2, :], label="high mean reversion")
plt.legend(loc="upper left")
plt.show()
time.sleep(3600)

print("Hello World!")
Data.rHeston_smiles_precise()
Data.plot_rBergomi_smiles()
time.sleep(3600)

K = np.exp(-1.3 + 0.01 * np.arange(161))
print("True rough Heston:")
tic = time.perf_counter()
true_heston = rHeston.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02, T=0.01,
                                         N_Riccati=3000, N_fourier=10000, L=50.)
toc = time.perf_counter()
print(true_heston)
print(f"Generating the true smile took {toc - tic} seconds.")

for N in [1, 2, 3, 4, 5, 6, 7, 8]:
    print(f"Approximation with {N} nodes, our scheme:")
    tic = time.perf_counter()
    approximated_heston = rHestonMarkov.implied_volatility(K=K, lambda_=0.3, rho=-0.7, nu=0.3, H=0.1, V_0=0.02, theta=0.02,
                                                       T=0.01, N=N, N_Riccati=3000, N_fourier=10000, L=50.)
    toc = time.perf_counter()
    print(approximated_heston)
    print(f"Generating the approximated smile with N={N} took {toc - tic} seconds.")
'''
