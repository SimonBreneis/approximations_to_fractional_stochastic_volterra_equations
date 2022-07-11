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


k_vec = np.linspace(-1.5, 0.75, 451)
with open('true surface.npy', 'rb') as f:
    true_surface = np.load(f)
true_surface = true_surface[-1, :]

# k_vec = np.linspace(-0.5, 0.1, 101)
# true_surface = rHeston.implied_volatility_smile(K=np.exp(k_vec), H=0.07, lambda_=0.6, rho=-0.8, nu=0.5, theta=0.01, V_0=0.01,
#                                                T=0.04, rel_tol=1e-04)
print(rHeston.implied_volatility_smile(K=np.exp(k_vec), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02,
                                                T=1, rel_tol=2e-05))
print((true_surface,))

for N in np.array([1, 2, 3, 4, 5, 6]):
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule(0.1, N, 1, mode='observation')
    smile = rHestonMarkov.implied_volatility_smile(K=np.exp(k_vec), H=0.07, lambda_=0.6,
                                                   rho=-0.8, nu=0.5, theta=0.01,
                                                   V_0=0.01, T=0.04, rel_tol=1e-04,
                                                   nodes=nodes, weights=weights, N=-1)
    dur = time.perf_counter() - tic
    kernel_error = np.sqrt(rk.error(0.1, nodes, weights, 1))/rk.kernel_norm(0.1, 1)
    smile_error = np.amax(np.abs(true_surface - smile) / true_surface)
    print(N, 'observation', np.amax(nodes), kernel_error, smile_error, dur)
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule(0.1, N, 1, mode='optimized')
    smile = rHestonMarkov.implied_volatility_smile(K=np.exp(k_vec), H=0.07, lambda_=0.6,
                                                   rho=-0.8, nu=0.5, theta=0.01,
                                                   V_0=0.01, T=0.04, rel_tol=1e-04,
                                                   nodes=nodes, weights=weights, N=-1)
    dur = time.perf_counter() - tic
    kernel_error = np.sqrt(rk.error(0.1, nodes, weights, 1))/rk.kernel_norm(0.1, 1)
    smile_error = np.amax(np.abs(true_surface - smile) / true_surface)
    print(N, 'optimized', np.amax(nodes), kernel_error, smile_error, dur)
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule(0.1, N, 1, mode='european')
    smile = rHestonMarkov.implied_volatility_smile(K=np.exp(k_vec), H=0.07, lambda_=0.6,
                                                   rho=-0.8, nu=0.5, theta=0.01,
                                                   V_0=0.01, T=0.04, rel_tol=1e-04,
                                                   nodes=nodes, weights=weights, N=-1)
    dur = time.perf_counter() - tic
    kernel_error = np.sqrt(rk.error(0.1, nodes, weights, 1))/rk.kernel_norm(0.1, 1)
    smile_error = np.amax(np.abs(true_surface - smile) / true_surface)
    print(N, 'european', np.amax(nodes), kernel_error, smile_error, dur)
time.sleep(36000)


nodes, weights = rk.quadrature_rule(0.1, 1, 1, 'optimized')
rule = np.empty(2*len(nodes))
rule[:len(nodes)] = nodes
rule[len(nodes):] = weights

k_vec = np.linspace(-1.5, 0.75, 451)
with open('true surface.npy', 'rb') as f:
    true_surface = np.load(f)
true_surface = true_surface[-1, :]
time.sleep(36000)


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
            nodes, weights = rk.quadrature_rule(0.1, N_vec[i], 0.2, 'observation')
            res_loc = rHestonMarkov.implied_volatility_smile(K=np.exp(k_loc), H=0.1, lambda_=0.3,
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
    nodes, weights = rk.quadrature_rule(0.1, N_vec[i], 0.2, 'optimized', fast=True, grad=True)
    for j in range(len(T)):
        print(N_vec[i], T[j])
        indices = slice(int((1-np.sqrt(T[j])) * 300), -int((1-np.sqrt(T[j])) * 150))
        if j == len(T)-1:
            indices = slice(0, len(k_vec))
        k_loc = k_vec[indices]
        res_loc = rHestonMarkov.implied_volatility_smile(K=np.exp(k_loc), H=0.1, lambda_=0.3,
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
    nodes, weights = rk.quadrature_rule(0.1, N_vec[i], T, 'optimized', fast=True, grad=False)
    for j in range(len(T)):
        print(N_vec[i], T[j])
        indices = slice(int((1-np.sqrt(T[j])) * 300), -int((1-np.sqrt(T[j])) * 150))
        if j == len(T)-1:
            indices = slice(0, len(k_vec))
        k_loc = k_vec[indices]
        res_loc = rHestonMarkov.implied_volatility_smile(K=np.exp(k_loc), H=0.1, lambda_=0.3,
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
time.sleep(360000)

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

S, V, _ = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, vol_behaviour='split kernel')
plt.plot(np.linspace(0, 1, 1001), S, label='stock price')
plt.plot(np.linspace(0, 1, 1001), V, label='volatility')
plt.xlabel('t')
plt.title('Sample path of split kernel implementation')
plt.show()

time.sleep(360000)

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
