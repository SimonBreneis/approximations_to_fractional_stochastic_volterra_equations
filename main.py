import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import ComputationalFinance as cf
import Data
import rBergomi
import rBergomiMarkov
import rHeston
import rHestonMarkov
import RoughKernel as rk
import rHestonMarkovSamplePaths as rHestonSP
import rHestonSplitKernel as sk
from numpy import nan
from os.path import exists


c = ['r', 'C1', 'y', 'g', 'b', 'purple']
c_ = ['darkred', 'r', 'C1', 'y', 'lime', 'g', 'deepskyblue', 'b', 'purple', 'deeppink']

k_vec = np.linspace(-1.5, 0.75, 451)[200:-70]
# k_vec = np.linspace(-0.5, 0.1, 181)
S, K, H, lambda_, rho, nu, theta, V_0, rel_tol = 1., np.exp(k_vec), 0.1, 0.3, -0.7, 0.3, 0.02, 0.02, 1e-05
# S, K, H, lambda_, rho, nu, theta, V_0, rel_tol = 1., np.exp(k_vec), 0.07, 0.6, -0.8, 0.5, 0.01, 0.01, 1e-05
# T = np.linspace(0.04, 1, 25)
T = 1
# T = 0.04

true_surface = Data.true_iv_surface_eur_call

nodes, weights = rk.quadrature_rule(H=H, N=10, T=1., mode='optimized')
print(nodes)
print(np.sqrt(rk.error(H=H, nodes=nodes, weights=weights, T=T)) / rk.kernel_norm(H=H, T=1))
print(rk.quadrature_rule(H=H, T=np.linspace(0.04, 1, 25), N=10, mode='optimized'))

'''
# print(rk.quadrature_rule(H=H, N=10, T=1., mode='european'))
print(rk.quadrature_rule(H=H, N=10, T=1., mode='european', optimal_weights=True))

time.sleep(360000)
'''
for N in np.array([1, 2, 3, 4, 5, 6, 7]):
    pass
    '''
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=0.52, mode='observation')
    print(N, 'observation', np.amax(nodes))
    approx_surface = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(np.amax(np.abs(true_surface - approx_surface) / true_surface))

    nodes, weights = rk.quadrature_rule(H=H, N=N, T=0.52, mode='optimized')
    print(N, 'optimized T=0.52', np.amax(nodes))
    approx_surface = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(np.amax(np.abs(true_surface - approx_surface) / true_surface))

    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='optimized')
    print(N, 'optimized, T=T', np.amax(nodes))
    approx_surface = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(np.amax(np.abs(true_surface - approx_surface) / true_surface))

    nodes, weights = rk.quadrature_rule(H=H, N=N, T=0.52, mode='european')
    print(N, 'european, T=0.52', np.amax(nodes))
    approx_surface = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(np.amax(np.abs(true_surface - approx_surface) / true_surface))
    
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='european')
    print(N, 'european T=T', np.amax(nodes))
    approx_surface = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(np.amax(np.abs(true_surface - approx_surface) / true_surface))

    nodes, weights = rk.quadrature_rule(H=H, N=N, T=0.52, mode='european', optimal_weights=True)
    print(N, 'european, T=0.52 new', np.amax(nodes))
    approx_surface = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(np.amax(np.abs(true_surface - approx_surface) / true_surface))
    
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='european', optimal_weights=True)
    print(N, 'european, T=T new', np.amax(nodes))
    approx_surface = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, rel_tol=rel_tol, nodes=nodes, weights=weights)
    print(np.amax(np.abs(true_surface - approx_surface) / true_surface))
    '''

# time.sleep(36000)
'''
true_smile = rHeston.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol)
print((true_smile,))

# true_smile = Data.true_iv_surface_eur_call[-1, :]
for N in np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    print('\nobservation', N)
    tic = time.perf_counter()
    smile = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, rel_tol=rel_tol, T=T, N=N, mode='observation')
    print(time.perf_counter() - tic)
    print(np.amax(np.abs(smile-true_smile)/true_smile))


    print('\noptimized', N)
    tic = time.perf_counter()
    smile = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, rel_tol=rel_tol, T=T, N=N, mode='optimized')
    print(time.perf_counter() - tic)
    print(np.amax(np.abs(smile-true_smile)/true_smile))


    print('\neuropean', N)
    tic = time.perf_counter()
    smile = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, rel_tol=rel_tol, T=T, N=N, mode='european')
    print(time.perf_counter() - tic)
    print(np.amax(np.abs(smile-true_smile)/true_smile))

    print('\neuro new', N)
    tic = time.perf_counter()
    smile = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, rel_tol=rel_tol, T=T, N=N, mode='euro new')
    print(time.perf_counter() - tic)
    print(np.amax(np.abs(smile-true_smile)/true_smile))

# rk.optimize_error(H, 512, T, iterative=True)
# rk.optimize_error_optimal_weights(H, 512, T, iterative=True, method='gradient')
# rk.optimize_error_optimal_weights(H, 512, T, iterative=True, method='error')
rk.optimize_error_optimal_weights(H, 512, T, iterative=True, method='hessian')
time.sleep(360000)
'''
'''
H, T = 0.1, 1.
for N in np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024]):
    tic = time.perf_counter()
    nodes, weights = rk.quadrature_rule(H, N, T, 'observation')
    duration = time.perf_counter() - tic
    error = np.sqrt(rk.error(H, nodes, weights, T)) / rk.kernel_norm(H, T)
    largest_node = np.amax(nodes)
    print('observation', N, largest_node, error, duration)

    tic = time.perf_counter()
    error, nodes, _ = rk.optimize_error(H, N, T)
    duration = time.perf_counter() - tic
    largest_node = np.amax(nodes)
    print('optimized', N, largest_node, error, duration)
    
    tic = time.perf_counter()
    error, nodes, _ = rk.optimize_error_optimal_weights(H, N, T, method='error')
    duration = time.perf_counter() - tic
    largest_node = np.amax(nodes)
    print('error', N, largest_node, error, duration)
    
    tic = time.perf_counter()
    error, nodes, _ = rk.optimize_error_optimal_weights(H, N, T, method='gradient')
    duration = time.perf_counter() - tic
    largest_node = np.amax(nodes)
    print('gradient', N, largest_node, error, duration)

    tic = time.perf_counter()
    error, nodes, _ = rk.optimize_error_optimal_weights(H, N, T, method='hessian')
    duration = time.perf_counter() - tic
    largest_node = np.amax(nodes)
    print('hessian', N, largest_node, error, duration)

time.sleep(360000)
'''

'''
tic = time.perf_counter()
S_, V_, V_components = nv.get_sample_path(H=0.49, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=2, S_0=S, N_time=1000000, mode='european')
print(time.perf_counter() - tic)
plt.plot(np.linspace(0, 1, 1000001), S_)
plt.plot(np.linspace(0, 1, 1000001), np.sqrt(V_))
plt.plot(np.linspace(0, 1, 1000001), V_components[0, :])
plt.plot(np.linspace(0, 1, 1000001), V_components[1, :])
plt.show()
'''

'''
vol_behaviour = 'hyperplane reset'
mode = 'european'
N = 1


final_S = np.empty(1000000)

truth = Data.true_iv_surface_eur_call[-1, 200:-70]
markov_truth = rHestonMarkov.iv_eur_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, mode=mode, rel_tol=rel_tol)

for i in range(18):
    N_time = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])[i]
    with open(f'rHeston samples {N} dim {mode} {vol_behaviour} {N_time} time steps.npy', 'rb') as f:
        final_S = np.load(f)

    vol, lower, upper = cf.iv_eur_call_MC(S=S, K=K, T=T, samples=final_S)
    plt.plot(k_vec, vol, color=c_[i % 10], label=f'N_time = {N_time}')
    if i == 17:
        plt.plot(k_vec, lower, '--', color=c_[i % 10])
        plt.plot(k_vec, upper, '--', color=c_[i % 10])
    print(N_time, 'truth', np.amax(np.abs(truth - vol)/truth))
    print(N_time, 'discretization', np.amax(np.abs(markov_truth - vol)/markov_truth))
    print(N_time, 'Markov', np.amax(np.abs(truth - markov_truth)/truth))

plt.plot(k_vec, markov_truth, color='grey', label='Exact Markovian smile')
plt.plot(k_vec, truth, color='k', label='Exact smile')
plt.legend(loc='upper right')
plt.xlabel('Log-moneyness')
plt.ylabel('Implied volatility')
plt.title(f'Rough Heston {N}-dimensional approximation with\nimplicit Euler and hyperplane reset')
plt.show()
'''

final_S = np.empty(1000000)
modes = ['european', 'optimized', 'observation']
vol_behaviours = ['hyperplane reset', 'ninomiya victoir']  # , 'sticky', 'hyperplane reflection', 'adaptive']

for N in np.array([1, 2, 6, 4, 3, 5]):
    for N_time in np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]):
        print(N_time)
        for vol_behaviour in vol_behaviours:
            print(N_time, vol_behaviour)
            for mode in modes:
                print(N_time, vol_behaviour, mode)
                filename = f'rHeston samples {N} dim {mode} {vol_behaviour} {N_time} time steps.npy'
                if exists(filename):
                    pass
                else:
                    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode=mode, optimal_weights=True)

                    for i in range(10):
                        print(N_time, vol_behaviour, mode, i)
                        with open(f'dW{i}.npy', 'rb') as f:
                            dW = np.load(f)
                        with open(f'dB{i}.npy', 'rb') as f:
                            dB = np.load(f)

                        WB_1 = np.empty((2, 100000, 2048))
                        WB_1[0, :, :] = dW
                        WB_1[1, :, :] = dB

                        n_samples = 100000
                        if N_time == 8192:
                            n_samples = 25000
                        elif N_time == 16384:
                            n_samples = 12500
                        elif N_time == 32768:
                            n_samples = 6250
                        elif N_time == 65536:
                            n_samples = 3125
                        elif N_time == 131072:
                            n_samples = 2000

                        WB = np.empty((2, n_samples, N_time))
                        for k in range(100000 // n_samples):
                            print(N_time, vol_behaviour, mode, i, k)
                            if N_time <= 256:
                                WB = WB_1[:, :, ::8] + WB_1[:, :, 1::8] + WB_1[:, :, 2::8] + WB_1[:, :, 3::8] + WB_1[:, :, 4::8] + WB_1[:, :, 5::8] + WB_1[:, :, 6::8] + WB_1[:, :, 7::8]
                            if N_time <= 128:
                                WB = WB[:, :, ::2] + WB[:, :, 1::2]
                            if N_time <= 64:
                                WB = WB[:, :, ::2] + WB[:, :, 1::2]
                            if N_time <= 32:
                                WB = WB[:, :, ::2] + WB[:, :, 1::2]
                            if N_time <= 16:
                                WB = WB[:, :, ::2] + WB[:, :, 1::2]
                            if N_time <= 8:
                                WB = WB[:, :, ::2] + WB[:, :, 1::2]
                            if N_time <= 4:
                                WB = WB[:, :, ::2] + WB[:, :, 1::2]
                            if N_time <= 2:
                                WB = WB[:, :, ::2] + WB[:, :, 1::2]
                            if N_time <= 1:
                                WB = WB[:, :, ::2] + WB[:, :, 1::2]
                            if N_time == 512:
                                WB = WB_1[:, :, ::4] + WB_1[:, :, 1::4] + WB_1[:, :, 2::4] + WB_1[:, :, 3::4]
                            if N_time == 1024:
                                WB = WB_1[:, :, ::2] + WB_1[:, :, 1::2]
                            if N_time == 2048:
                                WB = WB_1
                            if N_time == 4096:
                                incr = np.random.normal(0, 1/np.sqrt(8192), (2, 100000, 2048))
                                WB[:, :, 1::2] = WB_1/2 + incr
                                WB[:, :, ::2] = WB_1/2 - incr
                            if N_time >= 8192:
                                WB = np.empty((2, n_samples, 8192))
                                WB_1_ = WB_1[:, k*n_samples:(k+1)*n_samples, :]
                                incr_1 = np.random.normal(0, 1/np.sqrt(8192), (2, n_samples, 2048))
                                incr_2 = np.random.normal(0, 1/np.sqrt(16384), (2, n_samples, 2048))
                                incr_3 = np.random.normal(0, 1/np.sqrt(16384), (2, n_samples, 2048))
                                WB[:, :, ::4] = WB_1_/4 + incr_1/2 + incr_2
                                WB[:, :, 1::4] = WB_1_/4 + incr_1/2 - incr_2
                                WB[:, :, 2::4] = WB_1_/4 - incr_1/2 + incr_3
                                WB[:, :, 3::4] = WB_1_/4 - incr_1/2 - incr_3
                            if N_time >= 16384:
                                WB_ = np.empty((2, n_samples, 16384))
                                incr = np.random.normal(0, 1 / np.sqrt(32768), (2, n_samples, 8192))
                                WB_[:, :, ::2] = WB / 2 + incr
                                WB_[:, :, 1::2] = WB / 2 - incr
                                WB = WB_
                            if N_time >= 32768:
                                WB_ = np.empty((2, n_samples, 32768))
                                incr = np.random.normal(0, 1 / np.sqrt(65536), (2, n_samples, 16384))
                                WB_[:, :, ::2] = WB / 2 + incr
                                WB_[:, :, 1::2] = WB / 2 - incr
                                WB = WB_
                            if N_time >= 65536:
                                WB_ = np.empty((2, n_samples, 65536))
                                incr = np.random.normal(0, 1 / np.sqrt(131072), (2, n_samples, 32768))
                                WB_[:, :, ::2] = WB / 2 + incr
                                WB_[:, :, 1::2] = WB / 2 - incr
                                WB = WB_
                            if N_time >= 131072:
                                WB_ = np.empty((2, n_samples, 131072))
                                incr = np.random.normal(0, 1 / np.sqrt(262144), (2, n_samples, 65536))
                                WB_[:, :, ::2] = WB / 2 + incr
                                WB_[:, :, 1::2] = WB / 2 - incr
                                WB = WB_

                            final_S[i*100000 + k*n_samples:i*100000 + (k+1)*n_samples] = rHestonSP.sample_values(H=H, lambda_=lambda_,
                                                                                                                 rho=rho, nu=nu,
                                                                                                                 theta=theta, V_0=V_0,
                                                                                                                 T=T, N=N, S_0=S,
                                                                                                                 N_time=N_time, WB=WB,
                                                                                                                 m=n_samples, mode=mode,
                                                                                                                 vol_behaviour=vol_behaviour,
                                                                                                                 nodes=nodes,
                                                                                                                 weights=weights)

                    with open(filename, 'wb') as f:
                        np.save(f, final_S)
time.sleep(360000)
'''

with open(f'dW0.npy', 'rb') as f:
    dW = np.load(f)
with open(f'dB0.npy', 'rb') as f:
    dB = np.load(f)

WB = np.empty((2, 2, 1048576))
WB_1 = np.empty((2, 2, 2097152))

for i in range(1024):
    WB_1[0, :, i*2048:(i+1)*2048] = dW[2*i:2*i+1, :]
    WB_1[1, :, i*2048:(i+1)*2048] = dB[2*i:2*i+1, :]
WB_1 = WB_1 / 32

WB = WB_1[:, :, ::2] + WB_1[:, :, 1::2]

S_1, V_1, _ = nv.sample_func(H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=2, S_0=S, N_time=1048576, WB=WB, m=1, mode=mode, sample_paths=True)
S_2, V_2, _ = rHestonSP.sample_values(H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=2, S_0=S, N_time=1048576, WB=WB[:, :1, :], mode=mode, vol_behaviour='ninomiya victoir', sample_paths=True)

plt.plot(np.linspace(0, 1, 1048577), S_1[0, :], label='stock price 2097152 time steps')
plt.plot(np.linspace(0, 1, 1048577), V_1[0, :], label='volatility 2097152 time steps')
plt.plot(np.linspace(0, 1, 1048577), S_2[0, :], label='stock price 1048576 time steps')
plt.plot(np.linspace(0, 1, 1048577), V_2[0, :], label='volatility 1048576 time steps')
plt.legend(loc='best')
plt.show()
'''

'''
WB = np.empty((2, m, 256))
WB_1 = np.empty((2, m, 2048))
WB_1[0, :, :] = dW[:m, :]
WB_1[1, :, :] = dB[:m, :]
for i in range(256):
    WB[:, :, i] = np.sum(WB_1[:, :, 2 * i:2*(i+1)], axis=-1)
'''
'''
WB = np.empty((2, m, 8192))
WB[0, :, :2048] = dW[:m, :]
WB[1, :, :2048] = dB[:m, :]
with open('dW1.npy', 'rb') as f:
    dW = np.load(f)
with open('dB1.npy', 'rb') as f:
    dB = np.load(f)
WB[0, :, 2048:4096] = dW[:m, :]
WB[1, :, 2048:4096] = dB[:m, :]
with open('dW2.npy', 'rb') as f:
    dW = np.load(f)
with open('dB2.npy', 'rb') as f:
    dB = np.load(f)
WB[0, :, 4096:6144] = dW[:m, :]
WB[1, :, 4096:6144] = dB[:m, :]
with open('dW3.npy', 'rb') as f:
    dW = np.load(f)
with open('dB3.npy', 'rb') as f:
    dB = np.load(f)
WB[0, :, 6144:] = dW[:m, :]
WB[1, :, 6144:] = dB[:m, :]

WB = WB / 2
'''

'''
truth = Data.true_iv_surface_eur_call[-1, 200:-70]
vols = np.empty((3, 6, 181))
lowers = np.empty((3, 6, 181))
uppers = np.empty((3, 6, 181))

for N in np.array([2, 3, 4, 5, 6]):
    v, l, u = rHestonSP.call(K=np.exp(k_vec), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, H=H, N=N, S_0=S, T=T,
                             m=m, N_time=2048, WB=WB, mode='observation', vol_behaviour=vol_behaviour)
    vols[0, N - 1, :] = v
    lowers[0, N - 1, :] = l
    uppers[0, N - 1, :] = u
    print(N, 'observation', np.amax(np.abs(v-truth)/truth))
    plt.plot(k_vec, v, '-', color='red', label='Paper')
    plt.plot(k_vec, rHestonMarkov.iv_eur_call(S=S, K=np.exp(k_vec), H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, mode='observation', rel_tol=rel_tol), '--', color='red')
    v, l, u = rHestonSP.call(K=np.exp(k_vec), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, H=H, N=N, S_0=S, T=T,
                             m=m, N_time=2048, WB=WB, mode='optimized', vol_behaviour=vol_behaviour)
    vols[1, N - 1, :] = v
    lowers[1, N - 1, :] = l
    uppers[1, N - 1, :] = u
    print(N, 'optimized', np.amax(np.abs(v-truth)/truth))
    plt.plot(k_vec, v, '-', color='green', label='Kernel')
    plt.plot(k_vec, rHestonMarkov.iv_eur_call(S=S, K=np.exp(k_vec), H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, mode='optimized', rel_tol=rel_tol), '--', color='green')
    v, l, u = rHestonSP.call(K=np.exp(k_vec), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, H=H, N=N, S_0=S, T=T,
                             m=m, N_time=2048, WB=WB, mode='european', vol_behaviour=vol_behaviour)
    vols[2, N - 1, :] = v
    lowers[2, N - 1, :] = l
    uppers[2, N - 1, :] = u
    print(N, 'european', np.amax(np.abs(v-truth)/truth))
    plt.plot(k_vec, v, '-', color='blue', label='European')
    # plt.plot(k_vec, l, '--', color='blue')
    # plt.plot(k_vec, u, '--', color='blue')
    plt.plot(k_vec, rHestonMarkov.iv_eur_call(S=S, K=np.exp(k_vec), H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, mode='european', rel_tol=rel_tol), '--', color='blue')
    plt.plot(k_vec, truth, color='k', label='True rough smile')
    plt.xlabel('Log-moneyness')
    plt.ylabel('Implied volatility')
    plt.legend(loc='upper right')
    plt.title(f'Implied volatility for European call option\nwith {N} nodes and 2048 time steps sticky')
    plt.show()

print((vols, lowers, uppers))
time.sleep(360000)

# vol, lower, upper = ie.call(K=np.exp(k_vec), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, H=H, N=6, S_0=1., T=1, m=m, N_time=2048, WB=WB, mode='observation', vol_behaviour=vol_behaviour)
# vol_2, lower_2, upper_2 = ie.call(K=np.exp(k_vec), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, H=H, N=6, S_0=1., T=1, m=m, N_time=2048, WB=WB, mode='optimized', vol_behaviour=vol_behaviour)
vol_3_2, lower_3, upper_3 = rHestonSP.call(K=np.exp(k_vec), lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, H=H, N=6,
                                           S_0=1., T=1, m=m, N_time=2048, WB=WB, mode='european', vol_behaviour=vol_behaviour)

# print((vol, lower, upper))
# print((vol_2, lower_2, upper_2))
# print((vol_3_2, lower_3, upper_3))

vol = np.array([0.55047113, 0.54873696, 0.54700486, 0.54527302, 0.54354375,
                0.54181999, 0.54009645, 0.53837314, 0.53665106, 0.53493097,
                0.53321335, 0.53149593, 0.52977869, 0.52806199, 0.52634764,
                0.52463346, 0.52292086, 0.52120953, 0.51949834, 0.51778973,
                0.51608176, 0.51437603, 0.51267169, 0.51097282, 0.50927498,
                0.5075772, 0.50588184, 0.50418973, 0.50250016, 0.5008152,
                0.49913021, 0.49744814, 0.49577318, 0.4941018, 0.49243342,
                0.49076688, 0.48910465, 0.48744621, 0.48578969, 0.48413482,
                0.48248487, 0.48084003, 0.4791995, 0.47756185, 0.47592585,
                0.47429298, 0.47266299, 0.47103819, 0.4694155, 0.46779689,
                0.4661817, 0.46457062, 0.46296043, 0.46135452, 0.45975491,
                0.45816094, 0.45657138, 0.45498523, 0.45340224, 0.4518226,
                0.45025095, 0.44868028, 0.44711212, 0.44554524, 0.44398285,
                0.44242368, 0.44086357, 0.43930479, 0.43774839, 0.43619812,
                0.43465045, 0.43310826, 0.43157189, 0.43003767, 0.42850514,
                0.42697941, 0.42546396, 0.42395602, 0.42245157, 0.4209487,
                0.41945072, 0.41795677, 0.41646794, 0.41498334, 0.41350347,
                0.41202899, 0.41055905, 0.40909998, 0.40764496, 0.40619484,
                0.40474849, 0.40330835, 0.40187403, 0.40045365, 0.39903689,
                0.3976222, 0.39621066, 0.39480276, 0.39339427, 0.39198858,
                0.3905889, 0.38919574, 0.387814, 0.38643175, 0.3850555,
                0.38368639, 0.38232518, 0.38096942, 0.37961629, 0.37826639,
                0.37692324, 0.37558217, 0.37424678, 0.37291951, 0.37159384,
                0.3702693, 0.368946, 0.36762679, 0.36631067, 0.36499594,
                0.36368549, 0.3623854, 0.36109041, 0.35979543, 0.35850337,
                0.35721552, 0.35593021, 0.35466047, 0.35340234, 0.35215589,
                0.35091062, 0.34966368, 0.34841688, 0.34717578, 0.3459451,
                0.34472031, 0.34350152, 0.34228773, 0.34107517, 0.33985882,
                0.33863841, 0.33741791, 0.33620324, 0.3349891, 0.33378489,
                0.33259164, 0.33140625, 0.33022792, 0.32905567, 0.3278915,
                0.32673958, 0.32559744, 0.32445483, 0.32331685, 0.32218342,
                0.32105288, 0.31992608, 0.31880361, 0.31768444, 0.3165715,
                0.31545357, 0.31433512, 0.31321616, 0.31210124, 0.31098907,
                0.30987312, 0.30875977, 0.30764739, 0.30653336, 0.30541694,
                0.30430372, 0.30319338, 0.30208579, 0.30098408, 0.29988206,
                0.29877972, 0.29768092, 0.29658464, 0.29548987, 0.29439326,
                0.29329552, 0.29220009, 0.29110599, 0.29001317, 0.28892162,
                0.28782966, 0.2867397, 0.28565068, 0.28455844, 0.28346637,
                0.28237079, 0.2812791, 0.28018855, 0.27909964, 0.27800773,
                0.27691772, 0.27582928, 0.27475095, 0.2736732, 0.27259613,
                0.27151831, 0.27043994, 0.26935881, 0.26827274, 0.26718571,
                0.26610214, 0.26501247, 0.26392143, 0.26283225, 0.26174061,
                0.26064664, 0.25955351, 0.25846189, 0.25737106, 0.25628129,
                0.25519348, 0.2541027, 0.25300737, 0.25190866, 0.25080752,
                0.24970633, 0.24860023, 0.24748849, 0.24637296, 0.245253,
                0.24413408, 0.2430127, 0.24188546, 0.24075166, 0.23961728,
                0.23847747, 0.23733287, 0.23618191, 0.23502704, 0.23386953,
                0.23270669, 0.23153742, 0.23036483, 0.22918335, 0.22799666,
                0.22680672, 0.2256132, 0.22441771, 0.22321641, 0.22201189,
                0.2208042, 0.21958856, 0.218368, 0.21714239, 0.21591108,
                0.21467678, 0.21343739, 0.21219208, 0.21094085, 0.20968418,
                0.20842656, 0.20716516, 0.20589316, 0.20461389, 0.20332542,
                0.20202883, 0.20072419, 0.19941588, 0.1981051, 0.19678307,
                0.19545382, 0.1941204, 0.19277941, 0.19143337, 0.19008142,
                0.18872299, 0.18735713, 0.18598221, 0.1846027, 0.18321639,
                0.18182471, 0.18042493, 0.1790143, 0.17759983, 0.17618268,
                0.17475725, 0.17332583, 0.1718835, 0.1704385, 0.16898881,
                0.167535, 0.16607782, 0.16461773, 0.16315027, 0.16167457,
                0.16019255, 0.15870293, 0.15720973, 0.15571854, 0.15422476,
                0.1527264, 0.15122643, 0.14972356, 0.14822689, 0.14673087,
                0.14523501, 0.14374262, 0.14225476, 0.14077739, 0.13931147,
                0.13786293, 0.13642333, 0.13499862, 0.13360022, 0.13223124,
                0.13088903, 0.12957418, 0.12829411, 0.12705034, 0.12584725,
                0.12468907, 0.12357753, 0.12251062, 0.12148866, 0.12052638,
                0.11962416, 0.1187808, 0.11799576, 0.11726989, 0.11660637,
                0.1159823, 0.11541018, 0.11488704, 0.11442956, 0.1140269,
                0.11366995, 0.11335678, 0.11308251, 0.11285442, 0.11267774,
                0.11254834, 0.1124615, 0.11242429, 0.11241989, 0.11244419,
                0.11249842, 0.11258392, 0.1126984, 0.11284058, 0.11300607,
                0.11320345, 0.1134167, 0.11363733, 0.11388229, 0.11414002,
                0.11440357, 0.11466931, 0.11493231, 0.11516539, 0.11542221,
                0.1156984, 0.1160112, 0.11633125, 0.11665904, 0.11700481,
                0.11740423, 0.11783833, 0.11826157, 0.11869859, 0.11913779,
                0.11959875, 0.12004543, 0.12049344, 0.12095965, 0.12142775,
                0.12191049, 0.12245508, 0.12302626, 0.12363408, 0.12424668,
                0.12487184, 0.12550131, 0.12611627, 0.12672905, 0.12733594,
                0.12800737, 0.12865583, 0.12933483, 0.12999556, 0.13062962,
                0.13132158, 0.13201409, 0.13265665, 0.13323125, 0.13379332,
                0.13429054, 0.13474569, 0.13526066, 0.13574231, 0.13613358,
                0.13649521, 0.13699249, 0.13752692, 0.13798785, 0.13846326,
                0.13894461, 0.13934193, 0.13982326, 0.14029649, 0.14071897,
                0.14117858, 0.14155413, 0.14182874, 0.14220078, 0.14247618,
                0.14262913, 0.14273893, 0.14291623, 0.1429462, 0.14299863,
                0.14316094, 0.14317253, 0.14298425, 0.14251583, 0.14162111,
                0.13998379, 0.13912462, 0.1375963, 0.13436912, 0.02067377,
                0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                0.0250104])
lower = np.array([np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, 0.17719288, 0.21838372, 0.22900968, 0.23559619,
                  0.24028075, 0.24387674, 0.24674737, 0.24911368, 0.25109181,
                  0.25274177, 0.25414578, 0.2553407, 0.25635159, 0.25720279,
                  0.25792915, 0.25854502, 0.25906258, 0.25949918, 0.25984985,
                  0.26012279, 0.26033269, 0.26048304, 0.26057683, 0.26061243,
                  0.26059551, 0.26053578, 0.2604346, 0.26029463, 0.26011839,
                  0.25990547, 0.25966182, 0.25938764, 0.25907836, 0.25874093,
                  0.25837143, 0.25798219, 0.25757032, 0.25713766, 0.25667874,
                  0.25620151, 0.25570636, 0.25520559, 0.25468693, 0.25415129,
                  0.25359755, 0.2530267, 0.25243661, 0.25182522, 0.25119828,
                  0.25056188, 0.24990461, 0.24923295, 0.2485513, 0.24785485,
                  0.24714421, 0.24642362, 0.24569422, 0.24495549, 0.24420807,
                  0.24345328, 0.24268574, 0.24190395, 0.24110964, 0.24030419,
                  0.2394906, 0.23866364, 0.2378228, 0.23697044, 0.2361061,
                  0.23523615, 0.23435686, 0.23346471, 0.23255919, 0.23164707,
                  0.23072324, 0.2297886, 0.22884164, 0.22788522, 0.22692087,
                  0.22594585, 0.22495917, 0.2239643, 0.22295547, 0.22193676,
                  0.22091038, 0.21987612, 0.21883579, 0.21778545, 0.21672797,
                  0.21566348, 0.2145871, 0.21350212, 0.21240848, 0.21130565,
                  0.21019651, 0.209079, 0.20795232, 0.20681657, 0.20567234,
                  0.20452436, 0.20336975, 0.20220158, 0.20102335, 0.19983317,
                  0.19863221, 0.19742064, 0.19620301, 0.19498061, 0.19374452,
                  0.19249891, 0.19124696, 0.18998526, 0.18871644, 0.18743966,
                  0.1861544, 0.18485974, 0.18355406, 0.18224198, 0.18092127,
                  0.17959342, 0.17825573, 0.17690543, 0.17554967, 0.17418966,
                  0.17281976, 0.17144231, 0.17005239, 0.16865834, 0.16725816,
                  0.16585244, 0.16444197, 0.16302721, 0.16160372, 0.16017063,
                  0.15872988, 0.15728023, 0.15582571, 0.15437198, 0.15291441,
                  0.15145102, 0.14998479, 0.14851445, 0.14704912, 0.14558324,
                  0.14411632, 0.14265167, 0.14119034, 0.13973833, 0.13829657,
                  0.13687101, 0.13545316, 0.13404896, 0.13266986, 0.13131896,
                  0.12999358, 0.12869428, 0.12742848, 0.12619766, 0.12500622,
                  0.12385837, 0.12275581, 0.12169649, 0.1206807, 0.11972323,
                  0.11882446, 0.11798314, 0.11719875, 0.11647212, 0.11580647,
                  0.11517868, 0.11460131, 0.11407135, 0.11360568, 0.11319334,
                  0.11282513, 0.11249905, 0.11221012, 0.11196572, 0.11177119,
                  0.11162239, 0.11151451, 0.11145485, 0.11142623, 0.11142443,
                  0.11145069, 0.11150638, 0.11158916, 0.11169774, 0.11182751,
                  0.1119875, 0.11216096, 0.1123389, 0.11253901, 0.11274908,
                  0.11296161, 0.11317242, 0.11337587, 0.11354199, 0.11372789,
                  0.11392872, 0.11416335, 0.11439955, 0.11463735, 0.11488781,
                  0.1151914, 0.11552761, 0.11584433, 0.1161693, 0.11648858,
                  0.11682509, 0.11713523, 0.11743649, 0.1177489, 0.11805174,
                  0.11836038, 0.11873665, 0.11913742, 0.11957793, 0.12001651,
                  0.12046437, 0.12091034, 0.12132703, 0.12173089, 0.12211556,
                  0.12259034, 0.12302339, 0.12349983, 0.12394277, 0.12433541,
                  0.12482396, 0.12531591, 0.12571976, 0.12599285, 0.12623035,
                  0.12632013, 0.12628672, 0.12634833, 0.12629782, 0.12595318,
                  0.12540207, 0.12514838, 0.1249556, 0.12439039, 0.12372595,
                  0.12289795, 0.12111732, 0.11928495, 0.11612314, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, 0.02067377,
                  0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                  0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                  0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                  0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                  0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                  0.0250104])
upper = np.array([0.66096091, 0.65882631, 0.65669299, 0.65456038, 0.65242921,
                  0.6503004, 0.64817229, 0.64604486, 0.64391844, 0.64179326,
                  0.63966949, 0.63754637, 0.63542393, 0.63330225, 0.63118193,
                  0.62906226, 0.62694368, 0.62482611, 0.62270917, 0.62059365,
                  0.61847891, 0.61636548, 0.61425308, 0.61214302, 0.61003387,
                  0.60792532, 0.60581813, 0.60371256, 0.60160839, 0.59950629,
                  0.59740475, 0.59530474, 0.59320761, 0.59111222, 0.5890184,
                  0.58692575, 0.58483508, 0.58274624, 0.5806586, 0.57857209,
                  0.57648777, 0.57440572, 0.57232569, 0.5702472, 0.56816984,
                  0.56609411, 0.56401993, 0.56194808, 0.55987753, 0.55780894,
                  0.55574211, 0.55367727, 0.55161333, 0.54955146, 0.54749234,
                  0.54543577, 0.54338134, 0.54132871, 0.5392778, 0.5372287,
                  0.53518302, 0.53313833, 0.53109519, 0.52905315, 0.52701336,
                  0.52497537, 0.52293773, 0.52090124, 0.51886627, 0.51683418,
                  0.51480372, 0.51277593, 0.51075096, 0.50872751, 0.5067054,
                  0.50468652, 0.50267218, 0.50066139, 0.49865271, 0.49664543,
                  0.4946408, 0.49263854, 0.49063906, 0.48864206, 0.48664775,
                  0.48465641, 0.48266774, 0.48068422, 0.47870326, 0.47672522,
                  0.4747497, 0.47277768, 0.47080904, 0.46884706, 0.4668877,
                  0.46493037, 0.46297554, 0.46102342, 0.45907232, 0.45712364,
                  0.45517873, 0.45323785, 0.45130312, 0.4493696, 0.44744005,
                  0.44551502, 0.4435949, 0.44167872, 0.43976529, 0.43785493,
                  0.43594921, 0.43404614, 0.43214735, 0.43025402, 0.42836327,
                  0.42647493, 0.42458908, 0.42270704, 0.42082842, 0.41895247,
                  0.41708058, 0.41521568, 0.41335541, 0.41149736, 0.40964298,
                  0.40779295, 0.4059465, 0.40411014, 0.4022821, 0.40046254,
                  0.39864631, 0.396832, 0.39502054, 0.39321484, 0.39141746,
                  0.38962616, 0.38784111, 0.38606186, 0.38428648, 0.38251227,
                  0.38073908, 0.37896907, 0.37720555, 0.37544566, 0.37369474,
                  0.37195354, 0.37022045, 0.36849515, 0.36677718, 0.36506784,
                  0.36336975, 0.36168162, 0.35999746, 0.35832038, 0.35665044,
                  0.35498668, 0.35332973, 0.35168003, 0.35003702, 0.34840264,
                  0.34676981, 0.34514137, 0.34351734, 0.34190072, 0.34029075,
                  0.33868313, 0.33708211, 0.33548664, 0.33389498, 0.3323066,
                  0.33072535, 0.32915109, 0.32758376, 0.32602564, 0.32447242,
                  0.3229241, 0.32138346, 0.31984984, 0.31832252, 0.31679907,
                  0.31528001, 0.31376786, 0.31226194, 0.31076223, 0.30926874,
                  0.3077802, 0.30629844, 0.30482266, 0.30334965, 0.301882,
                  0.30041679, 0.29895984, 0.29750899, 0.29606466, 0.29462308,
                  0.29318821, 0.29175975, 0.2903447, 0.28893534, 0.28753171,
                  0.28613263, 0.28473821, 0.28334656, 0.2819558, 0.28056918,
                  0.27919046, 0.27781145, 0.27643611, 0.27506719, 0.27370092,
                  0.27233736, 0.27097922, 0.26962704, 0.26828019, 0.26693886,
                  0.2656038, 0.26427064, 0.26293788, 0.26160649, 0.26027724,
                  0.25895221, 0.25762698, 0.25630079, 0.25497523, 0.25364964,
                  0.25232897, 0.25100994, 0.24968938, 0.24836656, 0.24704691,
                  0.24572588, 0.24440401, 0.24307973, 0.24175527, 0.24043175,
                  0.23910657, 0.23777864, 0.2364508, 0.23511774, 0.23378287,
                  0.23244794, 0.23111261, 0.22977833, 0.22844142, 0.22710428,
                  0.22576688, 0.22442463, 0.22308033, 0.22173382, 0.2203844,
                  0.21903463, 0.21768246, 0.216327, 0.2149682, 0.21360648,
                  0.21224615, 0.2108844, 0.20951459, 0.20813987, 0.20675832,
                  0.20537093, 0.20397772, 0.20258291, 0.20118764, 0.1997833,
                  0.19837376, 0.19696197, 0.19554456, 0.19412397, 0.19269931,
                  0.19126998, 0.18983503, 0.18839278, 0.18694764, 0.18549737,
                  0.18404334, 0.18258286, 0.18111316, 0.17964113, 0.1781679,
                  0.1766879, 0.17520337, 0.17370942, 0.17221418, 0.17071562,
                  0.1692143, 0.16771093, 0.16620595, 0.16469493, 0.16317699,
                  0.16165401, 0.1601247, 0.15859304, 0.15706459, 0.15553475,
                  0.15400156, 0.15246794, 0.15093263, 0.14940467, 0.14787853,
                  0.14635374, 0.1448336, 0.14331917, 0.1418164, 0.14032625,
                  0.13885464, 0.13739321, 0.13594787, 0.13453003, 0.13314282,
                  0.13178361, 0.13045303, 0.1291585, 0.12790154, 0.12668656,
                  0.1255178, 0.12439699, 0.12332217, 0.1222937, 0.12132624,
                  0.12042017, 0.11957431, 0.11878813, 0.11806248, 0.11740051,
                  0.11677952, 0.11621194, 0.11569482, 0.11524468, 0.11485074,
                  0.114504, 0.11420257, 0.11394165, 0.11372843, 0.113568,
                  0.11345625, 0.11338852, 0.11337164, 0.1133891, 0.11343689,
                  0.11351621, 0.11362833, 0.11377099, 0.11394291, 0.11413982,
                  0.11436992, 0.11461776, 0.11487528, 0.11515871, 0.11545699,
                  0.11576365, 0.1160754, 0.11638783, 0.11667597, 0.11699045,
                  0.11732718, 0.11770196, 0.11808762, 0.11848484, 0.11890303,
                  0.11937404, 0.11988008, 0.12038029, 0.1208968, 0.12141946,
                  0.12196526, 0.12250306, 0.12304673, 0.12361072, 0.12418117,
                  0.12476878, 0.12541189, 0.12608003, 0.12678039, 0.12748654,
                  0.12820425, 0.12892664, 0.12963889, 0.13035077, 0.13105937,
                  0.13181758, 0.13255785, 0.13332002, 0.13406696, 0.13479293,
                  0.13555896, 0.13632158, 0.13704453, 0.13771645, 0.13837773,
                  0.13899258, 0.1395791, 0.140206, 0.14080889, 0.14135269,
                  0.14188067, 0.14249552, 0.14312916, 0.14371276, 0.14430112,
                  0.14488709, 0.14541778, 0.14599161, 0.14654982, 0.14706843,
                  0.14759421, 0.14805814, 0.14845361, 0.14888517, 0.14924413,
                  0.14951993, 0.14975752, 0.15000192, 0.15014308, 0.15026422,
                  0.15037811, 0.15034306, 0.15011991, 0.1496533, 0.14886554,
                  0.14766055, 0.14658733, 0.14471354, 0.14091269, 0.02067377,
                  0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                  0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                  0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                  0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                  0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                  0.0250104])
vol_2 = np.array([0.5578276, 0.55606766, 0.55430802, 0.55255054, 0.5507941,
                  0.54903991, 0.5472871, 0.54553885, 0.54379481, 0.54205311,
                  0.54031229, 0.53857451, 0.53683851, 0.53510769, 0.5333816,
                  0.5316586, 0.52993756, 0.52821794, 0.52649993, 0.52478416,
                  0.52307146, 0.5213602, 0.51965195, 0.51794818, 0.51624767,
                  0.51455066, 0.51285551, 0.5111615, 0.50947059, 0.50778102,
                  0.50609134, 0.50440591, 0.50272678, 0.50104839, 0.49937552,
                  0.49770554, 0.49603975, 0.49437683, 0.49272073, 0.49106794,
                  0.48941722, 0.48777358, 0.48613256, 0.48449676, 0.48286436,
                  0.48123613, 0.47961295, 0.47799521, 0.47638119, 0.47476792,
                  0.47315688, 0.47154673, 0.46994093, 0.46833909, 0.46674055,
                  0.46514627, 0.46356061, 0.46197719, 0.46039877, 0.45881991,
                  0.45724303, 0.4556721, 0.45410283, 0.4525419, 0.45098505,
                  0.44943078, 0.44787743, 0.4463237, 0.44477743, 0.4432405,
                  0.44170566, 0.44017367, 0.43864132, 0.43711246, 0.43558504,
                  0.43406001, 0.43253272, 0.43100691, 0.42948546, 0.42796753,
                  0.42645194, 0.42493543, 0.42341637, 0.42189657, 0.42037881,
                  0.41886613, 0.41735671, 0.4158511, 0.4143513, 0.4128547,
                  0.41136004, 0.40987246, 0.40838994, 0.40691185, 0.40544585,
                  0.403991, 0.40253748, 0.40108767, 0.39964451, 0.39820348,
                  0.3967723, 0.39534542, 0.39392666, 0.39251091, 0.39110149,
                  0.38970161, 0.38830737, 0.38691641, 0.38553473, 0.38415577,
                  0.38277602, 0.38139987, 0.38003417, 0.37867833, 0.37732617,
                  0.37598314, 0.37464594, 0.3733125, 0.37198426, 0.37065496,
                  0.36932928, 0.36800541, 0.36668636, 0.36537741, 0.36407115,
                  0.36277328, 0.36147931, 0.36018546, 0.35888961, 0.35759412,
                  0.3563077, 0.35502484, 0.35375098, 0.3524806, 0.3512123,
                  0.34994899, 0.34868741, 0.34742662, 0.34616661, 0.34491034,
                  0.34365846, 0.34241152, 0.34117194, 0.33993885, 0.33870978,
                  0.33749231, 0.33627893, 0.33507677, 0.33388214, 0.33269581,
                  0.33152174, 0.33035481, 0.32919352, 0.32803084, 0.32686793,
                  0.32570303, 0.32454385, 0.32339051, 0.32223973, 0.32109336,
                  0.31994606, 0.31879813, 0.3176547, 0.31651269, 0.31536926,
                  0.31422968, 0.31309133, 0.31196282, 0.31083431, 0.30970928,
                  0.30858012, 0.30746223, 0.30634822, 0.30523465, 0.30412294,
                  0.30301848, 0.30191402, 0.30080748, 0.29970081, 0.29859044,
                  0.29747792, 0.29636638, 0.29525895, 0.29415266, 0.29304383,
                  0.29193992, 0.29083464, 0.28972546, 0.28861582, 0.28750986,
                  0.28640921, 0.28530431, 0.28420086, 0.28309807, 0.28199289,
                  0.28088707, 0.27978386, 0.27867887, 0.27757736, 0.27647065,
                  0.27535872, 0.27424314, 0.27312965, 0.27201728, 0.27090275,
                  0.26978443, 0.26866466, 0.26754488, 0.266429, 0.26531274,
                  0.26419211, 0.2630736, 0.26195085, 0.26082811, 0.25970315,
                  0.25857441, 0.25743952, 0.25629844, 0.2551558, 0.25401841,
                  0.25287923, 0.25173886, 0.25059375, 0.24944682, 0.24829903,
                  0.24714759, 0.24599669, 0.24484277, 0.24369099, 0.2425392,
                  0.24137754, 0.24020949, 0.23904268, 0.23787077, 0.23669506,
                  0.23551155, 0.23432384, 0.23312776, 0.23192665, 0.23071921,
                  0.22950687, 0.22828942, 0.22706694, 0.22583945, 0.22460712,
                  0.22337389, 0.22213224, 0.2208808, 0.21962411, 0.21836035,
                  0.21709085, 0.21581312, 0.2145309, 0.21324556, 0.21195423,
                  0.21066342, 0.20936452, 0.20806128, 0.20675069, 0.2054295,
                  0.20410241, 0.20276624, 0.20142779, 0.20008166, 0.19872614,
                  0.19736805, 0.19600472, 0.19463538, 0.19326226, 0.1918823,
                  0.19049881, 0.18910696, 0.18770835, 0.18630323, 0.18489168,
                  0.18347241, 0.18204808, 0.1806152, 0.17917774, 0.1777387,
                  0.17629545, 0.17484358, 0.17338885, 0.17192723, 0.17045328,
                  0.1689741, 0.16749301, 0.16600935, 0.16452288, 0.16303206,
                  0.16153557, 0.16003582, 0.15854078, 0.15704486, 0.15554312,
                  0.15403796, 0.15253351, 0.15103426, 0.14954403, 0.14805493,
                  0.14656628, 0.14508099, 0.14360521, 0.14214155, 0.14069109,
                  0.13926184, 0.13784119, 0.13643615, 0.13505372, 0.13369241,
                  0.13235497, 0.13104569, 0.12976718, 0.12852125, 0.12730544,
                  0.12613629, 0.12500873, 0.12392071, 0.12287621, 0.12187946,
                  0.12094141, 0.12006121, 0.11923231, 0.1184666, 0.1177579,
                  0.11709869, 0.11648474, 0.1159299, 0.11543924, 0.11499974,
                  0.11460875, 0.11425183, 0.11393168, 0.11366882, 0.11346403,
                  0.11332108, 0.11321776, 0.11314986, 0.11314226, 0.11316849,
                  0.11323959, 0.11336748, 0.11352566, 0.11370753, 0.11392591,
                  0.11416391, 0.11440333, 0.11464106, 0.1148885, 0.11514837,
                  0.11543652, 0.1157291, 0.11604115, 0.11636968, 0.11671767,
                  0.11706902, 0.11743227, 0.11782833, 0.11824333, 0.11867355,
                  0.11911355, 0.11956943, 0.12003254, 0.12047678, 0.12093418,
                  0.12140294, 0.12186615, 0.12231876, 0.12280397, 0.12330529,
                  0.12382989, 0.12438487, 0.12494884, 0.12548612, 0.12604154,
                  0.12661541, 0.12716719, 0.12771934, 0.12827401, 0.12878528,
                  0.12930527, 0.12984633, 0.13032128, 0.13078448, 0.13129145,
                  0.13180596, 0.1323062, 0.13279351, 0.13335375, 0.13390737,
                  0.1344479, 0.13504255, 0.1356783, 0.13630187, 0.13686018,
                  0.13739528, 0.13794479, 0.13853946, 0.1391606, 0.13974116,
                  0.14032455, 0.14084223, 0.14128358, 0.14169131, 0.14207304,
                  0.14235396, 0.14251175, 0.14270111, 0.14274485, 0.14260033,
                  0.14220094, 0.14175217, 0.1412196, 0.1409737, 0.14042622,
                  0.13943622, 0.13870508, 0.13779858, 0.13736487, 0.13649093,
                  0.13481155, 0.1310477, 0.02034021, 0.02050699, 0.02067377,
                  0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                  0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                  0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                  0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                  0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                  0.0250104])
lower_2 = np.array([nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, nan, nan, nan,
                    np.nan, nan, 0.2193238, 0.232509, 0.23996739,
                    0.24513911, 0.24908382, 0.25223337, 0.25480617, 0.25695478,
                    0.25875313, 0.26027105, 0.26157403, 0.26268702, 0.26363002,
                    0.26443991, 0.26512664, 0.26572527, 0.26622257, 0.26663677,
                    0.26695972, 0.26723183, 0.26744265, 0.26759062, 0.26768356,
                    0.26773564, 0.26773685, 0.26768717, 0.26759346, 0.2674526,
                    0.26727013, 0.26705365, 0.26681016, 0.26653658, 0.26622896,
                    0.26590038, 0.26554264, 0.26515347, 0.26473937, 0.26430748,
                    0.26386095, 0.26338722, 0.26289529, 0.26238492, 0.26185279,
                    0.26130212, 0.26073796, 0.26015515, 0.2595612, 0.25894548,
                    0.2583087, 0.25765354, 0.25698781, 0.25631071, 0.2556186,
                    0.25491001, 0.25418824, 0.25345548, 0.25271675, 0.25196722,
                    0.25120252, 0.25043069, 0.24964454, 0.2488494, 0.24804294,
                    0.24722368, 0.24638922, 0.24553986, 0.24468116, 0.24382106,
                    0.24295178, 0.2420742, 0.24118457, 0.2402864, 0.23938092,
                    0.23846526, 0.23754421, 0.23661403, 0.23568055, 0.23474154,
                    0.23378644, 0.23281925, 0.23184838, 0.23086714, 0.22987709,
                    0.22887407, 0.22786212, 0.22683693, 0.2258022, 0.2247567,
                    0.22370208, 0.22263823, 0.22156535, 0.2204836, 0.21939323,
                    0.21829852, 0.21719164, 0.21607129, 0.21494233, 0.21380295,
                    0.21265463, 0.21149488, 0.21032765, 0.20915445, 0.2079724,
                    0.20678826, 0.20559324, 0.20439125, 0.20317927, 0.20195403,
                    0.20072044, 0.1994753, 0.19822565, 0.19696599, 0.19569465,
                    0.19441866, 0.19313534, 0.19184393, 0.19054678, 0.18924081,
                    0.18792943, 0.18660776, 0.1852775, 0.18393892, 0.18259215,
                    0.18123593, 0.17987298, 0.17849981, 0.17712046, 0.17573803,
                    0.17434985, 0.17295151, 0.17154886, 0.17013785, 0.16871301,
                    0.16728154, 0.16584682, 0.1644082, 0.16296545, 0.16151703,
                    0.16006163, 0.1586017, 0.15714529, 0.15568676, 0.15422114,
                    0.15275089, 0.15128013, 0.14981341, 0.14835455, 0.14689563,
                    0.14543597, 0.14397847, 0.14252932, 0.14109113, 0.13966496,
                    0.13825887, 0.13686015, 0.13547583, 0.13411293, 0.13276992,
                    0.13144953, 0.13015607, 0.1288921, 0.12765942, 0.12645552,
                    0.125297, 0.12417871, 0.12309857, 0.12206055, 0.12106885,
                    0.12013445, 0.11925651, 0.11842837, 0.11766203, 0.11695124,
                    0.11628837, 0.11566912, 0.11510747, 0.11460854, 0.11415916,
                    0.11375663, 0.11338628, 0.11305072, 0.11277081, 0.11254736,
                    0.11238432, 0.11225907, 0.11216727, 0.11213447, 0.1121336,
                    0.11217606, 0.11227428, 0.1124011, 0.11254969, 0.11273341,
                    0.11293479, 0.11313472, 0.11332967, 0.11353137, 0.11374247,
                    0.11397973, 0.11421776, 0.1144724, 0.11474044, 0.11502504,
                    0.11530868, 0.11560036, 0.11592304, 0.11626176, 0.11661246,
                    0.11696907, 0.11733848, 0.11771091, 0.118056, 0.11840998,
                    0.1187707, 0.11911806, 0.11944507, 0.11980315, 0.12017311,
                    0.12056431, 0.12098664, 0.12141427, 0.12180098, 0.12220374,
                    0.12262415, 0.12300804, 0.12338426, 0.12375589, 0.12405615,
                    0.12435637, 0.12467606, 0.1248809, 0.12504624, 0.12526023,
                    0.12546722, 0.12562734, 0.12573707, 0.12595813, 0.12614726,
                    0.12628796, 0.12652389, 0.12684535, 0.12714189, 0.1272784,
                    0.12734015, 0.12742706, 0.12764532, 0.12797589, 0.12822554,
                    0.12854458, 0.1287122, 0.12867132, 0.1285629, 0.12844155,
                    0.12796014, 0.12685216, 0.12594061, 0.12385031, 0.11678187,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, 0.02034021, 0.02050699, 0.02067377,
                    0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                    0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                    0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                    0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                    0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                    0.0250104])
upper_2 = np.array([0.66418859, 0.66204442, 0.65990097, 0.65775887, 0.65561775,
                    0.653478, 0.65133934, 0.64920284, 0.64706839, 0.64493535,
                    0.64280321, 0.64067272, 0.63854344, 0.63641654, 0.63429186,
                    0.63216884, 0.6300471, 0.62792644, 0.62580694, 0.6236888,
                    0.62157232, 0.61945693, 0.61734317, 0.61523156, 0.61312167,
                    0.61101359, 0.60890675, 0.6068009, 0.60469671, 0.60259359,
                    0.60049101, 0.59839051, 0.59629281, 0.59419596, 0.59210164,
                    0.59000894, 0.58791831, 0.5858293, 0.5837433, 0.58165909,
                    0.57957621, 0.57749648, 0.57541829, 0.57334259, 0.57126873,
                    0.56919701, 0.56712773, 0.56506106, 0.56299638, 0.56093262,
                    0.55887032, 0.55680898, 0.55474989, 0.55269291, 0.55063781,
                    0.54858495, 0.54653596, 0.54448849, 0.54244357, 0.5403992,
                    0.53835626, 0.53631627, 0.53427762, 0.53224285, 0.53021037,
                    0.52817962, 0.52614998, 0.52412095, 0.52209555, 0.52007454,
                    0.51805515, 0.51603767, 0.51402086, 0.51200624, 0.50999301,
                    0.50798156, 0.50597004, 0.50395995, 0.50195244, 0.4999472,
                    0.49794377, 0.49594084, 0.49393774, 0.4919352, 0.48993437,
                    0.48793651, 0.48594089, 0.48394776, 0.48195797, 0.47997048,
                    0.47798477, 0.47600302, 0.47402442, 0.47204874, 0.4700793,
                    0.46811576, 0.46615395, 0.46419492, 0.46223999, 0.46028721,
                    0.45834002, 0.45639603, 0.45445699, 0.45252067, 0.45058859,
                    0.44866228, 0.44674002, 0.4448208, 0.44290743, 0.44099693,
                    0.43908769, 0.4371818, 0.43528253, 0.43338972, 0.43150049,
                    0.42961753, 0.42773932, 0.42586493, 0.42399512, 0.42212683,
                    0.42026244, 0.41840106, 0.41654426, 0.41469479, 0.41284899,
                    0.41100982, 0.40917507, 0.40734281, 0.40551197, 0.40368378,
                    0.40186292, 0.40004654, 0.39823766, 0.39643339, 0.39463301,
                    0.39283819, 0.39104718, 0.38925952, 0.38747521, 0.38569598,
                    0.38392226, 0.38215446, 0.38039406, 0.37864066, 0.37689291,
                    0.3751554, 0.37342378, 0.37170246, 0.36998936, 0.36828506,
                    0.36659215, 0.36490764, 0.36323071, 0.36155705, 0.35988739,
                    0.35822062, 0.35656171, 0.35491084, 0.35326596, 0.35162834,
                    0.34999455, 0.34836477, 0.34674248, 0.34512566, 0.34351242,
                    0.34190637, 0.34030578, 0.33871666, 0.33713228, 0.3355551,
                    0.33397979, 0.3324173, 0.33086248, 0.3293129, 0.32776962,
                    0.32623663, 0.32470866, 0.32318419, 0.32166465, 0.32014733,
                    0.31863338, 0.31712514, 0.31562503, 0.31413079, 0.31263961,
                    0.31115724, 0.30967882, 0.30820232, 0.30673041, 0.30526638,
                    0.30381156, 0.30235831, 0.30091117, 0.29946951, 0.29803081,
                    0.29659647, 0.29516916, 0.29374523, 0.29232902, 0.29091328,
                    0.28949793, 0.28808418, 0.28667686, 0.28527513, 0.28387618,
                    0.28247854, 0.28108417, 0.27969428, 0.27831222, 0.27693427,
                    0.27555692, 0.27418577, 0.2728152, 0.27144892, 0.27008488,
                    0.26872167, 0.26735707, 0.26599097, 0.26462745, 0.26327261,
                    0.26192009, 0.26057035, 0.25922014, 0.25787205, 0.25652689,
                    0.25518207, 0.25384138, 0.25250149, 0.25116712, 0.24983626,
                    0.24849969, 0.24716054, 0.24582588, 0.24448973, 0.24315325,
                    0.24181261, 0.24047113, 0.2391248, 0.23777671, 0.23642554,
                    0.23507262, 0.23371766, 0.23236067, 0.23100164, 0.22964066,
                    0.22828146, 0.22691676, 0.2255452, 0.22417108, 0.22279258,
                    0.22141094, 0.22002368, 0.21863437, 0.21724429, 0.21585063,
                    0.21445963, 0.21306293, 0.2116641, 0.21026019, 0.20884795,
                    0.20743195, 0.20600902, 0.20458576, 0.20315685, 0.20172059,
                    0.2002836, 0.19884324, 0.19739871, 0.19595218, 0.19450059,
                    0.19304719, 0.19158717, 0.19012209, 0.18865214, 0.18717739,
                    0.18569654, 0.18421215, 0.18272079, 0.18122633, 0.17973172,
                    0.17823432, 0.17672978, 0.17522374, 0.17371221, 0.17218977,
                    0.17066344, 0.16913649, 0.16760825, 0.16607846, 0.16454559,
                    0.16300831, 0.16146899, 0.15993557, 0.15840246, 0.15686474,
                    0.15532482, 0.15378677, 0.15225507, 0.15073352, 0.14921427,
                    0.14769664, 0.14618354, 0.1446811, 0.14319193, 0.1417171,
                    0.14026462, 0.13882194, 0.13739605, 0.13599397, 0.13461421,
                    0.13325955, 0.13193428, 0.13064103, 0.12938163, 0.12815367,
                    0.12697364, 0.12583652, 0.1247403, 0.123689, 0.12268684,
                    0.12174473, 0.12086183, 0.12003167, 0.11926607, 0.11855889,
                    0.1179027, 0.11729332, 0.11674452, 0.11626127, 0.11583069,
                    0.11545017, 0.11510551, 0.11479943, 0.11455216, 0.11436439,
                    0.11423975, 0.11415638, 0.11411018, 0.11412537, 0.11417604,
                    0.11427288, 0.11442728, 0.11461337, 0.11482471, 0.11507363,
                    0.11534371, 0.11561752, 0.11589231, 0.11617911, 0.11648062,
                    0.11681189, 0.11715031, 0.11751016, 0.11788858, 0.11828832,
                    0.11869437, 0.11911478, 0.11956874, 0.12004319, 0.12053456,
                    0.12103781, 0.12155834, 0.12208825, 0.12260441, 0.12313557,
                    0.12367999, 0.12422287, 0.12476015, 0.12532923, 0.12591533,
                    0.1265241, 0.12716083, 0.12780689, 0.12843271, 0.12907556,
                    0.12973498, 0.13037785, 0.1310227, 0.13167086, 0.13228651,
                    0.1329113, 0.13355398, 0.13414818, 0.13473695, 0.13536139,
                    0.13599318, 0.13661648, 0.13723269, 0.13790165, 0.13856543,
                    0.1392194, 0.13990888, 0.14062205, 0.14132124, 0.14197167,
                    0.14260244, 0.14323696, 0.14389301, 0.14455467, 0.14517803,
                    0.14578846, 0.14634143, 0.14683027, 0.1472811, 0.14769476,
                    0.14802444, 0.14825779, 0.14848342, 0.14859245, 0.14856399,
                    0.14837282, 0.14814454, 0.14786393, 0.14769055, 0.14731322,
                    0.14670227, 0.14617674, 0.1455321, 0.14490684, 0.14377867,
                    0.14173332, 0.13732593, 0.02034021, 0.02050699, 0.02067377,
                    0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                    0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                    0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                    0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                    0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                    0.0250104])
vol_3 = np.array([0.5517277, 0.54999582, 0.54826725, 0.5465401, 0.54481425,
                  0.54309005, 0.54136845, 0.53965182, 0.53793567, 0.53622374,
                  0.53451483, 0.53280662, 0.53109952, 0.52939806, 0.52769823,
                  0.52600512, 0.52431557, 0.52262781, 0.52094006, 0.5192549,
                  0.51757402, 0.51589774, 0.51422762, 0.51256254, 0.51090104,
                  0.50924365, 0.50758846, 0.5059369, 0.50428661, 0.50264007,
                  0.50099707, 0.49935719, 0.49772289, 0.49609563, 0.49447175,
                  0.49284739, 0.49122254, 0.48959858, 0.48797662, 0.48635761,
                  0.48474023, 0.48312352, 0.48151494, 0.47991071, 0.4783084,
                  0.47670869, 0.4751094, 0.47351123, 0.47191639, 0.47032229,
                  0.46873129, 0.46714218, 0.46555213, 0.46396144, 0.46237448,
                  0.46079483, 0.45922182, 0.45765342, 0.45609377, 0.45453698,
                  0.45298364, 0.4514394, 0.44990531, 0.44837495, 0.44685123,
                  0.44532832, 0.44380589, 0.44228316, 0.4407629, 0.43924509,
                  0.43772993, 0.43621571, 0.43470666, 0.43320586, 0.43171319,
                  0.43022859, 0.42874941, 0.42727116, 0.42579438, 0.42432037,
                  0.42284802, 0.42137704, 0.41990679, 0.4184385, 0.41697348,
                  0.41550677, 0.41404281, 0.41258509, 0.41113267, 0.40967981,
                  0.40823187, 0.40678394, 0.40533938, 0.40389944, 0.40246175,
                  0.40102865, 0.39959943, 0.39818161, 0.39677477, 0.39537358,
                  0.39398046, 0.39259093, 0.39120242, 0.389821, 0.3884444,
                  0.38707303, 0.38570371, 0.3843376, 0.38297845, 0.38162732,
                  0.38028193, 0.37893628, 0.37759855, 0.37626797, 0.37494252,
                  0.37362151, 0.37230946, 0.37099835, 0.36969641, 0.36840073,
                  0.36711508, 0.36583275, 0.36455261, 0.36327949, 0.36200507,
                  0.36073144, 0.35945921, 0.3581891, 0.35691794, 0.35565386,
                  0.35439643, 0.35314233, 0.35189208, 0.35064192, 0.34938987,
                  0.34814222, 0.34690206, 0.34566546, 0.34442895, 0.34319332,
                  0.34196086, 0.34072777, 0.3394947, 0.33826815, 0.33704792,
                  0.33583808, 0.33463725, 0.33344149, 0.33225014, 0.33106437,
                  0.32988895, 0.32872307, 0.32756547, 0.32641658, 0.32527485,
                  0.32413891, 0.32299544, 0.32185191, 0.32071102, 0.31958002,
                  0.31845456, 0.31732748, 0.31619875, 0.31507231, 0.31395018,
                  0.31283507, 0.31172255, 0.3106103, 0.30950601, 0.30840724,
                  0.30730986, 0.30621389, 0.3051185, 0.30403111, 0.30294916,
                  0.30187234, 0.300795, 0.29971339, 0.29863219, 0.29755821,
                  0.29649211, 0.29543173, 0.2943732, 0.2933146, 0.29225324,
                  0.29118676, 0.29012218, 0.28906066, 0.28798978, 0.28691009,
                  0.28583795, 0.28476816, 0.28369781, 0.28262099, 0.2815422,
                  0.28045692, 0.27936554, 0.27827237, 0.27717874, 0.27608828,
                  0.27499761, 0.27390569, 0.27281, 0.27171382, 0.27061712,
                  0.26951785, 0.26841523, 0.26730995, 0.26619801, 0.26508548,
                  0.26397365, 0.26286604, 0.26175755, 0.26064485, 0.25952967,
                  0.25841252, 0.25729313, 0.25616733, 0.25503806, 0.25390659,
                  0.25277577, 0.25164653, 0.25051155, 0.249378, 0.24824466,
                  0.2471094, 0.2459723, 0.24483482, 0.24369137, 0.24255095,
                  0.2414057, 0.24025827, 0.23911176, 0.23796321, 0.23680668,
                  0.23563947, 0.23446279, 0.23328341, 0.23209983, 0.23091782,
                  0.2297327, 0.22853763, 0.22733433, 0.2261209, 0.22490098,
                  0.22367787, 0.2224517, 0.22121619, 0.21997305, 0.21872299,
                  0.21746597, 0.216203, 0.21494199, 0.21367644, 0.21240557,
                  0.2111268, 0.20983958, 0.20854758, 0.20724875, 0.20594869,
                  0.20464737, 0.20334379, 0.202029, 0.20070434, 0.1993717,
                  0.19803111, 0.19668491, 0.19533131, 0.19397454, 0.19261404,
                  0.19124624, 0.18987519, 0.18849871, 0.18712207, 0.18574042,
                  0.18435063, 0.18295685, 0.18155835, 0.18015082, 0.17873231,
                  0.17730623, 0.17587369, 0.17443759, 0.17299552, 0.17154911,
                  0.17009216, 0.16863092, 0.16716445, 0.16569564, 0.16422351,
                  0.16274721, 0.16126593, 0.15977887, 0.15828632, 0.15678865,
                  0.15528795, 0.15378818, 0.15229001, 0.15079582, 0.14930914,
                  0.14782461, 0.14633905, 0.14485903, 0.14338783, 0.14192603,
                  0.14048367, 0.1390617, 0.13766795, 0.13629147, 0.13492989,
                  0.13359268, 0.13228937, 0.13102175, 0.12978543, 0.12858172,
                  0.12740858, 0.12627753, 0.12519235, 0.12415302, 0.12316203,
                  0.12222029, 0.12133255, 0.12049408, 0.119713, 0.11899253,
                  0.11832643, 0.11771638, 0.1171605, 0.11665576, 0.11620413,
                  0.11580151, 0.11543782, 0.115111, 0.11482881, 0.11458555,
                  0.11439894, 0.1142576, 0.11418445, 0.1141504, 0.11414709,
                  0.11416486, 0.11420017, 0.11427867, 0.11439235, 0.11454104,
                  0.1147044, 0.11487558, 0.11505045, 0.11523548, 0.11541991,
                  0.11561582, 0.11583532, 0.11605736, 0.11630303, 0.11655207,
                  0.11681835, 0.11710094, 0.11740523, 0.11776409, 0.11816294,
                  0.1186147, 0.1190954, 0.11959967, 0.12010219, 0.12061142,
                  0.12110228, 0.12157532, 0.12206307, 0.1225572, 0.12305929,
                  0.12359314, 0.12414465, 0.1246912, 0.12519044, 0.12572199,
                  0.12631249, 0.12693731, 0.12758652, 0.1282172, 0.12883483,
                  0.12943212, 0.13002409, 0.1306133, 0.13119868, 0.13178499,
                  0.13241241, 0.1330067, 0.13363957, 0.13436914, 0.13508116,
                  0.13577222, 0.13648049, 0.13716695, 0.13779658, 0.13836164,
                  0.13887247, 0.1393547, 0.13979803, 0.14023707, 0.14072791,
                  0.14122425, 0.14171923, 0.1421337, 0.14245282, 0.14265671,
                  0.14280185, 0.14297957, 0.14320201, 0.14331041, 0.14325898,
                  0.14333028, 0.14342365, 0.14352024, 0.14363192, 0.14357775,
                  0.14330153, 0.14270969, 0.1416268, 0.1406158, 0.13906076,
                  0.13693846, 0.13488367, 0.12967636, 0.02050699, 0.02067377,
                  0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                  0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                  0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                  0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                  0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                  0.0250104])
lower_3 = np.array([nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, 0.21453874,
                    0.22905698, 0.23683377, 0.24217073, 0.24619672, 0.2494191,
                    0.25205763, 0.25423369, 0.25605126, 0.25759457, 0.25891835,
                    0.26006558, 0.26104999, 0.26188787, 0.26261675, 0.2632436,
                    0.26377042, 0.26420773, 0.26456269, 0.26485812, 0.26509427,
                    0.26527557, 0.26539605, 0.26545368, 0.26546143, 0.26543468,
                    0.26537692, 0.26528675, 0.26516005, 0.26499605, 0.26479288,
                    0.26454916, 0.26427775, 0.26398191, 0.26364482, 0.2632693,
                    0.26288042, 0.26247166, 0.26203991, 0.26157819, 0.26109384,
                    0.26058176, 0.26004357, 0.25948593, 0.25891133, 0.25832512,
                    0.25772348, 0.25710569, 0.25646918, 0.25581865, 0.2551546,
                    0.25447497, 0.25377931, 0.25306895, 0.25233955, 0.25159887,
                    0.2508488, 0.25009379, 0.24932812, 0.24854821, 0.24775644,
                    0.24695372, 0.24614003, 0.24531089, 0.24447001, 0.24361908,
                    0.24276158, 0.24189872, 0.24102253, 0.24014125, 0.23925368,
                    0.23835765, 0.23745345, 0.23654286, 0.23561992, 0.23469468,
                    0.2337587, 0.23281504, 0.2318672, 0.23091213, 0.22994359,
                    0.22895888, 0.22795949, 0.22695282, 0.22593741, 0.22491951,
                    0.22389424, 0.22285451, 0.22180227, 0.22073569, 0.21965869,
                    0.21857489, 0.2174845, 0.21638105, 0.21526645, 0.21414154,
                    0.21300638, 0.21186211, 0.21071702, 0.20956442, 0.20840361,
                    0.20723198, 0.20604904, 0.20485867, 0.20365883, 0.20245533,
                    0.20124821, 0.20003647, 0.19881101, 0.19757329, 0.19632528,
                    0.1950671, 0.19380117, 0.19252575, 0.19124517, 0.18995891,
                    0.1886634, 0.18736278, 0.1860549, 0.18474512, 0.18342858,
                    0.18210213, 0.18077002, 0.17943155, 0.17808238, 0.17672058,
                    0.17534964, 0.17397073, 0.17258678, 0.17119539, 0.16979825,
                    0.16838911, 0.16697431, 0.16555293, 0.1641279, 0.16269823,
                    0.16126311, 0.15982173, 0.1583733, 0.15691812, 0.15545658,
                    0.15399079, 0.15252474, 0.15105911, 0.14959628, 0.14813982,
                    0.14668433, 0.1452266, 0.14377324, 0.14232753, 0.14089004,
                    0.13947085, 0.13807088, 0.136698, 0.13534119, 0.13399803,
                    0.13267802, 0.13139071, 0.13013786, 0.12891505, 0.12772357,
                    0.12656132, 0.12543984, 0.1243629, 0.12333046, 0.122345,
                    0.12140741, 0.12052241, 0.11968525, 0.11890408, 0.11818214,
                    0.11751312, 0.11689872, 0.11633703, 0.11582498, 0.11536456,
                    0.11495162, 0.11457592, 0.11423532, 0.11393765, 0.11367709,
                    0.1134717, 0.11330991, 0.11321526, 0.11315812, 0.11312991,
                    0.11312068, 0.11312667, 0.11317433, 0.11325546, 0.11336996,
                    0.11349676, 0.11362852, 0.11376071, 0.11389982, 0.11403424,
                    0.11417634, 0.11433879, 0.1144989, 0.11467892, 0.11485678,
                    0.11504705, 0.11524843, 0.11546658, 0.11573805, 0.11604745,
                    0.11641012, 0.11679993, 0.11721127, 0.11761518, 0.11802114,
                    0.11839927, 0.11874881, 0.11910767, 0.11946571, 0.11982446,
                    0.1202136, 0.1206166, 0.12100451, 0.12132023, 0.12166581,
                    0.12208004, 0.12253279, 0.1230131, 0.12346073, 0.12388202,
                    0.12426418, 0.12462744, 0.12497468, 0.12530303, 0.12561888,
                    0.12599158, 0.12629587, 0.12665675, 0.1271929, 0.12770312,
                    0.12818049, 0.12870153, 0.12919349, 0.12958186, 0.12984049,
                    0.12997727, 0.13003802, 0.12998641, 0.12989665, 0.12993775,
                    0.13000947, 0.13010908, 0.12997595, 0.1294952, 0.12843457,
                    0.12673779, 0.12460996, 0.12194722, 0.1095842, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, nan, nan,
                    nan, nan, nan, 0.02050699, 0.02067377,
                    0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                    0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                    0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                    0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                    0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                    0.0250104])
upper_3 = np.array([0.66255523, 0.66041761, 0.65828168, 0.65614683, 0.65401303,
                    0.65188037, 0.64974918, 0.64762019, 0.64549199, 0.64336575,
                    0.64124111, 0.6391173, 0.63699447, 0.63487407, 0.63275481,
                    0.63063832, 0.62852361, 0.62641008, 0.62429718, 0.62218572,
                    0.62007626, 0.61796891, 0.61586417, 0.6137617, 0.61166101,
                    0.60956228, 0.60746489, 0.60536932, 0.60327478, 0.6011821,
                    0.59909121, 0.59700197, 0.59491521, 0.59283142, 0.59074938,
                    0.58866783, 0.58658675, 0.58450659, 0.58242774, 0.58035051,
                    0.57827446, 0.57619927, 0.57412748, 0.5720578, 0.56998942,
                    0.56792258, 0.56585654, 0.56379153, 0.56172831, 0.559666,
                    0.55760542, 0.55554615, 0.55348721, 0.5514287, 0.54937215,
                    0.54731882, 0.5452685, 0.54322049, 0.54117627, 0.53913378,
                    0.53709323, 0.53505667, 0.53302449, 0.53099442, 0.52896753,
                    0.52694173, 0.5249169, 0.52289276, 0.52087032, 0.51884958,
                    0.51683065, 0.51481288, 0.51279786, 0.51078677, 0.50877961,
                    0.50677638, 0.50477614, 0.50277719, 0.50077974, 0.49878432,
                    0.4967905, 0.49479819, 0.49280714, 0.49081784, 0.48883084,
                    0.48684417, 0.48485961, 0.48287856, 0.48090069, 0.47892372,
                    0.47694983, 0.47497704, 0.47300674, 0.47103947, 0.46907428,
                    0.46711217, 0.46515288, 0.46319959, 0.46125222, 0.45930859,
                    0.45736977, 0.45543392, 0.45349995, 0.45157053, 0.44964472,
                    0.44772276, 0.44580328, 0.44388683, 0.44197514, 0.44006876,
                    0.43816672, 0.43626635, 0.43437143, 0.43248168, 0.43059624,
                    0.42871485, 0.4268397, 0.42496709, 0.42310098, 0.42124008,
                    0.41938629, 0.41753647, 0.4156901, 0.41384962, 0.41201098,
                    0.41017523, 0.40834271, 0.40651382, 0.40468698, 0.40286642,
                    0.40105201, 0.3992421, 0.39743703, 0.39563486, 0.39383456,
                    0.39203953, 0.39025153, 0.38846852, 0.38668864, 0.38491236,
                    0.38314101, 0.38137248, 0.37960719, 0.37784887, 0.37609754,
                    0.37435567, 0.37262261, 0.37089616, 0.36917604, 0.36746303,
                    0.36576011, 0.36406698, 0.362383, 0.36070859, 0.35904293,
                    0.35738528, 0.3557273, 0.35407367, 0.35242617, 0.35078956,
                    0.34916116, 0.34753635, 0.34591511, 0.34430007, 0.34269265,
                    0.34109477, 0.33950351, 0.33791734, 0.33634161, 0.33477473,
                    0.33321392, 0.3316592, 0.33011002, 0.32857172, 0.32704259,
                    0.32552247, 0.32400731, 0.32249434, 0.32098696, 0.31949029,
                    0.3180049, 0.31652929, 0.31506059, 0.31359732, 0.31213742,
                    0.310679, 0.30922746, 0.30778371, 0.306338, 0.30489061,
                    0.30345447, 0.3020255, 0.30060133, 0.29917718, 0.29775659,
                    0.29633582, 0.29491508, 0.29349786, 0.2920852, 0.29068012,
                    0.28927978, 0.28788328, 0.28648845, 0.28509799, 0.28371186,
                    0.28232827, 0.28094646, 0.27956698, 0.2781863, 0.2768096,
                    0.27543797, 0.27407448, 0.27271464, 0.27135546, 0.2699984,
                    0.26864385, 0.26729152, 0.26593759, 0.26458464, 0.26323374,
                    0.26188742, 0.26054647, 0.25920422, 0.25786713, 0.25653404,
                    0.25520296, 0.25387392, 0.25254821, 0.2512206, 0.24989936,
                    0.24857718, 0.24725642, 0.24593995, 0.24462496, 0.24330582,
                    0.2419799, 0.24064824, 0.23931713, 0.23798511, 0.23665759,
                    0.23533008, 0.23399603, 0.23265699, 0.23131108, 0.22996169,
                    0.22861191, 0.22726183, 0.22590535, 0.22454406, 0.22317859,
                    0.22180882, 0.22043567, 0.21906674, 0.21769571, 0.21632179,
                    0.21494242, 0.213557, 0.21216906, 0.21077655, 0.20938488,
                    0.207994, 0.20660289, 0.20520276, 0.2037949, 0.20238108,
                    0.2009613, 0.19953779, 0.19810879, 0.19667838, 0.195246,
                    0.19380811, 0.19236864, 0.19092545, 0.18948366, 0.18803849,
                    0.18658681, 0.1851327, 0.1836754, 0.18221064, 0.18073643,
                    0.17925614, 0.17777083, 0.17628335, 0.17479128, 0.17329623,
                    0.17179202, 0.17028484, 0.16877374, 0.16726155, 0.16574729,
                    0.16423012, 0.1627092, 0.16118375, 0.15965401, 0.15812037,
                    0.15658489, 0.15505151, 0.15352088, 0.15199537, 0.1504785,
                    0.14896495, 0.14745154, 0.14594482, 0.14444808, 0.1429619,
                    0.1414963, 0.14005222, 0.13863749, 0.13724121, 0.13586106,
                    0.13450649, 0.133187, 0.1319044, 0.13065436, 0.1294382,
                    0.12825392, 0.12711303, 0.12601929, 0.12497274, 0.12397587,
                    0.1230296, 0.12213867, 0.12129842, 0.12051692, 0.11979737,
                    0.11913357, 0.1185272, 0.11797638, 0.11747813, 0.11703438,
                    0.11664108, 0.11628828, 0.11597401, 0.11570593, 0.11547845,
                    0.11530895, 0.11518619, 0.11513254, 0.11511938, 0.11513851,
                    0.11518054, 0.11524212, 0.11534814, 0.11549071, 0.11566956,
                    0.11586505, 0.11607064, 0.11628257, 0.1165072, 0.11673445,
                    0.11697609, 0.11724364, 0.11751738, 0.11781728, 0.11812451,
                    0.11845222, 0.11879962, 0.11917174, 0.11959842, 0.12006568,
                    0.12058455, 0.12113267, 0.1217048, 0.12227822, 0.12286046,
                    0.12342962, 0.12398684, 0.12456063, 0.12514366, 0.12573725,
                    0.1263613, 0.12700317, 0.12764383, 0.12824902, 0.12888477,
                    0.1295711, 0.1302868, 0.13102266, 0.13174457, 0.13245704,
                    0.13315483, 0.13384983, 0.13454393, 0.13523629, 0.13593046,
                    0.13665502, 0.13735517, 0.13808262, 0.13887606, 0.13965211,
                    0.14040822, 0.14117045, 0.14191041, 0.14260417, 0.14324748,
                    0.14384884, 0.14442616, 0.14497292, 0.14551242, 0.14607838,
                    0.14663736, 0.14718276, 0.14766392, 0.14807331, 0.14840217,
                    0.14868906, 0.1489863, 0.1492947, 0.14952025, 0.14964414,
                    0.14982156, 0.14998158, 0.15010684, 0.15018206, 0.15009914,
                    0.14981655, 0.14927618, 0.14839873, 0.14744251, 0.14602683,
                    0.14418023, 0.14170833, 0.13567184, 0.02050699, 0.02067377,
                    0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                    0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                    0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                    0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                    0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                    0.0250104])
vol_3_ = np.array([0.56120586, 0.5594352, 0.55766756, 0.55590126, 0.55413618,
                   0.55237265, 0.55061152, 0.54885486, 0.54709871, 0.54534454,
                   0.54359279, 0.54184174, 0.54009177, 0.53834691, 0.53660358,
                   0.53486631, 0.53313231, 0.53139998, 0.52966778, 0.52793797,
                   0.52621206, 0.52449036, 0.52277423, 0.52106269, 0.51935445,
                   0.51764998, 0.51594759, 0.51424854, 0.51255075, 0.5108564,
                   0.50916536, 0.5074772, 0.50579415, 0.50411751, 0.50244403,
                   0.50077025, 0.49909618, 0.49742302, 0.4957518, 0.49408335,
                   0.4924165, 0.49075038, 0.4890917, 0.48743705, 0.48578429,
                   0.484134, 0.48248424, 0.48083565, 0.47919036, 0.47754573,
                   0.47590405, 0.47426423, 0.47262372, 0.47098279, 0.46934538,
                   0.46771474, 0.46609025, 0.46447011, 0.46285805, 0.46124875,
                   0.45964278, 0.45804524, 0.4564571, 0.45487256, 0.45329425,
                   0.45171685, 0.45014018, 0.44856345, 0.4469892, 0.44541742,
                   0.44384828, 0.44228091, 0.44071987, 0.43916658, 0.43762093,
                   0.43608289, 0.43455006, 0.43301834, 0.43148822, 0.4299609,
                   0.42843536, 0.42691134, 0.42538827, 0.42386726, 0.42235101,
                   0.42083393, 0.41931963, 0.41781132, 0.41630815, 0.41480485,
                   0.41330633, 0.4118081, 0.41031304, 0.40882274, 0.40733481,
                   0.4058514, 0.40437188, 0.40290323, 0.40144507, 0.39999249,
                   0.39854774, 0.39710666, 0.39566688, 0.39423406, 0.39280609,
                   0.39138339, 0.38996294, 0.38854587, 0.38713568, 0.3857327,
                   0.38433588, 0.38294111, 0.38155424, 0.38017447, 0.37879994,
                   0.37742996, 0.37606944, 0.37471, 0.37335875, 0.3720127,
                   0.37067698, 0.36934452, 0.36801457, 0.3666906, 0.36536563,
                   0.36404315, 0.36272278, 0.3614049, 0.36008652, 0.35877506,
                   0.35747042, 0.35616944, 0.3548721, 0.35357556, 0.35227708,
                   0.35098326, 0.34969735, 0.34841659, 0.3471364, 0.34585761,
                   0.34458264, 0.3433073, 0.34203248, 0.34076448, 0.33950301,
                   0.33825319, 0.3370127, 0.33577862, 0.3345506, 0.3333279,
                   0.33211385, 0.33090996, 0.32971663, 0.32853293, 0.32735518,
                   0.32618114, 0.32500045, 0.32382161, 0.32264505, 0.32147861,
                   0.32031711, 0.31915539, 0.3179952, 0.31683781, 0.315685,
                   0.31453955, 0.31339545, 0.31224942, 0.31111214, 0.30998132,
                   0.30885072, 0.30772252, 0.30659836, 0.30548171, 0.30437215,
                   0.30326981, 0.30216798, 0.30106247, 0.29995572, 0.29885543,
                   0.29776256, 0.2966742, 0.29558838, 0.29450362, 0.29341568,
                   0.29232366, 0.29123326, 0.29014619, 0.2890502, 0.28794726,
                   0.28685201, 0.28575993, 0.28466851, 0.28357151, 0.28247366,
                   0.28136892, 0.28025719, 0.2791437, 0.27802991, 0.27692,
                   0.2758103, 0.27470115, 0.27358959, 0.27247669, 0.27136392,
                   0.27025023, 0.2691344, 0.26801645, 0.26689246, 0.26576817,
                   0.26464444, 0.26352641, 0.2624084, 0.26128667, 0.2601619,
                   0.25903567, 0.25790688, 0.25677308, 0.25563635, 0.25449683,
                   0.25335947, 0.2522248, 0.25108461, 0.24994377, 0.24880042,
                   0.24765342, 0.24650387, 0.24535133, 0.24419377, 0.24304018,
                   0.24188331, 0.24072321, 0.23956321, 0.23840344, 0.2372375,
                   0.23606122, 0.2348756, 0.23368638, 0.23249207, 0.23129911,
                   0.23010502, 0.22890344, 0.22769349, 0.2264731, 0.22524493,
                   0.2240125, 0.22277826, 0.22153547, 0.22028411, 0.21902448,
                   0.21775701, 0.2164852, 0.2152149, 0.21393989, 0.21266066,
                   0.21137712, 0.21008536, 0.20878761, 0.20748386, 0.20617826,
                   0.2048702, 0.20356016, 0.2022416, 0.20091468, 0.19958048,
                   0.19823953, 0.1968939, 0.1955404, 0.19418332, 0.19282156,
                   0.19145369, 0.19008282, 0.18870592, 0.18732752, 0.18594289,
                   0.18455028, 0.18315121, 0.18175021, 0.18034246, 0.178926,
                   0.17750067, 0.17606721, 0.17462576, 0.1731784, 0.17172902,
                   0.17026959, 0.1688082, 0.16734443, 0.165881, 0.16440878,
                   0.16293119, 0.16145032, 0.15996344, 0.15847454, 0.15697792,
                   0.15547577, 0.15397425, 0.15247311, 0.15097761, 0.14948796,
                   0.14800108, 0.14652192, 0.1450471, 0.14357978, 0.14211812,
                   0.14067413, 0.13925031, 0.13784381, 0.13645722, 0.13509232,
                   0.13375244, 0.13243784, 0.13115502, 0.12990361, 0.12868033,
                   0.1274894, 0.12633921, 0.12523768, 0.12418124, 0.12317347,
                   0.12221964, 0.12131984, 0.12048095, 0.11969779, 0.11896853,
                   0.118289, 0.11767087, 0.11710273, 0.11657566, 0.11609941,
                   0.11567234, 0.11528874, 0.1149464, 0.11466237, 0.1144269,
                   0.11424222, 0.11411519, 0.11404114, 0.11401182, 0.11400928,
                   0.11403424, 0.11407019, 0.11414592, 0.11425406, 0.11440125,
                   0.11456176, 0.11472928, 0.11490662, 0.11509594, 0.1152941,
                   0.11550892, 0.1157387, 0.11597193, 0.11621304, 0.11646106,
                   0.11671982, 0.11698647, 0.11727773, 0.11762418, 0.1180258,
                   0.11848689, 0.11897565, 0.1194786, 0.11999534, 0.12051679,
                   0.12101237, 0.12148629, 0.12196349, 0.12245258, 0.12296912,
                   0.1235144, 0.12408487, 0.12465113, 0.12517507, 0.12572769,
                   0.12632722, 0.1269607, 0.12761159, 0.12823417, 0.12882937,
                   0.12941531, 0.130006, 0.13059385, 0.13117774, 0.13176246,
                   0.13238823, 0.13298071, 0.1336117, 0.13434182, 0.13505207,
                   0.13574123, 0.13644751, 0.13713179, 0.13775894, 0.13832114,
                   0.1388287, 0.13930727, 0.13974646, 0.14019484, 0.1407279,
                   0.14122424, 0.14171922, 0.14213369, 0.14245281, 0.1426567,
                   0.14280185, 0.14297957, 0.14320201, 0.1433104, 0.14325898,
                   0.14333028, 0.14342365, 0.14352024, 0.14363192, 0.14357775,
                   0.14330153, 0.14270969, 0.1416268, 0.1406158, 0.13906076,
                   0.13693846, 0.13488367, 0.12967635, 0.02050699, 0.02067377,
                   0.02084056, 0.02100734, 0.02117412, 0.02134091, 0.02150769,
                   0.02167448, 0.02184127, 0.02200805, 0.02217484, 0.02234163,
                   0.02250843, 0.02267522, 0.02284201, 0.0230088, 0.0231756,
                   0.02334239, 0.02350919, 0.02367599, 0.02384279, 0.02400959,
                   0.02417639, 0.02434319, 0.02450999, 0.02467679, 0.0248436,
                   0.0250104])
print(np.amax(vol_3 - vol_3_))
print(np.amax(vol_3 - vol_3_2))
print(np.amax(vol_3_ - vol_3_2))
plt.plot(k_vec, vol, 'r-', label='Paper')
# plt.plot(k_vec, lower, 'r--')
# plt.plot(k_vec, upper, 'r--')
plt.plot(k_vec, vol_2, 'g-', label='Optimized')
# plt.plot(k_vec, lower_2, 'g--')
# plt.plot(k_vec, upper_2, 'g--')
plt.plot(k_vec, vol_3, 'b-', label='European')
plt.plot(k_vec, lower_3, 'b--')
plt.plot(k_vec, upper_3, 'b--')
plt.plot(k_vec, vol_3_, 'y-')
plt.plot(k_vec, vol_3_2, color='orange')
plt.plot(k_vec, Data.true_iv_surface_eur_call[-1, :], 'k-', label='Actual smile')
# plt.plot(k_vec, rHestonMarkov.iv_european_call(S=S, K=np.exp(k_vec), H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=6, mode=mode, rel_tol=rel_tol), 'b-')
plt.xlabel('Log-moneyness')
plt.ylabel('Implied volatility')
plt.legend(loc='upper right')
plt.title('Implied volatility smile using 6 dimensions\nand MC simulation')
plt.show()

methods = ['hyperplane reset']
mode = 'european'

with open('dW0.npy', 'rb') as f:
    dW = np.load(f)
with open('dB0.npy', 'rb') as f:
    dB = np.load(f)

WB_1 = np.empty((2, 1, 2048))
WB_2 = np.empty((2, 1, 1024))
WB_1[0, 0, :] = dW[0, :]
WB_1[1, 0, :] = dB[0, :]
for i in range(1024):
    WB_2[:, :, i] = WB_1[:, :, 2 * i] + WB_1[:, :, 2 * i + 1]
for vb in methods:
    for N in np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        S_1, V_1, _ = rHestonSP.sample_paths(H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, WB=WB_1,
                                             vol_behaviour=vb, mode=mode)
        # S_2, V_2, _ = ie.get_sample_paths(H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, N=N, WB=WB_2,
        # vol_behaviour=vb, mode=mode)
        plt.plot(np.linspace(0, 1, 2049), S_1[0, :], color=c_[N - 1], label=f'N={N}')
        # plt.plot(np.linspace(0, 1, 2049), V_1[0, :], label='volatility, N=2048')
        # plt.plot(np.linspace(0, 1, 1025), S_2[0, :], label='stock price, N=1024')
        # plt.plot(np.linspace(0, 1, 1025), V_2[0, :], label='volatility, N=1024')
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

'''
N_time = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
samples = np.empty(1000000)
k_vec = Data.k_rHeston
S = np.empty((len(N_time), 1000000))

print("Here")

for vb in methods:
    for N_t in N_time:
        WB = np.empty((2, 100000, N_t))
        factor = 2048 / N_t
        for i in range(10):
            with open(f'dW{i}.npy', 'rb') as f:
                dW = np.load(f)
            with open(f'dB{i}.npy', 'rb') as f:
                dB = np.load(f)
            for j in range(N_t):
                WB[0, :, j] = np.sum(dW[:, int(j * factor):int((j + 1) * factor)], axis=-1)
                WB[1, :, j] = np.sum(dB[:, int(j * factor):int((j + 1) * factor)], axis=-1)
            print(vb, N_t, i)
            samples[i * 100000:(i + 1) * 100000] = rHestonSP.sample_values(H=0.49, N=1, N_time=N_t, WB=WB, vol_behaviour=vb)
        np.save(f'samples of {vb} mode with N_time={N_t} and H=0.49 and N=1', samples)
time.sleep(3600000)
for vb in methods:
    for i in range(len(N_time)):
        print(N_time[i])
        with open(f'samples of {vb} mode with N_time={N_time[i]}.npy', 'rb') as f:
            S[i, :] = np.load(f)
        est, lower, upper = cf.iv_eur_call_MC(S[i, :], np.exp(k_vec), 1., 1.)
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
        factor = 2048 / N_t
        for i in range(10):
            with open(f'dW{i}.npy', 'rb') as f:
                dW = np.load(f)
            with open(f'dB{i}.npy', 'rb') as f:
                dB = np.load(f)
            for j in range(N_t):
                WB[0, :, j] = np.sum(dW[:, int(j * factor):int((j + 1) * factor)], axis=-1)
                WB[1, :, j] = np.sum(dB[:, int(j * factor):int((j + 1) * factor)], axis=-1)
            print(vb, N_t, i)
            samples[i * 100000:(i + 1) * 100000] = rHestonSP.sample_values(N_time=N_t, WB=WB, vol_behaviour=vb)
        np.save(f'samples of {vb} mode with N_time={N_t}', samples)

time.sleep(3600000)

S, V, _ = rHestonSP.sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6,
                                 vol_behaviour='split kernel')
plt.plot(np.linspace(0, 1, 1001), S, label='stock price')
plt.plot(np.linspace(0, 1, 1001), V, label='volatility')
plt.xlabel('t')
plt.title('Sample path of split kernel implementation')
plt.show()

time.sleep(360000)

K = np.exp(Data.k_rHeston)
N = 6
tic = time.perf_counter()
vol, lower, upper = rHestonSP.call(K, N=N, N_time=1000, m=200000, vol_behaviour='multiple time scales')
toc = time.perf_counter()
print(toc - tic)
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
    vol, lower, upper = rHestonSP.call(K, N=N, N_time=N_time, m=1000000, bounce_vol=False)
    toc = time.perf_counter()
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, vol.txt', vol, delimiter=',', header=f'time: {toc - tic}')
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, lower.txt', lower, delimiter=',', header=f'time: {toc - tic}')
    np.savetxt(f'rHestonIE N={N}, N_time={N_time}, upper.txt', upper, delimiter=',', header=f'time: {toc - tic}')
'''
