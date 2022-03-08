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


methods = ['sticky', 'hyperplane reset', 'split kernel', 'mean reversion', 'hyperplane reflection', 'split throw', 'multiple time scales']
N_time = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

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
            samples = ie.samples(N_time=N_t, WB=WB, vol_behaviour=vb)
            np.save(f'samples {i} of {vb} mode with N_time={N_t}')

Data.plot_rHeston_IE_smiles()

K = np.exp(Data.k_rrHeston)
N = 6
tic = time.perf_counter()
vol, lower, upper = ie.call(K, N=N, N_time=1000, m=200000, vol_behaviour='split kernel')
toc = time.perf_counter()
print(toc-tic)
print(vol)
print(lower)
print(upper)
time.sleep(360000)

S, V, _ = ie.get_sample_path(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, vol_behaviour='split kernel')
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

K = np.exp(Data.k_rrHeston)
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
