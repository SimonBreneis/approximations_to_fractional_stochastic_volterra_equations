import time
import numpy as np
import matplotlib.pyplot as plt
# import Data
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


nodes, weights = rk.quadrature_rule_geometric_good(0.1, 6)
nodes = rk.mp_to_np(nodes)
weights = rk.mp_to_np(weights)
nodes = nodes[:-1]
weights = weights[:-1]
i = np.sum(nodes < 200)
nodes = nodes[i:]
weights = weights[i:]
print(nodes)

T = 10/np.amin(nodes)
N_Riccati = 2000
times, psi = sk.solve_Riccati_high_mean_reversion(2j, nodes, weights, lambda_=0.3, nu=0.3, N_Riccati=10, T=T, adaptive=True)
plt.plot(times, psi[0, :], label='real part')
plt.plot(times, psi[1, :], label='imaginary part')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel(r'$\psi(t)$')
plt.show()

'''
true_psi = sk.solve_Riccati_high_mean_reversion(2j, nodes, weights, lambda_=0.3, nu=0.3, N_Riccati=1048576, T=T, adaptive=False)
true_ints = sk.psi_integrals(true_psi, T)
print(true_ints)
N_Riccatis = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
approx_ints = np.empty((2, len(N_Riccatis)), dtype=np.complex_)
for i in range(len(N_Riccatis)):
    approx_ints[:, i] = sk.psi_integrals(sk.solve_Riccati_high_mean_reversion(2j, nodes, weights, lambda_=0.3, nu=0.3, N_Riccati=N_Riccatis[i], T=T, adaptive=False), T)
print(approx_ints)
errors = np.array([[np.abs(approx_ints[i, j] - true_ints[i]) for j in range(len(N_Riccatis))] for i in range(2)])
a1, b1, _, _, _ = Data.log_linear_regression(N_Riccatis, errors[0, :])
a2, b2, _, _, _ = Data.log_linear_regression(N_Riccatis, errors[1, :])
print(a1, b1)
print(a2, b2)
plt.loglog(N_Riccatis, errors[0, :], 'b-', label='error of the first moment')
plt.loglog(N_Riccatis, b1*N_Riccatis**a1, 'b--')
plt.loglog(N_Riccatis, errors[1, :], 'r-', label='error of the second moment')
plt.loglog(N_Riccatis, b2*N_Riccatis**a2, 'r--')
plt.legend(loc='best')
plt.xlabel('Number of time steps')
plt.ylabel('Error')
plt.show()
'''

true_ints = sk.psi_integrals(*sk.solve_Riccati_high_mean_reversion(2j, nodes, weights, lambda_=0.3, nu=0.3, N_Riccati=1024, T=T, adaptive=True))
print(true_ints)
N_Riccatis = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
approx_ints = np.empty((2, len(N_Riccatis)), dtype=np.complex_)
for i in range(len(N_Riccatis)):
    approx_ints[:, i] = sk.psi_integrals(*sk.solve_Riccati_high_mean_reversion(2j, nodes, weights, lambda_=0.3, nu=0.3, N_Riccati=N_Riccatis[i], T=T, adaptive=True))
print(approx_ints)
errors = np.array([[np.abs(approx_ints[i, j] - true_ints[i]) for j in range(len(N_Riccatis))] for i in range(2)])
a1, b1, _, _, _ = Data.log_linear_regression(N_Riccatis, errors[0, :])
a2, b2, _, _, _ = Data.log_linear_regression(N_Riccatis[4:], errors[1, 4:])
print(a1, b1)
print(a2, b2)
plt.loglog(40*N_Riccatis, errors[0, :], 'b-', label='error of the first moment')
plt.loglog(40*N_Riccatis, b1*N_Riccatis**a1, 'b--')
plt.loglog(40*N_Riccatis, errors[1, :], 'r-', label='error of the second moment')
plt.loglog(40*N_Riccatis, b2*N_Riccatis**a2, 'r--')
plt.legend(loc='best')
plt.xlabel('Number of time steps')
plt.ylabel('Error')
plt.show()


'''
ints = sk.psi_integrals(psi, T)
print(2j - 0.3*ints[0] + 0.3**2*ints[1]/2)
us = np.linspace(-5, 5, 100)
cf = sk.chararacteristic_function_high_mean_reversion(us*1j, 1., nodes, weights, lambda_=0.3, nu=0.3, theta=0.02, T=T, N_Riccati=N_Riccati)
plt.plot(us, cf.real)
plt.plot(us, cf.imag)
plt.show()
'''

sk.regress_for_varying_p()
time.sleep(360000)

'''
S, V, _ = ie.get_sample_path(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, vol_behaviour='multiple time scales')
plt.plot(np.linspace(0, 1, 1001), S, label='stock price')
plt.plot(np.linspace(0, 1, 1001), V, label='volatility')
plt.xlabel('t')
plt.title('Sample path of multiple time scales implementation')
plt.show()

Data.plot_rHeston_IE_smiles()

S, V, _ = ie.get_sample_path(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, S_0=1., vol_behaviour='hyperplane reset')
plt.plot(np.linspace(0, 1, 1001), S, label='stock price')
plt.plot(np.linspace(0, 1, 1001), V, label='volatility')
plt.xlabel('t')
plt.title('Sample path of hyperplane reflection implementation')
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

tic_ = time.perf_counter()
vol_, lower_, upper_ = ie.call(K, m=200000, bounce_vol=False)
toc_ = time.perf_counter()
plt.plot(np.log(K), vol_, 'b-')
plt.plot(np.log(K), lower_, 'b--')
plt.plot(np.log(K), upper_, 'b--')
plt.show()

print(toc - tic)
print(vol)
print(lower)
print(upper)
print(toc_ - tic_)
print(vol_)
print(lower_)
print(upper_)
time.sleep(360000)


Data.plot_rHeston_IE_smiles()

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
