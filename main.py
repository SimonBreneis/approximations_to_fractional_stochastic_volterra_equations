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
print(rk.quadrature_rule_geometric_standard(0.1, 6, 1.))
time.sleep(3600)
S, V, _, bad = ie.get_sample_paths(H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1, N=1, vol_behaviour='adaptive')
print(bad)
plt.plot(np.linspace(0, 1, 1001), S[0, :])
plt.plot(np.linspace(0, 1, 1001), V[0, :])
plt.show()
ie.call(K=np.array([1.]), vol_behaviour='adaptive')
# methods = ['mean reversion', 'hyperplane reflection', 'split throw', 'multiple time scales']
methods = ['sticky', 'hyperplane reset', 'split kernel']

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
k_vec = Data.k_rrHeston
S = np.empty((len(N_time), 1000000))
'''
for vb in methods:
    for i in range(len(N_time)):
        print(N_time[i])
        with open(f'samples of {vb} mode with N_time={N_time[i]}.npy', 'rb') as f:
            S[i, :] = np.load(f)
        est, lower, upper = cf.volatility_smile_call(S[i, :], np.exp(k_vec), 1., 1.)
        plt.plot(k_vec, est, label=f'N_time={N_time[i]}')
    plt.plot(k_vec, lower, 'k--')
    plt.plot(k_vec, upper, 'k--')
    plt.plot(Data.k_rrHeston, Data.rrHeston_6, 'k-', label=f'Fourier inversion')
    plt.title(vb)
    plt.legend(loc='best')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.show()
'''
strikes = np.array([-0.3, 0., 0.15])
true_iv = np.array([np.interp(strike, Data.k_rrHeston, Data.rrHeston_6) for strike in strikes])
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
