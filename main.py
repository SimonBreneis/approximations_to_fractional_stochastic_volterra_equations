import time
import matplotlib.pyplot as plt
import numpy as np
import Data
import ComputationalFinance
import RoughKernel as rk
import rBergomi
import fBmMarkov
import functions
import rHestonFourier
import rHestonMarkovSimulation
import rHestonQESimulation
from functions import *
import rBergomiMarkov
import rHestonMomentMatching
import scipy.stats, scipy.optimize, scipy.integrate, scipy.linalg, scipy.special
from scipy.special import gamma
import fbm

import csv


rHestonFourier.eur_call_put(S_0=1., K=np.exp(np.linspace(-0.5, 0.3, 101)), lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., rel_tol=1e-05, H=0.1)

print(rk.quadrature_rule(H=0.1, N=2, T=1.))
print(rk.quadrature_rule(H=0.1, N=3, T=1.))
print(rk.quadrature_rule(H=-0.2, N=2, T=1.))
print(rk.quadrature_rule(H=-0.2, N=3, T=1.))
time.sleep(3600000)

def save_string_to_float(x):
    try:
        return float(x)
    except ValueError:
        pass
    return None


def get_data():
    data_ = np.empty((2583, 3))  # 2583 data points with maturity T, log-moneyness k and implied volatility sigma
    with open('ivols.csv', newline='') as csvfile:
        ivols_reader = csv.reader(csvfile, delimiter=',')
        i = -1
        for row in ivols_reader:
            print(', '.join(row))
            if i >= 0:
                data_[i, :] = np.array([save_string_to_float(row[2]),
                                       np.log(save_string_to_float(row[3]) / save_string_to_float(row[6])),
                                       save_string_to_float(row[5])])
            i = i + 1
    return data_


data = get_data()
data = data[np.abs(data[:, 1]) <= 0.5, :]
data = data[data[:, 0] > np.amin(data[:, 0]), :]
data = data[data[:, 0] > np.amin(data[:, 0]), :]
data = data[data[:, 0] > np.amin(data[:, 0]), :]



import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
print(data)


from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
print(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_trisurf(data[:, 1], data[:, 0], data[:, 2], cmap='hsv',
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_ylabel('Maturity T')
ax.set_xlabel('Log-moneyness k')
ax.set_zlabel('Implied volatility')
ax.set_zticks([0.1, 0.2, 0.3, 0.4, 0.5])

plt.tight_layout()
plt.show()

print(data[1000, 0])
data = data[data[:, 0] == data[1000, 0], :]
plt.plot(data[:, 1], data[:, 2])
plt.xlabel('Log-moneyness k')
plt.ylabel('Implied volatility')
plt.tight_layout()
plt.show()

k = np.linspace(-0.5, 0.3, 201)
iv_smile = rHestonFourier.eur_call_put(S_0=1., K=np.exp(k), lambda_=0.3, rho=-0.7, theta=0.02,
                                       V_0=0.02, nu=1., nodes=np.array([0.]), weights=np.array([1.]), T=0.23,
                                       rel_tol=1e-04, verbose=2)
plt.plot(k, iv_smile)
plt.xlabel('Log-moneyness k')
plt.title('Implied volatility smile for Heston')
plt.tight_layout()
plt.show()

data = get_data()
skew = []
maturities = []
current_T = 0
current_T_index = -1
while data.shape[0] > 0:
    current_data = data[data[:, 0] == data[0, 0], :]
    current_data_positive_k = current_data[current_data[:, 1] >= 0, :]
    current_data_negative_k = current_data[current_data[:, 1] < 0, :]
    maturities.append(current_data[0, 0])
    '''
    skew.append((
        np.abs((current_data_positive_k[0, 2] - current_data_negative_k[-1, 2]) / (current_data_positive_k[0, 1] - current_data_negative_k[-1, 1]))
        + np.abs((current_data_positive_k[1, 2] - current_data_negative_k[-1, 2]) / (current_data_positive_k[1, 1] - current_data_negative_k[-1, 1]))
        + np.abs((current_data_positive_k[2, 2] - current_data_negative_k[-1, 2]) / (current_data_positive_k[2, 1] - current_data_negative_k[-1, 1]))
        + np.abs((current_data_positive_k[0, 2] - current_data_negative_k[-2, 2]) / (current_data_positive_k[0, 1] - current_data_negative_k[-2, 1]))
        + np.abs((current_data_positive_k[1, 2] - current_data_negative_k[-2, 2]) / (current_data_positive_k[1, 1] - current_data_negative_k[-2, 1]))
        + np.abs((current_data_positive_k[2, 2] - current_data_negative_k[-2, 2]) / (current_data_positive_k[2, 1] - current_data_negative_k[-2, 1]))
        + np.abs((current_data_positive_k[0, 2] - current_data_negative_k[-3, 2]) / (current_data_positive_k[0, 1] - current_data_negative_k[-3, 1]))
        + np.abs((current_data_positive_k[1, 2] - current_data_negative_k[-3, 2]) / (current_data_positive_k[1, 1] - current_data_negative_k[-3, 1]))
        + np.abs((current_data_positive_k[2, 2] - current_data_negative_k[-3, 2]) / (current_data_positive_k[2, 1] - current_data_negative_k[-3, 1]))) / 9)
    '''
    skew.append((
        np.abs((current_data_positive_k[0, 2] - current_data_negative_k[-1, 2]) / (current_data_positive_k[0, 1] - current_data_negative_k[-1, 1]))
        + np.abs((current_data_positive_k[1, 2] - current_data_negative_k[-1, 2]) / (current_data_positive_k[1, 1] - current_data_negative_k[-1, 1]))
        + np.abs((current_data_positive_k[0, 2] - current_data_negative_k[-2, 2]) / (current_data_positive_k[0, 1] - current_data_negative_k[-2, 1]))
        + np.abs((current_data_positive_k[1, 2] - current_data_negative_k[-2, 2]) / (current_data_positive_k[1, 1] - current_data_negative_k[-2, 1]))) / 4)
    data = data[data[:, 0] != data[0, 0], :]
skew, maturities = np.array(skew), np.array(maturities)
exponent, constant, _, _, _ = functions.log_linear_regression(maturities, skew)
print(exponent, constant)
plt.scatter(maturities, skew)
plt.plot(maturities, constant * maturities ** exponent)
plt.ylim(0, 1.4)
plt.xlabel('Maturity T')
plt.ylabel('Skew')
plt.tight_layout()
plt.show()
'''
skew_heston = rHestonFourier.skew_eur_call_put(lambda_=0.3, rho=-0.7, nu=1, theta=0.02, V_0=0.02, T=maturities, nodes=np.array([0.]), weights=np.array([1.]), verbose=2)
plt.plot(maturities, skew_heston)
plt.show()
'''

seed = 801 * 42
np.random.seed(seed)
f = fbm.FBM(2 ** 12, 0.1)
fbm_sample_1 = f.fbm()
np.random.seed(seed)
f = fbm.FBM(2 ** 12, 0.3)
fbm_sample_2 = f.fbm()
np.random.seed(seed)
f = fbm.FBM(2 ** 12, 0.499)
fbm_sample_3 = f.fbm()
np.random.seed(seed)
f = fbm.FBM(2 ** 12, 0.7)
fbm_sample_4 = f.fbm()
np.random.seed(seed)
f = fbm.FBM(2 ** 12, 0.9)
fbm_sample_5 = f.fbm()
times = f.times()
plt.plot(times, fbm_sample_1, color=color(0, 5), label='H=0.1')
plt.plot(times, fbm_sample_2, color=color(1, 5), label='H=0.3')
plt.plot(times, fbm_sample_3, color=color(2, 5), label='H=0.5')
plt.plot(times, fbm_sample_4, color=color(3, 5), label='H=0.7')
plt.plot(times, fbm_sample_5, color=color(4, 5), label='H=0.9')
plt.xlabel('Time t')
plt.legend(loc='best')
plt.title('Sample paths of fractional Brownian motion')
plt.tight_layout()
plt.show()

n = 1000
BM_increments = np.random.normal(0, np.sqrt(1 / n), n)
n_mean_reversions = 201
mean_reversions = np.linspace(0, 10, n_mean_reversions)
OU_paths = np.zeros((n_mean_reversions, n + 1))

for i in range(1, n + 1):
    OU_paths[:, i] = OU_paths[:, i - 1] - mean_reversions * OU_paths[:, i - 1] / n + np.random.normal(0, n ** (-0.5))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
print(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

X, Y = np.meshgrid(np.linspace(0, 1, n + 1), mean_reversions)

# Plot the surface.
surf = ax.plot_surface(X, Y, OU_paths, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('Time t')
ax.set_ylabel('Mean reversion x')

plt.tight_layout()
plt.show()
print('Finished')
time.sleep(360000)


Data.illustrate_bermudan_option_prices()
Data.plot_for_simulation_paper_smile_errors()

'''
Data.illustrate_bermudan_option_prices()
generate_and_safe_samples_2()
print('Finished')
time.sleep(360000)
'''

H, T, S_0, V_0, theta, rho, nu, lambda_, r, rel_tol = 0.1, 1., 1., 0.02, 0.02, -0.7, 0.3, 0.3, 0., 1e-05
K = np.exp(np.linspace(-0.1, 0.05, 16))
'''
approx_smiles = np.empty((25, len(K)))
for i in range(25):
    print(i)
    stock_prices = np.load(f'rHeston HQE samples 1024 {i + 1}.npy')
    approx_smiles[i, :], _, _ = cf.eur_MC(S_0=S_0, K=K, T=T, samples=stock_prices, r=r, implied_vol=False)
approx_smile, lower, upper = cf.iv_eur(S_0=S_0, K=K, T=T, price=np.average(approx_smiles, axis=0), payoff='call', stat=np.std(approx_smiles, axis=0) / 5 * 1.95)
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                         r=r, rel_tol=rel_tol, H=H, implied_vol=True)
print(np.amax(np.abs(true_smile - approx_smile) / true_smile),
      np.fmax(np.amax(np.abs(approx_smile - lower) / approx_smile),
              np.amax(np.abs(approx_smile - upper) / approx_smile)))

T = np.linspace(0, 1, 17)[1:]
K_mat = S_0 * np.exp(np.sqrt(T[:, None]) * np.log(K / S_0)[None, :])
payoffs = np.empty((len(T), len(K), 5 * 220))
for i in range(25):
    samples = np.load(f'rHeston HQE samples 1024 {i + 1}.npy')
    payoffs[:, :, i * 220 + j], _, _ = cf.eur_MC(S_0=S_0, K=K_mat, T=T, samples=samples, r=r, implied_vol=False)
price, stat = cf.MC(payoffs)
approx_smile, lower, upper = cf.iv_eur(S_0=S_0, K=K_mat, T=T, price=price, payoff='call', r=r, stat=stat)
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K_mat, T=T, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                         r=r, rel_tol=rel_tol, H=H, implied_vol=True, verbose=2)
print(np.amax(np.abs(true_smile - approx_smile) / true_smile),
      np.fmax(np.amax(np.abs(approx_smile - lower) / approx_smile),
              np.amax(np.abs(approx_smile - upper) / approx_smile)))
'''
T = 1
payoffs = np.empty((len(K), 25))
for i in range(25):
    print(i)
    samples = np.load(f'rHeston HQE geom avg samples 1024 {i + 1}.npy')
    payoffs[:, i], _ = cf.MC(cf.payoff_call(S=samples, K=K))
true_smile = rHestonFourier.geom_asian_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                T=T, rel_tol=rel_tol, H=H)
approx_smile, stat = cf.MC(payoffs)
print(np.amax(np.abs(true_smile - approx_smile) / true_smile), np.amax(stat / approx_smile))

K = np.exp(np.linspace(-0.5, 0.5, 101))
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, r=r,
                                         rel_tol=rel_tol, H=H)
nodes, weights = rk.quadrature_rule(H=H, N=2, T=T)
markov_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                           r=r, rel_tol=rel_tol, H=H, nodes=nodes, weights=weights)
m = 2 ** 24
N_time = 32
HQE_smile, _, _ = rHestonQESimulation.eur(H=H, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T,
                                          r=r, m=m, N_time=N_time, implied_vol=True, qmc_error_estimators=1, verbose=2)
Weak_smile, _, _ = rHestonMarkovSimulation.eur(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T,
                                               nodes=nodes, weights=weights, r=r, m=m, N_time=N_time, euler=False,
                                               implied_vol=True, qmc_error_estimators=1, verbose=2)
Euler_smile, _, _ = rHestonMarkovSimulation.eur(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0,
                                                T=T, nodes=nodes, weights=weights, r=r, m=m, N_time=N_time, euler=True,
                                                implied_vol=True, qmc_error_estimators=1, verbose=2)
k = np.log(K)
plt.plot(k, true_smile, 'k-', label='True smile')
plt.plot(k, markov_smile, 'brown', label='Markov smile')
plt.plot(k, Euler_smile, color='red', marker='x', markevery=10, label='Euler smile')
plt.plot(k, Weak_smile, 'g-', label='Weak smile')
plt.plot(k, HQE_smile, color='blue', marker='o', markevery=10, label='HQE smile')
plt.xlabel('Log-moneyness')
plt.title('Implied volatility for European call options')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
print('Finished')
time.sleep(360000)




Data.plot_for_simulation_paper_smile_errors()


def plot_samples_volatility_markov():
    nodes = np.array([1., 10.])
    weights = np.array([1., 2.])
    samples = rHestonMarkovSimulation.samples(lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=100., m=1000, N_time=100000,
                                               nodes=nodes, weights=weights, vol_only=True, sample_paths=True,
                                               rho=-0.7, S_0=1., r=0., qmc=False, return_times=1000)[0][1:, :]
    # samples = weights[:, None] * (samples - np.array([0.01, 0.])[:, None])
    # samples = weights[:, None] * samples

    plt.scatter(samples[0, :].flatten(), samples[1, :].flatten())
    # plt.scatter(np.array([-0.01 * nodes[1] / np.sum(nodes)]), np.array([-0.01 * nodes[0] / np.sum(nodes)]))
    plt.plot(np.linspace(-1, 1, 100), -np.linspace(-1, 1, 100) / weights[1] * weights[0], 'k-') # - 0.01)
    plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), 'k-') # - 0.01)
    # plt.plot(np.linspace(-1, 1, 100), (weights[1] - weights[0]) / weights[0] * np.linspace(-1, 1, 100))
    plt.xlabel(r'$V_1$')
    plt.ylabel(r'$V_2$')
    plt.title("Samples of the two-dimensional volatility process")
    plt.tight_layout()
    plt.show()

    samples_ = np.empty((2, samples.shape[1] * samples.shape[2]))
    samples_[0, :] = samples[0, :, :].flatten()
    samples_[1, :] = samples[1, :, :].flatten()
    Q = np.empty((2, 2))
    Q[0, :] = weights
    Q[1, :] = np.array([1, -1])
    samples_transformed = np.einsum('ij,jk->ik', Q, samples_)
    plt.scatter(samples_transformed[0, :], samples_transformed[1, :])
    plt.xlabel(r'$U_1$')
    plt.ylabel(r'$U_2$')
    plt.title("Samples after linear transformation")
    plt.show()


def plot_samples_volatility_markov_3d():
    nodes = np.array([1., 5., 25.])
    weights = np.array([1., 2., 3.])
    samples = \
    rHestonMarkovSimulation.samples(lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=100., m=1000, N_time=100000,
                                    nodes=nodes, weights=weights, vol_only=True, sample_paths=True,
                                    rho=-0.7, S_0=1., r=0., qmc=False, return_times=1000)[0][1:, :]
    # samples = weights[:, None] * (samples - np.array([0.01, 0.])[:, None])
    # samples = weights[:, None] * samples
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(samples[0, :].flatten(), samples[1, :].flatten(), samples[2, :].flatten())
    # plt.scatter(samples[0, :].flatten(), samples[1, :].flatten(), samples[2, :].flatten())
    # plt.scatter(np.array([-0.01 * nodes[1] / np.sum(nodes)]), np.array([-0.01 * nodes[0] / np.sum(nodes)]))
    # plt.plot(np.linspace(-1, 1, 100), -np.linspace(-1, 1, 100) / weights[1] * weights[0], 'k-')  # - 0.01)
    # plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), 'k-')  # - 0.01)
    # plt.plot(np.linspace(-1, 1, 100), (weights[1] - weights[0]) / weights[0] * np.linspace(-1, 1, 100))
    # plt.xlabel(r'$V_1$')
    # plt.ylabel(r'$V_2$')
    plt.title("Samples of the two-dimensional volatility process")
    plt.tight_layout()
    plt.show()
    '''
    plt.scatter(samples[0, :].flatten(), samples[2, :].flatten())
    plt.show()

    samples_ = np.empty((3, samples.shape[1] * samples.shape[2]))
    samples_[0, :] = samples[0, :, :].flatten()
    samples_[1, :] = samples[1, :, :].flatten()
    samples_[2, :] = samples[2, :, :].flatten()
    Q = np.empty((3, 3))
    a = 1
    b = 2
    Q[0, :] = weights
    Q[1, :] = np.array([1, -a, -1 + a])
    Q[2, :] = np.array([1, b, -1 - b])
    samples_transformed = np.einsum('ij,jk->ik', Q, samples_)
    '''

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(samples[0, :].flatten(), samples[1, :].flatten(), samples[2, :].flatten())
    # plt.scatter(samples[0, :].flatten(), samples[1, :].flatten(), samples[2, :].flatten())
    # plt.scatter(np.array([-0.01 * nodes[1] / np.sum(nodes)]), np.array([-0.01 * nodes[0] / np.sum(nodes)]))
    # plt.plot(np.linspace(-1, 1, 100), -np.linspace(-1, 1, 100) / weights[1] * weights[0], 'k-')  # - 0.01)
    # plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), 'k-')  # - 0.01)
    # plt.plot(np.linspace(-1, 1, 100), (weights[1] - weights[0]) / weights[0] * np.linspace(-1, 1, 100))
    ax.set_xlabel(r'$U_1$')
    ax.set_ylabel(r'$U_2$')
    ax.set_zlabel(r'$U_2$')
    plt.title(r"Samples after linear transformation with $a=1$ and $b=2$")
    plt.tight_layout()
    plt.show()
    '''
    plt.scatter(samples_transformed[0, :], samples_transformed[1, :])
    plt.xlabel(r'$U_1$')
    plt.ylabel(r'$U_2$')
    plt.title(r"Samples after linear transformation with $a=1$ and $b=2$")
    plt.tight_layout()
    plt.show()

    plt.scatter(samples_transformed[0, :], samples_transformed[2, :])
    plt.xlabel(r'$U_1$')
    plt.ylabel(r'$U_3$')
    plt.title(r"Samples after linear transformation with $a=1$ and $b=2$")
    plt.tight_layout()
    plt.show()

    plt.scatter(samples_transformed[1, :], samples_transformed[2, :])
    plt.xlabel(r'$U_2$')
    plt.ylabel(r'$U_3$')
    plt.title(r"Samples after linear transformation with $a=1$ and $b=2$")
    plt.tight_layout()
    plt.show()

    Q = np.empty((3, 3))
    a = 6 / 5
    b = 5 / 2
    Q[0, :] = weights
    Q[1, :] = np.array([1, -a, -1 + a])
    Q[2, :] = np.array([1, b, -1 - b])
    samples_transformed = np.einsum('ij,jk->ik', Q, samples_)
    plt.scatter(samples_transformed[0, :], samples_transformed[1, :])
    plt.xlabel(r'$U_1$')
    plt.ylabel(r'$U_2$')
    plt.title(r"Samples after linear transformation with $a=1.2$ and $b=2.5$")
    plt.tight_layout()
    plt.show()

    plt.scatter(samples_transformed[0, :], samples_transformed[2, :])
    plt.xlabel(r'$U_1$')
    plt.ylabel(r'$U_3$')
    plt.title(r"Samples after linear transformation with $a=1.2$ and $b=2.5$")
    plt.tight_layout()
    plt.show()

    plt.scatter(samples_transformed[1, :], samples_transformed[2, :])
    plt.xlabel(r'$U_2$')
    plt.ylabel(r'$U_3$')
    plt.title(r"Samples after linear transformation with $a=1.2$ and $b=2.5$")
    plt.tight_layout()
    plt.show()

    Q = np.empty((3, 3))
    a = 2.5
    b = 2
    Q[0, :] = weights
    Q[1, :] = np.array([1, -a, -1 + a])
    Q[2, :] = np.array([1, b, -1 - b])
    samples_transformed = np.einsum('ij,jk->ik', Q, samples_)
    plt.scatter(samples_transformed[0, :], samples_transformed[1, :])
    plt.xlabel(r'$U_1$')
    plt.ylabel(r'$U_2$')
    plt.title(r"Samples after linear transformation with $a=2.5$ and $b=2$")
    plt.tight_layout()
    plt.show()

    plt.scatter(samples_transformed[0, :], samples_transformed[2, :])
    plt.xlabel(r'$U_1$')
    plt.ylabel(r'$U_3$')
    plt.title(r"Samples after linear transformation with $a=2.5$ and $b=2$")
    plt.tight_layout()
    plt.show()

    plt.scatter(samples_transformed[1, :], samples_transformed[2, :])
    plt.xlabel(r'$U_2$')
    plt.ylabel(r'$U_3$')
    plt.title(r"Samples after linear transformation with $a=2.5$ and $b=2$")
    plt.tight_layout()
    plt.show()
'''
nodes = np.array([1, 5, 25])
weights = np.array([1, 2, 3])
sample = rHestonMarkovSimulation.samples(lambda_=0.3, nu=0.3, theta=0.02, V_0=0.02, T=100., m=1, N_time=100000,
                                    nodes=nodes, weights=weights, vol_only=True, sample_paths=True,
                                    rho=-0.7, S_0=1., r=0., qmc=False, return_times=1000)[0][1:, 0, :]
t = np.linspace(0, 100, 1001)
plt.plot(t, sample[0, :])
plt.plot(t, sample[1, :])
plt.plot(t, sample[2, :])
plt.show()

plot_samples_volatility_markov_3d()

# Data.illustrate_bermudan_option_prices()

Data.plot_for_simulation_paper_smile_errors()
'''
H = 0.1
rho, nu, theta, V_0, lambda_, S_0 = -0.7, 0.3, 0.02, 0.02, 0.3, 1.
'''
H = 0.1
rel_tol = 1e-05
T = 1.
K = np.exp(np.linspace(-1.5, 0.75, 201))
N = np.array([1])  # np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
L1, L2 = np.empty((3, len(N))), np.empty((3, len(N)))
true_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, H=H)
for i in range(len(N)):
    print(i)
    nodes, weights = rk.quadrature_rule(H=H, N=N[i], T=T, mode='OL1')
    L1[0, i] = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T)[0]
    L1[1, i] = rk.error_l2(H=H, nodes=nodes, weights=weights, T=T)
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T, rel_tol=rel_tol, H=H, nodes=nodes, weights=weights)
    L1[2, i] = np.amax(np.abs(true_smile - approx_smile) / true_smile)

    nodes, weights = rk.quadrature_rule(H=H, N=N[i], T=T, mode='OL2')
    L2[0, i] = rk.error_l1(H=H, nodes=nodes, weights=weights, T=T)[0]
    L2[1, i] = rk.error_l2(H=H, nodes=nodes, weights=weights, T=T)
    approx_smile = rHestonFourier.eur_call_put(S_0=S_0, K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                               rel_tol=rel_tol, H=H, nodes=nodes, weights=weights)
    L2[2, i] = np.amax(np.abs(true_smile - approx_smile) / true_smile)

L1[0, :] = L1[0, :] / rk.kernel_norm(H=H, T=T, p=1.)
L2[0, :] = L2[0, :] / rk.kernel_norm(H=H, T=T, p=1.)
L1[1, :] = L1[1, :] / rk.kernel_norm(H=H, T=T, p=2.)
L2[1, :] = L2[1, :] / rk.kernel_norm(H=H, T=T, p=2.)

L1 = np.array([[1.63545001e-01, 5.73515295e-02, 2.39322992e-02, 1.11256367e-02,
        5.57139953e-03, 2.94864851e-03, 1.62979499e-03, 9.33230194e-04,
        5.75666147e-04, 3.75875939e-04],
       [5.30599323e-01, 3.49041452e-01, 2.47987474e-01, 1.84786594e-01,
        1.42499642e-01, 1.12322129e-01, 9.01983224e-02, 7.34771212e-02,
        5.46627181e-02, 4.17045503e-02],
       [6.96123849e-02, 1.87888397e-02, 6.71545691e-03, 2.77608995e-03,
        1.27131417e-03, 6.22275534e-04, 3.22276572e-04, 1.74123924e-04,
        7.09526680e-05, 3.60065071e-05]])
L2 = np.array([[0.26421043, 0.17125002, 0.11929625, 0.08644291, 0.06431098,
        0.04881763, 0.03767556, 0.02949276, 0.02337723, 0.01873652],
       [0.45700599, 0.20974588, 0.10854779, 0.06042393, 0.03540628,
        0.02157152, 0.01355739, 0.00874169, 0.0057599 , 0.00386659],
       [0.00897197, 0.00606565, 0.005255  , 0.00412311, 0.00313188,
        0.00234774, 0.00175021, 0.00130269, 0.00097046, 0.00072479]])

print((L1, L2))
L1 = L1 / L1[:, :1]
L2 = L2 / L2[:, :1]
N = np.arange(1, 11)
plt.loglog(N, L1[0, :], label=r'$L^1$-error')
plt.loglog(N, L1[1, :], label=r'$L^2$-error')
plt.loglog(N, L1[2, :], label=r'Weak error')
plt.xlabel('Number of dimensions N')
plt.title(r'Errors using $L^1$-optimized kernels')
plt.legend(loc='best')
plt.show()

plt.loglog(N, L2[0, :], label=r'$L^1$-error')
plt.loglog(N, L2[1, :], label=r'$L^2$-error')
plt.loglog(N, L2[2, :], label=r'Weak error')
plt.xlabel('Number of dimensions N')
plt.title(r'Errors using $L^2$-optimized kernels')
plt.legend(loc='best')
plt.show()
'''

# Data.plot_computed_skews()

# Data.convergence_rates_various_theorems()

# Data.plot_weak_errors_of_various_quadrature_rules()
'''
# T = np.linspace(0, 1, 17)[1:]
T = 1.
params = {'S': 1., 'K': np.exp(np.linspace(-0.1, 0.05, 16)), 'H': 0.1, 'T': T, 'lambda': 0.3, 'rho': -0.7,
                     'nu': 0.3, 'theta': 0.02, 'V_0': 0.02, 'rel_tol': 1e-05, 'r': 0.}

compute_smiles_given_stock_prices_QMC(params=params, Ns=np.array([3]),
                                      N_times=np.array([1024]),
                                      n_samples=2 ** 22, option='european call', simulator=['QE'], verbose=2)
'''
'''
T = np.linspace(0, 1, 17)[1:]
# T = 1.
params = {'S': 1., 'K': np.exp(np.linspace(-0.1, 0.05, 16)), 'H': 0.1, 'T': T, 'lambda': 0.3, 'rho': -0.7,
                     'nu': 0.3, 'theta': 0.02, 'V_0': 0.02, 'rel_tol': 1e-05, 'r': 0.}

compute_smiles_given_stock_prices_QMC(params=params, Ns=np.array([2, 3]),
                                      N_times=np.array([16, 32, 64, 128, 256, 512, 1024]),
                                      n_samples=2 ** 17, option='surface', simulator=['Weak'],
                                      verbose=2)
'''

T = np.linspace(0, 1, 17)[1:]
T = 1.
params = {'S': 1., 'K': np.exp(np.linspace(-0.1, 0.05, 16)), 'H': 0.1, 'T': T, 'lambda': 0.3, 'rho': -0.7,
                     'nu': 0.3, 'theta': 0.02, 'V_0': 0.02, 'rel_tol': 1e-05, 'r': 0.}

'''
samples, rng, kernel_dict = rHestonQESimulation.samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho,
                                                        S_0=S_0, r=0., m=2 ** 22, N_time=1024, sample_paths=True,
                                                        verbose=2)
filename = 'QE paths 1024 time steps batch 1.npy'
np.save(filename, samples)
for i in range(2, 26):
    print(f'Now generating batch {i}')
    samples, rng, kernel_dict = rHestonQESimulation.samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T,
                                                            rho=rho, S_0=S_0, r=0., m=2 ** 22, N_time=1024,
                                                            sample_paths=True, rng=rng, kernel_dict=kernel_dict,
                                                            rv_shift=True, verbose=2)
    np.save(f'QE paths 1024 time steps batch {i}.npy')
print('Finished!')
time.sleep(3600000)
'''
T = 1.
'''
for N in [2, 3]:
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        nodes, weights = rk.quadrature_rule(H=H, T=T, N=N)
        tic = time.perf_counter()
        rHestonMarkovSimulation.samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights,
                                        rho=rho, S_0=S_0, r=0., m=2 ** 22, N_time=M, euler=True, qmc=False)
        print(f'Euler, N={N}, M={M}, time={25*(time.perf_counter() - tic)}')

        tic = time.perf_counter()
        rHestonMarkovSimulation.samples(lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, nodes=nodes, weights=weights,
                                        rho=rho, S_0=S_0, r=0., m=2 ** 22, N_time=M, euler=False, qmc=False)
        print(f'Weak, N={N}, M={M}, time={25*(time.perf_counter() - tic)}')

        if N == 2:
            rHestonQESimulation.samples(H=H, lambda_=lambda_, nu=nu, theta=theta, V_0=V_0, T=T, rho=rho, S_0=S_0, r=0.,
                                        m=2 ** 22, N_time=M, qmc=False)
'''

params['rel_tol'] = 2e-04
compute_smiles_given_stock_prices_QMC(params=params, Ns=np.array([3]),
                                      N_times=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
                                      n_samples=2 ** 22, option='geometric asian call', simulator=['Euler', 'Weak'],
                                      verbose=2)

print('Finished!')
time.sleep(360000)


T, rel_tol, verbose = 1., 1e-04, 3
k = np.linspace(-1.5, 0.75, 201) * np.sqrt(T)
K = np.exp(k)

'''
Data.illustrate_bermudan_option_prices()
'''


H = 0.1
T, rel_tol, verbose = 1., 1e-05, 2
r = 0.06
N = 2
k = np.linspace(-0.1, 0.05, 16)
K = np.exp(k)
K = 1.05
# T = np.linspace(0, 1, 17)[1:]
# T = np.array([0.5, 1.])
# T = np.array([1.])
verbose = 1

params = {'S': 1., 'K': np.array([K]), 'H': H, 'T': T, 'lambda': lambda_, 'rho': rho, 'nu': nu, 'theta': theta, 'V_0': V_0,
          'rel_tol': rel_tol, 'r': r}
'''

for N_time in [2048]:
    est, stat, _, _, _, _ = \
        rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T,
                                     payoff='put', r=r, m=1_000_000, N_time=N_time, N_dates=4,
                                     feature_degree=8, antithetic=False)
    print('QE, 4 dates, ', N_time, 100 * est, 100 * stat)

for N_time in [2048]:
    est, stat, _, _, _, _ = \
        rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T,
                                     payoff='put', r=r, m=1_000_000, N_time=N_time, N_dates=16,
                                     feature_degree=8, antithetic=False)
    print('QE, 16 dates, ', N_time, 100 * est, 100 * stat)

for N_time in [2048]:
    est, stat, _, _, _, _ = \
        rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, S_0=S_0, T=T,
                                     payoff='put', r=r, m=1_000_000, N_time=N_time, N_dates=256,
                                     feature_degree=8, antithetic=False)
    print('QE, 256 dates, ', N_time, 100 * est, 100 * stat)
'''

'''
for N_time in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    if N_time >= 512:
        est, stat = rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                 S_0=S_0, T=T, payoff='put', r=r, m=2 ** 18, N_time=N_time, N_dates=4,
                                                 feature_degree=6, qmc=True, qmc_error_estimators=25, verbose=1)
        print('QE, 4 dates, ', N_time, 100 * est, 100 * stat)

    if N_time >= 512:
        est, stat = rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                 S_0=S_0, T=T, payoff='put', r=r, m=2 ** 18, N_time=N_time, N_dates=16,
                                                 feature_degree=6, qmc=True, qmc_error_estimators=25, verbose=1)
        print('QE, 16 dates, ', N_time, 100 * est, 100 * stat)
    if N_time >= 512:
        est, stat = rHestonQESimulation.price_am(K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                 S_0=S_0,
                                                 T=T, payoff='put', r=r, m=2 ** 17, N_time=N_time, N_dates=256,
                                                 feature_degree=6, qmc=True, qmc_error_estimators=50, verbose=1)
        print('QE, 256 dates, ', N_time, 100 * est, 100 * stat)
print('Finished')
time.sleep(360000)
verbose = 0
for N in [4]:
    nodes, weights = rk.quadrature_rule(H=H, N=N, T=T, mode='BL2')
    print(rHestonFourier.eur_call_put(S_0=100, K=np.array([100, 105, 110]), lambda_=lambda_, rho=rho, nu=nu,
                                      theta=theta, V_0=V_0,
                                      T=T, call=False, rel_tol=1e-05, implied_vol=False, r=r, nodes=nodes,
                                      weights=weights))

    for N_time in [256, 512, 1024, 2048]:
        for d in [6]:
            if N_time >= 2048:
                tic = time.perf_counter()
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=4, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=True, verbose=verbose)
                print(time.perf_counter() - tic)
                print('Euler, 4 dates, ', N, N_time, d, 100 * est, 100 * stat)
            if N_time >= 1024:
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=4, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=False, verbose=verbose)
                print('Weak, 4 dates, ', N, N_time, d, 100 * est, 100 * stat)
            if N_time >= 1024:
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=16, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=True, verbose=verbose)
                print('Euler, 16 dates, ', N, N_time, d, 100 * est, 100 * stat)
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=16, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=False, verbose=verbose)
                print('Weak, 16 dates, ', N, N_time, d, 100 * est, 100 * stat)
            if N_time >= 256:
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=256, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=True, verbose=verbose)
                print('Euler, 256 dates, ', N, N_time, d, 100 * est, 100 * stat)
                est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                             S_0=S_0, T=T, payoff='put', r=r, m=2 ** 20, N_time=N_time,
                                                             N_dates=256, feature_degree=d, qmc=True, nodes=nodes,
                                                             weights=weights, euler=False, verbose=verbose)
                print('Weak, 256 dates, ', N, N_time, d, 100 * est, 100 * stat)
'''
'''
    for N_time in [1024]:
        est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                     S_0=S_0, T=T, payoff='put', r=r, m=1_000_000, N_time=N_time,
                                                     N_dates=256, feature_degree=8, qmc=True, nodes=nodes,
                                                     weights=weights, euler=True)
        print('Euler, 256 dates, ', N_time, 100 * est, 100 * stat)
        est, stat = rHestonMarkovSimulation.price_am(K=K, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                                     S_0=S_0, T=T, payoff='put', r=r, m=1_000_000, N_time=N_time,
                                                     N_dates=256, feature_degree=8, qmc=True, nodes=nodes,
                                                     weights=weights, euler=False)
        print('Weak, 256 dates, ', N_time, 100 * est, 100 * stat)
        '''
