import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import qdarkstyle.style_rc
import werkzeug.http

import Data
import rBergomi
from functions import *
import rBergomiMarkov
import rHestonMomentMatching


nodes, weights = rk.quadrature_rule(H=0.1, N=3, T=1., mode='european')
print(nodes, weights)
V_0 = 0.02
V_0_vec = V_0 / (len(weights) * weights)
nu = 0.3
lambda_ = 0.3
theta = 0.02 * lambda_
V = V_0_vec * np.array([0.5, 1, 3])
dt = 0.1
mean_func = rHestonMomentMatching.mean_V(nodes=nodes, weights=weights, lambda_=lambda_, theta=theta, V_0=V_0_vec, dt=dt)
tic = time.perf_counter()
cov_func = rHestonMomentMatching.cov_V(nodes=nodes, weights=weights, lambda_=lambda_, theta=theta, nu=nu, V_0=V_0_vec, dt=dt)
print(f'Benchmark: {time.perf_counter() - tic}')
print(V_0_vec, weights @ V_0_vec)
print(V, weights @ V)
print(mean_func(V), weights @ mean_func(V))
print(cov_func(V))
time.sleep(360000)


def illustrate_Markovian_approximation(H=0.3, N_small=2, N_large=4, T=1., n=10000):
    nodes, weights = rk.quadrature_rule(H=H, N=N_small, T=T, mode='optimized')
    print(nodes, weights)
    nodes = np.array([1., 400])
    weights = np.array([1.6, 8])
    exact_nodes, exact_weights = rk.quadrature_rule(H=H, N=N_large, T=T, mode="optimized")
    print(exact_nodes, exact_weights)
    dt = T / n

    B = np.random.normal(0, np.sqrt(dt), n)
    exp_nodes = np.exp(-nodes * dt)
    div_nodes = (1 - np.exp(-2 * nodes * dt)) / (2 * nodes * dt)
    OU_1 = np.zeros((N_small, n + 1))
    for i in range(n):
        OU_1[:, i + 1] = exp_nodes * OU_1[:, i] + div_nodes * B[i]

    exp_nodes = np.exp(-exact_nodes * dt)
    div_nodes = (1 - np.exp(-2 * exact_nodes * dt)) / (2 * exact_nodes * dt)
    OU_2 = np.zeros((N_large, n + 1))
    for i in range(n):
        OU_2[:, i + 1] = exp_nodes * OU_2[:, i] + div_nodes * B[i]

    exact_fBm = np.einsum('i,ij->j', exact_weights, OU_2)
    approx_fBm = np.einsum('i,ij->j', weights, OU_1)

    time_vec = np.linspace(0, T, n + 1)
    BM = np.zeros(n + 1)
    BM[1:] = np.cumsum(B)
    plt.plot(time_vec, exact_fBm, 'k-', label='Fractional Brownian motion')
    plt.plot(time_vec, approx_fBm, 'g-', label='Markovian approximation')
    plt.plot(time_vec, weights[0] * OU_1[0, :], 'b-', label='Slow component')
    plt.plot(time_vec, weights[-1] * OU_1[-1, :], 'r-', label='Fast component')
    # plt.plot(time_vec, BM, color='grey', label='Underlying Brownian motion')
    plt.legend(loc='best')
    plt.show()

"""
largest_node = np.empty(10)
for i in range(1, 7):
    largest_node[i-1] = rk.quadrature_rule(H=0.1, T=1., N=i, mode='european')[0][-1]
print(largest_node)
time.sleep(36000)
"""
'''
params = rHeston_params('simple')
params['T'] = 0.04
params['K'] = np.exp(np.sqrt(params['T']) * np.log(params['K']))
# true_smile = rHeston_iv_eur_call(params, load=False, verbose=2)
# print((true_smile,))
true_smile = np.array([0.39928947, 0.39868438, 0.39807835, 0.39747139, 0.3968635 ,
       0.39625466, 0.39564488, 0.39503415, 0.39442247, 0.39380982,
       0.39319622, 0.39258165, 0.3919661 , 0.39134958, 0.39073208,
       0.3901136 , 0.38949412, 0.38887365, 0.38825218, 0.3876297 ,
       0.38700621, 0.38638171, 0.38575619, 0.38512965, 0.38450208,
       0.38387347, 0.38324382, 0.38261313, 0.38198139, 0.3813486 ,
       0.38071474, 0.38007982, 0.37944383, 0.37880676, 0.37816861,
       0.37752937, 0.37688904, 0.37624761, 0.37560508, 0.37496144,
       0.37431669, 0.37367081, 0.37302381, 0.37237567, 0.3717264 ,
       0.37107598, 0.37042441, 0.36977168, 0.36911779, 0.36846273,
       0.3678065 , 0.36714909, 0.36649049, 0.36583069, 0.36516969,
       0.36450749, 0.36384407, 0.36317944, 0.36251357, 0.36184648,
       0.36117814, 0.36050855, 0.35983771, 0.35916561, 0.35849224,
       0.3578176 , 0.35714167, 0.35646445, 0.35578594, 0.35510612,
       0.35442499, 0.35374254, 0.35305876, 0.35237365, 0.35168719,
       0.35099939, 0.35031022, 0.34961969, 0.34892779, 0.3482345 ,
       0.34753982, 0.34684375, 0.34614626, 0.34544736, 0.34474704,
       0.34404528, 0.34334208, 0.34263743, 0.34193132, 0.34122374,
       0.34051468, 0.33980414, 0.3390921 , 0.33837855, 0.33766349,
       0.33694691, 0.33622879, 0.33550913, 0.33478792, 0.33406514,
       0.33334079, 0.33261485, 0.33188732, 0.33115818, 0.33042743,
       0.32969506, 0.32896105, 0.32822539, 0.32748807, 0.32674908,
       0.32600842, 0.32526606, 0.32452199, 0.32377622, 0.32302871,
       0.32227947, 0.32152848, 0.32077573, 0.3200212 , 0.31926488,
       0.31850676, 0.31774683, 0.31698508, 0.31622149, 0.31545604,
       0.31468873, 0.31391955, 0.31314847, 0.31237549, 0.31160058,
       0.31082375, 0.31004496, 0.30926422, 0.30848149, 0.30769678,
       0.30691006, 0.30612132, 0.30533054, 0.30453771, 0.3037428 ,
       0.30294582, 0.30214673, 0.30134553, 0.3005422 , 0.29973671,
       0.29892906, 0.29811923, 0.29730719, 0.29649293, 0.29567644,
       0.29485769, 0.29403667, 0.29321335, 0.29238773, 0.29155977,
       0.29072947, 0.28989679, 0.28906173, 0.28822426, 0.28738436,
       0.28654201, 0.28569719, 0.28484987, 0.28400004, 0.28314768,
       0.28229276, 0.28143526, 0.28057515, 0.27971242, 0.27884704,
       0.27797898, 0.27710823, 0.27623475, 0.27535853, 0.27447953,
       0.27359774, 0.27271312, 0.27182565, 0.27093531, 0.27004205,
       0.26914587, 0.26824673, 0.2673446 , 0.26643945, 0.26553125,
       0.26461998, 0.2637056 , 0.26278808, 0.26186739, 0.2609435 ,
       0.26001638, 0.25908599, 0.25815231, 0.25721529, 0.2562749 ,
       0.25533111, 0.25438389, 0.25343319, 0.25247898, 0.25152122,
       0.25055987, 0.2495949 , 0.24862627, 0.24765393, 0.24667785,
       0.24569799, 0.24471429, 0.24372673, 0.24273525, 0.24173982,
       0.24074038, 0.23973689, 0.23872931, 0.23771759, 0.23670167,
       0.23568152, 0.23465707, 0.23362828, 0.23259511, 0.23155748,
       0.23051536, 0.22946868, 0.2284174 , 0.22736145, 0.22630077,
       0.22523532, 0.22416502, 0.22308981, 0.22200965, 0.22092445,
       0.21983416, 0.21873872, 0.21763804, 0.21653208, 0.21542075,
       0.214304  , 0.21318173, 0.21205389, 0.2109204 , 0.20978119,
       0.20863617, 0.20748527, 0.2063284 , 0.2051655 , 0.20399647,
       0.20282123, 0.2016397 , 0.20045179, 0.19925741, 0.19805648,
       0.1968489 , 0.19563458, 0.19441344, 0.19318537, 0.19195028,
       0.19070808, 0.18945866, 0.18820194, 0.18693781, 0.18566618,
       0.18438694, 0.18309999, 0.18180525, 0.18050259, 0.17919194,
       0.17787319, 0.17654625, 0.17521102, 0.17386741, 0.17251533,
       0.17115469, 0.16978542, 0.16840744, 0.16702067, 0.16562506,
       0.16422056, 0.16280711, 0.16138469, 0.15995328, 0.15851288,
       0.15706351, 0.15560519, 0.15413801, 0.15266203, 0.1511774 ,
       0.14968426, 0.14818283, 0.14667336, 0.14515615, 0.14363159,
       0.14210011, 0.14056225, 0.13901863, 0.13747001, 0.13591722,
       0.13436129, 0.13280335, 0.13124477, 0.12968706, 0.128132  ,
       0.12658158, 0.1250381 , 0.12350412, 0.12198252, 0.12047654,
       0.11898973, 0.11752601, 0.11608962, 0.11468513, 0.11331735,
       0.1119913 , 0.11071211, 0.10948488, 0.10831459, 0.10720591,
       0.10616304, 0.10518961, 0.10428851, 0.10346181, 0.10271071,
       0.1020355 , 0.10143563, 0.10090975, 0.10045583, 0.10007126,
       0.099753  , 0.09949768, 0.09930171, 0.09916142, 0.09907311,
       0.09903313, 0.09903792, 0.0990841 , 0.09916842, 0.09928782,
       0.09943946, 0.09962066, 0.09982895, 0.10006206, 0.10031786,
       0.10059442, 0.10088996, 0.10120284, 0.10153156, 0.10187475,
       0.10223114, 0.10259959, 0.10297903, 0.1033685 , 0.10376712,
       0.10417407, 0.10458861, 0.10501005, 0.10543776, 0.10587119,
       0.10630978, 0.10675306, 0.10720059, 0.10765194, 0.10810674,
       0.10856465, 0.10902534, 0.10948852, 0.10995391, 0.11042126,
       0.11089033, 0.11136092, 0.11183281, 0.11230584, 0.11277982,
       0.11325461, 0.11373004, 0.114206  , 0.11468235, 0.11515897,
       0.11563577, 0.11611263, 0.11658948, 0.11706621, 0.11754276,
       0.11801905, 0.11849501, 0.11897057, 0.11944568, 0.11992029,
       0.12039434, 0.12086779, 0.12134059, 0.12181271, 0.1222841 ,
       0.12275473, 0.12322458, 0.1236936 , 0.12416178, 0.12462909,
       0.12509551, 0.12556101, 0.12602558, 0.1264892 , 0.12695185,
       0.12741352, 0.12787419, 0.12833386, 0.1287925 , 0.12925012,
       0.1297067 , 0.13016223, 0.13061671, 0.13107013, 0.13152248,
       0.13197376, 0.13242397, 0.1328731 , 0.13332115, 0.13376812,
       0.134214  , 0.1346588 , 0.1351025 , 0.13554512, 0.13598665,
       0.13642709, 0.13686645, 0.13730472, 0.1377419 , 0.138178  ,
       0.13861302, 0.13904696, 0.13947982, 0.13991161, 0.14034232,
       0.14077196, 0.14120054, 0.14162806, 0.14205451, 0.14247991,
       0.14290426, 0.14332756, 0.14374982, 0.14417103, 0.14459121,
       0.14501035, 0.14542847, 0.14584556, 0.14626164, 0.1466767 ,
       0.14709075, 0.14750379, 0.14791583, 0.14832688, 0.14873694,
       0.149146  ])
'''
'''
for i in range(6):
    our_approx = rHestonMarkov_iv_eur_call(params=params, N=i+1, mode='paper', load=True, verbose=1)
    plt.plot(np.log(params['K']), our_approx, color=color(i, 6), label=f'N={i+1}')
plt.plot(np.log(params['K']), true_smile, 'k-', label='True smile')
plt.legend(loc='best')
plt.xlabel('Log-moneyness')
plt.ylabel('Implied volatility')
plt.show()
'''
'''
params['rel_tol'] = 1e-03
for i in range(6):
    plt.plot(np.log(params['K']), rHestonMarkov_iv_eur_call(params=params, N=4 ** i, mode='abi jaber', verbose=1), color=color(i, 6), label=f'Abi Jaber, El Euch, N={4 ** i}')
plt.plot(np.log(params['K']), rHestonMarkov_iv_eur_call(params=params, N=6, mode='paper', verbose=1), color='brown', label='Our approximation, N=6')
plt.plot(np.log(params['K']), true_smile, 'k-', label='True smile')
plt.legend(loc='best')
plt.xlabel('Log-moneyness')
plt.ylabel('Implied volatility')
plt.show()
'''
'''
for i in range(6):
    errors = np.abs(rHestonMarkov_iv_eur_call(params=params, N=i+1, mode='european', load=True, verbose=1) - true_smile)/true_smile
    plt.plot(np.log(params['K']), errors, color=color(i, 6), label=f'N={i+1}')
plt.plot(np.log(params['K']), 2e-05 * np.ones(len(params['K'])), 'k--', label='Discretization error')
plt.plot(np.log(params['K']), 2e-08 * np.ones(len(params['K'])), 'k:')

plt.legend(loc='best')
plt.xlabel('Log-moneyness')
plt.ylabel('Relative error of implied volatility')
plt.title('Relative error of implied volatility using bounded nodes')
plt.show()
'''
'''
errors = np.empty((4, 6))
modes = ['abi jaber', 'paper', 'optimized', 'european']
for i in range(len(modes)):
    for j in range(6):
        print(i, j)
        time.sleep(60)
        errors[i, j] = np.amax(np.abs(rHestonMarkov_iv_eur_call(params=params, N=j+1, mode=modes[i], load=True, verbose=1) - true_smile) / true_smile)
plt.plot(np.arange(1, 7), errors[0, :], 'k-', label='Abi Jaber, El Euch')
plt.plot(np.arange(1, 7), errors[1, :], 'b-', label='Paper')
plt.plot(np.arange(1, 7), errors[2, :], 'r-', label='Optimized')
plt.plot(np.arange(1, 7), errors[3, :], 'g-', label='Bounded')
plt.plot(np.arange(1, 7), 2e-05 * np.ones(6), 'k--', label='Discretization error')
plt.xlabel('Number of nodes N')
plt.ylabel('Maximal relative error of implied volatility')
plt.legend(loc='upper right')
plt.show()
'''
'''
illustrate_Markovian_approximation(H=0.2, T=1., n=300)
print('Finished')
time.sleep(360000)
'''

'''
T = 1.
k = np.linspace(-0.4, 0.4, 301) * np.sqrt(T)
tic = time.perf_counter()
smile, lower, upper = rBergomi.implied_volatility(K=np.exp(k), rel_tol=9e-02, T=T, verbose=1)
total = np.empty((3, len(smile)))
total[0, :] = smile
total[1, :] = lower
total[2, :] = upper
# np.save('rBergomi actual.npy', total)
print(time.perf_counter() - tic)
# print((smile, lower, upper))
plt.plot(k, smile, 'k-')
plt.plot(k, lower, 'k--')
plt.plot(k, upper, 'k--')
# plt.show()
total = np.load('rBergomi actual.npy')
plt.plot(k, total[0, :], 'k-')
plt.plot(k, total[1, :], 'k--')
plt.plot(k, total[2, :], 'k--')
'''
'''
functions.rHeston_iv_eur_call(params=functions.rHeston_params('simple'))
functions.rHestonMarkov_iv_eur_call(params=functions.rHeston_params('simple'), N=2, mode='european')
k = np.linspace(-0.4, 0.4, 301)
rBergomi.implied_volatility(rel_tol=1e-02, K=np.exp(k))
rBergomiMarkov.implied_volatility(rel_tol=1e-02, K=np.exp(k), N=2, mode='optimized')
disc = 0.7e-02 + 1.5 * (total[2, :] - total[1, :]) / total[1, :]
plt.plot(k, disc, 'k--', label='Discretization + MC error')
for i in np.array([0, 1, 2, 3, 4, 5]):
    approx = np.load(f'rBergomi actual N={i+1}.npy')
    # plt.plot(k, approx[0, :], '-', color=color(i, 6))
    err = np.abs(total[0, :] - approx[0, :]) / total[0, :]
    plt.plot(k, err, '-', color=color(i, 6), label=f'N={i+1}')
plt.yscale('log')
plt.title('Relative error in rough Bergomi implied volatility\nof Markovian approximations depending on dimension')
plt.xlabel('Log-moneyness')
plt.ylabel('Relative error')
plt.legend(loc='best')
plt.show()
for N in np.array([2]):
    print(N)
    tic = time.perf_counter()
    smile, l, u = rBergomiMarkov.implied_volatility(K=np.exp(k), mode='optimized', N=N, T=T, rel_tol=2e-02, verbose=1)
    # plt.plot(k, smile, '-', color=color(N, 4))
    total = np.empty((3, len(smile)))
    total[0, :] = smile
    total[1, :] = l
    total[2, :] = u
    np.save(f'rBergomi actual N={N}.npy', total)
    print(time.perf_counter() - tic)
    # print((smile, lower, upper))
# plt.show()
print('Finished')
time.sleep(360000)
'''
'''
tic = time.perf_counter()
smile, l, u = rBergomiMarkov.implied_volatility(K=np.exp(k), mode='optimized', N=3, T=T, rel_tol=1e-02, verbose=1)
# plt.plot(k, smile, 'g-')
print(time.perf_counter() - tic)
print((smile, lower, upper))
tic = time.perf_counter()
smile, l, u = rBergomiMarkov.implied_volatility(K=np.exp(k), mode='european', N=3, T=T, rel_tol=1e-02, verbose=1)
# plt.plot(k, smile, 'b-')
print(time.perf_counter() - tic)
print((smile, lower, upper))
print('Finished')
time.sleep(36000)
# plt.show()
'''
'''
params = {'H': 0.05, 'lambda': 0.2, 'rho': -0.6, 'nu': 0.6,
          'theta': 0.01, 'V_0': 0.01, 'S': 1., 'K': np.exp(np.linspace(-1, 0.5, 301)),
          'T': np.linspace(0.04, 1., 25), 'rel_tol': 1e-05}
print(params)
rHeston_iv_eur_call(params=params, load=True, save=True, verbose=1)
print('Finished!')
time.sleep(3600000)

tic = time.perf_counter()
print(rHeston_iv_eur_call(params=rHeston_params('simple'), load=False, save=False, verbose=1))
print(time.perf_counter() - tic)
print('finished')
time.sleep(36000)
'''

'''tic = time.perf_counter()
params = rHeston_params('simple')
rHeston_iv_eur_call(params)
print('time', time.perf_counter()-tic)
time.sleep(36000)
'''

# rHeston_iv_eur_call(params=rHeston_params('simple'), load=False, save=False, verbose=1)

if __name__ == '__main__':
    # 'nu': log_linspace(0.2, 0.6, 2),
    # 'theta': log_linspace(0.01, 0.03, 2),
    # 'V_0': log_linspace(0.01, 0.03, 2)
    # 'lambda': np.array([0.2, 1.0])
    # 'rho': np.array([-0.6, -0.8]),
    '''
    params = {'H': np.array([0.15]), 'lambda': np.array([0.2, 1.0]), 'rho': np.array([-0.6, -0.8]),
              'nu': np.array([0.8]), 'theta': np.array([0.01, 0.03]), 'V_0': np.array([0.01, 0.03])}
    for i in range(25):
        params['T'] = (i+1)/25
        params['K'] = np.exp(np.linspace(-1., 0.5, 301) * np.sqrt(params['T']))
        # print(params)
        rHestonMarkov_iv_eur_call_parallelized(params=params, Ns=np.arange(1, 11), modes=['paper', 'optixmized', 'european'], num_threads=1, verbose=1)
    '''
    '''
    params = {'H': np.array([0.1]), 'lambda': np.array([0.2, 1.0]), 'rho': np.array([-0.6, -0.8]),
              'nu': np.array([0.2]), 'theta': np.array([0.03]), 'V_0': np.array([0.03]),
              'T': np.linspace(0.04, 1., 25), 'K': np.exp(np.linspace(-1., 0.5, 301))}


    rHestonMarkov_iv_eur_call_parallelized(params=params, Ns=np.arange(1, 11),
                                           modes=['paper', 'optimized', 'european'], num_threads=1, verbose=1)
    '''
    '''
    params = {'H': np.array([0.15]), 'lambda': np.array([0.2, 1.0]), 'rho': np.array([-0.6, -0.8]),
              'nu': np.array([0.2, 0.4, 0.6, 0.8]), 'theta': np.array([0.01, 0.03]), 'V_0': np.array([0.01, 0.03]),
              'T': np.linspace(0.04, 1., 25), 'K': np.exp(np.linspace(-1., 0.5, 301))}

    rHeston_iv_eur_call_parallelized(params=params, num_threads=1, verbose=1)
    '''

print('Finished')

# compute_final_rHeston_stock_prices(params='simple', Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['paper', 'optimized', 'european'], vol_behaviours=['correct ninomiya victoir'], recompute=False)
# print('Finished')
# time.sleep(360000)
# print('Finished')
# time.sleep(360000)

print(rk.quadrature_rule(0.1, 2, 1))
k = np.sqrt(1) * np.linspace(-1.5, 0.75, 451)[220:-70]
params = {'K': np.exp(k), 'T': 1.}
params = rHeston_params(params)
true_smile = Data.true_iv_surface_eur_call[-1, 220:-70]
print(k, len(k))
# simulation_errors_depending_on_node_size(params=params, verbose=1, true_smile=true_smile, N_times=2**np.arange(4, 10), largest_nodes=np.linspace(0, 10, 101)/0.04, vol_behaviour='sticky')
# optimize_kernel_approximation_for_simulation_vector_inputs(Ns=np.array([1]), N_times=2 ** np.arange(6, 9), params=params, true_smile=true_smile, plot=True, recompute=True, vol_behaviours=['hyperplane reset'], m=10000000)

# compute_strong_discretization_errors(Ns=np.array([2]), N_times=2 ** np.arange(14), N_time_ref=2 ** 14, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(16), modes=None, vol_behaviours=['hyperplane reset', 'ninomiya victoir', 'sticky'], plot=True, true_smile=true_smile)
compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(16), modes=None, vol_behaviours=['sticky'], plot=True, true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european'], vol_behaviours=['ninomiya victoir', 'correct ninomiya victoir', 'sticky'], plot='vol_behaviour', true_smile=true_smile)
# compute_smiles_given_stock_prices(params=params, Ns=np.array([2]), N_times=2 ** np.arange(10), modes=['european', 'fitted'], vol_behaviours=['hyperplane reset'], plot='mode', true_smile=true_smile)
print('Finished')

