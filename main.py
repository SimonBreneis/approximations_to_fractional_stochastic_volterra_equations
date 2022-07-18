c = ['darkred', 'r', 'C1', 'y', 'lime', 'g', 'deepskyblue', 'b', 'purple', 'deeppink']

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
S, K, H, lambda_, rho, nu, theta, V_0, rel_tol = 1, np.exp(k_vec), 0.1, 0.3, -0.7, 0.3, 0.02, 0.02, 1e-05
T = np.linspace(0.04, 1., 25)
# k_vec = np.linspace(-0.5, 0.1, 101)
# S, K, H, lambda_, rho, nu, theta, V_0, T, rel_tol = 1, np.exp(k_vec), 0.07, 0.6, -0.8, 0.5, 0.01, 0.01, 0.04, 1e-04

largest_node = np.array([[3.32763862e+07, 1.66381931e+07, 1.10921287e+07, 8.31909656e+06,
        6.65527725e+06, 5.54606437e+06, 4.75376946e+06, 4.15954828e+06,
        3.69737625e+06, 3.32763862e+06, 3.02512602e+06, 2.77303219e+06,
        2.55972202e+06, 2.37688473e+06, 2.21842575e+06, 2.07977414e+06,
        1.95743448e+06, 1.84868812e+06, 1.75138875e+06, 1.66381931e+06,
        1.58458982e+06, 1.51256301e+06, 1.44679940e+06, 1.38651609e+06,
        1.33105545e+06],
       [4.77036024e+05, 2.38520358e+05, 1.59013162e+05, 1.19259704e+05,
        9.54081287e+04, 7.95066406e+04, 6.81483144e+04, 5.96298119e+04,
        5.29999630e+04, 4.77037067e+04, 4.33670771e+04, 3.97533059e+04,
        3.66960104e+04, 3.40738058e+04, 3.18027197e+04, 2.98142027e+04,
        2.80608605e+04, 2.65021553e+04, 2.51072086e+04, 2.38519660e+04,
        2.27160094e+04, 2.16833698e+04, 2.07407984e+04, 1.98768963e+04,
        1.90812219e+04],
       [1.89377558e+03, 9.46887840e+02, 6.31258502e+02, 4.73443878e+02,
        3.78755294e+02, 3.15629186e+02, 2.70539304e+02, 2.36721939e+02,
        2.10419515e+02, 1.89377594e+02, 1.72161414e+02, 1.57814605e+02,
        1.45675050e+02, 1.35269683e+02, 1.26251703e+02, 1.18360968e+02,
        1.11398561e+02, 1.05209865e+02, 9.96723954e+01, 9.46887745e+01,
        9.01797907e+01, 8.60807120e+01, 8.23380657e+01, 7.89073762e+01,
        7.57510270e+01]])
kernel_l_infinity = np.array([[1.96649963, 1.36339683, 1.09228965, 0.93230178, 0.82564424,
        0.74944564, 0.6925486 , 0.64875627, 0.61430636, 0.58676144,
        0.56446256, 0.54623638, 0.54566541, 0.54723797, 0.54854078,
        0.54961674, 0.55050025, 0.55121921, 0.55179651, 0.55225106,
        0.55259868, 0.55285265, 0.55302424, 0.55312305, 0.55315732],
       [0.66195922, 0.59552544, 0.54784999, 0.50867469, 0.47458403,
        0.44421507, 0.41703146, 0.39281528, 0.37144056, 0.35278481,
        0.33669223, 0.33196601, 0.33444584, 0.33667395, 0.33867597,
        0.34047661, 0.34209853, 0.34356255, 0.34488822, 0.34609269,
        0.34719191, 0.34820029, 0.34913045, 0.34999413, 0.35080206],
       [0.96124052, 0.63993795, 0.51242947, 0.45053948, 0.46065012,
        0.46909755, 0.47637828, 0.48278116, 0.48849056, 0.4936346 ,
        0.49830692, 0.50258059, 0.50651478, 0.51015797, 0.51355085,
        0.51672787, 0.51971786, 0.52254555, 0.52523284, 0.52779386,
        0.53024607, 0.53260158, 0.53486925, 0.53705827, 0.5391758 ]])
kernel_l_2 = np.array([[1.40010711, 0.97046608, 0.78792375, 0.68680658, 0.62389196,
        0.5821353 , 0.55328215, 0.53280731, 0.51801701, 0.50721131,
        0.4992699 , 0.49342913, 0.48915431, 0.48606263, 0.48387484,
        0.48238398, 0.48143452, 0.48090815, 0.48071391, 0.48078121,
        0.48105479, 0.4814911 , 0.48205555, 0.48272054, 0.48346392],
       [0.57027206, 0.4894851 , 0.43690332, 0.39849685, 0.36930119,
        0.34683713, 0.32957259, 0.31641064, 0.3064987 , 0.29915242,
        0.29381573, 0.29004074, 0.28746951, 0.28581901, 0.28486718,
        0.28444166, 0.28440932, 0.28466766, 0.28513847, 0.28576195,
        0.28649273, 0.28729656, 0.2881473 , 0.28902566, 0.28991749],
       [0.67827516, 0.48330514, 0.42234802, 0.3993012 , 0.39068766,
        0.38835033, 0.38893821, 0.39092826, 0.39357629, 0.39650658,
        0.39951627, 0.40250073, 0.40540659, 0.40820661, 0.4108902 ,
        0.41345551, 0.41590529, 0.41824486, 0.42048097, 0.42262026,
        0.42466975, 0.42663592, 0.42852492, 0.43034236, 0.43209352]])
smile_err = np.array([[0.35344619, 0.29088646, 0.2461029 , 0.2144478 , 0.21523023,
        0.22084815, 0.22428964, 0.22628439, 0.22727469, 0.22754492,
        0.22728645, 0.22663274, 0.22567963, 0.22449757, 0.22337635,
        0.22423035, 0.22483499, 0.22522905, 0.22544408, 0.22550605,
        0.22543644, 0.22525327, 0.2249717 , 0.22460459, 0.22416292],
       [0.31146992, 0.25086997, 0.20508167, 0.16989815, 0.1419354 ,
        0.11916956, 0.10032394, 0.0845348 , 0.07118483, 0.0598168 ,
        0.050079  , 0.0416977 , 0.03445417, 0.03110706, 0.02982506,
        0.03060562, 0.03148878, 0.03215019, 0.03261579, 0.03290732,
        0.0330446 , 0.03304466, 0.03292296, 0.03269391, 0.0323689 ],
       [0.13070527, 0.13123503, 0.130533  , 0.12958421, 0.12899442,
        0.1288475 , 0.12907192, 0.12957145, 0.13026044, 0.13107045,
        0.13195127, 0.1328594 , 0.13377479, 0.13467038, 0.13553673,
        0.13635957, 0.13713707, 0.13786166, 0.13855585, 0.13919564,
        0.13978152, 0.14031501, 0.14079704, 0.14122718, 0.14160698]])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(T, kernel_l_infinity[0, :], '--', color='red', label='Kernel error')
# ax1.plot(T, kernel_l_2[0, :], 'o-', color='red', label=r'$ker_2$')
ax1.plot(T, smile_err[0, :], color='red', label='Volatility error')
ax2.plot(T, largest_node[0, :], 'x-', color='red', label='Largest node')
ax1.plot(T, kernel_l_infinity[1, :], '--', color='green')
# ax1.plot(T, kernel_l_2[1, :], 'o-', color='green')
ax1.plot(T, smile_err[1, :], color='green')
ax2.plot(T, largest_node[1, :], 'x-', color='green')
ax1.plot(T, kernel_l_infinity[2, :], '--', color='blue')
# ax1.plot(T, kernel_l_2[2, :], 'o-', color='blue')
ax1.plot(T, smile_err[2, :], color='blue')
ax2.plot(T, largest_node[2, :], 'x-', color='blue')
ax1.set_xlabel('Time ' + r'$T_0$' + ' used for obtaining the Markovian approximation')
ax1.set_ylabel('Relative error')
ax2.set_ylabel('Largest node')
ax2.set_yscale('log')
ax1.set_yscale('log')
ax1.legend(loc='upper right')
ax2.legend(loc='right')
plt.show()

with open('true surface.npy', 'rb') as f:
    true_surface = np.load(f)

'''
tic = time.perf_counter()
true_surface = rHeston.iv_european_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                               rel_tol=rel_tol)
print("Duration:", time.perf_counter() - tic)
with open('true surface.npy', 'wb') as f:
    np.save(f, true_surface)
print("Finished")
time.sleep(360000)
'''
# print(np.amax(np.abs(true_surface-sur)/true_surface), time.perf_counter()-tic)
# print((true_surface,))

'''
true_surface = np.array([0.6193224 , 0.61516358, 0.61097976, 0.60677201, 0.60253891,
       0.59828059, 0.59399642, 0.58968578, 0.58534864, 0.58098384,
       0.57659171, 0.57217077, 0.56772152, 0.56324234, 0.55873372,
       0.55419403, 0.54962364, 0.54502102, 0.54038632, 0.53571816,
       0.53101644, 0.52627994, 0.5215083 , 0.51670045, 0.51185578,
       0.50697333, 0.50205229, 0.49709175, 0.49209071, 0.4870483 ,
       0.48196336, 0.47683499, 0.47166189, 0.46644312, 0.46117725,
       0.45586322, 0.45049953, 0.44508496, 0.43961792, 0.43409701,
       0.42852052, 0.42288689, 0.41719425, 0.41144082, 0.4056246 ,
       0.39974354, 0.39379544, 0.38777798, 0.38168874, 0.37552507,
       0.36928424, 0.36296326, 0.35655903, 0.35006813, 0.343487  ,
       0.33681174, 0.33003823, 0.32316192, 0.31617801, 0.3090812 ,
       0.30186581, 0.29452555, 0.28705363, 0.27944251, 0.27168392,
       0.26376868, 0.25568656, 0.24742611, 0.23897442, 0.23031687,
       0.22143673, 0.21231473, 0.20292844, 0.1932515 , 0.18325249,
       0.17289346, 0.16212772, 0.15089679, 0.13912555, 0.12671495,
       0.11353026, 0.09938329, 0.08401479, 0.06719413, 0.05074783,
       0.04680469, 0.05148665, 0.05743842, 0.06341134, 0.06917078,
       0.0746759 , 0.07993387, 0.08496362, 0.08978653, 0.09442177,
       0.09888792, 0.1031991 , 0.107371  , 0.11141233, 0.11533805,
       0.11915087])
'''

# largest_node = np.empty((3, len(T)))
# kernel_l_infinity = np.empty((3, len(T)))
# kernel_l_2 = np.empty((3, len(T)))
# smile_err = np.empty((3, len(T)))
T_avg = (np.amin(T) + np.amax(T))/2

s_e = np.empty(101)
k_e = np.empty(101)
l_n = np.empty(101)
smile_err = np.empty((3, 10, len(T)))

for N in np.array([7, 8, 9, 10]):
    for _ in range(1):
        '''
        if N == 1:
            T_init = 3/4 * np.amin(T) + 1/4 * np.amax(T)
        elif N == 2:
            T_init = 2/3 * np.amin(T) + 1/3 * np.amax(T)
        else:
            T_init = 1/(2*(N-2)) * np.amin(T) + (1-1/(2*(N-2))) * np.amax(T)
        '''
        if N == 1:
            T_init = 3/4 * np.amin(T) + 1/4 * np.amax(T)
        elif N == 2:
            T_init = 2/3 * np.amin(T) + 1/3 * np.amax(T)
        else:
            T_init = 1/(N-1) * np.amin(T) + (N-2)/(N-1) * np.amax(T)
        # L = np.exp(np.linspace(np.log(10), np.log(5e+08), 101))[i]
        tic = time.perf_counter()
        # _, nodes, weights = rk.optimize_error(H=H, N=N, T=T, bound=L, iterative=True, l2=True)
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=0.52, mode='observation')
        print((nodes, weights))
        smile = rHestonMarkov.iv_european_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0,
                                               T=T, rel_tol=rel_tol, nodes=nodes, weights=weights, N=-1)
        dur = time.perf_counter() - tic
        kernel_error = np.sqrt(rk.error(H, nodes, weights, T)) / rk.kernel_norm(H, T)
        smile_error = np.amax(np.abs(true_surface - smile) / true_surface)
        plt.plot(T, np.amax(np.abs(true_surface-smile)/true_surface, axis=1))
        plt.yscale('log')
        plt.xlabel('Maturity')
        plt.ylabel('Relative error')
        plt.title('Relative error in the implied volatility\nwhen optimizing ' + r'$ker_\infty$' + ' with 4 nodes, no bound')
        plt.show()
        largest_node = np.amax(nodes)
        kernel_l_infinity = np.amax(kernel_error)
        # s_e[i] = smile_error
        # k_e[i] = kernel_l_infinity
        # l_n[i] = largest_node
        # kernel_l_2 = np.sqrt(np.sum(kernel_error ** 2) / len(T))
        print(N, 'observation', largest_node, kernel_l_infinity, smile_error, dur)
        '''
        tic = time.perf_counter()
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T_avg, mode='observation')
        smile = rHestonMarkov.iv_european_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                               rel_tol=rel_tol, nodes=nodes, weights=weights, N=-1)
        dur = time.perf_counter() - tic
        kernel_error = np.sqrt(rk.error(H, nodes, weights, T))/rk.kernel_norm(H, T)
        smile_error = np.amax(np.abs(true_surface - smile) / true_surface)
        largest_node = np.amax(nodes)
        kernel_l_infinity = np.amax(kernel_error)
        kernel_l_2 = np.sqrt(np.sum(kernel_error**2)/len(T))
        print(N, 'observation', largest_node, kernel_l_infinity, kernel_l_2, smile_error, dur)
        
        tic = time.perf_counter()
        nodes, weights = rk.quadrature_rule(H=H, N=N, T=T_avg, mode='optimized')
        smile = rHestonMarkov.iv_european_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                               rel_tol=rel_tol, nodes=nodes, weights=weights, N=-1)
        dur = time.perf_counter() - tic
        kernel_error = np.sqrt(rk.error(H, nodes, weights, T))/rk.kernel_norm(H, T)
        smile_error = np.amax(np.abs(true_surface - smile) / true_surface)
        largest_node = np.amax(nodes)
        kernel_l_infinity = np.amax(kernel_error)
        kernel_l_2 = np.sqrt(np.sum(kernel_error**2)/len(T))
        smile_err[0, N-1, :] = np.amax(np.abs(true_surface-smile)/true_surface, axis=1)
        plt.plot(T, smile_err[0, N-1, :], '-', color=c[N-1], label=f'{N} dimensions')
        print(N, 'optimized fixed T', largest_node, kernel_l_infinity, kernel_l_2, smile_error, dur)
        print((smile_err[0, N-1, :],))
        tic = time.perf_counter()
        _, nodes, weights = rk.optimize_error(H=H, N=N, T=T, l2=False)
        smile = rHestonMarkov.iv_european_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                               rel_tol=rel_tol, nodes=nodes, weights=weights, N=-1)
        dur = time.perf_counter() - tic
        kernel_error = np.sqrt(rk.error(H, nodes, weights, T))/rk.kernel_norm(H, T)
        smile_error = np.amax(np.abs(true_surface - smile) / true_surface)
        largest_node = np.amax(nodes)
        kernel_l_infinity = np.amax(kernel_error)
        kernel_l_2 = np.sqrt(np.sum(kernel_error**2)/len(T))
        smile_err[1, N-1, :] = np.amax(np.abs(true_surface-smile)/true_surface, axis=1)
        plt.plot(T, smile_err[1, N-1, :], '--', color=c[N - 1])
        print(N, 'optimized l infinity', largest_node, kernel_l_infinity, kernel_l_2, smile_error, dur)
        print((smile_err[1, N-1, :],))
        tic = time.perf_counter()
        _, nodes, weights = rk.optimize_error(H, N, T, l2=True)
        smile = rHestonMarkov.iv_european_call(S=S, K=K, H=H, lambda_=lambda_, rho=rho, nu=nu, theta=theta, V_0=V_0, T=T,
                                               rel_tol=rel_tol, nodes=nodes, weights=weights, N=-1)[0, :]
        dur = time.perf_counter() - tic
        kernel_error = np.sqrt(rk.error(H, nodes, weights, T))/rk.kernel_norm(H, T)
        smile_error = np.amax(np.abs(true_surface - smile) / true_surface)
        largest_node = np.amax(nodes)
        kernel_l_infinity = np.amax(kernel_error)
        kernel_l_2 = np.sqrt(np.sum(kernel_error**2)/len(T))
        smile_err[2, N-1, :] = np.amax(np.abs(true_surface-smile)/true_surface, axis=1)
        plt.plot(T, smile_err[2, N-1, :], 'o-', color=c[N - 1])
        print(N, 'optimized l 2', largest_node, kernel_l_infinity, kernel_l_2, smile_error, dur)
        print((smile_err[2, N-1, :],))
        '''
print((smile_err,))

plt.xlabel('Maturity')
plt.ylabel('Relative error')
plt.yscale('log')
plt.legend(loc='best')
plt.title('Relative error in implied volatility depending on maturity')
plt.show()


print((largest_node,))
print((kernel_l_infinity,))
print((kernel_l_2,))
print((smile_err,))
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
            res_loc = rHestonMarkov.iv_european_call(K=np.exp(k_loc), H=0.1, lambda_=0.3,
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
        res_loc = rHestonMarkov.iv_european_call(K=np.exp(k_loc), H=0.1, lambda_=0.3,
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
        res_loc = rHestonMarkov.iv_european_call(K=np.exp(k_loc), H=0.1, lambda_=0.3,
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
