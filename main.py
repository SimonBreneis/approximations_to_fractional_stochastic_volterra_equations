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


truth = Data.rrHeston_1
recomputed_truth = np.array([0.36679211, 0.37151053, 0.37349597, 0.37300667, 0.37032276, 0.36581155,
 0.36003932, 0.35392561, 0.34881425, 0.34607352, 0.34613224, 0.34811188,
 0.35066277, 0.35275747, 0.35383477, 0.35367447, 0.35228275, 0.34983653,
 0.34666507, 0.34323466, 0.34009206, 0.33772914, 0.33639885, 0.33601041,
 0.33620251, 0.33652615, 0.33659838, 0.33617141, 0.33514212, 0.33353737,
 0.33149203, 0.32922015, 0.32697315, 0.32498276, 0.32339958, 0.32225261,
 0.32145111, 0.32082693, 0.32019298, 0.31939254, 0.3183283 , 0.3169722,
 0.31536127, 0.31358309, 0.31175343, 0.30998849, 0.30837712, 0.30696039,
 0.3057252 , 0.30461362, 0.30354326, 0.30243078, 0.30121143, 0.29985104,
 0.29834961, 0.29673734, 0.29506486, 0.29338986, 0.29176301, 0.29021636,
 0.28875734, 0.2873693 , 0.28601827, 0.28466305, 0.28326577, 0.28180032,
 0.28025685, 0.2786423 , 0.27697698, 0.27528857, 0.27360491, 0.27194739,
 0.27032638, 0.26873978, 0.26717468, 0.2656115 , 0.26402922, 0.26241048,
 0.26074513, 0.25903182, 0.25727743, 0.25549458, 0.25369811, 0.25190129,
 0.25011271, 0.24833462, 0.24656279, 0.2447881 , 0.24299894, 0.24118418,
 0.23933557, 0.23744935, 0.23552668, 0.23357276, 0.23159523, 0.22960199,
 0.2275992 , 0.22558976, 0.22357276, 0.22154379, 0.2194961 , 0.21742222,
 0.21531559, 0.2131719 , 0.2109898 , 0.20877079, 0.20651855, 0.20423768,
 0.20193233, 0.19960503, 0.19725593, 0.19488264, 0.1924807 , 0.19004452,
 0.18756857, 0.18504852, 0.18248203, 0.17986913, 0.17721198, 0.17451422,
 0.17178001, 0.16901299, 0.16621548, 0.16338817, 0.16053031, 0.15764051,
 0.15471793, 0.15176381, 0.14878291, 0.14578461, 0.14278366, 0.13980009,
 0.13685872, 0.13398793, 0.13121828, 0.12858084, 0.12610577, 0.12382093,
 0.12175093, 0.11991626, 0.11833263, 0.11701027, 0.11595315, 0.1151582,
 0.11461458, 0.1143035 , 0.11419869, 0.11426809, 0.11447666, 0.11479004,
 0.11517853, 0.11562054 ,0.11610477, 0.11663058, 0.11720621, 0.11784479,
 0.11855868, 0.11935326 ,0.12022168, 0.12114257, 0.12208199])
untruth = np.array([4.18877678e-01, 4.10908805e-01, 3.95918637e-01, 3.69571621e-01,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 3.33233398e-01, 3.68274123e-01, 3.82462863e-01, 3.88932231e-01,
 3.89972470e-01, 3.86400807e-01, 3.78593612e-01, 3.66766155e-01,
 3.51215599e-01, 3.33343645e-01, 3.20194735e-01, 3.23762569e-01,
 3.36580576e-01, 3.48452289e-01, 3.57060346e-01, 3.62214677e-01,
 3.64073994e-01, 3.62879258e-01, 3.58961358e-01, 3.52822238e-01,
 3.45278897e-01, 3.37655776e-01, 3.31802389e-01, 3.29370881e-01,
 3.30509697e-01, 3.33772062e-01, 3.37410007e-01, 3.40212555e-01,
 3.41545324e-01, 3.41175371e-01, 3.39148558e-01, 3.35738841e-01,
 3.31435426e-01, 3.26922023e-01, 3.22983931e-01, 3.20290407e-01,
 3.19118632e-01, 3.19234909e-01, 3.20053553e-01, 3.20918798e-01,
 3.21304959e-01, 3.20890582e-01, 3.19560744e-01, 3.17385701e-01,
 3.14592313e-01, 3.11523530e-01, 3.08573821e-01, 3.06096405e-01,
 3.04303237e-01, 3.03203600e-01, 3.02617723e-01, 3.02255231e-01,
 3.01811934e-01, 3.01044817e-01, 2.99812725e-01, 2.98088440e-01,
 2.95950569e-01, 2.93559817e-01, 2.91121348e-01, 2.88836608e-01,
 2.86853779e-01, 2.85231641e-01, 2.83930194e-01, 2.82830780e-01,
 2.81775665e-01, 2.80611392e-01, 2.79223484e-01, 2.77556833e-01,
 2.75621163e-01, 2.73483139e-01, 2.71247538e-01, 2.69030991e-01,
 2.66933630e-01, 2.65015499e-01, 2.63284206e-01, 2.61697034e-01,
 2.60175868e-01, 2.58629385e-01, 2.56975784e-01, 2.55160609e-01,
 2.53166547e-01, 2.51014235e-01, 2.48754717e-01, 2.46455558e-01,
 2.44183796e-01, 2.41989733e-01, 2.39895519e-01, 2.37891283e-01,
 2.35939419e-01, 2.33985292e-01, 2.31971162e-01, 2.29849674e-01,
 2.27593901e-01, 2.25202027e-01, 2.22696027e-01, 2.20114932e-01,
 2.17504368e-01, 2.14904854e-01, 2.12341666e-01, 2.09818669e-01,
 2.07317495e-01, 2.04802054e-01, 2.02227014e-01, 1.99548081e-01,
 1.96731599e-01, 1.93761349e-01, 1.90641082e-01, 1.87392272e-01,
 1.84047599e-01, 1.80641616e-01, 1.77200737e-01, 1.73734912e-01,
 1.70232994e-01, 1.66663022e-01, 1.62977558e-01, 1.59123162e-01,
 1.55052272e-01, 1.50735260e-01, 1.46170384e-01, 1.41389649e-01,
 1.36459316e-01, 1.31474882e-01, 1.26551514e-01, 1.21812003e-01,
 1.17374774e-01, 1.13344287e-01, 1.09805144e-01, 1.06819902e-01,
 1.04429341e-01, 1.02653257e-01, 1.01490012e-01, 1.00914081e-01,
 1.00872500e-01, 1.01282883e-01, 1.02036310e-01, 1.03007108e-01,
 1.04068559e-01, 1.05110847e-01, 1.06056778e-01, 1.06872098e-01,
 1.07569167e-01, 1.08203892e-01, 1.08865547e-01, 1.09658260e-01,
 1.10673678e-01, 1.11958914e-01, 1.13491305e-01, 1.15173999e-01,
 1.16857138e-01, 1.18374790e-01, 1.19581958e-01, 1.20382597e-01,
 1.20749756e-01])

rerecomputed_truth = np.array([0.36679211, 0.37151053, 0.37349597, 0.37300667, 0.37032276, 0.36581155,
 0.36003932, 0.35392561, 0.34881425, 0.34607352, 0.34613224, 0.34811188,
 0.35066277, 0.35275747, 0.35383477, 0.35367447, 0.35228275, 0.34983653,
 0.34666507, 0.34323466, 0.34009206, 0.33772914, 0.33639885, 0.33601041,
 0.33620251, 0.33652615, 0.33659838, 0.33617141, 0.33514212, 0.33353737,
 0.33149203, 0.32922015, 0.32697315, 0.32498276, 0.32339958, 0.32225261,
 0.32145111, 0.32082693, 0.32019298, 0.31939254, 0.3183283 , 0.3169722,
 0.31536127 ,0.31358309, 0.31175343, 0.30998849, 0.30837712, 0.30696039,
 0.3057252  ,0.30461362, 0.30354326, 0.30243078, 0.30121143, 0.29985104,
 0.29834961 ,0.29673734, 0.29506486, 0.29338986, 0.29176301, 0.29021636,
 0.28875734 ,0.2873693 , 0.28601827, 0.28466305, 0.28326577, 0.28180032,
 0.28025685 ,0.2786423 , 0.27697698, 0.27528857, 0.27360491, 0.27194739,
 0.27032638, 0.26873978, 0.26717468, 0.2656115 , 0.26402922, 0.26241048,
 0.26074513, 0.25903182, 0.25727743, 0.25549458, 0.25369811, 0.25190129,
 0.25011271, 0.24833462, 0.24656279, 0.2447881 , 0.24299894, 0.24118418,
 0.23933557, 0.23744935, 0.23552668, 0.23357276, 0.23159523, 0.22960199,
 0.2275992 , 0.22558976, 0.22357276, 0.22154379, 0.2194961 , 0.21742222,
 0.21531559, 0.2131719 , 0.2109898 , 0.20877079, 0.20651855, 0.20423768,
 0.20193233, 0.19960503, 0.19725593, 0.19488264, 0.1924807 , 0.19004452,
 0.18756857, 0.18504852, 0.18248203, 0.17986913, 0.17721198, 0.17451422,
 0.17178001, 0.16901299, 0.16621548, 0.16338817, 0.16053031, 0.15764051,
 0.15471793, 0.15176381, 0.14878291, 0.14578461, 0.14278366, 0.13980009,
 0.13685872, 0.13398793, 0.13121828, 0.12858084 ,0.12610577, 0.12382093,
 0.12175093, 0.11991626, 0.11833263, 0.11701027, 0.11595315, 0.1151582,
 0.11461458, 0.1143035 , 0.11419869, 0.11426809, 0.11447666, 0.11479004,
 0.11517853, 0.11562054, 0.11610477, 0.11663058, 0.11720621, 0.11784479,
 0.11855868, 0.11935326, 0.12022168, 0.12114257, 0.12208199])

totally_true = Data.rrHeston
optimized_points = np.array([0.40286746, 0.40315553, 0.40308528, 0.40241364, 0.40105379, 0.39906079,
 0.39660925, 0.39396115, 0.39141543, 0.38923719, 0.3875829 , 0.38645551,
 0.3857155 , 0.38514044, 0.38449857, 0.38360628, 0.38235887, 0.38073907,
 0.37880991, 0.37669597, 0.37455437, 0.37253804, 0.37075797, 0.3692561,
 0.36799891, 0.36689387, 0.36582057, 0.36466407, 0.36334071, 0.36181249,
 0.36008982, 0.35822465, 0.3562958 , 0.35438929, 0.35257758, 0.35090261,
 0.3493676 , 0.34793959, 0.34656135, 0.34516803, 0.34370336, 0.34213174,
 0.34044398, 0.33865677, 0.33680658, 0.33493966, 0.33310043, 0.33132103,
 0.32961446, 0.32797299, 0.32637195, 0.32477729, 0.32315465, 0.32147744,
 0.31973222, 0.31792053, 0.31605715, 0.3141655 , 0.3122714 , 0.31039673,
 0.30855452, 0.30674647, 0.30496351, 0.30318891, 0.30140294, 0.29958774,
 0.2977314 , 0.29583009, 0.29388826, 0.29191685, 0.2899301 , 0.28794191,
 0.28596245, 0.28399607, 0.28204059, 0.28008839, 0.27812858, 0.27614985,
 0.27414313, 0.27210347, 0.27003084, 0.26792966, 0.26580735, 0.2636723,
 0.26153167, 0.25938975, 0.25724707, 0.25510048, 0.25294413, 0.25077098,
 0.2485745 , 0.2463501 , 0.24409604, 0.2418135 , 0.23950607, 0.23717857,
 0.23483579, 0.23248119, 0.2301161 , 0.22773945, 0.22534805, 0.22293743,
 0.22050288, 0.21804053, 0.21554807, 0.21302519, 0.21047341, 0.20789551,
 0.2052947 , 0.20267375, 0.20003421, 0.19737604, 0.19469759, 0.19199604,
 0.18926804, 0.18651058, 0.18372166, 0.18090081, 0.17804921, 0.17516946,
 0.17226507, 0.16933982, 0.16639709, 0.16343944, 0.16046851, 0.15748529,
 0.15449079, 0.15148688, 0.14847735, 0.14546874, 0.14247103, 0.13949786,
 0.13656639, 0.13369676, 0.13091121, 0.1282331 , 0.12568588, 0.12329213,
 0.12107287, 0.11904694, 0.11723061, 0.11563707, 0.11427605, 0.11315308,
 0.11226869, 0.11161754, 0.11118766, 0.11096035, 0.11091089, 0.11101049,
 0.11122925, 0.11153981, 0.11192098, 0.11236044, 0.11285587, 0.11341391,
 0.11404667, 0.11476618, 0.11557767, 0.11647376, 0.11743149])

plt.plot(Data.k_rrHeston, totally_true)
plt.plot(Data.k_rrHeston, Data.rrHeston_1)
plt.plot(Data.k_rrHeston, optimized_points)
plt.show()

print(np.sum(np.abs(recomputed_truth - truth)))
print(np.sum(np.abs(untruth - truth)))
print(np.sum(np.abs(rerecomputed_truth - truth)))
# print(rHestonMarkov.implied_volatility(K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N=6, adaptive=True, mode='optimized'))
print(rHestonMarkov.implied_volatility(mode='optimized', N=2, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50, R=2))
# print(rHestonMarkov.implied_volatility(N=1, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50, R=2))
time.sleep(360000)

print(rk.quadrature_rule_geometric_good(0.1, 1, 1., 'observation'))
print(rk.quadrature_rule_geometric_standard(0.1, 1, 1., 'observation'))
Data.rHeston_smiles_precise()
print(rHestonMarkov.implied_volatility(N=1, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., N_Riccati=3000, N_fourier=10000, L=50, R=2))
time.sleep(360000)
print(rHeston.implied_volatility(K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02))
time.sleep(360000)
print(rHestonMarkov.implied_volatility(N=1, K=np.exp(Data.k_rrHeston), H=0.1, lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1.,
                                       adaptive=False, N_Riccati=1000, R=2., L=200., N_fourier=40000, mode="observation"))
time.sleep(36000)

rk.compare_approximations(0.1, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]), T=1.)
time.sleep(3600000)

print(rHestonMarkov.implied_volatility(Data.k_rrHeston, 0.1, 0.3, -0.7, 0.3, 0.02, 0.02, 1., 1, R=-2, N_Riccati=1000, L=10., N_fourier=1000))
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
k_vec = Data.k_rrHeston
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
    plt.plot(Data.k_rrHeston, Data.rrHeston_6, 'k-', label=f'Fourier inversion')
    plt.title(vb)
    plt.legend(loc='best')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.show()

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
