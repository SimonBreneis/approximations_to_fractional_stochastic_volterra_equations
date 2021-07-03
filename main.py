import time
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import stats
import mpmath as mp
import Data
import QuadratureRulesRoughKernel as qr
import ComputationalFinance as cf
import rBergomiBFG
import rBergomiAK
import rHeston
import rHestonAK
import fBmAK
import RoughKernel as rk


import orthopy
import quadpy


mp.mp.dps = 1000
#print(error_estimate_improved_exponential(0.1, 7, 73, 7.7444, 131.24, 1.))
#time.sleep(3600)


ms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ns = [[1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024],
      [1, 2, 3, 4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512],
      [1, 2, 3, 4, 5, 8, 11, 15, 21, 30, 43, 60, 85, 121, 171, 241, 341],
      [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256],
      [1, 2, 3, 5, 6, 9, 13, 18, 26, 36, 51, 72, 102, 145, 205],
      [1, 2, 3, 4, 5, 7, 11, 15, 21, 30, 43, 60, 85, 121, 171],
      [1, 2, 3, 5, 6, 9, 13, 18, 26, 37, 52, 73, 103, 146],
      [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128],
      [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 57, 80, 114],
      [1, 2, 3, 4, 6, 9, 13, 18, 26, 36, 51, 72, 102]]
#plot_errors_sparse(0.1, 1., ms, ns)


Ns = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048])
print(0.1306*np.sqrt(Ns))


alpha = 1.064184
beta = 0.4275
H = 0.1
A = np.sqrt(1/H + 1/(1.5-H))
rates = -2*alpha/A
exponent = np.exp(alpha*beta)/(8*(np.exp(alpha*beta)-1))
expp = 1/(3*exponent + 6*H - 4*H*H)
c_H = 1/(math.gamma(0.5-H)*math.gamma(0.5+H))
prefac = 1/(2*H) + 1/exponent + 1/(3-2*H)
fac1 = (3/H)**(exponent * (3-2*H))
fac2 = (5*np.pi**3/384 * np.exp(alpha*beta) * (np.exp(alpha*beta)-1) * (A/beta)**(2-2*H) / H)**(6*H - 4*H*H)
fac3 = (1/(1.5-H))**(exponent*2*H)
print(c_H**2 * prefac * (fac1*fac2*fac3)**expp)
print((1-H)*(6*H-4*H*H)*expp)
print(rates)

'''
betas = np.arange(1, 100001)/100000.
res = 2*betas * (-np.log(1/2 * (np.exp(alpha*betas) - 1))) - alpha
print(np.amax(res))
print((np.argmax(res) + 1)/100000.)
print((np.amax(res)+alpha))
plt.plot(betas, res)
plt.plot(betas, np.zeros(100000))
plt.show()
'''

def func(x):
    return np.exp(-x)


mp.mp.dps = 100
m = 10
n = 100
rule = qr.quadrature_rule_mpmath(mp.mpf(0.1), m, mp.matrix([mp.mpf(1+2.*x/n) for x in range(n+1)]))
rule = np.array([[float(x) for x in rule[i, :]] for i in range(2)])
true_integral = np.dot(rule[1, :], func(rule[0, :]))
ms = np.array([int(m) for m in np.sqrt(2)**np.arange(1, 11)], dtype=int)
for m in ms:
    m = int(m)
    rule = qr.quadrature_rule_interval_general(mp.mpf(0.1), m, mp.mpf(1.), mp.mpf(3.))
    rule = np.array([[float(x) for x in rule[i, :]] for i in range(2)])
    approx_integral = np.dot(rule[1, :], func(rule[0, :]))
    print(f"True integral: {true_integral}")
    print(f"Approximated integral: {approx_integral}")
    print(f"Error: {np.fabs(true_integral-approx_integral)}")
    print(f"Bound: {0.5 /math.factorial(2*m)}")


N = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048])
logximnn = np.array([4.8778, 9.1800, 12.063, 14.455, 18.342, 21.394, 23.917, 28.971, 33.278, 36.893, 43.621, 51.739, 58.953, 70.067, 79.889, 95.048, 110.73])

c_arr = scipy.stats.linregress(np.log(N[1:len(logximnn)]), np.log(logximnn[1:]))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(N[1:len(logximnn)], logximnn[1:], 'b-', label="error")
plt.loglog(N[1:len(logximnn)], np.exp(c02) * N[1:len(logximnn)] ** c12, 'or--', label="regression")
plt.loglog(N[1:len(logximnn)], np.exp(c02) * N[1:len(logximnn)] ** (0.5), 'sg--', label="order 0.5")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

print(np.log(2.1064*N**(-0.3067) * np.exp(1.987*np.sqrt(N))))

print("Stop the program!")
time.sleep(3600)

a = 1.
b = 10.


def funky(x):
    return np.exp(x/5.)*np.sin(x)


m=2
n=1000
partition = a + (b-a)*np.arange(n+1)/n
quad_rule = qr.quadrature_rule_mpmath(0.1, m, partition)
nodes = quad_rule[0, :]
nodes = np.array([float(node) for node in nodes])
weights = quad_rule[1, :]
weights = np.array([float(weight) for weight in weights])
integral = weights.dot(funky(nodes))
print(n)
print(integral)
error_old = 1.

n_vec = np.array([1, 2, 4, 8, 16, 32, 64, 128])
for n in n_vec:
    partition = a + (b-a)*np.arange(n+1)/n
    quad_rule = qr.quadrature_rule_mpmath(0.1, m, partition)
    nodes = quad_rule[0, :]
    nodes = np.array([float(node) for node in nodes])
    weights = quad_rule[1, :]
    weights = np.array([float(weight) for weight in weights])
    approx = weights.dot(funky(nodes))
    print(n)
    print(approx)
    error_new = np.fabs(integral-approx)
    print(error_new)
    print(error_old/error_new)
    error_old = error_new

'''
N = 2
moments = np.array([mp.mpf(rk.c_H(0.1) / (k + 0.5 - 0.1) * (2**(k+0.5-0.1) - 1**(k+0.5-0.1))) for k in range(2*N)])
print(moments)

alpha, beta, int_1 = orthopy.tools.chebyshev(moments)
points, weights = quadpy.tools.scheme_from_rc(alpha, beta, int_1, mode="mpmath")
print(points)
print(weights)
print(qr.quadrature_rule_geometric_mpmath(0.1, 2, 1, 1., 2.))
'''
print(qr.quadrature_rule_interval(0.1, 2, 1., 2.))



'''
# Computing strong approximation errors for different values of n, m
H = mp.mpf(0.1)
T = mp.mpf(1)
number_paths = 100000

for m in (1,):
    for n in (8,):
        result = strong_error_fBm_approximation_MC(H, T, m, n, number_paths)
        print('m={0}, n={1}, error={2}, std={3}'.format(m, n, result[0], result[1]))
'''

'''
# Plot loglog-plots of the L^2 errors of the OU-approximations of fBm for different values of m and n.
m1_errors = np.array(
    [0.8580795741857218, 0.5933883792763698, 0.3506089358024185, 0.17621325270440755, 0.07995320204075512,
     0.03430761416752963, 0.014829003039174117, 0.006666020657043866])
m2_errors = np.array(
    [1.0274299516166492, 0.4857599152180374, 0.29783742948598957, 0.12454282877701886, 0.03481046890585258,
     0.005911347425846188, 0.0008669864303425754])

m1_vec = np.array([2., 4., 8., 16., 32., 64., 128., 256.])
c_arr = scipy.stats.linregress(np.log(m1_vec), np.log(m1_errors))
c11 = c_arr[0]
c01 = c_arr[1]
print(c11, c01)
plt.loglog(m1_vec, m1_errors, 'b-', label="error")
plt.loglog(m1_vec, np.exp(c01) * m1_vec ** c11, 'or--', label="regression")
plt.loglog(m1_vec, np.exp(c01) * m1_vec ** (-1), 'sg--', label="order 1")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

m2_vec = np.array([4., 8., 16., 32., 64., 128., 256.])
c_arr = scipy.stats.linregress(np.log(m2_vec), np.log(m2_errors))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(m2_vec, m2_errors, 'b-', label="error")
plt.loglog(m2_vec, np.exp(c02) * m2_vec ** c12, 'or--', label="regression")
plt.loglog(m2_vec, np.exp(c02) * m2_vec ** (-2), 'sg--', label="order 2")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

plt.loglog(m2_vec, m1_errors[1:], 'b-', label='m=1')
plt.loglog(m2_vec, m2_errors, 'r-', label='m=2')
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

m2_vec = np.array([16., 32., 64., 128., 256.])
c_arr = scipy.stats.linregress(np.log(m2_vec), np.log(m2_errors[2:]))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(m2_vec, m2_errors[2:], 'b-', label="error")
plt.loglog(m2_vec, np.exp(c02) * m2_vec ** c12, 'or--', label="regression")
plt.loglog(m2_vec, np.exp(c02) * m2_vec ** (-2), 'sg--', label="order 2")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()
'''

sigma_imp = cf.implied_volatility_call(S=100., K=120., r=0.05, T=0.5, price=2.)
print('Test implied volatility: {0}'.format(sigma_imp))
print('Test call price: {0}'.format(cf.BS_call_price(100., 120., sigma_imp, 0.05, 0.5)))
H = 0.07
T = 0.9
N = 100
eta = 1.9
V_0 = 0.235 ** 2
S_0 = 1.
rho = -0.9
r = 0.
M = 10000
K = 1.
m = 1
n = 20
a = 1.
b = 1.
S_0 = 1.
rounds = 1

rHestonAK.plot_rHeston_AK()

k_vec = np.array([i / 100. for i in range(-40, 21)])
K_vec = S_0 * np.exp(k_vec)
n_vec = np.array([2, 4, 8])
'''
(mi, lo, up) = implied_volatility_call_rBergomi_BFG(H, T, N, eta, V_0, S_0, rho, K_vec, M, rounds)
print("Vector of k:")
print(k_vec)
print("Implied volatility of call options of the rBergomi model, mean estimates:")
print(mi)
#print("Implied volatility of call options of the rBergomi model, lower estimates:")
#print(lo)
#print("Implied volatility of call options of the rBergomi model, upper estimates:")
#print(up)

for n in n_vec:
    print(f"n={n}")
    (mi2, lo2, up2) = implied_volatility_call_rBergomi_AK(H, T, N, eta, V_0, S_0, rho, K_vec, m, n, a, b, M, rounds)
    print("Implied volatility of call options of the approximate rBergomi model, mean estimates:")
    print(mi2)
    #print("Implied volatility of call options of the approximate rBergomi model, lower estimates:")
    #print(lo2)
    #print("Implied volatility of call options of the approximate rBergomi model, upper estimates:")
    #print(up2)
    #print("Mean error estimate:")
    mean_error = np.fabs(mi2-mi)
    #print(mean_error)
    #print("Upper error estimate:")
    upper_error = np.fmax(np.fabs(up2-lo), np.fabs(up-lo2))
    #print(upper_error)
    print("Mean error integral:")
    print(np.average(mean_error))
    print("Upper error integral:")
    print(np.average(upper_error))
    print("Mean error for k=0.1:")
    print(mean_error[50])
    print("Upper error for k=0.1:")
    print(upper_error[50])

'''
k_vec = np.array([i / 100. for i in range(-40, 21)])
rBergomi = np.array([0.30878589, 0.30629082, 0.30378026, 0.30125732, 0.29872267, 0.29617548,
                     0.29361944, 0.29105047, 0.28846861, 0.28587805, 0.28327942, 0.28066776,
                     0.27804194, 0.27540347, 0.27275188, 0.27008338, 0.26740030, 0.26470467,
                     0.26199638, 0.25927082, 0.25653326, 0.25378723, 0.25102863, 0.24825490,
                     0.24546746, 0.24266456, 0.23984552, 0.23701176, 0.23416258, 0.23130497,
                     0.22843492, 0.22555297, 0.22265999, 0.21975504, 0.21684114, 0.21391494,
                     0.21097830, 0.20802965, 0.20506966, 0.20210868, 0.19914467, 0.19617844,
                     0.19321570, 0.19026870, 0.18734111, 0.18442839, 0.18153762, 0.17867544,
                     0.17585233, 0.17308232, 0.17037065, 0.16772805, 0.16516713, 0.16270101,
                     0.16034282, 0.15810844, 0.15601021, 0.15405349, 0.15223960, 0.15057060,
                     0.14905202])

rBergomi_2 = np.array([0.26954819, 0.26857467, 0.26759884, 0.26661924, 0.26563986, 0.26466083,
                       0.26367770, 0.26270334, 0.26173995, 0.26077652, 0.25980985, 0.25884828,
                       0.25788317, 0.25691491, 0.25594331, 0.25496806, 0.25398495, 0.25299145,
                       0.25199333, 0.25099640, 0.24999695, 0.24899307, 0.24798630, 0.24697772,
                       0.24596624, 0.24495475, 0.24394403, 0.24293329, 0.24191834, 0.24089753,
                       0.23987157, 0.23884129, 0.23780874, 0.23677682, 0.23574285, 0.23470682,
                       0.23366923, 0.23262696, 0.23158358, 0.23053741, 0.22949098, 0.22844434,
                       0.22739505, 0.22634232, 0.22528648, 0.22423024, 0.22317175, 0.22211158,
                       0.22105115, 0.21998431, 0.21891229, 0.21783576, 0.21675316, 0.21567167,
                       0.21458424, 0.21349738, 0.21241055, 0.21132085, 0.21023120, 0.20914157,
                       0.20805303])

rBergomi_4 = np.array([0.29148540, 0.28979548, 0.28810566, 0.28641570, 0.28471764, 0.28301367,
                       0.28130973, 0.27960403, 0.27789295, 0.27617730, 0.27445798, 0.27273219,
                       0.27099986, 0.26926220, 0.26751718, 0.26576665, 0.26401197, 0.26225347,
                       0.26048880, 0.25872220, 0.25695406, 0.25518142, 0.25340231, 0.25162107,
                       0.24983779, 0.24805124, 0.24626010, 0.24446533, 0.24266368, 0.24085262,
                       0.23903010, 0.23719933, 0.23536305, 0.23351918, 0.23166424, 0.22980830,
                       0.22794920, 0.22608380, 0.22421000, 0.22233015, 0.22043906, 0.21853473,
                       0.21662281, 0.21470572, 0.21278553, 0.21086296, 0.20893536, 0.20700288,
                       0.20506296, 0.20312356, 0.20118659, 0.19924979, 0.19730981, 0.19536559,
                       0.19341135, 0.19145382, 0.18948829, 0.18752153, 0.18556310, 0.18361192,
                       0.18166196])

rBergomi_8 = np.array([0.29263517, 0.29096081, 0.28927793, 0.28758692, 0.28588227, 0.28417107,
                       0.28244773, 0.28071616, 0.27897379, 0.27722156, 0.27545674, 0.27368181,
                       0.27189882, 0.27010457, 0.26830037, 0.26648627, 0.2646625, 0.26283184,
                       0.26099409, 0.25914671, 0.25728991, 0.25542474, 0.25355117, 0.25167086,
                       0.24978058, 0.24787653, 0.24596373, 0.24404046, 0.24210845, 0.24016533,
                       0.23821297, 0.23625370, 0.23428790, 0.23231638, 0.23033046, 0.22833328,
                       0.22633005, 0.22432302, 0.22230742, 0.22028815, 0.21826258, 0.21623074,
                       0.21419376, 0.21215383, 0.21011321, 0.20807370, 0.20602790, 0.20397752,
                       0.20192458, 0.19986895, 0.19780403, 0.19573423, 0.19366975, 0.19161056,
                       0.18955597, 0.18750978, 0.18547212, 0.18343419, 0.18140028, 0.17937711,
                       0.17736915])

rBergomi_16 = np.array([0.30225526, 0.30013381, 0.29799969, 0.29585726, 0.29370582, 0.29154356,
                        0.28937614, 0.28719628, 0.28500224, 0.28279380, 0.28057324, 0.27834435,
                        0.27610282, 0.27384913, 0.27158234, 0.26930375, 0.26701037, 0.26470713,
                        0.26239234, 0.26006992, 0.25773827, 0.25539224, 0.25303568, 0.25067045,
                        0.24829468, 0.24590539, 0.24350559, 0.24109616, 0.23867238, 0.23624039,
                        0.23379810, 0.23134269, 0.22887083, 0.22638790, 0.22389468, 0.22139110,
                        0.21887740, 0.21634786, 0.21381014, 0.21126753, 0.20871685, 0.20615744,
                        0.20359198, 0.20101957, 0.19844216, 0.19586753, 0.19329621, 0.19072116,
                        0.18815355, 0.18559310, 0.18304491, 0.18050839, 0.17799163, 0.17550204,
                        0.17304590, 0.17063451, 0.16827450, 0.16596925, 0.16371494, 0.16153584,
                        0.15944656])

rBergomi_32 = np.array([0.30448798, 0.30221138, 0.29992409, 0.2976232, 0.29530925, 0.29298094,
                        0.29064271, 0.28829284, 0.28592646, 0.28354444, 0.28115215, 0.27874806,
                        0.27633292, 0.27390937, 0.27147717, 0.26903089, 0.26656888, 0.26409849,
                        0.26161619, 0.25912092, 0.25661453, 0.25409455, 0.25156119, 0.24901402,
                        0.24645042, 0.24387062, 0.24127644, 0.23866657, 0.23604051, 0.23340071,
                        0.23074539, 0.22807996, 0.22540203, 0.22270992, 0.22000639, 0.21729123,
                        0.2145643, 0.21182864, 0.20908149, 0.20632356, 0.20355565, 0.20078378,
                        0.19801083, 0.19524012, 0.19247322, 0.18971951, 0.18698369, 0.18426788,
                        0.18157419, 0.1788987, 0.17624828, 0.17363767, 0.17106947, 0.16855319,
                        0.16610667, 0.16374242, 0.16147046, 0.15929046, 0.15720677, 0.15523531,
                        0.15338483])

rBergomi_64 = np.array([0.31009601, 0.30756878, 0.30503451, 0.30249607, 0.29995251, 0.29740012,
                        0.29483877, 0.29227058, 0.28969126, 0.2870982, 0.28449643, 0.28188293,
                        0.27926019, 0.27662655, 0.27398364, 0.27132884, 0.26866268, 0.26598539,
                        0.2632954, 0.26059775, 0.25789122, 0.255172, 0.25244314, 0.24970178,
                        0.24694694, 0.24418135, 0.2414026, 0.23860819, 0.23580387, 0.23298832,
                        0.23015886, 0.2273164, 0.22446544, 0.22160756, 0.21873789, 0.21585969,
                        0.21297178, 0.21007242, 0.20716663, 0.20425301, 0.20133059, 0.19840866,
                        0.19548851, 0.19257604, 0.18967043, 0.18678067, 0.18391097, 0.18106679,
                        0.17825139, 0.17547895, 0.17275316, 0.17008232, 0.16747954, 0.16495641,
                        0.16253136, 0.16020863, 0.15800481, 0.15592192, 0.15397283, 0.15217099,
                        0.15050673])

mean_error_avg = np.array([0.028134995629132267, 0.016455251456657768, 0.014743213390175258,
                           0.0066970065547930285, 0.0032726454643155745, 0.0017174343233870826])
upper_error_avg = np.array([0.030556598335120724, 0.018665475990593074, 0.016931657352149413,
                            0.008786826121979546, 0.005330713344092961, 0.0037475427520204875])
mean_error_01 = np.array([0.048541644864112886, 0.030815937716089370, 0.027433379145530457,
                          0.012674257595663874, 0.005877634321164282, 0.0023825084644830685])
upper_error_01 = np.array([0.049385873134685510, 0.031573935557073995, 0.028180269145823555,
                           0.013375876878720216, 0.006565842268107969, 0.0030643420177206004])

plt.plot(k_vec, rBergomi, "k", label="n=infinity")
plt.plot(k_vec, rBergomi_2, label="n=2")
plt.plot(k_vec, rBergomi_4, label="n=4")
plt.plot(k_vec, rBergomi_8, label="n=8")
plt.plot(k_vec, rBergomi_16, label="n=16")
plt.plot(k_vec, rBergomi_32, label="n=32")
plt.plot(k_vec, rBergomi_64, label="n=64")
plt.legend(loc='upper right')
plt.xlabel('Log-strike k')
plt.ylabel('Implied volatility')
plt.show()

n_array = np.array([2., 4., 8., 16., 32., 64.])
c_arr = scipy.stats.linregress(np.log(n_array), np.log(mean_error_avg))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(n_array, mean_error_avg, 'b-', label="average error")
plt.loglog(n_array, upper_error_avg, 'y-', label="upper bound error")
plt.loglog(n_array, np.exp(c02) * n_array ** c12, 'or--', label="regression")
plt.loglog(n_array, np.exp(c02) * n_array ** (-1), 'sg--', label="order 1")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

n_array = np.array([8., 16., 32., 64.])
c_arr = scipy.stats.linregress(np.log(n_array), np.log(mean_error_avg[2:]))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(n_array, mean_error_avg[2:], 'b-', label="average error")
plt.loglog(n_array, upper_error_avg[2:], 'y-', label="upper bound error")
plt.loglog(n_array, np.exp(c02) * n_array ** c12, 'or--', label="regression")
plt.loglog(n_array, np.exp(c02) * n_array ** (-1), 'sg--', label="order 1")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

n_array = np.array([2., 4., 8., 16., 32., 64.])
c_arr = scipy.stats.linregress(np.log(n_array), np.log(mean_error_avg))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(n_array, mean_error_avg, 'b-', label="average error")
plt.loglog(n_array, upper_error_avg, 'y-', label="upper bound error")
plt.loglog(n_array, np.exp(c02) * n_array ** c12, 'or--', label="regression")
plt.loglog(n_array, np.exp(c02) * n_array ** (-1), 'sg--', label="order 1")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

n_array = np.array([8., 16., 32., 64.])
c_arr = scipy.stats.linregress(np.log(n_array), np.log(mean_error_avg[2:]))
c12 = c_arr[0]
c02 = c_arr[1]
print(c12, c02)
plt.loglog(n_array, mean_error_avg[2:], 'b-', label="average error")
plt.loglog(n_array, upper_error_avg[2:], 'y-', label="upper bound error")
plt.loglog(n_array, np.exp(c02) * n_array ** c12, 'or--', label="regression")
plt.loglog(n_array, np.exp(c02) * n_array ** (-1), 'sg--', label="order 1")
plt.legend(loc='upper right')
plt.xlabel('Nodes')
plt.ylabel('Error')
plt.show()

# plt.plot(k_vec, np.array([mi, lo, up, mi2, lo2, up2]).transpose())
# plt.show()
'''
'''

H = 0.1
T = 1.
N = 1000
rho = -0.9
lambda_ = 0.3
theta = 0.02
nu = 0.3
V_0 = 0.02
S_0 = 1
K = 1.
m = 1
n = 20
a = 1.
b = 1.
M = 10000
rHestonAK.plot_rHeston_AK(H, T, N, rho, lambda_, theta, nu, V_0, S_0, m, n, a, b)
rounds = 1
rHestonAK.implied_volatility_call_rHeston_AK(H, T, N, rho, lambda_, theta, nu, V_0, S_0, K_vec, M, rounds)
(mi, lo, up) = rHeston.implied_volatility_call_rHeston(H, T, N, rho, lambda_, theta, nu, V_0, S_0, K_vec, M, rounds)
print("Vector of k:")
print(k_vec)
print("Implied volatility of call options of the rHeston model, mean estimates:")
print(mi)
# print("Implied volatility of call options of the rHeston model, lower estimates:")
# print(lo)
# print("Implied volatility of call options of the rHeston model, upper estimates:")
# print(up)
for n in n_vec:
    print(f"n={n}")
    (mi2, lo2, up2) = rHestonAK.implied_volatility_call_rHeston_AK(H, T, N, rho, lambda_, theta, nu, V_0, S_0, K_vec, m, n, a, b,
                                                         M, rounds)
    print("Implied volatility of call options of the approximate rHeston model, mean estimates:")
    print(mi2)
    # print("Implied volatility of call options of the approximate rHeston model, lower estimates:")
    # print(lo2)
    # print("Implied volatility of call options of the approximate rHeston model, upper estimates:")
    # print(up2)
    # print("Mean error estimate:")
    mean_error = np.fabs(mi2 - mi)
    # print(mean_error)
    # print("Upper error estimate:")
    upper_error = np.fmax(np.fabs(up2 - lo), np.fabs(up - lo2))
    # print(upper_error)
    print("Mean error integral:")
    print(np.average(mean_error))
    print("Upper error integral:")
    print(np.average(upper_error))
    print("Mean error for k=0.1:")
    print(mean_error[50])
    print("Upper error for k=0.1:")
    print(upper_error[50])
# plt.plot(k_vec, np.array([mi, lo, up, mi2, lo2, up2]).transpose())
# plt.show()
