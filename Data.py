import math

import numpy as np
import matplotlib.pyplot as plt
import scipy
import mpmath as mp
from scipy import stats
import RoughKernel as rk


def A_H(H):
    """
    Determines the constant A which is often used in the paper.
    :param H: Hurst parameter
    :return: A
    """
    return np.sqrt(1/H + 1/(1.5-H))


"""
The strong L^2-approximation errors of fBm with H=0.1 and T=1 using the AK scheme.
A node at x_0=0 is used, with optimal weight. Optimal xi_0 and xi_n are used.
The values of m used are contained in fBm_M, the values of n used for a specific m is contained in fBm_N[m-1].
The errors are given in fBm_errors, with the same indices as fBm_N.
An error entry with the value 0 is a numerical 0, i.e. the rounding error in the computation of the approximation 
error already exceeded the approximation error. This is, as it was already possible to choose xi_0 and xi_n such that
the computed error was negative before applying the root (which is necessary to compute the L^2-norm).
The xi_0s and xi_ns are given in fBm_a and fBm_b, respectively, where xi_0 = e^(-a) and xi_n = e^b.
fBm_errors_thm is the vector of errors achieved with the choices of m, xi_0 and xi_n as in the theorem.
fBm_errors_bound is the corresponding bound of the theorem.
fBm_errors_opt_1 are the errors achieved with the same m as in the theorem, but optimal xi_0 and xi_n.
fBm_errors_opt_2 are the errors achieved with optimal m, xi_0 and xi_n.
"""

fBm_m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fBm_n = [[1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024],
         [1, 2, 3, 4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512],
         [1, 2, 3, 4, 5, 8, 11, 15, 21, 30, 43, 60, 85, 121, 171, 241, 341],
         [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256],
         [1, 2, 3, 5, 6, 9, 13, 18, 26, 36, 51, 72, 102, 145, 205],
         [1, 2, 3, 4, 5, 7, 11, 15, 21, 30, 43, 60, 85, 121, 171],
         [1, 2, 3, 5, 6, 9, 13, 18, 26, 37, 52, 73, 103, 146],
         [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128],
         [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 57, 80, 114],
         [1, 2, 3, 4, 6, 9, 13, 18, 26, 36, 51, 72, 102]]

fBm_errors = [[0.683687, 0.528237, 0.420265, 0.346109, 0.253310, 0.199291, 0.149395, 0.098625, 0.065529, 0.043699,
               0.028083, 0.017409, 0.010589, 0.006437, 0.003830, 0.002251, 0.001309, 0.000754, 0.000431, 0.000245],
          [0.635407, 0.451472, 0.333850, 0.256923, 0.204204, 0.119592, 0.076692, 0.039571, 0.022815, 0.010167, 0.004749,
           0.002037, 0.000866, 0.000342, 0.000132, 4.91e-05, 1.78e-05, 6.26e-06],
          [0.604933, 0.405954, 0.285941, 0.210481, 0.160789, 0.087336, 0.049924, 0.025489, 0.012368, 0.004540, 0.001559,
           0.000543, 0.000158, 4.34e-05, 1.14e-05, 2.86e-06, 6.71e-07],
          [0.582543, 0.374196, 0.253831, 0.180499, 0.103376, 0.069539, 0.035804, 0.013985, 0.005244, 0.001812, 0.000526,
           0.000123, 2.74e-05, 5.37e-06, 9.46e-07, 1.57e-07],
          [0.564594, 0.350288, 0.230415, 0.115370, 0.087745, 0.044988, 0.017360, 0.006881, 0.002035, 0.000596, 0.000130,
           2.47e-05, 3.79e-06, 4.99e-07, 6.03e-08],
          [0.549739, 0.331373, 0.212379, 0.143411, 0.101948, 0.048047, 0.021629, 0.008499, 0.003022, 0.000708, 0.000140,
           2.59e-05, 3.50e-06, 3.66e-07, 3.42e-08],
          [0.537137, 0.315874, 0.197939, 0.091741, 0.068621, 0.031377, 0.010481, 0.003575, 0.000841, 0.000167, 2.69e-05,
           3.64e-06, 3.76e-07, 2.47e-08],
          [0.526234, 0.302840, 0.186042, 0.120975, 0.062400, 0.035876, 0.014809, 0.004062, 0.001044, 0.000225, 3.50e-05,
           3.82e-06, 3.47e-07, 2.09e-08],
          [0.516652, 0.291658, 0.176020, 0.112708, 0.077254, 0.043063, 0.017208, 0.005369, 0.001490, 0.000283, 4.10e-05,
           4.35e-06, 3.68e-07, 1.98e-08],
          [0.508122, 0.281912, 0.167430, 0.105750, 0.053486, 0.020847, 0.005956, 0.001722, 0.000307, 5.28e-05, 5.47e-06,
           4.40e-07, 2.21e-08]]

fBm_a = [[83.372, 1.4621, 0.9955, 0.7776, 0.5692, 0.5037, 2.1194, 1.6463, 2.2868, 2.3096, 2.7542, 3.1359, 3.5514,
          3.9615, 4.3852, 4.8150, 5.2495, 5.6885, 6.1308, 6.5766],
         [78.464, 0.9170, 0.5756, 0.4179, 0.3329, 0.2977, 2.3260, 1.8629, 1.8042, 2.7007, 3.4216, 4.4629, 5.1279,
          5.8933, 6.6406, 7.3999, 8.1777, 8.9733],
         [82.712, 0.7031, 0.4181, 0.2922, 0.2359, 0.3142, 2.4605, 2.0758, 1.9799, 3.1222, 3.9939, 4.6656, 6.5970,
          7.0621, 8.2385, 9.3413, 10.441],
         [4.8078, 0.5883, 0.3380, 0.2356, 0.2181, 0.4436, 2.6083, 2.1926, 3.9116, 3.4273, 6.0055, 6.2656, 7.4233,
          9.7630, 10.038, 11.276],
         [3.2387, 0.5156, 0.2903, 0.1957, 0.2360, 3.0594, 2.5352, 2.3270, 3.9750, 5.3305, 6.1260, 7.9805, 8.9325,
          10.210, 12.162],
         [2.5502, 0.4652, 0.2594, 0.1932, 0.2000, 1.4035, 2.8848, 2.5445, 4.7272, 4.0711, 5.3315, 7.8346, 9.4066,
          10.098, 11.873],
         [2.1460, 0.4278, 0.2384, 0.2106, 0.3022, 1.9567, 2.8012, 2.6517, 4.5182, 6.0340, 7.0503, 7.7444, 9.3630,
          12.204],
         [1.8755, 0.3990, 0.2235, 0.1852, 0.3460, 1.6109, 3.1203, 2.7761, 5.0011, 4.5961, 6.0147, 8.6366, 10.482,
          12.786],
         [1.6800, 0.3759, 0.2128, 0.1866, 0.2404, 1.3406, 3.3521, 2.9600, 4.0539, 4.8673, 6.4910, 7.6550, 9.9755,
          13.029],
         [1.5310, 0.3570, 0.2050, 0.1899, 0.4795, 2.1588, 3.1268, 3.0532, 5.1764, 7.0675, 8.3294, 11.065, 11.960]]

fBm_b = [[4.8778, 9.1800, 12.063, 14.455, 18.342, 21.394, 23.917, 28.971, 33.278, 37.865, 42.606, 47.765, 53.091,
          58.404, 63.928, 69.563, 75.288, 81.096, 86.970, 92.910],
         [5.9852, 10.528, 13.957, 16.896, 19.508, 26.004, 29.534, 36.893, 42.621, 51.739, 59.665, 68.195, 77.107,
          86.737, 96.621, 106.85, 117.35, 128.10],
         [6.6458, 11.451, 15.300, 18.659, 21.690, 29.339, 33.469, 40.685, 49.580, 58.953, 70.067, 81.434, 93.266,
          106.91, 120.50, 134.65, 149.46],
         [7.0473, 12.162, 16.350, 20.051, 26.561, 31.785, 36.574, 46.624, 56.342, 68.460, 79.889, 95.048, 110.51,
          126.96, 145.05, 163.28],
         [7.3466, 12.742, 17.214, 24.882, 28.306, 33.957, 43.829, 54.265, 65.829, 78.297, 93.953, 110.73, 130.02,
          150.94, 172.37],
         [7.5974, 13.233, 17.950, 22.193, 26.131, 33.826, 41.335, 51.080, 61.398, 76.958, 93.416, 109.87, 130.22,
          153.65, 175.89],
         [7.8138, 13.660, 18.591, 27.228, 31.098, 37.809, 48.623, 60.867, 74.334, 90.692, 109.36, 131.24, 154.98,
          180.16],
         [8.0042, 14.037, 19.161, 23.829, 32.230, 36.562, 44.946, 58.519, 71.899, 89.463, 107.94, 128.86, 153.06,
          180.82],
         [8.1745, 14.376, 19.673, 24.525, 29.088, 34.741, 43.315, 55.225, 68.712, 85.238, 104.73, 128.13, 152.45,
          180.30],
         [8.3284, 14.683, 20.139, 25.159, 33.989, 41.749, 54.035, 68.429, 84.115, 101.87, 124.85, 150.28, 179.76]]


fBm_errors_thm = [0.996490, 0.975001, 0.938847, 0.899764, 0.823498, 0.757286, 0.675217, 0.571030, 0.466534, 0.372303,
                  0.280328, 0.195570, 0.126123, 0.075222, 0.039853, 0.018481, 0.007324, 0.002405, 0.000633, 0.000128]
fBm_errors_bound = [4.190757, 4.089250, 3.933230, 3.773728, 3.477871, 3.218395, 2.888600, 2.455046, 2.007894, 1.599424,
                    1.199642, 0.833625, 0.534459, 0.316916, 0.167002, 0.077112, 0.030462, 0.009982, 0.002624, 0.000529]
fBm_errors_opt_1 = [0.683687, 0.528237, 0.420265, 0.346109, 0.253310, 0.199291, 0.149395, 0.098625, 0.065529,
                              0.043699, 0.028083, 0.010167, 0.004749, 0.002037, 0.000866, 0.000158, 4.34e-05, 1.14e-05,
                              9.46e-07, 6.03e-08]
fBm_errors_opt_2 = [0.683687, 0.528237, 0.420265, 0.346109, 0.253310, 0.199291, 0.149395, 0.098625, 0.065529,
                              0.039571, 0.022815, 0.010167, 0.004540, 0.001559, 0.000526, 0.000123, 2.47e-05, 3.50e-06,
                              3.47e-07, 1.98e-08]
fBm_errors_reg = [0.917761, 0.697745, 0.513551, 0.389907, 0.268290, 0.211681, 0.154789, 0.098789, 0.065748, 0.041534,
                  0.023458, 0.010345, 0.004874, 0.001611, 0.000558, 0.000124, 2.60e-05, 3.72e-06, 3.54e-07, 2.24e-08]
fBm_errors_reg_bound = [1.307860, 1.041449, 0.874442, 0.754638, 0.589374, 0.478511, 0.365847, 0.251243, 0.162193,
                        0.101018, 0.056665, 0.027849, 0.011944, 0.004502, 0.001388, 0.000342, 6.48e-05, 8.94e-06,
                        8.50e-07, 5.16e-08]
fBm_errors_Harms_1 = [np.nan, 1.408506, 1.355444, 1.318878, 1.269295, 1.235692, 1.200116, 1.160442, 1.124180, 1.092909,
                      1.062220, 1.032075, 1.003400, 0.976870, 0.951093, 0.926378, 0.902669, 0.879856, 0.857899,
                      0.836708]
fBm_errors_Harms_10 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.996490, np.nan, 0.836713, 0.632356, 0.521569,
                       0.398268, 0.304157, 0.238136, 0.191739, 0.150077, 0.120818, 0.095788, 0.076117, 0.060342]
fBm_errors_AK = [1.093956, 1.038641, 1.008279, 0.987654, 0.959794, 0.940801, 0.920437, 0.897258, 0.875517, 0.856277,
                 0.836913, 0.817406, 0.798393, 0.780402, 0.762558, 0.745117, 0.728091, 0.711447, 0.695196, 0.679308]


def plot_fBm_errors():
    """
    Plots a loglog-plot of the strong L^2-errors of approximating a fBm with H=0.1 and T=1 for varying n and m.
    Includes a node at x_0=0 and takes optimal values of xi_0 and xi_n.
    Also plot a loglog-plot comparing the errors of the choice of the theorem with the bound of the theorem, and the
    optimal choice for fBm with H=0.1 and T=1.
    """
    for m in fBm_m:
        n = np.array(fBm_n[m - 1])
        errors = np.array(fBm_errors[m-1])
        plt.loglog(m*n+1, errors, label=f"m={m}")

    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes")
    plt.ylabel("Error")
    plt.show()

    plt.loglog(fBm_n[0], fBm_errors_thm, 'b-', label="Theorem")
    plt.loglog(fBm_n[0], fBm_errors_bound, 'b--', label="Bound")
    plt.loglog(fBm_n[0], fBm_errors_opt_1, 'r-', label="Optimal xi, same m")
    plt.loglog(fBm_n[0], fBm_errors_opt_2, 'g-', label="Optimal xi and m")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes")
    plt.ylabel("Error")
    plt.show()

    plt.loglog(fBm_n[0], fBm_errors_thm, 'b-', label="Theorem")
    plt.loglog(fBm_n[0], fBm_errors_bound, 'b--', label="Theorem bound")
    plt.loglog(fBm_n[0], fBm_errors_reg, 'r-', label="Estimates")
    plt.loglog(fBm_n[0], fBm_errors_reg_bound, 'r--', label="Estimates bound")
    plt.loglog(fBm_n[0], fBm_errors_opt_2, 'g-', label="Optimal xi and m")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes")
    plt.ylabel("Error")
    plt.show()

    plt.loglog(fBm_n[0], fBm_errors_AK, 'y-', label="Alfonsi, Kebaier")
    plt.loglog(fBm_n[0], fBm_errors_Harms_1, 'c-', label="Harms, m=1")
    plt.loglog(fBm_n[0], fBm_errors_Harms_10, 'm-', label="Harms, m=10")
    plt.loglog(fBm_n[0], fBm_errors_thm, 'b-', label="Theorem")
    plt.loglog(fBm_n[0], fBm_errors_reg, 'r-', label="Estimates")
    plt.loglog(fBm_n[0], fBm_errors_opt_2, 'g-', label="Optimal xi and m")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes")
    plt.ylabel("Error")
    plt.show()


"""
The strong L^2-approximation errors of fBm with varying Hurst parameter H and T=1 using the AK scheme.
A node at x_0=0 is used, with optimal weight. Optimal xi_0 and xi_n are used.
Given N, optimal values of m and n are used, such that m*n is approximately N, and optimal xi_0 and xi_n are computed.
The values of H used are contained in fBm_H.
The values of m used are contained in fBm_m_best. This is a list, the ith entry of which corresponds to fBm_H[i].
The values of n used are contained in fBm_n_best. This is a list, the ith entry of which corresponds to fBm_H[i].
The values of N used are contained in fBm_N_best. This is a list, the ith entry of which corresponds to fBm_H[i].
Of course, fBm_N_best[i] = fBm_m_best[i] * fBm_n_best[i].
The values of a used are contained in fBm_a_best. This is a list, the ith entry of which corresponds to fBm_H[i].
Here, xi_0 = e^(-a).
The values of b used are contained in fBm_b_best. This is a list, the ith entry of which corresponds to fBm_H[i].
Here, xi_n = e^b.
The values of the computed errors are contained in fBm_error_best.
This is a list, the ith entry of which corresponds to fBm_H[i].
The optimization was performed for all H in fBm_H and all N that are (roughly) a power of sqrt(2). No optimization was
performed anymore if N was so large that the computed error was zero, indicating that the rounding errors in the
error computation already exceeded the approximation error.
"""

fBm_H = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])

fBm_m_best_005 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 6, 6])
fBm_n_best_005 = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 32, 45, 64, 60, 85, 90, 128, 121, 171])
fBm_N_best_005 = fBm_m_best_005 * fBm_n_best_005
fBm_a_best_005 = np.array([68.267, 1.6212, 1.1705, 0.9570, 0.7207, 0.5831, 0.4667, 0.4238, 1.7465, 1.4611, 2.0026,
                           1.7277, 2.9716, 2.5811, 2.9807, 3.8344, 4.2782, 6.0107, 7.5302, 7.7535])
fBm_b_best_005 = np.array([5.8092, 11.078, 14.803, 17.974, 23.319, 27.793, 33.424, 40.847, 47.970, 56.159, 64.609,
                           72.769, 85.834, 101.84, 117.20, 138.56, 161.81, 190.46, 222.80, 264.50])
fBm_error_best_005 = np.array([1.309316, 1.117131, 0.968190, 0.851984, 0.682578, 0.565580, 0.446300, 0.326527, 0.231697,
                               0.162855, 0.110980, 0.063447, 0.033678, 0.016317, 0.007062, 0.002514, 0.000782, 0.000185,
                               3.46e-05, 4.72e-06])

fBm_m_best_01 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 8, 9])
fBm_n_best_01 = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 16, 22, 32, 30, 43, 45, 64, 72, 85, 90, 114])
fBm_N_best_01 = fBm_m_best_01 * fBm_n_best_01
fBm_a_best_01 = np.array([83.372, 1.4621, 0.9955, 0.7776, 0.5692, 0.5037, 2.1194, 1.6463, 2.2868, 1.8629, 1.8042,
                          2.7007, 3.1222, 3.9939, 6.0055, 6.2656, 7.9805, 9.4066, 10.482, 13.029])
fBm_b_best_01 = np.array([4.8778, 9.1800, 12.063, 14.455, 18.342, 21.394, 23.917, 28.971, 33.278, 36.893, 43.621,
                          51.739, 58.953, 70.067, 79.889, 95.048, 110.73, 130.22, 153.06, 180.30])
fBm_error_best_01 = np.array([0.683687, 0.528237, 0.420265, 0.346109, 0.253310, 0.199291, 0.149395, 0.098625, 0.065529,
                              0.039571, 0.022815, 0.010167, 0.004540, 0.001559, 0.000526, 0.000123, 2.47e-05, 3.50e-06,
                              3.47e-07, 1.98e-08])

fBm_m_best_015 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 6, 7, 8, 9])
fBm_n_best_015 = np.array([1, 2, 3, 4, 6, 8, 11, 16, 11, 16, 22, 32, 30, 43, 45, 43, 52, 64, 80])
fBm_N_best_015 = fBm_n_best_015 * fBm_m_best_015
fBm_a_best_015 = np.array([97.792, 1.3180, 0.8893, 0.7042, 0.5828, 2.2974, 1.8262, 2.5641, 1.9724, 3.3083, 2.8285,
                           4.5125, 4.1137, 5.7362, 6.3405, 7.7740, 9.2238, 10.457, 12.534])
fBm_b_best_015 = np.array([4.5220, 8.2057, 10.703, 12.738, 15.888, 17.057, 20.231, 23.153, 25.328, 29.982, 35.606,
                           40.913, 48.118, 55.839, 64.978, 75.486, 88.702, 104.72, 121.64])
fBm_error_best_015 = np.array([0.408023, 0.291038, 0.217730, 0.172744, 0.123584, 0.093575, 0.064660, 0.042253, 0.026360,
                               0.013057, 0.006725, 0.002818, 0.000988, 0.000296, 7.39e-05, 1.38e-05, 1.90e-06,
                               1.97e-07, 9.62e-09])

fBm_m_best_02 = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 6, 7, 9])
fBm_n_best_02 = np.array([1, 2, 3, 4, 6, 8, 11, 8, 11, 16, 22, 21, 23, 32, 36, 43, 52, 57])
fBm_N_best_02 = fBm_m_best_02 * fBm_n_best_02
fBm_a_best_02 = np.array([4.3079, 1.1858, 0.8357, 0.7001, 2.6324, 2.1024, 1.8360, 2.1377, 1.9377, 3.0130, 3.8015,
                          4.4170, 4.7479, 6.7955, 7.0644, 8.4979, 11.165, 12.317])
fBm_b_best_02 = np.array([4.2163, 7.5560, 9.8627, 11.729, 12.987, 15.399, 17.871, 18.776, 22.910, 26.700, 30.468,
                          34.754, 41.188, 47.071, 55.554, 65.397, 75.529, 89.709])
fBm_error_best_02 = np.array([0.260448, 0.168793, 0.120762, 0.094786, 0.066514, 0.046403, 0.033040, 0.020377, 0.011986,
                              0.005112, 0.002467, 0.000910, 0.000290, 7.35e-05, 1.60e-05, 2.28e-06, 2.42e-07, 1.64e-08])

fBm_m_best_025 = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 6, 8])
fBm_n_best_025 = np.array([1, 2, 3, 4, 6, 8, 11, 8, 11, 16, 22, 21, 23, 32, 45, 43, 45])
fBm_N_best_025 = fBm_n_best_025 * fBm_m_best_025
fBm_a_best_025 = np.array([2.3321, 1.0710, 0.8200, 0.7376, 2.4732, 2.0544, 3.0027, 2.0776, 3.4274, 4.2612, 4.7492,
                           5.5759, 5.8976, 7.5928, 8.5448, 10.496, 11.752])
fBm_b_best_025 = np.array([3.8767, 7.0555, 9.3000, 11.155, 12.139, 14.370, 15.677, 17.500, 19.617, 23.350, 26.860,
                           30.554, 36.171, 41.634, 49.766, 56.788, 66.510])
fBm_error_best_025 = np.array([0.168614, 0.099223, 0.068853, 0.055027, 0.034511, 0.024598, 0.016651, 0.009904, 0.005078,
                               0.002157, 0.001009, 0.000335, 9.10e-05, 2.16e-05, 4.49e-06, 4.77e-07, 4.50e-08])

fBm_m_best_03 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 7])
fBm_n_best_03 = np.array([1, 2, 3, 4, 5, 6, 11, 16, 11, 16, 15, 21, 23, 32, 36, 37, 52])
fBm_N_best_03 = fBm_m_best_03 * fBm_n_best_03
fBm_a_best_03 = np.array([1.6429, 0.9825, 0.8302, 0.7937, 2.4023, 2.1053, 2.9969, 4.1563, 3.3196, 4.1298, 4.6673,
                          5.3973, 7.0179, 8.4061, 9.9614, 11.486, 13.662])
fBm_b_best_03 = np.array([3.6182, 6.6441, 8.9214, 10.967, 11.576, 13.866, 14.859, 16.050, 18.518, 22.071, 24.129,
                          28.906, 32.318, 37.557, 43.300, 51.217, 59.517])
fBm_error_best_03 = np.array([0.107225, 0.057602, 0.039323, 0.032875, 0.017865, 0.013574, 0.008582, 0.005222, 0.002396,
                              0.001009, 0.000416, 0.000148, 3.25e-05, 7.18e-06, 1.19e-06, 1.17e-07, 9.18e-09])

fBm_m_best_035 = np.array([1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 5, 7])
fBm_n_best_035 = np.array([1, 2, 3, 4, 6, 8, 11, 8, 8, 11, 15, 21, 23, 32, 36, 37])
fBm_N_best_035 = fBm_n_best_035 * fBm_m_best_035
fBm_a_best_035 = np.array([1.3019, 0.9261, 0.8562, 2.9885, 2.3901, 3.7381, 3.1191, 3.7219, 3.9612, 5.2135, 6.0331,
                           6.4369, 8.1268, 9.2378, 10.752, 12.438])
fBm_b_best_035 = np.array([3.4146, 6.3027, 8.6800, 8.1831, 11.232, 11.992, 14.563, 14.003, 15.770, 18.273, 21.410,
                           25.911, 29.225, 34.349, 39.673, 46.952])
fBm_error_best_035 = np.array([0.064887, 0.032008, 0.021847, 0.017005, 0.008927, 0.006111, 0.004370, 0.002417, 0.001256,
                               0.000526, 0.000184, 5.38e-05, 1.25e-05, 2.53e-06, 3.85e-07, 3.41e-08])

fBm_m_best_04 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 7])
fBm_n_best_04 = np.array([1, 2, 3, 4, 6, 8, 11, 16, 11, 16, 22, 21, 30, 32, 36, 37])
fBm_N_best_04 = fBm_m_best_04 * fBm_n_best_04
fBm_a_best_04 = np.array([1.1126, 0.9012, 0.8895, 2.7998, 2.4092, 3.6982, 3.3562, 5.5013, 4.6677, 5.2914, 6.6539,
                          7.4915, 9.4901, 10.089, 11.610, 14.602])
fBm_b_best_04 = np.array([3.2493, 6.0232, 8.5383, 7.8072, 11.045, 11.721, 15.645, 14.431, 15.459, 19.324, 21.535,
                          23.486, 27.174, 31.722, 36.828, 42.203])
fBm_error_best_04 = np.array([0.035262, 0.016065, 0.011161, 0.007855, 0.004067, 0.002482, 0.001996, 0.000948, 0.000460,
                              0.000172, 6.58e-05, 1.98e-05, 4.48e-06, 8.65e-07, 1.28e-07, 9.14e-09])

fBm_m_best_045 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
fBm_n_best_045 = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 16, 22, 32, 30, 43, 45])
fBm_N_best_045 = fBm_m_best_045 * fBm_n_best_045
fBm_a_best_045 = np.array([1.0050, 0.9038, 0.9256, 2.6467, 2.4356, 3.6513, 4.5937, 5.9468, 6.9425, 6.3889, 6.8332,
                           9.8940, 10.315, 11.997, 13.008])
fBm_b_best_045 = np.array([3.1111, 5.7994, 8.4604, 7.4577, 10.930, 11.475, 13.372, 14.892, 15.757, 17.238, 21.852,
                           22.632, 25.237, 29.762, 34.545])
fBm_error_best_045 = np.array([0.014476, 0.006129, 0.004411, 0.002771, 0.001430, 0.000736, 0.000410, 0.000224, 0.000132,
                               4.69e-05, 2.07e-05, 5.02e-06, 1.20e-06, 2.11e-07, 3.40e-08])

fBm_m_best = [fBm_m_best_005, fBm_m_best_01, fBm_m_best_015, fBm_m_best_02, fBm_m_best_025, fBm_m_best_03,
              fBm_m_best_035, fBm_m_best_04, fBm_m_best_045]
fBm_n_best = [fBm_n_best_005, fBm_n_best_01, fBm_n_best_015, fBm_n_best_02, fBm_n_best_025, fBm_n_best_03,
              fBm_n_best_035, fBm_n_best_04, fBm_n_best_045]
fBm_N_best = [fBm_N_best_005, fBm_N_best_01, fBm_N_best_015, fBm_N_best_02, fBm_N_best_025, fBm_N_best_03,
              fBm_N_best_035, fBm_N_best_04, fBm_N_best_045]
fBm_a_best = [fBm_a_best_005, fBm_a_best_01, fBm_a_best_015, fBm_a_best_02, fBm_a_best_025, fBm_a_best_03,
              fBm_a_best_035, fBm_a_best_04, fBm_a_best_045]
fBm_b_best = [fBm_b_best_005, fBm_b_best_01, fBm_b_best_015, fBm_b_best_02, fBm_b_best_025, fBm_b_best_03,
              fBm_b_best_035, fBm_b_best_04, fBm_b_best_045]
fBm_error_best = [fBm_error_best_005, fBm_error_best_01, fBm_error_best_015, fBm_error_best_02, fBm_error_best_025,
                  fBm_error_best_03, fBm_error_best_035, fBm_error_best_04, fBm_error_best_045]


def plot_beta(H, N, m):
    """
    Assumes that m = beta/A * sqrt(N), and determines beta using that formula. For every H, plots the thus
    determined beta as a function of N.
    :param H: Vector of Hurst parameters
    :param N: List of numpy-arrays of N-values (number of nodes, N=n*m), where N[i] corresponds to H[i]
    :param m: List of numpy-arrays of m-values (levels of quadrature nodes), where m[i] corresponds to H[i]
    :return: Nothing, produces a plot of beta
    """
    for i in range(len(H)):
        beta = m[i][2:] * A_H(H[i]) / np.sqrt(N[i][2:])  # excludes small N to avoid messing up the plot
        plt.plot(np.log10(N[i][2:]), beta, label=f"H={H[i]}")
    plt.xlabel("log_10(N)")
    plt.ylabel("Estimate for beta")
    plt.legend(loc="upper right")
    plt.show()


def plot_alpha(H, N, a, b, error):
    """
    Assumes that
    xi_0 = exp(-alpha/((1.5-H)*A) * sqrt(N)),
    xi_n = exp(alpha/(H*A) * sqrt(N)),
    error = exp(-alpha/A * sqrt(N)).
    Determines alpha using these formulas. For every H, plots the thus determined alpha as a function of N. Every H
    produces three such functions, one given by xi_0, one by xi_n and one by error.
    :param H: Vector of Hurst parameters
    :param N: List of numpy-arrays of N-values (number of nodes, N=n*m), where N[i] corresponds to H[i]
    :param a: List of numpy-arrays of a-values (xi_0 = e^(-a)), where a[i] corresponds to H[i]
    :param b: List of numpy-arrays of b-values (xi_n = e^b), where b[i] corresponds to H[i]
    :param error: List of numpy-arrays of estimated errors, where error[i] corresponds to H[i]
    :return: Nothing, produces a plot of alpha
    """
    for i in range(len(H)):
        H_ = H[i]
        N_ = N[i][3:]  # excludes the first 3 elements as these occasionally produce awkward results, messing up the
        a_ = a[i][3:]  # plot
        b_ = b[i][3:]
        error_ = error[i][3:]
        A = A_H(H_)
        error_ = np.log(error_)
        a_ = (1.5 - H_) * A / np.sqrt(N_) * a_
        b_ = H_ * A / np.sqrt(N_) * b_
        error_ = -A / np.sqrt(N_) * error_

        plt.plot(np.log10(N_), a_, label=f"H={H_}, xi_0")
        plt.plot(np.log10(N_), b_, label=f"H={H_}, xi_n")
        plt.plot(np.log10(N_), error_, label=f"H={H_}, error")
    plt.xlabel("log_10(N)")
    plt.ylabel("Estimate for alpha")
    plt.show()


def fit_observations():
    """
    Prints some approximations of the values of a, b, m determined by optimization, as well as the computed expected
    L^2 strong approximation errors of fBm with varying Hurst parameter using the AK approximation scheme with varying
    number of nodes N. Also shows some plots that illustrate the accuracy of the approximations.
    :return: Nothing, prints some approximations that are determined by regression, and shows some plots
    """
    H = fBm_H
    error = fBm_error_best.copy()
    b = fBm_b_best.copy()
    a = fBm_a_best.copy()
    m = fBm_m_best.copy()
    N = fBm_N_best.copy()

    A = A_H(H)

    plot_beta(H, N, m)

    plot_alpha(H, N, a, b, error)
    alpha = 1.8

    for i in range(len(H)):
        a[i] = a[i] - alpha / ((1.5 - H[i]) * A[i]) * np.sqrt(N[i])
        b[i] = b[i] - alpha / (H[i] * A[i]) * np.sqrt(N[i])
        error[i] = error[i] * np.exp(alpha / A[i] * np.sqrt(N[i]))

    for i in range(len(H)):
        plt.loglog(N[i][3:], np.exp(a[i][3:]), label=f"H={H[i]}")
    plt.xlabel("N")
    plt.ylabel("Factor by which xi_0 differs from its optimum")
    plt.legend(loc="upper right")
    plt.show()

    avg_factor_xi_0 = np.array([np.average(np.exp(a[i][3:])) for i in range(len(H))])
    res = scipy.stats.linregress(H, np.log(avg_factor_xi_0))
    gamma = res[0]
    C = res[1]
    print(f"A good fit is achieved using xi_0 = {np.round(np.exp(C), 4)} * exp({np.round(gamma, 4)}*H) * exp(-{alpha}/((1.5-H)*A) * sqrt(N))")
    plt.plot(H, np.log(avg_factor_xi_0), label="True log-factor")
    plt.plot(H, C + gamma*H, label="regression")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which xi_0 differs from its optimum")
    plt.legend(loc="upper left")
    plt.show()

    avg_factor_xi_0 = avg_factor_xi_0 / np.exp(C + gamma*H)
    plt.plot(H, np.log(avg_factor_xi_0), label="True log-factor")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which xi_0 differs from its optimum")
    # plt.legend(loc="upper right")
    plt.show()

    for i in range(len(H)):
        plt.loglog(N[i][3:], np.exp(b[i][3:]), label=f"H={H[i]}")
    plt.xlabel("N")
    plt.ylabel("Factor by which xi_n differs from its optimum")
    plt.legend(loc="upper right")
    plt.show()

    avg_factor_xi_n = np.array([np.average(np.exp(b[i][6:])) for i in range(len(H))])
    res = scipy.stats.linregress(np.log(H), np.log(np.log(avg_factor_xi_n)))
    gamma = -0.4
    C = 3.
    print(f"A good fit is achieved using xi_n = exp({np.round(C, 4)} * H^{np.round(gamma, 4)}) * exp(-{alpha}/(H*A) * sqrt(N))")
    plt.plot(H, np.log(avg_factor_xi_n), label="True log-factor")
    plt.plot(H, C * H**gamma, label="regression")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which xi_n differs from its optimum")
    plt.legend(loc="upper left")
    plt.show()

    avg_factor_xi_n = avg_factor_xi_n / np.exp(C * H**gamma)
    plt.plot(H, np.log(avg_factor_xi_n), label="True log-factor")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which xi_n differs from its optimum")
    # plt.legend(loc="upper right")
    plt.show()

    for i in range(len(H)):
        plt.loglog(N[i][3:], np.exp(error[i][3:]), label=f"H={H[i]}")
    plt.xlabel("N")
    plt.ylabel("Factor by which the error differs from its optimum")
    plt.legend(loc="upper right")
    plt.show()

    avg_factor_error = np.array([np.average(np.exp(error[i][3:])) for i in range(len(H))])
    res = scipy.stats.linregress(np.log(H), np.log(np.log(avg_factor_error)))
    gamma = -1.1
    C = 0.065
    print(f"A good fit is achieved using error = exp({np.round(C, 4)} * H^{np.round(gamma, 4)}) * exp(-{alpha}/A * sqrt(N))")
    plt.plot(H, np.log(avg_factor_error), label="True log-factor")
    plt.plot(H, C*H**gamma, label="approximation")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which the error differs from its true value")
    plt.legend(loc="upper right")
    plt.show()

    avg_factor_error = avg_factor_error / np.exp(C * H**gamma)
    plt.plot(H, np.log(avg_factor_error), label="True log-factor")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which the error differs from its true value")
    # plt.legend(loc="upper right")
    plt.show()


def q_quadrature_estimate(H=0.1, a=1., b=3., t=1.):
    mp.mp.dps = 100
    H = mp.mpf(H)
    a = mp.mpf(a)
    b = mp.mpf(b)
    t = mp.mpf(t)

    def func(x):
        return mp.matrix([mp.exp(-t*x_) for x_ in x])

    m = 15
    rule = rk.quadrature_rule_interval(H, m, a, b)
    nodes = rule[0, :]
    weights = rule[1, :]
    true_integral = float(np.dot(weights, func(nodes)))
    previous_error = true_integral

    m_vec = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    q_vec = np.empty(shape=(8,))

    for i in range(len(m_vec)):
        m = int(m_vec[i])
        rule = rk.quadrature_rule_interval(H, m, a, b)
        new_integral = np.dot(rule[1, :], func(rule[0, :]))
        new_error = np.fabs(float(new_integral - true_integral))
        q_vec[i] = np.sqrt(new_error / previous_error * (2 * m) * (2 * m - 1) / (float(t*(b - a))) ** 2)
        previous_error = new_error

    plt.plot(m_vec, q_vec)
    plt.xlabel("m")
    plt.ylabel("Estimate for q")
    plt.show()


def C_quadrature_estimate(H=0.1, a=1., b=3., t=1., q=0.25):
    mp.mp.dps = 100
    H = mp.mpf(H)
    a = mp.mpf(a)
    b = mp.mpf(b)
    t = mp.mpf(t)
    q = mp.mpf(q)

    def func(x):
        return mp.matrix([mp.exp(-t * x_) for x_ in x])

    m = 15
    rule = rk.quadrature_rule_interval(H, m, a, b)
    nodes = rule[0, :]
    weights = rule[1, :]
    true_integral = float(np.dot(weights, func(nodes)))
    error = true_integral

    m_vec = np.array([1, 2, 3, 4, 5, 6, 7])
    C_vec = np.empty(shape=(7,))

    for i in range(len(m_vec)):
        m = int(m_vec[i])
        rule = rk.quadrature_rule_interval(H, m, a, b)
        new_integral = np.dot(rule[1, :], func(rule[0, :]))
        error = np.fabs(float(new_integral - true_integral))
        C_vec[i] = t * error * mp.gamma(H+0.5) * mp.gamma(0.5-H) * math.factorial(2*m) * (q*t*(b-a))**(-2*m) * mp.exp(t*a) * a**(0.5+H)

    plt.plot(m_vec, C_vec)
    plt.xlabel("m")
    plt.ylabel("Estimate for C")
    plt.show()


'''
The rBergomi implied volatility smiles, pre-exponential approach. Parameters used are
H=0.07, T=0.9, number_time_steps=1000, eta=1.9, V_0=0.235**2, S_0=1, rho=-0.9.
The vector of log-strikes is given below (k_vec).
m is always chosen 1, n is in [2, 4, 8, 16, 32, 64].
We have xi_0 = n^(-1/(0.5-H)) and xi_n = n^(1/H).
rBergomi is the Bergomi smile for the BFG approximation (with log-strikes k_vec).
rBergomi_n is the Bergomi smile for the AK approximation (with log-strikes k_vec).
mean_error_avg is the arithmetic mean of the absolute values of the errors of the AK approximation compared to the 
BFG approximation, averaged over the log-strikes in k_vec, and indexed by n.
upper_error_avg is the 95% confidence upper bound of the mean_error_avg.
mean_error_01 and upper_error_01 are defined similarly, but they are not averages but the values for k=0.1.
10**6 samples are used for the MC estimates.
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

rBergomi_32 = np.array([0.30448798, 0.30221138, 0.29992409, 0.29762320, 0.29530925, 0.29298094,
                        0.29064271, 0.28829284, 0.28592646, 0.28354444, 0.28115215, 0.27874806,
                        0.27633292, 0.27390937, 0.27147717, 0.26903089, 0.26656888, 0.26409849,
                        0.26161619, 0.25912092, 0.25661453, 0.25409455, 0.25156119, 0.24901402,
                        0.24645042, 0.24387062, 0.24127644, 0.23866657, 0.23604051, 0.23340071,
                        0.23074539, 0.22807996, 0.22540203, 0.22270992, 0.22000639, 0.21729123,
                        0.21456430, 0.21182864, 0.20908149, 0.20632356, 0.20355565, 0.20078378,
                        0.19801083, 0.19524012, 0.19247322, 0.18971951, 0.18698369, 0.18426788,
                        0.18157419, 0.17889870, 0.17624828, 0.17363767, 0.17106947, 0.16855319,
                        0.16610667, 0.16374242, 0.16147046, 0.15929046, 0.15720677, 0.15523531,
                        0.15338483])

rBergomi_64 = np.array([0.31009601, 0.30756878, 0.30503451, 0.30249607, 0.29995251, 0.29740012,
                        0.29483877, 0.29227058, 0.28969126, 0.28709820, 0.28449643, 0.28188293,
                        0.27926019, 0.27662655, 0.27398364, 0.27132884, 0.26866268, 0.26598539,
                        0.26329540, 0.26059775, 0.25789122, 0.25517200, 0.25244314, 0.24970178,
                        0.24694694, 0.24418135, 0.24140260, 0.23860819, 0.23580387, 0.23298832,
                        0.23015886, 0.22731640, 0.22446544, 0.22160756, 0.21873789, 0.21585969,
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


def plot_rBergomi_smiles_BFG_AK():
    """
    Plots the rBergomi implied volatility smiles, pre-exponential approach. Parameters used are
    H=0.07, T=0.9, number_time_steps=1000, eta=1.9, V_0=0.235**2, S_0=1, rho=-0.9.
    Plots both the BFG approximation of the smile and the AK approximations for m=1, and n in [2, 4, 8, 16, 32, 64].
    We have xi_0 = n^(-1/(0.5-H)) and xi_n = n^(1/H).
    10**6 samples are used for the MC estimates.
    """
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


rBergomi_inf_ = np.array([0.30789030, 0.30542406, 0.30294705, 0.30045787, 0.29795548, 0.29543931,
                        0.29291404, 0.29037694, 0.28782456, 0.28525420, 0.28267079, 0.28007381,
                        0.27746035, 0.27483240, 0.27219063, 0.26953660, 0.26687116, 0.26419178,
                        0.26150058, 0.25879475, 0.25607083, 0.25333125, 0.25057885, 0.24781213,
                        0.24503297, 0.24224214, 0.23943759, 0.23661692, 0.23378269, 0.23093540,
                        0.22807654, 0.22520313, 0.22231260, 0.21941178, 0.21650186, 0.21358510,
                        0.21066112, 0.20772714, 0.20478393, 0.20182998, 0.19887752, 0.19593422,
                        0.19299473, 0.19006743, 0.18715063, 0.18424450, 0.18136372, 0.17851307,
                        0.17569899, 0.17293437, 0.17022934, 0.16760297, 0.16506612, 0.16262146,
                        0.16028505, 0.15807886, 0.15601016, 0.15408053, 0.15229629, 0.15065288,
                        0.14915747])

rBergomi_inf_lower = np.array([0.30506697, 0.30271901, 0.30035556, 0.29797540, 0.29557765, 0.2931619,
                        0.29073307, 0.28828855, 0.28582499, 0.28333980, 0.28083813, 0.27831958,
                        0.27578135, 0.27322556, 0.27065303, 0.26806541, 0.26546369, 0.26284541,
                        0.26021281, 0.25756315, 0.25489304, 0.25220500, 0.24950200, 0.24678258,
                        0.24404873, 0.24130128, 0.23853824, 0.23575726, 0.23296098, 0.23014996,
                        0.22732575, 0.22448542, 0.22162642, 0.21875568, 0.21587440, 0.21298492,
                        0.21008686, 0.20717749, 0.20425761, 0.20132571, 0.19839409, 0.19547044,
                        0.19254940, 0.18963940, 0.18673872, 0.18384752, 0.18098052, 0.17814247,
                        0.17533980, 0.17258537, 0.16988930, 0.16727062, 0.16474017, 0.16230056,
                        0.15996781, 0.15776385, 0.15569593, 0.15376554, 0.15197895, 0.15033149,
                        0.14883026])

rBergomi_inf_upper = np.array([0.31066643, 0.30808688, 0.30550079, 0.30290665, 0.30030328, 0.29769000,
                        0.29507127, 0.29244427, 0.28980549, 0.28715212, 0.28448891, 0.28181523,
                        0.27912811, 0.27642938, 0.27371963, 0.27100028, 0.26827210, 0.26553249,
                        0.26278348, 0.26002217, 0.25724504, 0.25445444, 0.25165312, 0.24883949,
                        0.24601539, 0.24318149, 0.24033571, 0.23747558, 0.23460359, 0.23172020,
                        0.22882683, 0.22592047, 0.22299849, 0.22006768, 0.21712918, 0.21418520,
                        0.21123534, 0.20827678, 0.20531026, 0.20233425, 0.19936096, 0.19639801,
                        0.19344005, 0.19049545, 0.18756251, 0.18464141, 0.18174682, 0.17888354,
                        0.17605802, 0.17328316, 0.17056913, 0.16793501, 0.16539170, 0.16294192,
                        0.16060177, 0.15839324, 0.15632365, 0.15439466, 0.15261262, 0.15097308,
                        0.14948326])

rBergomi_1_ = np.array([0.30480983, 0.30257585, 0.30033577, 0.29808625, 0.29583195, 0.29356831,
                        0.29129778, 0.28901699, 0.28672633, 0.28442703, 0.28212066, 0.27980567,
                        0.27748462, 0.27516124, 0.27282841, 0.27049066, 0.26814677, 0.26579317,
                        0.26342888, 0.26105500, 0.25867523, 0.25628713, 0.25389213, 0.25148849,
                        0.24907547, 0.24665616, 0.24422914, 0.24179205, 0.23934799, 0.23689405,
                        0.23442824, 0.23194897, 0.22945564, 0.22695648, 0.22444884, 0.22193449,
                        0.21940799, 0.21687013, 0.21432055, 0.21175899, 0.20918314, 0.20659262,
                        0.20399124, 0.20137817, 0.19875386, 0.19612049, 0.19348339, 0.19084385,
                        0.18819951, 0.18554479, 0.18288256, 0.18022130, 0.17755664, 0.17488855,
                        0.17222316, 0.16956790, 0.16691869, 0.16427007, 0.16163966, 0.15903724,
                        0.15646062])

rBergomi_2_ = np.array([0.30540629, 0.30312438, 0.30084257, 0.29855797, 0.29627244, 0.29398010,
                        0.29168131, 0.28937341, 0.28705826, 0.28474032, 0.28242078, 0.28009544,
                        0.27776093, 0.27541840, 0.27306955, 0.27071129, 0.26835072, 0.26598558,
                        0.26361251, 0.26122968, 0.25884188, 0.25644237, 0.25403074, 0.25160982,
                        0.24918297, 0.24674698, 0.24430076, 0.24185104, 0.23939331, 0.23692482,
                        0.23444783, 0.23196460, 0.22947369, 0.22697581, 0.22447223, 0.22196277,
                        0.21944435, 0.21691673, 0.21437813, 0.21183298, 0.20928082, 0.20672500,
                        0.20416655, 0.20160285, 0.19902902, 0.19644838, 0.19386380, 0.19127790,
                        0.18868710, 0.18609206, 0.18349292, 0.18089195, 0.17829127, 0.17569160,
                        0.17309729, 0.17051526, 0.16794620, 0.16539000, 0.16284859, 0.16033543,
                        0.15784705])

rBergomi_3_ = np.array([0.30348654, 0.30128177, 0.29906986, 0.29684959, 0.29462133, 0.29238440,
                        0.29014315, 0.28789627, 0.28564338, 0.28338318, 0.28111072, 0.27883436,
                        0.27655089, 0.27426278, 0.27197058, 0.26967345, 0.26736634, 0.26505150,
                        0.26272893, 0.26039739, 0.25805849, 0.25571483, 0.25335942, 0.25099583,
                        0.24862221, 0.24623805, 0.24384805, 0.24145288, 0.23904763, 0.23663293,
                        0.23420547, 0.23177131, 0.22932628, 0.22687529, 0.22441647, 0.22194437,
                        0.21946122, 0.21697109, 0.21447187, 0.21196519, 0.20945616, 0.20694319,
                        0.20442275, 0.20189042, 0.19935033, 0.19680522, 0.19425446, 0.19170011,
                        0.18914455, 0.18658421, 0.18402753, 0.18147582, 0.17893294, 0.17639966,
                        0.17387405, 0.17135757, 0.16885583, 0.16636839, 0.16390939, 0.16146288,
                        0.15904155])

rBergomi_4_ = np.array([0.30503112, 0.30274526, 0.30045623, 0.29816753, 0.29587669, 0.29358461,
                        0.29128738, 0.28898378, 0.28667802, 0.28437060, 0.28205819, 0.27973932,
                        0.27741341, 0.27508333, 0.27274801, 0.27040452, 0.26805546, 0.26569589,
                        0.26332884, 0.26095726, 0.25857779, 0.25618876, 0.25379459, 0.25139781,
                        0.24899037, 0.24657293, 0.24414908, 0.24171789, 0.23928302, 0.23684291,
                        0.23439682, 0.23193979, 0.22947752, 0.22700803, 0.22452832, 0.22204074,
                        0.21954479, 0.21704397, 0.21453850, 0.21202671, 0.20950767, 0.20697727,
                        0.20444127, 0.20190043, 0.19935525, 0.19680997, 0.19426885, 0.19173149,
                        0.18919499, 0.18666386, 0.18413860, 0.18160854, 0.17908336, 0.17657237,
                        0.17407214, 0.17159424, 0.16913060, 0.16668923, 0.16427547, 0.16189865,
                        0.15957161])

rBergomi_6_ = np.array([0.30100618, 0.29888191, 0.29674664, 0.29460153, 0.29244765, 0.29028309,
                        0.28811601, 0.28594475, 0.28376190, 0.28156368, 0.27935895, 0.27714667,
                        0.27492826, 0.27270142, 0.27046590, 0.26822184, 0.26596405, 0.26369848,
                        0.26141901, 0.25912989, 0.25683427, 0.25453191, 0.25221850, 0.24989498,
                        0.24756160, 0.24522339, 0.24287613, 0.24052060, 0.23815887, 0.23579139,
                        0.23341254, 0.23102359, 0.22862599, 0.22622412, 0.22381492, 0.22140092,
                        0.21898128, 0.21654982, 0.21410636, 0.21165469, 0.20920025, 0.20673816,
                        0.20427120, 0.20180326, 0.19934035, 0.19687492, 0.19440485, 0.19193745,
                        0.18946705, 0.18700228, 0.18454187, 0.18209442, 0.17966039, 0.17724452,
                        0.17484087, 0.17245167, 0.17009309, 0.16775695, 0.16543809, 0.16315510,
                        0.16091630])

rBergomi_44_ = np.array([0.30256652, 0.30032628, 0.29808558, 0.29584752, 0.29360803, 0.29136634,
                        0.28912116, 0.28687068, 0.28461533, 0.28235926, 0.28009874, 0.27783346,
                        0.27556198, 0.27328920, 0.27101202, 0.26873001, 0.26644111, 0.26414327,
                        0.26183895, 0.25953218, 0.25722115, 0.25490193, 0.25257843, 0.25025516,
                        0.24792705, 0.24559397, 0.24326257, 0.24093199, 0.23860107, 0.23626540,
                        0.23392712, 0.23158744, 0.22924503, 0.22690183, 0.22455993, 0.22221795,
                        0.21987435, 0.21752784, 0.21518150, 0.21283615, 0.21049424, 0.20815678,
                        0.20581820, 0.20348530, 0.20116175, 0.19884538, 0.19654078, 0.19424533,
                        0.19195921, 0.18968875, 0.18743863, 0.18521250, 0.18301019, 0.18083717,
                        0.17869708, 0.17659576, 0.17453566, 0.17252742, 0.17056675, 0.16864404,
                        0.16678240]
)


def plot_rBergomi_smiles_BFG_AK_():
    """
    Plots the rBergomi implied volatility smiles, pre-exponential approach. Parameters used are
    H=0.07, T=0.9, number_time_steps=1000, eta=1.9, V_0=0.235**2, S_0=1, rho=-0.9.
    Plots both the BFG approximation of the smile and the AK approximations for m=1, and n in [2, 4, 8, 16, 32, 64].
    We have xi_0 = n^(-1/(0.5-H)) and xi_n = n^(1/H).
    10**6 samples are used for the MC estimates.
    """
    plt.plot(k_vec, rBergomi_inf_, "k", label="N=infinity")
    plt.plot(k_vec, rBergomi_inf_upper, "k--", label="confidence interval")
    plt.plot(k_vec, rBergomi_inf_lower, "k--")
    plt.plot(k_vec, rBergomi_1_, label="N=1")
    plt.plot(k_vec, rBergomi_2_, label="N=2")
    plt.plot(k_vec, rBergomi_3_, label="N=3")
    plt.plot(k_vec, rBergomi_4_, label="N=4")
    plt.plot(k_vec, rBergomi_6_, label="N=6")
    plt.plot(k_vec, rBergomi_44_, label="N=44")
    plt.legend(loc='upper right')
    plt.xlabel('Log-strike k')
    plt.ylabel('Implied volatility')
    plt.show()
