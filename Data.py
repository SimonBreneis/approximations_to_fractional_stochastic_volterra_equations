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
fBm_errors_Harms_10 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.836713, 0.632356, 0.521569,
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

    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes N")
    plt.ylabel("Error")
    plt.show()

    plt.loglog(fBm_n[0], fBm_errors_thm, 'b-', label="Theorem")
    plt.loglog(fBm_n[0], fBm_errors_bound, 'b--', label="Bound")
    plt.loglog(fBm_n[0], fBm_errors_opt_1, 'r-', label=r"Optimal $\xi$, same m")
    plt.loglog(fBm_n[0], fBm_errors_opt_2, 'g-', label=r"Optimal $\xi$ and m")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes N")
    plt.ylabel("Error")
    plt.show()

    plt.loglog(fBm_n[0], fBm_errors_thm, 'b-', label="Theorem")
    plt.loglog(fBm_n[0], fBm_errors_bound, 'b--', label="Theorem bound")
    plt.loglog(fBm_n[0], fBm_errors_reg, 'r-', label="Estimates")
    plt.loglog(fBm_n[0], fBm_errors_reg_bound, 'r--', label="Estimates bound")
    plt.loglog(fBm_n[0], fBm_errors_opt_2, 'g-', label=r"Optimal $\xi$ and m")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes")
    plt.ylabel("Error")
    plt.show()

    plt.loglog(fBm_n[0], fBm_errors_AK, 'y-', label="Alfonsi, Kebaier")
    plt.loglog(fBm_n[0], fBm_errors_Harms_1, 'c-', label="Harms, m=1")
    plt.loglog(fBm_n[0], fBm_errors_Harms_10, 'm-', label="Harms, m=10")
    plt.loglog(fBm_n[0], fBm_errors_thm, 'b-', label="Theorem")
    plt.loglog(fBm_n[0], fBm_errors_reg, 'r-', label="Estimates")
    plt.loglog(fBm_n[0], fBm_errors_opt_2, 'g-', label=r"Optimal $\xi$ and m")
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
The rBergomi implied volatility smiles for European call options. Parameters used are
H=0.07, T=0.9, number_time_steps=2000, eta=1.9, V_0=0.235**2, S_0=1, rho=-0.9.
The vector of log-strikes is given below (k_vec).
generate_samples is the Bergomi smile for the BFG approximation (with log-strikes k_vec).
rBergomi_N is the Bergomi smile for our approximation (with log-strikes k_vec) where N points were used and 
m and xi were chosen according to the interpolation of the numerical results.
rBergomi_AK_16 is the approximation by Alfonsi and Kebaier with 16 points.
rBergomi_Harms_m_16 is the approximation by Harms with 16 points and where Gaussian quadrature of level m was used.
10**6 samples are used for the MC estimates.
'''

k_vec = np.array([i / 100. for i in range(-40, 21)])

rBergomi_BFG = np.array([0.30789030, 0.30542406, 0.30294705, 0.30045787, 0.29795548, 0.29543931,
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

rBergomi_BFG_lower = np.array([0.30506697, 0.30271901, 0.30035556, 0.29797540, 0.29557765, 0.2931619,
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

rBergomi_BFG_upper = np.array([0.31066643, 0.30808688, 0.30550079, 0.30290665, 0.30030328, 0.29769000,
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

rBergomi_1 = np.array([0.30029087, 0.29823023, 0.29615892, 0.29407957, 0.29198906, 0.28988828,
                       0.28777337, 0.28564312, 0.28350009, 0.28134005, 0.27916971, 0.27698854,
                       0.27479604, 0.27259011, 0.27037405, 0.26814552, 0.26590413, 0.26365214,
                       0.26139193, 0.25911849, 0.25683100, 0.25453357, 0.25222308, 0.24990075,
                       0.24756802, 0.24522826, 0.24287478, 0.24051250, 0.23813347, 0.23574089,
                       0.23334019, 0.23093015, 0.22851173, 0.22607929, 0.22363398, 0.22117538,
                       0.21870609, 0.21622689, 0.21374166, 0.21124813, 0.20874925, 0.20624422,
                       0.20372720, 0.20119832, 0.19865318, 0.19609618, 0.19352853, 0.19095218,
                       0.18837172, 0.18578618, 0.18319221, 0.18059428, 0.17799406, 0.17539352,
                       0.17279627, 0.17020166, 0.16761972, 0.16505155, 0.16250237, 0.15997672,
                       0.15747286])

rBergomi_2 = np.array([0.30336348, 0.30114712, 0.29892670, 0.29669841, 0.29446149, 0.29222059,
                       0.28997348, 0.28772251, 0.28546619, 0.28320453, 0.28093321, 0.27864860,
                       0.27635349, 0.27405119, 0.27174288, 0.26942439, 0.26710369, 0.26477759,
                       0.26244742, 0.26011118, 0.25776850, 0.25541727, 0.25305513, 0.25068590,
                       0.24830777, 0.24592034, 0.24352516, 0.24111766, 0.23870322, 0.23627801,
                       0.23384661, 0.23140599, 0.22895604, 0.22649882, 0.22402996, 0.22155169,
                       0.21906438, 0.21657007, 0.21406848, 0.21156508, 0.20905436, 0.20653346,
                       0.20400482, 0.20147182, 0.19893868, 0.19640436, 0.19386820, 0.19132690,
                       0.18878642, 0.18624839, 0.18371005, 0.18117391, 0.17864437, 0.17611694,
                       0.17360391, 0.17110750, 0.16863855, 0.16619983, 0.16379398, 0.16143676,
                       0.15913222])

rBergomi_3 = np.array([0.30659317, 0.30427233, 0.30194477, 0.29961108, 0.29726962, 0.29492234,
                       0.29257383, 0.29021881, 0.28786094, 0.28549788, 0.28312669, 0.28075116,
                       0.27836941, 0.27598024, 0.27358541, 0.27118356, 0.26877384, 0.26635089,
                       0.26391993, 0.26147667, 0.25902451, 0.25656442, 0.25409658, 0.25162243,
                       0.24914244, 0.24665368, 0.24415554, 0.24165162, 0.23914153, 0.23662275,
                       0.23409549, 0.23156084, 0.22901735, 0.22646439, 0.22390092, 0.22132803,
                       0.21874221, 0.21614590, 0.21354248, 0.21092958, 0.20831056, 0.20568527,
                       0.20305149, 0.20040852, 0.19775621, 0.19510378, 0.19244991, 0.18979090,
                       0.18713393, 0.18448248, 0.18183823, 0.17919740, 0.17656195, 0.17393707,
                       0.17133854, 0.16876195, 0.16621644, 0.16370445, 0.16123897, 0.15881141,
                       0.15643757])

rBergomi_4 = np.array([0.30467607, 0.30240884, 0.30013322, 0.29784767, 0.29555193, 0.29324793,
                       0.29093394, 0.28860606, 0.28626898, 0.28392112, 0.28156199, 0.27919271,
                       0.27681121, 0.27442187, 0.27202107, 0.26960405, 0.26717755, 0.26474423,
                       0.26230290, 0.25984894, 0.25738254, 0.25490000, 0.25240415, 0.24989289,
                       0.24736750, 0.24482870, 0.24227992, 0.23972134, 0.23714942, 0.23456314,
                       0.23196408, 0.22934942, 0.22671939, 0.22407626, 0.22142090, 0.21875533,
                       0.21607873, 0.21339172, 0.21069077, 0.20797919, 0.20525509, 0.20251492,
                       0.19975768, 0.19698447, 0.19420233, 0.19141707, 0.18862808, 0.18583565,
                       0.18303803, 0.18025024, 0.17747005, 0.17468617, 0.17191568, 0.16916777,
                       0.16644828, 0.16376887, 0.16113145, 0.15854968, 0.15602887, 0.15356288,
                       0.15118113])

rBergomi_6 = np.array([0.30900184, 0.30657181, 0.30412448, 0.30166430, 0.29919430, 0.29671582,
                       0.29422763, 0.29172822, 0.28921919, 0.28669891, 0.28416706, 0.28162372,
                       0.27906873, 0.27650012, 0.27391407, 0.27131724, 0.26870729, 0.26608582,
                       0.26344795, 0.26079425, 0.25812799, 0.25545296, 0.25276559, 0.25006069,
                       0.24734181, 0.24460563, 0.24185036, 0.23907902, 0.23628967, 0.23348524,
                       0.23066053, 0.22782142, 0.22496641, 0.22209502, 0.21920342, 0.21629616,
                       0.21337416, 0.21043658, 0.20748489, 0.20451962, 0.20153593, 0.19853630,
                       0.19552015, 0.19249268, 0.18946201, 0.18642738, 0.18338661, 0.18034136,
                       0.17730652, 0.17429009, 0.17129156, 0.16832357, 0.16538976, 0.16249959,
                       0.15967520, 0.15692729, 0.15425283, 0.15166490, 0.14918838, 0.14682617,
                       0.14459046])

rBergomi_8 = np.array([0.30985888, 0.30734841, 0.30482649, 0.30229171, 0.29974934, 0.29719803,
                       0.29463667, 0.29206537, 0.28948562, 0.28689444, 0.28429420, 0.28168375,
                       0.27906316, 0.27642839, 0.27377832, 0.27111583, 0.26844415, 0.26576110,
                       0.26306296, 0.26035052, 0.25762156, 0.25487627, 0.25211247, 0.24933856,
                       0.24655049, 0.24374600, 0.24092550, 0.23808586, 0.23523583, 0.23236818,
                       0.22948353, 0.22658307, 0.22366843, 0.22074188, 0.21780179, 0.21484890,
                       0.21188182, 0.20890152, 0.20590925, 0.20290207, 0.19988360, 0.19685930,
                       0.19382358, 0.19077587, 0.18772476, 0.18467325, 0.18162555, 0.17858686,
                       0.17556515, 0.17256530, 0.16959200, 0.16665632, 0.16376789, 0.16093749,
                       0.15818264, 0.15551711, 0.15295290, 0.15049779, 0.14818330, 0.14600920,
                       0.14398048])

rBergomi_11 = np.array([0.30698304, 0.30460176, 0.30220394, 0.29978920, 0.29736270, 0.29492094,
                        0.29246579, 0.28999694, 0.28751277, 0.28501514, 0.28250906, 0.27999079,
                        0.27746021, 0.27491469, 0.27235583, 0.26978570, 0.26720271, 0.26460317,
                        0.26198918, 0.25935968, 0.25671361, 0.25405379, 0.25137741, 0.24868627,
                        0.24598109, 0.24325778, 0.24051670, 0.23775773, 0.23497884, 0.23218234,
                        0.22936950, 0.22653783, 0.22369086, 0.22082945, 0.21795176, 0.21505834,
                        0.21215273, 0.20923488, 0.20630487, 0.20336422, 0.20041299, 0.19745553,
                        0.19449169, 0.19152443, 0.18855927, 0.18559958, 0.18265182, 0.17971075,
                        0.17679089, 0.17390346, 0.17104794, 0.16823639, 0.16548314, 0.16280207,
                        0.16019405, 0.15767335, 0.15525269, 0.15294624, 0.15076674, 0.14871932,
                        0.14680000])

rBergomi_16 = np.array([0.30769951, 0.30526242, 0.30281439, 0.30035945, 0.29789505, 0.29541776,
                        0.29292956, 0.29043329, 0.28792606, 0.28540466, 0.28286814, 0.28031903,
                        0.27775186, 0.27517199, 0.27258349, 0.26998194, 0.26736521, 0.26473509,
                        0.26208981, 0.25943436, 0.25676416, 0.25407617, 0.25137365, 0.24865440,
                        0.24591622, 0.24316573, 0.24039962, 0.23762124, 0.23482580, 0.23201589,
                        0.22919088, 0.22635244, 0.22350017, 0.22063245, 0.21775191, 0.21485654,
                        0.21195043, 0.20903572, 0.20611709, 0.20318873, 0.20025341, 0.19731071,
                        0.19436364, 0.19141692, 0.18847490, 0.18554083, 0.18261774, 0.17971141,
                        0.17682516, 0.17396845, 0.17115106, 0.16838528, 0.16567412, 0.16303243,
                        0.16047917, 0.15802543, 0.15568628, 0.15345781, 0.15135192, 0.14938261,
                        0.14756893])

rBergomi_23 = np.array([0.31123212, 0.30863029, 0.30602151, 0.30341065, 0.30079544, 0.29817026,
                        0.29554260, 0.29290695, 0.29026145, 0.28761038, 0.28495256, 0.28228916,
                        0.27961815, 0.27693890, 0.27424463, 0.27153757, 0.26882093, 0.26609349,
                        0.26335307, 0.26060270, 0.25784184, 0.25507069, 0.25228724, 0.24949250,
                        0.24668497, 0.24386381, 0.24103156, 0.23818644, 0.23532930, 0.23246028,
                        0.22958594, 0.22670062, 0.22380269, 0.22089517, 0.21797434, 0.21503885,
                        0.21209416, 0.20913988, 0.20617608, 0.20320507, 0.20023086, 0.19725564,
                        0.19428040, 0.19131238, 0.18834631, 0.18538729, 0.18244222, 0.17951199,
                        0.17660699, 0.17373703, 0.17091068, 0.16814732, 0.16545448, 0.16284967,
                        0.16034383, 0.15794975, 0.15567471, 0.15353706, 0.15154677, 0.14969188,
                        0.14797153])

rBergomi_32 = np.array([0.31085632, 0.30826034, 0.30565785, 0.30304654, 0.30042752, 0.29780207,
                        0.29516897, 0.29252719, 0.28987966, 0.28722946, 0.28456834, 0.28189712,
                        0.27921444, 0.27652304, 0.27382239, 0.27111073, 0.26838967, 0.26565821,
                        0.26291595, 0.26016468, 0.25740432, 0.25463578, 0.25185761, 0.24906957,
                        0.24627007, 0.24345932, 0.24063779, 0.23780214, 0.23495549, 0.23209637,
                        0.22922390, 0.22634073, 0.22344686, 0.22054466, 0.21762813, 0.21470091,
                        0.21176067, 0.20881175, 0.20585614, 0.20289666, 0.19993661, 0.19697134,
                        0.19400464, 0.19104262, 0.18808415, 0.18513978, 0.18221537, 0.17932058,
                        0.17645920, 0.17363927, 0.17086558, 0.16814933, 0.16551384, 0.16295966,
                        0.16050932, 0.15817242, 0.15596567, 0.15388615, 0.15194049, 0.15013040,
                        0.14846623])

rBergomi_Harms_1_16 = np.array([0.27482681, 0.27371927, 0.27260334, 0.27148488, 0.27036254, 0.26923712,
                                0.26810920, 0.26698023, 0.26584901, 0.26471447, 0.26357678, 0.26243125,
                                0.26128312, 0.26013407, 0.25898033, 0.25782198, 0.25666155, 0.25550070,
                                0.25433719, 0.25316389, 0.25198389, 0.25080388, 0.24961699, 0.24842772,
                                0.24723793, 0.24604527, 0.24484593, 0.24364305, 0.24243775, 0.24123573,
                                0.24002922, 0.23882036, 0.23761102, 0.23639835, 0.23517712, 0.23394897,
                                0.23271878, 0.23148399, 0.23024606, 0.22900690, 0.22776841, 0.22652867,
                                0.22528862, 0.22404275, 0.22279201, 0.22153981, 0.22028367, 0.21902055,
                                0.21775639, 0.21649168, 0.21522182, 0.21395324, 0.21267826, 0.21140199,
                                0.21012338, 0.20884673, 0.20756255, 0.20627449, 0.20498173, 0.20368292,
                                0.20238826])

rBergomi_AK_16 = np.array([0.30371212, 0.30157470, 0.29943079, 0.29728268, 0.29512383, 0.29295907,
                           0.29078264, 0.28860054, 0.28641279, 0.28421475, 0.28200946, 0.27979287,
                           0.27756660, 0.27533003, 0.27308287, 0.27082697, 0.26856091, 0.26628675,
                           0.26400139, 0.26170373, 0.25939205, 0.25707575, 0.25475304, 0.25241958,
                           0.25007410, 0.24771901, 0.24534799, 0.24296312, 0.24056070, 0.23814228,
                           0.23571047, 0.23326918, 0.23081991, 0.22836033, 0.22588443, 0.22339326,
                           0.22089004, 0.21837011, 0.21583698, 0.21328833, 0.21072406, 0.20815102,
                           0.20556544, 0.20297061, 0.20036338, 0.19774576, 0.19511536, 0.19247204,
                           0.18981487, 0.18714616, 0.18446458, 0.18177025, 0.17906627, 0.17634464,
                           0.17361743, 0.17088033, 0.16814380, 0.16540719, 0.16266981, 0.15993453,
                           0.15722911])

rBergomi_Harms_8_16 = np.array([0.29133591, 0.28970038, 0.28805877, 0.28641074, 0.28475876, 0.28309352,
                                0.28142104, 0.27974161, 0.27805547, 0.27636271, 0.27466345, 0.27295576,
                                0.27123737, 0.26951191, 0.26778095, 0.26603849, 0.26428380, 0.26252040,
                                0.26075058, 0.25897655, 0.25720139, 0.25541391, 0.25361976, 0.25181598,
                                0.25000329, 0.24818233, 0.24635593, 0.24451796, 0.24266937, 0.24080880,
                                0.23893800, 0.23705793, 0.23516849, 0.23326658, 0.23135528, 0.22943290,
                                0.22749953, 0.22555657, 0.22360830, 0.22165055, 0.21968262, 0.21770245,
                                0.21571000, 0.21370584, 0.21168755, 0.20965704, 0.20761593, 0.20556579,
                                0.20350155, 0.20142772, 0.19934538, 0.19724831, 0.19513755, 0.19301577,
                                0.19088036, 0.18873390, 0.18657440, 0.18440176, 0.18221648, 0.18002737,
                                0.17782914])


def plot_rBergomi_smiles():
    """
    Plots the rBergomi implied volatility smiles for European call options. Parameters used are
    H=0.07, T=0.9, number_time_steps=2000, eta=1.9, V_0=0.235**2, S_0=1, rho=-0.9.
    Plots both the BFG approximation of the smile and the approximations for N in [2, 4, 8, 16] (using the interpolation
    of the optimized values for xi and m). Also plots a 95% confidence interval for the BFG approximation.
    10**6 samples are used for the MC estimates.
    Afterwards, plots the BFG approximation with confidence intervals, our approximation, the Alfonsi-Kebaier
    approximation, and the Harms approximation with m=1 and m=8 for N=16.
    """
    plt.plot(k_vec, rBergomi_BFG, "k", label="Non-Markovian approximation")
    plt.plot(k_vec, rBergomi_BFG_upper, "k--", label="confidence interval")
    plt.plot(k_vec, rBergomi_BFG_lower, "k--")
    plt.plot(k_vec, rBergomi_2, label="N=2")
    plt.plot(k_vec, rBergomi_4, label="N=4")
    plt.plot(k_vec, rBergomi_8, label="N=8")
    plt.plot(k_vec, rBergomi_16, label="N=16")
    plt.legend(loc='upper right')
    plt.xlabel('Log-moneyness')
    plt.ylabel('Implied volatility')
    plt.show()

    plt.plot(k_vec, rBergomi_BFG, 'k', label="Non-Markovian approximation")
    plt.plot(k_vec, rBergomi_BFG_upper, "k--", label="confidence interval")
    plt.plot(k_vec, rBergomi_BFG_lower, "k--")
    plt.plot(k_vec, rBergomi_16, label="Our approach")
    plt.plot(k_vec, rBergomi_AK_16, label="Alfonsi, Kebaier")
    plt.plot(k_vec, rBergomi_Harms_1_16, label="Harms, m=1")
    plt.plot(k_vec, rBergomi_Harms_8_16, label="Harms, m=8")
    plt.legend(loc='upper right')
    plt.xlabel('Log-moneyness')
    plt.ylabel('Implied volatility')
    plt.show()


'''
The rHeston implied volatility smiles for European call options. Parameters used are
specified in the paper.
The vector of log-strikes is given below (k_vec).
rHeston is the Heston smile for the El Euch-Rosenbaum approximation (with log-strikes k_vec).
rHeston_N is the Bergomi smile for our approximation (with log-strikes k_vec) where N points were used and 
m and xi were chosen according to the interpolation of the numerical results.
rHeston_AE_16 is the approximation by Abi Jaber and El Euch with 16 points.
'''

k_rHeston = -1.3 + 0.02 * np.arange(81)

rHeston = np.array([0.39071594, 0.38782894, 0.38492131, 0.38199287, 0.37904301, 0.37607124,
                    0.37307725, 0.37006036, 0.36702011, 0.36395604, 0.36086744, 0.35775385,
                    0.35461463, 0.35144910, 0.34825669, 0.34503666, 0.34178829, 0.33851088,
                    0.33520362, 0.33186570, 0.32849631, 0.32509451, 0.32165941, 0.31819003,
                    0.31468533, 0.31114427, 0.30756570, 0.30394843, 0.30029125, 0.29659281,
                    0.29285173, 0.28906657, 0.28523576, 0.28135767, 0.27743056, 0.27345259,
                    0.26942181, 0.26533613, 0.26119333, 0.25699105, 0.25272676, 0.24839775,
                    0.24400112, 0.23953376, 0.23499232, 0.23037320, 0.22567251, 0.22088607,
                    0.21600932, 0.21103738, 0.20596492, 0.20078619, 0.19549495, 0.19008450,
                    0.18454760, 0.17887653, 0.17306318, 0.16709918, 0.16097627, 0.15468695,
                    0.14822569, 0.14159122, 0.13479062, 0.12784715, 0.12081460, 0.11380320,
                    0.10702149, 0.10082400, 0.09570101, 0.09209830, 0.09013840, 0.08958059,
                    0.09004443, 0.09119446, 0.09279002, 0.09467101, 0.09673306, 0.09890787,
                    0.10115026, 0.10342995, 0.10572636])

rHeston_1 = np.array([0.35472608, 0.35237628, 0.35003508, 0.34769471, 0.34529301, 0.34290571,
                      0.34049250, 0.33803998, 0.33559298, 0.33310895, 0.33060055, 0.32808436,
                      0.32552987, 0.32295746, 0.32036541, 0.31773793, 0.31509238, 0.31241884,
                      0.30971294, 0.30698498, 0.30422392, 0.30143175, 0.29861192, 0.29575588,
                      0.29286760, 0.28994581, 0.28698525, 0.28398920, 0.28095393, 0.27787685,
                      0.27475927, 0.27159679, 0.26838818, 0.26513270, 0.26182607, 0.25846729,
                      0.25505384, 0.25158177, 0.24804938, 0.24445277, 0.24078798, 0.23705199,
                      0.23323969, 0.22934647, 0.22536751, 0.22129628, 0.21712666, 0.21285142,
                      0.20846193, 0.20394939, 0.19930317, 0.19451120, 0.18956031, 0.18443474,
                      0.17911683, 0.17358655, 0.16782048, 0.16179255, 0.15547334, 0.14883021,
                      0.14182917, 0.13443712, 0.12662912, 0.11840573, 0.10983299, 0.10114255,
                      0.09295095, 0.08648798, 0.08296822, 0.08222430, 0.08320419, 0.08507607,
                      0.08738190, 0.08988635, 0.09246752, 0.09506271, 0.09763803, 0.10017555,
                      0.10266820, 0.10511094, 0.10750278])

rHeston_2 = np.array([0.36000595, 0.35755894, 0.35511776, 0.35266120, 0.35016264, 0.34766782,
                      0.34514341, 0.34259084, 0.34003288, 0.33744067, 0.33482813, 0.33220001,
                      0.32953825, 0.32685814, 0.32415423, 0.32141885, 0.31866303, 0.31587782,
                      0.31306262, 0.31022271, 0.30734990, 0.30444681, 0.30151403, 0.29854581,
                      0.29554520, 0.29250984, 0.28943651, 0.28632706, 0.28317791, 0.27998750,
                      0.27675597, 0.27347957, 0.27015740, 0.26678798, 0.26336783, 0.25989584,
                      0.25636926, 0.25278481, 0.24914055, 0.24543274, 0.24165793, 0.23781295,
                      0.23389314, 0.22989428, 0.22581169, 0.22163946, 0.21737198, 0.21300250,
                      0.20852332, 0.20392642, 0.19920219, 0.19433996, 0.18932794, 0.18415219,
                      0.17879717, 0.17324518, 0.16747560, 0.16146539, 0.15518829, 0.14861499,
                      0.14171445, 0.13445579, 0.12681486, 0.11878999, 0.11043868, 0.10196844,
                      0.09393114, 0.08743349, 0.08364250, 0.08254447, 0.08321322, 0.08484001,
                      0.08695739, 0.08931813, 0.09179123, 0.09430659, 0.09682463, 0.09932333,
                      0.10179161, 0.10422164, 0.10661067])

rHeston_4 = np.array([0.37828606, 0.37552890, 0.37275388, 0.36995541, 0.36713793, 0.36429937,
                      0.36143719, 0.35855456, 0.35564831, 0.35271803, 0.34976510, 0.34678654,
                      0.34378295, 0.34075418, 0.33769799, 0.33461512, 0.33150441, 0.32836442,
                      0.32519548, 0.32199595, 0.31876498, 0.31550224, 0.31220604, 0.30887572,
                      0.30551035, 0.30210833, 0.29866890, 0.29519067, 0.29167210, 0.28811213,
                      0.28450902, 0.28086118, 0.27716714, 0.27342486, 0.26963255, 0.26578823,
                      0.26188957, 0.25793437, 0.25392011, 0.24984401, 0.24570332, 0.24149484,
                      0.23721518, 0.23286078, 0.22842758, 0.22391131, 0.21930730, 0.21461032,
                      0.20981479, 0.20491450, 0.19990256, 0.19477147, 0.18951283, 0.18411736,
                      0.17857480, 0.17287368, 0.16700140, 0.16094410, 0.15468678, 0.14821373,
                      0.14150951, 0.13456116, 0.12736327, 0.11992870, 0.11231228, 0.10466161,
                      0.09731420, 0.09091269, 0.08628859, 0.08386015, 0.08329830, 0.08398718,
                      0.08543259, 0.08732180, 0.08946881, 0.09176286, 0.09413671, 0.09654897,
                      0.09897323, 0.10139266, 0.10379681])

rHeston_8 = np.array([0.38493220, 0.38210898, 0.37926473, 0.37639912, 0.37351262, 0.37060355,
                      0.36767204, 0.36471779, 0.36173949, 0.35873729, 0.35571038, 0.35265784,
                      0.34957959, 0.34647456, 0.34334208, 0.34018173, 0.33699239, 0.33377347,
                      0.33052422, 0.32724354, 0.32393076, 0.32058486, 0.31720474, 0.31378957,
                      0.31033810, 0.30684922, 0.30332180, 0.29975445, 0.29614590, 0.29249476,
                      0.28879945, 0.28505845, 0.28127009, 0.27743253, 0.27354395, 0.26960231,
                      0.26560545, 0.26155114, 0.25743690, 0.25326014, 0.24901807, 0.24470768,
                      0.24032577, 0.23586886, 0.23133321, 0.22671478, 0.22200920, 0.21721170,
                      0.21231716, 0.20731995, 0.20221396, 0.19699253, 0.19164838, 0.18617359,
                      0.18055952, 0.17479681, 0.16887542, 0.16278471, 0.15651373, 0.15005186,
                      0.14338999, 0.13652297, 0.12945445, 0.12220640, 0.11483834, 0.10748467,
                      0.10041987, 0.09413337, 0.08928184, 0.08633305, 0.08520746, 0.08544609,
                      0.08657580, 0.08825589, 0.09026902, 0.09248060, 0.09480709, 0.09719585,
                      0.09961309, 0.10203701, 0.10445331])

rHeston_16 = np.array([0.38735198, 0.38448981, 0.38160661, 0.37870235, 0.37577689, 0.37282904,
                       0.36985888, 0.36686579, 0.36384891, 0.36080816, 0.35774268, 0.35465182,
                       0.35153525, 0.34839201, 0.34522156, 0.34202330, 0.33879627, 0.33553987,
                       0.33225329, 0.32893557, 0.32558600, 0.32220356, 0.31878728, 0.31533624,
                       0.31184929, 0.30832535, 0.30476330, 0.30116182, 0.29751967, 0.29383549,
                       0.29010778, 0.28633507, 0.28251571, 0.27864797, 0.27473008, 0.27076006,
                       0.26673587, 0.26265532, 0.25851605, 0.25431557, 0.25005121, 0.24572007,
                       0.24131909, 0.23684493, 0.23229402, 0.22766248, 0.22294614, 0.21814046,
                       0.21324051, 0.20824095, 0.20313596, 0.19791918, 0.19258369, 0.18712197,
                       0.18152583, 0.17578642, 0.16989428, 0.16383943, 0.15761168, 0.15120127,
                       0.14460002, 0.13780378, 0.13081698, 0.12366163, 0.11639505, 0.10914349,
                       0.10215982, 0.09588968, 0.09093856, 0.08777573, 0.08639141, 0.08639036,
                       0.08732176, 0.08884368, 0.09073124, 0.09284252, 0.09508820, 0.09741128,
                       0.09977482, 0.10215464, 0.10453459])

rHeston_32 = np.array([0.38932190, 0.38643611, 0.38352932, 0.38060151, 0.37765226, 0.37468071,
                       0.37168679, 0.36866980, 0.36562910, 0.36256442, 0.35947495, 0.35636014,
                       0.35321953, 0.35005224, 0.34685776, 0.34363539, 0.34038428, 0.33710382,
                       0.33379313, 0.33045137, 0.32707775, 0.32367128, 0.32023104, 0.31675607,
                       0.31324525, 0.30969754, 0.30611178, 0.30248672, 0.29882111, 0.29511360,
                       0.29136274, 0.28756707, 0.28372495, 0.27983471, 0.27589457, 0.27190262,
                       0.26785685, 0.26375510, 0.25959508, 0.25537433, 0.25109025, 0.24674002,
                       0.24232063, 0.23782884, 0.23326117, 0.22861386, 0.22388285, 0.21906375,
                       0.21415179, 0.20914182, 0.20402821, 0.19880488, 0.19346519, 0.18800195,
                       0.18240739, 0.17667313, 0.17079027, 0.16474952, 0.15854150, 0.15215740,
                       0.14559019, 0.13883698, 0.13190346, 0.12481247, 0.11762056, 0.11044872,
                       0.10353407, 0.09728896, 0.09227728, 0.08896328, 0.08738801, 0.08720900,
                       0.08799605, 0.08940687, 0.09121006, 0.09325719, 0.09545375, 0.09773897,
                       0.10007322, 0.10243029, 0.10479262])

rHeston_AE_16 = np.array([0.36348225, 0.36102425, 0.35854778, 0.35605248, 0.35353778, 0.35100336,
                          0.34844872, 0.34587331, 0.34327674, 0.34065842, 0.33801779, 0.33535436,
                          0.33266747, 0.32995653, 0.32722091, 0.32445992, 0.32167288, 0.31885905,
                          0.31601765, 0.31314790, 0.31024896, 0.30731992, 0.30435990, 0.30136788,
                          0.29834287, 0.29528378, 0.29218948, 0.28905877, 0.28589039, 0.28268300,
                          0.27943520, 0.27614548, 0.27281227, 0.26943387, 0.26600851, 0.26253427,
                          0.25900914, 0.25543094, 0.25179739, 0.24810600, 0.24435415, 0.24053899,
                          0.23665751, 0.23270643, 0.22868226, 0.22458123, 0.22039927, 0.21613201,
                          0.21177472, 0.20732232, 0.20276931, 0.19810980, 0.19333743, 0.18844540,
                          0.18342650, 0.17827310, 0.17297729, 0.16753111, 0.16192685, 0.15615779,
                          0.15021932, 0.14411105, 0.13784041, 0.13142910, 0.12492446, 0.11841864,
                          0.11207813, 0.10617675, 0.10109605, 0.09722380, 0.09475759, 0.09360455,
                          0.09348975, 0.09411704, 0.09524849, 0.09671462, 0.09840009, 0.10022733,
                          0.10214408, 0.10411482, 0.10611520])


def plot_rHeston_smiles():
    """
    Plots the rHeston implied volatility smiles for European call options. Parameters used are specified in the paper.
    Plots both the ER approximation of the smile and the approximations for N in [1, 4, 16] (using the interpolation
    of the optimized values for xi and m).
    Afterwards, plots the ER approximation, our approximation, and the Abi Jaber-El Euch approximation for N=16.
    """
    plt.plot(k_rHeston, rHeston, 'k', label="Non-Markovian approximation")
    plt.plot(k_rHeston, rHeston_1, label="N=1")
    # plt.plot(k_rHeston, rHeston_AK_2, label="N=2")
    plt.plot(k_rHeston, rHeston_4, label="N=4")
    # plt.plot(k_rHeston, rHeston_AK_8, label="N=8")
    plt.plot(k_rHeston, rHeston_16, label="N=16")
    # plt.plot(k_rHeston, rHeston_AK_32, label="N=32")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("implied volatility")
    plt.show()

    plt.plot(k_rHeston, rHeston, 'k', label="Non-Markovian approximation")
    plt.plot(k_rHeston, rHeston_16, label="Our approximation")
    plt.plot(k_rHeston, rHeston_AE_16, label="Abi Jaber, El Euch")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("implied volatility")
    plt.show()
