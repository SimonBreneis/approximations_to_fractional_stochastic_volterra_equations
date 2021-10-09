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
    return np.sqrt(1 / H + 1 / (1.5 - H))


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
              [0.635407, 0.451472, 0.333850, 0.256923, 0.204204, 0.119592, 0.076692, 0.039571, 0.022815, 0.010167,
               0.004749,
               0.002037, 0.000866, 0.000342, 0.000132, 4.91e-05, 1.78e-05, 6.26e-06],
              [0.604933, 0.405954, 0.285941, 0.210481, 0.160789, 0.087336, 0.049924, 0.025489, 0.012368, 0.004540,
               0.001559,
               0.000543, 0.000158, 4.34e-05, 1.14e-05, 2.86e-06, 6.71e-07],
              [0.582543, 0.374196, 0.253831, 0.180499, 0.103376, 0.069539, 0.035804, 0.013985, 0.005244, 0.001812,
               0.000526,
               0.000123, 2.74e-05, 5.37e-06, 9.46e-07, 1.57e-07],
              [0.564594, 0.350288, 0.230415, 0.115370, 0.087745, 0.044988, 0.017360, 0.006881, 0.002035, 0.000596,
               0.000130,
               2.47e-05, 3.79e-06, 4.99e-07, 6.03e-08],
              [0.549739, 0.331373, 0.212379, 0.143411, 0.101948, 0.048047, 0.021629, 0.008499, 0.003022, 0.000708,
               0.000140,
               2.59e-05, 3.50e-06, 3.66e-07, 3.42e-08],
              [0.537137, 0.315874, 0.197939, 0.091741, 0.068621, 0.031377, 0.010481, 0.003575, 0.000841, 0.000167,
               2.69e-05,
               3.64e-06, 3.76e-07, 2.47e-08],
              [0.526234, 0.302840, 0.186042, 0.120975, 0.062400, 0.035876, 0.014809, 0.004062, 0.001044, 0.000225,
               3.50e-05,
               3.82e-06, 3.47e-07, 2.09e-08],
              [0.516652, 0.291658, 0.176020, 0.112708, 0.077254, 0.043063, 0.017208, 0.005369, 0.001490, 0.000283,
               4.10e-05,
               4.35e-06, 3.68e-07, 1.98e-08],
              [0.508122, 0.281912, 0.167430, 0.105750, 0.053486, 0.020847, 0.005956, 0.001722, 0.000307, 5.28e-05,
               5.47e-06,
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
        errors = np.array(fBm_errors[m - 1])
        plt.loglog(m * n + 1, errors, label=f"m={m}")

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
    plt.xlabel(r"$\log_{10}(N)$")
    plt.ylabel(r"Estimate for $\beta$")
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
    plt.xlabel(r"$\log_{10}(N)$")
    plt.ylabel(r"Estimate for $\alpha$")
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
    print(
        f"A good fit is achieved using xi_0 = {np.round(np.exp(C), 4)} * exp({np.round(gamma, 4)}*H) * exp(-{alpha}/((1.5-H)*A) * sqrt(N))")
    plt.plot(H, np.log(avg_factor_xi_0), label="True log-factor")
    plt.plot(H, C + gamma * H, label="regression")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which xi_0 differs from its optimum")
    plt.legend(loc="upper left")
    plt.show()

    avg_factor_xi_0 = avg_factor_xi_0 / np.exp(C + gamma * H)
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
    print(
        f"A good fit is achieved using xi_n = exp({np.round(C, 4)} * H^{np.round(gamma, 4)}) * exp(-{alpha}/(H*A) * sqrt(N))")
    plt.plot(H, np.log(avg_factor_xi_n), label="True log-factor")
    plt.plot(H, C * H ** gamma, label="regression")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which xi_n differs from its optimum")
    plt.legend(loc="upper left")
    plt.show()

    avg_factor_xi_n = avg_factor_xi_n / np.exp(C * H ** gamma)
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
    print(
        f"A good fit is achieved using error = exp({np.round(C, 4)} * H^{np.round(gamma, 4)}) * exp(-{alpha}/A * sqrt(N))")
    plt.plot(H, np.log(avg_factor_error), label="True log-factor")
    plt.plot(H, C * H ** gamma, label="approximation")
    plt.xlabel("H")
    plt.ylabel("Logarithm of average factor by which the error differs from its true value")
    plt.legend(loc="upper right")
    plt.show()

    avg_factor_error = avg_factor_error / np.exp(C * H ** gamma)
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
        return mp.matrix([mp.exp(-t * x_) for x_ in x])

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
        q_vec[i] = np.sqrt(new_error / previous_error * (2 * m) * (2 * m - 1) / (float(t * (b - a))) ** 2)
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
        C_vec[i] = t * error * mp.gamma(H + 0.5) * mp.gamma(0.5 - H) * math.factorial(2 * m) * (q * t * (b - a)) ** (
                -2 * m) * mp.exp(t * a) * a ** (0.5 + H)

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
    plt.plot(k_vec, rBergomi_16, 'r', label="Our approach")
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


k_rrHeston = -1.3 + 0.01 * np.arange(161)

rrHeston = np.array([0.4008664, 0.40133726, 0.40138185, 0.40074782, 0.39935361, 0.39726891,
                     0.39469064, 0.39190993, 0.38925826, 0.38702849, 0.38538926, 0.38433434,
                     0.38369854, 0.38322914, 0.38266999, 0.38182348, 0.38058122, 0.37893055,
                     0.37694544, 0.37476574, 0.37256558, 0.37051289, 0.36872756, 0.3672516,
                     0.36604309, 0.36499611, 0.3639765, 0.3628588, 0.36155391, 0.36002355,
                     0.35828236, 0.35638963, 0.35443306, 0.35250709, 0.35069011, 0.34902612,
                     0.3475161, 0.34612165, 0.34477863, 0.34341583, 0.34197263, 0.34041174,
                     0.33872497, 0.33693217, 0.33507423, 0.33320202, 0.33136371, 0.32959345,
                     0.32790411, 0.32628595, 0.3247111, 0.32314203, 0.32154153, 0.31988137,
                     0.31814793, 0.31634388, 0.31448607, 0.31260035, 0.31071478, 0.30885274,
                     0.30702771, 0.3052408, 0.30348156, 0.30173147, 0.29996912, 0.29817545,
                     0.29633806, 0.2944534, 0.29252681, 0.29057046, 0.28859987, 0.2866299,
                     0.28467128, 0.28272826, 0.28079813, 0.27887239, 0.27693922, 0.2749865,
                     0.27300472, 0.27098889, 0.26893929, 0.26686095, 0.264762, 0.26265143,
                     0.2605368, 0.25842253, 0.25630893, 0.25419246, 0.25206672, 0.24992417,
                     0.24775793, 0.24556326, 0.2433385, 0.24108509, 0.23880697, 0.23650934,
                     0.23419725, 0.23187429, 0.22954174, 0.22719828, 0.22484042, 0.22246333,
                     0.22006201, 0.21763236, 0.21517205, 0.21268081, 0.2101603, 0.20761349,
                     0.20504373, 0.20245386, 0.19984535, 0.197218, 0.19456989, 0.19189784,
                     0.18919819, 0.18646764, 0.18370397, 0.18090661, 0.17807668, 0.1752168,
                     0.1723305, 0.1694215, 0.16649305, 0.16354748, 0.16058611, 0.15760954,
                     0.15461837, 0.15161415, 0.14860043, 0.14558373, 0.14257421, 0.1395859,
                     0.13663661, 0.13374722, 0.13094086, 0.12824178, 0.12567429, 0.12326176,
                     0.12102584, 0.11898586, 0.11715833, 0.11555656, 0.11419012, 0.11306418,
                     0.11217866, 0.11152735, 0.11109726, 0.11086852, 0.11081531, 0.11090788,
                     0.11111577, 0.11141152, 0.11177435, 0.11219287, 0.1126661, 0.11320218,
                     0.11381464, 0.11451643, 0.11531294, 0.11619583, 0.11714026])

rrrHeston = np.array([0.40082795, 0.40130254, 0.40134906, 0.40071517, 0.39931958, 0.39723243,
                      0.39465122, 0.39186775, 0.3892142, 0.38698393, 0.3853457, 0.38429289,
                      0.38365965, 0.38319254, 0.38263498, 0.38178911, 0.38054658, 0.37889493,
                      0.37690842, 0.37472727, 0.37252599, 0.37047277, 0.36868766, 0.36721255,
                      0.36600529, 0.36495962, 0.36394109, 0.36282404, 0.36151931, 0.35998868,
                      0.3582469, 0.35635347, 0.35439628, 0.35246995, 0.35065296, 0.34898931,
                      0.34747991, 0.34608618, 0.34474385, 0.34338157, 0.34193865, 0.34037778,
                      0.33869083, 0.33689774, 0.3350395, 0.33316709, 0.33132875, 0.32955864,
                      0.32786962, 0.32625188, 0.32467746, 0.32310877, 0.32150854, 0.31984852,
                      0.31811509, 0.31631097, 0.31445305, 0.31256726, 0.31068168, 0.30881973,
                      0.30699489, 0.30520824, 0.30344928, 0.30169948, 0.29993735, 0.29814385,
                      0.29630654, 0.29442192, 0.29249533, 0.29053898, 0.28856842, 0.28659854,
                      0.28464005, 0.2826972, 0.28076728, 0.27884175, 0.27690876, 0.27495621,
                      0.27297455, 0.27095879, 0.26890924, 0.26683095, 0.26473206, 0.26262156,
                      0.26050705, 0.2583929, 0.25627946, 0.25416315, 0.25203757, 0.24989516,
                      0.24772903, 0.24553446, 0.24330977, 0.24105643, 0.23877837, 0.23648081,
                      0.23416881, 0.23184596, 0.22951352, 0.22717018, 0.22481244, 0.22243546,
                      0.22003423, 0.21760467, 0.21514442, 0.21265323, 0.21013277, 0.20758601,
                      0.20501631, 0.2024265, 0.19981806, 0.19719079, 0.19454274, 0.19187075,
                      0.18917116, 0.18644063, 0.18367698, 0.18087962, 0.17804968, 0.17518979,
                      0.17230347, 0.16939446, 0.16646599, 0.16352041, 0.16055902, 0.15758242,
                      0.15459121, 0.15158692, 0.14857313, 0.14555634, 0.14254673, 0.13955835,
                      0.13660899, 0.13371957, 0.13091323, 0.12821423, 0.12564688, 0.12323455,
                      0.1209989, 0.11895925, 0.11713213, 0.11553083, 0.11416494, 0.11303959,
                      0.11215471, 0.11150408, 0.11107469, 0.11084664, 0.11079408, 0.11088725,
                      0.11109565, 0.11139181, 0.11175497, 0.11217377, 0.11264723, 0.11318355,
                      0.11379629, 0.11449846, 0.11529541, 0.1161788, 0.11712369])

rrHeston_1 = np.array([0.36679211, 0.37151053, 0.37349597, 0.37300667, 0.37032276, 0.36581155,
                       0.36003932, 0.35392561, 0.34881425, 0.34607352, 0.34613224, 0.34811188,
                       0.35066277, 0.35275747, 0.35383477, 0.35367447, 0.35228275, 0.34983653,
                       0.34666507, 0.34323466, 0.34009206, 0.33772914, 0.33639885, 0.33601041,
                       0.33620251, 0.33652615, 0.33659838, 0.33617141, 0.33514212, 0.33353737,
                       0.33149203, 0.32922015, 0.32697315, 0.32498276, 0.32339958, 0.32225261,
                       0.32145111, 0.32082693, 0.32019298, 0.31939254, 0.3183283, 0.3169722,
                       0.31536127, 0.31358309, 0.31175343, 0.30998849, 0.30837712, 0.30696039,
                       0.3057252, 0.30461362, 0.30354326, 0.30243078, 0.30121143, 0.29985104,
                       0.29834961, 0.29673734, 0.29506486, 0.29338986, 0.29176301, 0.29021636,
                       0.28875734, 0.2873693, 0.28601827, 0.28466305, 0.28326577, 0.28180032,
                       0.28025685, 0.2786423, 0.27697698, 0.27528857, 0.27360491, 0.27194739,
                       0.27032638, 0.26873978, 0.26717468, 0.2656115, 0.26402922, 0.26241048,
                       0.26074513, 0.25903182, 0.25727743, 0.25549458, 0.25369811, 0.25190129,
                       0.25011271, 0.24833462, 0.24656279, 0.2447881, 0.24299894, 0.24118418,
                       0.23933557, 0.23744935, 0.23552668, 0.23357276, 0.23159523, 0.22960199,
                       0.2275992, 0.22558976, 0.22357276, 0.22154379, 0.2194961, 0.21742222,
                       0.21531559, 0.2131719, 0.2109898, 0.20877079, 0.20651855, 0.20423768,
                       0.20193233, 0.19960503, 0.19725593, 0.19488264, 0.1924807, 0.19004452,
                       0.18756857, 0.18504852, 0.18248203, 0.17986913, 0.17721198, 0.17451422,
                       0.17178001, 0.16901299, 0.16621548, 0.16338817, 0.16053031, 0.15764051,
                       0.15471793, 0.15176381, 0.14878291, 0.14578461, 0.14278366, 0.13980009,
                       0.13685872, 0.13398793, 0.13121828, 0.12858084, 0.12610577, 0.12382093,
                       0.12175093, 0.11991626, 0.11833263, 0.11701027, 0.11595315, 0.1151582,
                       0.11461458, 0.1143035, 0.11419869, 0.11426809, 0.11447666, 0.11479004,
                       0.11517853, 0.11562054, 0.11610477, 0.11663058, 0.11720621, 0.11784479,
                       0.11855868, 0.11935326, 0.12022168, 0.12114257, 0.12208199])

rrHeston_2 = np.array([0.37123133, 0.37515465, 0.37692873, 0.37659932, 0.37435366, 0.37051441,
                       0.36559576, 0.36037974, 0.35590733, 0.35318187, 0.35259186, 0.35364273,
                       0.35537571, 0.35692277, 0.35772658, 0.35752065, 0.35625898, 0.35406703,
                       0.35121555, 0.34809564, 0.34516394, 0.34283504, 0.34134307, 0.34065554,
                       0.34050997, 0.34054253, 0.34041614, 0.33989375, 0.33886081, 0.33732028,
                       0.33537559, 0.33320494, 0.33102424, 0.32903822, 0.3273882, 0.32611548,
                       0.32515757, 0.32437853, 0.3236165, 0.3227281, 0.32161756, 0.32024938,
                       0.31864724, 0.31688262, 0.31505562, 0.31327059, 0.31161108, 0.31012063,
                       0.30879509, 0.30758862, 0.30643025, 0.30524446, 0.30396955, 0.30256984,
                       0.30104034, 0.29940435, 0.29770544, 0.29599561, 0.29432247, 0.2927182,
                       0.29119315, 0.2897354, 0.288316, 0.28689776, 0.285445, 0.28393168,
                       0.28234626, 0.28069279, 0.27898834, 0.27725779, 0.27552723, 0.27381766,
                       0.27214044, 0.27049535, 0.26887178, 0.26725217, 0.26561686, 0.26394887,
                       0.26223753, 0.2604803, 0.25868253, 0.25685543, 0.25501282, 0.25316758,
                       0.25132859, 0.2494989, 0.2476754, 0.24585004, 0.24401207, 0.24215073,
                       0.24025769, 0.23832873, 0.23636427, 0.23436876, 0.23234926, 0.23031336,
                       0.22826727, 0.22621428, 0.22415407, 0.22208288, 0.21999457, 0.21788201,
                       0.21573877, 0.21356041, 0.21134525, 0.20909441, 0.2068112, 0.20450002,
                       0.20216503, 0.19980897, 0.19743238, 0.19503336, 0.19260792, 0.19015087,
                       0.1876569, 0.18512169, 0.18254277, 0.17991992, 0.17725501, 0.17455145,
                       0.17181325, 0.16904406, 0.16624635, 0.16342104, 0.16056757, 0.15768464,
                       0.15477132, 0.15182844, 0.14886002, 0.14587444, 0.1428851, 0.13991062,
                       0.13697427, 0.13410302, 0.13132614, 0.12867371, 0.12617515, 0.12385794,
                       0.12174663, 0.11986202, 0.11822048, 0.11683329, 0.11570595, 0.11483729,
                       0.11421878, 0.11383408, 0.11365932, 0.1136644, 0.11381551, 0.11407859,
                       0.11442326, 0.11482649, 0.11527517, 0.11576697, 0.11630905, 0.1169145,
                       0.11759685, 0.1183636, 0.11921039, 0.12011788, 0.12105276])

rrHeston_3 = np.array([0.38078185, 0.38315765, 0.38432447, 0.38407032, 0.38241926, 0.3795764,
                       0.37591797, 0.37198672, 0.36843585, 0.36586289, 0.36456539, 0.36440085,
                       0.36489509, 0.36548967, 0.36573024, 0.36533638, 0.3642008, 0.36236705,
                       0.36000496, 0.35738076, 0.35481036, 0.35258819, 0.35090435, 0.34978713,
                       0.34910627, 0.3486341, 0.34812631, 0.34738528, 0.34629359, 0.34482288,
                       0.34302691, 0.34102346, 0.33896664, 0.33701116, 0.33527464, 0.33380877,
                       0.33259044, 0.33153573, 0.33052962, 0.32945893, 0.3282384, 0.32682555,
                       0.32522427, 0.32347891, 0.32166096, 0.3198509, 0.31811865, 0.31650742,
                       0.31502506, 0.31364554, 0.31231896, 0.31098656, 0.30959569, 0.30811128,
                       0.3065217, 0.30483883, 0.30309306, 0.30132459, 0.29957324, 0.29786894,
                       0.29622545, 0.29463839, 0.29308814, 0.291546, 0.28998203, 0.28837212,
                       0.286703, 0.28497415, 0.28319667, 0.28138953, 0.27957443, 0.27777038,
                       0.27598938, 0.27423409, 0.27249805, 0.27076797, 0.26902752, 0.26726141,
                       0.26545885, 0.26361559, 0.26173427, 0.25982311, 0.25789347, 0.25595678,
                       0.25402178, 0.25209257, 0.25016796, 0.24824211, 0.24630621, 0.24435078,
                       0.24236785, 0.24035266, 0.23830439, 0.236226, 0.23412312, 0.23200236,
                       0.22986961, 0.22772854, 0.22557978, 0.22342082, 0.22124676, 0.21905147,
                       0.21682904, 0.21457502, 0.21228724, 0.20996607, 0.20761402, 0.20523484,
                       0.2028324, 0.20040962, 0.19796759, 0.19550526, 0.19301965, 0.19050646,
                       0.18796104, 0.18537937, 0.18275893, 0.18009915, 0.1774014, 0.1746686,
                       0.17190447, 0.16911262, 0.16629583, 0.16345559, 0.16059204, 0.15770452,
                       0.15479246, 0.15185656, 0.14890012, 0.14593009, 0.14295782, 0.13999933,
                       0.13707501, 0.13420877, 0.13142701, 0.12875721, 0.12622667, 0.12386133,
                       0.12168484, 0.11971778, 0.1179771, 0.11647553, 0.11522093, 0.11421557,
                       0.11345527, 0.11292872, 0.11261731, 0.11249574, 0.11253375, 0.11269904,
                       0.11296086, 0.11329386, 0.11368128, 0.11411671, 0.11460394, 0.11515442,
                       0.11578249, 0.11649899, 0.11730454, 0.11818489, 0.11911002])

rrHeston_4 = np.array([0.38965446, 0.39094881, 0.39154371, 0.39116412, 0.38975239, 0.38742717,
                       0.38445783, 0.38123653, 0.37821991, 0.37581963, 0.37426573, 0.37352273,
                       0.37332514, 0.37330344, 0.3731121, 0.37250552, 0.37136346, 0.36968809,
                       0.36758789, 0.36525183, 0.36291093, 0.36078599, 0.35903042, 0.35768899,
                       0.35669246, 0.35589034, 0.35510335, 0.35417261, 0.35299203, 0.35152243,
                       0.34979068, 0.3478777, 0.34589752, 0.34396995, 0.34219169, 0.34061337,
                       0.33922982, 0.33798673, 0.33680015, 0.33558086, 0.33425621, 0.33278435,
                       0.33115984, 0.32941093, 0.32759021, 0.32576071, 0.32398037, 0.32228851,
                       0.32069767, 0.31919289, 0.31773812, 0.31628724, 0.31479612, 0.31323274,
                       0.31158315, 0.30985284, 0.30806355, 0.30624677, 0.30443558, 0.30265661,
                       0.3009242, 0.29923801, 0.29758438, 0.29594092, 0.29428263, 0.29258807,
                       0.29084397, 0.28904752, 0.28720598, 0.28533402, 0.28344967, 0.28156963,
                       0.27970546, 0.27786112, 0.27603273, 0.27421007, 0.27237956, 0.27052776,
                       0.26864442, 0.26672463, 0.26476937, 0.26278477, 0.26078017, 0.25876554,
                       0.25674901, 0.25473504, 0.25272352, 0.25071013, 0.24868762, 0.2466477,
                       0.24458302, 0.24248871, 0.24036334, 0.23820882, 0.23602966, 0.23383158,
                       0.23161996, 0.22939847, 0.22716822, 0.22492753, 0.22267241, 0.2203976,
                       0.21809772, 0.21576852, 0.21340765, 0.21101502, 0.20859251, 0.20614334,
                       0.20367102, 0.2011784, 0.19866685, 0.19613586, 0.19358315, 0.19100514,
                       0.1883978, 0.18575751, 0.18308188, 0.18037023, 0.17762368, 0.17484484,
                       0.1720372, 0.16920438, 0.16634938, 0.16347416, 0.16057956, 0.15766565,
                       0.15473248, 0.15178116, 0.14881497, 0.14584034, 0.14286761, 0.13991129,
                       0.13698987, 0.13412513, 0.13134124, 0.12866355, 0.12611742, 0.12372719,
                       0.12151532, 0.11950169, 0.1177031, 0.11613277, 0.11479979, 0.11370839,
                       0.1128572, 0.11223835, 0.11183706, 0.11163178, 0.11159535, 0.11169741,
                       0.11190759, 0.11219934, 0.11255338, 0.11296009, 0.11342012, 0.11394274,
                       0.11454182, 0.11522973, 0.11601048, 0.11687402, 0.117794])

rrHeston_5 = np.array([0.39511924, 0.39595051, 0.39624821, 0.39574386, 0.39436226, 0.39219243,
                       0.38946183, 0.3865044, 0.38370407, 0.38140331, 0.37979768, 0.37887189,
                       0.37842218, 0.37814825, 0.37775548, 0.37702401, 0.37583847, 0.37419103,
                       0.37216939, 0.36993365, 0.36768172, 0.36560388, 0.36383474, 0.3624186,
                       0.3613034, 0.36036542, 0.35945177, 0.35842255, 0.35718077, 0.355687,
                       0.35396038, 0.35206883, 0.35011085, 0.34819151, 0.34639711, 0.34477486,
                       0.34332375, 0.34199921, 0.3407291, 0.3394347, 0.33805016, 0.33653597,
                       0.33488474, 0.33311953, 0.33128589, 0.32943968, 0.32763316, 0.3259028,
                       0.32426156, 0.3226978, 0.32118043, 0.31966847, 0.31812182, 0.31651059,
                       0.31482089, 0.31305637, 0.31123578, 0.30938719, 0.30754071, 0.30572107,
                       0.30394221, 0.30220474, 0.30049702, 0.29879901, 0.29708789, 0.29534362,
                       0.29355339, 0.2917139, 0.28983118, 0.28791835, 0.2859919, 0.28406744,
                       0.28215602, 0.28026184, 0.27838176, 0.27650665, 0.27462402, 0.27272126,
                       0.27078855, 0.2688209, 0.26681886, 0.26478786, 0.26273645, 0.26067399,
                       0.25860827, 0.2565437, 0.25448048, 0.25241476, 0.25033987, 0.248248,
                       0.24613208, 0.24398735, 0.24181221, 0.23960828, 0.23737968, 0.23513179,
                       0.23286976, 0.23059721, 0.22831535, 0.22602274, 0.22371573, 0.22138934,
                       0.21903844, 0.2166589, 0.21424837, 0.21180663, 0.20933543, 0.20683779,
                       0.20431711, 0.2017762, 0.1992165, 0.19663769, 0.19403772, 0.19141329,
                       0.1887606, 0.18607625, 0.18335796, 0.1806051, 0.17781878, 0.17500158,
                       0.17215698, 0.16928864, 0.16639969, 0.16349231, 0.16056764, 0.15762606,
                       0.15466799, 0.15169478, 0.14870988, 0.14571974, 0.14273451, 0.13976835,
                       0.13683921, 0.13396825, 0.1311789, 0.12849573, 0.12594338, 0.12354555,
                       0.12132414, 0.11929865, 0.11748568, 0.11589846, 0.11454636, 0.11343414,
                       0.11256122, 0.11192077, 0.11149917, 0.11127601, 0.11122512, 0.11131671,
                       0.11152055, 0.1118097, 0.11216406, 0.11257292, 0.11303585, 0.11356126,
                       0.11416263, 0.11485257, 0.11563592, 0.11650386, 0.11743128])

rrHeston_6 = np.array([0.39701921, 0.39772145, 0.39792862, 0.39737866, 0.39599473, 0.39385957,
                       0.39119021, 0.38830518, 0.38556874, 0.3833039, 0.3816948, 0.38072851,
                       0.38021534, 0.37987299, 0.37942169, 0.37865043, 0.377447, 0.37580229,
                       0.37379926, 0.37159061, 0.36936518, 0.36730438, 0.36553665, 0.36410481,
                       0.36296027, 0.36198602, 0.36103646, 0.3599775, 0.35871554, 0.35721194,
                       0.35548439, 0.35359767, 0.35164609, 0.34973046, 0.34793372, 0.34630152,
                       0.34483335, 0.34348689, 0.34219315, 0.34087644, 0.33947317, 0.33794491,
                       0.33628416, 0.33451284, 0.33267469, 0.33082355, 0.32900993, 0.32726918,
                       0.32561408, 0.32403362, 0.32249794, 0.32096749, 0.31940341, 0.31777661,
                       0.31607341, 0.31429719, 0.31246595, 0.31060688, 0.30874918, 0.30691695,
                       0.30512383, 0.30337056, 0.30164593, 0.29993053, 0.29820213, 0.29644119,
                       0.29463512, 0.29278057, 0.29088336, 0.2889562, 0.28701517, 0.28507551,
                       0.28314809, 0.28123708, 0.27933946, 0.27744635, 0.27554556, 0.27362472,
                       0.27167416, 0.26968896, 0.26766956, 0.26562126, 0.26355242, 0.26147221,
                       0.2593883, 0.25730508, 0.25522275, 0.25313759, 0.25104302, 0.24893138,
                       0.2467957, 0.24463126, 0.24243644, 0.24021281, 0.23796442, 0.23569657,
                       0.23341434, 0.23112133, 0.22881874, 0.2265052, 0.2241771, 0.22182954,
                       0.21945746, 0.21705677, 0.21462514, 0.21216236, 0.20967015, 0.2071515,
                       0.2046098, 0.20204786, 0.19946713, 0.19686734, 0.19424649, 0.19160136,
                       0.18892822, 0.18622374, 0.1834857, 0.18071351, 0.17790833, 0.17507277,
                       0.17221036, 0.16932478, 0.16641925, 0.16349602, 0.16055632, 0.15760068,
                       0.15462962, 0.15164464, 0.14864928, 0.14565007, 0.14265723, 0.1396849,
                       0.13675099, 0.13387657, 0.13108492, 0.12840049, 0.12584776, 0.12345026,
                       0.12122976, 0.11920565, 0.11739447, 0.11580945, 0.11445997, 0.11335091,
                       0.11248179, 0.11184593, 0.11142985, 0.11121327, 0.11117007, 0.11127046,
                       0.11148415, 0.11178405, 0.11214989, 0.11257074, 0.11304593, 0.11358364,
                       0.11419712, 0.11489882, 0.11569348, 0.11657232, 0.11751041])

rrHeston_AE_1 = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 3.03684829e-01,
                          3.12612699e-01, 3.15755234e-01, 3.15520034e-01, 3.12607767e-01,
                          3.07245806e-01, 2.99435825e-01, 2.89102571e-01, 2.76990266e-01,
                          2.69798583e-01, 2.75298190e-01, 2.83529414e-01, 2.89706879e-01,
                          2.93610691e-01, 2.95537294e-01, 2.95763615e-01, 2.94538067e-01,
                          2.92129806e-01, 2.88889792e-01, 2.85313230e-01, 2.82053268e-01,
                          2.79772423e-01, 2.78806144e-01, 2.78947576e-01, 2.79643223e-01,
                          2.80344521e-01, 2.80683549e-01, 2.80479802e-01, 2.79696062e-01,
                          2.78399927e-01, 2.76737457e-01, 2.74908615e-01, 2.73132449e-01,
                          2.71597713e-01, 2.70411490e-01, 2.69572233e-01, 2.68984652e-01,
                          2.68505851e-01, 2.67995093e-01, 2.67347445e-01, 2.66507947e-01,
                          2.65471684e-01, 2.64275418e-01, 2.62984037e-01, 2.61673579e-01,
                          2.60413144e-01, 2.59249344e-01, 2.58197538e-01, 2.57242323e-01,
                          2.56346253e-01, 2.55462856e-01, 2.54549343e-01, 2.53575790e-01,
                          2.52529548e-01, 2.51415126e-01, 2.50250482e-01, 2.49061030e-01,
                          2.47872819e-01, 2.46706411e-01, 2.45572800e-01, 2.44472148e-01,
                          2.43395329e-01, 2.42327473e-01, 2.41252332e-01, 2.40156242e-01,
                          2.39030846e-01, 2.37874170e-01, 2.36690099e-01, 2.35486611e-01,
                          2.34273341e-01, 2.33059087e-01, 2.31849852e-01, 2.30647795e-01,
                          2.29451228e-01, 2.28255508e-01, 2.27054504e-01, 2.25842188e-01,
                          2.24613983e-01, 2.23367584e-01, 2.22103138e-01, 2.20822839e-01,
                          2.19530108e-01, 2.18228591e-01, 2.16921236e-01, 2.15609644e-01,
                          2.14293816e-01, 2.12972299e-01, 2.11642655e-01, 2.10302091e-01,
                          2.08948100e-01, 2.07578944e-01, 2.06193912e-01, 2.04793296e-01,
                          2.03378142e-01, 2.01949850e-01, 2.00509741e-01, 1.99058675e-01,
                          1.97596829e-01, 1.96123644e-01, 1.94637956e-01, 1.93138241e-01,
                          1.91622928e-01, 1.90090682e-01, 1.88540604e-01, 1.86972311e-01,
                          1.85385888e-01, 1.83781729e-01, 1.82160323e-01, 1.80522032e-01,
                          1.78866919e-01, 1.77194659e-01, 1.75504547e-01, 1.73795605e-01,
                          1.72066745e-01, 1.70316965e-01, 1.68545513e-01, 1.66752014e-01,
                          1.64936511e-01, 1.63099435e-01, 1.61241513e-01, 1.59363651e-01,
                          1.57466822e-01, 1.55552006e-01, 1.53620192e-01, 1.51672476e-01,
                          1.49710236e-01, 1.47735367e-01, 1.45750548e-01, 1.43759487e-01,
                          1.41767123e-01, 1.39779743e-01, 1.37804986e-01, 1.35851753e-01,
                          1.33930011e-01, 1.32050524e-01, 1.30224531e-01, 1.28463414e-01,
                          1.26778361e-01, 1.25180047e-01, 1.23678346e-01, 1.22282047e-01,
                          1.20998588e-01, 1.19833775e-01, 1.18791491e-01, 1.17873405e-01,
                          1.17078715e-01, 1.16403977e-01, 1.15843113e-01, 1.15387671e-01,
                          1.15027404e-01, 1.14751188e-01, 1.14548191e-01, 1.14409169e-01,
                          1.14327619e-01, 1.14300530e-01, 1.14328418e-01, 1.14414411e-01,
                          1.14562279e-01])

rrHeston_AE_2 = np.array([1.36379788e-10, 3.05421580e-01, 3.21230787e-01, 3.28288893e-01,
                          3.31300973e-01, 3.31451666e-01, 3.29258660e-01, 3.25057136e-01,
                          3.19213835e-01, 3.12385180e-01, 3.05945186e-01, 3.02114459e-01,
                          3.02256490e-01, 3.05076713e-01, 3.08439476e-01, 3.11098773e-01,
                          3.12581908e-01, 3.12785958e-01, 3.11781902e-01, 3.09753578e-01,
                          3.06993653e-01, 3.03912209e-01, 3.01012312e-01, 2.98780242e-01,
                          2.97496340e-01, 2.97102742e-01, 2.97266696e-01, 2.97573610e-01,
                          2.97682177e-01, 2.97381860e-01, 2.96589916e-01, 2.95330104e-01,
                          2.93709900e-01, 2.91895887e-01, 2.90081361e-01, 2.88443517e-01,
                          2.87098286e-01, 2.86071208e-01, 2.85299854e-01, 2.84665566e-01,
                          2.84036241e-01, 2.83301721e-01, 2.82393875e-01, 2.81292537e-01,
                          2.80021326e-01, 2.78636630e-01, 2.77211938e-01, 2.75819816e-01,
                          2.74515019e-01, 2.73322987e-01, 2.72236998e-01, 2.71224246e-01,
                          2.70237889e-01, 2.69230564e-01, 2.68165498e-01, 2.67023128e-01,
                          2.65802812e-01, 2.64520245e-01, 2.63201764e-01, 2.61877018e-01,
                          2.60571689e-01, 2.59301916e-01, 2.58071629e-01, 2.56873172e-01,
                          2.55690671e-01, 2.54504900e-01, 2.53298231e-01, 2.52058460e-01,
                          2.50780839e-01, 2.49468129e-01, 2.48128958e-01, 2.46775051e-01,
                          2.45418099e-01, 2.44066988e-01, 2.42726029e-01, 2.41394462e-01,
                          2.40067231e-01, 2.38736667e-01, 2.37394570e-01, 2.36034143e-01,
                          2.34651350e-01, 2.33245469e-01, 2.31818805e-01, 2.30375743e-01,
                          2.28921430e-01, 2.27460426e-01, 2.25995643e-01, 2.24527789e-01,
                          2.23055385e-01, 2.21575297e-01, 2.20083584e-01, 2.18576447e-01,
                          2.17051026e-01, 2.15505897e-01, 2.13941175e-01, 2.12358245e-01,
                          2.10759223e-01, 2.09146301e-01, 2.07521133e-01, 2.05884403e-01,
                          2.04235663e-01, 2.02573446e-01, 2.00895606e-01, 1.99199777e-01,
                          1.97483852e-01, 1.95746342e-01, 1.93986567e-01, 1.92204638e-01,
                          1.90401246e-01, 1.88577345e-01, 1.86733776e-01, 1.84870964e-01,
                          1.82988723e-01, 1.81086229e-01, 1.79162142e-01, 1.77214861e-01,
                          1.75242828e-01, 1.73244825e-01, 1.71220194e-01, 1.69168938e-01,
                          1.67091692e-01, 1.64989580e-01, 1.62864008e-01, 1.60716453e-01,
                          1.58548307e-01, 1.56360837e-01, 1.54155277e-01, 1.51933063e-01,
                          1.49696174e-01, 1.47447536e-01, 1.45191408e-01, 1.42933708e-01,
                          1.40682199e-01, 1.38446528e-01, 1.36238081e-01, 1.34069701e-01,
                          1.31955278e-01, 1.29909284e-01, 1.27946282e-01, 1.26080453e-01,
                          1.24325175e-01, 1.22692638e-01, 1.21193502e-01, 1.19836555e-01,
                          1.18628363e-01, 1.17572893e-01, 1.16671115e-01, 1.15920651e-01,
                          1.15315556e-01, 1.14846365e-01, 1.14500517e-01, 1.14263268e-01,
                          1.14119047e-01, 1.14053152e-01, 1.14053501e-01, 1.14112110e-01,
                          1.14225891e-01, 1.14396438e-01, 1.14628534e-01, 1.14927394e-01,
                          1.15294986e-01])

rrHeston_AE_3 = np.array([0.31317234, 0.32692543, 0.33476824, 0.33902852, 0.34056479, 0.33982951,
                          0.33715189, 0.33288125, 0.32752982, 0.32195781, 0.31747092, 0.31535525,
                          0.31581232, 0.31778876, 0.32001562, 0.32167152, 0.32237316, 0.32201125,
                          0.32064554, 0.31846199, 0.31575981, 0.3129347, 0.31042112, 0.30857308,
                          0.30752835, 0.30716081, 0.30716727, 0.30721073, 0.30702344, 0.30644607,
                          0.30542759, 0.30401009, 0.3023086, 0.30048528, 0.29871506, 0.29714462,
                          0.29585518, 0.29484493, 0.29403947, 0.29332369, 0.29257902, 0.29171232,
                          0.29067159, 0.28944964, 0.28807875, 0.28661898, 0.28514222, 0.28371478,
                          0.28238208, 0.28115973, 0.28003309, 0.27896494, 0.27790788, 0.27681715,
                          0.27566053, 0.2744238, 0.27311133, 0.27174264, 0.27034615, 0.26895159,
                          0.26758288, 0.26625316, 0.26496288, 0.26370122, 0.26245011, 0.26118945,
                          0.25990198, 0.25857694, 0.25721165, 0.25581106, 0.25438569, 0.25294846,
                          0.25151139, 0.25008281, 0.24866583, 0.24725813, 0.24585312, 0.24444195,
                          0.24301579, 0.24156788, 0.24009478, 0.23859672, 0.23707713, 0.23554137,
                          0.23399525, 0.23244352, 0.23088881, 0.2293312, 0.22776837, 0.22619633,
                          0.22461048, 0.22300668, 0.22138207, 0.21973556, 0.2180678, 0.21638079,
                          0.21467718, 0.21295948, 0.21122939, 0.2094874, 0.20773261, 0.20596304,
                          0.20417603, 0.20236882, 0.20053913, 0.19868551, 0.19680752, 0.19490563,
                          0.19298091, 0.19103462, 0.18906777, 0.18708076, 0.18507322, 0.18304403,
                          0.1809915, 0.17891373, 0.17680901, 0.17467612, 0.17251458, 0.17032474,
                          0.16810766, 0.16586491, 0.16359831, 0.16130963, 0.15900046, 0.15667219,
                          0.15432616, 0.15196397, 0.14958796, 0.14720167, 0.14481031, 0.14242105,
                          0.14004326, 0.13768841, 0.13536985, 0.13310242, 0.13090191, 0.12878452,
                          0.12676626, 0.12486245, 0.12308728, 0.1214534, 0.11997153, 0.11865013,
                          0.11749495, 0.11650864, 0.11569023, 0.11503484, 0.11453348, 0.11417331,
                          0.1139384, 0.11381104, 0.11377358, 0.11381041, 0.11390994, 0.11406585,
                          0.11427751, 0.11454897, 0.11488651, 0.11529501, 0.11577372])

rrHeston_AE_4 = np.array([0.32893336, 0.33720989, 0.34278842, 0.34585798, 0.3466959, 0.34556155,
                          0.34273432, 0.33858673, 0.33368761, 0.32890342, 0.32532746, 0.32378317,
                          0.32419876, 0.32571628, 0.32737628, 0.32852368, 0.32882731, 0.32818737,
                          0.32666477, 0.32444627, 0.32182581, 0.31917457, 0.31687259, 0.31519868,
                          0.31422929, 0.31382271, 0.31370306, 0.31357669, 0.31321369, 0.31248072,
                          0.31134228, 0.30984813, 0.30811395, 0.30629497, 0.30455177, 0.30301183,
                          0.30173788, 0.30071613, 0.29986946, 0.29908794, 0.29826235, 0.29730971,
                          0.29618685, 0.29489298, 0.29346394, 0.29196027, 0.2904514, 0.2889989,
                          0.28764245, 0.28639231, 0.2852302, 0.28411751, 0.28300751, 0.28185765,
                          0.28063871, 0.27933951, 0.27796703, 0.27654252, 0.275095, 0.27365371,
                          0.2722412, 0.27086869, 0.26953466, 0.26822672, 0.26692585, 0.26561176,
                          0.26426769, 0.26288389, 0.26145891, 0.25999892, 0.2585154, 0.25702177,
                          0.25553005, 0.25404813, 0.25257831, 0.25111733, 0.24965778, 0.24819018,
                          0.24670548, 0.24519703, 0.24366179, 0.24210066, 0.24051769, 0.2389188,
                          0.23731013, 0.23569646, 0.23408021, 0.23246102, 0.23083604, 0.22920078,
                          0.22755028, 0.22588023, 0.22418783, 0.22247223, 0.22073446, 0.2189769,
                          0.21720251, 0.21541398, 0.21361302, 0.21179994, 0.20997356, 0.20813155,
                          0.20627094, 0.20438878, 0.20248268, 0.20055129, 0.19859434, 0.19661255,
                          0.19460727, 0.19257994, 0.19053169, 0.18846288, 0.18637302, 0.18426079,
                          0.18212429, 0.17996147, 0.17777051, 0.17555025, 0.17330038, 0.17102149,
                          0.16871495, 0.16638264, 0.16402662, 0.16164885, 0.15925103, 0.15683459,
                          0.15440089, 0.15195166, 0.14948945, 0.14701818, 0.14454367, 0.14207392,
                          0.13961931, 0.13719244, 0.13480791, 0.13248176, 0.13023093, 0.12807259,
                          0.1260236, 0.12409991, 0.12231612, 0.12068507, 0.11921744, 0.11792141,
                          0.11680214, 0.11586134, 0.11509675, 0.11450179, 0.11406549, 0.11377286,
                          0.11360592, 0.11354526, 0.11357218, 0.11367085, 0.11383029, 0.11404553,
                          0.11431758, 0.1146519, 0.11505541, 0.1155323, 0.11607958])

rrHeston_AE_5 = np.array([0.33770618, 0.34396278, 0.34839799, 0.35078946, 0.35121951, 0.34986852,
                          0.34699692, 0.34299032, 0.33843316, 0.33415165, 0.33107244, 0.32978011,
                          0.33011392, 0.33133693, 0.33264793, 0.3334822, 0.33354234, 0.33273781,
                          0.33113159, 0.32890932, 0.32635678, 0.32382299, 0.32164884, 0.32006713,
                          0.31912257, 0.31867044, 0.31845513, 0.31821048, 0.31772994, 0.31689597,
                          0.3156822, 0.31414216, 0.31239031, 0.31057564, 0.30884806, 0.30732223,
                          0.30604968, 0.30501035, 0.3041269, 0.30329382, 0.30240852, 0.30139473,
                          0.3002148, 0.29887168, 0.29740292, 0.29586869, 0.29433622, 0.29286361,
                          0.29148668, 0.29021245, 0.28902063, 0.28787216, 0.28672116, 0.28552676,
                          0.2842618, 0.28291709, 0.28150115, 0.28003611, 0.2785512, 0.27707511,
                          0.27562934, 0.27422383, 0.27285581, 0.27151196, 0.2701728, 0.26881802,
                          0.26743132, 0.26600364, 0.26453435, 0.26303041, 0.26150383, 0.25996831,
                          0.2584358, 0.25691383, 0.25540413, 0.25390286, 0.25240205, 0.25089188,
                          0.24936317, 0.24780937, 0.2462278, 0.24461975, 0.24298973, 0.24134399,
                          0.23968887, 0.23802916, 0.23636711, 0.23470202, 0.2330307, 0.23134834,
                          0.22964973, 0.22793046, 0.22618782, 0.22442113, 0.22263168, 0.22082212,
                          0.21899562, 0.21715501, 0.21530196, 0.21343666, 0.21155773, 0.2096626,
                          0.20774809, 0.20581111, 0.20384926, 0.20186121, 0.19984688, 0.19780716,
                          0.1957436, 0.19365779, 0.1915509, 0.1894233, 0.1872744, 0.18510274,
                          0.18290626, 0.18068281, 0.17843054, 0.17614832, 0.17383597, 0.17149429,
                          0.16912485, 0.16672976, 0.16431125, 0.1618714, 0.15941195, 0.15693436,
                          0.15443999, 0.15193063, 0.14940896, 0.14687919, 0.14434755, 0.14182263,
                          0.13931549, 0.13683956, 0.13441026, 0.13204446, 0.12975986, 0.12757432,
                          0.12550523, 0.12356893, 0.12178028, 0.12015223, 0.11869542, 0.11741778,
                          0.11632404, 0.11541522, 0.11468811, 0.11413493, 0.1137433, 0.11349674,
                          0.11337586, 0.11336016, 0.11343033, 0.11357053, 0.11377035, 0.11402584,
                          0.11433919, 0.11471681, 0.1151659, 0.11568995, 0.11628426])

rrHeston_AE_6 = np.array([0.34375837, 0.34891649, 0.35264659, 0.35459547, 0.35475804, 0.35327483,
                          0.35039694, 0.34651585, 0.34221479, 0.33827371, 0.33549887, 0.33433677,
                          0.33459228, 0.33560831, 0.33667998, 0.33730002, 0.3371947, 0.33628113,
                          0.33462395, 0.33240683, 0.32990737, 0.32745623, 0.3253657, 0.32383803,
                          0.32289931, 0.32240549, 0.32211605, 0.32178391, 0.32121842, 0.32031275,
                          0.31904649, 0.31747522, 0.31571171, 0.31389976, 0.31218117, 0.31066154,
                          0.30938495, 0.30832766, 0.30741282, 0.30653853, 0.3056071, 0.30454703,
                          0.30332444, 0.30194471, 0.30044627, 0.29888873, 0.29733748, 0.295848,
                          0.29445338, 0.29315842, 0.29194167, 0.29076392, 0.28958009, 0.28835062,
                          0.28704985, 0.28566995, 0.28422046, 0.28272406, 0.28120996, 0.27970641,
                          0.27823409, 0.27680201, 0.27540655, 0.27403379, 0.27266398, 0.27127687,
                          0.26985652, 0.26839438, 0.26689042, 0.26535212, 0.26379187, 0.26222352,
                          0.26065892, 0.25910532, 0.25756406, 0.25603087, 0.25449739, 0.25295357,
                          0.25139017, 0.24980075, 0.24818286, 0.24653809, 0.24487128, 0.24318894,
                          0.24149752, 0.2398018, 0.23810388, 0.23640286, 0.23469527, 0.23297605,
                          0.23123983, 0.22948217, 0.22770038, 0.22589395, 0.22406436, 0.22221447,
                          0.22034761, 0.21846666, 0.21657331, 0.21466763, 0.2127481, 0.21081198,
                          0.20885592, 0.20687674, 0.20487201, 0.2028405, 0.2007822, 0.1986982,
                          0.19659015, 0.19445978, 0.19230831, 0.19013608, 0.18794242, 0.18572578,
                          0.18348399, 0.18121481, 0.17891638, 0.1765876, 0.1742284, 0.17183972,
                          0.16942331, 0.16698142, 0.16451643, 0.16203047, 0.15952533, 0.15700246,
                          0.15446322, 0.15190942, 0.14934385, 0.14677092, 0.14419716, 0.14163159,
                          0.13908581, 0.13657384, 0.13411171, 0.1317169, 0.12940769, 0.1272024,
                          0.12511881, 0.12317356, 0.12138167, 0.11975616, 0.11830761, 0.11704377,
                          0.11596904, 0.1150839, 0.11438444, 0.11386194, 0.11350296, 0.11328992,
                          0.11320241, 0.11321917, 0.11332048, 0.11349057, 0.11371953, 0.11400422,
                          0.11434773, 0.11475713, 0.11523976, 0.1157985, 0.11642728])

rrHeston_AE_7 = np.array([0.34832146, 0.35277610, 0.35602267, 0.35765808, 0.35763152, 0.35606067,
                          0.35319104, 0.34941581, 0.34531045, 0.34161249, 0.33904017, 0.33795247,
                          0.33813993, 0.33900167, 0.33989778, 0.34036102, 0.34013528, 0.33914379,
                          0.33745247, 0.33524272, 0.33278470, 0.33039412, 0.32836163, 0.32686795,
                          0.32592715, 0.32539700, 0.32504855, 0.32464871, 0.32401848, 0.32305875,
                          0.32175332, 0.32015891, 0.31838666, 0.31657601, 0.31486252, 0.31334500,
                          0.31206226, 0.31098817, 0.31004665, 0.30913867, 0.30817032, 0.30707366,
                          0.30581758, 0.30440912, 0.30288725, 0.30131094, 0.29974410, 0.29824015,
                          0.29683013, 0.29551729, 0.29427938, 0.29307722, 0.29186639, 0.29060839,
                          0.28927869, 0.28787047, 0.28639399, 0.28487228, 0.28333448, 0.28180845,
                          0.28031424, 0.27886014, 0.27744194, 0.27604527, 0.27465021, 0.27323662,
                          0.27178882, 0.27029867, 0.26876660, 0.26720046, 0.26561289, 0.26401785,
                          0.26242711, 0.26084770, 0.25928065, 0.25772135, 0.25616117, 0.25458991,
                          0.25299826, 0.25137989, 0.24973254, 0.24805803, 0.24636145, 0.24464949,
                          0.24292868, 0.24120380, 0.23947685, 0.23774673, 0.23600976, 0.23426072,
                          0.23249411, 0.23070545, 0.22889210, 0.22705367, 0.22519180, 0.22330950,
                          0.22141023, 0.21949693, 0.21757127, 0.21563326, 0.21368123, 0.21171232,
                          0.20972306, 0.20771021, 0.20567133, 0.20360521, 0.20151198, 0.19939283,
                          0.19724954, 0.19508391, 0.19289721, 0.19068976, 0.18846084, 0.18620880,
                          0.18393140, 0.18162633, 0.17929171, 0.17692648, 0.17453065, 0.17210527,
                          0.16965224, 0.16717392, 0.16467279, 0.16215103, 0.15961044, 0.15705246,
                          0.15447846, 0.15189024, 0.14929067, 0.14668430, 0.14407793, 0.14148090,
                          0.13890522, 0.13636537, 0.13387787, 0.13146067, 0.12913248, 0.12691200,
                          0.12481728, 0.12286519, 0.12107088, 0.11944740, 0.11800531, 0.11675219,
                          0.11569217, 0.11482531, 0.11414712, 0.11364818, 0.11331420, 0.11312674,
                          0.11306460, 0.11310595, 0.11323080, 0.11342348, 0.11367452, 0.11398142,
                          0.11434797, 0.11478177, 0.11529022, 0.11587565, 0.11653091])

rrHeston_AE_8 = np.array([0.35194159, 0.35590221, 0.35879435, 0.36019515, 0.36002752, 0.35839488,
                          0.35553886, 0.35185205, 0.34789987, 0.34438318, 0.34195394, 0.34091169,
                          0.34104114, 0.34178295, 0.34254415, 0.34288709, 0.34256932, 0.3415191,
                          0.3398032, 0.33760085, 0.33517564, 0.33283095, 0.33084063, 0.32936942,
                          0.32842312, 0.32786159, 0.32746499, 0.32701102, 0.32632949, 0.32532721,
                          0.32399109, 0.32237856, 0.32059918, 0.31878884, 0.31707805, 0.31556043,
                          0.31427081, 0.31318148, 0.3122171, 0.31128102, 0.3102823, 0.30915579,
                          0.30787248, 0.30644064, 0.30489954, 0.3033076, 0.30172746, 0.3002109,
                          0.2987874, 0.29745901, 0.29620291, 0.29498004, 0.29374655, 0.29246477,
                          0.29111106, 0.28967938, 0.28818053, 0.28663778, 0.28508021, 0.2835353,
                          0.28202261, 0.2805499, 0.27911245, 0.27769559, 0.2762793, 0.2748435,
                          0.27337276, 0.27185928, 0.2703038, 0.26871448, 0.26710417, 0.26548687,
                          0.26387431, 0.26227331, 0.26068467, 0.25910351, 0.25752101, 0.25592682,
                          0.25431161, 0.25266914, 0.25099728, 0.24929807, 0.24757676, 0.2458402,
                          0.244095, 0.24234592, 0.24059485, 0.23884057, 0.23707923, 0.23530545,
                          0.23351366, 0.23169933, 0.22985988, 0.22799502, 0.2261065, 0.22419747,
                          0.2222715, 0.22033156, 0.21837932, 0.21641471, 0.21443599, 0.21244015,
                          0.21042367, 0.20838321, 0.20631636, 0.20422195, 0.20210019, 0.19995236,
                          0.19778034, 0.19558602, 0.19337067, 0.19113462, 0.18887709, 0.18659636,
                          0.18429013, 0.18195602, 0.17959215, 0.17719749, 0.17477211, 0.17231717,
                          0.16983465, 0.16732704, 0.16479686, 0.16224635, 0.15967732, 0.15709118,
                          0.15448928, 0.15187344, 0.14924659, 0.1466134, 0.14398085, 0.14135856,
                          0.13875887, 0.13619664, 0.13368878, 0.13125362, 0.12891021, 0.12667754,
                          0.12457391, 0.12261634, 0.12082009, 0.11919824, 0.11776131, 0.11651675,
                          0.11546847, 0.1146162, 0.11395495, 0.11347473, 0.11316057, 0.11299332,
                          0.11295119, 0.11301187, 0.1131552, 0.11336559, 0.11363396, 0.11395835,
                          0.11434314, 0.11479632, 0.11532531, 0.11593201, 0.1166083])

rrHeston_AE_16 = np.array([0.36698365, 0.36936034, 0.37103536, 0.37160195, 0.37093779, 0.3691138,
                           0.36635702, 0.36304026, 0.35966094, 0.35676168, 0.35476797, 0.3538113,
                           0.35368284, 0.35396465, 0.35421872, 0.35411045, 0.35344921, 0.35218379,
                           0.35038359, 0.3482154, 0.34591203, 0.34372584, 0.34186751, 0.3404474,
                           0.3394473, 0.33873912, 0.33813814, 0.33746192, 0.33657292, 0.3353995,
                           0.33393939, 0.33225114, 0.33043664, 0.32861644, 0.32690114, 0.32536479,
                           0.32402848, 0.32285984, 0.32178772, 0.32072531, 0.31959339, 0.31833738,
                           0.31693603, 0.31540186, 0.31377464, 0.31210995, 0.31046508, 0.3088853,
                           0.307394, 0.30598902, 0.30464583, 0.3033261, 0.30198851, 0.30059885,
                           0.29913699, 0.29759984, 0.29600012, 0.2943618, 0.29271346, 0.29108109,
                           0.28948217, 0.28792229, 0.28639493, 0.28488431, 0.28337008, 0.28183258,
                           0.28025738, 0.27863801, 0.27697659, 0.27528242, 0.27356909, 0.2718508,
                           0.27013893, 0.26843956, 0.26675245, 0.26507171, 0.26338767, 0.2616895,
                           0.25996782, 0.25821671, 0.25643472, 0.2546247, 0.25279268, 0.25094611,
                           0.24909186, 0.24723464, 0.24537591, 0.24351385, 0.24164391, 0.23976013,
                           0.23785655, 0.23592858, 0.23397382, 0.23199244, 0.22998677, 0.22796051,
                           0.22591764, 0.22386134, 0.22179321, 0.2197129, 0.21761821, 0.21550569,
                           0.2133714, 0.21121181, 0.20902449, 0.2068085, 0.20456443, 0.202294,
                           0.19999953, 0.19768317, 0.19534633, 0.19298925, 0.19061091, 0.18820926,
                           0.18578166, 0.18332554, 0.18083891, 0.17832089, 0.17577183, 0.17319329,
                           0.17058771, 0.16795796, 0.16530681, 0.16263664, 0.15994919, 0.15724577,
                           0.1545276, 0.15179648, 0.14905553, 0.14630994, 0.14356751, 0.14083905,
                           0.13813831, 0.13548178, 0.13288804, 0.13037704, 0.12796928, 0.12568499,
                           0.12354343, 0.12156232, 0.11975729, 0.11814154, 0.11672538, 0.11551569,
                           0.11451536, 0.11372261, 0.1131304, 0.11272612, 0.1124919, 0.11240565,
                           0.11244297, 0.11257985, 0.1127955, 0.11307505, 0.11341122, 0.11380461,
                           0.11426209, 0.11479328, 0.11540553, 0.11609847, 0.11685982])

rrHeston_AE_32 = np.array([0.37737074, 0.37897779, 0.38001823, 0.3801323, 0.37920041, 0.37728672,
                           0.37460782, 0.37151177, 0.36843915, 0.36583611, 0.36401903, 0.36305549,
                           0.36275222, 0.36276124, 0.36272011, 0.36234824, 0.36148515, 0.3600926,
                           0.35824097, 0.35608724, 0.35384296, 0.35172895, 0.3499208, 0.34850166,
                           0.34744367, 0.34662753, 0.3458886, 0.34506658, 0.34404223, 0.34275601,
                           0.34121145, 0.33946698, 0.33761901, 0.33577835, 0.33404364, 0.33247784,
                           0.33109476, 0.32986012, 0.32870573, 0.32755086, 0.32632324, 0.32497448,
                           0.32348782, 0.32187806, 0.32018501, 0.3184623, 0.31676378, 0.31513076,
                           0.31358305, 0.31211605, 0.31070436, 0.30931024, 0.307894, 0.30642356,
                           0.30488098, 0.30326493, 0.30158932, 0.2998785, 0.29816068, 0.29646089,
                           0.29479529, 0.29316809, 0.29157165, 0.28998948, 0.28840102, 0.28678693,
                           0.28513343, 0.28343492, 0.28169443, 0.27992204, 0.27813182, 0.27633812,
                           0.27455208, 0.27277924, 0.27101869, 0.26926383, 0.26750441, 0.26572927,
                           0.26392901, 0.26209797, 0.26023516, 0.25834403, 0.25643117, 0.25450444,
                           0.25257093, 0.25063523, 0.24869855, 0.24675859, 0.24481032, 0.24284736,
                           0.24086349, 0.23885406, 0.23681687, 0.23475243, 0.23266348, 0.23055414,
                           0.22842868, 0.22629043, 0.22414093, 0.22197957, 0.21980384, 0.21760992,
                           0.2153936, 0.21315118, 0.21088025, 0.20858006, 0.20625145, 0.20389651,
                           0.20151784, 0.19911779, 0.19669782, 0.19425808, 0.19179734, 0.18931326,
                           0.18680296, 0.18426366, 0.18169333, 0.17909115, 0.17645769, 0.17379479,
                           0.1711052, 0.16839202, 0.1656582, 0.16290614, 0.1601375, 0.15735344,
                           0.15455503, 0.15174404, 0.14892371, 0.14609958, 0.14328008, 0.14047686,
                           0.13770477, 0.13498147, 0.13232678, 0.12976181, 0.12730812, 0.12498682,
                           0.12281784, 0.12081936, 0.11900728, 0.11739485, 0.11599218, 0.11480572,
                           0.11383759, 0.11308485, 0.11253891, 0.11218522, 0.11200378, 0.11197036,
                           0.11205883, 0.11224404, 0.112505, 0.11282756, 0.11320599, 0.11364287,
                           0.11414693, 0.11472887, 0.11539578, 0.11614542, 0.11696224])

rrHeston_AE_64 = np.array([0.38451111, 0.38569831, 0.38638032, 0.3862322, 0.38514329, 0.38317711,
                           0.38054314, 0.37757078, 0.37466312, 0.37221034, 0.37047214, 0.36948662,
                           0.36907117, 0.36891488, 0.36869573, 0.36816342, 0.36717543, 0.36570174,
                           0.36381333, 0.36166086, 0.35944258, 0.35736046, 0.35557005, 0.35413966,
                           0.35303657, 0.35214678, 0.35131708, 0.35039988, 0.34928656, 0.34792478,
                           0.34632154, 0.3445352, 0.3426588, 0.34079725, 0.33904209, 0.33744974,
                           0.33602976, 0.33474676, 0.33353432, 0.33231524, 0.33102138, 0.32960812,
                           0.32806142, 0.32639754, 0.32465638, 0.32289041, 0.32115143, 0.31947833,
                           0.31788868, 0.31637636, 0.31491536, 0.31346822, 0.31199622, 0.31046859,
                           0.30886874, 0.30719655, 0.30546669, 0.30370386, 0.30193609, 0.30018781,
                           0.29847435, 0.29679899, 0.29515333, 0.29352035, 0.29187933, 0.29021107,
                           0.28850223, 0.28674781, 0.28495146, 0.28312384, 0.28127938, 0.27943253,
                           0.27759431, 0.27576989, 0.27395786, 0.27215111, 0.27033898, 0.26851006,
                           0.26665492, 0.26476808, 0.26284891, 0.26090129, 0.25893221, 0.25694984,
                           0.2549614, 0.25297146, 0.25098096, 0.2489873, 0.24698507, 0.24496757,
                           0.24292841, 0.24086291, 0.23876901, 0.23664745, 0.23450131, 0.23233501,
                           0.23015303, 0.22795878, 0.22575374, 0.22353713, 0.22130618, 0.2190568,
                           0.21678456, 0.21448567, 0.2121577, 0.20980004, 0.20741377, 0.20500119,
                           0.20256511, 0.20010804, 0.19763146, 0.19513542, 0.19261852, 0.19007821,
                           0.18751139, 0.18491516, 0.18228742, 0.17962741, 0.17693586, 0.17421481,
                           0.17146723, 0.16869641, 0.16590537, 0.16309652, 0.16027145, 0.15743117,
                           0.15457666, 0.15170962, 0.14883337, 0.14595374, 0.14307959, 0.14022325,
                           0.13740034, 0.13462941, 0.13193117, 0.12932761, 0.12684104, 0.12449321,
                           0.12230453, 0.1202935, 0.1184762, 0.11686592, 0.11547261, 0.1143024,
                           0.11335681, 0.11263206, 0.11211835, 0.11179975, 0.11165465, 0.11165729,
                           0.11178031, 0.11199781, 0.11228876, 0.11263966, 0.11304598, 0.11351181,
                           0.11404721, 0.11466364, 0.11536787, 0.11615614, 0.11701045])

rrHeston_AE_128 = np.array([0.38941682, 0.39035698, 0.39082359, 0.39051484, 0.38932794, 0.38732723,
                            0.38471764, 0.38181601, 0.37900157, 0.37663103, 0.37493198, 0.37392674,
                            0.3734392, 0.3731795, 0.37284892, 0.37221516, 0.37114686, 0.36961932,
                            0.36770408, 0.36554789, 0.36334076, 0.36127313, 0.3594885, 0.35804638,
                            0.3569112, 0.35597206, 0.3550825, 0.35410253, 0.35293002, 0.35151714,
                            0.34987315, 0.34805644, 0.34615806, 0.34427934, 0.3425074, 0.34089462,
                            0.33944793, 0.33813115, 0.33687883, 0.3356159, 0.33427676, 0.33281912,
                            0.33123074, 0.32952886, 0.32775355, 0.3259566, 0.32418858, 0.32248682,
                            0.32086748, 0.31932339, 0.31782809, 0.31634421, 0.3148336, 0.31326632,
                            0.31162667, 0.30991533, 0.30814757, 0.30634833, 0.30454559, 0.30276343,
                            0.3010166, 0.29930777, 0.29762798, 0.29595982, 0.29428242, 0.29257665,
                            0.29082945, 0.28903624, 0.28720114, 0.2853352, 0.28345314, 0.28156951,
                            0.27969524, 0.27783523, 0.27598774, 0.27414526, 0.27229682, 0.27043084,
                            0.26853785, 0.26661253, 0.26465448, 0.26266789, 0.26066007, 0.25863941,
                            0.25661322, 0.25458603, 0.25255863, 0.25052815, 0.24848892, 0.24643402,
                            0.2443569, 0.24225289, 0.24012, 0.23795919, 0.23577376, 0.23356835,
                            0.23134759, 0.22911495, 0.22687186, 0.2246174, 0.22234861, 0.22006121,
                            0.21775059, 0.21541288, 0.21304569, 0.21064848, 0.20822249, 0.20577021,
                            0.2032946, 0.20079826, 0.19828267, 0.19574783, 0.19319219, 0.19061304,
                            0.18800714, 0.18537146, 0.18270388, 0.18000368, 0.17727169, 0.17451012,
                            0.17172209, 0.168911, 0.16607996, 0.16323135, 0.16036671, 0.15748693,
                            0.1545929, 0.15168628, 0.14877046, 0.14585145, 0.14293847, 0.1400443,
                            0.13718516, 0.13438025, 0.13165094, 0.12901985, 0.12650984, 0.12414312,
                            0.12194044, 0.11992054, 0.11809963, 0.116491, 0.11510451, 0.11394603,
                            0.11301666, 0.11231196, 0.11182129, 0.11152764, 0.11140823, 0.11143623,
                            0.11158336, 0.11182326, 0.11213489, 0.11250527, 0.11293082, 0.11341671,
                            0.11397401, 0.11461464, 0.11534509, 0.11616042, 0.11704079])

rrHeston_AE_256 = np.array([0.39280753, 0.39359406, 0.39392475, 0.39351315, 0.39226238, 0.39023784,
                            0.38764159, 0.3847822, 0.38202348, 0.37970126, 0.37802364, 0.37700356,
                            0.37646887, 0.37614241, 0.37573967, 0.37503962, 0.37391811, 0.372354,
                            0.37041933, 0.36825841, 0.36605603, 0.36399531, 0.36221218, 0.36076061,
                            0.35960317, 0.3586308, 0.35770121, 0.35667917, 0.35546668, 0.35401884,
                            0.35234645, 0.35050804, 0.34859342, 0.34670166, 0.34491709, 0.34328948,
                            0.34182397, 0.34048384, 0.33920418, 0.33791124, 0.33654102, 0.33505276,
                            0.33343544, 0.33170699, 0.32990763, 0.3280888, 0.32630026, 0.32457836,
                            0.32293828, 0.32137213, 0.31985309, 0.31834384, 0.31680653, 0.31521178,
                            0.31354449, 0.31180591, 0.31001171, 0.30818708, 0.30635997, 0.30455423,
                            0.30278423, 0.30105222, 0.29934881, 0.29765632, 0.29595374, 0.29422198,
                            0.29244818, 0.29062804, 0.28876601, 0.28687346, 0.2849653, 0.28305618,
                            0.28115695, 0.27927234, 0.27740034, 0.27553318, 0.27365965, 0.27176803,
                            0.26984885, 0.26789685, 0.26591185, 0.26389825, 0.26186359, 0.25981641,
                            0.25776411, 0.25571118, 0.2536583, 0.25160239, 0.2495376, 0.24745683,
                            0.24535345, 0.24322276, 0.24106286, 0.23887484, 0.23666218, 0.23442966,
                            0.23218204, 0.22992282, 0.22765339, 0.22537274, 0.22307775, 0.22076399,
                            0.21842675, 0.2160621, 0.21366765, 0.21124294, 0.20878932, 0.20630941,
                            0.20380629, 0.20128261, 0.19873988, 0.19617801, 0.19359539, 0.19098916,
                            0.18835596, 0.1856927, 0.18299722, 0.18026885, 0.1775085, 0.17471847,
                            0.17190201, 0.16906261, 0.16620342, 0.16332682, 0.16043427, 0.15752661,
                            0.15460463, 0.15166998, 0.14872609, 0.14577909, 0.14283848, 0.13991737,
                            0.13703243, 0.1342033, 0.13145186, 0.12880117, 0.12627451, 0.12389439,
                            0.12168184, 0.11965575, 0.11783244, 0.11622518, 0.11484378, 0.11369391,
                            0.11277635, 0.1120862, 0.11161216, 0.11133645, 0.11123546, 0.11128154,
                            0.11144579, 0.1117015, 0.11202769, 0.11241178, 0.11285087, 0.11335096,
                            0.11392381, 0.11458168, 0.11533079, 0.11616532, 0.11706404])

rrHeston_AE_512 = np.array([0.39516938, 0.39585638, 0.39609811, 0.39561843, 0.39432476, 0.39228343,
                            0.38969471, 0.38686162, 0.38413786, 0.3818457, 0.38018078, 0.37915009,
                            0.37858395, 0.3782132, 0.37776244, 0.37701795, 0.37586037, 0.37427101,
                            0.37232233, 0.3701571, 0.3679567, 0.36589948, 0.3641164, 0.36265778,
                            0.36148494, 0.36048995, 0.35953317, 0.35848245, 0.35724254, 0.35577051,
                            0.35407824, 0.35222437, 0.35029799, 0.34839665, 0.34660288, 0.34496468,
                            0.34348601, 0.3421297, 0.34083117, 0.33951751, 0.33812577, 0.33661624,
                            0.3349787, 0.33323161, 0.33141532, 0.32958105, 0.32777806, 0.32604202,
                            0.32438745, 0.32280595, 0.32127043, 0.31974354, 0.31818767, 0.31657377,
                            0.31488719, 0.31312955, 0.31131685, 0.30947444, 0.30763028, 0.30580807,
                            0.30402192, 0.30227377, 0.30055393, 0.29884452, 0.2971244, 0.29537453,
                            0.29358216, 0.29174321, 0.28986237, 0.28795122, 0.28602484, 0.28409794,
                            0.28218131, 0.28027958, 0.27839052, 0.27650618, 0.27461518, 0.27270569,
                            0.27076823, 0.26879762, 0.26679379, 0.26476132, 0.26270791, 0.26064224,
                            0.25857173, 0.25650087, 0.25443022, 0.2523566, 0.25027399, 0.24817519,
                            0.24605348, 0.24390415, 0.24172538, 0.23951834, 0.23728663, 0.23503516,
                            0.23276877, 0.23049099, 0.22820317, 0.22590422, 0.22359092, 0.22125874,
                            0.21890288, 0.21651936, 0.21410581, 0.21166181, 0.20918882, 0.20668953,
                            0.20416711, 0.20162425, 0.19906247, 0.19648164, 0.19388006, 0.1912548,
                            0.18860241, 0.18591973, 0.1832046, 0.18045636, 0.17767598, 0.17486587,
                            0.17202933, 0.16916993, 0.16629084, 0.16339443, 0.16048213, 0.15755471,
                            0.15461291, 0.15165835, 0.1486945, 0.1457276, 0.14276731, 0.13982702,
                            0.13692369, 0.13407732, 0.13131012, 0.1286455, 0.12610702, 0.12371744,
                            0.12149795, 0.11946758, 0.1176427, 0.1160366, 0.11465902, 0.11351549,
                            0.11260658, 0.11192701, 0.11146503, 0.11120229, 0.11111456, 0.11117362,
                            0.1113501, 0.1116171, 0.11195364, 0.11234746, 0.11279619, 0.11330642,
                            0.11389039, 0.1145606, 0.11532305, 0.11617126, 0.11708301])

rrHeston_AE_1024 = np.array([0.39682618, 0.39744679, 0.39762871, 0.39710287, 0.39577976, 0.3937265,
                             0.39114217, 0.38832608, 0.38562508, 0.38335243, 0.3816955, 0.38065735,
                             0.38006986, 0.37966912, 0.37918571, 0.37841085, 0.3772284, 0.3756214,
                             0.37366265, 0.37149391, 0.3692943, 0.36723895, 0.3654555, 0.36399178,
                             0.36280824, 0.36179766, 0.36082215, 0.35975163, 0.35849269, 0.35700376,
                             0.35529748, 0.35343261, 0.35149775, 0.3495895, 0.34778911, 0.34614341,
                             0.3446555, 0.34328791, 0.34197626, 0.34064817, 0.33924141, 0.33771697,
                             0.33606522, 0.33430499, 0.33247675, 0.33063157, 0.32881838, 0.3270724,
                             0.32540766, 0.32381542, 0.32226837, 0.32072916, 0.31916029, 0.31753296,
                             0.31583281, 0.31406177, 0.31223605, 0.31038113, 0.30852499, 0.30669121,
                             0.30489374, 0.30313428, 0.30140295, 0.29968167, 0.29794927, 0.29618668,
                             0.29438127, 0.2925291, 0.29063503, 0.28871081, 0.28677163, 0.28483224,
                             0.28290341, 0.28098967, 0.27908866, 0.27719228, 0.27528902, 0.273367,
                             0.2714167, 0.269433, 0.26741593, 0.2653702, 0.26330362, 0.26122494,
                             0.25914165, 0.2570582, 0.25497509, 0.25288903, 0.25079392, 0.24868245,
                             0.24654785, 0.24438542, 0.24219336, 0.23997293, 0.23772782, 0.23546301,
                             0.23318342, 0.23089257, 0.22859182, 0.22628, 0.22395382, 0.22160867,
                             0.21923969, 0.21684287, 0.21441585, 0.21195825, 0.20947158, 0.20695862,
                             0.20442257, 0.20186618, 0.19929095, 0.19669674, 0.19408177, 0.19144306,
                             0.1887771, 0.18608068, 0.18335164, 0.18058932, 0.17779476, 0.1749704,
                             0.17211962, 0.16924603, 0.16635282, 0.16344235, 0.16051603, 0.15757457,
                             0.15461868, 0.15164996, 0.1486719, 0.14569081, 0.1427165, 0.13976253,
                             0.13684609, 0.13398743, 0.13120902, 0.1285345, 0.12598764, 0.12359138,
                             0.12136703, 0.1193337, 0.11750781, 0.11590266, 0.11452794, 0.1133891,
                             0.1124865, 0.11181464, 0.11136141, 0.11110805, 0.11102989, 0.11109828,
                             0.11128354, 0.1115586, 0.11190253, 0.11230331, 0.11275893, 0.11327642,
                             0.11386837, 0.11454743, 0.11531945, 0.11617743, 0.11709844])


def rHeston_smiles_precise():
    plt.plot(k_rrHeston[80:], rrHeston[80:], 'k', label="Non-Markovian approximation")
    plt.plot(k_rrHeston[80:], rrHeston_1[80:], label="N=1")
    plt.plot(k_rrHeston[80:], rrHeston_2[80:], label="N=2")
    plt.plot(k_rrHeston[80:], rrHeston_3[80:], label="N=3")
    plt.plot(k_rrHeston[80:], rrHeston_4[80:], label="N=4")
    plt.plot(k_rrHeston[80:], rrHeston_5[80:], label="N=5")
    plt.plot(k_rrHeston[80:], rrHeston_6[80:], label="N=6")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("implied volatility")
    plt.show()

    plt.plot(k_rrHeston[80:], rrHeston[80:], 'k', label="Non-Markovian approximation")
    plt.plot(k_rrHeston[80:], rrHeston_6[80:], 'k--', label="N=6, our method")
    plt.plot(k_rrHeston[80:], rrHeston_AE_1[80:], label="N=1, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], rrHeston_AE_4[80:], label="N=4, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], rrHeston_AE_16[80:], label="N=16, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], rrHeston_AE_64[80:], label="N=64, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], rrHeston_AE_256[80:], label="N=256, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], rrHeston_AE_1024[80:], label="N=1024, Abi Jaber, El Euch")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("implied volatility")
    plt.show()

    def relative_error(x, y):
        return np.abs((x-y)/x)

    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_1[80:]), label="N=1")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_2[80:]), label="N=2")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_3[80:]), label="N=3")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_4[80:]), label="N=4")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_5[80:]), label="N=5")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_6[80:]), label="N=6")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("relative error of implied volatility")
    plt.show()

    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_6[80:]), "k-", label="N=6, our method")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_AE_1[80:]), label="N=1, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_AE_4[80:]), label="N=4, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_AE_16[80:]), label="N=16, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_AE_64[80:]), label="N=64, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_AE_256[80:]), label="N=256, Abi Jaber, El Euch")
    plt.plot(k_rrHeston[80:], relative_error(rrHeston[80:], rrHeston_AE_1024[80:]), label="N=1024, Abi Jaber, El Euch")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("relative error of implied volatility")
    plt.show()

    error = np.zeros(6)
    N = np.arange(1, 7)
    error[0] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_1[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 1 node: {error[0]}")
    error[1] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_2[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 2 nodes: {error[1]}")
    error[2] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_3[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 3 nodes: {error[2]}")
    error[3] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_4[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 4 nodes: {error[3]}")
    error[4] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_5[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 5 nodes: {error[4]}")
    error[5] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_6[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 6 nodes: {error[5]}")

    N_AE = np.array([1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128, 256, 512, 1024])
    error_AE = np.zeros(14)
    error_AE[0] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_1[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 1 node: {error_AE[0]}")
    error_AE[1] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_2[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 2 nodes: {error_AE[1]}")
    error_AE[2] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_3[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 3 nodes: {error_AE[2]}")
    error_AE[3] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_4[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 4 nodes: {error_AE[3]}")
    error_AE[4] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_5[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 5 nodes: {error_AE[4]}")
    error_AE[5] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_6[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 8 nodes: {error_AE[5]}")
    error_AE[6] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_8[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 8 nodes: {error_AE[6]}")
    error_AE[7] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_16[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 16 nodes: {error_AE[7]}")
    error_AE[8] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_32[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 32 nodes: {error_AE[8]}")
    error_AE[9] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_64[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 64 nodes: {error_AE[9]}")
    error_AE[10] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_128[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 128 nodes: {error_AE[10]}")
    error_AE[11] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_256[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 256 nodes: {error_AE[11]}")
    error_AE[12] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_512[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 512 nodes: {error_AE[12]}")
    error_AE[13] = np.sqrt(np.trapz((rrHeston[80:] - rrHeston_AE_1024[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 1024 nodes: {error_AE[13]}")

    plt.loglog(N, error, label="our approximation")
    plt.loglog(N_AE, error_AE, label="Abi Jaber, El Euch")
    plt.legend(loc="upper right")
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.show()
