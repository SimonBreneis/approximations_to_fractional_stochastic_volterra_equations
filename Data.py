import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import mpmath as mp
from scipy import stats
import RoughKernel as rk
import ComputationalFinance as cf


c = ['r', 'C1', 'y', 'g', 'b']

def log_linear_regression(x, y):
    """
    Applies log-linear regression of y against x.
    :param x: The argument
    :param y: The function value
    :return: Exponent, constant, R-value, p-value, empirical standard deviation
    """
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(x), np.log(y))
    return slope, np.exp(intercept), r_value, p_value, std_err


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
fBm_errors_AK_original = [1.093956, 1.038641, 1.008279, 0.987654, 0.959794, 0.940801, 0.920437, 0.897258, 0.875517, 0.856277,
                          0.836913, 0.817406, 0.798393, 0.780402, 0.762558, 0.745117, 0.728091, 0.711447, 0.695196, 0.679308]
fBm_errors_AK_improved = [np.nan, 0.656022, np.nan, 0.471510, 0.362307, 0.290318, 0.201996, 0.151103, 0.106656,
                          0.068764, 0.043808, 0.028360, 0.017721, 0.010653, 0.006299, 0.003793, 0.002340, 0.001506,
                          0.001392, 0.000933]


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

    plt.loglog(fBm_n[0], fBm_errors_AK_original, color='orange', label="Alfonsi, Kebaier, original")
    plt.loglog(fBm_n[0], fBm_errors_AK_improved, color='blue', label="Alfonsi, Kebaier, improved")
    plt.loglog(fBm_n[0], fBm_errors_Harms_1, color='r', label="Harms, m=1")
    plt.loglog(fBm_n[0], fBm_errors_Harms_10, color='y', label="Harms, m=10")
    plt.loglog(fBm_n[0], fBm_errors_thm, color='g', label="Theorem")
    plt.loglog(fBm_n[0], fBm_errors_reg, color='purple', label="Estimates")
    plt.loglog(fBm_n[0], fBm_errors_opt_2, color='k', label=r"Optimal $\xi$ and m")
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
lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., S_0=1.
The vector of log-strikes is given below (k_vec).
rHeston is the Heston smile for the El Euch-Rosenbaum approximation (with log-strikes k_vec).
rHeston_N is the Heston smile for our approximation (with log-strikes k_vec) where N points were used and 
m and xi were chosen according to the interpolation of the numerical results.
rHeston_AE_N is the approximation by Abi Jaber and El Euch with N points.
'''

k_rHeston = -1.3 + 0.01 * np.arange(161)

rHeston = np.array([0.4008664, 0.40133726, 0.40138185, 0.40074782, 0.39935361, 0.39726891,
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

rHeston_1 = np.array([0.36679211, 0.37151053, 0.37349597, 0.37300667, 0.37032276, 0.36581155,
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

rHeston_2 = np.array([0.37123133, 0.37515465, 0.37692873, 0.37659932, 0.37435366, 0.37051441,
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

rHeston_3 = np.array([0.38078185, 0.38315765, 0.38432447, 0.38407032, 0.38241926, 0.3795764,
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

rHeston_4 = np.array([0.38965446, 0.39094881, 0.39154371, 0.39116412, 0.38975239, 0.38742717,
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

rHeston_5 = np.array([0.39511924, 0.39595051, 0.39624821, 0.39574386, 0.39436226, 0.39219243,
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

rHeston_6 = np.array([0.39701921, 0.39772145, 0.39792862, 0.39737866, 0.39599473, 0.39385957,
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

rHeston_AE_1 = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 3.03684829e-01,
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

rHeston_AE_2 = np.array([1.36379788e-10, 3.05421580e-01, 3.21230787e-01, 3.28288893e-01,
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

rHeston_AE_3 = np.array([0.31317234, 0.32692543, 0.33476824, 0.33902852, 0.34056479, 0.33982951,
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

rHeston_AE_4 = np.array([0.32893336, 0.33720989, 0.34278842, 0.34585798, 0.3466959, 0.34556155,
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

rHeston_AE_5 = np.array([0.33770618, 0.34396278, 0.34839799, 0.35078946, 0.35121951, 0.34986852,
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

rHeston_AE_6 = np.array([0.34375837, 0.34891649, 0.35264659, 0.35459547, 0.35475804, 0.35327483,
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

rHeston_AE_7 = np.array([0.34832146, 0.35277610, 0.35602267, 0.35765808, 0.35763152, 0.35606067,
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

rHeston_AE_8 = np.array([0.35194159, 0.35590221, 0.35879435, 0.36019515, 0.36002752, 0.35839488,
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

rHeston_AE_16 = np.array([0.36698365, 0.36936034, 0.37103536, 0.37160195, 0.37093779, 0.3691138,
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

rHeston_AE_32 = np.array([0.37737074, 0.37897779, 0.38001823, 0.3801323, 0.37920041, 0.37728672,
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

rHeston_AE_64 = np.array([0.38451111, 0.38569831, 0.38638032, 0.3862322, 0.38514329, 0.38317711,
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

rHeston_AE_128 = np.array([0.38941682, 0.39035698, 0.39082359, 0.39051484, 0.38932794, 0.38732723,
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

rHeston_AE_256 = np.array([0.39280753, 0.39359406, 0.39392475, 0.39351315, 0.39226238, 0.39023784,
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

rHeston_AE_512 = np.array([0.39516938, 0.39585638, 0.39609811, 0.39561843, 0.39432476, 0.39228343,
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

rHeston_AE_1024 = np.array([0.39682618, 0.39744679, 0.39762871, 0.39710287, 0.39577976, 0.3937265,
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


def rHeston_smiles():
    plt.plot(k_rHeston[80:], rHeston[80:], 'k', label="Non-Markovian approximation")
    plt.plot(k_rHeston[80:], rHeston_1[80:], label="N=1")
    plt.plot(k_rHeston[80:], rHeston_2[80:], label="N=2")
    plt.plot(k_rHeston[80:], rHeston_3[80:], label="N=3")
    plt.plot(k_rHeston[80:], rHeston_4[80:], label="N=4")
    plt.plot(k_rHeston[80:], rHeston_5[80:], label="N=5")
    plt.plot(k_rHeston[80:], rHeston_6[80:], label="N=6")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("implied volatility")
    plt.show()

    plt.plot(k_rHeston[80:], rHeston[80:], 'k', label="Non-Markovian approximation")
    plt.plot(k_rHeston[80:], rHeston_6[80:], 'k--', label="N=6, our method")
    plt.plot(k_rHeston[80:], rHeston_AE_1[80:], label="N=1, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], rHeston_AE_4[80:], label="N=4, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], rHeston_AE_16[80:], label="N=16, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], rHeston_AE_64[80:], label="N=64, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], rHeston_AE_256[80:], label="N=256, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], rHeston_AE_1024[80:], label="N=1024, Abi Jaber, El Euch")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("implied volatility")
    plt.show()

    def relative_error(x, y):
        return np.abs((x - y) / x)

    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_1[80:]), label="N=1")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_2[80:]), label="N=2")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_3[80:]), label="N=3")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_4[80:]), label="N=4")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_5[80:]), label="N=5")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_6[80:]), label="N=6")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("relative error of implied volatility")
    plt.show()

    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_6[80:]), "k-", label="N=6, our method")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_AE_1[80:]), label="N=1, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_AE_4[80:]), label="N=4, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_AE_16[80:]), label="N=16, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_AE_64[80:]), label="N=64, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_AE_256[80:]), label="N=256, Abi Jaber, El Euch")
    plt.plot(k_rHeston[80:], relative_error(rHeston[80:], rHeston_AE_1024[80:]), label="N=1024, Abi Jaber, El Euch")
    plt.legend(loc="upper right")
    plt.xlabel("log-moneyness")
    plt.ylabel("relative error of implied volatility")
    plt.show()

    error = np.zeros(6)
    N = np.arange(1, 7)
    error[0] = np.sqrt(np.trapz((rHeston[80:] - rHeston_1[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 1 node: {error[0]}")
    error[1] = np.sqrt(np.trapz((rHeston[80:] - rHeston_2[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 2 nodes: {error[1]}")
    error[2] = np.sqrt(np.trapz((rHeston[80:] - rHeston_3[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 3 nodes: {error[2]}")
    error[3] = np.sqrt(np.trapz((rHeston[80:] - rHeston_4[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 4 nodes: {error[3]}")
    error[4] = np.sqrt(np.trapz((rHeston[80:] - rHeston_5[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 5 nodes: {error[4]}")
    error[5] = np.sqrt(np.trapz((rHeston[80:] - rHeston_6[80:]) ** 2, dx=0.01))
    print(f"Error for our approximation with 6 nodes: {error[5]}")

    N_AE = np.array([1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128, 256, 512, 1024])
    error_AE = np.zeros(14)
    error_AE[0] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_1[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 1 node: {error_AE[0]}")
    error_AE[1] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_2[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 2 nodes: {error_AE[1]}")
    error_AE[2] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_3[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 3 nodes: {error_AE[2]}")
    error_AE[3] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_4[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 4 nodes: {error_AE[3]}")
    error_AE[4] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_5[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 5 nodes: {error_AE[4]}")
    error_AE[5] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_6[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 8 nodes: {error_AE[5]}")
    error_AE[6] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_8[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 8 nodes: {error_AE[6]}")
    error_AE[7] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_16[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 16 nodes: {error_AE[7]}")
    error_AE[8] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_32[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 32 nodes: {error_AE[8]}")
    error_AE[9] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_64[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 64 nodes: {error_AE[9]}")
    error_AE[10] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_128[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 128 nodes: {error_AE[10]}")
    error_AE[11] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_256[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 256 nodes: {error_AE[11]}")
    error_AE[12] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_512[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 512 nodes: {error_AE[12]}")
    error_AE[13] = np.sqrt(np.trapz((rHeston[80:] - rHeston_AE_1024[80:]) ** 2, dx=0.01))
    print(f"Error for AE approximation with 1024 nodes: {error_AE[13]}")

    plt.loglog(N, error, label="our approximation")
    plt.loglog(N_AE, error_AE, label="Abi Jaber, El Euch")
    plt.legend(loc="upper right")
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.show()


'''
The rHeston implied volatility smiles for European call options. Parameters used are
lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, T=1., S_0=1., N_time=1000, m=200000
The vector of log-strikes is given below (k_vec).
rHeston_IE_x is the Heston smile for our approximation (with log-strikes k_vec) where 6 (hence, 7) points were used and 
m and xi were chosen according to the interpolation of the numerical results. If x is bouncy, it was ensured that the
volatility is not zero twice in a row. If x is sticky, no such intervention was done.
The duration notes how long it took to compute those smiles. No parallelization was used.
lower and upper refer to the 95% confidence bounds.
'''

k_rHeston_IE = -1.3 + 0.01 * np.arange(161)

rHeston_IE_bouncy_duration = 3594.0693

rHeston_IE_bouncy = np.array([0.49990642, 0.49645114, 0.49300103, 0.48956375, 0.48614694, 0.48275102,
                              0.47936615, 0.47599658, 0.47264186, 0.46930649, 0.46599066, 0.46268921,
                              0.45940016, 0.45612925, 0.45287146, 0.4496269, 0.44639667, 0.44318875,
                              0.43999088, 0.43681144, 0.43364873, 0.43049944, 0.42736865, 0.42425768,
                              0.42117699, 0.41811364, 0.41506535, 0.41203811, 0.40903508, 0.40604997,
                              0.4030834, 0.40014233, 0.39722944, 0.39434161, 0.39148034, 0.38864319,
                              0.38581372, 0.38300284, 0.38020589, 0.37742584, 0.37467338, 0.37194789,
                              0.36925887, 0.36659178, 0.36396216, 0.3613568, 0.35877155, 0.35620677,
                              0.35366749, 0.35113924, 0.34863122, 0.34613704, 0.34366236, 0.34120953,
                              0.33877582, 0.33635261, 0.33394291, 0.33155028, 0.32918393, 0.32683724,
                              0.32449748, 0.32216838, 0.31983194, 0.3175021, 0.3152002, 0.31290854,
                              0.31061918, 0.30834231, 0.3060946, 0.30385877, 0.30163053, 0.29942023,
                              0.29721891, 0.2950102, 0.29279271, 0.29057501, 0.28836239, 0.28615437,
                              0.28395358, 0.28174723, 0.27953421, 0.27731571, 0.27509463, 0.27286803,
                              0.27063353, 0.26839283, 0.26614527, 0.26389809, 0.2616422, 0.25938637,
                              0.25712511, 0.25485775, 0.25257727, 0.25029215, 0.24799454, 0.24568779,
                              0.24337963, 0.24106076, 0.23874244, 0.23641308, 0.23407015, 0.23172078,
                              0.22936578, 0.22699554, 0.22460255, 0.22219874, 0.21978458, 0.21736222,
                              0.21492661, 0.2124728, 0.21000308, 0.20751201, 0.20500808, 0.20249194,
                              0.19995153, 0.19739247, 0.19480744, 0.19220451, 0.18958221, 0.18695215,
                              0.18430626, 0.18164579, 0.17898293, 0.17631999, 0.1736539, 0.17098546,
                              0.16831605, 0.16565249, 0.16298879, 0.16033982, 0.15772212, 0.15514191,
                              0.15259386, 0.15007683, 0.14759492, 0.14517794, 0.14281609, 0.14052424,
                              0.13831072, 0.13619251, 0.13418312, 0.13229522, 0.1305199, 0.12887322,
                              0.12735955, 0.1259978, 0.12478513, 0.12372379, 0.12282723, 0.12209829,
                              0.12154071, 0.12114906, 0.12088788, 0.12074556, 0.12070695, 0.12073709,
                              0.12085191, 0.12106102, 0.12134018, 0.12166956, 0.12205733])

rHeston_IE_bouncy_lower = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 2.57021498e-01, 2.69234651e-01, 2.76201397e-01,
                                    2.80889013e-01, 2.84198797e-01, 2.86634453e-01, 2.88423763e-01,
                                    2.89741997e-01, 2.90696474e-01, 2.91350197e-01, 2.91735767e-01,
                                    2.91902809e-01, 2.91890750e-01, 2.91741911e-01, 2.91461350e-01,
                                    2.91042369e-01, 2.90507233e-01, 2.89838802e-01, 2.89072945e-01,
                                    2.88252254e-01, 2.87355398e-01, 2.86377638e-01, 2.85340505e-01,
                                    2.84272484e-01, 2.83152897e-01, 2.81980220e-01, 2.80772275e-01,
                                    2.79520458e-01, 2.78207232e-01, 2.76834864e-01, 2.75417628e-01,
                                    2.73964833e-01, 2.72478287e-01, 2.70963325e-01, 2.69406777e-01,
                                    2.67809618e-01, 2.66175322e-01, 2.64509107e-01, 2.62809286e-01,
                                    2.61074803e-01, 2.59309112e-01, 2.57512863e-01, 2.55695340e-01,
                                    2.53847684e-01, 2.51980604e-01, 2.50089105e-01, 2.48173435e-01,
                                    2.46227017e-01, 2.44259864e-01, 2.42264419e-01, 2.40245086e-01,
                                    2.38210731e-01, 2.36152237e-01, 2.34082057e-01, 2.31988631e-01,
                                    2.29869941e-01, 2.27733994e-01, 2.25582121e-01, 2.23404890e-01,
                                    2.21195090e-01, 2.18965491e-01, 2.16717009e-01, 2.14452233e-01,
                                    2.12166393e-01, 2.09854791e-01, 2.07520153e-01, 2.05157275e-01,
                                    2.02775122e-01, 2.00374645e-01, 1.97943910e-01, 1.95488874e-01,
                                    1.93002423e-01, 1.90492940e-01, 1.87959181e-01, 1.85413055e-01,
                                    1.82846614e-01, 1.80261294e-01, 1.77669519e-01, 1.75073746e-01,
                                    1.72471039e-01, 1.69862297e-01, 1.67249027e-01, 1.64638167e-01,
                                    1.62023795e-01, 1.59420897e-01, 1.56846126e-01, 1.54305772e-01,
                                    1.51794525e-01, 1.49311234e-01, 1.46860018e-01, 1.44470767e-01,
                                    1.42133604e-01, 1.39863397e-01, 1.37668420e-01, 1.35565610e-01,
                                    1.33568421e-01, 1.31689460e-01, 1.29919622e-01, 1.28274832e-01,
                                    1.26759293e-01, 1.25391810e-01, 1.24169310e-01, 1.23093793e-01,
                                    1.22178634e-01, 1.21426588e-01, 1.20841406e-01, 1.20417667e-01,
                                    1.20119335e-01, 1.19934476e-01, 1.19847471e-01, 1.19821905e-01,
                                    1.19873647e-01, 1.20012337e-01, 1.20212488e-01, 1.20452415e-01,
                                    1.20739745e-01, ])

rHeston_IE_bouncy_upper = np.array([0.5681805, 0.56409666, 0.56001708, 0.55594533, 0.55188499, 0.54783632,
                                    0.54379479, 0.53976243, 0.53573909, 0.53172695, 0.52772621, 0.5237345,
                                    0.51975093, 0.51577836, 0.51181446, 0.50785935, 0.50391368, 0.49998147,
                                    0.49605682, 0.492144, 0.4882423, 0.48435022, 0.48047046, 0.47660387,
                                    0.47275605, 0.46892063, 0.46509663, 0.4612874, 0.45749487, 0.45371602,
                                    0.44995145, 0.44620521, 0.44247916, 0.43877204, 0.4350851, 0.43141743,
                                    0.42776016, 0.42411973, 0.42049382, 0.41688448, 0.41329842, 0.40973583,
                                    0.40620305, 0.402692, 0.39921279, 0.39575807, 0.39232582, 0.38891681,
                                    0.38553486, 0.38217119, 0.37883224, 0.3755143, 0.37222163, 0.36895639,
                                    0.36571733, 0.36249893, 0.35930369, 0.35613456, 0.35299868, 0.34989191,
                                    0.34680537, 0.34374205, 0.34068865, 0.33765564, 0.33465966, 0.33168757,
                                    0.32873333, 0.32580505, 0.32291637, 0.32005385, 0.31721416, 0.31440591,
                                    0.31162189, 0.3088484, 0.30608394, 0.30333547, 0.30060735, 0.29789907,
                                    0.29521281, 0.29253721, 0.28987096, 0.28721477, 0.28457091, 0.2819364,
                                    0.27930877, 0.27668917, 0.27407663, 0.27147745, 0.26888297, 0.26630094,
                                    0.26372596, 0.26115705, 0.25858727, 0.25602421, 0.25346008, 0.2508977,
                                    0.24834407, 0.24579002, 0.24324599, 0.24070058, 0.23815097, 0.23560375,
                                    0.23305938, 0.23050825, 0.22794272, 0.22537404, 0.22280242, 0.22022966,
                                    0.21765057, 0.21506001, 0.21245998, 0.20984483, 0.20722269, 0.20459396,
                                    0.20194653, 0.19928569, 0.19660402, 0.19390927, 0.19119981, 0.18848698,
                                    0.18576261, 0.1830278, 0.1802945, 0.17756491, 0.17483587, 0.17210805,
                                    0.16938274, 0.16666665, 0.16395374, 0.16125876, 0.15859814, 0.15597805,
                                    0.15339316, 0.15084229, 0.14832957, 0.14588472, 0.14349801, 0.14118433,
                                    0.13895204, 0.13681817, 0.13479627, 0.13289908, 0.13111787, 0.1294688,
                                    0.12795642, 0.12659969, 0.12539601, 0.1243478, 0.12346856, 0.12276118,
                                    0.1222293, 0.12186743, 0.1216406, 0.12153736, 0.12154291, 0.12162348,
                                    0.12179486, 0.12206639, 0.1224147, 0.12282124, 0.12329423])

rHeston_IE_sticky_duration = 3378.70300559

rHeston_IE_sticky = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                              1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                              1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                              1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                              1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                              1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                              1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                              1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 2.56902567e-01,
                              2.82672929e-01, 2.92588720e-01, 2.98623033e-01, 3.02733452e-01,
                              3.05701604e-01, 3.07931560e-01, 3.09590360e-01, 3.10781829e-01,
                              3.11581698e-01, 3.12063642e-01, 3.12317646e-01, 3.12364813e-01,
                              3.12236914e-01, 3.11990630e-01, 3.11606401e-01, 3.11107701e-01,
                              3.10497268e-01, 3.09786813e-01, 3.08978424e-01, 3.08096233e-01,
                              3.07160937e-01, 3.06196330e-01, 3.05176931e-01, 3.04094978e-01,
                              3.02943881e-01, 3.01754612e-01, 3.00525015e-01, 2.99259565e-01,
                              2.97961874e-01, 2.96640143e-01, 2.95290478e-01, 2.93907432e-01,
                              2.92484630e-01, 2.91034185e-01, 2.89550623e-01, 2.88039917e-01,
                              2.86510740e-01, 2.84961785e-01, 2.83372347e-01, 2.81742849e-01,
                              2.80076208e-01, 2.78384576e-01, 2.76663506e-01, 2.74925189e-01,
                              2.73163532e-01, 2.71380534e-01, 2.69584062e-01, 2.67764426e-01,
                              2.65924608e-01, 2.64056453e-01, 2.62172913e-01, 2.60266880e-01,
                              2.58340736e-01, 2.56396421e-01, 2.54423108e-01, 2.52430114e-01,
                              2.50412021e-01, 2.48372144e-01, 2.46314920e-01, 2.44240105e-01,
                              2.42135772e-01, 2.40010786e-01, 2.37866645e-01, 2.35704654e-01,
                              2.33529485e-01, 2.31329241e-01, 2.29103768e-01, 2.26862241e-01,
                              2.24606143e-01, 2.22323245e-01, 2.20009796e-01, 2.17666213e-01,
                              2.15291229e-01, 2.12880288e-01, 2.10437800e-01, 2.07968426e-01,
                              2.05480501e-01, 2.02970372e-01, 2.00432806e-01, 1.97873990e-01,
                              1.95282684e-01, 1.92662138e-01, 1.90013185e-01, 1.87335445e-01,
                              1.84639317e-01, 1.81919118e-01, 1.79179425e-01, 1.76420760e-01,
                              1.73641111e-01, 1.70834154e-01, 1.68006701e-01, 1.65159308e-01,
                              1.62294496e-01, 1.59409312e-01, 1.56508573e-01, 1.53601930e-01,
                              1.50687349e-01, 1.47775381e-01, 1.44871153e-01, 1.41982940e-01,
                              1.39130420e-01, 1.36333458e-01, 1.33609997e-01, 1.30972630e-01,
                              1.28442254e-01, 1.26046622e-01, 1.23810906e-01, 1.21764253e-01,
                              1.19921150e-01, 1.18293411e-01, 1.16900845e-01, 1.15728307e-01,
                              1.14781741e-01, 1.14051319e-01, 1.13530823e-01, 1.13213182e-01,
                              1.13033580e-01, 1.13013312e-01, 1.13118852e-01, 1.13334462e-01,
                              1.13636919e-01, 1.14042807e-01, 1.14534014e-01, 1.15075570e-01,
                              1.15637073e-01, 1.16268481e-01, 1.16969412e-01, 1.17719637e-01,
                              1.18538815e-01])

rHeston_IE_sticky_lower = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
                                    1.36379788e-10, 2.24727529e-01, 2.36382596e-01, 2.43093351e-01,
                                    2.47564215e-01, 2.50749187e-01, 2.53061268e-01, 2.54752377e-01,
                                    2.55984462e-01, 2.56849818e-01, 2.57381805e-01, 2.57634837e-01,
                                    2.57653500e-01, 2.57485851e-01, 2.57148148e-01, 2.56675624e-01,
                                    2.56074269e-01, 2.55358952e-01, 2.54549825e-01, 2.53642930e-01,
                                    2.52649514e-01, 2.51566140e-01, 2.50414167e-01, 2.49189875e-01,
                                    2.47900400e-01, 2.46551745e-01, 2.45134927e-01, 2.43663702e-01,
                                    2.42134701e-01, 2.40554127e-01, 2.38929056e-01, 2.37261122e-01,
                                    2.35539145e-01, 2.33774476e-01, 2.31970204e-01, 2.30129056e-01,
                                    2.28257193e-01, 2.26343103e-01, 2.24387746e-01, 2.22401765e-01,
                                    2.20387561e-01, 2.18333182e-01, 2.16235573e-01, 2.14095966e-01,
                                    2.11913786e-01, 2.09685033e-01, 2.07414897e-01, 2.05108739e-01,
                                    2.02775599e-01, 2.00412178e-01, 1.98013563e-01, 1.95586472e-01,
                                    1.93119840e-01, 1.90617352e-01, 1.88080194e-01, 1.85508296e-01,
                                    1.82912472e-01, 1.80287225e-01, 1.77637421e-01, 1.74963800e-01,
                                    1.72264538e-01, 1.69533450e-01, 1.66777580e-01, 1.63997646e-01,
                                    1.61196324e-01, 1.58370770e-01, 1.55525930e-01, 1.52671588e-01,
                                    1.49805760e-01, 1.46939082e-01, 1.44076715e-01, 1.41226949e-01,
                                    1.38409487e-01, 1.35644188e-01, 1.32948955e-01, 1.30336297e-01,
                                    1.27827021e-01, 1.25448794e-01, 1.23226694e-01, 1.21189796e-01,
                                    1.19352486e-01, 1.17726499e-01, 1.16331649e-01, 1.15152658e-01,
                                    1.14195482e-01, 1.13450267e-01, 1.12910852e-01, 1.12570300e-01,
                                    1.12363082e-01, 1.12310842e-01, 1.12379634e-01, 1.12553421e-01,
                                    1.12808320e-01, 1.13161329e-01, 1.13593910e-01, 1.14069708e-01,
                                    1.14556126e-01, 1.15104752e-01, 1.15715226e-01, 1.16366136e-01,
                                    1.17078736e-01])

rHeston_IE_sticky_upper = np.array([0.52091173, 0.51723326, 0.51356013, 0.5098991, 0.50624399, 0.50259789,
                                    0.49896793, 0.49534884, 0.49174148, 0.48814498, 0.48456034, 0.48098574,
                                    0.47742317, 0.47387052, 0.47032568, 0.46679956, 0.46328981, 0.45979161,
                                    0.45631474, 0.4528548, 0.44940927, 0.44598194, 0.44256751, 0.43916729,
                                    0.43577547, 0.43239981, 0.42903501, 0.42568883, 0.42236332, 0.41905871,
                                    0.41577423, 0.41250718, 0.40925683, 0.40603501, 0.40284089, 0.39966922,
                                    0.39652202, 0.3934052, 0.39031279, 0.38723802, 0.3841752, 0.38112379,
                                    0.37809347, 0.37508029, 0.37208503, 0.36911927, 0.36617129, 0.36324396,
                                    0.36033351, 0.35743999, 0.35456002, 0.35170038, 0.34886793, 0.34607254,
                                    0.34330027, 0.34054471, 0.33779946, 0.33507898, 0.33238043, 0.32970476,
                                    0.32705262, 0.32442777, 0.32182658, 0.31924409, 0.31667437, 0.3141239,
                                    0.31158743, 0.30906764, 0.30656951, 0.304091, 0.30161526, 0.29914073,
                                    0.29666787, 0.29420467, 0.29174629, 0.2893013, 0.28686369, 0.28443399,
                                    0.28201774, 0.27960587, 0.27719983, 0.27479157, 0.27239127, 0.26999182,
                                    0.26759438, 0.26519979, 0.26279748, 0.2603949, 0.2579863, 0.25557387,
                                    0.25316093, 0.25074657, 0.24831902, 0.24588576, 0.24344758, 0.24100508,
                                    0.23856214, 0.23610692, 0.23363869, 0.23116566, 0.22868877, 0.22619582,
                                    0.22368266, 0.22114918, 0.21859367, 0.21601123, 0.2134057, 0.21078121,
                                    0.20814555, 0.20549482, 0.2028236, 0.20013763, 0.19742559, 0.19469036,
                                    0.19193251, 0.18915139, 0.18635706, 0.18354369, 0.1807156, 0.17787315,
                                    0.17501416, 0.17213219, 0.16923385, 0.16631956, 0.1633917, 0.16044723,
                                    0.15749085, 0.15453209, 0.15156888, 0.14861168, 0.14566561, 0.14273893,
                                    0.13985129, 0.13702258, 0.13427076, 0.13160852, 0.12905686, 0.12664359,
                                    0.12439398, 0.12233724, 0.12048795, 0.11885799, 0.11746714, 0.11630037,
                                    0.11536359, 0.11464695, 0.11414415, 0.11384795, 0.11369415, 0.11370366,
                                    0.11384328, 0.11409746, 0.11444349, 0.11489743, 0.11544143, 0.11604159,
                                    0.11666925, 0.11737261, 0.11815098, 0.11898482, 0.11989205])

rHeston_IE_hreset_duration = 5291.35  # not accurate timing!

rHeston_IE_hreset = np.array([0.514767  , 0.5111496 , 0.50754331, 0.5039442 , 0.5003525 , 0.49676591,
 0.49318745, 0.48962358, 0.48607066, 0.48252702, 0.47899439, 0.47546691,
 0.47195031, 0.46844647, 0.46495587, 0.46147793, 0.45801565, 0.45456822,
 0.45113152, 0.4477106 , 0.44430341, 0.44090176, 0.43751207, 0.43413165,
 0.43077142, 0.42742824, 0.42410967, 0.4208115 , 0.41754276, 0.41430415,
 0.41109573, 0.40791186, 0.4047476 , 0.40159828, 0.39847467, 0.39536783,
 0.39228477, 0.38924164, 0.38622695, 0.38322981, 0.38025636, 0.377313,
 0.37440343, 0.37152324, 0.3686618 , 0.3658221 , 0.36300276, 0.36020392,
 0.357429  , 0.35467344, 0.3519378 , 0.34920859, 0.34649297, 0.34379835,
 0.34112594, 0.33847463, 0.33583836, 0.33322366, 0.33063134, 0.32806586,
 0.32551837, 0.32299102, 0.3204739 , 0.31796757, 0.31546381, 0.31298622,
 0.31051895, 0.3080574 , 0.30559983, 0.3031576 , 0.30073438, 0.29831567,
 0.29590743, 0.29350142, 0.29110073, 0.28870465, 0.28631773, 0.2839313,
 0.2815456 , 0.27916357, 0.2767925 , 0.27441913, 0.27205431, 0.26968648,
 0.26731502, 0.26495499, 0.26259432, 0.26022747, 0.25785657, 0.25547901,
 0.25309478, 0.25070767, 0.24830325, 0.24589029, 0.24346226, 0.24102118,
 0.23856987, 0.23612382, 0.2336702 , 0.23120078, 0.22871958, 0.2262347,
 0.22373605, 0.221224  , 0.21869704, 0.2161518 , 0.21358311, 0.21099067,
 0.20837583, 0.205743  , 0.20308741, 0.20040902, 0.19771349, 0.19499459,
 0.19225035, 0.18948398, 0.18669049, 0.1838708 , 0.18103164, 0.17816503,
 0.17527472, 0.17236135, 0.1694317 , 0.16649249, 0.16354067, 0.16057995,
 0.15761077, 0.15460763, 0.1515962 , 0.14858218, 0.14558876, 0.1426077,
 0.13966715, 0.13677083, 0.13394321, 0.13119459, 0.12856524, 0.12608073,
 0.123761  , 0.12162382, 0.11970721, 0.11802231, 0.11658468, 0.11537473,
 0.11438427, 0.11360565, 0.11304639, 0.11266614, 0.11246199, 0.11240947,
 0.11248667, 0.11268007, 0.11297517, 0.11333829, 0.11376822, 0.11422018,
 0.114712  , 0.11523933, 0.11582222, 0.11644482, 0.11709012])

rHeston_IE_hreset_lower = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 2.62780847e-01, 2.78159594e-01, 2.85962282e-01,
 2.90997699e-01, 2.94567182e-01, 2.97216127e-01, 2.99197540e-01,
 3.00640005e-01, 3.01677525e-01, 3.02386807e-01, 3.02827721e-01,
 3.03054033e-01, 3.03085573e-01, 3.02951058e-01, 3.02640175e-01,
 3.02191385e-01, 3.01636486e-01, 3.00989243e-01, 3.00256775e-01,
 2.99435861e-01, 2.98546858e-01, 2.97597285e-01, 2.96600032e-01,
 2.95544442e-01, 2.94438882e-01, 2.93271935e-01, 2.92049317e-01,
 2.90763074e-01, 2.89452312e-01, 2.88096392e-01, 2.86692139e-01,
 2.85240494e-01, 2.83759924e-01, 2.82257332e-01, 2.80715562e-01,
 2.79144712e-01, 2.77536311e-01, 2.75896425e-01, 2.74226096e-01,
 2.72532635e-01, 2.70807095e-01, 2.69051488e-01, 2.67270884e-01,
 2.65475129e-01, 2.63649979e-01, 2.61809276e-01, 2.59940868e-01,
 2.58045350e-01, 2.56140725e-01, 2.54214366e-01, 2.52261166e-01,
 2.50284513e-01, 2.48282505e-01, 2.46256066e-01, 2.44210159e-01,
 2.42130080e-01, 2.40026109e-01, 2.37892059e-01, 2.35730867e-01,
 2.33546198e-01, 2.31354996e-01, 2.29144232e-01, 2.26905879e-01,
 2.24644714e-01, 2.22369657e-01, 2.20070717e-01, 2.17748744e-01,
 2.15402648e-01, 2.13029406e-01, 2.10624138e-01, 2.08186987e-01,
 2.05719738e-01, 2.03227276e-01, 2.00705073e-01, 1.98153420e-01,
 1.95578389e-01, 1.92973933e-01, 1.90338332e-01, 1.87675103e-01,
 1.84979435e-01, 1.82252512e-01, 1.79501337e-01, 1.76718052e-01,
 1.73906631e-01, 1.71067890e-01, 1.68208805e-01, 1.65336266e-01,
 1.62447331e-01, 1.59545819e-01, 1.56632266e-01, 1.53681132e-01,
 1.50718263e-01, 1.47749393e-01, 1.44797825e-01, 1.41855261e-01,
 1.38949885e-01, 1.36085377e-01, 1.33286149e-01, 1.30562397e-01,
 1.27954351e-01, 1.25487489e-01, 1.23181622e-01, 1.21054415e-01,
 1.19143841e-01, 1.17461011e-01, 1.16021490e-01, 1.14805564e-01,
 1.13804985e-01, 1.13012048e-01, 1.12434468e-01, 1.12031633e-01,
 1.11800750e-01, 1.11717217e-01, 1.11758895e-01, 1.11912201e-01,
 1.12162416e-01, 1.12474989e-01, 1.12848495e-01, 1.13236023e-01,
 1.13655150e-01, 1.14100528e-01, 1.14592463e-01, 1.15113729e-01,
 1.15644824e-01])

rHeston_IE_hreset_upper = np.array([0.57269805, 0.56857169, 0.5644529 , 0.56033957, 0.55623184, 0.55212847,
 0.54803108, 0.54394319, 0.53986285, 0.53578917, 0.53172311, 0.52766149,
 0.52360746, 0.51956209, 0.51552567, 0.51149794, 0.5074806 , 0.50347327,
 0.49947373, 0.49548484, 0.49150554, 0.48753127, 0.4835657 , 0.47960738,
 0.47566259, 0.47172968, 0.46781314, 0.46391071, 0.46002785, 0.4561652,
 0.4523231 , 0.44849853, 0.44468879, 0.44089132, 0.43711282, 0.43334815,
 0.42960185, 0.4258842 , 0.42218862, 0.41850873, 0.41484869, 0.41121297,
 0.40760444, 0.40402083, 0.40045576, 0.39691157, 0.3933878 , 0.38988495,
 0.38640577, 0.38294767, 0.37951148, 0.37608828, 0.37268331, 0.36930212,
 0.36594606, 0.36261487, 0.35930462, 0.35602049, 0.35276359, 0.34953776,
 0.34633695, 0.34316324, 0.34000947, 0.33687634, 0.33375767, 0.33067215,
 0.32760772, 0.3245609 , 0.3215304 , 0.31852555, 0.31554962, 0.31259095,
 0.30965453, 0.30673357, 0.30383063, 0.30094512, 0.29808093, 0.2952306,
 0.29239421, 0.28957423, 0.28677701, 0.28399076, 0.28122502, 0.2784693,
 0.27572282, 0.27299904, 0.27028686, 0.26758099, 0.26488308, 0.26219045,
 0.25950279, 0.25682332, 0.25413827, 0.25145546, 0.24876842, 0.2460787,
 0.24338863, 0.24071273, 0.23803849, 0.23535776, 0.23267407, 0.22999493,
 0.22731035, 0.22462038, 0.22192328, 0.21921549, 0.21649164, 0.21375115,
 0.21099505, 0.20822739, 0.20544322, 0.20264227, 0.19982987, 0.19699965,
 0.19414947, 0.19128226, 0.1883929 , 0.18548212, 0.18255638, 0.17960763,
 0.17663942, 0.17365224, 0.17065269, 0.16764734, 0.16463307, 0.16161347,
 0.15858891, 0.15553395, 0.15247409, 0.14941497, 0.14637972, 0.14336014,
 0.14038435, 0.13745614, 0.13460001, 0.13182636, 0.12917552, 0.12667313,
 0.12433925, 0.12219179, 0.12026874, 0.11858131, 0.11714501, 0.11594035,
 0.11495921, 0.11419391, 0.11365176, 0.11329263, 0.11311342, 0.11308977,
 0.11319986, 0.1134302 , 0.11376635, 0.11417535, 0.11465601, 0.11516534,
 0.11572109, 0.11631952, 0.1169801 , 0.11768774, 0.118427  ])

rHeston_IE_hreflection_duration = 3524.94

rHeston_IE_hreflection = np.array([0.41168636, 0.41006579, 0.40844579, 0.40681311, 0.40518776, 0.40355666,
 0.40191351, 0.4002574 , 0.39862827, 0.39700564, 0.39539422, 0.39376694,
 0.39215317, 0.39056331, 0.38898436, 0.38741214, 0.38583882, 0.38424862,
 0.38265685, 0.38105356, 0.3794424 , 0.37784743, 0.37627263, 0.37467966,
 0.3730859 , 0.37149549, 0.36990893, 0.36829758, 0.36668488, 0.3650653,
 0.36344812, 0.36182009, 0.3601841 , 0.35853555, 0.35687336, 0.35522132,
 0.35355799, 0.3518933 , 0.35023167, 0.34854534, 0.34686116, 0.34516252,
 0.34345783, 0.34176158, 0.34006364, 0.33837382, 0.33667795, 0.33498307,
 0.33329289, 0.33158753, 0.32987185, 0.32816403, 0.32645172, 0.32472641,
 0.32300151, 0.32128379, 0.31956571, 0.31784287, 0.31611128, 0.31436413,
 0.31259659, 0.31082787, 0.30906487, 0.30728929, 0.30550893, 0.30373346,
 0.30195542, 0.30016476, 0.29836847, 0.29657259, 0.29477725, 0.29296559,
 0.29112943, 0.28928746, 0.28743826, 0.28556943, 0.28367921, 0.28178203,
 0.27988794, 0.27798764, 0.27607738, 0.27415903, 0.27222637, 0.2702882,
 0.26833688, 0.26637218, 0.26439942, 0.26241662, 0.26042191, 0.2584209,
 0.25641333, 0.2544028 , 0.25238058, 0.25034726, 0.24829819, 0.24623945,
 0.24416837, 0.24208252, 0.23997186, 0.23784925, 0.23572714, 0.23360309,
 0.23147069, 0.22934417, 0.22721667, 0.22508009, 0.22294123, 0.22079936,
 0.21865562, 0.21650811, 0.21435536, 0.21220063, 0.21003606, 0.20786045,
 0.20569146, 0.20354262, 0.20140514, 0.19927622, 0.19716911, 0.19507451,
 0.1929931 , 0.19093503, 0.18890855, 0.18691496, 0.18495479, 0.18302496,
 0.18113864, 0.17929817, 0.17751055, 0.17577919, 0.17410841, 0.17249895,
 0.17094749, 0.16945999, 0.16806033, 0.16673976, 0.16550265, 0.1643602,
 0.16330863, 0.16235802, 0.16147946, 0.16069477, 0.16002409, 0.15945887,
 0.15900525, 0.15866449, 0.15842802, 0.1582839 , 0.15825727, 0.15835333,
 0.15854631, 0.15885964, 0.15928461, 0.1598063 , 0.16045177, 0.16122128,
 0.16209742, 0.16305267, 0.16414368, 0.16536235, 0.16667277])

rHeston_IE_hreflection_lower = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 2.12863820e-01, 2.38043741e-01, 2.48113302e-01,
 2.54253866e-01, 2.58506407e-01, 2.61620319e-01, 2.63912567e-01,
 2.65619891e-01, 2.66896633e-01, 2.67816967e-01, 2.68428507e-01,
 2.68793780e-01, 2.68959523e-01, 2.68953764e-01, 2.68773893e-01,
 2.68428694e-01, 2.67961917e-01, 2.67384324e-01, 2.66690344e-01,
 2.65887976e-01, 2.65004501e-01, 2.64059151e-01, 2.63045749e-01,
 2.61964945e-01, 2.60823718e-01, 2.59618906e-01, 2.58364727e-01,
 2.57055563e-01, 2.55694331e-01, 2.54289883e-01, 2.52842427e-01,
 2.51352141e-01, 2.49827308e-01, 2.48269418e-01, 2.46684061e-01,
 2.45063248e-01, 2.43409119e-01, 2.41718099e-01, 2.39997960e-01,
 2.38247051e-01, 2.36463904e-01, 2.34639028e-01, 2.32786959e-01,
 2.30921522e-01, 2.29040876e-01, 2.27139039e-01, 2.25231390e-01,
 2.23311398e-01, 2.21371303e-01, 2.19418637e-01, 2.17453171e-01,
 2.15476543e-01, 2.13487251e-01, 2.11484238e-01, 2.09471246e-01,
 2.07440643e-01, 2.05391625e-01, 2.03342410e-01, 2.01306976e-01,
 1.99276674e-01, 1.97248928e-01, 1.95237334e-01, 1.93232748e-01,
 1.91236040e-01, 1.89257646e-01, 1.87306027e-01, 1.85382655e-01,
 1.83488191e-01, 1.81619694e-01, 1.79790520e-01, 1.78003123e-01,
 1.76264627e-01, 1.74578550e-01, 1.72949314e-01, 1.71377730e-01,
 1.69860531e-01, 1.68403740e-01, 1.67031336e-01, 1.65734581e-01,
 1.64517864e-01, 1.63392435e-01, 1.62354485e-01, 1.61414120e-01,
 1.60542271e-01, 1.59760772e-01, 1.59089772e-01, 1.58520606e-01,
 1.58059378e-01, 1.57707244e-01, 1.57455482e-01, 1.57291913e-01,
 1.57241723e-01, 1.57310073e-01, 1.57470816e-01, 1.57747528e-01,
 1.58131338e-01, 1.58607012e-01, 1.59202035e-01, 1.59916795e-01,
 1.60733724e-01, 1.61624745e-01, 1.62648096e-01, 1.63796073e-01,
 1.65032208e-01])

rHeston_IE_hreflection_upper = np.array([0.55062949, 0.54671313, 0.54280455, 0.53890235, 0.53500914, 0.53112351,
 0.52724479, 0.52337293, 0.5195138 , 0.51566499, 0.51182762, 0.50799782,
 0.50418052, 0.5003781 , 0.496589  , 0.49281304, 0.48904924, 0.48529486,
 0.48155312, 0.47782231, 0.47410342, 0.47040218, 0.46672038, 0.46304966,
 0.45939446, 0.45575635, 0.45213607, 0.44852632, 0.4449337 , 0.44135708,
 0.43779967, 0.43425791, 0.430733  , 0.42722372, 0.42372988, 0.42026012,
 0.41680746, 0.41337585, 0.40996756, 0.40657212, 0.40320032, 0.39984576,
 0.39651214, 0.39320628, 0.38992435, 0.38667145, 0.3834415 , 0.38023829,
 0.37706421, 0.3739095 , 0.37077661, 0.36767554, 0.36460003, 0.36154535,
 0.35851923, 0.35552603, 0.35256165, 0.34962355, 0.34670918, 0.34381392,
 0.3409341 , 0.33808204, 0.33526248, 0.33246283, 0.32968817, 0.32694525,
 0.32422867, 0.3215307 , 0.31885602, 0.31620891, 0.31358923, 0.31098343,
 0.3083842 , 0.3058056 , 0.30324598, 0.30069455, 0.29814895, 0.29562041,
 0.29311695, 0.29063022, 0.28815643, 0.28569653, 0.28324446, 0.28080729,
 0.27837761, 0.27595452, 0.27354217, 0.27113815, 0.26874013, 0.26635265,
 0.26397492, 0.26160977, 0.25924855, 0.25689122, 0.25453284, 0.25217855,
 0.24982532, 0.24747029, 0.24510331, 0.24273614, 0.24038029, 0.23803305,
 0.23568784, 0.23335806, 0.23103676, 0.2287157 , 0.22640111, 0.22409199,
 0.22178911, 0.2194903 , 0.2171938 , 0.21490255, 0.21260847, 0.21031013,
 0.20802467, 0.20576526, 0.20352302, 0.20129498, 0.19909408, 0.19691093,
 0.19474602, 0.19260927, 0.19050875, 0.18844562, 0.18642026, 0.1844295,
 0.18248635, 0.18059304, 0.17875643, 0.17697986, 0.17526756, 0.17362019,
 0.1720344 , 0.17051608, 0.16908901, 0.16774444, 0.1664867 , 0.16532697,
 0.16426147, 0.16330027, 0.16241459, 0.16162626, 0.16095539, 0.16039351,
 0.15994682, 0.15961664, 0.15939453, 0.1592688,  0.15926449, 0.15938682,
 0.15961036, 0.15995836, 0.16042223, 0.16098728, 0.16168016, 0.16250092,
 0.16343229, 0.16444718, 0.1656007 , 0.16688438, 0.16826272])

rHeston_IE_splitkernel_duration = 4080.4631821

rHeston_IE_splitkernel = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 2.84541748e-01, 3.16515347e-01, 3.26990903e-01,
 3.33031782e-01, 3.37152593e-01, 3.40121710e-01, 3.42319907e-01,
 3.43940818e-01, 3.45185914e-01, 3.46038775e-01, 3.46586762e-01,
 3.46882703e-01, 3.46983294e-01, 3.46926729e-01, 3.46730182e-01,
 3.46465997e-01, 3.46094657e-01, 3.45630724e-01, 3.45105119e-01,
 3.44502304e-01, 3.43809128e-01, 3.43070123e-01, 3.42253623e-01,
 3.41395184e-01, 3.40475819e-01, 3.39503243e-01, 3.38487624e-01,
 3.37435236e-01, 3.36346235e-01, 3.35223805e-01, 3.34063296e-01,
 3.32844700e-01, 3.31580776e-01, 3.30300227e-01, 3.28983264e-01,
 3.27638001e-01, 3.26263684e-01, 3.24848429e-01, 3.23394105e-01,
 3.21909573e-01, 3.20419626e-01, 3.18919433e-01, 3.17391191e-01,
 3.15834818e-01, 3.14273277e-01, 3.12696365e-01, 3.11102226e-01,
 3.09486547e-01, 3.07860351e-01, 3.06222144e-01, 3.04575104e-01,
 3.02924695e-01, 3.01276493e-01, 2.99608813e-01, 2.97917415e-01,
 2.96200667e-01, 2.94473452e-01, 2.92732345e-01, 2.90982863e-01,
 2.89211756e-01, 2.87427609e-01, 2.85635007e-01, 2.83821245e-01,
 2.81994797e-01, 2.80155390e-01, 2.78307496e-01, 2.76452186e-01,
 2.74595772e-01, 2.72722466e-01, 2.70847426e-01, 2.68945655e-01,
 2.67021784e-01, 2.65096954e-01, 2.63167809e-01, 2.61214518e-01,
 2.59239969e-01, 2.57255669e-01, 2.55257768e-01, 2.53248767e-01,
 2.51225246e-01, 2.49191577e-01, 2.47137972e-01, 2.45054828e-01,
 2.42940532e-01, 2.40812603e-01, 2.38664749e-01, 2.36499038e-01,
 2.34323688e-01, 2.32135301e-01, 2.29934647e-01, 2.27718808e-01,
 2.25481454e-01, 2.23220157e-01, 2.20934061e-01, 2.18624792e-01,
 2.16310090e-01, 2.13968263e-01, 2.11606580e-01, 2.09217136e-01,
 2.06804336e-01, 2.04374197e-01, 2.01919731e-01, 1.99441781e-01,
 1.96946983e-01, 1.94442246e-01, 1.91914332e-01, 1.89369183e-01,
 1.86806974e-01, 1.84216857e-01, 1.81601408e-01, 1.78974327e-01,
 1.76325245e-01, 1.73656605e-01, 1.70980215e-01, 1.68289066e-01,
 1.65582721e-01, 1.62866706e-01, 1.60143608e-01, 1.57425621e-01,
 1.54709487e-01, 1.51988403e-01, 1.49266600e-01, 1.46573613e-01,
 1.43916334e-01, 1.41303937e-01, 1.38749816e-01, 1.36268522e-01,
 1.33872371e-01, 1.31574076e-01, 1.29392051e-01, 1.27346544e-01,
 1.25430661e-01, 1.23676008e-01, 1.22074373e-01, 1.20643891e-01,
 1.19424697e-01, 1.18412310e-01, 1.17583240e-01, 1.16902974e-01,
 1.16371294e-01, 1.15990268e-01, 1.15770552e-01, 1.15697673e-01,
 1.15710087e-01, 1.15828593e-01, 1.16049128e-01, 1.16351961e-01,
 1.16695593e-01, 1.17102550e-01, 1.17535544e-01, 1.18034179e-01,
 1.18600963e-01])

rHeston_IE_splitkernel_lower = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 2.24376546e-01,
 2.37659321e-01, 2.44958541e-01, 2.49729351e-01, 2.53063281e-01,
 2.55453983e-01, 2.57203753e-01, 2.58462291e-01, 2.59344581e-01,
 2.59898713e-01, 2.60194411e-01, 2.60278598e-01, 2.60161708e-01,
 2.59881879e-01, 2.59458125e-01, 2.58912560e-01, 2.58259370e-01,
 2.57517502e-01, 2.56675118e-01, 2.55759938e-01, 2.54747248e-01,
 2.53649772e-01, 2.52498829e-01, 2.51294521e-01, 2.50017419e-01,
 2.48675091e-01, 2.47284358e-01, 2.45843703e-01, 2.44358627e-01,
 2.42827681e-01, 2.41257937e-01, 2.39640576e-01, 2.37967110e-01,
 2.36237787e-01, 2.34473301e-01, 2.32668272e-01, 2.30826254e-01,
 2.28957172e-01, 2.27058444e-01, 2.25131851e-01, 2.23175234e-01,
 2.21182853e-01, 2.19153050e-01, 2.17085761e-01, 2.14983438e-01,
 2.12865088e-01, 2.10708900e-01, 2.08522977e-01, 2.06299762e-01,
 2.04044315e-01, 2.01763267e-01, 1.99449925e-01, 1.97105575e-01,
 1.94737359e-01, 1.92352635e-01, 1.89938285e-01, 1.87500653e-01,
 1.85040201e-01, 1.82546240e-01, 1.80021646e-01, 1.77480491e-01,
 1.74912529e-01, 1.72320431e-01, 1.69716273e-01, 1.67093163e-01,
 1.64450815e-01, 1.61794927e-01, 1.59128214e-01, 1.56463031e-01,
 1.53796185e-01, 1.51120919e-01, 1.48441522e-01, 1.45787659e-01,
 1.43166249e-01, 1.40586479e-01, 1.38061744e-01, 1.35606584e-01,
 1.33233279e-01, 1.30954480e-01, 1.28788548e-01, 1.26755685e-01,
 1.24848813e-01, 1.23099512e-01, 1.21499352e-01, 1.20066367e-01,
 1.18840856e-01, 1.17818327e-01, 1.16975109e-01, 1.16276267e-01,
 1.15721386e-01, 1.15312477e-01, 1.15060392e-01, 1.14950635e-01,
 1.14920302e-01, 1.14990437e-01, 1.15156929e-01, 1.15399418e-01,
 1.15674420e-01, 1.16004670e-01, 1.16350219e-01, 1.16751870e-01,
 1.17212355e-01])

rHeston_IE_splitkernel_upper = np.array([0.53444051, 0.53061535, 0.52680042, 0.5229946 , 0.51919597, 0.51540809,
 0.5116297 , 0.50786066, 0.50410268, 0.50035576, 0.49662301, 0.49290623,
 0.48919763, 0.48550236, 0.48182003, 0.47815084, 0.47449348, 0.47085323,
 0.46722191, 0.46360057, 0.45998923, 0.45639003, 0.4528046 , 0.44923279,
 0.44568351, 0.44215   , 0.43863319, 0.43513763, 0.43165979, 0.4281958,
 0.42475406, 0.42132686, 0.41792171, 0.41453342, 0.41116313, 0.40781304,
 0.40448458, 0.40117754, 0.39789267, 0.3946281 , 0.39137612, 0.3881401,
 0.38493022, 0.38173874, 0.37856838, 0.37541842, 0.37228313, 0.36916233,
 0.36605926, 0.36298535, 0.35993856, 0.35690981, 0.35389838, 0.35091608,
 0.34795739, 0.34502101, 0.34210401, 0.33921243, 0.33634516, 0.33350393,
 0.33069206, 0.3279132 , 0.32515297, 0.32240776, 0.31967549, 0.3169657,
 0.31427544, 0.31160814, 0.30895348, 0.30631701, 0.30370158, 0.3010968,
 0.2985084 , 0.29593547, 0.2933809 , 0.29084499, 0.28833238, 0.28582947,
 0.28334819, 0.28086678, 0.27838815, 0.27592969, 0.27348801, 0.27104507,
 0.26860248, 0.2661697 , 0.2637426 , 0.26132279, 0.25890649, 0.25649705,
 0.25408484, 0.25166023, 0.24922089, 0.24678241, 0.24433832, 0.24188995,
 0.23944454, 0.23699838, 0.23455172, 0.2321013 , 0.22964058, 0.2271667,
 0.22467833, 0.22217657, 0.21967818, 0.21716174, 0.21463391, 0.21208657,
 0.20952366, 0.20695071, 0.20436054, 0.20175367, 0.19913632, 0.19651504,
 0.19387652, 0.19122638, 0.18856455, 0.18588009, 0.1831753 , 0.18046357,
 0.17773442, 0.1749901 , 0.17224218, 0.16948356, 0.16671366, 0.16393787,
 0.16115864, 0.15838803, 0.15562273, 0.1528559 , 0.1500917 , 0.14735957,
 0.14466636, 0.14202125, 0.13943762, 0.13693004, 0.13451087, 0.13219286,
 0.12999449, 0.12793605, 0.12601081, 0.1242504 , 0.1226468 , 0.12121823,
 0.12000465, 0.11900156, 0.11818561, 0.11752267, 0.11701264, 0.11665761,
 0.11646796, 0.11642916, 0.11648087, 0.11664351, 0.11691294, 0.11726983,
 0.11767426, 0.11814831, 0.11865669, 0.11923754, 0.11989274])

rHeston_IE_splitthrow_duration = 3736.1708509

rHeston_IE_splitthrow = np.array([0.48987684, 0.48653616, 0.48320603, 0.47987969, 0.47656713, 0.473269,
 0.46998549, 0.46671768, 0.46346858, 0.46023417, 0.45700789, 0.45378783,
 0.45058109, 0.4474022 , 0.44424854, 0.44110748, 0.4379942 , 0.434905,
 0.43183492, 0.42878052, 0.42575355, 0.42273861, 0.41974673, 0.41678369,
 0.413833  , 0.41089381, 0.40797969, 0.40508591, 0.40220481, 0.39934733,
 0.39651116, 0.39370102, 0.39091497, 0.38814861, 0.38540306, 0.38268416,
 0.37997224, 0.37727714, 0.37460828, 0.37197301, 0.36937766, 0.36680084,
 0.36424095, 0.36169889, 0.3591719 , 0.35667271, 0.35420639, 0.35175337,
 0.3493105 , 0.34689134, 0.34449285, 0.34210637, 0.33973479, 0.33738147,
 0.33505062, 0.33274084, 0.33044712, 0.32816213, 0.32589091, 0.32363001,
 0.32137946, 0.31913572, 0.31689559, 0.31466698, 0.31245583, 0.31025044,
 0.30804198, 0.30582817, 0.30362059, 0.30140315, 0.29918486, 0.29697019,
 0.29476349, 0.29256586, 0.29037482, 0.28818978, 0.28600181, 0.28381136,
 0.28162043, 0.2794288 , 0.27723427, 0.27503602, 0.27284051, 0.27064651,
 0.26844142, 0.26622319, 0.26399694, 0.26175886, 0.2595176 , 0.25726698,
 0.25502358, 0.25277306, 0.25051865, 0.24825998, 0.24598489, 0.24369909,
 0.24140217, 0.23908521, 0.23675516, 0.23440182, 0.23202815, 0.22964313,
 0.22725137, 0.22484207, 0.22242014, 0.21999059, 0.21754698, 0.21508837,
 0.21262084, 0.21013436, 0.20763267, 0.20511762, 0.20258335, 0.2000247,
 0.19744706, 0.194855  , 0.19223385, 0.1895867 , 0.18691653, 0.18422822,
 0.18153031, 0.17882225, 0.17610314, 0.17337335, 0.17063112, 0.16788696,
 0.16514357, 0.16239638, 0.15964769, 0.15690138, 0.15416523, 0.15144433,
 0.14873566, 0.14606679, 0.14344652, 0.14089863, 0.13844275, 0.13608263,
 0.13383399, 0.13172606, 0.12976154, 0.12794663, 0.12628707, 0.12480174,
 0.12349038, 0.1223648 , 0.1214387 , 0.12070035, 0.12010771, 0.1196379,
 0.11930242, 0.11913479, 0.11907262, 0.11907147, 0.11917568, 0.11938784,
 0.11968616, 0.12004598, 0.12042065, 0.12085729, 0.12135859])

rHeston_IE_splitthrow_lower = np.array([1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 1.36379788e-10,
 1.36379788e-10, 1.36379788e-10, 1.36379788e-10, 2.32442845e-01,
 2.56490675e-01, 2.66315104e-01, 2.72347802e-01, 2.76469730e-01,
 2.79439644e-01, 2.81635690e-01, 2.83278358e-01, 2.84493355e-01,
 2.85357479e-01, 2.85920193e-01, 2.86243463e-01, 2.86357761e-01,
 2.86293303e-01, 2.86067677e-01, 2.85695576e-01, 2.85207269e-01,
 2.84625174e-01, 2.83941260e-01, 2.83152454e-01, 2.82265311e-01,
 2.81305052e-01, 2.80255649e-01, 2.79136461e-01, 2.77958915e-01,
 2.76733192e-01, 2.75464459e-01, 2.74152852e-01, 2.72800768e-01,
 2.71400136e-01, 2.69954552e-01, 2.68469206e-01, 2.66946279e-01,
 2.65385430e-01, 2.63787843e-01, 2.62162927e-01, 2.60510923e-01,
 2.58819152e-01, 2.57087091e-01, 2.55322148e-01, 2.53521524e-01,
 2.51696135e-01, 2.49840357e-01, 2.47973338e-01, 2.46080428e-01,
 2.44166087e-01, 2.42230796e-01, 2.40262482e-01, 2.38268132e-01,
 2.36248120e-01, 2.34193841e-01, 2.32113395e-01, 2.29996802e-01,
 2.27847888e-01, 2.25676621e-01, 2.23488294e-01, 2.21272268e-01,
 2.19034104e-01, 2.16779400e-01, 2.14501986e-01, 2.12201289e-01,
 2.09883950e-01, 2.07540066e-01, 2.05173834e-01, 2.02787473e-01,
 2.00375334e-01, 1.97932516e-01, 1.95464763e-01, 1.92977003e-01,
 1.90454634e-01, 1.87901045e-01, 1.85319496e-01, 1.82715116e-01,
 1.80096694e-01, 1.77463861e-01, 1.74815852e-01, 1.72153187e-01,
 1.69474239e-01, 1.66789676e-01, 1.64102333e-01, 1.61407705e-01,
 1.58708173e-01, 1.56007692e-01, 1.53314116e-01, 1.50632584e-01,
 1.47960055e-01, 1.45324165e-01, 1.42733708e-01, 1.40212468e-01,
 1.37780059e-01, 1.35440148e-01, 1.33208416e-01, 1.31114068e-01,
 1.29159697e-01, 1.27351408e-01, 1.25694812e-01, 1.24208755e-01,
 1.22892850e-01, 1.21758900e-01, 1.20820710e-01, 1.20066562e-01,
 1.19453987e-01, 1.18959672e-01, 1.18595102e-01, 1.18394540e-01,
 1.18294642e-01, 1.18249635e-01, 1.18304648e-01, 1.18462610e-01,
 1.18701301e-01, 1.18995213e-01, 1.19295071e-01, 1.19649747e-01,
 1.20062473e-01])

rHeston_IE_splitthrow_upper = np.array([0.562767  , 0.55872932, 0.55469813, 0.55067059, 0.54665097, 0.54263959,
 0.53863659, 0.5346425 , 0.5306587 , 0.52668353, 0.52271418, 0.51874986,
 0.51479374, 0.51085238, 0.50692479, 0.50300548, 0.49910142, 0.49521117,
 0.49133266, 0.48746449, 0.48361229, 0.47976913, 0.47594037, 0.47212905,
 0.46832759, 0.46453575, 0.46076036, 0.45699944, 0.45324948, 0.44951621,
 0.44579885, 0.44210019, 0.43841969, 0.43475549, 0.43110859, 0.42748255,
 0.42386727, 0.42026835, 0.41669142, 0.41314121, 0.40962206, 0.40612262,
 0.40264247, 0.39918264, 0.395742  , 0.39232869, 0.38894658, 0.38558445,
 0.38224079, 0.3789246 , 0.37563461, 0.37236584, 0.36912063, 0.36590171,
 0.36271252, 0.3595528 , 0.35641971, 0.3533086 , 0.35022337, 0.34716195,
 0.34412472, 0.34110935, 0.33811366, 0.33514373, 0.33220439, 0.32928703,
 0.32638484, 0.32349592, 0.32062928, 0.31777205, 0.31493121, 0.3121103,
 0.30931289, 0.30653992, 0.30378934, 0.30106061, 0.29834602, 0.29564569,
 0.29296109, 0.2902918 , 0.28763561, 0.28499149, 0.2823649 , 0.2797545,
 0.27714865, 0.27454503, 0.27194786, 0.26935322, 0.26676866, 0.26418812,
 0.26162663, 0.25907058, 0.25652264, 0.2539821 , 0.25143713, 0.2488927,
 0.24634803, 0.24379421, 0.24123746, 0.2386676 , 0.23608706, 0.2335041,
 0.23092287, 0.22833261, 0.22573776, 0.22314288, 0.22054145, 0.21793222,
 0.21532091, 0.21269739, 0.21006508, 0.20742554, 0.20477276, 0.20210142,
 0.19941656, 0.1967225 , 0.19400453, 0.19126546, 0.18850808, 0.18573702,
 0.18296059, 0.18017812, 0.17738857, 0.17459218, 0.1717871 , 0.16898365,
 0.16618446, 0.16338489, 0.16058716, 0.15779508, 0.15501636, 0.15225608,
 0.14951121, 0.14680927, 0.14415907, 0.1415844 , 0.13910487, 0.13672432,
 0.13445854, 0.13233675, 0.13036174, 0.12853982, 0.12687684, 0.12539169,
 0.12408422, 0.12296624, 0.12205128, 0.12132761, 0.12075354, 0.12030658,
 0.11999816, 0.11986103, 0.11983364, 0.11987268, 0.12002165, 0.12028267,
 0.12063421, 0.12105219, 0.12149208, 0.12199913, 0.12257528])


def plot_rHeston_IE_smiles():
    """
    Plots the smiles for the implicit Euler scheme applied to the rough Heston model.
    """
    plt.plot(k_rHeston, rHeston_6, 'k-', label='Fourier inversion')
    plt.plot(k_rHeston_IE, rHeston_IE_bouncy, 'b-', label='Implicit Euler - bouncy')
    plt.plot(k_rHeston_IE, rHeston_IE_sticky, 'r-', label='Implicit Euler - sticky')
    plt.plot(k_rHeston, rHeston_IE_hreset, 'y-', label='Implicit Euler - hyperplane reset')
    plt.plot(k_rHeston, rHeston_IE_hreflection, 'g-', label='Implicit Euler - hyperplane reflection')
    plt.plot(k_rHeston, rHeston_IE_splitkernel, 'c-', label='Implicit Euler - split kernel')
    plt.plot(k_rHeston, rHeston_IE_splitthrow, 'orange', label='Implicit Euler - split throw')
    plt.plot(k_rHeston_IE, rHeston_IE_sticky_lower, 'r--')
    plt.plot(k_rHeston_IE, rHeston_IE_sticky_upper, 'r--')
    plt.legend(loc='upper right')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.show()


'''
High mean-reversion psis in the split kernel rHeston IE scheme. Uses parameters
H=0.1, nu=0.3, theta=0.02, lambda=0.3, T=1 (globally), N=6, throws away nodes smaller than 100,
T=10/min(nodes) (locally).
rHeston_psi_n is the first and second moment of the psi with n equidistant time intervals in the Riccati equation.
rHeston_psi_adaptive_q is similar. 
'''

rHeston_psi_1048576 = np.array([-4.786493744137755e-05+0.009001370091122604j, -0.23626350731140552-0.0010043112607593682j])

n_Riccati_vec = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
rHeston_psi_n_Riccati_vec = np.array([[-4.97984846e-06+4.13759872e-03j, -1.69728027e-05+8.19854819e-03j,
  -2.67722020e-05+8.85185258e-03j, -3.14222142e-05+8.96669591e-03j,
  -3.27480190e-05+8.99105866e-03j, -3.36237016e-05+8.99764251e-03j,
  -3.57348529e-05+8.99985661e-03j, -3.91242318e-05+9.00071119e-03j,
  -4.14544957e-05+9.00108062e-03j, -4.21681839e-05+9.00126302e-03j,
  -4.24481025e-05+9.00135352e-03j, -4.31611972e-05+9.00138998e-03j,
  -4.46116863e-05+9.00139203e-03j, -4.59265580e-05+9.00138435e-03j,
  -4.64062272e-05+9.00138401e-03j, -4.65193974e-05+9.00138616e-03j,
  -4.66949162e-05+9.00138550e-03j],
 [-2.67446315e-02-3.21888162e-05j, -9.17841760e-02-1.88750894e-04j,
  -1.34479253e-01-3.99321604e-04j, -1.55717768e-01-5.25300169e-04j,
  -1.61840399e-01-5.51411685e-04j, -1.66046185e-01-5.67290500e-04j,
  -1.76426605e-01-6.26977659e-04j, -1.93138848e-01-7.35288838e-04j,
  -2.04632477e-01-8.09562944e-04j, -2.08151435e-01-8.23443999e-04j,
  -2.09531313e-01-8.25061652e-04j, -2.13049870e-01-8.45837544e-04j,
  -2.20208094e-01-8.95717412e-04j, -2.26697199e-01-9.42053001e-04j,
  -2.29064417e-01-9.56543503e-04j, -2.29622897e-01-9.58065175e-04j,
  -2.30489112e-01-9.63093147e-04j]])

rHeston_psi_adaptive_1024 = np.array([-4.808868651379933e-05+0.009002184905100123j, -0.2373427877401934-0.0010120619893089458j])

q_Riccati_vec_adaptive = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
rHeston_psi_adaptive_q_Riccati_vec = np.array([[-5.27872583e-05+0.00984926j, -5.02510598e-05+0.00942111j,
  -4.90599705e-05+0.00921081j, -4.85359999e-05+0.00910605j,
  -4.83002287e-05+0.00905371j, -4.81900600e-05+0.00902754j,
  -4.81370591e-05+0.00901445j, -4.81110994e-05+0.00900791j,
  -4.80982573e-05+0.00900464j],
 [-2.38052642e-01-0.00103711j, -2.36091327e-01-0.0010199j,
  -2.36001795e-01-0.00101363j, -2.36440742e-01-0.00101212j,
  -2.36829055e-01-0.00101189j, -2.37072600e-01-0.00101193j,
  -2.37207704e-01-0.00101199j, -2.37278717e-01-0.00101202j,
  -2.37315106e-01-0.00101204j]])

'''
The rHeston implied volatility smiles for European call options. Parameters used are
lambda_=0.3, rho=-0.7, nu=0.3, theta=0.02, V_0=0.02, S_0=1.
The vector of log-strikes is given by k_T, where T/100 is the corresponding maturity.
rHeston_T is the Heston smile for the El Euch-Rosenbaum approximation (with log-strikes k_vec) and maturity T/100.
rHeston_N_T is the Heston smile using the optimal quadrature rule with N points and maturity T/100.
Estimated upper bounds for the relative discretization errors are specified for the rHeston_T arrays, they are generally 
similar for the rHeston_N_T arrays.
'''

k_100 = np.linspace(-1.5, 0.75, 201)

rHeston_100 = np.array([0.42919066 ,0.42770626, 0.42621689, 0.42472251, 0.42322305, 0.42171848,
 0.42020874, 0.41869376, 0.41717351, 0.41564792, 0.41411695, 0.41258052,
 0.41103859, 0.4094911 , 0.40793798, 0.40637918, 0.40481464, 0.40324428,
 0.40166806, 0.40008589, 0.39849772, 0.39690348, 0.39530309, 0.3936965,
 0.39208361, 0.39046437, 0.3888387 , 0.38720652, 0.38556776, 0.38392233,
 0.38227016, 0.38061116, 0.37894525, 0.37727235, 0.37559236, 0.3739052,
 0.37221077, 0.37050899, 0.36879975, 0.36708297, 0.36535854, 0.36362637,
 0.36188634, 0.36013836, 0.35838231, 0.35661809, 0.35484557, 0.35306466,
 0.35127522, 0.34947715, 0.3476703 , 0.34585457, 0.34402981, 0.3421959,
 0.34035271, 0.33850008, 0.33663788, 0.33476596, 0.33288418, 0.33099238,
 0.3290904 , 0.32717808, 0.32525525, 0.32332175, 0.3213774 , 0.31942203,
 0.31745544, 0.31547745, 0.31348787, 0.31148649, 0.30947311, 0.30744752,
 0.30540951, 0.30335885, 0.3012953 , 0.29921865, 0.29712863, 0.29502501,
 0.29290752, 0.2907759 , 0.28862987, 0.28646916, 0.28429347, 0.2821025,
 0.27989594, 0.27767348, 0.27543478, 0.27317951, 0.27090731, 0.26861782,
 0.26631067, 0.26398547, 0.26164182, 0.25927931, 0.2568975 , 0.25449598,
 0.25207426, 0.2496319 , 0.24716839, 0.24468325, 0.24217594, 0.23964594,
 0.23709269, 0.23451564, 0.23191418, 0.22928772, 0.22663564, 0.22395731,
 0.22125207, 0.21851927, 0.21575822, 0.21296825, 0.21014867, 0.20729879,
 0.20441793, 0.20150541, 0.19856059, 0.19558285, 0.19257164, 0.18952646,
 0.18644694, 0.18333281, 0.18018399, 0.17700064, 0.17378321, 0.17053255,
 0.16724999, 0.16393755, 0.16059806, 0.15723543, 0.15385493, 0.15046357,
 0.14707053, 0.1436877 , 0.14033026, 0.1370173 , 0.13377238, 0.13062383,
 0.12760459, 0.12475132, 0.12210242, 0.11969498, 0.11756099, 0.11572353,
 0.11419415, 0.11297202, 0.11204521, 0.11139326, 0.11099052, 0.11080901,
 0.11082082, 0.11099958, 0.11132132, 0.11176478, 0.11231148, 0.11294549,
 0.11365322, 0.11442309, 0.1152453 , 0.11611153, 0.11701471, 0.11794888,
 0.11890893, 0.11989054, 0.12089001, 0.12190419, 0.12293037, 0.12396623,
 0.12500978, 0.1260593 , 0.12711331, 0.12817053, 0.12922986, 0.13029033,
 0.13135113, 0.13241152, 0.1334709 , 0.13452871, 0.13558449, 0.13663784,
 0.1376884 , 0.13873587, 0.13977998, 0.14082053, 0.1418573 , 0.14289013,
 0.14391891, 0.14494348, 0.14596375, 0.14697967, 0.14799116, 0.14899811,
 0.15000056, 0.15099846, 0.15199173, 0.15298039, 0.15396451, 0.15494396,
 0.15591876, 0.15688908, 0.15785478])

rHeston_100_errors = np.array([1.92122011e-05, 1.92283548e-05, 1.88950598e-05 ,1.89205985e-05,
 1.92245360e-05, 1.91992669e-05, 1.89477442e-05, 1.89999684e-05,
 1.92404792e-05, 1.91967907e-05, 1.89973612e-05, 1.90703989e-05,
 1.92672635e-05, 1.92023713e-05, 1.90486062e-05, 1.91442328e-05,
 1.92946097e-05, 1.92151156e-05, 1.91119121e-05, 1.92153159e-05,
 1.93194668e-05, 1.92438135e-05, 1.91833088e-05, 1.92785947e-05,
 1.93512190e-05, 1.92891030e-05, 1.92546686e-05, 1.93403420e-05,
 1.93953696e-05, 1.93435592e-05 ,1.93268554e-05, 1.94078470e-05,
 1.94490060e-05, 1.94041044e-05, 1.94052731e-05, 1.94813539e-05,
 1.95084634e-05, 1.94743499e-05, 1.94918063e-05, 1.95587843e-05,
 1.95770161e-05, 1.95572202e-05, 1.95843037e-05, 1.96421851e-05,
 1.96584397e-05, 1.96504141e-05, 1.96823413e-05, 1.97354603e-05,
 1.97523144e-05, 1.97531745e-05, 1.97901186e-05, 1.98407914e-05,
 1.98580475e-05, 1.98673017e-05, 1.99101197e-05, 1.99580438e-05,
 1.99768052e-05, 1.99961384e-05, 2.00431667e-05, 2.00889200e-05,
 2.01123862e-05, 2.01405420e-05, 2.01904844e-05, 2.02367382e-05,
 2.02661213e-05, 2.03020297e-05, 2.03553717e-05, 2.04043505e-05,
 2.04403137e-05, 2.04835851e-05, 2.05418482e-05, 2.05945929e-05,
 2.06372037e-05, 2.06891643e-05, 2.07531029e-05, 2.08105681e-05,
 2.08616939e-05, 2.09231404e-05, 2.09931698e-05, 2.10572356e-05,
 2.11187643e-05, 2.11899204e-05, 2.12676290e-05, 2.13409864e-05,
 2.14138254e-05, 2.14961276e-05, 2.15842000e-05, 2.16689050e-05,
 2.17547680e-05, 2.18505318e-05, 2.19513103e-05, 2.20498458e-05,
 2.21516943e-05, 2.22634573e-05, 2.23800887e-05, 2.24960728e-05,
 2.26172320e-05, 2.27490355e-05, 2.28854008e-05, 2.30232772e-05,
 2.31691069e-05, 2.33253574e-05, 2.34872449e-05, 2.36535074e-05,
 2.38294222e-05, 2.40170235e-05, 2.42120303e-05, 2.44145167e-05,
 2.46291740e-05, 2.48579406e-05, 2.50967976e-05, 2.53463138e-05,
 2.56122941e-05, 2.58959612e-05, 2.61929296e-05, 2.65058006e-05,
 2.68410074e-05, 2.71975782e-05, 2.75741222e-05, 2.79737824e-05,
 2.84022617e-05, 2.88604995e-05, 2.93462777e-05, 2.98650257e-05,
 3.04225535e-05, 3.10195301e-05, 3.16553110e-05, 3.23353104e-05,
 3.30661377e-05, 3.38459398e-05, 3.46723619e-05, 3.55505706e-05,
 3.64805808e-05, 3.74503365e-05, 3.84484883e-05, 3.94657575e-05,
 4.04763822e-05, 4.14392136e-05, 4.23125061e-05, 4.30522177e-05,
 4.35988673e-05, 4.38964700e-05, 4.39147011e-05, 4.36531559e-05,
 4.31170365e-05, 4.23378822e-05, 4.13860931e-05, 4.03245674e-05,
 3.91879845e-05, 3.80210097e-05, 3.68811234e-05, 3.57873581e-05,
 3.47257503e-05, 3.37175687e-05, 3.27987112e-05, 3.19439849e-05,
 3.11208831e-05, 3.03614543e-05, 2.96940622e-05, 2.90677296e-05,
 2.84441732e-05, 2.78833138e-05, 2.74171022e-05, 2.69571525e-05,
 2.64622376e-05, 2.60457194e-05, 2.57424770e-05 ,2.53929658e-05,
 2.49580742e-05, 2.46585108e-05, 2.45101402e-05, 2.42057850e-05,
 2.37692055e-05, 2.36025453e-05, 2.36102381e-05, 2.32722463e-05,
 2.27855272e-05, 2.27989040e-05, 2.29771617e-05, 2.25092853e-05,
 2.19089774e-05, 2.21956359e-05, 2.26164775e-05, 2.18427746e-05,
 2.09927628e-05, 2.18227604e-05, 2.26242951e-05 ,2.10839072e-05,
 1.98623151e-05, 2.19036617e-05, 2.30814052e-05, 1.98582440e-05,
 1.84561257e-05, 2.28255381e-05, 2.39425541e-05, 1.77251729e-05,
 1.67678594e-05, 2.50190931e-05, 2.53409919e-05, 1.40520629e-05,
 1.43459013e-05])

rHeston_1_100 = np.array([0.43300045, 0.43150788, 0.4300102,  0.42850718, 0.42699872, 0.42548492,
 0.42396576, 0.42244103, 0.42091068, 0.41937475 ,0.41783319, 0.41628583,
 0.41473262, 0.41317359, 0.41160866, 0.41003768, 0.4084606 , 0.40687743,
 0.40528807, 0.4036924 , 0.40209036, 0.40048194, 0.39886703, 0.39724552,
 0.39561735, 0.39398248, 0.3923408 , 0.39069221, 0.38903664, 0.38737403,
 0.38570428, 0.38402727, 0.38234293, 0.3806512 , 0.37895195, 0.37724508,
 0.3755305 , 0.37380814, 0.37207786, 0.37033956, 0.36859315, 0.36683852,
 0.36507555, 0.36330412, 0.36152414, 0.35973548, 0.35793801, 0.35613161,
 0.35431616, 0.35249154, 0.3506576 , 0.34881421, 0.34696124, 0.34509856,
 0.343226  , 0.34134343, 0.3394507 , 0.33754765, 0.33563413, 0.33370997,
 0.33177501, 0.32982909, 0.32787202, 0.32590363, 0.32392376, 0.32193219,
 0.31992876, 0.31791325, 0.31588548, 0.31384525, 0.31179233, 0.30972651,
 0.30764759, 0.30555533, 0.3034495 , 0.30132986, 0.29919617, 0.29704819,
 0.29488565, 0.29270829, 0.29051585, 0.28830804, 0.28608459, 0.28384521,
 0.28158959, 0.27931742, 0.2770284 , 0.27472219, 0.27239847, 0.2700569,
 0.26769711, 0.26531876, 0.26292148, 0.26050487, 0.25806856, 0.25561214,
 0.2531352 , 0.25063732, 0.24811806, 0.24557699, 0.24301364, 0.24042755,
 0.23781824, 0.23518523, 0.23252802, 0.22984609 ,0.22713895, 0.22440606,
 0.22164689, 0.21886092, 0.21604761, 0.21320644, 0.21033687, 0.2074384,
 0.20451054, 0.20155283, 0.19856483, 0.19554619, 0.19249659, 0.18941583,
 0.18630381, 0.18316059, 0.17998644, 0.17678185, 0.17354766, 0.17028509,
 0.16699588, 0.16368242, 0.16034793, 0.15699665, 0.15363413, 0.15026753,
 0.14690602, 0.14356126, 0.14024784, 0.13698384, 0.13379121, 0.13069611,
 0.12772867, 0.12492227, 0.12231191, 0.11993173, 0.11781179, 0.11597476,
 0.11443342, 0.1131895 , 0.11223431, 0.11155071, 0.11111592, 0.11090415,
 0.11088895, 0.1110448 , 0.1113481 , 0.11177764, 0.11231473, 0.11294317,
 0.11364899, 0.11442024, 0.11524673, 0.11611976, 0.11703197, 0.11797705,
 0.11894963, 0.11994514, 0.12095966, 0.12198984, 0.12303279, 0.12408604,
 0.12514748, 0.12621524, 0.12728773, 0.12836359, 0.12944166, 0.13052086,
 0.1316003 , 0.13267924, 0.13375702, 0.13483299, 0.13590665, 0.13697764,
 0.13804557, 0.13911001, 0.14017071, 0.14122756, 0.1422803 , 0.14332859,
 0.14437234, 0.14541169, 0.14644639, 0.14747602, 0.14850076, 0.14952097,
 0.15053625, 0.15154601, 0.15255081, 0.15355133, 0.15454671, 0.15553608,
 0.15652073, 0.15750176, 0.15847729])

rHeston_2_100 = np.array([0.43117435, 0.42969094, 0.42820256, 0.42670893, 0.42520998, 0.42370586,
 0.42219651, 0.4206817 , 0.41916138, 0.41763566, 0.41610446, 0.41456757,
 0.41302496, 0.4114767 , 0.40992268, 0.40836274, 0.40679686, 0.40522505,
 0.4036472 , 0.40206318, 0.40047294, 0.3988765  ,0.39727373, 0.39566449,
 0.39404877, 0.39242653, 0.39079764, 0.38916199, 0.38751955, 0.38587025,
 0.38421397, 0.3825506 , 0.3808801 , 0.37920238, 0.37751733, 0.37582483,
 0.37412483, 0.37241722, 0.37070189, 0.36897873, 0.36724765, 0.36550856,
 0.36376132, 0.36200582, 0.36024197, 0.35846965, 0.35668872, 0.35489907,
 0.35310058, 0.35129313, 0.34947657, 0.34765078, 0.34581562, 0.34397096,
 0.34211665, 0.34025253, 0.33837848, 0.33649433, 0.33459991, 0.33269508,
 0.33077968, 0.32885352, 0.32691644, 0.32496825, 0.32300879, 0.32103786,
 0.31905526, 0.31706081 ,0.3150543 , 0.31303552, 0.31100426, 0.30896031,
 0.30690343, 0.3048334 , 0.30274998, 0.30065293, 0.298542  , 0.29641693,
 0.29427745, 0.29212329, 0.28995419, 0.28776983, 0.28556994, 0.28335421,
 0.28112232, 0.27887395, 0.27660876, 0.27432643, 0.27202659, 0.26970888,
 0.26737294, 0.26501837, 0.26264478, 0.26025176, 0.2578389 , 0.25540577,
 0.25295191, 0.25047688, 0.2479802 , 0.2454614 , 0.24291997, 0.24035541,
 0.23776719, 0.23515479, 0.23251765, 0.22985522, 0.22716692, 0.22445218,
 0.22171041, 0.218941  , 0.21614337, 0.21331692, 0.21046104, 0.20757516,
 0.2046587 , 0.20171112, 0.19873191, 0.1957206 , 0.19267681, 0.18960022,
 0.18649064, 0.18334803, 0.18017255, 0.17696458, 0.17372485, 0.17045448,
 0.16715511, 0.16382904, 0.1604794 , 0.15711039, 0.15372754, 0.15033808,
 0.1469513 , 0.14357912, 0.14023653, 0.13694225 ,0.13371911, 0.13059445,
 0.12759987, 0.12477044, 0.12214295, 0.1197531 , 0.11763207, 0.11580291,
 0.11427793, 0.11305768, 0.11213181, 0.11148141, 0.11108195, 0.11090623,
 0.1109267 , 0.11111704, 0.11145313, 0.11191344, 0.11247914, 0.11313395,
 0.11386393, 0.11465719, 0.11550363, 0.11639467, 0.11732306, 0.11828261,
 0.11926807, 0.12027498, 0.12129954, 0.1223385 , 0.12338906, 0.12444887,
 0.12551587, 0.12658831, 0.12766466, 0.12874365, 0.12982416, 0.13090519,
 0.13198591, 0.13306563, 0.13414375, 0.13521965, 0.13629288, 0.13736314,
 0.13843007, 0.13949324, 0.14055247, 0.14160769, 0.14265864, 0.14370497,
 0.14474668, 0.14578395, 0.14681646, 0.14784381, 0.14886628, 0.14988428,
 0.15089726, 0.15190463, 0.15290719, 0.15390561, 0.15489873, 0.15588574,
 0.15686837 ,0.15784759, 0.15882096])

rHeston_3_100 = np.array([0.43019463, 0.42871329, 0.42722692, 0.42573546, 0.42423886, 0.42273707,
 0.42123004, 0.41971771, 0.41820002, 0.41667692, 0.41514836, 0.41361426,
 0.41207458, 0.41052925, 0.40897821, 0.40742141, 0.40585876, 0.40429022,
 0.4027157 , 0.40113516, 0.39954851, 0.39795569, 0.39635662, 0.39475124,
 0.39313946, 0.39152122, 0.38989643, 0.38826502, 0.38662691, 0.38498202,
 0.38333026, 0.38167154, 0.38000579, 0.37833291, 0.37665281, 0.3749654,
 0.37327059, 0.37156828, 0.36985837, 0.36814076, 0.36641535, 0.36468203,
 0.3629407 , 0.36119125, 0.35943357, 0.35766754, 0.35589305, 0.35410997,
 0.35231818, 0.35051757, 0.34870799, 0.34688933, 0.34506144, 0.34322419,
 0.34137744, 0.33952105, 0.33765486, 0.33577873, 0.3338925 , 0.33199601,
 0.3300891 , 0.3281716 , 0.32624334, 0.32430415, 0.32235385, 0.32039224,
 0.31841915, 0.31643437, 0.31443771, 0.31242895, 0.31040789, 0.30837432,
 0.306328  , 0.30426871, 0.30219621, 0.30011026, 0.29801061, 0.29589701,
 0.29376918, 0.29162685, 0.28946976, 0.28729759, 0.28511007, 0.28290688,
 0.28068771, 0.27845223 ,0.27620011, 0.27393099, 0.27164454, 0.26934037,
 0.26701811, 0.26467737, 0.26231774, 0.25993881, 0.25754016, 0.25512134,
 0.25268189, 0.25022135, 0.24773922, 0.24523503, 0.24270824, 0.24015834,
 0.23758478, 0.234987  , 0.23236444, 0.2297165 , 0.22704258, 0.22434208,
 0.22161438, 0.21885883, 0.21607481, 0.21326166, 0.21041876, 0.20754545,
 0.20464112, 0.20170516, 0.198737  , 0.19573611, 0.19270203, 0.18963438,
 0.18653289, 0.18339742, 0.18022805, 0.17702508, 0.17378913, 0.17052125,
 0.16722297, 0.16389651, 0.16054492, 0.15717234, 0.15378425, 0.15038787,
 0.14699254, 0.14361026, 0.14025623, 0.13694944, 0.13371319, 0.13057535,
 0.12756824, 0.12472774, 0.12209143, 0.1196957 , 0.11757212, 0.11574379,
 0.11422271, 0.11300877, 0.11209084, 0.11144919, 0.11105863, 0.11089143,
 0.11091965, 0.11111676, 0.11145853, 0.1119234 , 0.11249256, 0.11314978,
 0.11388121, 0.11467502, 0.11552122, 0.11641131, 0.1173381 , 0.1182955,
 0.11927834, 0.12028221, 0.12130337, 0.12233862, 0.12338525, 0.12444091,
 0.1255036 , 0.1265716 , 0.12764345, 0.12871785, 0.12979373, 0.13087013,
 0.13194625, 0.13302138, 0.13409491, 0.13516633, 0.13623518, 0.13730107,
 0.13836369, 0.13942273, 0.14047796, 0.14152917, 0.14257619, 0.14361886,
 0.14465707, 0.14569072, 0.1467197 , 0.14774398, 0.1487635 , 0.14977819,
 0.15078805, 0.15179308, 0.15279321, 0.15378847, 0.15477891, 0.15576447,
 0.15674515, 0.15772109, 0.15869223])

rHeston_4_100 = np.array([0.42969539, 0.42821401, 0.42672764, 0.42523622, 0.42373969, 0.42223802,
 0.42073114, 0.419219  , 0.41770154, 0.41617872, 0.41465047 ,0.41311673,
 0.41157745, 0.41003257, 0.40848202, 0.40692575, 0.40536369, 0.40379577,
 0.40222193, 0.40064212, 0.39905624, 0.39746425, 0.39586606, 0.3942616,
 0.39265081, 0.39103361, 0.38940991, 0.38777965, 0.38614275, 0.38449912,
 0.38284868, 0.38119134, 0.37952704, 0.37785566, 0.37617713, 0.37449136,
 0.37279824, 0.37109769, 0.36938962, 0.36767391, 0.36595047, 0.3642192,
 0.36247999, 0.36073273, 0.35897732, 0.35721363, 0.35544156, 0.35366098,
 0.35187178, 0.35007383, 0.34826701, 0.34645118, 0.34462621, 0.34279197,
 0.34094832, 0.33909512, 0.33723221, 0.33535946, 0.3334767 , 0.33158378,
 0.32968054, 0.32776682, 0.32584244, 0.32390723, 0.32196101, 0.3200036,
 0.31803481, 0.31605444, 0.31406231, 0.31205819, 0.31004189, 0.30801319,
 0.30597187, 0.3039177 , 0.30185043, 0.29976985, 0.29767568, 0.29556769,
 0.2934456 , 0.29130914, 0.28915804, 0.28699201, 0.28481075, 0.28261395,
 0.2804013 , 0.27817248, 0.27592715, 0.27366496, 0.27138556, 0.26908858,
 0.26677363, 0.26444034, 0.26208828, 0.25971706, 0.25732622, 0.25491534,
 0.25248394, 0.25003157, 0.24755771, 0.24506189, 0.24254356, 0.24000221,
 0.23743727, 0.23484818, 0.23223436, 0.2295952 , 0.2269301 , 0.22423843,
 0.22151955, 0.21877281, 0.21599754, 0.21319309, 0.21035878, 0.20749395,
 0.20459795, 0.20167014, 0.19870991, 0.19571669, 0.19268998, 0.18962935,
 0.18653447, 0.18340518, 0.18024147, 0.17704359, 0.17381212, 0.170548,
 0.16725274, 0.16392846, 0.16057817, 0.15720594, 0.15381722, 0.15041918,
 0.14702118, 0.14363522, 0.1402766 , 0.13696443, 0.13372222, 0.13057814,
 0.12756486, 0.12471866, 0.12207753, 0.11967819, 0.11755239, 0.11572322,
 0.11420242, 0.11298954, 0.11207302, 0.11143273, 0.11104317, 0.11087638,
 0.11090434, 0.11110047, 0.11144055, 0.11190308, 0.11246933, 0.11312315,
 0.11385074, 0.11464038, 0.11548214, 0.11636757, 0.11728955, 0.11824203,
 0.11921987, 0.12021873, 0.12123487, 0.12226513, 0.12330682, 0.1243576,
 0.1254155 , 0.12647881, 0.12754606, 0.12861599, 0.12968751, 0.13075968,
 0.13183169, 0.13290284, 0.13397254, 0.13504025, 0.13610552, 0.13716798,
 0.13822729, 0.13928315, 0.14033532, 0.14138361, 0.14242783, 0.14346782,
 0.14450347, 0.14553467, 0.14656132, 0.14758338, 0.14860078, 0.14961346,
 0.15062141, 0.15162462, 0.15262304, 0.15361666, 0.15460558, 0.15558968,
 0.15656899, 0.15754365, 0.15851358])

rHeston_5_100 = np.array([0.4294413 , 0.42795935, 0.42647242, 0.42498047, 0.42348343, 0.42198126,
 0.42047391, 0.41896132, 0.41744343, 0.4159202 , 0.41439156, 0.41285745,
 0.41131784, 0.40977264, 0.40822179, 0.40666525, 0.40510294, 0.40353481,
 0.40196078, 0.40038079, 0.39879477, 0.39720266, 0.39560439, 0.39399988,
 0.39238906, 0.39077185, 0.38914819, 0.38751799, 0.38588118, 0.38423768,
 0.38258739, 0.38093025, 0.37926617, 0.37759506, 0.37591682, 0.37423138,
 0.37253864, 0.37083849, 0.36913086, 0.36741564, 0.36569273, 0.36396202,
 0.36222342, 0.36047681, 0.35872209, 0.35695914, 0.35518785, 0.3534081,
 0.35161978, 0.34982275, 0.3480169 , 0.3462021 , 0.3443782 , 0.34254509,
 0.34070262, 0.33885065, 0.33698903, 0.33511762, 0.33323627, 0.33134482,
 0.3294431 , 0.32753096, 0.32560822, 0.32367472, 0.32173027, 0.3197747,
 0.31780782, 0.31582943, 0.31383934, 0.31183734, 0.30982323, 0.30779679,
 0.3057578 , 0.30370604, 0.30164127, 0.29956325, 0.29747174, 0.29536647,
 0.2932472 , 0.29111364, 0.28896552, 0.28680256, 0.28462445, 0.2824309,
 0.28022158, 0.27799619, 0.27575437, 0.27349579, 0.27122008, 0.26892689,
 0.26661584, 0.26428652, 0.26193854, 0.25957148 ,0.25718491, 0.25477837,
 0.25235142, 0.24990358, 0.24743435, 0.24494323, 0.2424297 , 0.23989322,
 0.23733324, 0.23474918, 0.23214045, 0.22950646, 0.22684658, 0.22416019,
 0.22144662, 0.21870523, 0.21593533, 0.21313627, 0.21030734, 0.20744789,
 0.20455723, 0.20163471, 0.19867971, 0.19569163, 0.19266994, 0.18961419,
 0.18652404, 0.18339926, 0.18023983, 0.17704597, 0.17381819, 0.17055743,
 0.16726512, 0.16394335, 0.16059509, 0.15722435, 0.15383655, 0.15043882,
 0.1470405 , 0.14365359, 0.14029339, 0.13697909, 0.13373427, 0.13058724,
 0.12757084, 0.12472157, 0.12207763, 0.11967591, 0.11754824, 0.1157177,
 0.11419591, 0.1129822 , 0.11206478, 0.11142333, 0.1110322 , 0.11086337,
 0.11088877, 0.11108187, 0.11141849, 0.11187721, 0.11243935, 0.11308883,
 0.11381192, 0.11459697, 0.11543406, 0.11631482, 0.11723215, 0.11818002,
 0.11915333, 0.12014773, 0.12115953, 0.12218556, 0.12322313, 0.12426993,
 0.12532398, 0.12638356, 0.12744723, 0.1285137 , 0.1295819 , 0.13065089,
 0.13171985, 0.13278807, 0.13385496, 0.13491999, 0.1359827 , 0.1370427,
 0.13809966, 0.13915329, 0.14020334, 0.1412496 , 0.14229187, 0.14333002,
 0.14436392, 0.14539345, 0.14641852, 0.14743907, 0.14845504, 0.14946635,
 0.15047302, 0.15147502, 0.15247227, 0.1534648 , 0.15445268, 0.15543581,
 0.15641419, 0.15738798, 0.15835708])

rHeston_obs_2_100 = np.array([0.38774008, 0.38655288, 0.38537618, 0.38418844, 0.38298162,
       0.38177158, 0.38057017, 0.37936741, 0.37814992, 0.37692196,
       0.37569631, 0.37447324, 0.37324179, 0.37199816, 0.37075052,
       0.36950461, 0.36825514, 0.3669954 , 0.36572773, 0.36445858,
       0.36318784, 0.36190985, 0.36062279, 0.3593309 , 0.35803696,
       0.35673808, 0.35543087, 0.35411661, 0.35279855, 0.35147647,
       0.3501474 , 0.34881045, 0.34746779, 0.34612077, 0.34476777,
       0.34340703, 0.34203922, 0.34066596, 0.33928703, 0.33790083,
       0.33650687, 0.33510627, 0.33369962, 0.332286  , 0.33086443,
       0.32943524, 0.32799922, 0.32655614, 0.32510509, 0.32364577,
       0.3221787 , 0.32070412, 0.31922144, 0.31773007, 0.31623013,
       0.31472194, 0.3132053 , 0.31167962, 0.31014466, 0.30860062,
       0.30704752, 0.30548493, 0.30391245, 0.30233004, 0.30073778,
       0.29913542, 0.29752253, 0.29589888, 0.29426445, 0.29261913,
       0.29096254, 0.28929433, 0.28761434, 0.28592248, 0.28421845,
       0.28250187, 0.28077243, 0.27902999, 0.27727428, 0.27550493,
       0.27372154, 0.27192386, 0.27011161, 0.26828443, 0.26644186,
       0.26458354, 0.26270913, 0.26081824, 0.2589104 , 0.25698513,
       0.25504198, 0.25308053, 0.25110025, 0.24910057, 0.24708094,
       0.24504082, 0.24297964, 0.24089672, 0.23879139, 0.23666298,
       0.23451081, 0.23233409, 0.23013203, 0.22790378, 0.22564852,
       0.22336534, 0.22105327, 0.21871131, 0.21633842, 0.21393355,
       0.21149556, 0.20902323, 0.20651536, 0.20397068, 0.20138791,
       0.19876567, 0.19610259, 0.1933973 , 0.19064842, 0.18785459,
       0.18501449, 0.18212691, 0.17919078, 0.17620527, 0.17316979,
       0.17008421, 0.166949  , 0.16376538, 0.16053562, 0.15726335,
       0.15395408, 0.15061566, 0.14725904, 0.14389904, 0.14055536,
       0.13725356, 0.13402584, 0.13091135, 0.12795554, 0.12520784,
       0.12271743, 0.12052734, 0.11866842, 0.11715497, 0.11598364,
       0.11513581, 0.11458225, 0.1142883 , 0.11421813, 0.11433755,
       0.11461564, 0.11502549, 0.11554417, 0.11615243, 0.11683413,
       0.11757596, 0.118367  , 0.11919825, 0.1200622 , 0.12095257,
       0.1218643 , 0.12279326, 0.12373594, 0.12468924, 0.12565065,
       0.12661829, 0.12759058, 0.12856589, 0.12954274, 0.13052022,
       0.13149787, 0.13247491, 0.13345017, 0.13442294, 0.13539348,
       0.13636187, 0.13732703, 0.13828778, 0.13924454, 0.14019865,
       0.14114964, 0.14209515, 0.14303464, 0.1439709 , 0.14490566,
       0.1458355 , 0.14675659, 0.14767208, 0.14858854, 0.14950408,
       0.15040874, 0.15130115, 0.15219485, 0.15309722, 0.15399202,
       0.1548624 , 0.15572458, 0.15660926, 0.15750554, 0.15836554,
       0.15918535])

rHeston_obs_4_100 = np.array([0.40646287, 0.40512765, 0.40378795, 0.40244367, 0.40109482,
       0.39974134, 0.39838316, 0.39702025, 0.39565256, 0.39428002,
       0.3929026 , 0.39152025, 0.39013289, 0.38874048, 0.38734297,
       0.38594029, 0.38453239, 0.38311922, 0.3817007 , 0.38027677,
       0.37884739, 0.37741248, 0.37597196, 0.37452579, 0.3730739 ,
       0.3716162 , 0.37015263, 0.36868313, 0.36720761, 0.36572601,
       0.36423824, 0.36274423, 0.3612439 , 0.35973716, 0.35822394,
       0.35670414, 0.35517769, 0.35364448, 0.35210444, 0.35055747,
       0.34900347, 0.34744235, 0.34587401, 0.34429835, 0.34271525,
       0.34112462, 0.33952635, 0.33792032, 0.33630643, 0.33468455,
       0.33305456, 0.33141634, 0.32976976, 0.3281147 , 0.32645103,
       0.3247786 , 0.32309727, 0.32140691, 0.31970736, 0.31799848,
       0.3162801 , 0.31455208, 0.31281423, 0.31106641, 0.30930842,
       0.3075401 , 0.30576126, 0.30397172, 0.30217126, 0.3003597 ,
       0.29853683, 0.29670244, 0.2948563 , 0.29299819, 0.29112787,
       0.28924511, 0.28734966, 0.28544125, 0.28351962, 0.28158451,
       0.27963561, 0.27767265, 0.27569531, 0.27370329, 0.27169625,
       0.26967387, 0.26763578, 0.26558164, 0.26351105, 0.26142365,
       0.25931901, 0.25719673, 0.25505636, 0.25289746, 0.25071956,
       0.24852216, 0.24630476, 0.24406684, 0.24180785, 0.2395272 ,
       0.23722432, 0.23489859, 0.23254935, 0.23017595, 0.22777769,
       0.22535384, 0.22290366, 0.22042637, 0.21792116, 0.21538718,
       0.21282359, 0.21022947, 0.20760392, 0.20494598, 0.20225469,
       0.19952908, 0.19676816, 0.19397093, 0.19113645, 0.18826376,
       0.18535199, 0.18240036, 0.1794082 , 0.17637505, 0.17330068,
       0.17018524, 0.16702932, 0.16383419, 0.16060192, 0.15733574,
       0.15404033, 0.15072232, 0.14739081, 0.14405808, 0.14074038,
       0.13745876, 0.13423983, 0.13111622, 0.12812631, 0.12531283,
       0.12271996, 0.12038895, 0.11835309, 0.11663323, 0.11523549,
       0.11415178, 0.11336259, 0.11284102, 0.11255666, 0.11247869,
       0.1125779 , 0.11282784, 0.11320522, 0.11368991, 0.11426476,
       0.11491526, 0.11562918, 0.11639622, 0.11720774, 0.11805646,
       0.11893626, 0.11984196, 0.12076919, 0.12171423, 0.12267393,
       0.12364561, 0.12462696, 0.12561602, 0.1266111 , 0.12761076,
       0.12861375, 0.129619  , 0.13062557, 0.13163266, 0.13263958,
       0.13364572, 0.13465058, 0.13565369, 0.13665467, 0.13765318,
       0.13864893, 0.13964167, 0.14063117, 0.14161726, 0.14259979,
       0.14357861, 0.14455359, 0.14552468, 0.14649176, 0.14745476,
       0.14841368, 0.14936843, 0.15031895, 0.15126531, 0.15220744,
       0.15314526, 0.15407892, 0.15500835, 0.15593342, 0.15685437,
       0.15777118])

rHeston_obs_8_100 = np.array([0.42452982, 0.42308304, 0.42163124, 0.42017456, 0.41871293,
       0.41724612, 0.41577415, 0.41429709, 0.41281482, 0.41132716,
       0.40983417, 0.40833585, 0.40683205, 0.40532266, 0.40380773,
       0.4022872 , 0.40076093, 0.39922886, 0.397691  , 0.39614726,
       0.39459751, 0.39304172, 0.39147986, 0.38991184, 0.38833752,
       0.38675689, 0.3851699 , 0.38357642, 0.38197637, 0.3803697 ,
       0.37875633, 0.37713615, 0.37550908, 0.37387506, 0.37223399,
       0.37058576, 0.36893028, 0.36726749, 0.36559728, 0.36391952,
       0.36223414, 0.36054104, 0.3588401 , 0.35713121, 0.35541428,
       0.3536892 , 0.35195583, 0.35021407, 0.3484638 , 0.34670489,
       0.34493722, 0.34316066, 0.34137509, 0.33958035, 0.33777632,
       0.33596286, 0.33413982, 0.33230705, 0.3304644 , 0.32861171,
       0.32674883, 0.32487559, 0.32299182, 0.32109734, 0.31919199,
       0.31727558, 0.31534791, 0.31340881, 0.31145807, 0.30949548,
       0.30752084, 0.30553394, 0.30353455, 0.30152245, 0.2994974 ,
       0.29745916, 0.29540748, 0.29334212, 0.2912628 , 0.28916926,
       0.28706121, 0.28493837, 0.28280044, 0.28064712, 0.27847809,
       0.27629302, 0.27409159, 0.27187343, 0.2696382 , 0.26738552,
       0.26511502, 0.26282629, 0.26051893, 0.25819253, 0.25584663,
       0.2534808 , 0.25109457, 0.24868747, 0.24625898, 0.24380861,
       0.24133582, 0.23884007, 0.2363208 , 0.23377742, 0.23120934,
       0.22861595, 0.2259966 , 0.22335066, 0.22067747, 0.21797633,
       0.21524657, 0.21248749, 0.20969838, 0.20687854, 0.20402727,
       0.20114387, 0.19822769, 0.19527808, 0.19229448, 0.18927637,
       0.18622335, 0.18313513, 0.18001161, 0.17685293, 0.17365951,
       0.17043218, 0.16717225, 0.16388172, 0.1605634 , 0.15722119,
       0.15386037, 0.15048796, 0.14711317, 0.14374796, 0.14040763,
       0.13711142, 0.13388311, 0.13075132, 0.12774933, 0.12491418,
       0.1222846 , 0.11989791, 0.1177861 , 0.11597206, 0.11446689,
       0.11326921, 0.11236644, 0.11173761, 0.11135661, 0.1111952 ,
       0.1112253 , 0.11142048, 0.11175679, 0.11221303, 0.1127708 ,
       0.11341428, 0.11412999, 0.11490645, 0.11573396, 0.1166043 ,
       0.11751051, 0.11844667, 0.11940777, 0.12038956, 0.1213884 ,
       0.12240118, 0.12342527, 0.12445839, 0.12549859, 0.12654417,
       0.12759373, 0.12864601, 0.12969992, 0.13075454, 0.1318091 ,
       0.13286288, 0.13391525, 0.13496574, 0.13601393, 0.13705937,
       0.1381017 , 0.13914075, 0.14017625, 0.14120787, 0.14223547,
       0.14325908, 0.14427845, 0.14529328, 0.14630369, 0.14730983,
       0.14831129, 0.14930782, 0.15029992, 0.15128773, 0.15227044,
       0.15324803, 0.15422161, 0.15519091, 0.15615447, 0.15711308,
       0.15806858])

rHeston_obs_16_100 = np.array([0.42626127, 0.42479755, 0.42332881, 0.4218552 , 0.42037663,
       0.41889289, 0.41740398, 0.41590999, 0.41441079, 0.4129062 ,
       0.41139629, 0.40988105, 0.40836033, 0.40683403, 0.40530219,
       0.40376477, 0.40222161, 0.40067266, 0.39911792, 0.39755732,
       0.39599072, 0.39441809, 0.39283941, 0.39125456, 0.38966345,
       0.38806604, 0.38646227, 0.38485204, 0.38323525, 0.38161186,
       0.3799818 , 0.37834495, 0.37670123, 0.37505057, 0.3733929 ,
       0.37172809, 0.37005606, 0.36837675, 0.36669004, 0.36499582,
       0.36329401, 0.36158451, 0.35986721, 0.358142  , 0.35640879,
       0.35466746, 0.35291788, 0.35115996, 0.34939357, 0.3476186 ,
       0.34583491, 0.34404238, 0.34224088, 0.34043029, 0.33861045,
       0.33678124, 0.33494252, 0.33309413, 0.33123592, 0.32936774,
       0.32748944, 0.32560085, 0.32370181, 0.32179214, 0.31987168,
       0.31794023, 0.31599763, 0.31404367, 0.31207816, 0.31010091,
       0.3081117 , 0.30611034, 0.30409659, 0.30207023, 0.30003104,
       0.29797877, 0.29591319, 0.29383404, 0.29174106, 0.28963399,
       0.28751254, 0.28537644, 0.2832254 , 0.28105911, 0.27887725,
       0.27667952, 0.27446558, 0.27223508, 0.26998767, 0.26772298,
       0.26544065, 0.26314027, 0.26082144, 0.25848376, 0.25612678,
       0.25375007, 0.25135315, 0.24893557, 0.24649682, 0.24403639,
       0.24155377, 0.23904841, 0.23651975, 0.23396722, 0.23139021,
       0.22878813, 0.22616033, 0.22350618, 0.220825  , 0.21811613,
       0.21537888, 0.21261254, 0.20981641, 0.20698979, 0.20413196,
       0.20124224, 0.19831995, 0.19536446, 0.19237517, 0.18935157,
       0.18629324, 0.18319987, 0.18007136, 0.17690781, 0.17370962,
       0.17047759, 0.16721302, 0.16391785, 0.16059486, 0.1572479 ,
       0.15388219, 0.1505047 , 0.14712457, 0.14375367, 0.14040721,
       0.13710433, 0.13386869, 0.13072878, 0.12771779, 0.12487261,
       0.1222319 , 0.11983296, 0.11770783, 0.11587953, 0.11435937,
       0.11314622, 0.11222778, 0.11158334, 0.11118702, 0.11101076,
       0.11102659, 0.11120819, 0.11153164, 0.1119758 , 0.11252224,
       0.11315517, 0.11386106, 0.11462844, 0.11544758, 0.11631022,
       0.11720937, 0.1181391 , 0.11909436, 0.12007088, 0.12106498,
       0.12207355, 0.1230939 , 0.12412376, 0.12516113, 0.12620432,
       0.12725189, 0.12830255, 0.1293552 , 0.13040892, 0.13146289,
       0.1325164 , 0.13356879, 0.13461959, 0.13566834, 0.1367146 ,
       0.13775799, 0.13879832, 0.13983532, 0.14086864, 0.14189813,
       0.14292382, 0.14394544, 0.14496269, 0.14597566, 0.14698451,
       0.14798883, 0.14898834, 0.14998354, 0.15097458, 0.15196064,
       0.15294165, 0.15391876, 0.15489171, 0.15585899, 0.15682136,
       0.15778076])

rHeston_obs_32_100 = np.array([0.42856104, 0.42708194, 0.42559778, 0.42410868, 0.42261459,
       0.4211153 , 0.4196108 , 0.41810116, 0.41658627, 0.41506597,
       0.41354028, 0.41200923, 0.41047265, 0.40893046, 0.40738268,
       0.40582927, 0.40427008, 0.40270505, 0.40113419, 0.39955741,
       0.3979746 , 0.39638571, 0.39479072, 0.39318952, 0.391582  ,
       0.38996814, 0.38834788, 0.3867211 , 0.38508772, 0.38344769,
       0.38180093, 0.38014734, 0.37848682, 0.37681933, 0.37514476,
       0.373463  , 0.37177398, 0.37007762, 0.36837381, 0.36666244,
       0.36494343, 0.36321668, 0.36148207, 0.3597395 , 0.35798887,
       0.35623007, 0.35446297, 0.35268747, 0.35090345, 0.34911079,
       0.34730935, 0.34549902, 0.34367968, 0.34185117, 0.34001337,
       0.33816614, 0.33630933, 0.3344428 , 0.3325664 , 0.33067997,
       0.32878337, 0.32687641, 0.32495894, 0.3230308 , 0.3210918 ,
       0.31914176, 0.3171805 , 0.31520783, 0.31322356, 0.31122748,
       0.30921939, 0.30719908, 0.30516634, 0.30312093, 0.30106263,
       0.2989912 , 0.2969064 , 0.29480798, 0.29269567, 0.29056922,
       0.28842835, 0.28627277, 0.28410219, 0.28191632, 0.27971484,
       0.27749743, 0.27526377, 0.27301351, 0.27074629, 0.26846177,
       0.26615956, 0.26383927, 0.2615005 , 0.25914284, 0.25676587,
       0.25436914, 0.25195219, 0.24951456, 0.24705575, 0.24457527,
       0.24207259, 0.23954718, 0.23699849, 0.23442594, 0.23182895,
       0.22920692, 0.22655922, 0.22388523, 0.22118428, 0.21845572,
       0.21569887, 0.21291305, 0.21009756, 0.20725172, 0.20437483,
       0.20146623, 0.19852527, 0.19555134, 0.19254386, 0.18950236,
       0.18642645, 0.18331585, 0.1801705 , 0.17699055, 0.17377644,
       0.17052901, 0.16724962, 0.16394026, 0.16060377, 0.15724406,
       0.15386639, 0.15047778, 0.14708742, 0.14370719, 0.1403523 ,
       0.13704184, 0.13379941, 0.13065335, 0.12763665, 0.124786  ,
       0.12213983, 0.11973525, 0.11760422, 0.11576981, 0.1142435 ,
       0.11302441, 0.11210055, 0.11145143, 0.11105134, 0.11087231,
       0.1108864 , 0.11106725, 0.11139088, 0.11183605, 0.11238427,
       0.11301962, 0.11372853, 0.11449941, 0.11532249, 0.11618943,
       0.1170932 , 0.11802782, 0.1189882 , 0.11997003, 0.12096961,
       0.12198378, 0.12300987, 0.12404555, 0.12508882, 0.12613798,
       0.12719155, 0.12824828, 0.12930702, 0.13036685, 0.13142695,
       0.1324866 , 0.13354514, 0.13460208, 0.13565698, 0.13670937,
       0.13775888, 0.13880532, 0.13984842, 0.14088782, 0.14192338,
       0.14295509, 0.14398273, 0.14500598, 0.14602492, 0.1470397 ,
       0.14804995, 0.14905536, 0.15005642, 0.15105328, 0.15204516,
       0.15303198, 0.15401481, 0.15499347, 0.15596649, 0.15693455,
       0.15789952])

k_010 = np.linspace(-1, 0.4, 201)

rHeston_010 = np.array([0.58293893, 0.58097686, 0.57900811 ,0.57703265, 0.57505037, 0.57306123,
 0.57106515, 0.56906205, 0.56705186, 0.56503451, 0.56300991, 0.560978,
 0.55893868, 0.55689189, 0.55483753, 0.55277553, 0.55070579, 0.54862823,
 0.54654277, 0.5444493 , 0.54234775, 0.54023801, 0.53811998, 0.53599358,
 0.5338587 , 0.53171524, 0.52956309, 0.52740216, 0.52523232, 0.52305348,
 0.52086552, 0.51866832, 0.51646177, 0.51424574, 0.51202012, 0.50978478,
 0.50753959, 0.50528442, 0.50301915, 0.50074362, 0.49845771, 0.49616126,
 0.49385414, 0.49153619, 0.48920727, 0.48686721, 0.48451585, 0.48215304,
 0.4797786 , 0.47739236, 0.47499414, 0.47258376, 0.47016105, 0.4677258,
 0.46527782, 0.46281692, 0.46034288, 0.4578555 , 0.45535456, 0.45283984,
 0.4503111 , 0.44776812, 0.44521065, 0.44263844, 0.44005124, 0.43744878,
 0.4348308 , 0.43219701, 0.42954713, 0.42688086, 0.4241979 , 0.42149792,
 0.41878062, 0.41604565, 0.41329267, 0.41052131, 0.40773122, 0.404922,
 0.40209327, 0.39924461, 0.39637561, 0.39348582, 0.3905748 , 0.38764207,
 0.38468714, 0.38170952, 0.37870867, 0.37568405, 0.3726351 , 0.36956121,
 0.36646179, 0.36333619, 0.36018375, 0.35700376, 0.35379552, 0.35055826,
 0.3472912 , 0.3439935 , 0.34066431, 0.33730271, 0.33390777, 0.33047849,
 0.32701382, 0.32351267, 0.31997388, 0.31639624, 0.31277847, 0.30911922,
 0.30541706, 0.30167048, 0.29787789, 0.2940376 , 0.29014782, 0.28620663,
 0.28221202, 0.27816182, 0.27405372, 0.26988528, 0.26565386, 0.26135665,
 0.25699061, 0.25255251, 0.24803886, 0.24344588, 0.23876953, 0.23400541,
 0.2291488 , 0.22419456, 0.21913714, 0.21397054, 0.20868828, 0.20328337,
 0.19774833, 0.19207519, 0.18625564, 0.18028116, 0.17414347, 0.16783518,
 0.16135111, 0.15469052, 0.14786126, 0.140887  , 0.13382027, 0.12676441,
 0.11990708, 0.11355575, 0.10813095, 0.10404199, 0.10147301, 0.10029854,
 0.10021636, 0.10091406, 0.10214431, 0.10373154, 0.10555601, 0.10753683,
 0.10961909, 0.11176516, 0.11394896, 0.11615223, 0.11836209, 0.12056939,
 0.12276763, 0.12495219, 0.1271198 , 0.12926819, 0.13139582, 0.13350167,
 0.13558514, 0.13764592, 0.13968391, 0.14169918, 0.14369192, 0.14566242,
 0.14761102, 0.14953812, 0.15144413, 0.1533295 , 0.15519469, 0.15704016,
 0.15886638, 0.16067381, 0.16246289, 0.16423409, 0.16598784, 0.16772457,
 0.16944469, 0.17114862, 0.17283675, 0.17450947, 0.17616714, 0.17781012,
 0.17943877, 0.18105342, 0.18265441, 0.18424201, 0.18581657, 0.18737845,
 0.18892779, 0.19046485, 0.19199023])

rHeston_1_010 = np.array([0.58510233, 0.58316736, 0.5812256 , 0.57927699, 0.57732145, 0.57535891,
 0.57338929, 0.57141252, 0.56942852, 0.56743721, 0.56543851, 0.56343235,
 0.56141862, 0.55939726, 0.55736817, 0.55533127, 0.55328646, 0.55123365,
 0.54917276, 0.54710368, 0.54502632, 0.54294057, 0.54084635, 0.53874354,
 0.53663204, 0.53451174, 0.53238254, 0.53024432, 0.52809696, 0.52594036,
 0.52377439, 0.52159893, 0.51941385, 0.51721904, 0.51501436, 0.51279967,
 0.51057485, 0.50833975, 0.50609424, 0.50383817, 0.50157138, 0.49929374,
 0.49700508, 0.49470524, 0.49239407, 0.49007139, 0.48773704, 0.48539084,
 0.48303262, 0.48066218, 0.47827934, 0.47588391, 0.47347569, 0.47105448,
 0.46862007, 0.46617224, 0.46371077, 0.46123545, 0.45874603, 0.45624228,
 0.45372395, 0.4511908 , 0.44864255, 0.44607896, 0.44349973, 0.4409046,
 0.43829326, 0.43566542, 0.43302076, 0.43035898, 0.42767974, 0.4249827,
 0.42226751, 0.41953381, 0.41678123, 0.41400937, 0.41121785, 0.40840625,
 0.40557414, 0.40272107, 0.3998466 , 0.39695025, 0.39403152, 0.3910899,
 0.38812488, 0.38513588, 0.38212236, 0.3790837 , 0.3760193 , 0.37292851,
 0.36981067, 0.36666508, 0.363491  , 0.3602877 , 0.35705437, 0.35379019,
 0.3504943 , 0.3471658 , 0.34380374, 0.34040715, 0.33697497, 0.33350613,
 0.32999949, 0.32645386, 0.32286797, 0.3192405 , 0.31557007, 0.31185521,
 0.30809438, 0.30428594, 0.30042818, 0.29651928, 0.29255732, 0.28854027,
 0.28446598, 0.28033217, 0.27613643, 0.27187619, 0.26754873, 0.26315118,
 0.25868046, 0.25413332, 0.24950629, 0.24479569, 0.23999761, 0.23510788,
 0.2301221 , 0.22503557, 0.21984333, 0.21454013, 0.20912045, 0.20357853,
 0.1979084 , 0.19210397, 0.18615922, 0.18006845, 0.17382676, 0.16743086,
 0.16088037, 0.15418016, 0.14734426, 0.14040256, 0.13341219, 0.12647605,
 0.11976948, 0.11356728, 0.10823837, 0.10415217, 0.10150585, 0.10022774,
 0.10006105, 0.10071097, 0.10192948, 0.10353377, 0.10539618, 0.10742911,
 0.10957247, 0.11178479, 0.11403719, 0.11630941, 0.11858715, 0.12086028,
 0.12312162, 0.12536613, 0.12759032, 0.12979178, 0.13196896, 0.1341209,
 0.13624707, 0.1383473 , 0.14042163, 0.1424703  ,0.14449366, 0.14649215,
 0.14846629, 0.15041661, 0.15234371, 0.15424815, 0.15613054, 0.15799147,
 0.15983151, 0.16165123, 0.1634512 , 0.16523196, 0.16699404, 0.16873794,
 0.17046417, 0.17217319, 0.17386548, 0.17554147, 0.17720159, 0.17884625,
 0.18047586, 0.18209078, 0.1836914 , 0.18527806, 0.18685108, 0.1884109,
 0.18995766, 0.1914919 , 0.19301363])

rHeston_2_010 = np.array([0.58304915, 0.5811093 , 0.57916276, 0.57720949, 0.57524942, 0.57328247,
 0.57130856, 0.56932762, 0.56733958, 0.56534436, 0.56334189, 0.56133207,
 0.55931483, 0.5572901 , 0.55525777, 0.55321777, 0.55117002, 0.54911441,
 0.54705087, 0.54497929, 0.54289959, 0.54081166, 0.53871541, 0.53661074,
 0.53449755, 0.53237573, 0.53024517, 0.52810578, 0.52595743, 0.52380001,
 0.52163342, 0.51945752, 0.51727221, 0.51507735, 0.51287282, 0.5106585,
 0.50843425, 0.50619993, 0.50395542, 0.50170056, 0.49943523, 0.49715926,
 0.49487251, 0.49257482, 0.49026604, 0.48794601, 0.48561456, 0.48327151,
 0.48091671, 0.47854996, 0.47617109, 0.47377992, 0.47137624, 0.46895986,
 0.46653057, 0.46408818, 0.46163247, 0.45916322, 0.4566802  ,0.45418319,
 0.45167194, 0.44914622, 0.44660576, 0.44405032, 0.44147963, 0.43889341,
 0.43629137, 0.43367324, 0.43103871, 0.42838747, 0.4257192 , 0.42303358,
 0.42033026, 0.41760889, 0.41486912, 0.41211057, 0.40933285, 0.40653556,
 0.40371829, 0.40088062, 0.39802209, 0.39514224, 0.39224061, 0.38931669,
 0.38636998, 0.38339994, 0.38040602, 0.37738764, 0.3743442 , 0.37127509,
 0.36817966, 0.36505722, 0.36190708, 0.3587285 , 0.35552072, 0.35228294,
 0.34901431, 0.34571397, 0.34238099, 0.33901442, 0.33561325, 0.33217642,
 0.32870282, 0.32519129, 0.3216406 , 0.31804946, 0.3144165 , 0.31074029,
 0.30701931, 0.30325196, 0.29943654, 0.29557125, 0.29165419, 0.28768333,
 0.28365653, 0.2795715 , 0.27542581, 0.27121687, 0.26694191, 0.26259798,
 0.25818191, 0.25369034, 0.24911964, 0.24446595, 0.23972511, 0.23489268,
 0.22996387, 0.22493359, 0.21979636, 0.21454635, 0.20917735, 0.20368279,
 0.19805579, 0.19228922, 0.18637588, 0.18030874, 0.17408144, 0.1676891,
 0.16112966, 0.15440627, 0.14753135, 0.1405337 , 0.13347065, 0.12644822,
 0.11965073, 0.1133714 , 0.10800589, 0.10394446, 0.101379  , 0.10020954,
 0.10015134, 0.10089466, 0.1021871 , 0.10384664, 0.10574818, 0.10780698,
 0.10996561, 0.11218486, 0.11443775, 0.11670558, 0.1189753 , 0.12123782,
 0.12348679, 0.12571781, 0.12792789, 0.13011505, 0.13227801, 0.13441606,
 0.13652883, 0.13861627, 0.14067852, 0.14271584, 0.14472864, 0.14671736,
 0.14868252, 0.15062466, 0.15254432, 0.15444208, 0.15631849, 0.15817413,
 0.16000955, 0.16182528, 0.16362186, 0.16539979, 0.16715959, 0.16890174,
 0.1706267 , 0.17233492, 0.17402685, 0.1757029 , 0.17736349, 0.17900899,
 0.1806398 , 0.18225627, 0.18385876, 0.1854476 , 0.18702312, 0.1885857,
 0.19013547, 0.19167288, 0.19319824])

rHeston_3_010 = np.array([0.58254701, 0.58059925, 0.57864486, 0.57668376, 0.57471589, 0.57274117,
 0.57075955, 0.56877093, 0.56677525, 0.56477244, 0.56276241, 0.56074508,
 0.55872039, 0.55668823, 0.55464855, 0.55260123, 0.55054621, 0.54848339,
 0.54641268, 0.544334  , 0.54224724, 0.54015232, 0.53804914, 0.53593759,
 0.53381758, 0.53168901, 0.52955177, 0.52740576, 0.52525085, 0.52308696,
 0.52091395, 0.51873172, 0.51654014, 0.51433909, 0.51212846, 0.50990811,
 0.50767792, 0.50543775, 0.50318746, 0.50092693, 0.498656  , 0.49637454,
 0.49408239, 0.49177941, 0.48946544, 0.48714032, 0.48480388, 0.48245597,
 0.48009641, 0.47772502, 0.47534163, 0.47294606, 0.4705381 , 0.46811758,
 0.46568429, 0.46323803, 0.46077859, 0.45830576, 0.45581931, 0.45331901,
 0.45080464, 0.44827595, 0.4457327 , 0.44317464, 0.4406015 , 0.43801301,
 0.4354089 , 0.43278889, 0.43015267, 0.42749995, 0.42483041, 0.42214373,
 0.41943959, 0.41671763, 0.4139775 , 0.41121883, 0.40844125, 0.40564436,
 0.40282777, 0.39999104, 0.39713374, 0.39425544, 0.39135564, 0.38843389,
 0.38548966, 0.38252245, 0.3795317 , 0.37651687, 0.37347735, 0.37041254,
 0.36732181, 0.3642045 , 0.36105992, 0.35788735, 0.35468604, 0.35145521,
 0.34819404, 0.34490166, 0.3415772 , 0.33821969, 0.33482817, 0.33140159,
 0.32793887, 0.32443887, 0.32090039, 0.31732215, 0.31370283, 0.31004101,
 0.30633522, 0.30258387, 0.29878529, 0.29493774, 0.29103932, 0.28708805,
 0.28308182, 0.27901836, 0.27489529, 0.27071003, 0.26645984, 0.26214179,
 0.25775274, 0.25328931, 0.2487479 , 0.2441246 , 0.23941524, 0.23461532,
 0.22971998, 0.22472401, 0.21962178, 0.21440727, 0.20907399, 0.20361506,
 0.19802317, 0.19229066, 0.18640968, 0.18037243, 0.1741716 , 0.16780118,
 0.16125782, 0.1545432 , 0.14766822, 0.14066026, 0.13357584, 0.12652162,
 0.11968568, 0.11336988, 0.10798266, 0.1039234 , 0.10138009, 0.10023923,
 0.10020476, 0.10096237, 0.10225962, 0.10391616, 0.10580889, 0.10785487,
 0.10999808, 0.11220041, 0.1144357 , 0.11668583, 0.1189382 , 0.12118398,
 0.12341702, 0.12563304, 0.12782911, 0.13000327, 0.13215425, 0.1342813,
 0.13638404, 0.13846236, 0.14051635, 0.14254625, 0.1445524 , 0.1465352,
 0.14849512, 0.15043264, 0.15234829, 0.15424259, 0.15611606, 0.15796923,
 0.15980263, 0.16161677, 0.16341214, 0.16518924, 0.16694854, 0.16869051,
 0.17041558, 0.1721242 , 0.17381679, 0.17549374, 0.17715545, 0.1788023,
 0.18043465, 0.18205286, 0.18365725, 0.18524819, 0.18682594, 0.18839087,
 0.18994314, 0.19148327, 0.19301129])

rHeston_4_010 = np.array([0.58251347, 0.58055991, 0.57859971, 0.57663283, 0.57465918, 0.57267869,
 0.57069129, 0.56869691, 0.56669548, 0.56468692, 0.56267115, 0.5606481,
 0.55861768, 0.55657981, 0.55453442, 0.55248142, 0.55042072, 0.54835223,
 0.54627587, 0.54419154, 0.54209916, 0.53999862, 0.53788984, 0.53577271,
 0.53364714, 0.53151302, 0.52937025, 0.52721872, 0.52505833, 0.52288896,
 0.5207105 , 0.51852284, 0.51632586, 0.51411944, 0.51190345, 0.50967777,
 0.50744228, 0.50519684, 0.50294131, 0.50067556, 0.49839946, 0.49611285,
 0.49381559, 0.49150753, 0.48918852, 0.48685839, 0.484517  , 0.48216417,
 0.47979973, 0.47742351, 0.47503534, 0.47263502, 0.47022238, 0.46779723,
 0.46535936, 0.46290857, 0.46044467, 0.45796742, 0.45547663, 0.45297205,
 0.45045347, 0.44792064, 0.44537332, 0.44281126, 0.4402342 , 0.43764187,
 0.43503401, 0.43241033, 0.42977053, 0.42711433, 0.4244414 , 0.42175144,
 0.41904411, 0.41631908, 0.41357599, 0.41081448, 0.40803418, 0.4052347,
 0.40241564, 0.39957659, 0.39671711, 0.39383676, 0.39093509, 0.38801161,
 0.38506583, 0.38209723, 0.37910528, 0.37608943, 0.37304909, 0.36998367,
 0.36689253, 0.36377504, 0.3606305 , 0.35745822, 0.35425745, 0.35102741,
 0.34776731, 0.34447629, 0.34115348, 0.33779794, 0.33440871, 0.33098476,
 0.32752503, 0.32402839, 0.32049365, 0.31691956 ,0.31330482, 0.30964803,
 0.30594772, 0.30220235, 0.29841027, 0.29456974, 0.29067891, 0.28673582,
 0.28273837, 0.27868434, 0.27457136, 0.27039689, 0.26615822, 0.26185244,
 0.25747644, 0.25302687, 0.24850015, 0.24389239, 0.23919943, 0.23441677,
 0.22953956, 0.22456255, 0.21948011, 0.21428612, 0.20897404, 0.20353683,
 0.19796701, 0.19225671, 0.18639774, 0.18038191, 0.17420139, 0.16784951,
 0.16132213, 0.15461996, 0.14775281, 0.14074693, 0.13365788, 0.12659211,
 0.11973911, 0.11340491, 0.1080045 , 0.10394186, 0.10140248, 0.10026586,
 0.10023103, 0.10098258, 0.10226911, 0.10391181, 0.10578897, 0.10781866,
 0.10994558, 0.1121321 , 0.11435233, 0.11658835, 0.11882761, 0.12106132,
 0.12328332, 0.12548929, 0.12767625, 0.12984217, 0.13198575, 0.13410616,
 0.13620296, 0.138276  , 0.1403253 , 0.14235106, 0.14435356, 0.14633317,
 0.14829032, 0.15022547, 0.15213908, 0.15403167, 0.15590373, 0.15775576,
 0.15958826, 0.16140173, 0.16319665, 0.16497348, 0.16673269, 0.16847472,
 0.17020002, 0.171909  , 0.17360206, 0.17527962, 0.17694204, 0.17858969,
 0.18022294, 0.18184213, 0.18344758, 0.1850396 , 0.18661857, 0.18818471,
 0.18973841, 0.19127985, 0.19280926])

rHeston_5_010 = np.array([0.58260124, 0.58064404, 0.57868019, 0.57670963, 0.57473232, 0.57274816,
 0.57075709, 0.56875903, 0.5667539 , 0.56474165, 0.56272219, 0.56069543,
 0.55866131, 0.55661973, 0.55457063, 0.55251391, 0.55044948, 0.54837727,
 0.54629718, 0.54420913, 0.54211301, 0.54000875, 0.53789623, 0.53577536,
 0.53364605, 0.5315082 , 0.52936169, 0.52720642, 0.52504229, 0.52286918,
 0.52068698, 0.51849559, 0.51629487, 0.51408472, 0.511865  , 0.5096356,
 0.50739638, 0.50514722, 0.50288798, 0.50061852, 0.49833871, 0.49604841,
 0.49374746, 0.49143572, 0.48911304, 0.48677925, 0.4844342 , 0.48207773,
 0.47970966, 0.47732983, 0.47493805, 0.47253414, 0.47011793, 0.46768921,
 0.4652478 , 0.46279349, 0.46032608, 0.45784535, 0.45535109, 0.45284308,
 0.45032108, 0.44778486, 0.44523417, 0.44266878, 0.44008841, 0.43749281,
 0.43488171, 0.43225482, 0.42961186, 0.42695252, 0.42427651, 0.4215835,
 0.41887317, 0.41614518, 0.41339919, 0.41063483, 0.40785173, 0.40504951,
 0.40222777, 0.3993861 , 0.39652407, 0.39364124, 0.39073716, 0.38781136,
 0.38486333, 0.38189257, 0.37889854, 0.37588071, 0.37283849, 0.36977129,
 0.36667848, 0.36355943, 0.36041346, 0.35723987, 0.35403792, 0.35080684,
 0.34754585, 0.34425409, 0.3409307 , 0.33757476, 0.3341853 , 0.33076131,
 0.32730174, 0.32380546, 0.32027131, 0.31669805, 0.31308437, 0.30942891,
 0.3057302 , 0.30198671, 0.29819682, 0.29435879, 0.2904708 , 0.2865309,
 0.28253702, 0.27848695, 0.27437835, 0.2702087 , 0.2659753 , 0.26167527,
 0.25730554, 0.25286276, 0.24834338, 0.24374354, 0.2390591 , 0.23428558,
 0.22941813, 0.22445153, 0.21938013, 0.21419783, 0.20889804, 0.20347369,
 0.19791724, 0.1922207 , 0.18637576, 0.18037402, 0.17420739, 0.16786885,
 0.16135379, 0.15466235, 0.14780362, 0.14080306, 0.13371541, 0.12664662,
 0.1197865 , 0.11344293, 0.10803424, 0.10396681, 0.10142484, 0.10028431,
 0.10024219, 0.10098321, 0.10225706 ,0.10388607, 0.10574937, 0.10776558,
 0.10987975, 0.11205441, 0.11426376, 0.11648985, 0.11872012, 0.12094573,
 0.12316045, 0.12535989, 0.12754102, 0.12970174, 0.13184067, 0.13395695,
 0.13605008, 0.13811986, 0.14016627, 0.14218947, 0.14418972, 0.14616734,
 0.14812275, 0.15005636, 0.15196865, 0.15386008 ,0.15573114, 0.15758232,
 0.15941409, 0.16122695, 0.16302135, 0.16479777, 0.16655665, 0.16829843,
 0.17002354, 0.17173239, 0.17342539, 0.17510292, 0.17676536, 0.17841308,
 0.18004642, 0.18166573, 0.18327134, 0.18486357, 0.18644269, 0.18800908,
 0.18956299, 0.19110469, 0.19263433])

rHeston_obs_2_010 = np.array([0.49556916, 0.49407051, 0.49256615, 0.49105635, 0.48954126,
       0.48802182, 0.48649593, 0.48496533, 0.48342846, 0.48188607,
       0.4803388 , 0.47878537, 0.47722615, 0.47566117, 0.47409042,
       0.47251361, 0.47093102, 0.46934204, 0.46774694, 0.4661457 ,
       0.46453808, 0.46292417, 0.46130377, 0.45967681, 0.45804323,
       0.45640299, 0.45475593, 0.45310201, 0.45144117, 0.44977329,
       0.44809827, 0.44641602, 0.44472647, 0.4430295 , 0.44132503,
       0.43961294, 0.43789315, 0.43616554, 0.43443   , 0.43268643,
       0.43093472, 0.42917474, 0.42740638, 0.42562952, 0.42384404,
       0.42204981, 0.4202467 , 0.41843457, 0.41661329, 0.41478273,
       0.41294273, 0.41109315, 0.40923384, 0.40736465, 0.40548541,
       0.40359596, 0.40169613, 0.39978576, 0.39786466, 0.39593265,
       0.39398954, 0.39203514, 0.39006926, 0.38809168, 0.3861022 ,
       0.38410059, 0.38208664, 0.38006012, 0.37802078, 0.37596838,
       0.37390266, 0.37182337, 0.36973024, 0.36762297, 0.36550129,
       0.3633649 , 0.36121348, 0.35904671, 0.35686426, 0.35466579,
       0.35245094, 0.35021934, 0.3479706 , 0.34570432, 0.34342009,
       0.34111747, 0.33879601, 0.33645525, 0.33409469, 0.33171383,
       0.32931213, 0.32688904, 0.32444398, 0.32197634, 0.31948548,
       0.31697074, 0.31443141, 0.31186677, 0.30927605, 0.30665843,
       0.30401306, 0.30133904, 0.29863542, 0.2959012 , 0.29313532,
       0.29033665, 0.28750401, 0.28463614, 0.28173168, 0.2787892 ,
       0.27580718, 0.27278398, 0.26971786, 0.26660694, 0.26344921,
       0.2602425 , 0.25698446, 0.25367257, 0.25030407, 0.24687599,
       0.24338505, 0.23982771, 0.23620005, 0.23249776, 0.22871611,
       0.22484981, 0.220893  , 0.21683912, 0.2126808 , 0.20840973,
       0.20401649, 0.19949037, 0.19481914, 0.18998879, 0.18498329,
       0.17978435, 0.17437127, 0.16872115, 0.16280969, 0.15661342,
       0.15011473, 0.14331252, 0.13624396, 0.12902617, 0.12192619,
       0.11543497, 0.11021176, 0.1067384 , 0.10498461, 0.10455166,
       0.10500514, 0.10602179, 0.10739106, 0.10898104, 0.11070909,
       0.11252267, 0.11438769, 0.11628156, 0.11818903, 0.12009961,
       0.12200606, 0.12390327, 0.12578769, 0.12765684, 0.12950901,
       0.13134304, 0.13315822, 0.13495412, 0.13673053, 0.13848742,
       0.1402249 , 0.14194314, 0.1436424 , 0.14532298, 0.14698523,
       0.14862951, 0.15025619, 0.15186566, 0.15345832, 0.15503455,
       0.15659474, 0.15813928, 0.15966854, 0.1611829 , 0.16268271,
       0.16416833, 0.1656401 , 0.16709835, 0.16854341, 0.16997558,
       0.17139518, 0.1728025 , 0.17419787, 0.17558146, 0.17695357,
       0.17831458, 0.17966451, 0.18100425, 0.18233268, 0.18365118,
       0.1849601 ])

rHeston_obs_4_010 = np.array([0.53729266, 0.53555209, 0.53380556, 0.53205328, 0.53029499,
       0.52853077, 0.5267604 , 0.52498397, 0.52320132, 0.5214124 ,
       0.51961714, 0.51781551, 0.51600745, 0.51419281, 0.51237158,
       0.51054367, 0.50870901, 0.5068675 , 0.5050191 , 0.50316371,
       0.50130126, 0.49943166, 0.49755483, 0.49567068, 0.49377911,
       0.49188007, 0.48997343, 0.48805912, 0.48613703, 0.48420708,
       0.48226917, 0.48032319, 0.47836904, 0.47640661, 0.47443581,
       0.47245652, 0.47046862, 0.46847202, 0.46646657, 0.46445218,
       0.46242871, 0.46039605, 0.45835405, 0.4563026 , 0.45424156,
       0.45217079, 0.45009015, 0.4479995 , 0.44589868, 0.44378755,
       0.44166596, 0.43953374, 0.43739073, 0.43523676, 0.43307167,
       0.43089527, 0.42870739, 0.42650783, 0.42429641, 0.42207294,
       0.4198372 , 0.417589  , 0.41532811, 0.41305432, 0.41076741,
       0.40846714, 0.40615327, 0.40382555, 0.40148373, 0.39912755,
       0.39675674, 0.39437101, 0.39197008, 0.38955365, 0.38712141,
       0.38467305, 0.38220823, 0.37972661, 0.37722785, 0.37471157,
       0.37217741, 0.36962496, 0.36705382, 0.36446356, 0.36185376,
       0.35922396, 0.35657367, 0.35390242, 0.35120968, 0.34849493,
       0.34575761, 0.34299714, 0.34021292, 0.3374043 , 0.33457064,
       0.33171125, 0.32882539, 0.32591232, 0.32297123, 0.3200013 ,
       0.31700165, 0.31397137, 0.31090947, 0.30781494, 0.30468671,
       0.30152363, 0.29832451, 0.29508808, 0.29181297, 0.28849777,
       0.28514095, 0.2817409 , 0.27829588, 0.27480406, 0.27126347,
       0.26767199, 0.26402737, 0.26032717, 0.25656878, 0.25274935,
       0.24886583, 0.2449149 , 0.24089296, 0.23679609, 0.23261999,
       0.22835998, 0.22401091, 0.21956711, 0.21502234, 0.21036971,
       0.20560157, 0.20070945, 0.19568395, 0.19051468, 0.18519015,
       0.17969779, 0.17402408, 0.1681549 , 0.16207645, 0.15577711,
       0.1492514 , 0.1425078 , 0.13558431, 0.12857782, 0.12169345,
       0.11530161, 0.1099243 , 0.10602597, 0.10372344, 0.10277705,
       0.10282499, 0.10355273, 0.10473387, 0.10621587, 0.10789755,
       0.10971144, 0.11161193, 0.11356771, 0.11555703, 0.11756452,
       0.11957927, 0.12159341, 0.12360131, 0.12559888, 0.1275832 ,
       0.12955217, 0.13150432, 0.13343864, 0.13535449, 0.13725147,
       0.13912938, 0.14098819, 0.14282795, 0.14464884, 0.14645106,
       0.14823488, 0.1500006 , 0.15174855, 0.15347907, 0.15519251,
       0.15688922, 0.15856957, 0.16023391, 0.16188261, 0.163516  ,
       0.16513443, 0.16673826, 0.16832779, 0.16990337, 0.17146531,
       0.1730139 , 0.17454945, 0.17607226, 0.17758259, 0.17908072,
       0.18056698, 0.18204153, 0.18350464, 0.18495668, 0.18639769,
       0.18782775])

rHeston_obs_8_010 = np.array([0.57146395, 0.56956977, 0.56766909, 0.5657619 , 0.56384811,
       0.56192768, 0.56000049, 0.55806652, 0.55612567, 0.55417786,
       0.55222305, 0.55026112, 0.54829202, 0.54631566, 0.54433196,
       0.54234085, 0.54034222, 0.53833601, 0.53632212, 0.53430047,
       0.53227096, 0.5302335 , 0.52818799, 0.52613436, 0.52407248,
       0.52200227, 0.51992362, 0.51783642, 0.51574058, 0.51363599,
       0.51152253, 0.50940009, 0.50726855, 0.5051278 , 0.50297772,
       0.50081818, 0.49864906, 0.49647024, 0.49428157, 0.49208292,
       0.48987417, 0.48765516, 0.48542575, 0.4831858 , 0.48093515,
       0.47867366, 0.47640116, 0.47411749, 0.47182248, 0.46951597,
       0.46719778, 0.46486774, 0.46252565, 0.46017134, 0.4578046 ,
       0.45542524, 0.45303306, 0.45062786, 0.4482094 , 0.44577748,
       0.44333186, 0.44087233, 0.43839862, 0.43591051, 0.43340773,
       0.43089002, 0.42835713, 0.42580876, 0.42324463, 0.42066445,
       0.41806792, 0.41545473, 0.41282454, 0.41017703, 0.40751186,
       0.40482866, 0.40212707, 0.39940671, 0.39666719, 0.3939081 ,
       0.39112901, 0.38832949, 0.38550909, 0.38266734, 0.37980375,
       0.37691781, 0.374009  , 0.37107677, 0.36812056, 0.36513977,
       0.36213378, 0.35910197, 0.35604365, 0.35295813, 0.34984469,
       0.34670256, 0.34353095, 0.34032902, 0.33709591, 0.3338307 ,
       0.33053244, 0.32720012, 0.32383269, 0.32042903, 0.31698798,
       0.31350831, 0.30998871, 0.30642783, 0.30282421, 0.29917632,
       0.29548253, 0.29174113, 0.28795029, 0.28410806, 0.28021238,
       0.27626104, 0.27225169, 0.26818181, 0.26404872, 0.25984951,
       0.2555811 , 0.25124016, 0.24682309, 0.24232603, 0.2377448 ,
       0.23307489, 0.22831143, 0.22344911, 0.21848221, 0.21340454,
       0.20820938, 0.20288951, 0.19743717, 0.19184409, 0.1861016 ,
       0.18020082, 0.17413302, 0.16789034, 0.16146704, 0.15486181,
       0.14808184, 0.14115024, 0.13411916, 0.1270923 , 0.12025916,
       0.11393162, 0.1085366 , 0.1044868 , 0.10196188, 0.10082765,
       0.10077614, 0.10149361, 0.1027337 , 0.10432241, 0.10614155,
       0.1081115 , 0.11017837, 0.11230532, 0.11446695, 0.11664553,
       0.1188286 , 0.1210074 , 0.12317572, 0.12532921, 0.1274648 ,
       0.12958043, 0.13167469, 0.13374673, 0.13579603, 0.13782239,
       0.1398258 , 0.14180639, 0.14376443, 0.14570023, 0.1476142 ,
       0.14950677, 0.15137837, 0.1532295 , 0.15506062, 0.15687222,
       0.15866477, 0.16043875, 0.16219463, 0.16393286, 0.16565388,
       0.16735814, 0.16904605, 0.17071802, 0.17237445, 0.17401573,
       0.17564222, 0.17725428, 0.17885226, 0.18043652, 0.18200735,
       0.18356505, 0.18511   , 0.18664245, 0.18816264, 0.18967081,
       0.19116749])

rHeston_obs_16_010 = np.array([0.57651092, 0.57458339, 0.57264931, 0.5707086 , 0.56876123,
       0.56680709, 0.56484613, 0.56287827, 0.56090344, 0.55892157,
       0.55693258, 0.55493639, 0.55293292, 0.5509221 , 0.54890385,
       0.54687808, 0.5448447 , 0.54280363, 0.54075479, 0.53869808,
       0.53663341, 0.53456069, 0.53247983, 0.53039073, 0.52829328,
       0.5261874 , 0.52407298, 0.5219499 , 0.51981808, 0.51767739,
       0.51552773, 0.51336899, 0.51120104, 0.50902378, 0.50683707,
       0.5046408 , 0.50243484, 0.50021906, 0.49799333, 0.49575751,
       0.49351147, 0.49125507, 0.48898816, 0.4867106 , 0.48442223,
       0.48212289, 0.47981244, 0.47749071, 0.47515753, 0.47281273,
       0.47045614, 0.46808759, 0.46570687, 0.46331382, 0.46090823,
       0.45848991, 0.45605866, 0.45361426, 0.4511565 , 0.44868517,
       0.44620003, 0.44370085, 0.4411874 , 0.43865943, 0.43611669,
       0.4335589 , 0.43098582, 0.42839716, 0.42579263, 0.42317195,
       0.42053482, 0.41788091, 0.41520991, 0.41252149, 0.4098153 ,
       0.407091  , 0.40434821, 0.40158655, 0.39880564, 0.39600508,
       0.39318443, 0.39034328, 0.38748116, 0.38459762, 0.38169217,
       0.3787643 , 0.3758135 , 0.37283923, 0.36984092, 0.36681798,
       0.36376981, 0.36069578, 0.35759522, 0.35446744, 0.35131173,
       0.34812732, 0.34491344, 0.34166927, 0.33839394, 0.33508656,
       0.33174617, 0.3283718 , 0.32496239, 0.32151686, 0.31803406,
       0.31451277, 0.31095172, 0.30734956, 0.30370486, 0.30001613,
       0.29628175, 0.29250005, 0.28866923, 0.28478737, 0.28085246,
       0.27686232, 0.27281465, 0.26870698, 0.26453668, 0.2603009 ,
       0.25599661, 0.25162054, 0.24716918, 0.24263874, 0.23802511,
       0.23332386, 0.22853021, 0.22363896, 0.21864448, 0.21354066,
       0.20832091, 0.20297808, 0.19750453, 0.19189206, 0.18613209,
       0.18021578, 0.17413443, 0.16788014, 0.1614471 , 0.15483382,
       0.14804717, 0.1411098 , 0.1340731 , 0.12703965, 0.1201973 ,
       0.11385572, 0.10843966, 0.10436132, 0.1018028 , 0.10063399,
       0.1005502 , 0.10123935, 0.10245561, 0.10402494, 0.10582884,
       0.1077873 , 0.10984603, 0.11196784, 0.11412695, 0.11630531,
       0.1184902 , 0.12067257, 0.12284599, 0.12500591, 0.1271491 ,
       0.12927331, 0.13137702, 0.13345925, 0.13551937, 0.1375571 ,
       0.13957233, 0.14156514, 0.14353571, 0.14548434, 0.14741135,
       0.14931715, 0.15120214, 0.15306676, 0.15491148, 0.15673673,
       0.158543  , 0.16033071, 0.16210034, 0.16385232, 0.16558708,
       0.16730504, 0.16900663, 0.17069224, 0.17236226, 0.17401707,
       0.17565704, 0.17728251, 0.17889385, 0.18049137, 0.1820754 ,
       0.18364626, 0.18520423, 0.18674961, 0.18828278, 0.18980379,
       0.19131297])

rHeston_obs_32_010 = np.array([0.58135997, 0.57940746, 0.57744831, 0.57548247, 0.57350985,
       0.5715304 , 0.56954403, 0.56755068, 0.56555027, 0.56354273,
       0.56152798, 0.55950594, 0.55747653, 0.55543967, 0.55339529,
       0.55134329, 0.54928359, 0.5472161 , 0.54514074, 0.54305742,
       0.54096603, 0.5388665 , 0.53675872, 0.53464259, 0.53251802,
       0.53038491, 0.52824314, 0.52609262, 0.52393324, 0.52176489,
       0.51958746, 0.51740082, 0.51520487, 0.51299948, 0.51078454,
       0.50855991, 0.50632548, 0.5040811 , 0.50182665, 0.49956199,
       0.49728699, 0.49500149, 0.49270535, 0.49039843, 0.48808058,
       0.48575162, 0.48341142, 0.48105979, 0.47869658, 0.47632161,
       0.47393471, 0.4715357 , 0.46912438, 0.46670057, 0.46426408,
       0.46181471, 0.45935224, 0.45687648, 0.45438719, 0.45188417,
       0.44936718, 0.44683599, 0.44429035, 0.44173003, 0.43915475,
       0.43656427, 0.4339583 , 0.43133658, 0.4286988 , 0.42604469,
       0.42337392, 0.4206862 , 0.41798119, 0.41525855, 0.41251795,
       0.40975903, 0.40698141, 0.40418471, 0.40136855, 0.39853251,
       0.39567616, 0.39279908, 0.38990081, 0.38698088, 0.38403879,
       0.38107405, 0.37808613, 0.37507448, 0.37203854, 0.36897771,
       0.36589139, 0.36277893, 0.35963966, 0.3564729 , 0.35327791,
       0.35005394, 0.3468002 , 0.34351587, 0.34020007, 0.3368519 ,
       0.33347041, 0.33005461, 0.32660344, 0.32311582, 0.31959057,
       0.31602649, 0.3124223 , 0.30877663, 0.30508806, 0.30135507,
       0.29757607, 0.29374936, 0.28987314, 0.28594549, 0.2819644 ,
       0.27792768, 0.27383303, 0.26967797, 0.26545988, 0.26117593,
       0.25682307, 0.25239805, 0.24789737, 0.24331725, 0.23865362,
       0.23390207, 0.22905784, 0.2241158 , 0.21907036, 0.2139155 ,
       0.2086447 , 0.20325097, 0.19772676, 0.19206411, 0.18625463,
       0.18028979, 0.17416125, 0.16786159, 0.16138554, 0.15473235,
       0.14790977, 0.14094144, 0.13387985, 0.12682838, 0.11997487,
       0.11362721, 0.10820655, 0.10412242, 0.1015585 , 0.10038853,
       0.10030986, 0.10100997, 0.1022416 , 0.10382929, 0.10565343,
       0.10763324, 0.10971389, 0.11185784, 0.11403909, 0.11623943,
       0.11844605, 0.12064985, 0.12284435, 0.12502499, 0.12718852,
       0.1293327 , 0.13145601, 0.13355747, 0.13563648, 0.13769274,
       0.13972618, 0.14173687, 0.14372502, 0.14569092, 0.14763492,
       0.14955742, 0.15145886, 0.15333967, 0.15520032, 0.15704127,
       0.158863  , 0.16066597, 0.16245062, 0.16421743, 0.16596681,
       0.16769921, 0.16941505, 0.17111472, 0.17279863, 0.17446717,
       0.1761207 , 0.17775957, 0.17938416, 0.18099477, 0.18259176,
       0.18417542, 0.18574606, 0.18730399, 0.18884949, 0.19038288,
       0.19190447])

k_001 = np.linspace(-0.35, 0.12, 201)

rHeston_001 = np.array([0.56283269, 0.5610481 , 0.55925809, 0.55746257, 0.5556606 , 0.55385223,
 0.55203892, 0.550219  , 0.54839257, 0.54655984, 0.54472131, 0.54287657,
 0.54102504, 0.53916709, 0.53730277, 0.5354316 , 0.53355381, 0.53166912,
 0.52977767, 0.52787922, 0.52597387, 0.52406141, 0.52214179, 0.52021494,
 0.51828071, 0.51633911, 0.51439002, 0.51243333, 0.51046899, 0.50849688,
 0.50651692, 0.504529  , 0.50253306, 0.50052897, 0.49851663, 0.49649595,
 0.49446681, 0.49242912, 0.49038276, 0.48832762, 0.4862636 , 0.48419056,
 0.4821084 , 0.48001699, 0.47791621, 0.47580593, 0.47368602 ,0.47155635,
 0.46941678, 0.46726718, 0.46510739, 0.46293728, 0.4607567  ,0.45856549,
 0.45636349, 0.45415055, 0.45192651, 0.44969118, 0.4474444  ,0.445186,
 0.44291579, 0.44063357, 0.43833917, 0.43603239, 0.43371301, 0.43138084,
 0.42903566, 0.42667726, 0.42430539, 0.42191984, 0.41952036, 0.41710671,
 0.41467864, 0.41223587, 0.40977815, 0.4073052 , 0.40481673, 0.40231245,
 0.39979205, 0.39725521, 0.39470162, 0.39213094, 0.38954282, 0.38693691,
 0.38431283, 0.3816702 , 0.37900862, 0.37632768, 0.37362696, 0.37090601,
 0.36816438, 0.36540158, 0.36261713, 0.35981051, 0.35698118, 0.35412859,
 0.35125216, 0.34835129, 0.34542535, 0.34247367, 0.33949558, 0.33649037,
 0.33345727, 0.33039552, 0.32730429, 0.32418271, 0.3210299 , 0.31784491,
 0.31462673, 0.31137433, 0.30808661, 0.30476241, 0.3014005 , 0.29799959,
 0.29455832, 0.29107524, 0.28754882, 0.28397744, 0.28035936, 0.27669276,
 0.27297568, 0.26920603, 0.26538157, 0.26149994, 0.25755856, 0.2535547,
 0.24948541, 0.24534751, 0.24113758, 0.23685192, 0.23248653, 0.22803707,
 0.22349884, 0.21886674, 0.2141352 , 0.2092982 , 0.20434917, 0.19928101,
 0.19408602, 0.18875596, 0.18328207, 0.17765521, 0.17186621, 0.16590639,
 0.15976867, 0.15344951, 0.14695241, 0.14029432, 0.13351685, 0.12670571,
 0.12002066, 0.11373025, 0.10821684, 0.10388405, 0.1009643 , 0.09940281,
 0.09894535, 0.09929718, 0.10021202, 0.10150992, 0.10306617, 0.10479573,
 0.10664048, 0.10856041, 0.11052766, 0.11252265, 0.11453146, 0.11654412,
 0.1185535 , 0.12055443, 0.12254322, 0.12451721, 0.12647455, 0.12841394,
 0.13033454, 0.13223581, 0.13411747, 0.1359794 , 0.13782165, 0.13964434,
 0.14144771, 0.143232  , 0.14499754, 0.14674467, 0.14847374, 0.15018513,
 0.15187921, 0.15355638, 0.15521702, 0.1568615 , 0.15849018, 0.16010352,
 0.16170168, 0.1632853 , 0.16485442, 0.16640983, 0.16795069, 0.1694782,
 0.17099095, 0.1724971 , 0.17398089])

rHeston_1_001 = np.array([0.56386411, 0.56210861 ,0.56034916, 0.55858238 ,0.55680997 ,0.55503188,
 0.55324768, 0.55145702, 0.54965998, 0.54785686, 0.54604756, 0.54423153,
 0.54240896, 0.54058023, 0.53874461, 0.53690225, 0.53505316, 0.53319717,
 0.53133433, 0.52946448, 0.5275875 , 0.5257033 , 0.52381182, 0.52191306,
 0.52000687, 0.51809317, 0.51617185, 0.51424286, 0.51230606, 0.51036137,
 0.50840873, 0.50644798, 0.50447908, 0.50250189, 0.50051631, 0.49852224,
 0.49651958, 0.49450819, 0.49248798, 0.49045883, 0.48842062, 0.48637322,
 0.48431651, 0.48225037, 0.48017466, 0.47808925, 0.475994  , 0.47388878,
 0.47177344, 0.46964784, 0.46751182, 0.46536524, 0.46320793, 0.46103973,
 0.45886049, 0.45667002, 0.45446816, 0.45225473, 0.45002954, 0.44779241,
 0.44554314, 0.44328154, 0.44100741, 0.43872053, 0.4364207 , 0.43410768,
 0.43178126, 0.42944119, 0.42708725, 0.42471918, 0.42233673, 0.41993964,
 0.41752763, 0.41510043, 0.41265775, 0.4101993 , 0.40772476, 0.40523383,
 0.40272618, 0.40020147, 0.39765936, 0.39509948, 0.39252146, 0.38992492,
 0.38730947, 0.38467467, 0.38202012, 0.37934537 ,0.37664995, 0.37393338,
 0.37119519, 0.36843483, 0.3656518 , 0.36284551, 0.3600154 , 0.35716087,
 0.35428127, 0.35137597, 0.34844426, 0.34548545, 0.34249877, 0.33948346,
 0.33643868, 0.3333636 , 0.3302573 , 0.32711886, 0.32394727, 0.32074151,
 0.31750049, 0.31422305, 0.31090799, 0.30755403, 0.30415982, 0.30072395,
 0.2972449 , 0.29372109, 0.29015083, 0.28653233, 0.28286368, 0.27914288,
 0.27536776, 0.27153604, 0.26764528, 0.26369288, 0.25967605, 0.25559182,
 0.25143701, 0.24720822, 0.24290181, 0.23851388, 0.23404026, 0.22947648,
 0.22481778, 0.22005906 ,0.21519489, 0.21021953, 0.20512689, 0.19991059,
 0.19456401, 0.18908038, 0.18345295, 0.17767524, 0.17174153, 0.16564757,
 0.15939184, 0.15297761, 0.14641641, 0.13973398, 0.13298018, 0.12624501,
 0.11968198, 0.11353327, 0.10813212, 0.10383466, 0.1008716 , 0.09923446,
 0.09871674, 0.09904481, 0.09997255, 0.10131184, 0.10292871, 0.10473037,
 0.10665292, 0.10865229, 0.11069786, 0.11276828, 0.11484854, 0.11692809,
 0.11899952, 0.12105763, 0.12309883, 0.12512068, 0.12712155, 0.12910045,
 0.1310568 , 0.13299036, 0.13490112, 0.13678923, 0.13865497, 0.1404987,
 0.14232083, 0.14412183, 0.14590217, 0.14766237, 0.1494029 , 0.15112429,
 0.15282703, 0.1545116 , 0.15617848, 0.15782814, 0.15946104, 0.16107759,
 0.16267824, 0.16426349, 0.16583348, 0.16738857, 0.16892984, 0.17045694,
 0.17197247, 0.17346981, 0.17495632])

rHeston_2_001 = np.array([0.562253 ,  0.56049198, 0.55872379, 0.55695043, 0.55517092, 0.55338541,
 0.5515939 , 0.54979652, 0.54799282, 0.54618283, 0.54436671, 0.54254436,
 0.54071531, 0.5388798 , 0.53703796, 0.53518929, 0.53333408, 0.53147207,
 0.52960324, 0.52772746, 0.52584476, 0.52395491, 0.52205792, 0.52015379,
 0.51824224, 0.51632333, 0.51439692, 0.51246293, 0.51052127, 0.50857185,
 0.50661458, 0.50464935, 0.5026761 , 0.50069469, 0.49870502, 0.49670701,
 0.49470053, 0.49268548, 0.49066176, 0.48862924, 0.48658782, 0.48453736,
 0.48247776, 0.48040888 ,0.4783306 , 0.47624279, 0.47414532, 0.47203806,
 0.46992085, 0.46779357, 0.46565607, 0.46350819, 0.46134979, 0.45918071,
 0.45700078, 0.45480984, 0.45260773, 0.45039427, 0.44816929, 0.44593259,
 0.44368401, 0.44142333, 0.43915037, 0.43686492, 0.43456678, 0.43225573,
 0.42993155, 0.42759402, 0.4252429 , 0.42287795, 0.42049893, 0.41810558,
 0.41569765, 0.41327485, 0.41083691, 0.40838356, 0.40591448, 0.40342938,
 0.40092793, 0.39840982, 0.39587471, 0.39332225, 0.39075207, 0.38816382,
 0.38555709, 0.3829315 , 0.38028663, 0.37762204, 0.37493731, 0.37223195,
 0.3695055 , 0.36675745, 0.3639873 , 0.36119449, 0.35837847, 0.35553866,
 0.35267445, 0.3497852 , 0.34687026, 0.34392893, 0.34096048, 0.33796418,
 0.33493922, 0.33188478, 0.32879999, 0.32568395, 0.3225357 , 0.31935424,
 0.31613851, 0.3128874 , 0.30959974, 0.3062743 , 0.30290977, 0.29950479,
 0.29605788, 0.29256751, 0.28903203, 0.28544972, 0.28181871, 0.27813704,
 0.27440262, 0.2706132 , 0.2667664 , 0.26285964, 0.2588902 , 0.25485512,
 0.25075125, 0.24657519, 0.24232328, 0.23799157, 0.23357581, 0.22907141,
 0.22447342, 0.21977649, 0.21497485, 0.21006228, 0.20503214, 0.19987728,
 0.19459017, 0.18916289, 0.18358729, 0.17785525, 0.1719591 , 0.16589235,
 0.15965098, 0.15323559, 0.14665507, 0.13993294, 0.13311817, 0.12630286,
 0.11964863, 0.11341555, 0.10796459, 0.10367706, 0.10078433, 0.0992512,
 0.0988398 , 0.09925782, 0.10025316, 0.10163862, 0.1032837 , 0.10509948,
 0.1070256 , 0.10902092, 0.11105717, 0.11311479, 0.11518013, 0.11724367,
 0.11929873, 0.12134064, 0.12336616, 0.1253731 , 0.12735998, 0.12932589,
 0.13127029, 0.13319295, 0.13509384, 0.13697309, 0.13883092, 0.14066765,
 0.14248365, 0.14427934, 0.14605515, 0.14781152, 0.14954892, 0.1512678,
 0.15296862, 0.15465184, 0.1563179 , 0.15796721, 0.15960023, 0.16121737,
 0.16281904, 0.16440549, 0.16597727, 0.16753522, 0.16907817, 0.17060725,
 0.17212312, 0.17362812, 0.17511909])

rHeston_3_001 = np.array([0.56208134, 0.56031065, 0.55853441, 0.5567522 , 0.55496393, 0.55316983,
 0.55136967, 0.54956317, 0.54775094, 0.54593265, 0.544108  , 0.54227677,
 0.54043935, 0.53859541, 0.53674494, 0.53488799, 0.53302419, 0.53115365,
 0.52927647, 0.52739227, 0.52550118, 0.52360302, 0.52169771, 0.51978521,
 0.51786545, 0.51593836, 0.51400375, 0.51206164, 0.51011189, 0.50815437,
 0.50618907, 0.50421586, 0.50223465, 0.50024533, 0.49824779, 0.49624195,
 0.4942277 , 0.49220492, 0.49017351, 0.48813335, 0.48608434, 0.48402635,
 0.48195927, 0.47988297, 0.47779733, 0.47570222, 0.47359751, 0.47148306,
 0.46935875, 0.46722442, 0.46507994, 0.46292515, 0.46075992, 0.45858407,
 0.45639747, 0.45419993, 0.4519913 , 0.44977141, 0.44754008, 0.44529714,
 0.44304239, 0.44077566, 0.43849673, 0.43620543, 0.43390154, 0.43158485,
 0.42925514, 0.4269122 , 0.42455579, 0.42218568, 0.41980162, 0.41740337,
 0.41499067, 0.41256325, 0.41012085, 0.40766317, 0.40518993, 0.40270083,
 0.40019556, 0.39767379, 0.39513521, 0.39257946, 0.3900062 , 0.38741505,
 0.38480565, 0.3821776 , 0.37953048, 0.3768639 , 0.3741774 , 0.37147053,
 0.36874283, 0.3659938 , 0.36322295, 0.36042974, 0.35761362, 0.35477402,
 0.35191036, 0.349022  , 0.3461083 , 0.34316858, 0.34020215, 0.33720825,
 0.33418613, 0.33113496, 0.32805391, 0.32494208, 0.32179855, 0.31862233,
 0.31541239, 0.31216765, 0.30888697, 0.30556914, 0.30221288, 0.29881686,
 0.29537965, 0.29189974, 0.28837553, 0.28480533, 0.28118732, 0.27751959,
 0.27380009, 0.27002663, 0.26619686, 0.26230828, 0.25835822, 0.25434377,
 0.25026185, 0.24610911, 0.24188194, 0.23757647, 0.23318847, 0.22871341,
 0.22414635, 0.21948194, 0.2147144 , 0.20983745, 0.20484431, 0.19972769,
 0.19447974, 0.18909218, 0.18355632, 0.17786332, 0.17200454, 0.16597227,
 0.15976097, 0.15336936, 0.14680419, 0.14008672, 0.13326399, 0.12642743,
 0.11974093, 0.11347198, 0.1079943 , 0.10370064, 0.10082192, 0.09931067,
 0.09891675, 0.09934236, 0.10033536, 0.10171069, 0.10334037, 0.10513764,
 0.10704377, 0.10901875, 0.11103509, 0.11307369, 0.11512122, 0.11716828,
 0.11920824, 0.12123642, 0.12324956, 0.12524538, 0.12722233, 0.12917941,
 0.13111601, 0.13303181, 0.1349267 , 0.13680074, 0.1386541 , 0.14048702,
 0.14229983, 0.14409289, 0.14586658, 0.1476213 , 0.14935749, 0.15107557,
 0.15277595, 0.15445907, 0.15612534, 0.15777515, 0.15940903, 0.16102714,
 0.16262996, 0.16421808, 0.16579143, 0.16735112, 0.16889579, 0.17042898,
 0.17194572, 0.17345655, 0.17494845])

rHeston_4_001 = np.array([0.56224217, 0.5604657 , 0.55868268, 0.55689558, 0.5551011 , 0.55330092,
 0.55149486, 0.54968275, 0.54786449, 0.54604002, 0.54420938, 0.54237228,
 0.54052882, 0.53867908, 0.53682272, 0.53495956, 0.53308993, 0.53121353,
 0.52933025, 0.52744018, 0.52554308, 0.52363897, 0.52172773, 0.51980931,
 0.51788357, 0.51595051, 0.51400996, 0.51206186, 0.51010617, 0.50814272,
 0.50617149, 0.50419233, 0.50220517, 0.50020993, 0.49820646, 0.4961947,
 0.49417453, 0.49214585, 0.49010854, 0.4880625 , 0.48600761, 0.48394375,
 0.48187081, 0.47978866, 0.47769718, 0.47559625, 0.47348572, 0.47136548,
 0.46923539, 0.46709529, 0.46494506, 0.46278455, 0.46061361, 0.45843207,
 0.4562398 , 0.45403662, 0.45182237, 0.44959689, 0.44735999, 0.44511151,
 0.44285125, 0.44057903, 0.43829466, 0.43599795, 0.43368868, 0.43136665,
 0.42903164, 0.42668344, 0.42432181, 0.42194653, 0.41955735, 0.41715403,
 0.4147363 , 0.41230392, 0.4098566 , 0.40739407, 0.40491604, 0.40242222,
 0.39991229, 0.39738595 ,0.39484286, 0.39228268, 0.38970507, 0.38710967,
 0.38449609, 0.38186396, 0.37921287, 0.37654241, 0.37385214, 0.37114163,
 0.36841039, 0.36565796, 0.36288383, 0.36008748, 0.35726836, 0.35442592,
 0.35155957, 0.34866869, 0.34575264, 0.34281077, 0.33984236, 0.3368467,
 0.33382302, 0.33077053, 0.32768838, 0.32457571, 0.32143159, 0.31825506,
 0.31504511, 0.31180066, 0.30852059, 0.30520371, 0.30184877, 0.29845445,
 0.29501934, 0.29154196, 0.28802073, 0.28445399, 0.28083996, 0.27717674,
 0.27346232, 0.26969454, 0.26587111, 0.26198955, 0.25804722, 0.25404129,
 0.24996869, 0.24582615, 0.2416101 , 0.2373167 , 0.23294182, 0.22848094,
 0.22392918, 0.21928125, 0.21453139, 0.20967336, 0.20470038, 0.19960513,
 0.19437972, 0.18901572, 0.18350428, 0.17783624, 0.17200256, 0.16599493,
 0.15980698, 0.15343636, 0.14688847, 0.140183  , 0.13336531, 0.12652564,
 0.11982809, 0.11354313, 0.10805053, 0.10374874, 0.1008683 , 0.09935588,
 0.09895642, 0.09937101, 0.10034872, 0.10170632, 0.10331736, 0.10509614,
 0.1069846 , 0.10894311, 0.11094434, 0.11296926, 0.11500449, 0.11704056,
 0.11907076, 0.12109031, 0.12309582, 0.12508494, 0.12705601, 0.12900797,
 0.13094011, 0.13285204, 0.13474361, 0.13661481, 0.13846576, 0.14029666,
 0.1421078 , 0.1438995 , 0.14567212, 0.14742603, 0.14916164, 0.15087935,
 0.15257955, 0.15426266, 0.15592908, 0.1575792 , 0.15921342, 0.16083212,
 0.16243568, 0.16402428, 0.16559882, 0.16715905, 0.16870523, 0.17023817,
 0.17175541, 0.17326537, 0.17475577])

rHeston_5_001 = np.array([0.56243905, 0.56065729, 0.55887214, 0.55708028, 0.55528268, 0.55347943,
 0.55166962, 0.54985408, 0.54803174, 0.546204  , 0.54436989, 0.54252937,
 0.54068212, 0.53882863, 0.53696874, 0.53510197, 0.5332286 , 0.53134862,
 0.52946167, 0.52756783, 0.52566717, 0.52375937, 0.52184436, 0.51992224,
 0.51799281, 0.516056  , 0.51411168, 0.51215987, 0.51020039, 0.50823318,
 0.50625819, 0.50427523, 0.5022843 , 0.50028525, 0.49827798, 0.49626241,
 0.49423843, 0.49220591, 0.49016477, 0.48811489, 0.48605615, 0.48398845,
 0.48191165, 0.47982564, 0.4777303 , 0.47562549, 0.47351109, 0.47138697,
 0.46925299, 0.46710901, 0.46495489, 0.46279049, 0.46061565, 0.45843022,
 0.45623405, 0.45402698, 0.45180883, 0.44957945, 0.44733866, 0.44508628,
 0.44282214, 0.44054603, 0.43825778, 0.43595719, 0.43364404, 0.43131815,
 0.42897928, 0.42662723, 0.42426176, 0.42188265, 0.41948964, 0.41708251,
 0.41466099, 0.41222483, 0.40977375, 0.40730747, 0.40482572, 0.40232818,
 0.39981457, 0.39728456, 0.39473783, 0.39217405, 0.38959285, 0.3869939,
 0.38437681, 0.3817412 , 0.37908667, 0.3764128 , 0.37371918, 0.37100535,
 0.36827085, 0.3655152 , 0.36273792, 0.35993847, 0.35711632, 0.35427092,
 0.35140167, 0.34850798, 0.34558919, 0.34264467, 0.3396737 , 0.33667558,
 0.33364954, 0.3305948 , 0.32751052, 0.32439585, 0.32124986, 0.3180716,
 0.31486006, 0.31161419, 0.30833287, 0.30501492, 0.3016591 , 0.29826411,
 0.29482855, 0.29135096, 0.28782977, 0.28426334, 0.28064989, 0.27698758,
 0.27327438, 0.26950819, 0.26568671, 0.26180752, 0.25786798, 0.25386531,
 0.24979647, 0.24565821, 0.24144701, 0.23715909, 0.23279032, 0.22833625,
 0.22379205, 0.21915245, 0.21441176, 0.20956376, 0.20460172, 0.19951833,
 0.19430571, 0.18895542, 0.18345854, 0.17780583, 0.17198804, 0.1659966,
 0.15982473, 0.15346949, 0.14693549, 0.14024139, 0.1334314 , 0.12659462,
 0.11989463, 0.11360281, 0.10810166, 0.10379248, 0.10090574, 0.09938556,
 0.09897516, 0.09937596, 0.1003383 , 0.10168017, 0.10327599, 0.10504055,
 0.10691602, 0.10886285, 0.11085368, 0.11286938, 0.11489649, 0.11692543,
 0.11894937, 0.12096344, 0.12296415, 0.12494906, 0.12691646, 0.1288652,
 0.13079453, 0.13270402, 0.13459345, 0.13646279, 0.13831212, 0.14014162,
 0.14195155, 0.1437422 , 0.14551392, 0.14726707, 0.14900203, 0.15071918,
 0.15241892, 0.15410165, 0.15576778, 0.15741765, 0.15905167, 0.16067028,
 0.16227363, 0.16386242, 0.16543652, 0.16699682, 0.16854405, 0.17007633,
 0.17159592, 0.17310453, 0.17459738])


def plot_rHeston_optimized_smiles():
    plt.plot(k_100, rHeston_100, 'k-', label='rough Heston')
    plt.plot(k_100, rHeston_1_100, c[0] + '-', label=f'{1}-dimensional approximation')
    plt.plot(k_100, rHeston_2_100, c[1] + '-', label=f'{2}-dimensional approximation')
    plt.plot(k_100, rHeston_3_100, c[2] + '-', label=f'{3}-dimensional approximation')
    plt.plot(k_100, rHeston_4_100, c[3] + '-', label=f'{4}-dimensional approximation')
    plt.plot(k_100, rHeston_5_100, c[4] + '-', label=f'{5}-dimensional approximation')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.legend(loc='upper right')
    plt.show()

    rel_errors_1_100 = np.abs(rHeston_1_100 - rHeston_100) / rHeston_100
    rel_errors_2_100 = np.abs(rHeston_2_100 - rHeston_100) / rHeston_100
    rel_errors_3_100 = np.abs(rHeston_3_100 - rHeston_100) / rHeston_100
    rel_errors_4_100 = np.abs(rHeston_4_100 - rHeston_100) / rHeston_100
    rel_errors_5_100 = np.abs(rHeston_5_100 - rHeston_100) / rHeston_100

    print(np.amax(rel_errors_1_100))
    print(np.amax(rel_errors_2_100))
    print(np.amax(rel_errors_3_100))
    print(np.amax(rel_errors_4_100))
    print(np.amax(rel_errors_5_100))

    plt.plot(k_100, rHeston_100_errors, 'k--', label='estimated discretization error')
    plt.plot(k_100, rel_errors_1_100, c[0] + '-', label=f'{1}-dimensional quadrature error')
    plt.plot(k_100, rel_errors_2_100, c[1] + '-', label=f'{2}-dimensional quadrature error')
    plt.plot(k_100, rel_errors_3_100, c[2] + '-', label=f'{3}-dimensional quadrature error')
    plt.plot(k_100, rel_errors_4_100, c[3] + '-', label=f'{4}-dimensional quadrature error')
    plt.plot(k_100, rel_errors_5_100, c[4] + '-', label=f'{5}-dimensional quadrature error')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='upper right')
    plt.show()

    print(np.amax(rel_errors_1_100[62:-40]))
    print(np.amax(rel_errors_2_100[62:-40]))
    print(np.amax(rel_errors_3_100[62:-40]))
    print(np.amax(rel_errors_4_100[62:-40]))
    print(np.amax(rel_errors_5_100[62:-40]))

    plt.plot(k_100[62:-40], rHeston_100_errors[62:-40], 'k--', label='estimated discretization error')
    plt.plot(k_100[62:-40], rel_errors_1_100[62:-40], c[0] + '-', label=f'{1}-dimensional quadrature error')
    plt.plot(k_100[62:-40], rel_errors_2_100[62:-40], c[1] + '-', label=f'{2}-dimensional quadrature error')
    plt.plot(k_100[62:-40], rel_errors_3_100[62:-40], c[2] + '-', label=f'{3}-dimensional quadrature error')
    plt.plot(k_100[62:-40], rel_errors_4_100[62:-40], c[3] + '-', label=f'{4}-dimensional quadrature error')
    plt.plot(k_100[62:-40], rel_errors_5_100[62:-40], c[4] + '-', label=f'{5}-dimensional quadrature error')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(k_100, rHeston_100, 'k-', label='rough Heston')
    plt.plot(k_100, rHeston_obs_2_100, c[0] + '-', label=f'{2}-dimensional approximation')
    plt.plot(k_100, rHeston_obs_4_100, c[1] + '-', label=f'{4}-dimensional approximation')
    plt.plot(k_100, rHeston_obs_8_100, c[2] + '-', label=f'{8}-dimensional approximation')
    plt.plot(k_100, rHeston_obs_16_100, c[3] + '-', label=f'{16}-dimensional approximation')
    plt.plot(k_100, rHeston_obs_32_100, c[4] + '-', label=f'{32}-dimensional approximation')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.legend(loc='upper right')
    plt.show()

    rel_errors_obs_2_100 = np.abs(rHeston_obs_2_100 - rHeston_100) / rHeston_100
    rel_errors_obs_4_100 = np.abs(rHeston_obs_4_100 - rHeston_100) / rHeston_100
    rel_errors_obs_8_100 = np.abs(rHeston_obs_8_100 - rHeston_100) / rHeston_100
    rel_errors_obs_16_100 = np.abs(rHeston_obs_16_100 - rHeston_100) / rHeston_100
    rel_errors_obs_32_100 = np.abs(rHeston_obs_32_100 - rHeston_100) / rHeston_100

    print(np.amax(rel_errors_obs_2_100))
    print(np.amax(rel_errors_obs_4_100))
    print(np.amax(rel_errors_obs_8_100))
    print(np.amax(rel_errors_obs_16_100))
    print(np.amax(rel_errors_obs_32_100))

    plt.plot(k_100, rel_errors_1_100, 'k-', label='1 dimension, optimized')
    plt.plot(k_100, rel_errors_5_100, color='brown', label='5 dimensions, optimized')
    plt.plot(k_100, rel_errors_obs_2_100, c[0] + '-', label=f'2 dimensions, paper')
    plt.plot(k_100, rel_errors_obs_4_100, c[1] + '-', label=f'4 dimensions, paper')
    plt.plot(k_100, rel_errors_obs_8_100, c[2] + '-', label=f'8 dimensions, paper')
    plt.plot(k_100, rel_errors_obs_16_100, c[3] + '-', label=f'16 dimensions, paper')
    plt.plot(k_100, rel_errors_obs_32_100, c[4] + '-', label=f'32 dimensions, paper')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()

    print(np.amax(rel_errors_obs_2_100[62:-40]))
    print(np.amax(rel_errors_obs_4_100[62:-40]))
    print(np.amax(rel_errors_obs_8_100[62:-40]))
    print(np.amax(rel_errors_obs_16_100[62:-40]))
    print(np.amax(rel_errors_obs_32_100[62:-40]))

    plt.plot(k_010, rHeston_010, 'k-', label='rough Heston')
    plt.plot(k_010, rHeston_1_010, c[0] + '-', label=f'{1}-dimensional approximation')
    plt.plot(k_010, rHeston_2_010, c[1] + '-', label=f'{2}-dimensional approximation')
    plt.plot(k_010, rHeston_3_010, c[2] + '-', label=f'{3}-dimensional approximation')
    plt.plot(k_010, rHeston_4_010, c[3] + '-', label=f'{4}-dimensional approximation')
    plt.plot(k_010, rHeston_5_010, c[4] + '-', label=f'{5}-dimensional approximation')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.legend(loc='upper right')
    plt.show()

    rel_errors_1_010 = np.abs(rHeston_1_010 - rHeston_010) / rHeston_010
    rel_errors_2_010 = np.abs(rHeston_2_010 - rHeston_010) / rHeston_010
    rel_errors_3_010 = np.abs(rHeston_3_010 - rHeston_010) / rHeston_010
    rel_errors_4_010 = np.abs(rHeston_4_010 - rHeston_010) / rHeston_010
    rel_errors_5_010 = np.abs(rHeston_5_010 - rHeston_010) / rHeston_010

    print(np.amax(rel_errors_1_010))
    print(np.amax(rel_errors_2_010))
    print(np.amax(rel_errors_3_010))
    print(np.amax(rel_errors_4_010))
    print(np.amax(rel_errors_5_010))

    plt.plot(k_010, rel_errors_1_010, c[0] + '-', label=f'{1}-dimensional quadrature error')
    plt.plot(k_010, rel_errors_2_010, c[1] + '-', label=f'{2}-dimensional quadrature error')
    plt.plot(k_010, rel_errors_3_010, c[2] + '-', label=f'{3}-dimensional quadrature error')
    plt.plot(k_010, rel_errors_4_010, c[3] + '-', label=f'{4}-dimensional quadrature error')
    plt.plot(k_010, rel_errors_5_010, c[4] + '-', label=f'{5}-dimensional quadrature error')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='upper right')
    plt.show()

    print(np.amax(rel_errors_1_010[:-46]))
    print(np.amax(rel_errors_2_010[:-46]))
    print(np.amax(rel_errors_3_010[:-46]))
    print(np.amax(rel_errors_4_010[:-46]))
    print(np.amax(rel_errors_5_010[:-46]))

    plt.plot(k_010[:-46], rel_errors_1_010[:-46], c[0] + '-', label=f'{1}-dimensional quadrature error')
    plt.plot(k_010[:-46], rel_errors_2_010[:-46], c[1] + '-', label=f'{2}-dimensional quadrature error')
    plt.plot(k_010[:-46], rel_errors_3_010[:-46], c[2] + '-', label=f'{3}-dimensional quadrature error')
    plt.plot(k_010[:-46], rel_errors_4_010[:-46], c[3] + '-', label=f'{4}-dimensional quadrature error')
    plt.plot(k_010[:-46], rel_errors_5_010[:-46], c[4] + '-', label=f'{5}-dimensional quadrature error')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(k_010, rHeston_010, 'k-', label='rough Heston')
    plt.plot(k_010, rHeston_obs_2_010, c[0] + '-', label=f'{2}-dimensional approximation')
    plt.plot(k_010, rHeston_obs_4_010, c[1] + '-', label=f'{4}-dimensional approximation')
    plt.plot(k_010, rHeston_obs_8_010, c[2] + '-', label=f'{8}-dimensional approximation')
    plt.plot(k_010, rHeston_obs_16_010, c[3] + '-', label=f'{16}-dimensional approximation')
    plt.plot(k_010, rHeston_obs_32_010, c[4] + '-', label=f'{32}-dimensional approximation')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.legend(loc='upper right')
    plt.show()

    rel_errors_obs_2_010 = np.abs(rHeston_obs_2_010 - rHeston_010) / rHeston_010
    rel_errors_obs_4_010 = np.abs(rHeston_obs_4_010 - rHeston_010) / rHeston_010
    rel_errors_obs_8_010 = np.abs(rHeston_obs_8_010 - rHeston_010) / rHeston_010
    rel_errors_obs_16_010 = np.abs(rHeston_obs_16_010 - rHeston_010) / rHeston_010
    rel_errors_obs_32_010 = np.abs(rHeston_obs_32_010 - rHeston_010) / rHeston_010

    print(np.amax(rel_errors_obs_2_010))
    print(np.amax(rel_errors_obs_4_010))
    print(np.amax(rel_errors_obs_8_010))
    print(np.amax(rel_errors_obs_16_010))
    print(np.amax(rel_errors_obs_32_010))

    plt.plot(k_010, rel_errors_1_010, 'k-', label='1 dimension, optimized')
    plt.plot(k_010, rel_errors_5_010, color='brown', label='5 dimensions, optimized')
    plt.plot(k_010, rel_errors_obs_2_010, c[0] + '-', label=f'2 dimensions, paper')
    plt.plot(k_010, rel_errors_obs_4_010, c[1] + '-', label=f'4 dimensions, paper')
    plt.plot(k_010, rel_errors_obs_8_010, c[2] + '-', label=f'8 dimensions, paper')
    plt.plot(k_010, rel_errors_obs_16_010, c[3] + '-', label=f'16 dimensions, paper')
    plt.plot(k_010, rel_errors_obs_32_010, c[4] + '-', label=f'32 dimensions, paper')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()

    print(np.amax(rel_errors_obs_2_010[:-46]))
    print(np.amax(rel_errors_obs_4_010[:-46]))
    print(np.amax(rel_errors_obs_8_010[:-46]))
    print(np.amax(rel_errors_obs_16_010[:-46]))
    print(np.amax(rel_errors_obs_32_010[:-46]))

    plt.plot(k_001, rHeston_001, 'k-', label='rough Heston')
    plt.plot(k_001, rHeston_1_001, c[0] + '-', label=f'{1}-dimensional approximation')
    plt.plot(k_001, rHeston_2_001, c[1] + '-', label=f'{2}-dimensional approximation')
    plt.plot(k_001, rHeston_3_001, c[2] + '-', label=f'{3}-dimensional approximation')
    plt.plot(k_001, rHeston_4_001, c[3] + '-', label=f'{4}-dimensional approximation')
    plt.plot(k_001, rHeston_5_001, c[4] + '-', label=f'{5}-dimensional approximation')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.legend(loc='upper right')
    plt.show()

    rel_errors_1_001 = np.abs(rHeston_1_001 - rHeston_001) / rHeston_001
    rel_errors_2_001 = np.abs(rHeston_2_001 - rHeston_001) / rHeston_001
    rel_errors_3_001 = np.abs(rHeston_3_001 - rHeston_001) / rHeston_001
    rel_errors_4_001 = np.abs(rHeston_4_001 - rHeston_001) / rHeston_001
    rel_errors_5_001 = np.abs(rHeston_5_001 - rHeston_001) / rHeston_001

    print(np.amax(rel_errors_1_001))
    print(np.amax(rel_errors_2_001))
    print(np.amax(rel_errors_3_001))
    print(np.amax(rel_errors_4_001))
    print(np.amax(rel_errors_5_001))

    plt.plot(k_001, rel_errors_1_001, c[0] + '-', label=f'{1}-dimensional quadrature error')
    plt.plot(k_001, rel_errors_2_001, c[1] + '-', label=f'{2}-dimensional quadrature error')
    plt.plot(k_001, rel_errors_3_001, c[2] + '-', label=f'{3}-dimensional quadrature error')
    plt.plot(k_001, rel_errors_4_001, c[3] + '-', label=f'{4}-dimensional quadrature error')
    plt.plot(k_001, rel_errors_5_001, c[4] + '-', label=f'{5}-dimensional quadrature error')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='upper right')
    plt.show()

    print(np.amax(rel_errors_1_001[:-43]))
    print(np.amax(rel_errors_2_001[:-43]))
    print(np.amax(rel_errors_3_001[:-43]))
    print(np.amax(rel_errors_4_001[:-43]))
    print(np.amax(rel_errors_5_001[:-43]))

    plt.plot(k_001[:-43], rel_errors_1_001[:-43], c[0] + '-', label=f'{1}-dimensional quadrature error')
    plt.plot(k_001[:-43], rel_errors_2_001[:-43], c[1] + '-', label=f'{2}-dimensional quadrature error')
    plt.plot(k_001[:-43], rel_errors_3_001[:-43], c[2] + '-', label=f'{3}-dimensional quadrature error')
    plt.plot(k_001[:-43], rel_errors_4_001[:-43], c[3] + '-', label=f'{4}-dimensional quadrature error')
    plt.plot(k_001[:-43], rel_errors_5_001[:-43], c[4] + '-', label=f'{5}-dimensional quadrature error')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='upper right')
    plt.show()

    '''

    plt.plot(k_001, rHeston_001, 'k-', label='rough Heston')
    plt.plot(k_001, rHeston_obs_2_001, c[0] + '-', label=f'{2}-dimensional approximation')
    plt.plot(k_001, rHeston_obs_4_001, c[1] + '-', label=f'{4}-dimensional approximation')
    plt.plot(k_001, rHeston_obs_8_001, c[2] + '-', label=f'{8}-dimensional approximation')
    plt.plot(k_001, rHeston_obs_16_001, c[3] + '-', label=f'{16}-dimensional approximation')
    plt.plot(k_001, rHeston_obs_32_001, c[4] + '-', label=f'{32}-dimensional approximation')
    plt.xlabel('log-moneyness')
    plt.ylabel('implied volatility')
    plt.legend(loc='upper right')
    plt.show()

    rel_errors_obs_2_001 = np.abs(rHeston_obs_2_001 - rHeston_001) / rHeston_001
    rel_errors_obs_4_001 = np.abs(rHeston_obs_4_001 - rHeston_001) / rHeston_001
    rel_errors_obs_8_001 = np.abs(rHeston_obs_8_001 - rHeston_001) / rHeston_001
    rel_errors_obs_16_001 = np.abs(rHeston_obs_16_001 - rHeston_001) / rHeston_001
    rel_errors_obs_32_001 = np.abs(rHeston_obs_32_001 - rHeston_001) / rHeston_001

    print(np.amax(rel_errors_obs_2_001))
    print(np.amax(rel_errors_obs_4_001))
    print(np.amax(rel_errors_obs_8_001))
    print(np.amax(rel_errors_obs_16_001))
    print(np.amax(rel_errors_obs_32_001))

    plt.plot(k_001, rel_errors_1_001, 'k-', label='1 dimension, optimized')
    plt.plot(k_001, rel_errors_5_001, color='brown', label='5 dimensions, optimized')
    plt.plot(k_001, rel_errors_obs_2_001, c[0] + '-', label=f'2 dimensions, paper')
    plt.plot(k_001, rel_errors_obs_4_001, c[1] + '-', label=f'4 dimensions, paper')
    plt.plot(k_001, rel_errors_obs_8_001, c[2] + '-', label=f'8 dimensions, paper')
    plt.plot(k_001, rel_errors_obs_16_001, c[3] + '-', label=f'16 dimensions, paper')
    plt.plot(k_001, rel_errors_obs_32_001, c[4] + '-', label=f'32 dimensions, paper')
    plt.xlabel('log-moneyness')
    plt.ylabel('relative error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()

    print(np.amax(rel_errors_obs_2_001[:-43]))
    print(np.amax(rel_errors_obs_4_001[:-43]))
    print(np.amax(rel_errors_obs_8_001[:-43]))
    print(np.amax(rel_errors_obs_16_001[:-43]))
    print(np.amax(rel_errors_obs_32_001[:-43]))
'''