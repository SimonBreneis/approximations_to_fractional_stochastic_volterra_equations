import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats


# Improved errors for fBm with H=0.1 and T=1.
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

errors = [[0.683692, 0.528237, 0.420265, 0.346109, 0.284132, 0.211246, 0.149395, 0.098625, 0.065529, 0.043699, 0.028083, 0.017409, 0.010589, 0.006437, 0.003830, 0.002251, 0.001309, 0.000754, 0.000431, 0.000245],
          [0.635407, 0.451472, 0.333850, 0.256923, 0.204204, 0.119592, 0.076692, 0.039571, 0.022207, 0.010167, 0.004737, 0.002037, 0.000866, 0.000342, 0.000132, 4.91e-05, 1.78e-05, 6.26e-06],
          [0.604934, 0.405954, 0.285941, 0.210481, 0.152756, 0.093002, 0.049924, 0.025489, 0.012130, 0.004540, 0.001559, 0.000528, 0.000158, 4.34e-05, 1.14e-05, 2.86e-06, 6.61e-07],
          [0.582543, 0.374196, 0.253831, 0.180499, 0.103376, 0.069539, 0.035804, 0.016080, 0.006019, 0.001846, 0.000526, 0.000123, 2.74e-05, 5.25e-06, 9.42e-07, 1.20e-07],
          [0.564594, 0.350288, 0.230415, 0.115370, 0.087745, 0.044988, 0.017360, 0.006881, 0.002035, 0.000596, 0.000130, 2.47e-05, 3.79e-06, 4.90e-07, 0.000000],
          [0.549799, 0.331372, 0.212379, 0.143411, 0.101948, 0.061172, 0.021629, 0.008499, 0.003022, 0.000708, 0.000140, 2.59e-05, 3.50e-06, 3.65e-07, 0.000000],
          [0.537380, 0.315872, 0.197939, 0.091741, 0.068621, 0.031377, 0.010481, 0.003575, 0.000841, 0.000167, 2.69e-05, 3.57e-06, 3.36e-07, 0.000000],
          [0.527443, 0.302775, 0.186042, 0.120975, 0.062400, 0.035876, 0.014809, 0.004062, 0.001044, 0.000225, 3.62e-05, 3.82e-06, 3.38e-07, 4.30e-08],
          [0.496858, 0.291659, 0.176010, 0.112708, 0.077254, 0.043063, 0.017208, 0.005369, 0.001490, 0.000283, 4.10e-05, 4.35e-06, 3.59e-07, 4.96e-09],
          [0.549315, 0.281985, 0.167504, 0.105820, 0.053539, 0.020872, 0.005968, 0.001729, 0.000309, 5.35e-05, 5.65e-06, 4.69e-07, 0.000000]]


for m in ms:
    n_here = np.array(ns[m-1])
    errors_here = np.array(errors[m-1])
    plt.loglog(m*n_here[:-1]+1, errors_here[:-1], label=f"m={m}")

plt.legend(loc="upper right")
plt.xlabel("Number of nodes")
plt.ylabel("Error")
plt.show()
