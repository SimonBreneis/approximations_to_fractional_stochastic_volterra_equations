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


mp.mp.dps = 1000
rk.plot_errors_sparse(0.1, 1., [5], [[102]])
print(rk.error_estimate_improved(0.1, 7, 73, 7.7444, 131.24, 1.))
time.sleep(3600)


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
