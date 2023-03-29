#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 19:19:22 2022

@author: bayerc

The generalized Mittag-Leffler function called through the R-package
MittagLeffleR and rpy2.
"""

import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter


def mittag_leffler(z, alpha, beta):
    """
    Compute the generalized Mittag-Leffler function E_{alpha,beta}(z), using
    the R package MittagLeffleR called with rpy2.

    Parameters
    ----------
    z : numpy array or scalar.
        Argument of the Mittag-Leffler function.
    alpha : double
        Alpha parameter.
    beta : double
        Beta parameter.

    Returns
    -------
    e : numpy array or scalar.

    """
    mlfr = importr('MittagLeffleR')
    if isinstance(z, np.ndarray):
        np_cv_rules = default_converter + numpy2ri.converter
        with localconverter(np_cv_rules) as cv:
            e = mlfr.mlf(z, alpha, beta)
    else:
        e = mlfr.mlf(z, alpha, beta)[0]

    return e


if __name__ == "__main__":
    pass