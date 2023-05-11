#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:28:56 2022

@author: bayerc

Andersen's QE algorithm.'
"""

import numpy as np
from scipy.special import ndtri


def psi_minus(psi, m, rv):
    """
    The chi-squared branch of Andersen's QE scheme.
    :param psi:
    :param m:
    :param rv: Uniform random variables of shape psi.shape
    """
    assert np.all(psi <= 2.0), f"Failure of psi <= 2 for psi = {psi}."
    beta = np.sqrt(2 / psi - 1 + np.sqrt(2 / psi) * np.sqrt(2 / psi - 1))
    # rv = rng.standard_normal(size=psi.shape)
    index = rv != 0.
    rv[index] = ndtri(rv[index])
    return m / (1 + beta * beta) * (beta + rv) ** 2


def psi_plus(psi, m, rv):
    """
    The exponential branch of Andersen's QE scheme.
    :param psi:
    :param m:
    :param rv: Uniformly distributed random variables of shape psi.shape
    """
    assert np.all(psi >= 1.0), f"Failure of psi >= 1 for psi = {psi}."
    p = 2 / (1 + psi)
    # rv = rng.uniform(size=psi.shape)
    return m * (1 + psi) / 2 * np.log(p / rv) * (rv < p)


def psi_QE(psi, m, rv):
    """
    Andersen's QE scheme. We choose the two branches based on the traditional
    threshold of psi = 1.5.

    Parameters
    ----------
    psi : numpy array.
        psi (i.e., variance / mean^2) values
    m : numpy array.
        Means of the distribution.
    rv : np.random.rng
        Uniform random variables of shape psi.shape

    Returns
    -------
    numpy array.
        Independent samples obtained by the QE algorithm, one for each input
        value psi/m.

    """
    assert np.all(psi > 0.0), f"Failure of positivity of psi = {psi}."
    v = np.zeros_like(psi)
    index = (psi >= 3 / 2)
    v[index] = psi_plus(psi[index], m[index], rv[index])
    v[~index] = psi_minus(psi[~index], m[~index], rv[~index])
    return v


if __name__ == '__main__':
    pass