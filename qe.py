#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:28:56 2022

@author: bayerc

Andersen's QE algorithm.'
"""

import numpy as np


def psi_minus(psi, m, rng):
    """
    The chi-squared branch of Andersen's QE scheme.

    """
    assert np.all(psi <= 2.0), f"Failure of psi <= 2 for psi = {psi}."
    beta = np.sqrt(2 / psi - 1 + np.sqrt(2 / psi) * np.sqrt(2 / psi - 1))
    alpha = m / (1 + beta * beta)
    Z = rng.standard_normal(size=psi.shape)
    return alpha * (beta + Z) ** 2


def psi_plus(psi, m, rng):
    """
    The exponential branch of Andersen's QE scheme.

    """
    assert np.all(psi >= 1.0), f"Failure of psi >= 1 for psi = {psi}."
    p = 2 / (1 + psi)
    gamma = m * (1 + psi) / 2
    U = rng.uniform(size=psi.shape)
    v = gamma * np.log(p / U) * (U < p)
    return v


def psi_QE(psi, m, rng):
    """
    Andersen's QE scheme. We choose the two branches based on the traditional
    threshold of psi = 1.5.

    Parameters
    ----------
    psi : numpy array.
        psi (i.e., variance / mean^2) values
    m : numpy array.
        Means of the distribution.
    rng : np.random.rng
        Random number generator.

    Returns
    -------
    numpy array.
        Independent samples obtained by the QE algorithm, one for each input
        value psi/m.

    """
    assert np.all(psi > 0.0), f"Failure of positivity of psi = {psi}."
    v = np.zeros_like(psi)
    index = (psi >= 3 / 2)
    v[index] = psi_plus(psi[index], m[index], rng)
    v[~index] = psi_minus(psi[~index], m[~index], rng)
    return v


if __name__ == '__main__':
    pass