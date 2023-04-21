#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 20:46:00 2022

@author: bayerc

Jim's RSQE and HQE schemes.
"""
import time
import numpy as np
import multiprocessing as mp
from qe import psi_QE


def discrete_convolution(b, chi):
    """
    Discrete convolution of b and chi.

    Parameters
    ----------
    b : numpy array
        Numpy array of dimension n.
    chi : numpy array
        Numpy array of dimension (m,n).

    Returns
    -------
    dc : numpy array
        numpy array of dimension m. dc[j] = sum_k b[n-1-k] * chi[j,k]

    """
    m, n = chi.shape
    if n > 0:
        b_reversed = b[::-1]  # Note that this is a view, not a copy!
        dc = np.sum(b_reversed * chi, axis=1)
    else:
        dc = np.zeros(m)
    return dc


def _b_star(kernel, Delta, N):
    b2 = kernel.K_diag(Delta, N) / Delta
    return np.sqrt(np.maximum(b2, 0.0))


def _rsqe_step(xi, kernel, rho, rho_bar, j, Delta, eps, b_star, v_old,
               chi_old, X_old, rng, M):
    """Time step of the RSQE scheme.
    """
    xi_hat = np.maximum(eps, xi(j * Delta) +
                        discrete_convolution(b_star[1:j], chi_old[:, :(j - 1)]))
    v_bar = (xi_hat + 2 * kernel.H * v_old) / (2 * kernel.H + 1)
    s2_qe = b_star[0] ** 2 * v_bar * Delta
    psi = s2_qe / xi_hat ** 2
    v = psi_QE(psi, xi_hat, rng)
    u = v - xi_hat
    chi = u / b_star[0]
    Z_orthogonal = rng.standard_normal(M)
    v_mean = 0.5 * (v_old + v)
    X = X_old - 0.5 * v_mean * Delta + \
        rho_bar * np.sqrt(v_mean * Delta) * Z_orthogonal + rho * chi
    return X, v, chi


def rsqe(xi, kernel, rho, X0, t_final, N, M, eps, rng):
    """
    Jim's RSQE algorithm for sampling from the rough Heston method.

    Parameters
    ----------
    xi : function
        Forward variance curve at time 0.
    kernel : class
        Represents the Volterra kernel.
    rho : double
        Correlation between variance and log-price.
    X0 : double (or numpy array)
        Initial value of the log-price.
    t_final : double
        Terminal time.
    N : int
        Number of time steps.
    M : int
        Number of samples.
    eps : double
        Cutoff tolerance at 0.
    rng : numpy.random.rng
        RNG.

    Returns
    -------
    X : numpy array.
        log-price samples, array of dimension (M,N+1).
    v : numpy array.
        variance samples, array of dimension (M,N+1).

    """
    # set up the algorithm
    Delta = t_final / N
    b_star = _b_star(kernel, Delta, N)
    rho_bar = np.sqrt(1 - rho ** 2)
    # allocate memory
    v = np.zeros((M, N + 1))
    v[:, 0] = xi(0)
    X = np.zeros((M, N + 1))
    X[:, 0] = X0
    chi = np.zeros((M, N))

    # The time-stepping
    for j in range(1, N + 1):
        X_new, v_new, chi_new = _rsqe_step(xi, kernel, rho, rho_bar, j, Delta,
                                           eps, b_star, v[:, j - 1],
                                           chi[:, :(j - 1)], X[:, j - 1], rng, M)
        X[:, j] = X_new
        v[:, j] = v_new
        chi[:, j - 1] = chi_new
    return X, v


def rsqe_batch(xi, kernel, rho, X0, t_final, N, M, num_batch, eps, rng=None):
    """
    Jim's RSQE algorithm for sampling from the rough Heston method.

    The simulation is performed in num_batch batches to preserve memory. Only
    the terminal values of log-price and variance are returned.

    Parameters
    ----------
    xi : function
        Forward variance curve at time 0.
    kernel : class
        Represents the Volterra kernel.
    rho : double
        Correlation between variance and log-price.
    X0 : double (or numpy array)
        Initial value of the log-price.
    t_final : double
        Terminal time.
    N : int
        Number of time steps.
    M : int
        Number of samples.
    num_batch : int
        Number of batches for the simulation.
    eps : double
        Cutoff tolerance at 0.
    rng : numpy.random.rng, optional.
        RNG. If None, one is generated.

    Returns
    -------
    X : numpy array.
        log-price samples, array of dimension (M).
    v : numpy array.
        variance samples, array of dimension (M).

    """
    if rng is None:
        rng = np.random.default_rng()
    # set up the algorithm
    Delta = t_final / N
    b_star = _b_star(kernel, Delta, N)
    rho_bar = np.sqrt(1 - rho ** 2)
    M_batch = int(np.ceil(M / num_batch))  # number of samples per batch
    # allocate memory
    v = np.zeros(M_batch * num_batch)
    X = np.zeros(M_batch * num_batch)
    # the stepping over the number of batches
    for n in range(num_batch):
        v_old = np.full(M_batch, xi(0))
        X_old = np.zeros(M_batch)
        X_old = X0
        chi = np.zeros((M_batch, N))
        # The time-stepping
        for j in range(1, N + 1):
            X_new, v_new, chi_new = _rsqe_step(xi, kernel, rho, rho_bar, j,
                                               Delta, eps, b_star, v_old,
                                               chi[:, :(j - 1)], X_old, rng,
                                               M_batch)
            X_old = X_new
            v_old = v_new
            chi[:, j - 1] = chi_new
        v[(n * M_batch):((n + 1) * M_batch)] = v_new
        X[(n * M_batch):((n + 1) * M_batch)] = X_new
    return X[:M], v[:M]  # discard those values which were unnecessarily generated


def bivariate_QE(xi_hat, v_bar, Delta, beta, gamma, rng):
    """
    Apply QE algorithm twice, as needed for the HQE scheme.
    """
    m = 0.5 * xi_hat  # the mean for both bchi_tilde and epsilon_tilde
    s2_chi = beta ** 2 * v_bar * Delta
    s2_epsilon = v_bar * gamma
    psi_chi = s2_chi / m ** 2
    psi_epsilon = s2_epsilon / m ** 2
    assert np.all(psi_chi > 0.0), \
        f"Failure of positivity. psi_chi = {psi_chi}."
    assert np.all(psi_epsilon > 0.0), \
        f"Failure of positivity. psi_epsilon = {psi_epsilon}."
    bchi_tilde = psi_QE(psi_chi, m, rng)
    epsilon_tilde = psi_QE(psi_epsilon, m, rng)
    return bchi_tilde, epsilon_tilde


def _hqe_step(xi, kernel, rho, rho_bar, j, Delta, eps, b_star, v_old,
              chi_old, X_old, rng, M, beta, gamma, r=0.):
    xi_hat = np.maximum(eps, xi(j * Delta) +
                        discrete_convolution(b_star[1:j], chi_old))
    assert np.all(xi_hat > 0.0), f"xi_hat fails positivity, xihat = {xi_hat}."
    v_bar = (xi_hat + 2 * kernel.H * v_old) / (2 * kernel.H + 1)
    assert np.all(v_bar > 0.0), f"v_bar fails positivity at {v_bar}."
    bchi_tilde, epsilon_tilde = bivariate_QE(xi_hat, v_bar, Delta, beta,
                                             gamma, rng)
    # bchi_tilde is beta*chi_tilde
    chi_new = (bchi_tilde - 0.5 * xi_hat) / beta
    v_new = bchi_tilde + epsilon_tilde
    Z_orthogonal = rng.standard_normal(M)
    v_mean = 0.5 * (v_old + v_new)
    X_new = X_old - 0.5 * v_mean * Delta + \
            rho_bar * np.sqrt(v_mean * Delta) * Z_orthogonal + rho * chi_new + r * Delta
    return X_new, v_new, chi_new


def hqe(xi, kernel, rho, X0, t_final, N, M, eps, rng=None, r=0.):
    """
    Jim's HQE algorithm for sampling from the rough Heston method.

    Parameters
    ----------
    xi : function
        Forward variance curve at time 0.
    kernel : class
        Represents the Volterra kernel.
    rho : double
        Correlation between variance and log-price.
    X0 : double (or numpy array)
        Initial value of the log-price.
    t_final : double
        Terminal time.
    N : int
        Number of time steps.
    M : int
        Number of samples.
    eps : double
        Cutoff tolerance at 0.
    rng : numpy.random.rng, optional.
        RNG. If None, an RNG is generated.
    r : double
        Interest rate

    Returns
    -------
    X : numpy array.
        log-price samples, array of dimension (M,N+1).
    v : numpy array.
        variance samples, array of dimension (M,N+1).

    """
    if rng is None:
        rng = np.random.default_rng()
    # set up the algorithm
    Delta = t_final / N
    b_star = _b_star(kernel, Delta, N)
    beta = kernel.K_0(Delta) / Delta
    gamma = Delta * b_star[0] ** 2 - kernel.K_0(Delta) ** 2 / Delta
    rho_bar = np.sqrt(1 - rho ** 2)
    # allocate memory
    v = np.zeros((M, N + 1))
    v[:, 0] = xi(0)
    X = np.zeros((M, N + 1))
    X[:, 0] = X0
    chi = np.zeros((M, N))
    # The time-stepping
    for j in range(1, N + 1):
        X_new, v_new, chi_new = _hqe_step(xi, kernel, rho, rho_bar, j, Delta,
                                          eps, b_star, v[:, j - 1],
                                          chi[:, :(j - 1)], X[:, j - 1], rng, M,
                                          beta, gamma, r=r)
        X[:, j] = X_new
        v[:, j] = v_new
        chi[:, j - 1] = chi_new
    return X, v


def hqe_batch(xi, kernel, rho, X0, t_final, N, M, num_batch, eps, rng=None, sample_paths=False, r=0., verbose=0):
    """
    Batched version of Jim's HQE algorithm for sampling from the rough Heston
    method. Only the terminal value of log-price and variance are returned.

    In order to keep memory bounded, the M samples are generated in num_batch
    batches, with roughly M/num_batch samples each.

    Parameters
    ----------
    xi : function
        Forward variance curve at time 0.
    kernel : class
        Represents the Volterra kernel.
    rho : double
        Correlation between variance and log-price.
    X0 : double (or numpy array)
        Initial value of the log-price.
    t_final : double
        Terminal time.
    N : int
        Number of time steps.
    M : int
        Number of samples.
    num_batch : int
        Number of batches.
    eps : double
        Cutoff tolerance at 0.
    rng : numpy.random.rng, optional
        RNG. If not present, will be generated.
    sample_paths : bool
        If True, returns the sample paths that were generated. If False, only returns the final values
    r : double
        Interest rate
    verbose: int
        Determines the number of intermediary results printed to the console

    Returns
    -------
    X : numpy array.
        log-price samples, array of dimension (M, N+1) if sample_paths is True, else dimension (M).
    v : numpy array.
        variance samples, array of dimension (M, N+1) if sample_paths is False, else dimension (M).

    """
    if rng is None:
        rng = np.random.default_rng()
    # set up the algorithm
    Delta = t_final / N
    tic = time.perf_counter()
    b_star = _b_star(kernel, Delta, N)
    beta = kernel.K_0(Delta) / Delta
    print(f'Other precomputations took {time.perf_counter() - tic} seconds.')
    gamma = Delta * b_star[0] ** 2 - kernel.K_0(Delta) ** 2 / Delta
    assert gamma > 0.0, f"gamma fails positivity, gamma = {gamma}."
    rho_bar = np.sqrt(1 - rho ** 2)
    M_batch = int(np.ceil(M / num_batch))  # number of samples per batch
    # allocate memory
    if sample_paths:
        v = np.zeros((M_batch * num_batch, N + 1))
        X = np.zeros((M_batch * num_batch, N + 1))
        v[:, 0] = xi(0)
        X[:, 0] = X0
    else:
        v = np.zeros(M_batch * num_batch)
        X = np.zeros(M_batch * num_batch)
    # the stepping over the number of batches
    tic = time.perf_counter()
    for n in range(num_batch):
        if verbose >= 1:
            print(f'Simulating batch {n + 1} of {num_batch}.')
        v_old = np.full(M_batch, xi(0))
        X_old = np.zeros(M_batch)
        X_old = X0
        chi = np.zeros((M_batch, N))
        # The time-stepping
        for j in range(1, N + 1):
            X_new, v_new, chi_new = _hqe_step(xi, kernel, rho, rho_bar, j,
                                              Delta, eps, b_star, v_old,
                                              chi[:, :(j - 1)], X_old, rng,
                                              M_batch, beta, gamma, r=r)
            X_old = X_new
            v_old = v_new
            chi[:, j - 1] = chi_new
            if sample_paths:
                v[(n * M_batch):((n + 1) * M_batch), j] = v_new
                X[(n * M_batch):((n + 1) * M_batch), j] = X_new
        if not sample_paths:
            v[(n * M_batch):((n + 1) * M_batch)] = v_new
            X[(n * M_batch):((n + 1) * M_batch)] = X_new
    if verbose >= 1:
        print(f'Simulation took {time.perf_counter() - tic} seconds')
    return X[:M, ...], v[:M, ...]  # discard those values which were unnecessarily generated


def hqe_par(xi, kernel, rho, X0, t_final, N, M, num_batch, num_threads, eps):
    """
    Batched and parallelized version of Jim's HQE algorithm for sampling from
    the rough Heston method. Only the terminal value of log-price and variance
    are returned.

    In order to keep memory bounded, the M samples are generated in num_batch
    batches, with roughly M/num_batch samples each.

    Parameters
    ----------
    xi : function
        Forward variance curve at time 0.
    kernel : class
        Represents the Volterra kernel.
    rho : double
        Correlation between variance and log-price.
    X0 : double (or numpy array)
        Initial value of the log-price.
    t_final : double
        Terminal time.
    N : int
        Number of time steps.
    M : int
        Number of samples.
    num_batch : int
        Number of batches.
    num_threads : int
        Number of threads.
    eps : double
        Cutoff tolerance at 0.

    Returns
    -------
    X : numpy array.
        log-price samples, array of dimension (M).
    v : numpy array.
        variance samples, array of dimension (M).

    """
    M_thread = int(np.ceil(M / num_threads))  # samples per thread
    # helper function to run in parallel; Needs to be global in order to be
    # picklable
    global _f_hqe_parallel

    def _f_hqe_parallel(i):
        return hqe_batch(xi, kernel, rho, X0, t_final, N, M_thread, num_batch,
                         eps)

    with mp.Pool(processes=num_threads) as pool:
        res_list = pool.map(_f_hqe_parallel, range(num_threads))
    # This is a list of pair of (X,v). Now organize them in (X,v) form.
    XV = list(zip(*res_list))
    X = np.array(XV[0]).ravel()
    v = np.array(XV[1]).ravel()
    return X[:M], v[:M]
