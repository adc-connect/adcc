#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

from adcc import AdcMatrix, AmplitudeVector, linear_combination
from adcc import empty_like
from .preconditioner import JacobiPreconditioner
import numpy as np
import scipy.sparse.linalg as sla
import scipy.linalg as la
import sys


class JacobiState:
    def __init__(self, guess):
        self.eigenvector = guess
        self.eigenvalues = None         # Current eigenvalues
        self.residuals = None           # Current residuals
        self.residual_norms = None       # Corrunt residual norms
        self.iterates = []
        self.converged = None            # Flag whether iteration is converged
        self.n_iter = 0                  # Number of iterations
        self.n_applies = 0               # Number of applies


def __jacobi_step(matrix, state, callback=None, debug_checks=False):
    out = empty_like(state.eigenvector)

    # TODO: HACK
    matrix.compute_diis_apply(state.eigenvector['s'],
                              state.eigenvalue, out['s'])
    vnorm2 = state.eigenvector @ state.eigenvector
    vnorm = np.sqrt(vnorm2)
    state.eigenvalue = (state.eigenvector @ out) / vnorm2
    residual = out - state.eigenvalue * state.eigenvector

    rnorm = residual @ residual / vnorm2
    # state.eigenvector['s'] = 1.0 / vnorm * (state.eigenvector['s'] - (residual['s'] / matrix.mp.df("o1v1")))
    state.eigenvector['s'] = 1.0 / vnorm * (state.eigenvector['s'] - (residual['s'] / matrix.diagonal('s')))
    # print("eigenvector norm: ", state.eigenvector @ state.eigenvector, rnorm, state.eigenvalue)
    state.residual_norms = np.array([rnorm])
    state.n_applies += 1
    return state


def jacobi_solver(matrix, guesses, n_ep=None, max_subspace=None,
         conv_tol=1e-12, max_iter=300,
         callback=None, debug_checks=False):
    """
    Davidson eigensolver for ADC problems

    @param matrix        ADC matrix instance
    @param guesses       Guess vectors (fixes the block size)
    @param n_ep          Number of eigenpairs to be computed
    @param max_subspace  Maximal subspace size
    @param conv_tol      Convergence tolerance on the l2 norm of residuals
                         to consider them converged
    @param max_iter      Maximal numer of iterations
    @param callback      Callback to run after each iteration
    @param debug_checks  Enable some potentially costly debug checks
                         (Loss of orthogonality etc.)
    """
    if not isinstance(matrix, AdcMatrix):
        raise TypeError("matrix is not of type AdcMatrix")
    for guess in guesses:
        if not isinstance(guess, AmplitudeVector):
            raise TypeError("One of the guesses is not of type AmplitudeVector")

    if n_ep is None:
        n_ep = len(guesses)
    elif n_ep > len(guesses):
        raise ValueError("n_ep cannot exceed the number of guess vectors.")

    def convergence_test(state):
        state.residuals_converged = state.residual_norms < conv_tol
        state.converged = np.all(state.residuals_converged)
        return state.converged

    results = []
    # guess_energies = np.sort(np.unique(matrix.mp.df("o1v1").to_ndarray()))
    # if guess_energies.size < len(guesses):
    #     raise RuntimeError("Not enough guess energies found.")
    for guess in guesses:
        # Hack to take only singles guesses
        state = JacobiState(AmplitudeVector(guess['s']))
        state.eigenvalue = np.random.rand(1)[0]
        while not state.converged:
            state.n_iter += 1
            state = __jacobi_step(matrix, state,
                                  callback=callback,
                                  debug_checks=debug_checks)
            if convergence_test(state):
                print("Energy: ", state.eigenvalue)
                print("Applies: ", state.n_applies)
                results.append(state)
            if state.n_iter == max_iter:
                print("Maximum number of iterations (== " +
                                     str(max_iter) + " reached in Jacobi "
                                     "procedure.")
                break
                # raise la.LinAlgError("Maximum number of iterations (== " +
                #                      str(max_iter) + " reached in Jacobi "
                #                      "procedure.")
    return results
