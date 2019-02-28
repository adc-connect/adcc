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
    def __init__(self, guess, previous_results):
        self.eigenvector = guess
        self.previous_results = previous_results
        self.eigenvalues = None         # Current eigenvalues
        self.residual = None           # Current residuals
        self.residual_norm = None       # Corrunt residual norms
        self.iterates = []
        self.converged = None            # Flag whether iteration is converged
        self.n_iter = 0                  # Number of iterations
        self.n_applies = 0               # Number of applies


def __jacobi_step(matrix, state, callback=None, debug_checks=False,
                  u2guess=None):
    out = empty_like(state.eigenvector)
    u2 = empty_like(u2guess)
    matrix.compute_apply("ds", state.eigenvector['s'], u2)
    diag = matrix.diagonal('d')
    # e need to have the same symmetry as the diagonal!
    e = diag.ones_like()
    u2 = u2 / (e * state.eigenvalue - diag)
    tmp1 = state.eigenvector['s'].empty_like()
    matrix.compute_apply("ss", state.eigenvector['s'], out['s'])
    matrix.compute_apply("sd", u2, tmp1)
    out['s'] += tmp1

    for r in state.previous_results:
        state.eigenvector -= (state.eigenvector @ r.eigenvector) * r.eigenvector

    vnorm2 = state.eigenvector @ state.eigenvector
    vnorm = np.sqrt(vnorm2)
    state.eigenvalue = (state.eigenvector @ out) / vnorm2
    residual = out - state.eigenvalue * state.eigenvector

    rnorm = np.sqrt(residual @ residual)
    state.residual = residual
    state.residual_norm = rnorm
    # state.eigenvector['s'] = 1.0 / vnorm * (state.eigenvector['s'] - (residual['s'] / matrix.ground_state.df("o1v1")))
    state.eigenvector['s'] = 1.0 / vnorm * (state.eigenvector['s'] - (residual['s'] / matrix.diagonal('s')))
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
    for guess in guesses:
        # Hack to take only singles guesses
        state = JacobiState(AmplitudeVector(guess['s']), results)
        state.eigenvalue = guess @ (matrix @ guess)
        diis_maxvec = 8
        diis_vectors = []
        diis_residuals = []
        while not state.converged:
            state.n_iter += 1
            state = __jacobi_step(matrix, state,
                                  callback=callback,
                                  debug_checks=debug_checks,
                                  u2guess=guess['d'].empty_like())
            diis_vectors.append(state.eigenvector)
            diis_residuals.append(state.residual)
            if len(diis_vectors) > diis_maxvec:
                diis_vectors.pop(0)
                diis_residuals.pop(0)

            if len(diis_vectors) > 1:
                diis_size = len(diis_vectors) + 1
                bmat = np.zeros((diis_size, diis_size))
                for i in range(1, diis_size):
                    bmat[i, 0] = bmat[0, i] = -1.0
                for i in range(1, diis_size):
                    for j in range(1, diis_size):
                        bmat[i, j] = bmat[j, i] = diis_residuals[i - 1] @ diis_residuals[j - 1]
                rhs = np.zeros(diis_size)
                rhs[0] = -1.0
                l, U = np.linalg.eigh(bmat)
                mask = np.where(np.abs(l) > 1e-14)[0]
                weights = ((U[:, mask] @ np.diag(1. / l[mask]) @ U[:, mask].T) @ rhs)[1:]
                new_vec = state.eigenvector.zeros_like()
                for i in range(diis_size - 1):
                    new_vec += weights[i] * diis_vectors[i]
                state.eigenvector = new_vec
            if convergence_test(state):
                print("Energy: ", state.eigenvalue)
                print("Applies: ", state.n_applies)
                results.append(state)
            if state.n_iter == max_iter:
                print("Maximum number of iterations (== " +
                                     str(max_iter) + " reached in Jacobi "
                                     "procedure.")
                break
    return results
