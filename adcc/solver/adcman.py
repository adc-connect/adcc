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
import numpy as np

from .SolverStateBase import SolverStateBase

import libadcc

from adcc import AdcMatrix, AmplitudeVector


class AdcmanSolverState(SolverStateBase):
    def __init__(self, matrix, cppstate):
        super().__init__(matrix)
        self.eigenvalues = cppstate.eigenvalues
        self.converged = np.all(cppstate.residuals_converged)
        self.eigenvectors = [AmplitudeVector(st, dt)
                             for st, dt in zip(cppstate.singles_block,
                                               cppstate.doubles_block)]
        self.residual_norms = cppstate.residual_norms
        self.residuals_converged = cppstate.residuals_converged
        self.kind = cppstate.kind
        self.ctx = cppstate.ctx


def eigh(matrix, n_singlets=None, n_triplets=None, n_states=None,
         max_subspace=0, conv_tol=1e-6, max_iter=60,
         print_level=1, residual_min_norm=1e-12,
         n_guess_singles=0, n_guess_doubles=0):
    """
    Davidson eigensolver for ADC problems

    @param matrix        ADC matrix instance
    @param n_singlets    Number of singlets to solve for
                         (has to be None for UHF reference)
    @param n_triplets    Number of triplets to solve for
                         (has to be None for UHF reference)
    @param n_states      Number of states to solve for
                         (has to be None for RHF reference)
    @param max_subspace  Maximal subspace size
                         (0 means choose automatically depending
                          on the number of states to compute)
    @param conv_tol      Convergence tolerance on the l2 norm of residuals
                         to consider them converged
    @param max_iter      Maximal numer of iterations
    @param print_level   ADCman print level
    @param residual_min_norm   Minimal norm a residual needs to have in order
                               to be accepted as a new subspace vector
                               (defaults to 1e-12)
    @param n_guess_singles   Number of singles block guesses
                             If this plus n_guess_doubles is less
                             than then the number of states to be
                             computed, then n_guess_singles = number of
                             excited states to compute
    @param n_guess doubles   Number of doubles block guesses
                             If this plus n_guess_singles is less
                             than then the number of states to be
                             computed, then n_guess_singles = number of
                             excited states to compute
    """
    if not isinstance(matrix, AdcMatrix):
        raise TypeError("matrix is not of type AdcMatrix")

    if not matrix.reference_state.restricted or \
       matrix.reference_state.spin_multiplicity != 1:
        if n_singlets is not None:
            raise ValueError("The key \"n_singlets\" may only be used in "
                             "combination with an restricted ground state "
                             "reference of singlet spin to provide the number "
                             "of excited states to compute. Use \"n_states\" "
                             "for an UHF reference.")
        if n_triplets is not None:
            raise ValueError("The key \"n_triplets\" may only be used in "
                             "combination with an restricted ground state "
                             "reference of singlet spin to provide the number "
                             "of excited states to compute. Use \"n_states\" "
                             "for an UHF reference.")
        n_triplets = 0
        n_singlets = n_states
    else:
        if n_states is not None:
            raise ValueError("The key \"n_states\" may only be used in "
                             "combination with an unrestricted ground state "
                             "or a non-singlet ground state to provide the "
                             "number of excited states to compute. Use "
                             "\"n_singlets\" and \"n_triplets\".")
        if n_triplets is None:
            n_triplets = 0
        if n_singlets is None:
            n_singlets = 0

    if n_singlets + n_triplets == 0:
        raise ValueError("No excited states to compute.")

    res = libadcc.solve_adcman_davidson(matrix, n_singlets, n_triplets,
                                        conv_tol, max_iter, max_subspace,
                                        print_level, residual_min_norm,
                                        n_guess_singles, n_guess_doubles)

    return [AdcmanSolverState(matrix, state) for state in res]


jacobi_davidson = eigh
