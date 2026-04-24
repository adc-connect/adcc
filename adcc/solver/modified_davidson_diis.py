#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import warnings
import numpy as np
import scipy.linalg as la
from .davidson import DavidsonState, davidson_iterations
from .explicit_symmetrisation import IndexSymmetrisation
from .fixed_point_diis import diis, default_print, DIISError, SubspaceError


class ModifiedDavidsonDiisState:
    def __init__(self, matrix, n_states):
        self.matrix = matrix
        self.eigenvalues = np.full((n_states, ), np.nan, dtype=np.float64)
        self.eigenvectors = [None] * n_states
        self.residual_norms = np.full((n_states, ), np.nan, dtype=np.float64)
        self.converged = np.full((n_states, ), False, dtype=np.float64)
        self.n_iter = np.full((n_states, ), 0, dtype=np.int64)
        self.n_applies = np.full((n_states, ), 0, dtype=np.int64)
        self.n_iter_diis = np.full((n_states, ), 0, dtype=np.int64)


def convergence_test_modified_davidson(omega_macro, n_state, conv_tol_davidson):
    def convergence_test(state):
        if state.eigenvalues_history.maxlen is not None:
            assert state.eigenvalues_history.maxlen >= 2
        if len(state.eigenvalues_history) < 2:
            return False
        delta_omega_first = omega_macro - state.eigenvalues_initial[n_state]
        delta_omega_subsequent = (
            state.eigenvalues_history[-1][n_state]
            - state.eigenvalues_history[-2][n_state]
        )
        state.converged = np.abs(delta_omega_first) > np.abs(delta_omega_subsequent)
        return state.converged
    return convergence_test


def diis_updater(state, n_state, preconditioner):
    def updater(vec):
        state.n_iter_diis[n_state] += 1
        vnorm = np.sqrt(vec @ vec)
        vec /= vnorm
        mvp = (state.matrix @ vec).evaluate()
        state.n_applies[n_state] += 1
        residual = mvp - state.eigenvalues[n_state] * vec
        new_vec = (vec - (preconditioner @ residual)).evaluate()
        state.eigenvectors[n_state] = vec
        state.eigenvalues[n_state] = vec @ mvp
        state.matrix.omega = state.eigenvalues[n_state]
        return new_vec
    return updater


def modified_davidson_diis(matrix, guess_vectors, guess_omegas,
                           n_ep=None, conv_tol=1e-9,
                           conv_tol_davidson=1e-3, max_davidson_runs=50,
                           n_block_davidson=None, max_subspace_davidson=None,
                           which_davidson="SA", residual_min_norm_davidson=None,
                           max_iter_davidson=70, max_subspace_iter_davidson=None,
                           max_subspace_diis=20, max_iter_diis=300,
                           diis_start_size=3,
                           callback=None, preconditioner=None, debug_checks=False,
                           explicit_symmetrisation=IndexSymmetrisation):
    if preconditioner is not None and isinstance(preconditioner, type):
        preconditioner = preconditioner(matrix)

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    if len(guess_omegas) != len(guess_vectors):
        raise ValueError(f"The number of guess vectors (= {len(guess_vectors)}) "
                         f"and the number of guess omegas (= {len(guess_omegas)}) "
                         "must be equal.")

    if n_ep is None:
        n_ep = len(guess_vectors)
    elif n_ep > len(guess_vectors):
        raise ValueError(f"n_ep (= {n_ep}) cannot exceed the number of guess "
                         f"vectors (= {len(guess_vectors)}).")

    if n_block_davidson is None:
        n_block_davidson = n_ep
    elif n_block_davidson < n_ep:
        raise ValueError(f"n_block_davidson (= {n_block_davidson}) cannot be "
                         f"smaller than the number of states requested (= {n_ep}).")
    elif n_block_davidson > len(guess_vectors):
        raise ValueError(f"n_block_davidson (= {n_block_davidson}) cannot exceed "
                         f"the number of guess vectors (= {len(guess_vectors)}).")

    if not max_subspace_davidson:
        # TODO Arnoldi uses this:
        # max_subspace = max(2 * n_ep + 1, 20)
        max_subspace_davidson = max(6 * n_ep, 20, 5 * len(guess_vectors))
    elif max_subspace_davidson < 2 * n_block_davidson:
        raise ValueError(f"max_subspace (= {max_subspace_davidson}) needs to be at "
                         "least twice as large as n_block_davidson "
                         f"(n_block_davidson = {n_block_davidson}).")
    elif max_subspace_davidson < len(guess_vectors):
        raise ValueError(f"max_subspace (= {max_subspace_davidson}) cannot be "
                         "smaller than the number of guess vectors "
                         f"(= {len(guess_vectors)}).")

    if conv_tol < matrix.shape[1] * np.finfo(float).eps:
        warnings.warn(la.LinAlgWarning(
            "Convergence tolerance (== {:5.2g}) lower than "
            "estimated maximal numerical accuracy (== {:5.2g}). "
            "Convergence might be hard to achieve."
            "".format(conv_tol, matrix.shape[1] * np.finfo(float).eps)
        ))

    state = ModifiedDavidsonDiisState(matrix, n_ep)
    for n_state in range(n_ep):
        print(f"====================== {n_state} ======================")
        matrix.omega = guess_omegas[n_state]
        davidson_guesses = guess_vectors.copy()
        while state.n_iter[n_state] < max_davidson_runs:
            state.n_iter[n_state] += 1
            omega_macro = matrix.omega
            state_i = DavidsonState(matrix, davidson_guesses)
            is_converged = convergence_test_modified_davidson(omega_macro, n_state,
                                                              conv_tol_davidson)
            davidson_iterations(matrix, state_i, max_subspace_davidson,
                                max_iter_davidson, n_ep=n_ep,
                                n_block=n_block_davidson, is_converged=is_converged,
                                which=which_davidson, callback=callback,
                                preconditioner=preconditioner,
                                debug_checks=debug_checks,
                                residual_min_norm=residual_min_norm_davidson,
                                explicit_symmetrisation=explicit_symmetrisation,
                                max_subspace_iter=max_subspace_iter_davidson)
            state.eigenvalues[n_state] = state_i.eigenvalues[n_state]
            matrix.omega = state.eigenvalues[n_state]
            state.eigenvectors[n_state] = state_i.eigenvectors[n_state]
            state.residual_norms[n_state] = state_i.residual_norms[n_state]
            state.n_applies[n_state] += state_i.n_applies
            print(f"number of micro iterations: {state_i.n_iter}")
            if state.residual_norms[n_state] < conv_tol_davidson:
                break
            davidson_guesses = state_i.eigenvectors.copy()

        preconditioner.update_shifts(0.0)
        try:
            state.eigenvectors[n_state] = diis(
                diis_updater(state, n_state, preconditioner),
                state.eigenvectors[n_state],
                diis_start_size=diis_start_size,
                max_subspace_size=max_subspace_diis, conv_tol=conv_tol,
                n_max_iterations=max_iter_diis, callback=default_print
            )
            state.converged[n_state] = True
        except SubspaceError as e:
            raise e
        except DIISError:
            pass
    return state
