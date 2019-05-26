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
import sys
import numpy as np

from .preconditioner import JacobiPreconditioner
from .SolverStateBase import SolverStateBase
from .explicit_symmetrisation import IndexSymmetrisation

import scipy.linalg as la
import scipy.sparse.linalg as sla

from adcc import AdcMatrix, AmplitudeVector, linear_combination


def select_eigenpairs(vectors, n_ep, which):
    if which in ["LM", "LA"]:
        return vectors[-n_ep:]
    elif which in ["SM", "SA"]:
        return vectors[:n_ep]
    else:
        raise ValueError("For now only the values 'LM', 'LA', 'SM' and 'SA' "
                         "are understood.")


class DavidsonState(SolverStateBase):
    def __init__(self, matrix, guesses):
        super().__init__(matrix)
        self.residuals = None                   # Current residuals
        self.residual_norms = None              # Current residual norms
        self.subspace_vectors = guesses.copy()  # Current subspace vectors


def default_print(state, identifier, file=sys.stdout):
    """
    A default print function for the davidson callback
    """
    from adcc.timings import strtime, strtime_short

    # TODO Use colour!

    if identifier == "start" and state.n_iter == 0:
        print("Niter n_ss  max_residual  time  Ritz values",
              file=file)
    elif identifier == "next_iter":
        time_iter = state.timer.current("davidson/iteration")
        fmt = "{n_iter:3d}  {ss_size:4d}  {residual:12.5g}  {tstr:5s}"
        print(fmt.format(n_iter=state.n_iter, tstr=strtime_short(time_iter),
                         ss_size=len(state.subspace_vectors),
                         residual=np.max(state.residual_norms)),
              "", state.eigenvalues[:7], file=file)
    elif identifier == "is_converged":
        soltime = state.timer.current("davidson/total")
        print("=== Converged ===", file=file)
        print("    Number of matrix applies:   ", state.n_applies)
        print("    Total solver time:          ", strtime(soltime))
    elif identifier == "restart":
        print("=== Restart ===", file=file)


def davidson_iterations(matrix, state, max_subspace, max_iter, n_ep,
                        is_converged, which, callback=None, preconditioner=None,
                        preconditioning_method="Davidson", debug_checks=False,
                        residual_min_norm=None, explicit_symmetrisation=None):
    """
    @param matrix        Matrix to diagonalise
    @param state         DavidsonState containing the eigenvector guess
                         to propagate
    @param max_subspace  Maximal subspace size
    @param max_iter      Maximal numer of iterations
    @param n_ep          Number of eigenpairs to be computed
    @param is_converged  Function to test for convergence
    @param callback      Callback to run after each iteration
    @param which         Which eigenvectors to converge to.
                         Needs to be chosen such that it agrees with
                         the selected preconditioner.
    @param preconditioner           Preconditioner (type or instance)
    @param preconditioning_method   Precondititoning method. Valid values are
                                    "Davidson" or "Sleijpen-van-der-Vorst"
    @param debug_checks  Enable some potentially costly debug checks
                         (loss of orthogonality in subspace etc)
    @param residual_min_norm   Minimal norm a residual needs to have in order
                               to be accepted as a new subspace vector
                               (defaults to 2 * len(matrix) * machine_expsilon)
    @param explicit_symmetrisation   Explicit symmetrisation to perform
                                     on new subspace vectors before adding
                                     them to the subspace.
    """
    if preconditioning_method not in ["Davidson", "Sleijpen-van-der-Vorst"]:
        raise ValueError("Only 'Davidson' and 'Sleijpen-van-der-Vorst' "
                         "are valid preconditioner methods")
    if preconditioning_method == "Sleijpen-van-der-Vorst":
        raise NotImplementedError("Sleijpen-van-der-Vorst preconditioning "
                                  "not yet implemented.")

    if callback is None:
        def callback(state, identifier):
            pass

    # The problem size
    n_problem = matrix.shape[1]

    # The block size
    n_block = len(state.subspace_vectors)

    # The current subspace size
    n_ss_vec = n_block

    # The current subspace
    SS = state.subspace_vectors

    # The matrix A projected into the subspace
    # as a continuous array. Only the view
    # Ass[:n_ss_vec, :n_ss_vec] contains valid data.
    Ass_cont = np.empty((max_subspace, max_subspace))

    if residual_min_norm is None:
        residual_min_norm = 2 * n_problem * np.finfo(float).eps

    callback(state, "start")
    state.timer.restart("davidson/total")
    state.timer.restart("davidson/iteration")
    while state.n_iter < max_iter:
        state.n_iter += 1

        assert len(SS) >= n_block
        assert len(SS) <= max_subspace

        # Project A onto the subspace, keeping in mind
        # that the values Ass[:-n_block, :-n_block] are already valid,
        # since they have been computed in the previous iterations already.
        state.n_applies += n_block
        AsBlock = matrix @ SS[-n_block:]

        # Increase the view we work with and set the extra column and rows
        Ass = Ass_cont[:n_ss_vec, :n_ss_vec]
        for i in range(n_block):
            Ass[:, -n_block + i] = AsBlock[i] @ SS
        Ass[-n_block:, :] = np.transpose(Ass[:, -n_block:])
        del AsBlock

        # Compute the which(== largest, smallest, ...) eigenpair of Ass
        # and the associated ritz vector as well as residual
        if Ass.shape == (n_block, n_block):
            rvals, rvecs = la.eigh(Ass)  # Do a full diagonalisation
        else:
            # TODO Maybe play with precision a little here
            # TODO Maybe use previous vectors somehow
            v0 = None
            rvals, rvecs = sla.eigsh(Ass, k=n_block, which=which, v0=v0)

        # Transform new vectors to the full basis (form ritz vectors)
        fvecs = [linear_combination(v, SS) for v in np.transpose(rvecs)]
        assert len(fvecs) == n_block

        # Form residuals
        Afvecs = [matrix @ fvecs[i] for i in range(len(fvecs))]
        state.n_applies += n_block
        residuals = [Afvecs[i] - rvals[i] * fvecs[i]
                     for i in range(len(fvecs))]

        # Update te state's eigenpairs and residuals
        state.eigenvalues = select_eigenpairs(rvals, n_ep, which)
        state.eigenvectors = select_eigenpairs(fvecs, n_ep, which)
        state.residuals = select_eigenpairs(residuals, n_ep, which)
        state.residual_norms = np.array([res @ res for res in state.residuals])

        callback(state, "next_iter")
        state.timer.restart("davidson/iteration")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            state.timer.stop("davidson/total")
            return state

        if state.n_iter == max_iter:
            raise la.LinAlgError("Maximum number of iterations (== "
                                 + str(max_iter) + " reached in davidson "
                                 "procedure.")

        if n_ss_vec + n_block > max_subspace:
            # The addition of the preconditioned vectors
            # would go beyond the max_subspace size => collapse first
            state.subspace_vectors = SS = fvecs
            n_ss_vec = len(SS)

            # Update projection of ADC matrix A onto subspace
            Ass = Ass_cont[:n_ss_vec, :n_ss_vec]
            for i in range(n_ss_vec):
                Ass[:, i] = Afvecs[i] @ SS
            callback(state, "restart")
            # continue to add residuals to space
        del Afvecs

        # Apply a preconditioner to the residuals.
        if preconditioner:
            if hasattr(preconditioner, "update_shifts"):
                preconditioner.update_shifts(rvals)
            preconds = preconditioner.apply(residuals)
        else:
            preconds = residuals

        # Explicitly symmetries the new vectors if requested
        if explicit_symmetrisation:
            explicit_symmetrisation.symmetrise(preconds, SS)

        # Project the components of the preconditioned vectors away
        # which are already contained in the subspace.
        # Then add those, which have a significant norm to the subspace.
        n_ss_added = 0
        for i in range(n_block):
            pvec = preconds[i]
            # Project out the components of the current subspace
            pvec = pvec - linear_combination(pvec @ SS, SS)
            pnorm = np.sqrt(pvec @ pvec)
            if pnorm > residual_min_norm:
                # Extend the subspace
                SS.append(pvec / pnorm)
                n_ss_added += 1
                n_ss_vec = len(SS)

        if debug_checks:
            orth = np.array([[SS[i] @ SS[j] for i in range(n_ss_vec)]
                             for j in range(n_ss_vec)])
            orth -= np.eye(n_ss_vec)
            if np.max(np.abs(orth)) > 1e-14:
                raise la.LinAlgWarning(
                    "Subspace in davidson has lost orthogonality. "
                    "Expect inaccurate results."
                )

        if n_ss_added == 0:
            state.converged = False
            raise la.LinAlgError(
                "Davidson procedure could not generate any further vectors for "
                "the subpace. Iteration cannot be continued like this and will "
                "be aborted without convergence. Try a different guess.")


def eigsh(matrix, guesses, n_ep=None, max_subspace=None,
          conv_tol=1e-9, which="SA", max_iter=100,
          callback=None, preconditioner=None,
          preconditioning_method="Davidson", debug_checks=False,
          residual_min_norm=None, explicit_symmetrisation=IndexSymmetrisation):
    """
    Davidson eigensolver for ADC problems

    @param matrix        ADC matrix instance
    @param guesses       Guess vectors (fixes the block size)
    @param n_ep          Number of eigenpairs to be computed
    @param max_subspace  Maximal subspace size
    @param conv_tol      Convergence tolerance on the l2 norm of residuals
                         to consider them converged
    @param which         Which eigenvectors to converge to.
                         Needs to be chosen such that it agrees with
                         the selected preconditioner.
    @param max_iter      Maximal numer of iterations
    @param callback      Callback to run after each iteration
    @param preconditioner           Preconditioner (type or instance)
    @param preconditioning_method   Precondititoning method. Valid values are
                                    "Davidson" or "Sleijpen-van-der-Vorst"
    @param explicit_symmetrisation   Explicit symmetrisation to apply to new
                                     subspace vectors before adding them to
                                     the subspace. Allows to correct for
                                     loss of index or spin symmetries
                                     (type or instance)
    @param debug_checks  Enable some potentially costly debug checks
                         (Loss of orthogonality etc.)
    @param residual_min_norm   Minimal norm a residual needs to have in order
                               to be accepted as a new subspace vector
                               (defaults to 2 * len(matrix) * machine_expsilon)
    """
    if not isinstance(matrix, AdcMatrix):
        raise TypeError("matrix is not of type AdcMatrix")
    for guess in guesses:
        if not isinstance(guess, AmplitudeVector):
            raise TypeError("One of the guesses is not of type AmplitudeVector")

    if preconditioner is not None and isinstance(preconditioner, type):
        preconditioner = preconditioner(matrix)

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    if n_ep is None:
        n_ep = len(guesses)
    elif n_ep > len(guesses):
        raise ValueError("n_ep cannot exceed the number of guess vectors.")
    if not max_subspace:
        # TODO Arnoldi uses this:
        # max_subspace = max(2 * n_ep + 1, 20)
        max_subspace = max(6 * n_ep, 20, 5 * len(guesses))

    def convergence_test(state):
        state.residuals_converged = state.residual_norms < conv_tol
        state.converged = np.all(state.residuals_converged)
        return state.converged

    state = DavidsonState(matrix, guesses)
    davidson_iterations(matrix, state, max_subspace, max_iter,
                        n_ep=n_ep, is_converged=convergence_test,
                        callback=callback, which=which,
                        preconditioner=preconditioner,
                        preconditioning_method=preconditioning_method,
                        debug_checks=debug_checks,
                        residual_min_norm=residual_min_norm,
                        explicit_symmetrisation=explicit_symmetrisation)

    # Free memory occupied by subspace_vectors and return
    state.subspace_vectors = None
    return state


def jacobi_davidson(*args, **kwargs):
    return eigsh(*args, preconditioner=JacobiPreconditioner,
                 preconditioning_method="Davidson", **kwargs)


def davidson(*args, **kwargs):
    return eigsh(*args, preconditioner=None, **kwargs)
