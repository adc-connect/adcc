#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import sys
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from adcc import evaluate, lincomb
from adcc.AdcMatrix import AdcMatrixlike
from adcc.AmplitudeVector import AmplitudeVector

from .common import select_eigenpairs
from .preconditioner import JacobiPreconditioner
from .SolverStateBase import EigenSolverStateBase
from .explicit_symmetrisation import IndexSymmetrisation


class DavidsonState(EigenSolverStateBase):
    def __init__(self, matrix, guesses):
        super().__init__(matrix)
        self.residuals = None                   # Current residuals
        self.subspace_vectors = guesses.copy()  # Current subspace vectors
        self.algorithm = "davidson"


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
        time_iter = state.timer.current("iteration")
        fmt = "{n_iter:3d}  {ss_size:4d}  {residual:12.5g}  {tstr:5s}"
        print(fmt.format(n_iter=state.n_iter, tstr=strtime_short(time_iter),
                         ss_size=len(state.subspace_vectors),
                         residual=np.max(state.residual_norms)),
              "", state.eigenvalues[:7], file=file)
        if hasattr(state, "subspace_orthogonality"):
            print(33 * " " + "nonorth: {:5.3g}"
                  "".format(state.subspace_orthogonality))
    elif identifier == "is_converged":
        soltime = state.timer.total("iteration")
        print("=== Converged ===", file=file)
        print("    Number of matrix applies:   ", state.n_applies, file=file)
        print("    Total solver time:          ", strtime(soltime), file=file)
    elif identifier == "restart":
        print("=== Restart ===", file=file)


# TODO This function should be merged with eigsh
def davidson_iterations(matrix, state, max_subspace, max_iter, n_ep,
                        is_converged, which, callback=None, preconditioner=None,
                        preconditioning_method="Davidson", debug_checks=False,
                        residual_min_norm=None, explicit_symmetrisation=None):
    """Drive the davidson iterations

    Parameters
    ----------
    matrix
        Matrix to diagonalise
    state
        DavidsonState containing the eigenvector guess
    max_subspace : int or NoneType, optional
        Maximal subspace size
    max_iter : int, optional
        Maximal number of iterations
    n_ep : int or NoneType, optional
        Number of eigenpairs to be computed
    is_converged
        Function to test for convergence
    callback : callable, optional
        Callback to run after each iteration
    which : str, optional
        Which eigenvectors to converge to. Needs to be chosen such that
        it agrees with the selected preconditioner.
    preconditioner
        Preconditioner (type or instance)
    preconditioning_method : str, optional
        Precondititoning method. Valid values are "Davidson"
        or "Sleijpen-van-der-Vorst"
    debug_checks : bool, optional
        Enable some potentially costly debug checks
        (Loss of orthogonality etc.)
    residual_min_norm : float or NoneType, optional
        Minimal norm a residual needs to have in order to be accepted as
        a new subspace vector
        (defaults to 2 * len(matrix) * machine_expsilon)
    explicit_symmetrisation
        Explicit symmetrisation to apply to new subspace vectors before
        adding them to the subspace. Allows to correct for loss of index
        or spin symmetries (type or instance)
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

    eps = np.finfo(float).eps
    if residual_min_norm is None:
        residual_min_norm = 2 * n_problem * eps

    callback(state, "start")
    state.timer.restart("iteration")

    with state.timer.record("projection"):
        # Initial application of A to the subspace
        Ax = evaluate(matrix @ SS)
        state.n_applies += n_ss_vec

    while state.n_iter < max_iter:
        state.n_iter += 1

        assert len(SS) >= n_block
        assert len(SS) <= max_subspace

        # Project A onto the subspace, keeping in mind
        # that the values Ass[:-n_block, :-n_block] are already valid,
        # since they have been computed in the previous iterations already.
        with state.timer.record("projection"):
            Ass = Ass_cont[:n_ss_vec, :n_ss_vec]  # Increase the work view size
            for i in range(n_block):
                Ass[:, -n_block + i] = Ax[-n_block + i] @ SS
            Ass[-n_block:, :] = np.transpose(Ass[:, -n_block:])

        # Compute the which(== largest, smallest, ...) eigenpair of Ass
        # and the associated ritz vector as well as residual
        with state.timer.record("rayleigh_ritz"):
            if Ass.shape == (n_block, n_block):
                rvals, rvecs = la.eigh(Ass)  # Do a full diagonalisation
            else:
                # TODO Maybe play with precision a little here
                # TODO Maybe use previous vectors somehow
                v0 = None
                rvals, rvecs = sla.eigsh(Ass, k=n_block, which=which, v0=v0)

        with state.timer.record("residuals"):
            # Form residuals, A * SS * v - λ * SS * v = Ax * v + SS * (-λ*v)
            def form_residual(rval, rvec):
                coefficients = np.hstack((rvec, -rval * rvec))
                return lincomb(coefficients, Ax + SS, evaluate=True)
            residuals = [form_residual(rvals[i], v)
                         for i, v in enumerate(np.transpose(rvecs))]
            assert len(residuals) == n_block

            # Update the state's eigenpairs and residuals
            epair_mask = select_eigenpairs(rvals, n_ep, which)
            state.eigenvalues = rvals[epair_mask]
            state.residuals = [residuals[i] for i in epair_mask]
            state.residual_norms = np.array([r @ r for r in state.residuals])
            # TODO This is misleading ... actually residual_norms contains
            #      the norms squared. That's also the used e.g. in adcman to
            #      check for convergence, so using the norm squared is fine,
            #      in theory ... it should just be consistent. I think it is
            #      better to go for the actual norm (no squared) inside the code
            #
            #      If this adapted, also change the conv_tol to tol conversion
            #      inside the Lanczos procedure.

        callback(state, "next_iter")
        state.timer.restart("iteration")
        if is_converged(state):
            # Build the eigenvectors we desire from the subspace vectors:
            state.eigenvectors = [lincomb(v, SS, evaluate=True)
                                  for i, v in enumerate(np.transpose(rvecs))
                                  if i in epair_mask]

            state.converged = True
            callback(state, "is_converged")
            state.timer.stop("iteration")
            return state

        if state.n_iter == max_iter:
            warnings.warn(la.LinAlgWarning(
                f"Maximum number of iterations (== {max_iter}) "
                "reached in davidson procedure."))
            state.eigenvectors = [lincomb(v, SS, evaluate=True)
                                  for i, v in enumerate(np.transpose(rvecs))
                                  if i in epair_mask]
            state.timer.stop("iteration")
            state.converged = False
            return state

        if n_ss_vec + n_block > max_subspace:
            callback(state, "restart")
            with state.timer.record("projection"):
                # The addition of the preconditioned vectors goes beyond max.
                # subspace size => Collapse first, ie keep current Ritz vectors
                # as new subspace
                SS = [lincomb(v, SS, evaluate=True) for v in np.transpose(rvecs)]
                state.subspace_vectors = SS
                Ax = [lincomb(v, Ax, evaluate=True) for v in np.transpose(rvecs)]
                n_ss_vec = len(SS)

                # Update projection of ADC matrix A onto subspace
                Ass = Ass_cont[:n_ss_vec, :n_ss_vec]
                for i in range(n_ss_vec):
                    Ass[:, i] = Ax[i] @ SS
            # continue to add residuals to space

        with state.timer.record("preconditioner"):
            if preconditioner:
                if hasattr(preconditioner, "update_shifts"):
                    # Epsilon factor to make sure that 1 / (shift - diagonal)
                    # does not become ill-conditioned as soon as the shift
                    # approaches the actual diagonal values (which are the
                    # eigenvalues for the ADC(2) doubles part if the coupling
                    # block are absent)
                    rvals_eps = 1e-6
                    preconditioner.update_shifts(rvals - rvals_eps)

                preconds = evaluate(preconditioner @ residuals)
            else:
                preconds = residuals

            # Explicitly symmetrise the new vectors if requested
            if explicit_symmetrisation:
                explicit_symmetrisation.symmetrise(preconds)

        # Project the components of the preconditioned vectors away
        # which are already contained in the subspace.
        # Then add those, which have a significant norm to the subspace.
        with state.timer.record("orthogonalisation"):
            n_ss_added = 0
            for i in range(n_block):
                pvec = preconds[i]
                # Project out the components of the current subspace
                # That is form (1 - SS * SS^T) * pvec = pvec + SS * (-SS^T * pvec)
                coefficients = np.hstack(([1], -(pvec @ SS)))
                pvec = lincomb(coefficients, [pvec] + SS, evaluate=True)
                pnorm = np.sqrt(pvec @ pvec)
                if pnorm > residual_min_norm:
                    # Extend the subspace
                    SS.append(evaluate(pvec / pnorm))
                    n_ss_added += 1
                    n_ss_vec = len(SS)

            if debug_checks:
                orth = np.array([[SS[i] @ SS[j] for i in range(n_ss_vec)]
                                 for j in range(n_ss_vec)])
                orth -= np.eye(n_ss_vec)
                state.subspace_orthogonality = np.max(np.abs(orth))
                if state.subspace_orthogonality > n_problem * eps:
                    warnings.warn(la.LinAlgWarning(
                        "Subspace in davidson has lost orthogonality. "
                        "Expect inaccurate results."
                    ))

        if n_ss_added == 0:
            state.timer.stop("iteration")
            state.converged = False
            state.eigenvectors = [lincomb(v, SS, evaluate=True)
                                  for i, v in enumerate(np.transpose(rvecs))
                                  if i in epair_mask]
            warnings.warn(la.LinAlgWarning(
                "Davidson procedure could not generate any further vectors for "
                "the subspace. Iteration cannot be continued like this and will "
                "be aborted without convergence. Try a different guess."))
            return state

        with state.timer.record("projection"):
            Ax.extend(matrix @ SS[-n_ss_added:])
            state.n_applies += n_ss_added


def eigsh(matrix, guesses, n_ep=None, max_subspace=None,
          conv_tol=1e-9, which="SA", max_iter=70,
          callback=None, preconditioner=None,
          preconditioning_method="Davidson", debug_checks=False,
          residual_min_norm=None, explicit_symmetrisation=IndexSymmetrisation):
    """Davidson eigensolver for ADC problems

    Parameters
    ----------
    matrix
        ADC matrix instance
    guesses : list
        Guess vectors (fixes also the Davidson block size)
    n_ep : int or NoneType, optional
        Number of eigenpairs to be computed
    max_subspace : int or NoneType, optional
        Maximal subspace size
    conv_tol : float, optional
        Convergence tolerance on the l2 norm squared of residuals to consider
        them converged
    which : str, optional
        Which eigenvectors to converge to (e.g. LM, LA, SM, SA)
    max_iter : int, optional
        Maximal number of iterations
    callback : callable, optional
        Callback to run after each iteration
    preconditioner
        Preconditioner (type or instance)
    preconditioning_method : str, optional
        Precondititoning method. Valid values are "Davidson"
        or "Sleijpen-van-der-Vorst"
    explicit_symmetrisation
        Explicit symmetrisation to apply to new subspace vectors before
        adding them to the subspace. Allows to correct for loss of index
        or spin symmetries (type or instance)
    debug_checks : bool, optional
        Enable some potentially costly debug checks
        (Loss of orthogonality etc.)
    residual_min_norm : float or NoneType, optional
        Minimal norm a residual needs to have in order to be accepted as
        a new subspace vector
        (defaults to 2 * len(matrix) * machine_expsilon)
    """
    if not isinstance(matrix, AdcMatrixlike):
        raise TypeError("matrix is not of type AdcMatrixlike")
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

    if conv_tol < matrix.shape[1] * np.finfo(float).eps:
        warnings.warn(la.LinAlgWarning(
            "Convergence tolerance (== {:5.2g}) lower than "
            "estimated maximal numerical accuracy (== {:5.2g}). "
            "Convergence might be hard to achieve."
            "".format(conv_tol, matrix.shape[1] * np.finfo(float).eps)
        ))

    state = DavidsonState(matrix, guesses)
    davidson_iterations(matrix, state, max_subspace, max_iter,
                        n_ep=n_ep, is_converged=convergence_test,
                        callback=callback, which=which,
                        preconditioner=preconditioner,
                        preconditioning_method=preconditioning_method,
                        debug_checks=debug_checks,
                        residual_min_norm=residual_min_norm,
                        explicit_symmetrisation=explicit_symmetrisation)
    return state


def jacobi_davidson(*args, **kwargs):
    return eigsh(*args, preconditioner=JacobiPreconditioner,
                 preconditioning_method="Davidson", **kwargs)


def davidson(*args, **kwargs):
    return eigsh(*args, preconditioner=None, **kwargs)
