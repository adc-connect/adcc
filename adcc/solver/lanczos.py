#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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

from adcc import lincomb

from .common import select_eigenpairs
from .LanczosIterator import LanczosIterator
from .SolverStateBase import EigenSolverStateBase
from .explicit_symmetrisation import IndexSymmetrisation


class LanczosState(EigenSolverStateBase):
    def __init__(self, iterator):
        super().__init__(iterator.matrix)
        self.n_restart = 0
        self.residual = None          # Lanczos residual vector(s)
        self.subspace_vectors = None  # Current subspace vectors
        self.algorithm = "lanczos"


def default_print(state, identifier, file=sys.stdout):
    """
    A default print function for the lanczos callback
    """
    from adcc.timings import strtime, strtime_short

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


def lanczos_iterations(iterator, n_ep, min_subspace, max_subspace, conv_tol=1e-9,
                       which="LA", max_iter=100, callback=None,
                       debug_checks=False, state=None):
    if callback is None:
        def callback(state, identifier):
            pass

    # TODO For consistency with the Davidson the conv_tol is interpreted
    #      as the residual norm *squared*. Arnoldi, however, uses the actual norm
    #      to check for convergence and so on. See also the comment in Davidson
    #      around the line computing state.residual_norms
    #
    #      See also the squaring of the residual norms below
    tol = np.sqrt(conv_tol)

    if state is None:
        state = LanczosState(iterator)
        callback(state, "start")
        state.timer.restart("iteration")
        n_applies_offset = 0
    else:
        n_applies_offset = state.n_applies

    for subspace in iterator:
        T = subspace.subspace_matrix
        b = subspace.rayleigh_extension
        eps = np.finfo(float).eps
        rvals, rvecs = np.linalg.eigh(T)

        if debug_checks:
            orth = np.array([[SSi @ SSj for SSi in subspace.subspace]
                             for SSj in subspace.subspace])
            orth -= np.eye(len(subspace.subspace))
            state.subspace_orthogonality = np.max(np.abs(orth))
            orthotol = max(tol / 1000, subspace.n_problem * eps)
            if state.subspace_orthogonality > orthotol:
                warnings.warn(la.LinAlgWarning(
                    "Subspace in lanczos has lost orthogonality. "
                    "Expect inaccurate results."
                ))

        # Norm of the residual vector block
        norm_residual = np.sqrt(np.sum(subspace.residual[p] @ subspace.residual[p]
                                       for p in range(subspace.n_block)))

        # Minimal tolerance for convergence criterion
        # same settings as in ARPACK are used:
        #    norm(r) * norm(b^T * rvec) <= max(mintol, tol * abs(rval)
        mintol = eps * np.max(np.abs(rvals))
        eigenpair_error = []
        is_rval_converged = np.ones_like(rvals, dtype=bool)
        for i, rval in enumerate(rvals):
            lhs = norm_residual * np.linalg.norm(b.T @ rvecs[:, i])
            rhs = max(mintol, tol * abs(rval))
            if lhs > rhs:
                is_rval_converged[i] = False
            if mintol < tol * abs(rval):
                eigenpair_error.append(lhs / abs(rval))
            else:
                eigenpair_error.append(lhs * tol / mintol)
        eigenpair_error = np.array(eigenpair_error)

        # Update state
        state.n_iter += 1
        state.n_applies = subspace.n_applies + n_applies_offset
        state.converged = False
        state.eigenvectors = None  # Not computed in Lanczos
        state.subspace_vectors = subspace.subspace
        state.residual = subspace.residual
        state.eigenvalues = select_eigenpairs(rvals, n_ep, which)
        state.residual_norms = select_eigenpairs(eigenpair_error, n_ep, which)
        converged = np.all(select_eigenpairs(is_rval_converged, n_ep, which))

        # TODO For consistency with the Davidson the residual norms are squared
        #      again to give output in the same order of magnitude.
        state.residual_norms = state.residual_norms**2

        callback(state, "next_iter")
        state.timer.restart("iteration")

        if converged:
            # TODO Optimise: No need to compute *all* residuals here
            V = subspace.subspace
            AV = subspace.matrix_product

            def form_residual(rval, rvec):
                coefficients = np.hstack((rvec, -rval * rvec))
                return lincomb(coefficients, AV + V, evaluate=True)
            state.residuals = [form_residual(rvals[i], v)
                               for i, v in enumerate(np.transpose(rvecs))]
            state.residuals = select_eigenpairs(state.residuals, n_ep, which)

            selected = select_eigenpairs(np.transpose(rvecs), n_ep, which)
            state.eigenvectors = [lincomb(v, V, evaluate=True) for v in selected]

            rnorms = np.array([np.sqrt(r @ r) for r in state.residuals])
            state.residual_norms = rnorms

            # TODO For consistency with the Davidson the residual norms are
            #      squared again to give output in the same order of magnitude.
            state.residual_norms = state.residual_norms**2

            state.converged = True
            callback(state, "is_converged")
            state.timer.stop("iteration")
            return state

        if len(rvecs) + subspace.n_block > max_subspace:
            callback(state, "restart")

            V = subspace.subspace
            vn, betan = subspace.ortho.qr(subspace.residual)
            rvecsT = select_eigenpairs(np.transpose(rvecs), min_subspace, which)

            Y = [lincomb(rvec, V, evaluate=True) for rvec in rvecsT]
            Theta = select_eigenpairs(rvals, min_subspace, which)
            Sigma = rvecsT @ b @ betan.T

            iterator = LanczosIterator(
                iterator.matrix, vn, ritz_vectors=Y, ritz_values=Theta,
                ritz_overlaps=Sigma,
                explicit_symmetrisation=iterator.explicit_symmetrisation
            )
            state.n_restart += 1
            return lanczos_iterations(
                iterator, n_ep, min_subspace, max_subspace, conv_tol, which,
                max_iter, callback, debug_checks, state)

    state.timer.stop("iteration")
    state.converged = False
    warnings.warn(la.LinAlgWarning(
        "Lanczos procedure found maximal subspace possible. Iteration cannot be "
        "continued like this and will be aborted without convergence. "
        "Try a different guess."))
    return state


def lanczos(matrix, guesses, n_ep, max_subspace=None,
            conv_tol=1e-9, which="LA", max_iter=100,
            callback=None, debug_checks=False,
            explicit_symmetrisation=IndexSymmetrisation,
            min_subspace=None):
    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)
    iterator = LanczosIterator(matrix, guesses,
                               explicit_symmetrisation=explicit_symmetrisation)

    if not isinstance(guesses, list):
        guesses = [guesses]
    if not max_subspace:
        max_subspace = max(2 * n_ep + len(guesses), 20)
    if not min_subspace:
        min_subspace = max(int(max_subspace / 2), n_ep + len(guesses))
    if conv_tol < matrix.shape[1] * np.finfo(float).eps:
        warnings.warn(la.LinAlgWarning(
            "Convergence tolerance (== {:5.2g}) lower than "
            "estimated maximal numerical accuracy (== {:5.2g}). "
            "Convergence might be hard to achieve."
            "".format(conv_tol, matrix.shape[1] * np.finfo(float).eps)
        ))

    return lanczos_iterations(iterator, n_ep, min_subspace, max_subspace,
                              conv_tol, which, max_iter, callback, debug_checks)
