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
import copy
import warnings
import numpy as np
import scipy.linalg as la

from adcc import evaluate, lincomb
from adcc.AdcMatrix import AdcMatrixlike
from adcc.AmplitudeVector import AmplitudeVector

from .common import select_eigenpairs
from .SolverStateBase import EigenSolverStateBase
from .explicit_symmetrisation import IndexSymmetrisation

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig_banded.html


class GramSchmidtOrthogonaliser:
    def orthogonalise(self, vectors):
        if len(vectors) == 0:
            return []
        subspace = [evaluate(vectors[0] / np.sqrt(vectors[0] @ vectors[0]))]
        for v in vectors[1:]:
            subspace.append(self.orthogonalise_against(v, subspace))
        return subspace

    def orthogonalise_against(self, vector, subspace):
        # Project out the components of the current subspace
        # That is form (1 - SS * SS^T) * vector = vector + SS * (-SS^T * vector)
        coefficients = np.hstack(([1], -(vector @ subspace)))
        return lincomb(coefficients, [vector] + subspace, evaluate=True)


class LanczosIterator:
    def __init__(self, matrix, guesses, ritz_vectors=None, ritz_values=None,
                 ritz_overlaps=None, explicit_symmetrisation=None):
        if not isinstance(matrix, AdcMatrixlike):
            raise TypeError("matrix is not of type AdcMatrixlike")
        n_problem = matrix.shape[1]   # Problem size

        if not isinstance(guesses, list):
            guesses = [guesses]
        for guess in guesses:
            if not isinstance(guess, AmplitudeVector):
                raise TypeError("One of the guesses is not an AmplitudeVector")
        n_block = len(guesses)  # Lanczos block size

        # For thick restarts, defaults to no restart
        if ritz_values is None:
            n_restart = 0
            ritz_vectors = []  # Y
            ritz_overlaps = np.empty((0, n_block))  # Sigma
            ritz_values = np.empty((0, ))  # Theta
        else:
            n_restart = len(ritz_values)
            if ritz_vectors.shape != (n_problem, n_restart) or \
               ritz_overlaps.shape != (n_restart, n_block):
                raise ValueError("Restart vector shape does not agree "
                                 "with problem.")

        self.matrix = matrix
        self.ritz_values = ritz_values
        self.ritz_vectors = ritz_vectors
        self.ritz_overlaps = ritz_overlaps
        self.n_problem = n_problem
        self.n_block = n_block
        self.n_restart = n_restart
        self.ortho = GramSchmidtOrthogonaliser()
        self.explicit_symmetrisation = explicit_symmetrisation

        # To be initialised by __iter__
        self.lanczos_subspace = []
        self.alphas = []  # Diagonal matrix block of subspace matrix
        self.betas = []   # Side-diagonal matrix blocks of subspace matrix
        self.residual = guesses
        self.n_iter = 0
        self.n_applies = 0

    def __iter__(self):
        iterator = copy.copy(self)
        v = self.ortho.orthogonalise(self.residual)

        # Initialise Lanczos subspace
        iterator.lanczos_subspace = v
        r = evaluate(self.matrix @ v)
        alpha = np.empty((self.n_block, self.n_block))
        for p in range(self.n_block):
            alpha[p, :] = v[p] @ r

        # r = r - v * alpha - Y * Sigma
        Sigma, Y = self.ritz_overlaps, self.ritz_vectors
        r = [lincomb(np.hstack(([1], -alpha[:, p], -Sigma[:, p])),
                     [r[p]] + v + Y, evaluate=True) for p in range(self.n_block)]
        iterator.residual = r
        iterator.n_iter = 0
        iterator.n_applies = self.n_block
        iterator.alphas = [alpha]
        iterator.betas = []
        return iterator

    def __next__(self):
        # First iteration already done at class setup
        if self.n_iter == 0:
            self.n_iter += 1
            return LanczosSubspace(self)

        q = self.lanczos_subspace[-self.n_block:]
        if self.n_block == 1:
            beta = np.array([[np.sqrt(self.residual[0] @ self.residual[0])]])
            v = [self.residual[0] / beta[0, 0]]
        else:
            # v, beta = qr(self.residual)
            raise NotImplementedError()
        if self.explicit_symmetrisation is not None:
            # TODO Is this needed?
            self.explicit_symmetrisation.symmetrise(v)

        if np.linalg.norm(beta) < np.finfo(float).eps * self.n_problem:
            # No point to go on ... new vectors will be decoupled from old ones
            raise StopIteration()

        # r = A * v - q * beta^T
        self.n_applies += self.n_block
        r = self.matrix @ v
        r = [lincomb(np.hstack(([1], -(beta.T)[:, p])), [r[p]] + q, evaluate=True)
             for p in range(self.n_block)]

        # alpha = v^T * r
        alpha = np.empty((self.n_block, self.n_block))
        for p in range(self.n_block):
            alpha[p, :] = v[p] @ r

        # r = r - v * alpha
        r = [lincomb(np.hstack(([1], -alpha[:, p])), [r[p]] + v, evaluate=True)
             for p in range(self.n_block)]

        # Full reorthogonalisation
        self.ortho.orthogonalise_against(
            r[p], self.lanczos_subspace + self.ritz_vectors
        )
        if self.explicit_symmetrisation is not None:
            # TODO Is this needed?
            self.explicit_symmetrisation.symmetrise(r)

        # Commit results
        self.n_iter += 1
        self.lanczos_subspace.extend(v)
        self.residual = r
        self.alphas.append(alpha)
        self.betas.append(beta)
        return LanczosSubspace(self)


class LanczosSubspace:
    def __init__(self, iterator):
        self.__iterator = iterator
        self.n_iter = iterator.n_iter
        self.n_restart = iterator.n_restart
        self.residual = iterator.residual
        self.n_block = iterator.n_block
        self.n_problem = iterator.n_problem
        self.matrix = iterator.matrix
        self.ortho = iterator.ortho
        self.alphas = iterator.alphas
        self.betas = iterator.betas
        self.n_applies = iterator.n_applies

        # Combined set of subspace vectors
        self.subspace = iterator.ritz_vectors + iterator.lanczos_subspace

    @property
    def subspace_matrix(self):
        # TODO Use sparse representation

        n_k = len(self.__iterator.lanczos_subspace)
        n_restart = len(self.__iterator.ritz_values)
        n_ss = n_k + n_restart
        ritz_values = self.__iterator.ritz_values
        ritz_overlaps = self.__iterator.ritz_overlaps

        T = np.zeros((n_ss, n_ss))
        if n_restart > 0:
            T[:n_restart, :n_restart] = np.diag(ritz_values)
            T[:n_restart, n_restart:n_restart + self.n_block] = ritz_overlaps

        for i, alpha in enumerate(self.alphas):
            rnge = range(i * self.n_block, (i + 1) * self.n_block)
            T[rnge, rnge] = alpha
        for i, beta in enumerate(self.betas):
            rnge = range(i * self.n_block, (i + 1) * self.n_block)
            rnge_plus = range(rnge.start + self.n_block, rnge.stop + self.n_block)
            T[rnge, rnge_plus] = beta.T
            T[rnge_plus, rnge] = beta
        return T

    @property
    def rayleigh_extension(self):
        n_k = len(self.__iterator.lanczos_subspace)
        n_restart = len(self.__iterator.ritz_values)
        n_ss = n_k + n_restart
        b = np.zeros((n_ss, self.n_block))
        for i in range(self.n_block):
            b[n_restart + (n_k - 1) * self.n_block + i, i] = 1
        return b

    @property
    def matrix_product(self):
        """
        Return the reconstructed matrix-vector product for all subspace vectors
        (using the Lanczos relation).
        """
        r, T = self.residual, self.subspace_matrix
        # b = self.rayleigh_extension; V = self.subspace
        # Form AV  = V * T + r * b'
        AV = []
        for i in range(len(self.subspace)):
            # Compute AV[:, i]
            coefficients = []
            vectors = []
            for (j, v) in enumerate(self.subspace):
                if T[j, i] != 0:
                    coefficients.append(T[j, i])
                    vectors.append(v)
            if i >= len(self.subspace) - self.n_block:
                ires = i - (len(self.subspace) - self.n_block)
                coefficients.append(1)
                vectors.append(r[ires])
            AV.append(lincomb(np.array(coefficients), vectors, evaluate=True))
        return AV


#
# Lanczos eigensolver from here
#

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


def lanczos_iterations(iterator, n_ep, max_subspace, conv_tol=1e-9, which="LA",
                       max_iter=100, callback=None, debug_checks=False,
                       state=None, explicit_symmetrisation=IndexSymmetrisation):
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
        state.n_restart += 1

    for subspace in iterator:
        T = subspace.subspace_matrix
        b = subspace.rayleigh_extension
        rvals, rvecs = np.linalg.eigh(T)

        # Norm of the residual vector block
        norm_residual = np.sqrt(np.sum(subspace.residual[p] @ subspace.residual[p]
                                       for p in range(subspace.n_block)))

        # Minimal tolerance for convergence criterion
        # same settings as in ARPACK are used:
        #    norm(r) * norm(b^T * rvec) <= max(mintol, tol * abs(rval)
        mintol = np.finfo(float).eps * np.max(np.abs(rvals))
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
            # TODO Do a thick restart
            raise NotImplementedError()

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
            explicit_symmetrisation=IndexSymmetrisation):
    iterator = LanczosIterator(matrix, guesses)
    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    if not max_subspace:
        max_subspace = max(2 * n_ep + 1, 20)
    if conv_tol < matrix.shape[1] * np.finfo(float).eps:
        warnings.warn(la.LinAlgWarning(
            "Convergence tolerance (== {:5.2g}) lower than "
            "estimated maximal numerical accuracy (== {:5.2g}). "
            "Convergence might be hard to achieve."
            "".format(conv_tol, matrix.shape[1] * np.finfo(float).eps)
        ))

    return lanczos_iterations(iterator, n_ep, max_subspace, conv_tol, which,
                              max_iter, callback, debug_checks,
                              explicit_symmetrisation=explicit_symmetrisation)
