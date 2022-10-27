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
import warnings
import numpy as np
import scipy.linalg as la

from adcc import evaluate, lincomb
from adcc.timings import Timer
from adcc.AmplitudeVector import AmplitudeVector

from .orthogonaliser import GramSchmidtOrthogonaliser


class LanczosIterator:
    def __init__(self, matrix, guesses, ritz_vectors=None, ritz_values=None,
                 ritz_overlaps=None, explicit_symmetrisation=None):
        """
        Initialise an iterator generating :py:class:`LanczosSubspace` objects,
        which represent a growing Krylov subspace started from the `matrix`
        and guess vectors `guesses` or optionally using a thick restart from
        the passed Ritz vectors, Ritz values and Ritz overlaps.

        Parameters
        ----------
        matrix
            Matrix to build the Krylov subspace
        guesses : list
            Vectors to build the Krylov subspace
        ritz_vectors : list or NoneType, optional
            Ritz vectors for thick restarts
        ritz_values : numpy.ndarray or NoneType, optional
            Ritz values corresponding to the `ritz_vectors` for thick restarts
        ritz_overlaps : numpy.ndarray or NoneType, optional
            The values ``ritz_vectors.T @ matrix @ guesses`` (which can also
            be computed purely inside the previous Lanczos subspace before the
            collapse, see the code of the :py:`lanczos_iterations` function).
        explicit_symmetrisation : optional
            Explicit symmetrisation to use after orthogonalising the
            subspace vectors. Allows to correct for loss of index or spin
            symmetries during orthogonalisation (type or instance).
        """
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
            if len(ritz_vectors) != n_restart or \
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
        self.ortho = GramSchmidtOrthogonaliser(explicit_symmetrisation)
        self.explicit_symmetrisation = explicit_symmetrisation
        self.timer = Timer()  # TODO More fine-grained timings

        # To be initialised by first call to __next__
        self.lanczos_subspace = []
        self.alphas = []  # Diagonal matrix block of subspace matrix
        self.betas = []   # Side-diagonal matrix blocks of subspace matrix
        self.residual = guesses
        self.n_iter = 0
        self.n_applies = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Advance the iterator, i.e. extend the Lanczos subspace"""
        if self.n_iter == 0:
            # Initialise Lanczos subspace
            v = self.ortho.orthogonalise(self.residual)
            self.lanczos_subspace = v
            r = evaluate(self.matrix @ v)
            alpha = np.empty((self.n_block, self.n_block))
            for p in range(self.n_block):
                alpha[p, :] = v[p] @ r

            # r = r - v * alpha - Y * Sigma
            Sigma, Y = self.ritz_overlaps, self.ritz_vectors
            r = [lincomb(np.hstack(([1], -alpha[:, p], -Sigma[:, p])),
                         [r[p]] + v + Y, evaluate=True)
                 for p in range(self.n_block)]

            # r = r - Y * Y'r (Full reorthogonalisation)
            for p in range(self.n_block):
                r[p] = self.ortho.orthogonalise_against(r[p], self.ritz_vectors)

            self.residual = r
            self.n_iter = 1
            self.n_applies = self.n_block
            self.alphas = [alpha]  # Diagonal matrix block of subspace matrix
            self.betas = []        # Side-diagonal matrix blocks
            return LanczosSubspace(self)

        # Iteration 1 and onwards:
        q = self.lanczos_subspace[-self.n_block:]
        v, beta = self.ortho.qr(self.residual)
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
        for p in range(self.n_block):
            r[p] = self.ortho.orthogonalise_against(
                r[p], self.lanczos_subspace + self.ritz_vectors
            )

        # Commit results
        self.n_iter += 1
        self.lanczos_subspace.extend(v)
        self.residual = r
        self.alphas.append(alpha)
        self.betas.append(beta)
        return LanczosSubspace(self)


class LanczosSubspace:
    """Container for the Lanczos subspace."""
    def __init__(self, iterator):
        self.__iterator = iterator
        self.n_iter = iterator.n_iter        # Number of iterations
        self.n_restart = iterator.n_restart  # Number of restart vectors
        self.residual = iterator.residual    # Lanczos residual vector(s)
        self.n_block = iterator.n_block      # Block size
        self.n_problem = iterator.n_problem  # Problem size
        self.matrix = iterator.matrix        # Problem matrix
        self.ortho = iterator.ortho          # Orthogonaliser
        self.alphas = iterator.alphas        # Diagonal blocks
        self.betas = iterator.betas          # Side-diagonal blocks
        self.n_applies = iterator.n_applies  # Number of applies

        # Combined set of subspace vectors
        self.subspace = iterator.ritz_vectors + iterator.lanczos_subspace

    @property
    def subspace_matrix(self):
        """
        Return the projection of the problem matrix into the Lanczos subspace
        """
        # TODO Use sparse representation

        n_k = len(self.__iterator.lanczos_subspace)
        n_restart = self.n_restart
        n_ss = n_k + n_restart
        ritz_values = self.__iterator.ritz_values
        ritz_overlaps = self.__iterator.ritz_overlaps

        T = np.zeros((n_ss, n_ss))
        if n_restart > 0:
            T[:n_restart, :n_restart] = np.diag(ritz_values)
            T[:n_restart, n_restart:n_restart + self.n_block] = ritz_overlaps
            T[n_restart:n_restart + self.n_block, :n_restart] = ritz_overlaps.T

        for i, alpha in enumerate(self.alphas):
            rnge = slice(n_restart + i * self.n_block,
                         n_restart + (i + 1) * self.n_block)
            T[rnge, rnge] = alpha
        for i, beta in enumerate(self.betas):
            rnge = slice(n_restart + i * self.n_block,
                         n_restart + (i + 1) * self.n_block)
            rnge_plus = slice(rnge.start + self.n_block, rnge.stop + self.n_block)
            T[rnge, rnge_plus] = beta.T
            T[rnge_plus, rnge] = beta
        return T

    @property
    def rayleigh_extension(self):
        n_k = len(self.__iterator.lanczos_subspace)
        n_ss = n_k + self.n_restart
        b = np.zeros((n_ss, self.n_block))
        for i in range(self.n_block):
            b[n_ss - self.n_block + i, i] = 1
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

    def check_orthogonality(self, tolerance=None):
        if tolerance is None:
            tolerance = self.n_problem * np.finfo(float).eps
        orth = np.array([[SSi @ SSj for SSi in self.subspace]
                         for SSj in self.subspace])
        orth -= np.eye(len(self.subspace))
        orth = np.max(np.abs(orth))
        if orth > tolerance:
            warnings.warn(la.LinAlgWarning(
                "LanczosSubspace has lost orthogonality. "
                "Expect inaccurate results."
            ))
        return orth
