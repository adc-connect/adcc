#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2025-2026 by the adcc authors
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
from typing import (
    Callable, Optional, TextIO, Protocol, Deque, TypeVar, Generic, cast
)
from collections import deque
import sys

import numpy as np


class DIISError(Exception):
    """is the base class of errors visible from this module."""
    pass


class SubspaceError(DIISError):
    """is raised when encountering problems related to the subspace setup."""
    pass


Vector = TypeVar("Vector", bound="DIISVector")


class DIISVector(Protocol):
    def dot(self: Vector, other: Vector) -> float:
        ...

    def zeros_like(self: Vector) -> Vector:
        ...

    def __sub__(self: Vector, other: Vector) -> Vector:
        ...

    def __mul__(self: Vector, other: float) -> Vector:
        ...

    def __iadd__(self: Vector, other: Vector) -> Vector:
        ...


class DIISCallback(Protocol):
    def __call__(self, state: "DIISSubspace", identifier: str,
                 file: TextIO = sys.stdout) -> None:
        ...


class DIISSubspace(Generic[Vector]):
    """
    Manages the DIIS matrix and the subspace and error vectors.

    Parameters
    ----------
    max_size: int
        The maximum number of vectors to keep in the subspace simultaneously.
    start_size: int
        The minimal number of subspace vectors required in the subspace
        in order to start extrapolating. If less vectors are in the
        subspace, linear steps will be performed.
    """
    def __init__(self, max_size: int, start_size: int):
        if max_size < 2:
            raise SubspaceError(f"The DIIS maximum subspace size ({max_size=}) "
                                "has to be greater than 1.")
        if start_size < 2:
            raise SubspaceError(f"The DIIS start size ({start_size=}) has to "
                                "be greater than 1.")
        if start_size > max_size:
            raise SubspaceError(f"The maximum subspace size ({max_size=}) has to "
                                "be larger than the DIIS start size "
                                f"({start_size=}).")

        self.max_size: int = max_size
        self.start_size: int = start_size
        self.subspace_vectors: Deque[Vector] = deque(maxlen=max_size)
        self.error_vectors: Deque[Vector] = deque(maxlen=max_size)
        self.overlap: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
            (0, 0), dtype=np.float64
        )
        self.step_info: Optional[str] = None
        self.converged: bool = False
        self.n_iter: int = 0

    def is_converged(self, conv_tol: float) -> bool:
        """
        Computes the Frobenius norm of the last error vector and checks
        whether it is smaller than the convergence criterion passed as
        conv_tol.
        """
        if self.n_iter >= 1 and self.residual_norm < conv_tol:
            self.converged = True
        return self.converged

    @property
    def residual_norm(self) -> np.float64:
        """
        Returns the Frobenius norm of the most recent error vector, which is
        just the square root of the last diagonal element of the current
        overlap matrix.
        """
        return np.sqrt(self.overlap[-1, -1])

    @property
    def size(self) -> int:
        """Returns the current subspace size"""
        return len(self.subspace_vectors)

    @property
    def last_vector(self) -> Vector:
        """
        Returns the most recent subspace vector, that is, in case of
        convergence, the solution vector.
        """
        return self.subspace_vectors[-1]

    def add(self, subspace_vector: Vector, error_vector: Vector) -> None:
        """
        Adds a new vector and corresponding error vector to the subspace and
        updates the DIIS matrix. Also the iteration counter is incremented.
        """
        self.n_iter += 1
        self.subspace_vectors.append(subspace_vector)
        self.error_vectors.append(error_vector)
        self._update_overlap()

    def _update_overlap(self) -> None:
        """
        Computes the matrix of error vectors in the current subspace.
        Only new elements are computed, elements which have been previously
        computed are transferred from the old overlap matrix.

        Here, two cases can occur. The dimension of the currently stored overlap
        matrix of the previous step ...
        1) equals the subspace size of the current step, that is, the subspace has
           already reached its maximum size in the previous step. In this case,
           the oldest entries (first row and column) are discarded.
        2) is one smaller than the subspace size of the current step, that is, the
           subspace has grown by one. In this case, the overlap matrix is extended
           to the right and bottom.

        The logic of this function is written in a general way in order to
        simultaneously cover both cases.
        """
        if not self.error_vectors:
            raise SubspaceError("No error vectors in subspace.")

        cur_size = len(self.error_vectors)
        copy_size = cur_size - 1
        prev_size = self.overlap.shape[0]
        assert prev_size == copy_size or prev_size == cur_size
        # init new overlap matrix, copy elements and add new elements
        new_overlap = np.zeros((cur_size, cur_size), dtype=np.float64)
        new_overlap.fill(float("NaN"))
        if copy_size > 0:
            new_overlap[:copy_size, :copy_size] = (
                self.overlap[-copy_size:, -copy_size:]
            )
        for ind, error_vec in enumerate(self.error_vectors):
            overlap_value = error_vec.dot(self.error_vectors[-1])
            new_overlap[ind, -1] = overlap_value
            # create transposed element if this is not the diagonal element
            if ind != cur_size - 1:
                new_overlap[-1, ind] = overlap_value
        # Check that there are no NaN elements in the updated overlap matrix,
        # i.e., that all elements have been properly populated
        assert not np.any(np.isnan(new_overlap))
        self.overlap = new_overlap

    def compute_guess(self, n_omit_vectors: int = 0) -> Vector:
        """
        Computes a DIIS guess from the current subspace of size n by solving a
        system of linear equations A * x = b, where A is a matrix of size
        (n+1 x n+1) and b is a vector of size (n+1). Using the error overlap
        matrix elements Sij, the A and b quantities are set up according to

            (S00/S00 S01/S00 ... S0n/S00  -1  )           (  0  )
            (S10/S00 S11/S00 ... S1n/S00  -1  )           (  0  )
        A = (  ...     ...   ...   ...    ... )  and  b = ( ... )
            (Sn0/S00 Sn1/S00 ... Snn/S00  -1  )           (  0  )
            (   -1      -1   ...    -1     0  )           ( -1  )

        The guess is then computed as linear combination of the current subspace
        vectors, using the first (n - n_omit_vectors) elements of the solution
        vector x as coefficients.

        Returns the guess vector.

        Parameters
        ----------
        n_omit_vectors: int
            In the DIIS extrapolation, omit the n subspace vectors associated
            with the error vectors with the largest Frobenius norm.
        """
        if not self.size:
            raise SubspaceError("No vectors in subspace.")
        elif n_omit_vectors >= self.size:
            raise SubspaceError(f"Can not omit {n_omit_vectors} vectors "
                                f"because there are only {self.size} vectors "
                                "in the subspace.")

        # we only have a single vector -> perform a linear step
        if self.size == 1 or self.size - n_omit_vectors <= 1 or \
                self.size < self.start_size:
            self.step_info = "Linear step"
            return self.last_vector

        fill_value = -1.0
        diis_size = self.size
        # build the DIIS matrix
        A: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
            (diis_size + 1, diis_size + 1), dtype=np.float64
        )
        A.fill(fill_value)
        A[-1, -1] = 0.0
        A[:diis_size, :diis_size] = self.overlap / self.overlap[0, 0]
        # build the right hand side
        b: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(
            diis_size + 1, dtype=np.float64
        )
        b[-1] = fill_value
        # determine the indices of vectors to discard
        inds_to_discard = []
        if n_omit_vectors > 0:
            inds_to_discard = np.argsort(self.overlap.diagonal())[-n_omit_vectors:]
            # shrink the system of linear equations by removing rows/columns
            # collected in inds_to_discard
            A = np.delete(A, inds_to_discard, axis=0)
            A = np.delete(A, inds_to_discard, axis=1)
            b = np.delete(b, inds_to_discard, axis=0)
        # solve the system of linear equations
        coefficients = np.linalg.solve(A, b)
        # discard the Lagrange multiplier, which is not needed for DIIS
        # extrapolation
        # also: the coefficients have to be a 1D array with dtype float64
        # the type checker unfortunately can not infer this automatically
        coefficients = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]], coefficients[:-1]
        )
        # build the linear combination:
        # determine the indices of vectors to use for the generation of the
        # DIIS-extrapolated guess, i.e., vectors which have not beeen
        # discarded above
        vec_indices = [i for i in range(self.size) if i not in inds_to_discard]
        guess = self.subspace_vectors[0].zeros_like()
        assert len(vec_indices) == len(coefficients)
        for vec_ind, coeff in zip(vec_indices, coefficients):
            guess += self.subspace_vectors[vec_ind] * coeff

        self.step_info = (
            f"DIIS step from {len(vec_indices)} error vectors, subspace size: "
            f"{self.size}"
        )
        return guess


def default_print(state: DIISSubspace, identifier: str, file: TextIO = sys.stdout):
    if identifier == "start":
        print("Niter   DIIS_error  comment", file=file)
    elif identifier == "next_iter":
        fmt = "{n_iter:4d} {residual: >13.5E}  {step_info:s}"
        print(fmt.format(n_iter=state.n_iter,
                         residual=state.residual_norm,
                         step_info=state.step_info), file=file)
    elif identifier == "is_converged":
        print("=== Converged ===", file=file)


def _no_print(state: DIISSubspace, identifier: str, file: TextIO = sys.stdout):
    pass


def diis(updater: Callable[[Vector], Vector], guess_vector: Vector,
         diis_start_size: int = 3, max_subspace_size: int = 7,
         conv_tol: float = 1e-9, n_max_iterations: int = 100,
         callback: Optional[DIISCallback] = None) -> Vector:
    """
    Implementation of the direct inversion of the iterative subspace algorithm.

    Parameters
    ----------
    updater: callable
        Callable that takes the guess vector and updates it once.
    guess_vector
        The guess vector to start with.
    diis_start_size: int, optional
        Perform n linear steps until the DIIS subspace contains at least
        n vectors. Has to be smaller than or equal to max_subspace_size
        (default: 3).
    max_subspace_size: int, optional
        The maximum number of vectors to keep in the subspace simultaneously
        (default: 7).
    conv_tol: float, optional
        The convergence tolerance on the Frobenius norm of the
        error vector (default: 1e-9).
    n_max_iterations: int, optional
        The maximum number of allowed iterations (default: 100).
    callback: DIISCallback, optional
        A callable that is called after each iteration, e.g., to produce
        printout.
    """
    if callback is None:
        callback = _no_print
    if n_max_iterations < 1:
        raise DIISError(f"The maximum number of iterations ({n_max_iterations=}) "
                        "can not be smaller than 1.")

    # initialize DIIS subspace
    diis_subspace: DIISSubspace[Vector] = DIISSubspace(
        max_size=max_subspace_size, start_size=diis_start_size
    )
    # perform an initial linear step
    new_subspace_vector = updater(guess_vector)
    new_error_vector = new_subspace_vector - guess_vector
    diis_subspace.add(new_subspace_vector, new_error_vector)
    diis_subspace.step_info = "Linear step from guess"
    callback(diis_subspace, "start")
    callback(diis_subspace, "next_iter")
    # check for convergence (very unlikely)
    if diis_subspace.is_converged(conv_tol):
        callback(diis_subspace, "is_converged")
        return diis_subspace.last_vector

    # iterate
    while diis_subspace.n_iter < n_max_iterations:
        diis_guess = diis_subspace.compute_guess()
        new_subspace_vector = updater(diis_guess)
        new_error_vector = new_subspace_vector - diis_guess
        diis_subspace.add(new_subspace_vector, new_error_vector)
        callback(diis_subspace, "next_iter")

        # check for convergence
        if diis_subspace.is_converged(conv_tol):
            callback(diis_subspace, "is_converged")
            break
    else:  # no convergence achieved within n_max_iterations iterations
        raise DIISError(
            f"No convergence detected after {n_max_iterations} iterations"
        )
    return diis_subspace.last_vector
