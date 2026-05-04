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
from collections.abc import Callable, Sequence
from enum import Enum
from typing import (
    cast, runtime_checkable, Generic, Literal, Protocol, TypeVar, TextIO
)
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
import warnings
import math
import sys

from ..functions import evaluate, lincomb
from ..timings import strtime, strtime_short, Timer
from .common import select_eigenpairs


class DavidsonError(Exception):
    pass


Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
Array1D_INT = np.ndarray[tuple[int], np.dtype[np.int64]]
Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]

VectorT = TypeVar("VectorT", bound="DavidsonVector")


class DavidsonMatrix(Protocol):
    @property
    def shape(self) -> tuple[int, int]:
        ...

    def __matmul__(self, vectors: Sequence[VectorT]) -> Sequence[VectorT]:
        ...


class DavidsonVector(Protocol):
    def __matmul__(self: VectorT, other: VectorT | Sequence[VectorT]) -> float:
        ...

    def __rmul__(self: VectorT, other: float) -> VectorT:
        ...

    def __add__(self: VectorT, other: VectorT) -> VectorT:
        ...


class DavidsonPreconditioner(Protocol):
    def __matmul__(self, vectors: Sequence[VectorT]) -> Sequence[VectorT]:
        ...


@runtime_checkable
class ShiftUpdatingPreconditioner(DavidsonPreconditioner, Protocol):
    def update_shifts(self, shifts: Array1D) -> None:
        ...


PrecondT = TypeVar("PrecondT", bound=DavidsonPreconditioner, covariant=True)


@runtime_checkable
class SupportsInstantiation(Protocol[PrecondT]):
    def __call__(self, matrix: DavidsonMatrix, *args, **kwargs) -> PrecondT:
        ...


class DavidsonSymmetrisation(Protocol):
    def symmetrise(self, vectors: Sequence[VectorT]) -> Sequence[VectorT]:
        ...


class DavidsonCallback(Protocol):
    def __call__(self, davidson: "Davidson", state: "State",
                 file: TextIO = sys.stdout) -> None:
        ...


class State(Enum):
    START = "start"
    NEXT_ITER = "next_iter"
    IS_CONVERGED = "is_converged"
    RESTART = "restart"


class Which(Enum):
    SA = "SA"
    LA = "LA"
    LM = "LM"
    SM = "SM"


class Davidson(Generic[VectorT, PrecondT]):
    def __init__(self, matrix: DavidsonMatrix, n_states: int,
                 max_subspace_size: int, block_size: int,
                 n_max_subspace_iter: int | None, which: Which,
                 debug_checks: bool = False,
                 preconditioner: PrecondT | None = None,
                 explicit_symmetrisation: DavidsonSymmetrisation | None = None
                 ) -> None:
        # some sanity checks
        assert n_states > 0
        assert max_subspace_size > 0
        assert block_size > 0
        assert n_max_subspace_iter is None or n_max_subspace_iter > 0
        assert block_size >= n_states
        assert max_subspace_size >= 2 * block_size
        # the matrix for which to determine eigenpairs
        self.matrix: DavidsonMatrix = matrix
        # parameters for the davidson configuration
        self.n_states: int = n_states
        self.max_subspace_size: int = max_subspace_size
        self.block_size: int = block_size
        self.n_max_subspace_iter: int | None = n_max_subspace_iter
        self.which: Which = which
        # preconditioning of the new subspace vectors
        self.preconditioner: PrecondT | None = preconditioner
        self.explicit_symmetrisation: DavidsonSymmetrisation | None = (
            explicit_symmetrisation
        )
        # data related to max_subspace_size
        self.subspace_vectors: list[VectorT] = []
        self.mvps: list[VectorT] = []
        self._projected_matrix: Array2D = np.full(
            (max_subspace_size, max_subspace_size), np.nan, dtype=np.float64
        )
        # rayleigh ritz data
        self.ritz_values: Array1D = np.full(
            (n_states,), np.nan, dtype=np.float64
        )
        self.ritz_vectors: Array2D = np.full(
            (max_subspace_size, max_subspace_size), np.nan, dtype=np.float64
        )
        # mask containing the indices of the relevant ritz eigenpairs
        self.ritz_pair_mask: Array1D_INT = np.full(
            (n_states,), fill_value=np.iinfo(np.int64).max, dtype=np.int64
        )
        # some debug information
        self.debug_checks: bool = debug_checks
        self.subspace_orthogonality: float = float("NaN")
        # data related to the number of states
        self.residuals: Sequence[VectorT] = []
        self.residual_norms: Array1D = np.full(
            (n_states,), np.nan, dtype=np.float64
        )
        self.eigenvalues: Array1D = np.full(
            (n_states,), np.nan, dtype=np.float64
        )
        self.eigenvectors: Sequence[VectorT] = []
        # timings and statistics
        self.n_iter: int = 0
        self.n_matrix_applies: int = 0
        self.timer: Timer = Timer()
        self.residuals_converged: np.ndarray[tuple[int], np.dtype[np.bool]] = (
            np.full((n_states,), False, dtype=np.bool)
        )
        self.reortho_triggers: list[float] = []

    @property
    def projected_matrix(self) -> Array2D:
        """Returns the valid work view of the projected matrix"""
        view = self._projected_matrix[
            :len(self.subspace_vectors), :len(self.subspace_vectors)
        ]
        assert not np.any(np.isnan(view))
        return view

    @property
    def converged(self) -> bool:
        """Whether all residuals are converged."""
        return bool(np.all(self.residuals_converged))

    @property
    def need_collapse(self) -> bool:
        """Whether growing the subpsace would exceed the maxuimum subspace sice."""
        return len(self.subspace_vectors) + self.block_size > self.max_subspace_size

    def add_to_subspace(self, vectors: Sequence[VectorT]) -> None:
        """
        Add the given vectors to the subspace, compute the matrix vector
        products and update the projected matrix.
        This also increments the iteration counter.
        """
        # add the given vectors to the subspace and update the projected
        # matrix. Also update the iteration counter
        if len(self.subspace_vectors) + len(vectors) > self.max_subspace_size:
            raise DavidsonError(f"Adding {len(vectors)} to the subspace would "
                                "exceed the max_subspace_size "
                                f"(= {self.max_subspace_size}).")
        self.n_iter += 1
        with self.timer.record("projection"):
            # compute the mvps
            mvps = self.matrix @ vectors
            assert len(mvps) == len(vectors)
            self.mvps.extend(mvps)
            self.n_matrix_applies += len(vectors)
            assert len(self.mvps) <= self.max_subspace_size
            # and update the projected matrix
            to_compute: Array2D = self._projected_matrix[
                len(self.subspace_vectors):, len(self.subspace_vectors):
            ]
            assert np.all(np.isnan(to_compute))
            for i in range(len(vectors)):
                for j in range(i, len(vectors)):
                    to_compute[i, j] = vectors[i] @ mvps[j]
                    if i != j:
                        to_compute[j, i] = to_compute[i, j]
            # ensure that the matrix is still in a valid state
            assert not np.any(np.isnan(to_compute[:len(vectors), :len(vectors)]))
            assert np.all(np.isnan(to_compute[len(vectors):, len(vectors):]))
        # finally update the subspace
        self.subspace_vectors.extend(vectors)
        assert len(self.subspace_vectors) <= self.max_subspace_size

    def collapse_subspace(self) -> None:
        """
        Collapse the subspace replacing the current subspace vectors
        using the provided ritz eigenvecators. Also updates the
        stored matrix vector products and the projected matrix.
        """
        with self.timer.record("projection"):
            # the new subspace vectors can not be added using
            # add_to_subspace, because that would trigger the
            # evaluation of mvps
            self.subspace_vectors = [
                lincomb(v, self.subspace_vectors, evaluate=True)
                for v in np.transpose(self.ritz_vectors)
            ]
            assert len(self.subspace_vectors) <= self.max_subspace_size
            assert len(self.subspace_vectors) >= self.block_size
            # update the mvps without actually reevaluating them
            self.mvps = [
                lincomb(v, self.mvps, evaluate=True)
                for v in np.transpose(self.ritz_vectors)
            ]
            assert len(self.mvps) == len(self.subspace_vectors)
            # reset the projected matrix and recompute relevant entries
            self._projected_matrix.fill(np.nan)
            for i in range(len(self.subspace_vectors)):
                for j in range(i, len(self.subspace_vectors)):
                    self._projected_matrix[i, j] = (
                        self.subspace_vectors[i] @ self.mvps[j]
                    )
                    if i != j:
                        self._projected_matrix[j, i] = self._projected_matrix[i, j]
            assert not np.any(np.isnan(self._projected_matrix[
                :len(self.subspace_vectors), :len(self.subspace_vectors)
            ]))
            assert np.all(np.isnan(self._projected_matrix[
                len(self.subspace_vectors):, len(self.subspace_vectors):
            ]))

    def compute_residuals(self) -> None:
        """
        Compute the residuals using the current ritz eigenvalues and eigenvectors.
        """
        assert len(self.ritz_pair_mask) == self.n_states
        with self.timer.record("residuals"):
            vectors = np.transpose(self.ritz_vectors)
            assert len(self.ritz_values) == len(vectors)
            self.residuals = [
                self._compute_residual(rval, vec)
                for i, (rval, vec) in enumerate(zip(self.ritz_values, vectors))
                if i in self.ritz_pair_mask
            ]
            assert len(self.residuals) == self.n_states
            self.residual_norms = np.array([
                np.sqrt(r @ r) for r in self.residuals
            ])
            assert len(self.residual_norms) == self.n_states

    def _compute_residual(self, ritz_value: float, ritz_vector: Array1D) -> VectorT:
        # Form residuals, A * SS * v - λ * SS * v = Ax * v + SS * (-λ*v)
        coefficients = (
            np.hstack((ritz_vector, -ritz_value * ritz_vector))
        )
        return lincomb(
            coefficients, self.mvps + self.subspace_vectors, evaluate=True
        )

    def compute_eigenvectors(self) -> None:
        """
        Compute the eigenvectors for the current ritz eigenpairs.
        """
        assert len(self.ritz_pair_mask) == self.n_states
        self.eigenvectors = [
            lincomb(v, self.subspace_vectors, evaluate=True)
            for i, v in enumerate(np.transpose(self.ritz_vectors))
            if i in self.ritz_pair_mask
        ]

    def solve_rayleigh_ritz(self) -> None:
        """
        Determine the eigenvalues and eigenstates of the projected matrix.
        Note that only the eigenvalues are updated automatically, while
        the eigenvalues of the Davidson procedure are only computed
        on request.
        """
        with self.timer.record("rayleigh_ritz"):
            matrix = self.projected_matrix
            if matrix.shape == (self.block_size, self.block_size):
                ritz_values, ritz_vectors = scipy.linalg.eigh(matrix)
            else:
                ritz_values, ritz_vectors = scipy.sparse.linalg.eigsh(
                    matrix, k=self.block_size, which=self.which.value,
                    maxiter=self.n_max_subspace_iter
                )
            self.ritz_values = ritz_values
            self.ritz_vectors = ritz_vectors
            # set the new eigenvalues and the mask
            self.ritz_pair_mask = select_eigenpairs(
                eigenvalues=ritz_values, n_ep=self.n_states, which=self.which.value
            )
            assert len(self.ritz_pair_mask) == self.n_states
            self.eigenvalues = ritz_values[self.ritz_pair_mask]
            assert len(self.eigenvalues) == self.n_states

    def apply_preconditioner(self) -> None:
        """Apply the preconditioner to the current residual vectors."""
        with self.timer.record("preconditioner"):
            if self.preconditioner is not None:
                if isinstance(self.preconditioner, ShiftUpdatingPreconditioner):
                    # Epsilon factor to make sure that 1 / (shift - diagonal)
                    # does not become ill-conditioned as soon as the shift
                    # approaches the actual diagonal values (which are the
                    # eigenvalues for the ADC(2) doubles part if the coupling
                    # block are absent)
                    self.preconditioner.update_shifts(self.ritz_values - 1e-6)
                self.residuals = cast(
                    list[VectorT],
                    evaluate(self.preconditioner @ self.residuals)
                )
            if self.explicit_symmetrisation is not None:
                self.residuals = self.explicit_symmetrisation.symmetrise(
                    self.residuals
                )

    def orthogonalise_residuals(self, residual_min_norm: float) -> list[VectorT]:
        """
        Orthogonalize the residuals with respect to the subspace and filter
        out residuals with a norm smaller than residual_min_norm.
        """
        new_vectors: list[VectorT] = []
        with self.timer.record("orthogonalisation"):
            assert len(self.residuals) >= self.block_size
            for i in range(self.block_size):
                residual = self.residuals[i]
                # Project out the components of the current subspace using
                # conventional Gram-Schmidt (CGS) procedure.
                # That is form (1 - SS * SS^T) * pvec = pvec + SS * (-SS^T * pvec)
                coefficients = np.hstack(([1], -(residual @ self.subspace_vectors)))
                proj_vec = lincomb(
                    coefficients, [residual] + self.subspace_vectors, evaluate=True
                )
                proj_norm = np.sqrt(proj_vec @ proj_vec)
                if proj_norm < residual_min_norm:
                    continue
                # Perform reorthogonalisation if loss of orthogonality is
                # detected; this comes at the expense of computing n_ss_vec
                # additional scalar products but avoids linear dependence
                # within the subspace.
                with self.timer.record("reorthogonalisation"):
                    ss_overlap = np.array(proj_vec @ self.subspace_vectors)
                    max_ortho_loss = np.max(np.abs(ss_overlap)) / proj_norm
                    if max_ortho_loss > self.matrix.shape[1] * np.finfo(float).eps:
                        # Update pvec by instance reorthogonalised against SS
                        # using a second CGS. Also update pnorm.
                        coefficients = np.hstack(([1], -ss_overlap))
                        proj_vec = lincomb(
                            coefficients, [proj_vec] + self.subspace_vectors,
                            evaluate=True
                        )
                        proj_norm = np.sqrt(proj_vec @ proj_vec)
                        self.reortho_triggers.append(max_ortho_loss)
                # Extend the subspace if the norm is sufficiently large
                if proj_norm >= residual_min_norm:
                    new_vectors.append(cast(
                        VectorT,
                        evaluate(proj_vec / proj_norm)
                    ))
            if self.debug_checks:
                SS = self.subspace_vectors + new_vectors
                orth = np.array([
                    [SS[i] @ SS[j] for i in range(len(SS))] for j in range(len(SS))
                ])
                orth -= np.eye(len(SS))
                self.subspace_orthogonality = np.max(np.abs(orth))
                if self.subspace_orthogonality > \
                        self.matrix.shape[1] * np.finfo(float).eps:
                    warnings.warn(scipy.linalg.LinAlgWarning(
                        "Subspace in Davidson has lost orthogonality. "
                        "Max. deviation from orthogonality is "
                        f"{self.subspace_orthogonality:.4E}. "
                        "Expect inaccurate results."
                    ))
        return new_vectors


def default_print(davidson: Davidson, state: State,
                  file: TextIO = sys.stdout) -> None:
    if state is state.START:
        print("Niter n_ss  max_residual  time  Ritz values",
              file=file)
    elif state is state.NEXT_ITER:
        time_iter = davidson.timer.current("iteration")
        fmt = "{n_iter:3d}  {ss_size:4d}  {residual:12.5g}  {tstr:5s}"
        print(fmt.format(n_iter=davidson.n_iter, tstr=strtime_short(time_iter),
                         ss_size=len(davidson.subspace_vectors),
                         residual=np.max(davidson.residual_norms)),
              "", davidson.eigenvalues[:7], file=file)
        if not math.isnan(davidson.subspace_orthogonality):
            print(33 * " " + "nonorth: {:5.3g}"
                  "".format(davidson.subspace_orthogonality))
    elif state is state.IS_CONVERGED:
        soltime = davidson.timer.total("iteration")
        print("=== Converged ===", file=file)
        print("    Number of matrix applies:   ", davidson.n_matrix_applies,
              file=file)
        print("    Total solver time:          ", strtime(soltime), file=file)
    elif state is state.RESTART:
        print("=== Restart ===", file=file)
    else:
        raise NotImplementedError(f"Invalid state {state}")


def no_print(davidson: Davidson, state: State, file: TextIO = sys.stdout) -> None:
    pass


def davidson_iterations(
    matrix: DavidsonMatrix, guess_vectors: Sequence[VectorT],
    n_states: int, block_size: int, max_subspace_size: int,
    is_converged: Callable[[Davidson], bool],
    callback: DavidsonCallback | None = None,
    which: Literal["SA", "LA", "LM", "SM"] | Which = "SA",
    n_max_iterations: int = 70,
    debug_checks: bool = False, residual_min_norm: float | None = None,
    preconditioner: PrecondT | SupportsInstantiation[PrecondT] | None = None,
    explicit_symmetrisation: DavidsonSymmetrisation | None = None,
    n_max_subspace_iter: int | None = None
) -> Davidson[VectorT, PrecondT]:

    validate_davidson_params(
        guess_vectors=guess_vectors, n_states=n_states, block_size=block_size,
        max_subspace_size=max_subspace_size
    )
    if callback is None:
        callback = no_print
    if not isinstance(which, Which):
        which = Which(which)
    if residual_min_norm is None:
        residual_min_norm = 2 * matrix.shape[1] * np.finfo(float).eps
    if isinstance(preconditioner, SupportsInstantiation):
        preconditioner = preconditioner(matrix)

    # init the davidson and add the guess vectors to the subspace
    davidson: Davidson[VectorT, PrecondT] = Davidson(
        matrix=matrix, n_states=n_states, max_subspace_size=max_subspace_size,
        block_size=block_size, n_max_subspace_iter=n_max_subspace_iter, which=which,
        debug_checks=debug_checks, preconditioner=preconditioner,
        explicit_symmetrisation=explicit_symmetrisation
    )
    callback(davidson, State.START)
    davidson.timer.restart("iteration")
    # compute the initial mvps and the projected matrix for the guess vectors
    davidson.add_to_subspace(guess_vectors)

    while True:
        # solve rayleigh_ritz and identify the n_states relevant
        # eigenvalues and vectors
        davidson.solve_rayleigh_ritz()
        # compute the residuals for these relevant ritz eigenpairs
        davidson.compute_residuals()
        callback(davidson, State.NEXT_ITER)
        davidson.timer.restart("iteration")
        # check for convergence and return if necessary
        if is_converged(davidson):
            davidson.compute_eigenvectors()
            callback(davidson, State.IS_CONVERGED)
            davidson.timer.stop("iteration")
            return davidson
        # maximum number of iterations reached? return
        if davidson.n_iter == n_max_iterations:
            warnings.warn(scipy.linalg.LinAlgWarning(
                f"Maximum number of iterations (== {n_max_iterations}) "
                "reached in davidson procedure."
            ))
            davidson.compute_eigenvectors()
            davidson.timer.stop("iteration")
            return davidson
        # do we need to collapse the subspace?
        if davidson.need_collapse:
            callback(davidson, State.RESTART)
            davidson.collapse_subspace()
        # apply the preconditioner to the residuals
        davidson.apply_preconditioner()
        # orthogonalise the new vectors
        new_vectors = davidson.orthogonalise_residuals(
            residual_min_norm=residual_min_norm
        )
        if len(new_vectors) == 0:
            warnings.warn(scipy.linalg.LinAlgWarning(
                "Davidson procedure could not generate any further vectors for "
                "the subspace. Iteration cannot be continued like this and will "
                "be aborted without convergence. Try a different guess."
            ))
            davidson.compute_eigenvectors()
            davidson.timer.stop("iteration")
            return davidson
        # add the new vectors to the subspace, compute the additional
        # matrix vector products and update the projected matrix
        davidson.add_to_subspace(new_vectors)


def validate_davidson_params(guess_vectors: Sequence[DavidsonVector],
                             n_states: int, block_size: int,
                             max_subspace_size: int):
    if n_states > len(guess_vectors):
        raise ValueError(f"n_states (= {n_states}) cannot exceed the number of "
                         f"guess vectors (= {len(guess_vectors)}).")

    if block_size < n_states:
        raise ValueError(f"block_size (= {block_size}) cannot be smaller than the "
                         f"number of states requested (= {n_states}).")
    elif block_size > len(guess_vectors):
        raise ValueError(f"block_size (= {block_size}) cannot exceed the number "
                         f"of guess vectors (= {len(guess_vectors)}).")

    if max_subspace_size < 2 * block_size:
        raise ValueError(f"max_subspace_size (= {max_subspace_size}) needs to be "
                         "at least twice as large as n_block "
                         f"(n_block = {block_size}).")
    elif max_subspace_size < len(guess_vectors):
        raise ValueError(f"max_subspace_size (= {max_subspace_size}) cannot be "
                         "smaller than the number of guess vectors "
                         f"(= {len(guess_vectors)}).")


def default_convergence_check(conv_tol: float) -> Callable[[Davidson], bool]:
    def is_converged(davidson: Davidson) -> bool:
        davidson.residuals_converged = davidson.residual_norms < conv_tol
        return davidson.converged
    return is_converged


def davidson(
    matrix: DavidsonMatrix, guess_vectors: Sequence[VectorT],
    n_states: int | None = None, block_size: int | None = None,
    max_subspace_size: int | None = None, conv_tol: float = 1e-9,
    which: Literal["SA", "LA", "LM", "SM"] = "SA",
    n_max_iterations: int = 70,
    callback: DavidsonCallback | None = None,
    preconditioner: PrecondT | SupportsInstantiation[PrecondT] | None = None,
    debug_checks: bool = False, residual_min_norm: float | None = None,
    explicit_symmetrisation: DavidsonSymmetrisation | None = None,
    n_max_subspace_iter: int | None = None
) -> Davidson[VectorT, PrecondT]:
    """
    Parameters
    ----------
    which: Literal["SA", "LA", "LM", "SM"]
        Which eigenvalues to compute:
        - SA: Smallest algebraic
        - LA: Largest algebraic
        - LM: Largest magnitude
        - SM: Smallest magnitude
    """
    # apply defaults
    if n_states is None:
        n_states = len(guess_vectors)
    if block_size is None:
        block_size = n_states
    if max_subspace_size is None:
        # TODO Arnoldi uses this:
        # max_subspace = max(2 * n_ep + 1, 20)
        max_subspace_size = max(6 * n_states, 20, 5 * len(guess_vectors))

    if conv_tol < matrix.shape[1] * np.finfo(float).eps:
        warnings.warn(scipy.linalg.LinAlgWarning(
            "Convergence tolerance (== {:5.2g}) lower than "
            "estimated maximal numerical accuracy (== {:5.2g}). "
            "Convergence might be hard to achieve."
            "".format(conv_tol, matrix.shape[1] * np.finfo(float).eps)
        ))
    return davidson_iterations(
        matrix=matrix, guess_vectors=guess_vectors, n_states=n_states,
        block_size=block_size, max_subspace_size=max_subspace_size,
        is_converged=default_convergence_check(conv_tol=conv_tol),
        callback=callback, which=which, n_max_iterations=n_max_iterations,
        debug_checks=debug_checks, residual_min_norm=residual_min_norm,
        preconditioner=preconditioner,
        explicit_symmetrisation=explicit_symmetrisation,
        n_max_subspace_iter=n_max_subspace_iter
    )
