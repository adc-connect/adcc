#!/usr/bin/env python3
# vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import numpy as np
from collections import deque
import sys


class DIISError(Exception):
    """is the base class of errors visible from this module."""
    pass


class SubspaceError(DIISError):
    """is raised when encountering problems related to the subspace setup."""
    pass


class DIISSubspace:
    """
    Manages the DIIS matrix and the subspace and error vectors.

    Parameters
    ----------
    max_size: int
        The maximum number of vectors to keep in the subspace simultaneously.
    """
    def __init__(self, max_size: int):
        if max_size < 2:
            raise SubspaceError("DIIS size has to be greater than 1.")
        self.max_size: int = max_size
        self.subspace_vectors = deque(maxlen=max_size)
        self.error_vectors = deque(maxlen=max_size)
        self.overlap: np.ndarray = np.zeros((0, 0))
        self.step_info: str = None
        self.converged: bool = False
        self.n_iter: int = 0

    def is_converged(self, conv_tol: float) -> bool:
        """
        Computes the square root of the norm of the last error vector and checks
        whether it is smaller than the given tolerance.
        """
        if self.n_iter >= 2 and self.residual_norm < conv_tol:
            self.converged = True
        return self.converged

    @property
    def residual_norm(self) -> float:
        """Returns the norm of the most recent error vector"""
        return np.sqrt(self.error_vectors[-1].dot(self.error_vectors[-1]))

    @property
    def size(self) -> int:
        """Returns the current subspace size"""
        return len(self.subspace_vectors)

    @property
    def last_vector(self):
        """
        Returns the most recent subspace vector, that is, in case of
        convergence, the solution vector.
        """
        return self.subspace_vectors[-1]

    def add(self, subspace_vector, error_vector) -> None:
        """
        Add a new vector and error vector to the subspace and updates the
        DIIS matrix. Also the iteration counter is incremented.
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
        new_overlap = np.zeros((cur_size, cur_size))
        new_overlap.fill(float("NaN"))
        new_overlap[:copy_size, :copy_size] = self.overlap[-copy_size:, -copy_size:]
        for ind, error_vec in enumerate(self.error_vectors):
            overlap_value = error_vec.dot(self.error_vectors[-1])
            new_overlap[ind, -1] = overlap_value
            # create transposed element if this is not the diagonal element
            if ind != cur_size - 1:
                new_overlap[-1, ind] = overlap_value
        # Check that there are no NaN elements in the updated overlap matrix,
        # i.e., that all elements have been properly populated
        assert not np.isnan(np.sum(new_overlap))
        self.overlap = new_overlap

    def compute_guess(self, n_omit_vectors: int = 0):
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
        n_omit_vectors
            Omit the n subspace vectors with the largest error vectors in the
            extrapolation.
        """
        if not self.size:
            raise SubspaceError("No vectors in subspace.")

        # we only have a single vector -> perform a linear step
        if self.size == 1 or self.size - n_omit_vectors == 1:
            self.step_info = "Linear step"
            return self.last_vector

        fill_value = -1.0
        diis_size = self.size
        # build the DIIS matrix
        A = np.zeros((diis_size + 1, diis_size + 1))
        A.fill(fill_value)
        A[-1, -1] = 0.0
        A[:diis_size, :diis_size] = self.overlap / self.overlap[0, 0]
        # build the right hand side
        b = np.zeros(diis_size + 1)
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
        # build the linear combination:
        # determine the indices of not discarded vectors to combine with the
        # corresponding coefficients
        vec_indices = [i for i in range(self.size) if i not in inds_to_discard]
        guess = self.subspace_vectors[0].zeros_like()
        for vec_ind, coeff in zip(vec_indices, coefficients):
            guess += self.subspace_vectors[vec_ind] * coeff

        self.step_info = (
            f"DIIS step from {len(vec_indices)} error vectors, subspace size: "
            f"{self.size}"
        )
        return guess


def default_print(state: DIISSubspace, identifier: str, file=sys.stdout):
    if identifier == "start":
        print("Niter   DIIS_error  comment", file=file)
    elif identifier == "next_iter":
        fmt = "{n_iter:4d} {residual: >13.5E}  {step_info:s}"
        print(fmt.format(n_iter=state.n_iter,
                         residual=state.residual_norm,
                         step_info=state.step_info), file=file)
    elif identifier == "is_converged":
        print("=== Converged ===", file=file)


def diis(updater: callable, guess_vector, diis_start_size: int = 3,
         max_subspace_size: int = 7, conv_tol: float = 1e-9,
         n_max_iterations: int = 100, callback: callable = None):
    """
    Implementation of the direct inversion of the iterative subspace algorithm.

    Parameters
    ----------
    updater: callable
        Callable that takes the guess vector and updates it once.
    guess_vector
        The guess vector to start with.
    diis_start_size: int
        Perform n linear steps until the DIIS subspace contains at least
        n vectors. Has to be smaller than max_subspace_size (default: 3).
    max_subspace_size: int
        The maximum number of vectors to keep in the subspace simultaneously
        (default 7).
    conv_tol: float
        The convergence tolerance on the euclidean norm of the
        error vector. (default: 1e-9)
    n_max_iterations: int
        The maximum number of allowed iterations (default: 100).
    callback: callable
        A callable that is called after each iteration, e.g., to produce
        printout.
    """

    if callback is None:
        def callback(state, hint):
            pass

    if diis_start_size > max_subspace_size:
        raise SubspaceError("diis_start_size cannot be greater than "
                            "max_subspace_size")

    # initialize DIIS subspace
    diis_subspace = DIISSubspace(max_size=max_subspace_size)
    # perform an initial linear step
    new_subspace_vector = updater(guess_vector)
    new_error_vector = new_subspace_vector - guess_vector
    diis_subspace.add(new_subspace_vector, new_error_vector)
    diis_subspace.step_info = "Linear step from guess"
    callback(diis_subspace, "start")
    callback(diis_subspace, "next_iter")

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
