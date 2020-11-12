#!/usr/bin/env python3
import sys
import numpy as np
import scipy.linalg as la

from adcc import copy, evaluate

from ..functions import dot
from .preconditioner import PreconditionerIdentity
from .explicit_symmetrisation import IndexSymmetrisation


def guess_from_previous(matrix, rhs, previous_cgstate):
    """
    Gets passed the matrix, the cgstate from the previous solution
    and the new RHS and may use this information to construct
    a guess for the next CG to perform.
    """
    if previous_cgstate is None:
        return rhs
    else:
        return previous_cgstate.solution


def guess_from_rhs(matrix, rhs, previous_cgstate):
    return rhs


class IterativeInverse:
    def __init__(self, matrix, construct_guess=guess_from_rhs, **kwargs):
        """Initialise an iterative inverse

        This object mimics to be the inverse of a passed matrix
        by solving linear system's of equations each time the `@`
        operator is used in conjunction with this matrix.

        Parameters
        ----------
        matrix
            Matrix object
        """
        self.matrix = matrix
        self.kwargs = kwargs
        self.construct_guess = construct_guess
        self.cgstate = None

    @property
    def shape(self):
        return self.matrix.shape  # Inversion does not change the shape

    def __matmul__(self, x):
        if isinstance(x, list):
            return [self.__matmul__(xi) for xi in x]
        else:
            guess = self.construct_guess(self.matrix,  x, self.cgstate)
            self.cgstate = conjugate_gradient(self.matrix, x, guess,
                                              **self.kwargs)
            return self.cgstate.solution


class State:
    def __init__(self):
        self.solution = None       # Current approximation to the solution
        self.residual = None       # Current residual
        self.residual_norm = None  # Current residual norm
        self.converged = False     # Flag whether iteration is converged
        self.n_iter = 0            # Number of iterations
        self.n_applies = 0         # Number of applies


def default_print(state, identifier, file=sys.stdout):
    if identifier == "start" and state.n_iter == 0:
        print("Niter residual_norm", file=file)
    elif identifier == "next_iter":
        fmt = "{n_iter:3d}  {residual:12.5g}"
        print(fmt.format(n_iter=state.n_iter,
                         residual=np.max(state.residual_norm)), file=file)
    elif identifier == "is_converged":
        print("=== Converged ===", file=file)
        print("    Number of matrix applies:   ", state.n_applies)


def conjugate_gradient(matrix, rhs, x0=None, conv_tol=1e-9, max_iter=100,
                       callback=None, Pinv=None, cg_type="polak_ribiere",
                       explicit_symmetrisation=IndexSymmetrisation):
    """An implementation of the conjugate gradient algorithm.

    This algorithm implements the "flexible" conjugate gradient using the
    Polak-Ribière formula, but allows to employ the "traditional"
    Fletcher-Reeves formula as well.
    It solves `matrix @ x = rhs` for `x` by minimising the residual
    `matrix @ x - rhs`.

    Parameters
    ----------
    matrix
        Matrix object. Should be an ADC matrix.
    rhs
        Right-hand side, source.
    x0
        Initial guess
    conv_tol : float
        Convergence tolerance on the l2 norm of residuals to consider
        them converged.
    max_iter : int
        Maximum number of iterations
    callback
        Callback to call after each iteration
    Pinv
        Preconditioner to A, typically an estimate for A^{-1}
    cg_type : string
        Identifier to select between polak_ribiere and fletcher_reeves
    explicit_symmetrisation
        Explicit symmetrisation to perform during iteration to ensure
        obtaining an eigenvector with matching symmetry criteria.
    """
    if callback is None:
        def callback(state, identifier):
            pass

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    if x0 is None:
        # Start with random guess
        raise NotImplementedError("Random guess is not yet implemented.")
    else:
        x0 = copy(x0)

    if Pinv is None:
        Pinv = PreconditionerIdentity()
    if Pinv is not None and isinstance(Pinv, type):
        Pinv = Pinv(matrix)

    def is_converged(state):
        state.converged = state.residual_norm < conv_tol
        return state.converged

    state = State()

    # Initialise iterates
    state.solution = x0
    state.residual = evaluate(rhs - matrix @ state.solution)
    state.n_applies += 1
    state.residual_norm = np.sqrt(state.residual @ state.residual)
    pk = zk = Pinv @ state.residual

    if explicit_symmetrisation:
        # TODO Not sure this is the right spot ... also this syntax is ugly
        pk = explicit_symmetrisation.symmetrise(pk)

    callback(state, "start")
    while state.n_iter < max_iter:
        state.n_iter += 1

        # Update ak and iterated solution
        # TODO This needs to be modified for general optimisations,
        #      i.e. where A is non-linear
        # https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
        Apk = matrix @ pk
        state.n_applies += 1
        res_dot_zk = dot(state.residual, zk)
        ak = float(res_dot_zk / dot(pk, Apk))
        state.solution = evaluate(state.solution + ak * pk)

        residual_old = state.residual
        state.residual = evaluate(residual_old - ak * Apk)
        state.residual_norm = np.sqrt(state.residual @ state.residual)

        callback(state, "next_iter")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            return state

        if state.n_iter == max_iter:
            raise la.LinAlgError("Maximum number of iterations (== "
                                 + str(max_iter) + " reached in conjugate "
                                 "gradient procedure.")

        zk = evaluate(Pinv @ state.residual)

        if explicit_symmetrisation:
            # TODO Not sure this is the right spot ... also this syntax is ugly
            zk = explicit_symmetrisation.symmetrise(zk)

        if cg_type == "fletcher_reeves":
            bk = float(dot(zk, state.residual) / res_dot_zk)
        elif cg_type == "polak_ribiere":
            bk = float(dot(zk, (state.residual - residual_old)) / res_dot_zk)
        pk = zk + bk * pk
