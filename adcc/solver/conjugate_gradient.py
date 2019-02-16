#!/usr/bin/env python3
import numpy as np

from ..functions import dot
from .preconditioner import PreconditionerIdentity
from .explicit_symmetrisation import IndexSymmetrisation

import scipy.linalg as la


class State:
    def __init__(self):
        self.solution = None             # Current approximation to the solution
        self.residual = None             # Current residual
        self.residual_norm = None        # Current residual norm
        self.converged = False           # Flag whether iteration is converged
        self.n_iter = 0                  # Number of iterations
        self.n_applies = 0               # Number of applies


def conjugate_gradient(matrix, rhs, x0=None, conv_tol=1e-9, max_iter=100,
                       callback=None, Pinv=None, cg_type="polak_ribiere",
                       explicit_symmetrisation=IndexSymmetrisation):
    """
    Implements the "flexible" conjugate gradient using the Polak-Ribière
    formula, but allows to employ the "traditional" Fletcher-Reeves
    formula as well.

    Solves matrix @ x = rhs for x by minimising the residual matrix @ x - rhs

    matrix           system matrix
    rhs              right-hand side, source
    x0               initial guess, random if not specified
    conv_tol         Convergence tolerance (l2-norm of the residual)
    max_iter         Maximum number of iterations used
    Pinv             Preconditioner to A, typically an estimate for A^{-1}
    cg_type          Select between polak_ribiere and fletcher_reeves
    """
    raise NotImplementedError("Not yet properly implemented")

    if callback is None:
        def callback(state, identifier):
            pass

    # The problem size
    n_problem = matrix.shape[1]

    if x0 is None:
        # Start with random guess
        raise NotImplementedError("Random guess is not yet implemented.")
        x0 = np.random.rand((n_problem))

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
    state.residual = rhs - matrix @ state.solution
    state.n_applies += 1
    state.residual_norm = np.sqrt(state.residual @ state.residual)
    pk = zk = Pinv @ state.residual

    # TODO Index symmetrisation

    callback(state, "start")
    while state.n_iter < max_iter:
        state.n_iter += 1

        # Update ak and iterated solution
        # TODO This needs to be modified for general optimisations,
        #      i.e. where A is non-linear
        Apk = matrix @ pk
        res_dot_zk = dot(state.residual, zk)
        ak = float(res_dot_zk / dot(pk, Apk))
        state.solution += ak * pk

        residual_old = state.residual
        state.residual = residual_old - ak * Apk
        state.residual_norm = np.sqrt(state.residual @ state.residual)

        callback(state, "next_iter")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            return state

        if state.n_iter == max_iter:
            raise la.LinAlgError("Maximum number of iterations (== " +
                                 str(max_iter) + " reached in davidson "
                                 "procedure.")

        zk = Pinv @ state.residual
        if cg_type == "fletcher_reeves":
            bk = float(dot(zk, state.residual) / res_dot_zk)
        elif cg_type == "polak_ribiere":
            bk = float(dot(zk, (state.residual - residual_old)) / res_dot_zk)
        pk = zk + bk * pk

        # TODO Index symmetrisation
