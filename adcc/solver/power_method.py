#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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

from .SolverStateBase import EigenSolverStateBase
from .explicit_symmetrisation import IndexSymmetrisation

import scipy.linalg as la


class PowerMethodState(EigenSolverStateBase):
    def __init__(self, A):
        super().__init__(A)
        self.residuals = None
        self.algorithm = "power_method"


def default_print(state, identifier, file=sys.stdout):
    """
    A default print function for the power_method callback
    """
    from adcc.timings import strtime, strtime_short

    if identifier == "start" and state.n_iter == 0:
        print("Niter residual  time  Ritz value", file=file)
    elif identifier == "next_iter":
        time_iter = state.timer.current("power_method/iteration")
        fmt = "{n_iter:3d}  {residual:12.5g}  {tstr:5s}"
        print(fmt.format(n_iter=state.n_iter, tstr=strtime_short(time_iter),
                         residual=np.max(state.residual_norms)),
              "", state.eigenvalues[:7], file=file)
    elif identifier == "is_converged":
        soltime = state.timer.total("power_method/iteration")
        print("=== Converged ===", file=file)
        print("    Number of matrix applies:   ", state.n_applies)
        print("    Total solver time:          ", strtime(soltime))


def power_method(A, guess, conv_tol=1e-9, max_iter=70, callback=None,
                 explicit_symmetrisation=IndexSymmetrisation):
    """Use the power iteration to solve for the largest eigenpair of A.

    The power method is a very simple diagonalisation method, which solves
    for the (by magnitude) largest eigenvalue of the matrix `A`.

    Parameters
    ----------
    A
        Matrix object. Only the `@` operator needs to be implemented.
    guess
        Matrix used as a guess
    conv_tol : float
        Convergence tolerance on the l2 norm of residuals to consider
        them converged.
    max_iter : int
        Maximal numer of iterations
    callback
        Callback function called after each iteration
    explicit_symmetrisation
        Explicit symmetrisation to perform during iteration to ensure
        obtaining an eigenvector with matching symmetry criteria.
    """
    if callback is None:
        def callback(state, identifier):
            pass

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(A)

    x = guess / np.sqrt(guess @ guess)
    state = PowerMethodState(A)

    def is_converged(state):
        return state.residual_norms[0] < conv_tol

    callback(state, "start")
    state.timer.restart("power_method/iteration")
    for i in range(max_iter):
        state.n_iter += 1
        Ax = A @ x
        state.n_applies += 1

        eigval = x @ (Ax)
        residual = Ax - eigval * x
        residual_norm = np.sqrt(residual @ residual)
        state.eigenvalues = np.array([eigval])
        state.eigenvectors = np.array([x])
        state.residual_norms = np.array([residual_norm])

        callback(state, "next_iter")
        state.timer.restart("power_method/iteration")
        if is_converged(state):
            state.converged = True
            callback(state, "is_converged")
            state.timer.stop("power_method/iteration")
            return state

        if explicit_symmetrisation:
            x = explicit_symmetrisation.symmetrise(Ax)
        else:
            x = Ax
        x = x / np.sqrt(x @ x)

    warnings.warn(la.LinAlgWarning(
        "Power method not converged. Returning intermediate results."))
    return state
