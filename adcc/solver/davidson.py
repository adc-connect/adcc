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
from adcc.AdcMatrix import AdcMatrixlike, AdcMatrixFolded
from adcc.AmplitudeVector import AmplitudeVector

from .common import select_eigenpairs
from .preconditioner import JacobiPreconditioner
from .SolverStateBase import EigenSolverStateBase
from .explicit_symmetrisation import IndexSymmetrisation
from itertools import product


class DavidsonState(EigenSolverStateBase):
    def __init__(self, matrix, guesses):
        super().__init__(matrix)
        self.residuals = None                   # Current residuals
        self.subspace_vectors = guesses.copy()  # Current subspace vectors
        self.algorithm = "davidson"
        self.DIIS_iter = 0                      # Total number of DIIS iterations
        self.folded = False                     # Folded or normal ADC matrix


class FoldedDavidsonState(DavidsonState):
    def __init__(self, matrix, guesses):
        super().__init__(matrix, guesses)       # with folded ADC matrix
        self.folded = True                      # Folded or normal ADC matrix
        self.n_state = None                     # Current state
        self.macro_iter = 0                     # Number of macro iterations
        self.history_rval = []                  # Previous Ritz values
        self.energy_diff = None                 # Difference between Ritz values
        self.residual_norm = None               # Current residual norm of one state
        self.residual = None                    # Current residual of one state
        self.eigenvector = None                 # Current eigenvector of one state
        self.converged_macro = False            # Convergence of macro iteration
        self.converged_diis = False             # Convergence of DIIS


def default_print(state, identifier, file=sys.stdout):
    """
    A default print function for the davidson callback
    """
    from adcc.timings import strtime, strtime_short

    # TODO Use colour!

    if identifier == "start" and state.n_iter == 0:
        if not state.folded:
            print("Niter n_ss  max_residual  time  Ritz values", file=file)
        else:
            print("Niter n_ss  energy_difference  time  Ritz values",
                  file=file)
    elif identifier == "next_iter":
        time_iter = state.timer.current("iteration")
        if not state.folded:
            fmt = "{n_iter:3d}  {ss_size:4d}  {residual:12.5g}  {tstr:5s}"
            print(fmt.format(n_iter=state.n_iter,
                             tstr=strtime_short(time_iter),
                             ss_size=len(state.subspace_vectors),
                             residual=np.max(state.residual_norms)),
                  "", state.eigenvalues[:10], file=file)
        else:
            fmt = "{n_iter:3d}  {ss_size:4d}  {energy_diff:12.5g}  {tstr:5s}"
            print(fmt.format(n_iter=state.n_iter,
                             tstr=strtime_short(time_iter),
                             ss_size=len(state.subspace_vectors),
                             energy_diff=state.energy_diff),
                  "", state.eigenvalues[:10], file=file)
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
    # For folded matrix
    elif identifier == "folded_start":
        print(f"============ State {state.n_state} ============", file=file)
        print("folded_matrix.omega(initial):", state.matrix.omega, file=file)
    elif identifier == "micro":
        print(f"--macro iteration {state.macro_iter}, ",
              f"Number of micro iterations: {state.n_iter}, "
              f"Ritz_value: {state.eigenvalues[state.n_state]}, ",
              f"residual_norm: {state.residual_norm}, ",
              f"Converged: {state.converged}--", file=file)
    elif identifier == "macro_stop":
        print("== Summary of macro iterations ==", file=file)
        print(" Number of macro-iterations:", state.macro_iter,
              " n_applies:", state.n_applies, file=file)
        print(" Ritz value:", state.eigenvalues[state.n_state],
              " residual norm:", state.residual_norm,
              " Converged or not:", state.converged_macro, file=file)
        print(" time:", strtime_short(state.timer.current
                                      ("folded iterations")), file=file)
    elif identifier == "DIIS_steps":
        print(f"--DIIS, Omega: {state.matrix.omega}, "
              f"residual_norm: {state.residual_norm}--", file=file)
    elif identifier == "DIIS_stop":
        print("== Summary of DIIS ==", file=file)
        print(" Number of DIIS iterations:", state.DIIS_iter, file=file)
        print(" Omega:", state.eigenvalues[state.n_state],
              " residual norm:", state.residual_norm,
              " Converged or not:", state.converged_diis, file=file)
        print(" time:", strtime_short(state.timer.current
                                      ("folded iterations")), file=file)
    elif identifier == "sum_folded":
        print("========= Converged (folded matrix) =========", file=file)
        print(" Number of matrix applies: ", state.n_applies, file=file)
        print(" Total solver time: ", strtime(state.timer.total
                                              ("folded iterations")), file=file)
        print(" Number of Davidson iterations: ", state.n_iter, file=file)
        print(" Number of DIIS: ", state.DIIS_iter, file=file)


# TODO This function should be merged with eigsh
def davidson_iterations(matrix, state, max_subspace, max_iter, n_ep, n_block,
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
    max_subspace : int
        Maximal subspace size
    max_iter : int
        Maximal number of iterations
    n_ep : int
        Number of eigenpairs to be computed
    n_block : int
        Davidson block size: the number of vectors that are added to the subspace
        in each iteration
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
    if state.folded:  # initialization for each micro iteration
        state.history_rval = [matrix.omega]
        state.n_iter = 0

    # The problem size
    n_problem = matrix.shape[1]

    # The current subspace size == Number of guesses
    n_ss_vec = len(state.subspace_vectors)

    # Sanity checks for block size
    assert n_block >= n_ep and n_block <= n_ss_vec

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

    # Get the worksize view for the first iteration
    Ass = Ass_cont[:n_ss_vec, :n_ss_vec]

    # Initial projection of Ax onto the subspace exploiting the hermiticity
    with state.timer.record("projection"):
        for i in range(n_ss_vec):
            for j in range(i, n_ss_vec):
                Ass[i, j] = SS[i] @ Ax[j]
                if i != j:
                    Ass[j, i] = Ass[i, j]

    while state.n_iter < max_iter:
        state.n_iter += 1

        assert len(SS) >= n_block
        assert len(SS) <= max_subspace

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
            if not state.folded:
                state.residual_norms = np.array([np.sqrt(r @ r)
                                                 for r in state.residuals])
            else:
                state.energy_diff = np.abs(state.eigenvalues[state.n_state]
                                           - state.history_rval[-1])
                state.history_rval.append(state.eigenvalues[state.n_state])

        callback(state, "next_iter")
        state.timer.restart("iteration")
        if is_converged(state):
            # Build the eigenvectors we desire from the subspace vectors:
            if not state.folded:
                state.eigenvectors = [lincomb(v, SS, evaluate=True)
                                      for i, v in enumerate(np.transpose(rvecs))
                                      if i in epair_mask]
                callback(state, "is_converged")
            else:
                # update guesses vectors for next macro iteration
                state.subspace_vectors = [lincomb(v, SS, evaluate=True)
                                          for v in np.transpose(rvecs)]
                assert len(state.subspace_vectors) == n_block
                state.eigenvector = state.subspace_vectors[epair_mask
                                                           [state.n_state]]
            state.converged = True
            state.timer.stop("iteration")
            return state

        if state.n_iter == max_iter:
            warnings.warn(la.LinAlgWarning(
                f"Maximum number of iterations (== {max_iter}) "
                "reached in davidson procedure."))
            if not state.folded:
                state.eigenvectors = [lincomb(v, SS, evaluate=True)
                                      for i, v in enumerate(np.transpose(rvecs))
                                      if i in epair_mask]
            else:
                # update guesses vectors for next macro iteration
                state.subspace_vectors = [lincomb(v, SS, evaluate=True)
                                          for v in np.transpose(rvecs)]
                assert len(state.subspace_vectors) == n_block
                state.eigenvector = state.subspace_vectors[epair_mask
                                                           [state.n_state]]
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
                    for j in range(i, n_ss_vec):
                        Ass[i, j] = SS[i] @ Ax[j]
                        if i != j:
                            Ass[j, i] = Ass[i, j]
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
                # Project out the components of the current subspace using
                # conventional Gram-Schmidt (CGS) procedure.
                # That is form (1 - SS * SS^T) * pvec = pvec + SS * (-SS^T * pvec)
                coefficients = np.hstack(([1], -(pvec @ SS)))
                pvec = lincomb(coefficients, [pvec] + SS, evaluate=True)
                pnorm = np.sqrt(pvec @ pvec)
                if pnorm < residual_min_norm:
                    continue
                # Perform reorthogonalisation if loss of orthogonality is
                # detected; this comes at the expense of computing n_ss_vec
                # additional scalar products but avoids linear dependence
                # within the subspace.
                with state.timer.record("reorthogonalisation"):
                    ss_overlap = np.array(pvec @ SS)
                    max_ortho_loss = np.max(np.abs(ss_overlap)) / pnorm
                    if max_ortho_loss > n_problem * eps:
                        # Update pvec by instance reorthogonalised against SS
                        # using a second CGS. Also update pnorm.
                        coefficients = np.hstack(([1], -ss_overlap))
                        pvec = lincomb(coefficients, [pvec] + SS, evaluate=True)
                        pnorm = np.sqrt(pvec @ pvec)
                        state.reortho_triggers.append(max_ortho_loss)
                if pnorm >= residual_min_norm:
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
                        "Subspace in Davidson has lost orthogonality. "
                        "Max. deviation from orthogonality is {:.4E}. "
                        "Expect inaccurate results.".format(
                            state.subspace_orthogonality)
                    ))

        if n_ss_added == 0:
            state.timer.stop("iteration")
            state.converged = False
            if not state.folded:
                state.eigenvectors = [lincomb(v, SS, evaluate=True)
                                      for i, v in enumerate(np.transpose(rvecs))
                                      if i in epair_mask]
            else:
                # Compute all eigenvectors as guesses vectors
                # for the next macro iteration.
                state.subspace_vectors = [lincomb(v, SS, evaluate=True)
                                          for v in np.transpose(rvecs)]
                assert len(state.subspace_vectors) == n_block
                state.eigenvector = state.subspace_vectors[epair_mask
                                                           [state.n_state]]
            warnings.warn(la.LinAlgWarning(
                "Davidson procedure could not generate any further vectors for "
                "the subspace. Iteration cannot be continued like this and will "
                "be aborted without convergence. Try a different guess."))
            return state

        # Matrix applies for the new vectors
        with state.timer.record("projection"):
            Ax.extend(matrix @ SS[-n_ss_added:])
            state.n_applies += n_ss_added

        # Update the worksize view for the next iteration
        Ass = Ass_cont[:n_ss_vec, :n_ss_vec]

        # Project Ax onto the subspace, keeping in mind
        # that the values Ass[:-n_ss_added, :-n_ss_added] are already valid,
        # since they have been computed in the previous iterations already.
        with state.timer.record("projection"):
            for i in range(n_ss_vec - n_ss_added, n_ss_vec):
                for j in range(i + 1):
                    Ass[i, j] = SS[i] @ Ax[j]
                    if i != j:
                        Ass[j, i] = Ass[i, j]


def eigsh(matrix, guesses, n_ep=None, n_block=None, max_subspace=None,
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
    n_block : int or NoneType, optional
        The solver block size: the number of vectors that are added to the subspace
        in each iteration
    max_subspace : int or NoneType, optional
        Maximal subspace size
    conv_tol : float, optional
        Convergence tolerance on the l2 norm of residuals to consider
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
        raise ValueError(f"n_ep (= {n_ep}) cannot exceed the number of guess "
                         f"vectors (= {len(guesses)}).")

    if n_block is None:
        n_block = n_ep
    elif n_block < n_ep:
        raise ValueError(f"n_block (= {n_block}) cannot be smaller than the number "
                         f"of states requested (= {n_ep}).")
    elif n_block > len(guesses):
        raise ValueError(f"n_block (= {n_block}) cannot exceed the number of guess "
                         f"vectors (= {len(guesses)}).")

    if not max_subspace:
        # TODO Arnoldi uses this:
        # max_subspace = max(2 * n_ep + 1, 20)
        max_subspace = max(6 * n_ep, 20, 5 * len(guesses))
    elif max_subspace < 2 * n_block:
        raise ValueError(f"max_subspace (= {max_subspace}) needs to be at least "
                         f"twice as large as n_block (n_block = {n_block}).")
    elif max_subspace < len(guesses):
        raise ValueError(f"max_subspace (= {max_subspace}) cannot be smaller than "
                         f"the number of guess vectors (= {len(guesses)}).")

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
                        n_ep=n_ep, n_block=n_block, is_converged=convergence_test,
                        callback=callback, which=which,
                        preconditioner=preconditioner,
                        preconditioning_method=preconditioning_method,
                        debug_checks=debug_checks,
                        residual_min_norm=residual_min_norm,
                        explicit_symmetrisation=explicit_symmetrisation)
    return state


def eigsh_folded(matrix, guesses, omegas=None, n_ep=None, n_block=None,
                 max_subspace=None,
                 conv_tol=1e-9, which="SA", max_iter=70,
                 callback=None, preconditioner=None,
                 preconditioning_method="Davidson",
                 debug_checks=False, residual_min_norm=None,
                 explicit_symmetrisation=IndexSymmetrisation,
                 macro_conv_tol=1e-3, macro_max_iter=30,
                 num_diis_vecs=50, diis_max_iter=200):
    """Davidson eigensolver for ADC problems with doubles-folding

    Parameters
    ----------
    matrix
        ADC(2) matrix instance
    guesses : list
        Guess vectors (fixes also the Davidson block size)
    n_ep : int or NoneType, optional
        Number of eigenpairs to be computed
    n_block : int or NoneType, optional
        The solver block size: the number of vectors that are added to the subspace
        in each iteration
    max_subspace : int or NoneType, optional
        Maximal subspace size
    conv_tol : float, optional
        Convergence tolerance on the l2 norm of residuals to consider
        them converged during the final DIIS iterations.
    which : str, optional
        Which eigenvectors to converge to (e.g. LM, LA, SM, SA)
    max_iter : int, optional
        Maximal number of Davidson iterations
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
    macro_conv_tol : float, optional (default=1e-3)
        Convergence tolerance on the l2 norm of residuals to consider
        them converged during macro iterations.
    macro_max_iter : int, optional (default=30)
        Maximal number of macro iterations
    num_diis_vecs: int, optional (default=50)
        Maximal number of DIIS vectors to keep
    diis_max_iter : int, optional (default=200)
        Maximal number of DIIS iterations
    """
    if callback is None:
        def callback(state, identifier):
            pass
    if not isinstance(matrix, AdcMatrixlike):
        raise TypeError("matrix is not of type AdcMatrixlike")
    for guess in guesses:
        if not isinstance(guess, AmplitudeVector):
            raise TypeError("One of the guesses is not of type AmplitudeVector")

    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)

    if n_ep is None:
        n_ep = len(guesses)
    elif n_ep > len(guesses):
        raise ValueError(f"n_ep (= {n_ep}) cannot exceed the number of guess "
                         f"vectors (= {len(guesses)}).")

    if n_block is None:
        n_block = n_ep
    elif n_block < n_ep:
        raise ValueError(f"n_block (= {n_block}) cannot be smaller than the number "
                         f"of states requested (= {n_ep}).")
    elif n_block > len(guesses):
        raise ValueError(f"n_block (= {n_block}) cannot exceed the number of guess "
                         f"vectors (= {len(guesses)}).")

    if not max_subspace:
        # TODO Arnoldi uses this:
        # max_subspace = max(2 * n_ep + 1, 20)
        max_subspace = max(6 * n_ep, 20, 5 * len(guesses))

    def convergence_test(state):
        state.residuals_converged = state.residual_norms < conv_tol
        state.converged = np.all(state.residuals_converged)
        return state.converged

    def convergence_micro(state):  # really rough
        state.converged = np.abs(state.history_rval[0]
                                 - state.history_rval[1]) > state.energy_diff
        return state.converged

    def residualNorm_folded(state, diis_omegaUpdate=False):
        state.eigenvector /= np.sqrt(state.eigenvector @ state.eigenvector)
        Av = folded_matrix @ state.eigenvector
        state.n_applies += 1
        # residual: r_i = A*v_i - w_i*v_i
        state.residual = lincomb([1, -folded_matrix.omega],
                                 [Av, state.eigenvector], evaluate=True)
        state.residual_norm = np.sqrt(state.residual @ state.residual)
        # For DIIS, update the eigenvalue corresponding to the new eigenvector.
        if diis_omegaUpdate:
            state.eigenvalues[state.n_state] = Av @ state.eigenvector
            folded_matrix.update_omega(state.eigenvalues[state.n_state])
        return state.residual_norm

    if conv_tol < matrix.shape[1] * np.finfo(float).eps:
        warnings.warn(la.LinAlgWarning(
            "Convergence tolerance (== {:5.2g}) lower than "
            "estimated maximal numerical accuracy (== {:5.2g}). "
            "Convergence might be hard to achieve."
            "".format(conv_tol, matrix.shape[1] * np.finfo(float).eps)
        ))

    state = DavidsonState(matrix, guesses)
    state.eigenvalues = np.empty(n_ep)
    state.eigenvectors = []
    state.residual_norms = np.empty(n_ep)

    folded_matrix = AdcMatrixFolded(matrix)
    if preconditioner is not None and isinstance(preconditioner, type):
        preconditioner = preconditioner(matrix)
        preconditioner.diagonal = folded_matrix.diagonal()

    # Retain single part of guess vectors
    guesses_i = [AmplitudeVector(ph=guess.__getitem__("ph")) for guess in guesses]
    
    if omegas is None:
        # Calculate the initial (guess) eigenvalue for state 0.
        Avi = matrix.block_apply("ph_ph", guesses_i[0].ph)
        state.eigenvalues[0] = Avi.dot(guesses_i[0].ph)

    state.timer.restart("folded iterations")
    for n_state in range(n_ep):
        # Initialize guess omega for excited states.
        if omegas is None:
            folded_matrix.update_omega(state.eigenvalues[n_state])
        else:
            folded_matrix.update_omega(omegas[n_state])

        state_i = FoldedDavidsonState(folded_matrix, guesses_i)
        state_i.n_state = n_state
        callback(state_i, "folded_start")

        # Macro iterations for state i
        state_i.timer.restart("folded iterations")
        while state_i.macro_iter < macro_max_iter:
            state_i.macro_iter += 1
            # Micro davidson iteration for diagonalising A(w_i)
            state_i = davidson_iterations(
                folded_matrix,
                state_i,
                max_subspace,
                max_iter,
                n_ep=n_ep,
                n_block=n_block,
                is_converged=convergence_micro,
                callback=callback,
                which=which,
                preconditioner=preconditioner,
                preconditioning_method=preconditioning_method,
                debug_checks=debug_checks,
                residual_min_norm=residual_min_norm,
                explicit_symmetrisation=explicit_symmetrisation)

            state.n_iter += state_i.n_iter
            # Update omega and calculate the residual_norm
            # under the latest omega for state i.
            folded_matrix.update_omega(state_i.eigenvalues[state_i.n_state])
            residualNorm_folded(state_i)
            callback(state_i, "micro")
            if state_i.residual_norm < macro_conv_tol:
                state_i.converged_macro = True
                break
            if state_i.macro_iter == macro_max_iter:
                warnings.warn(la.LinAlgWarning(
                    f"Maximum number of macro iterations ({macro_max_iter}) "
                    "reached in modified davidson procedure."))

        callback(state_i, "macro_stop")
        # DIIS to further converge
        state_i.timer.restart("folded iterations")
        diis = DIIS(num_diis_vecs=num_diis_vecs, start_iter=4)
        if not state_i.converged_macro:
            warnings.warn(la.LinAlgWarning(
                "Macro iterations with Davidson diagonalization "
                "is not converged yet."))

        preconditioner.update_shifts(float(0))
        while diis.iter_idx < diis_max_iter:
            b_i = state_i.eigenvector + preconditioner @ state_i.residual
            # corrected vector: b_i = u_i + residual_i / D11
            state_i.eigenvector = diis.compute_new_vec(b_i, state_i.residual)
            residualNorm_folded(state_i, diis_omegaUpdate=True)
            callback(state_i, "DIIS_steps")
            if state_i.residual_norm < conv_tol:
                state_i.converged_diis = True
                break
            if diis.iter_idx == diis_max_iter:
                warnings.warn(la.LinAlgWarning(
                    f"Maximum number of iterations (== {diis_max_iter}) "
                    "reached in DIIS procedure."))
        state_i.DIIS_iter = diis.iter_idx
        state.DIIS_iter += state_i.DIIS_iter
        callback(state_i, "DIIS_stop")

        guesses_i = state_i.subspace_vectors.copy()
        # Orthonormalize and update guesses_i for the next state:
        # for state 0, taking initial guesses as guess vectors;
        # for the higher-excited states, taking all eigenvectors of
        # current A(w_i) as guess vectors.
        guess_vecs = guesses_i.copy()
        del guess_vecs[n_state]
        coefficient = np.hstack(([1], -(state_i.eigenvector @ guess_vecs)))
        newVec = lincomb(coefficient, [state_i.eigenvector]
                         + guess_vecs, evaluate=True)
        state_i.eigenvector = newVec / np.sqrt(newVec @ newVec)
        guesses_i[n_state] = state_i.eigenvector

        # Collect results into the "DavidsonState"
        state.n_applies += state_i.n_applies
        state.eigenvalues[n_state:] = state_i.eigenvalues[n_state:]
        state.residual_norms[n_state] = state_i.residual_norm
        state_i.eigenvector = folded_matrix.unfold(state_i.eigenvector)
        state_i.eigenvector /= np.sqrt(state_i.eigenvector @ state_i.eigenvector)
        state.eigenvectors.append(state_i.eigenvector)

    if convergence_test(state):
        callback(state, "sum_folded")
    state.timer.stop("folded iterations")
    return state


def jacobi_davidson(*args, **kwargs):
    return eigsh(*args, preconditioner=JacobiPreconditioner,
                 preconditioning_method="Davidson", **kwargs)


def davidson(*args, **kwargs):
    return eigsh(*args, preconditioner=None, **kwargs)


def davidson_folded_DIIS(*args, **kwargs):
    return eigsh_folded(*args, preconditioner=JacobiPreconditioner,
                        preconditioning_method="Davidson", **kwargs)


class DIIS:
    """
    An implementation of DIIS acceleration, adapted from
    https://github.com/edeprince3/pdaggerq/blob/master/examples/full_cc_codes/diis.py
    """
    def __init__(self, num_diis_vecs: int, start_iter=4):
        """
        Initialize DIIS updater

        :params num_diis_vecs: Integer number representing number of DIIS
                               vectors to keep
        :param start_iter: optional (default=4) number to start DIIS iterations
        """
        self.nvecs = num_diis_vecs
        self.error_vecs = []
        self.prev_vecs = []
        self.start_iter = start_iter
        self.iter_idx = 0

    def compute_new_vec(self, iterate, error):
        """
        Compute a DIIS update.  Only perform diis update after start_iter
        have been accumulated.
        """
        # don't start DIIS until start_iter
        if self.iter_idx < self.start_iter:
            self.iter_idx += 1
            return iterate

        # add iterate and error to the list of error and iterates
        self.prev_vecs.append(iterate)
        self.error_vecs.append(error)
        self.iter_idx += 1

        # if prev_vecs is larger than the diis space size, then pop the oldest
        if len(self.prev_vecs) > self.nvecs:
            self.prev_vecs.pop(0)
            self.error_vecs.pop(0)

        # construct bmat and solve ax=b diis problem
        b_mat, rhs = self.get_bmat()
        c = np.linalg.solve(b_mat, rhs)
        c = c.flatten()

        # construct new iterate from solution to diis ax=b and previous vecs.
        new_iterate = self.prev_vecs[0].zeros_like()
        for ii in range(len(self.prev_vecs)):
            new_iterate += c[ii] * self.prev_vecs[ii]
        return new_iterate

    def get_bmat(self):
        """
        Compute b-mat
        """
        dim = len(self.prev_vecs)
        b = np.zeros((dim, dim))
        for i, j in product(range(dim), repeat=2):
            if i <= j:
                b[i, j] = self.error_vecs[i].dot(self.error_vecs[j])
                b[j, i] = b[i, j]
        b = np.hstack((b, -1 * np.ones((dim, 1))))
        b = np.vstack((b, -1 * np.ones((1, dim + 1))))
        b[-1, -1] = 0
        rhs = np.zeros((dim + 1, 1))
        rhs[-1, 0] = -1
        return b, rhs
