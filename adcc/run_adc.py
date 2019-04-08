#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import sys

from . import solver
from .guess import (guesses_any, guesses_singlet, guesses_spin_flip,
                    guesses_triplet)
from .backends import import_scf_results
from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod
from .caching_policy import DefaultCachingPolicy, GatherStatisticsPolicy
from .tmp_run_prelim import tmp_run_prelim

from .solver.davidson import jacobi_davidson
from .solver.explicit_symmetrisation import (IndexSpinSymmetrisation,
                                             IndexSymmetrisation)

from libadcc import HartreeFockSolution_i


def __call_jacobi_davidson(matrix, guesses, kind, output, **kwargs):
    """
    Internal function to kick off a Jacobi-Davidson sove from run_adc.
    """

    # Output callback for jacobi_davidson
    jd_callback = None
    if output:
        def jd_callback(state, identifier):
            solver.davidson.default_print(state, identifier, output)
        space = " " if kind else ""
        print("Starting " + matrix.method.name + " " + kind + space +
              "Jacobi-Davidson ...", file=output)
    return jacobi_davidson(matrix, guesses, callback=jd_callback, **kwargs)


def __call_guesses_functions(guessfctn, matrix, n_guess_singles,
                             n_guess_doubles):
    # later n_guesses, n_guesses_doubles
    singles_guesses = guessfctn(matrix, n_guess_singles, block="s")
    n_guess_doubles += n_guess_singles - len(singles_guesses)
    if n_guess_doubles <= 0:
        return singles_guesses
    doubles_guesses = guessfctn(matrix, n_guess_doubles, block="d")
    return singles_guesses + doubles_guesses


def run_adc(method, hfdata, n_singlets=None, n_triplets=None,
            n_states=None, solver_method=None,
            n_guess_singles=0, n_guess_doubles=0, output=sys.stdout,
            n_core_orbitals=None, caching_policy=DefaultCachingPolicy,
            **solverargs):
    """
    Run an ADC calculation on top of Hartree-Fock data.

    Required Parameters
    -------------------
    @param hfdata
    Data with the SCF reference to use. adcc is pretty flexible here.
    Can be e.g. a molsturm scf State, a pyscf SCF object, a class implementing
    the adcc.HartreeFockProvider interface or a pointer to any C++ object
    derived derived off adcc::HartreeFockSolution_i.

    @param n_singlets
    @param n_triplets
    @param n_states
    Specify the number and kind of states to compute. For unrestricted
    references clamping spin-pure singlets/triplets is currently not
    possible and n_states has to be used to specify the number of states.
    For restricted references n_states cannot be used at the moment.

    Optional parameters
    -------------------
    @param solver_method
    The eigensolver algorithm to use.

    @param n_guess_singles
    Number of singles block guesses. If this plus n_guess_doubles is less
    than then the number of states to be computed, then
    n_guess_singles = min(6,2 * number of excited states to compute)

    @param n_guess doubles
    Number of doubles block guesses. If this plus n_guess_singles is less
    than then the number of states to be computed, then
    n_guess_singles = number of excited states to compute

    @param output
    Python stream to which output will be written. If None all output
    is disabled.

    @param n_core_orbitals
    Number of (spatial) core orbitals. Required if apply_core-valence
    separation is applied. Notice that this number denotes spatial orbitals.
    Thus a value of 1 will put 1 alpha and 1 beta electron into the core region.

    @param caching_policy
    The policy to use for caching intermediate Tensors. Altering this value
    influences the balance between memory footprint and runtime.

    Solver keyword arguments
    ------------------------
    Other keyword arguments for the solver can be passed as well. An important
    selection of such arguments includes
       conv_tol       Convergence tolerance
       max_subspace   Maximal subspace size
       max_iter       Maximal numer of iterations
    """
    if not isinstance(hfdata, HartreeFockSolution_i):
        hfdata = import_scf_results(hfdata)

    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)

    if solver_method is None:
        solver_method = "jacobi_davidson"

    if method.is_core_valence_separated and n_core_orbitals is None:
        raise ValueError("If core-valence separation approximation is applied "
                         "then the number of core orbitals needs to be "
                         "specified via the parameter n_core_orbitals.")

    if solver_method != "jacobi_davidson":
        raise NotImplementedError("Only the jacobi_davidson solver_method "
                                  "is implemented.")

    # Parse state argument and bring the data into a common framework
    # of various lists, such that the remainder of the code works
    # for both restricted and unrestricted ground states.
    if not hfdata.restricted or hfdata.spin_multiplicity == 0:
        if n_singlets is not None:
            raise ValueError("The key \"n_singlets\" may only be used in "
                             "combination with an restricted ground state "
                             "reference of singlet spin to provide the number "
                             "of excited states to compute. Use \"n_states\" "
                             "for an UHF reference.")
        if n_triplets is not None:
            raise ValueError("The key \"n_triplets\" may only be used in "
                             "combination with an restricted ground state "
                             "reference of singlet spin to provide the number "
                             "of excited states to compute. Use \"n_states\" "
                             "for an UHF reference.")
        n_singlets = 0
        n_triplets = 0

        if n_states is None or n_states == 0:
            raise ValueError("No excited states to compute.")

        # Solve for a number of states of unspeciffied spin
        state_kinds = [""]
        n_kinds = [n_states]
        guess_keys = ["guesses_state"]  # Guess key for getting guesses
        #                                 from tmp_run_prelim
    else:
        if n_states is not None:
            raise ValueError("The key \"n_states\" may only be used in "
                             "combination with an unrestricted ground state "
                             "or a non-singlet ground state to provide the "
                             "number of excited states to compute. Use "
                             "\"n_singlets\" and \"n_triplets\".")
        n_states = 0

        state_kinds = []
        n_kinds = []
        guess_keys = []
        if n_singlets is None or n_singlets == 0:
            n_singlets = 0
        else:
            state_kinds.append("singlet")
            n_kinds.append(n_singlets)
            guess_keys.append("guesses_singlet")

        if n_triplets is None or n_triplets == 0:
            n_triplets = 0
        else:
            state_kinds.append("triplet")
            n_kinds.append(n_triplets)
            guess_keys.append("guesses_triplet")

        if n_singlets + n_triplets == 0:
            raise ValueError("No excited states to compute.")

    if n_guess_singles + n_guess_doubles == 0:
        # The maximum over all state parameters
        n_max_states = max(n_singlets, n_triplets, n_states)

        # Try to use at least 4 or twice the number of singlets or
        # triplets to be computed as guesses
        n_guess_singles = max(4, 2 * n_max_states)

        # Compute the maximal number of sensible singles block guesses.
        # This is roughly the number of occupied alpha orbitals
        # times the number of virtual alpha orbitals
        #
        # If the system is core valence separated, then only the
        # core electrons count as "occupied".
        n_virt_a = hfdata.n_orbs_alpha - hfdata.n_alpha
        if method.is_core_valence_separated:
            n_occ_a = n_core_orbitals
        else:
            n_occ_a = hfdata.n_alpha
        n_guess_singles_max = n_occ_a * n_virt_a

        # Adjust if we overshoot the maximal number of sensible
        # singles block guesses
        if n_guess_singles >= n_guess_singles_max:
            n_guess_singles = n_guess_singles_max

        # Make sure at least the number of requested states is also
        # requested as the number of guess vectors
        if n_guess_singles < n_max_states:
            n_guess_singles = n_max_states

    if method.base_method in ["adc0", "adc1"] and n_guess_doubles > 0:
        raise ValueError("n_guess_doubles > 0 is only sensible if the ADC "
                         "method is not adc0 or adc1 or a variant thereof.")

    # Do not copy caches if we want to make some statistics
    copy_caches = True
    if caching_policy == GatherStatisticsPolicy or \
       isinstance(caching_policy, GatherStatisticsPolicy):
        copy_caches = False

    # Obtain guesses and preliminary data
    prelim = tmp_run_prelim(hfdata, method,
                            n_guess_singles=n_guess_singles,
                            n_guess_doubles=n_guess_doubles,
                            n_core_orbitals=n_core_orbitals,
                            caching_policy=caching_policy,
                            copy_caches=copy_caches)

    # Setup ADC problem matrix
    matrix = AdcMatrix(method, prelim.ground_state)
    matrix.intermediates = prelim.intermediates

    if n_states > 0:
        prelim.guesses_state = __call_guesses_functions(guesses_any, matrix,
                                                        n_guess_singles,
                                                        n_guess_doubles)
    if n_singlets > 0:
        prelim.guesses_singlet = __call_guesses_functions(guesses_singlet,
                                                          matrix,
                                                          n_guess_singles,
                                                          n_guess_doubles)
    if n_triplets > 0:
        prelim.guesses_triplet = __call_guesses_functions(guesses_triplet,
                                                          matrix,
                                                          n_guess_singles,
                                                          n_guess_doubles)
    # TODO spin-flip

    # Solve for each spin kind:
    ret = []
    for kind, guess_key, n_kind in zip(state_kinds, guess_keys, n_kinds):
        # Get guesses
        guesses = getattr(prelim, guess_key)

        # Setup index and spin symmetrisation for obtaining spin-pure
        # singlet and triplet states in case of a restricted ground state
        explicit_symmetrisation = IndexSymmetrisation
        if hfdata.restricted:
            explicit_symmetrisation = IndexSpinSymmetrisation(
                matrix, enforce_spin_kind=kind
            )

        # Call solver
        res = __call_jacobi_davidson(
            matrix, guesses, kind, n_ep=n_kind, output=output,
            explicit_symmetrisation=explicit_symmetrisation, **solverargs
        )

        res.kind = kind
        ret.append(res)
    return ret
