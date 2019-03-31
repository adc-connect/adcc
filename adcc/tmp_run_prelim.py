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
import warnings

import libadcc

from .backends import import_scf_results
from .AdcMethod import AdcMethod
from .memory_pool import StdAllocatorWarning, memory_pool
from .caching_policy import DefaultCachingPolicy
from .AmplitudeVector import AmplitudeVector


class TmpRunPrelimResult:
    def __init__(self, method, cpp_return, caching_policy):
        self.method = method
        self.reference = cpp_return.reference
        self.ground_state = cpp_return.mp
        self.intermediates = cpp_return.intermediates
        self.ctx = cpp_return.ctx

        self.caching_policy = caching_policy
        self.reference.caching_policy = caching_policy
        self.ground_state.caching_policy = caching_policy

        if cpp_return.have_singlet_and_triplet:
            self.guesses_singlet = [
                AmplitudeVector(*guess)
                for guess in cpp_return.guesses_singlet
            ]
            self.guesses_triplet = [
                AmplitudeVector(*guess)
                for guess in cpp_return.guesses_triplet
            ]
            assert len(self.guesses_singlet) > 0
            assert len(self.guesses_triplet) > 0
        else:
            self.guesses_state = [
                AmplitudeVector(*guess)
                for guess in cpp_return.guesses_state
            ]


def tmp_run_prelim(hfdata, adcmethod, n_guess_singles=0,
                   n_guess_doubles=0, print_level=0,
                   n_core_orbitals=None,
                   caching_policy=DefaultCachingPolicy,
                   copy_caches=True):
    """
    Temporary function to generate all the required data for an actional
    ADC calculation (MP2, guesses, intermediates).

    hfdata           Python object representing the C++ class implementing
                     the HartreeFockSolution_i interface
    adcmethod        string or adcc.AdcMethod object
    n_guess_singles  Number of guesses obtained looking at the singles part
    n_guess_doubles  Number of guesses obtained looking at the doubles part
    print_level      The adcman print level for this step, no real reason
                     to go beyond 0 (quiet) except for debugging
    n_core_orbitals  Number of (spatial) core orbitals. Required if
                     apply_core_valence_separation = True.
                     Notice that this number denotes spatial orbitals.
                     Thus a value of 1
                     will put 1 alpha and 1 beta electron into the core region.
    copy_caches      Should caches be copied from the preliminary ADC run (True)
                     or recomputed (False)
    """
    if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
        hfdata = import_scf_results(hfdata)

    if not isinstance(adcmethod, AdcMethod):
        adcmethod = AdcMethod(adcmethod)

    if isinstance(caching_policy, type):
        caching_policy = caching_policy()
    if not isinstance(caching_policy, libadcc.CachingPolicy_i):
        raise TypeError("caching_policy needs to be a CachingPolicy_i")

    if n_core_orbitals is None:
        n_core_orbitals = 0

    if adcmethod.is_core_valence_separated and n_core_orbitals == 0:
        raise ValueError("Core-valence separation requires n_core_orbitals > 0")

    if memory_pool.use_std_allocator:
        warnings.warn(StdAllocatorWarning(
            "Standard allocator will be used for "
            "computations. Results might be off, "
            "since this is not well-tested. "
            "Try initialising the memory_pool using "
            "adcc.memory_pool.initialise before "
            "doing any computations."
        ))

    cpp_return = libadcc.tmp_run_prelim(hfdata, adcmethod.name, memory_pool,
                                        print_level, n_guess_singles,
                                        n_guess_doubles, n_core_orbitals,
                                        caching_policy, copy_caches)
    return TmpRunPrelimResult(adcmethod, cpp_return, caching_policy)
