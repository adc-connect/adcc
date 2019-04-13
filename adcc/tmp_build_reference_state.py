#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
from .memory_pool import StdAllocatorWarning, memory_pool
from .caching_policy import DefaultCachingPolicy


def tmp_build_reference_state(hfdata, n_core_orbitals=None,
                              caching_policy=DefaultCachingPolicy):
    """
    Temporary function to import HF data into the libtensor infrastructure
    and to return a ReferenceState object for further use inside adcc.

    hfdata           Python object representing the C++ class implementing
                     the HartreeFockSolution_i interface
    n_core_orbitals  Number of (spatial) core orbitals. Required if
                     apply_core_valence_separation = True.
                     Notice that this number denotes spatial orbitals.
                     Thus a value of 1
                     will put 1 alpha and 1 beta electron into the core region.
    caching_policy   Policy to use for caching LazyMp and ADC intermediates.
    """
    if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
        hfdata = import_scf_results(hfdata)

    if isinstance(caching_policy, type):
        caching_policy = caching_policy()
    if not isinstance(caching_policy, libadcc.CachingPolicy_i):
        raise TypeError("caching_policy needs to be a CachingPolicy_i")

    if n_core_orbitals is None:
        n_core_orbitals = 0

    if memory_pool.use_std_allocator:
        warnings.warn(StdAllocatorWarning(
            "Standard allocator will be used for "
            "computations. Results might be off, "
            "since this is not well-tested. "
            "Try initialising the memory_pool using "
            "adcc.memory_pool.initialise(<memory in bytes>) before "
            "doing any computations."
        ))

    reference = libadcc.tmp_build_reference_state(hfdata, memory_pool,
                                                  n_core_orbitals,
                                                  caching_policy)
    reference.caching_policy = caching_policy
    return reference
