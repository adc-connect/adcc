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
import libadcc

from .backends import import_scf_results
from .memory_pool import memory_pool

__all__ = ["MoSpaces"]


class MoSpaces(libadcc.MoSpaces):
    def __init__(self, hfdata, core_orbitals=None, frozen_core=None,
                 frozen_virtual=None):
        """
        Construct an MoSpaces object.

        @param hfdata
        Object with Hartree-Fock data (e.g. a molsturm scf state, a pyscf SCF
        object or any class implementing the adcc.HartreeFockProvider interface
        or in fact any python object representing a pointer to a C++ object
        derived off the adcc::HartreeFockSolution_i.

        @param core_orbitals
        (a) The number of alpha and beta core orbitals to use. The first
        orbitals (in the original ordering of the hfdata object), which are not
        part of the frozen_core will be selected.
        (b) Explicit list of orbital indices (in the ordering of the hfdata
        object) to put into the core-occupied orbital space. The same number of
        alpha and beta orbitals have to be selected. These will be forcibly
        occupied.

        @param frozen_core
        (a) The number of alpha and beta frozen core orbitals to use. The first
        orbitals (in the original ordering of the hfdata object) will be
        selected.
        (b) Explicit list of orbital indices (in the ordering of the hfdata
        object) to put into the fropen core. The same number of alpha and beta
        orbitals have to be selected. These will be forcibly occupied.

        @param frozen_virtuals
        (a) The number of alpha and beta frozen virtual orbitals to use. The
        last orbitals will be selected.
        (b) Explicit list of orbital indices to put into the frozen virtual
        orbital subspace. The same number of alpha and beta orbitals have to be
        selected. These will be forcibly unoccupied.
        """
        if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
            hfdata = import_scf_results(hfdata)

        if not isinstance(frozen_core, (list, int)) and frozen_core is not None:
            raise TypeError("frozen_core should be an int or a list")
        if not isinstance(core_orbitals, (list, int)) \
           and core_orbitals is not None:
            raise TypeError("core_orbitals should be an int or a list")
        if not isinstance(frozen_virtual, (list, int)) \
           and frozen_virtual is not None:
            raise TypeError("frozen_virtual should be an int or a list")

        if any(isinstance(k, int) for k in [frozen_core, core_orbitals,
                                            frozen_virtual]):
            if frozen_core is None:
                frozen_core = 0
            if core_orbitals is None:
                core_orbitals = 0
            if frozen_virtual is None:
                frozen_virtual = 0
        else:
            if frozen_core is None:
                frozen_core = []
            if core_orbitals is None:
                core_orbitals = []
            if frozen_virtual is None:
                frozen_virtual = []
        super().__init__(hfdata, memory_pool, core_orbitals, frozen_core,
                         frozen_virtual)


# TODO some nice describe method
#      (should return the core_orbitals, frozen_core, ...)
