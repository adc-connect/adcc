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

from .MoSpaces import auto_core, auto_frozen_core, auto_frozen_virtual
from .backends import import_scf_results
from .memory_pool import memory_pool

__all__ = ["ReferenceState"]


class ReferenceState(libadcc.ReferenceState):
    def __init__(self, hfdata, core_orbitals=None, frozen_core=None,
                 frozen_virtual=None, symmetry_check_on_import=False,
                 import_all_below_n_orbs=0):
        """
        Construct a ReferenceState object. The object is lazy and will only
        import orbital energies and coefficients. Fock matrix blocks and
        electron-repulsion integral blocks are imported as needed.

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

        @param symmetry_check_on_import
        Should symmetry of the imported objects be checked explicitly during the
        import process. This massively slows down the import and has a dramatic
        impact on memory usage. Thus one should enable this only for debugging
        (e.g. for testing import routines from the host programs). Do not enable
        this unless you know what you are doing.

        @import_all_below_n_orbs
        For small problem sizes lazy make less sense, since the memory
        requirement for storing the ERI tensor is neglibile and thus the
        flexiblity gained by having the full tensor in memory is advantageous.
        Below the number of orbitals specified by this parameter, the class
        will thus automatically import all ERI tensor and fock matrix blocks.
        """
        if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
            hfdata = import_scf_results(hfdata)
            # TODO This is needed right now to keep the hfdata object alive.
            #      One can probarly drop this if the other hacky
            #      keepalive-workaround for pybind11 is implemented
            self._hfdata = hfdata

        if frozen_core is None:
            frozen_core = []
        elif isinstance(frozen_core, int):
            frozen_core = auto_frozen_core(hfdata, core_orbitals, frozen_core)

        if core_orbitals is None:
            core_orbitals = []
        elif isinstance(core_orbitals, int):
            core_orbitals = auto_core(hfdata, core_orbitals, frozen_core)

        if frozen_virtual is None:
            frozen_virtual = []
        elif isinstance(frozen_virtual, int):
            frozen_virtual = auto_frozen_virtual(hfdata, frozen_virtual)

        super().__init__(hfdata, memory_pool, core_orbitals, frozen_core,
                         frozen_virtual, symmetry_check_on_import)

        if import_all_below_n_orbs is not None and \
           hfdata.n_orbs < import_all_below_n_orbs:
            super().import_all()


# TODO some nice describe method
