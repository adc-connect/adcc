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


def auto_core(hfdata, core_orbitals, frozen_core):
    if not frozen_core:
        frozen_core = 0

    if isinstance(frozen_core, list):
        raise NotImplementedError

    ret = []
    for i in range(frozen_core, frozen_core + core_orbitals):
        ret.append(i)
        ret.append(i + hfdata.n_orbs_alpha)
    return sorted(ret)


def auto_frozen_core(hfdata, core_orbitals, frozen_core):
    if not core_orbitals:
        core_orbitals = 0

    if isinstance(core_orbitals, list):
        raise NotImplementedError

    ret = []
    for i in range(frozen_core):
        ret.append(i)
        ret.append(i + hfdata.n_orbs_alpha)
    return sorted(ret)


def auto_frozen_virtual(hfdata, frozen_virtual):
    ret = []
    for i in range(frozen_virtual):
        ret.append(hfdata.n_orbs_alpha - i - 1)
        ret.append(hfdata.n_orbs - i - 1)
    return sorted(ret)


class MoSpaces(libadcc.MoSpaces):
    def __init__(self, hfdata, core_orbitals=None, frozen_core=None,
                 frozen_virtual=None):
        if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
            hfdata = import_scf_results(hfdata)

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
                         frozen_virtual)

# TODO some nice describe method
