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
import numpy as np
import textwrap
from itertools import product

from .backends import import_scf_results
from .memory_pool import memory_pool

from collections.abc import Iterable

import libadcc

__all__ = ["MoSpaces"]

__valid_spaces = list(map(''.join, product(["o", "v"], ["1", "2", "3"])))


def split_spaces(spacestr):
    """Utility function to split a string with
    multiple spaces into the individual space strings
    """
    sp = textwrap.wrap(spacestr, 2)
    try:
        assert all(s in __valid_spaces for s in sp)
        return sp
    except AssertionError:
        raise ValueError(f"Invalid space string '{spacestr}'.")


def expand_spaceargs(hfdata, **spaceargs):
    if isinstance(spaceargs.get("frozen_core", None), bool) \
       and spaceargs.get("frozen_core", None):
        # Determine number of frozen core electrons automatically
        # TODO The idea is to look at the energy gap in the HF orbital
        #      energies and exclude the ones, which are very far from the
        #      HOMO-LUMO gap.
        raise NotImplementedError("Automatic determination of frozen-core "
                                  "electrons not implemented.")
        #
        # TODO One could also adopt the idea in the paper by Andreas and Chong
        #      how to automatically select the frozen_virtual orbitals in a
        #      clever way.
        #

    def expand_to_list(space, entry, from_min=None, from_max=None):
        if from_min is None and from_max is None:
            raise ValueError("Both from_min and from_max is None")
        if entry is None:
            return np.array([])
        elif isinstance(entry, int):
            if from_min is not None:
                return from_min + np.arange(entry)
            return np.arange(from_max - entry, from_max)
        elif isinstance(entry, Iterable):
            return np.array(entry)
        else:
            raise TypeError("Unsupported type {} passed to argument {}"
                            "".format(type(entry), space))

    for key in spaceargs:
        if not isinstance(spaceargs[key], tuple):
            spaceargs[key] = (spaceargs[key], spaceargs[key])

    any_iterable = False
    for key in spaceargs:
        for spin in [0, 1]:
            if isinstance(spaceargs[key][spin], Iterable):
                any_iterable = True
            elif spaceargs[key][spin] is not None:
                if any_iterable:
                    raise ValueError("If one of the values of frozen_core, "
                                     "core_orbitals, frozen_virtual is an "
                                     "iterable, all must be.")

    noa = hfdata.n_orbs_alpha
    n_orbs = [0, 0]
    for key in ["frozen_core", "core_orbitals"]:
        if key not in spaceargs or not spaceargs[key]:
            continue
        list_alpha = expand_to_list(key, spaceargs[key][0],
                                    from_min=n_orbs[0])
        list_beta = noa + expand_to_list(key, spaceargs[key][1],
                                         from_min=n_orbs[1])
        spaceargs[key] = np.concatenate((list_alpha, list_beta)).tolist()
        n_orbs[0] += len(list_alpha)
        n_orbs[1] += len(list_beta)

    key = "frozen_virtual"
    if key in spaceargs and spaceargs[key]:
        spaceargs[key] = np.concatenate((
            expand_to_list(key, spaceargs[key][0], from_max=noa),
            expand_to_list(key, spaceargs[key][1], from_max=noa) + noa
        )).tolist()
    return spaceargs


class MoSpaces(libadcc.MoSpaces):
    def __init__(self, hfdata, core_orbitals=None, frozen_core=None,
                 frozen_virtual=None):
        """Construct an MoSpaces object, which holds information for
        translating between the adcc convention of arranging molecular
        orbitals and the convention of the host program.

        The documentation of each field here is only brief. Details
        can be found in :py:`adcc.ReferenceState.__init__`.

        Parameters
        ----------
        hfdata
            Host-program object with Hartree-Fock data.

        core_orbitals : int or list or tuple, optional
            The orbitals to be put into the core-occupied space.

        frozen_core : int or list or tuple, optional
            The orbitals to be put into the frozen core space.

        frozen_virtuals : int or list or tuple, optional
            The orbitals to be put into the frozen virtual space.
        """
        if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
            hfdata = import_scf_results(hfdata)

        spaceargs = expand_spaceargs(hfdata, frozen_core=frozen_core,
                                     frozen_virtual=frozen_virtual,
                                     core_orbitals=core_orbitals)
        super().__init__(hfdata, memory_pool, spaceargs["core_orbitals"],
                         spaceargs["frozen_core"], spaceargs["frozen_virtual"])

    @property
    def frozen_core(self):
        """
        The frozen (inactive) occupied spin orbitals
        (in the index convention of the host provider).
        """
        return self.map_index_hf_provider.get("o3", [])

    @property
    def core_orbitals(self):
        """
        The spin orbitals selected to reside in the core space
        (in the index convention of the host provider).
        """
        return self.map_index_hf_provider.get("o2", [])

    @property
    def occupied_orbitals(self):
        """
        The active valence-occupied spin orbitals
        (in the index convention of the host provider).
        """
        return self.map_index_hf_provider["o1"]

    @property
    def virtual_orbitals(self):
        """
        The active virtual spin orbitals
        (in the index convention of the host provider).
        """
        return self.map_index_hf_provider["v1"]

    @property
    def frozen_virtual(self):
        """
        The frozen (inactive) virtual spin orbitals
        (in the index convention of the host provider).
        """
        return self.map_index_hf_provider.get("v2", [])


# TODO some nice describe method
#      (should return the core_orbitals, frozen_core, ...)
