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
from collections.abc import Collection
from operator import index
from typing import SupportsIndex

import libadcc

from .backends import import_scf_results
from .memory_pool import memory_pool

__all__ = ["MoSpaces"]


# explicit selection of orbitals by index
Indices = Collection[SupportsIndex]
# Orbitals for one spin: a count, explicit indices or nothing
SpaceSelection = SupportsIndex | Indices | None
# Same selection for both spins, or differing (alpha, beta) index lists.
# No pair of counts: libadcc rejects unequal number of alpha/beta orbitals in a space,
# so such a pair is either invalid or equal to a single count. It would also
# make the input classification of type tuple[int, int] ambiguous:
# It can be either a single explicit index selection or a pair of counts?
SpaceSelectionArg = SpaceSelection | tuple[Indices, Indices]


def _as_pair(arg: SpaceSelectionArg) -> tuple[SpaceSelection, SpaceSelection]:
    # only a tuple of Indices is not expanded
    match arg:
        case tuple((Collection() as alpha, Collection() as beta)):
            return alpha, beta
        case _:
            return arg, arg


def _expand_to_list(
    space: str, entry: SpaceSelection, start: int | None = None, stop: int | None = None
) -> list[int]:
    """
    Convert the space selection ``entry`` to a list of orbital indices.
    If ``entry`` is a count, ``start`` or ``stop`` have to be provided to
    define the bounds of the range which is constructed.
    If ``start`` is provided [start, ..., count) is constructed.
    If ``stop`` is provided [stop - count, ..., stop) is constructed.
    """
    if entry is None:
        return []
    elif isinstance(entry, Collection):
        try:
            indices = [index(i) for i in entry]
            if any(i < 0 for i in indices):
                raise ValueError(
                    f"Orbital indices for space {space} must be non-negative, got {entry}."
                )
            return indices
        except TypeError:
            raise TypeError(
                f"Invalid orbital indices {entry} passed to argument {space}."
            ) from None
    elif isinstance(entry, SupportsIndex):
        if (start is None) == (stop is None):
            raise ValueError(f"Exactly one of start or stop has to be provided for space {space}.")
        n = index(entry)
        if n < 0:
            raise ValueError(f"Number of orbitals for space {space} must be non-negative, got {n}.")
        if start is not None:
            return list(range(start, start + n))
        else:
            assert stop is not None
            return list(range(stop - n, stop))
    raise TypeError(f"Unsupported type {(type(entry))} passed to argument {space}.")


def expand_spaceargs(
    hfdata_or_n_orbs: tuple[int, int] | libadcc.HartreeFockSolution_i,
    *,
    core_orbitals: SpaceSelectionArg = None,
    frozen_core: SpaceSelectionArg = None,
    frozen_virtual: SpaceSelectionArg = None,
) -> dict[str, list[int]]:
    if isinstance(hfdata_or_n_orbs, libadcc.HartreeFockSolution_i):
        noa = hfdata_or_n_orbs.n_orbs_alpha
        nob = hfdata_or_n_orbs.n_orbs_beta
    else:
        noa, nob = hfdata_or_n_orbs

    if frozen_core is True:
        # Determine number of frozen core electrons automatically
        # TODO The idea is to look at the energy gap in the HF orbital
        #      energies and exclude the ones, which are very far from the
        #      HOMO-LUMO gap.
        raise NotImplementedError(
            "Automatic determination of frozen-core electrons not implemented."
        )
        #
        # TODO One could also adopt the idea in the paper by Andreas and Chong
        #      how to automatically select the frozen_virtual orbitals in a
        #      clever way.
        #

    # convert the input to alpha, beta pairs
    pairs: dict[str, tuple[SpaceSelection, SpaceSelection]] = {
        "core_orbitals": _as_pair(core_orbitals),
        "frozen_core": _as_pair(frozen_core),
        "frozen_virtual": _as_pair(frozen_virtual),
    }
    # ensure that if we have an explicit selection for 1 space, we also have to have
    # explicit selections for the other spaces.
    entries = tuple(entry for pair in pairs.values() for entry in pair if entry is not None)
    if 0 < sum(isinstance(entry, Collection) for entry in entries) < len(entries):
        raise ValueError(
            "If one of the values of core_orbitals, frozen_core and frozen_virtual is an "
            "iterable, all must be, got "
            f"{core_orbitals = }, {frozen_core = } and {frozen_virtual = }."
        )

    spaces: dict[str, list[int]] = {}
    n_alpha, n_beta = 0, 0  # occupied orbitals already assigned per spin
    for key in ["frozen_core", "core_orbitals"]:  # the order is crucial!
        alpha = _expand_to_list(space=key, entry=pairs[key][0], start=n_alpha)
        beta = _expand_to_list(space=key, entry=pairs[key][1], start=n_beta)
        spaces[key] = alpha + [noa + i for i in beta]
        n_alpha += len(alpha)
        n_beta += len(beta)

    key = "frozen_virtual"
    alpha = _expand_to_list(space=key, entry=pairs[key][0], stop=noa)
    beta = _expand_to_list(space=key, entry=pairs[key][1], stop=nob)
    spaces[key] = alpha + [noa + i for i in beta]
    return spaces


class MoSpaces(libadcc.MoSpaces):
    def __init__(
        self,
        hfdata: object,
        core_orbitals: SpaceSelectionArg = None,
        frozen_core: SpaceSelectionArg = None,
        frozen_virtual: SpaceSelectionArg = None,
    ) -> None:
        """
        Construct an MoSpaces object, which holds information for
        translating between the adcc convention of arranging molecular
        orbitals and the convention of the host program.

        The documentation of each field here is only brief. Details
        can be found in :py:meth:`adcc.ReferenceState.__init__`.

        Parameters
        ----------
        hfdata
            Host-program object with Hartree-Fock data.

        core_orbitals : SpaceSelectionArg, optional
            The orbitals to be put into the core-occupied space.

        frozen_core : SpaceSelectionArg, optional
            The orbitals to be put into the frozen core space.

        frozen_virtual : SpaceSelectionArg, optional
            The orbitals to be put into the frozen virtual space.
        """
        hfdata = import_scf_results(hfdata)

        spaceargs = expand_spaceargs(
            hfdata,
            frozen_core=frozen_core,
            frozen_virtual=frozen_virtual,
            core_orbitals=core_orbitals,
        )
        super().__init__(
            hfdata,
            memory_pool,
            spaceargs["core_orbitals"],
            spaceargs["frozen_core"],
            spaceargs["frozen_virtual"],
        )

    @property
    def frozen_core(self) -> list[int]:
        """
        The frozen (inactive) occupied spin orbitals (in the index convention of the host provider).
        """
        return self.map_index_hf_provider.get("o3", [])

    @property
    def core_orbitals(self) -> list[int]:
        """
        The spin orbitals selected to reside in the core space
        (in the index convention of the host provider).
        """
        return self.map_index_hf_provider.get("o2", [])

    @property
    def occupied_orbitals(self) -> list[int]:
        """
        The active valence-occupied spin orbitals (in the index convention of the host provider).
        """
        return self.map_index_hf_provider["o1"]

    @property
    def virtual_orbitals(self) -> list[int]:
        """
        The active virtual spin orbitals (in the index convention of the host provider).
        """
        return self.map_index_hf_provider["v1"]

    @property
    def frozen_virtual(self) -> list[int]:
        """
        The frozen (inactive) virtual spin orbitals (in the index convention of the host provider).
        """
        return self.map_index_hf_provider.get("v2", [])


# TODO some nice describe method
#      (should return the core_orbitals, frozen_core, ...)
