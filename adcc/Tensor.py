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
from .Symmetry import Symmetry

import libadcc


class Tensor(libadcc.Tensor):
    def __init__(self, sym_or_mo, space=None,
                 permutations=None, spin_block_maps=None,
                 spin_blocks_forbidden=None):
        """Construct an uninitialised Tensor from an :class:`MoSpaces` or
        a :class:`Symmetry` object.

        More information about the last four, symmetry-related parameters
        see the documentation of the :class:`Symmetry` object.

        Parameters
        ----------
        sym_or_mo
            Symmetry or MoSpaces object
        spaces : str, optional
            Space of the tensor, can be None if the first argument is
            a :class:`Symmetry` object.
        permutations : list, optional
            List of permutational symmetries of the Tensor.
        spin_block_maps : list, optional
            List of mappings between spin blocks
        spin_blocks_forbidden : list, optional
            List of forbidden (i.e. forced-to-zero) spin blocks.

        Notes
        -----
        An :class:`MoSpaces` object is contained in many datastructures
        of adcc, including the :class:`AdcMatrix`, the :class:`LazyMp`,
        the :class:`ReferenceState` and any solver or ADC results state.

        Examples
        --------
        Construct a symmetric tensor in the "o1o1" (occupied-occupied) spaces:

        >>> Tensor(mospaces, "o1o1", permutations=["ij", "ji"])

        Construct an anti-symmetric tensor in the "v1v1" spaces:

        >>> Tensor(mospaces, "v1v1", permutations=["ab", "-ba"])

        Construct a tensor in "o1v1", which maps the alpha-alpha block
        anti-symmetrically to the beta-beta block and which has the
        other spin blocks set to zero:

        >>> Tensor(mospaces, "o1v1", spin_block_maps=[("aa", "bb", -1)],
        ...        spin_blocks_forbidden=["ab", "ba"])

        """
        if not isinstance(sym_or_mo, (libadcc.MoSpaces, libadcc.Symmetry)):
            raise TypeError("The first argument needs to be a Symmetry or an "
                            "MoSpaces object.")
        if not isinstance(sym_or_mo, libadcc.Symmetry):
            if space is None:
                raise ValueError("If the first argument to Tensor is no "
                                 "Symmetry object, the second argument (spaces)"
                                 "needs to be given")
            sym_or_mo = Symmetry(sym_or_mo, space, permutations,
                                 spin_block_maps, spin_blocks_forbidden)

        if space is not None:
            if sym_or_mo.space != space:
                raise ValueError("Value passed to space needs to agree with "
                                 "space value from Symmetry object.")

        super().__init__(sym_or_mo)


def _tensor_select_below_absmax(tensor, tolerance):
    """
    Select the absolute maximal values in the tensor,
    which are below the given tolerance.
    """
    n = min(10, tensor.size)
    res = []
    while n <= tensor.size:
        res = tensor.select_n_absmax(n)
        minampl = min(abs(r[1]) for r in res)
        if minampl < tolerance:
            break
        else:
            n = max(n + 1, min(tensor.size, 2 * n))
    return [r for r in res if abs(r[1]) >= tolerance]


Tensor.select_below_absmax = _tensor_select_below_absmax
