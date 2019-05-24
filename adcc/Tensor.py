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
from .Symmetry import Symmetry

import libadcc

# TODO
# Along with it a constructor for the Tensor object, which takes
# an mospaces, a space string and the 4 properties of Symmetry,
# which completely allow to setup a tensor and its symmetry in one
# constructor call.


class Tensor(libadcc.Tensor):
    def __init__(self, sym_or_mo, space=None, irreps_allowed=None,
                 permutations=None, spin_block_maps=None,
                 spin_blocks_forbidden=None):
        """
        Construct an uninitialised tensor from an MoSpaces or a Symmetry object

        sym_or_mo    Symmetry or Mospaces object
        space        Space of the tensor, can be None if first argument is
                     a Symmetry object.
        irreps_allowed   List of allowed irreducible representations,
        permutations     List of permutational symmetries of the Tensor,
        spin_block_maps  List of mappings between spin blocks
        spin_blocks_forbidden   List of forbidden (i.e. forced-to-zero)
                                spin blocks.

        For the last four symmetry-related arguments see the documentation
        of the Symmetry object for details.
        """
        if not isinstance(sym_or_mo, (libadcc.MoSpaces, libadcc.Symmetry)):
            raise TypeError("The first argument needs to be a Symmetry or an "
                            "MoSpaces object.")
        if not isinstance(sym_or_mo, libadcc.Symmetry):
            if space is None:
                raise ValueError("If the first argument to Tensor is no "
                                 "Symmetry object, the second argument (spaces)"
                                 "needs to be given")
            sym_or_mo = Symmetry(sym_or_mo, space, irreps_allowed, permutations,
                                 spin_block_maps, spin_blocks_forbidden)

        if space is not None:
            if sym_or_mo.space != space:
                raise ValueError("Value passed to space needs to agree with "
                                 "space value from Symmetry object.")

        super().__init__(sym_or_mo)
