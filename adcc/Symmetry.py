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
import libadcc


class Symmetry(libadcc.Symmetry):
    def __init__(self, mospaces, space, permutations=None,
                 spin_block_maps=None, spin_blocks_forbidden=None):
        if not isinstance(mospaces, libadcc.MoSpaces):
            raise TypeError("mospaces needs to be an MoSpaces instance.")

        super().__init__(mospaces, space)
        if permutations is not None:
            self.permutations = permutations
        if spin_block_maps is not None:
            self.spin_block_maps = spin_block_maps
        if spin_blocks_forbidden is not None:
            self.spin_blocks_forbidden = spin_blocks_forbidden

    def _repr_pretty_(self, pp, cycle):
        pp.text(self.describe())
