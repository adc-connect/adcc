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

from collections import namedtuple
import numpy as np

# Helper namedtuple for slices of spin blocks
SpinBlockSlice = namedtuple('SpinBlockSlice', ['spins', 'slices'])

# Helper class to contain a specific ERI permutation with the resp. sign
EriPermutation = namedtuple('EriPermutation', ['sign', 'transposition'])

_valid_notations = ["chem", "phys", "phys_asym"]

# ERIs in Chemists' notation
# (ij|kl) = (kl|ij) = (ji|lk) = (lk|ji)
# = (ji|kl) = (lk|ij) = (ij|lk) = (kl|ji)
_chem_allowed = [EriPermutation(1, [0, 1, 2, 3]),  # (ij|kl)
                 EriPermutation(1, [2, 3, 0, 1]),  # (kl|ij)
                 EriPermutation(1, [1, 0, 3, 2]),  # (ji|lk)
                 EriPermutation(1, [3, 2, 1, 0]),  # (lk|ji)
                 EriPermutation(1, [1, 0, 2, 3]),  # (ji|kl)
                 EriPermutation(1, [3, 2, 0, 1]),  # (lk|ij)
                 EriPermutation(1, [0, 1, 3, 2]),  # (ij|lk)
                 EriPermutation(1, [2, 3, 1, 0])]  # (kl|ji)


class EriBlock:
    """
    Helper class for ERI Blocks and their permutational symmetries
    """
    def __init__(self, block_name, notation="chem"):
        self.block_name = block_name
        self.notation = notation
        self.build_permutations()

    def build_permutations(self):
        bsp = np.array(tuple(self.block_name))
        allowed_permutations_strings = []
        allowed_transpositions = []
        # TODO: take care of sign when phys_asym is implemented
        for permutation in _chem_allowed:
            t = np.take(bsp, permutation.transposition)
            tstring = "".join(t.tolist())
            if tstring not in allowed_permutations_strings:
                allowed_permutations_strings.append(tstring)
                allowed_transpositions.append(tuple(permutation.transposition))
        return (allowed_permutations_strings, allowed_transpositions)


def get_symmetry_equivalent_transpositions_for_block(block, notation="chem"):
    """
    Obtains the symmetry-equivalent blocks of a canonical ERI tensor
    block in the given notation.

    Currently, only Chemists' notation is implemented
    """
    if not isinstance(block, str):
        raise ValueError("Please specify the block as string!")
    if notation not in _valid_notations:
        raise ValueError("""Invalid notation type {}.
                         Valid notations are:
                         {}""".format(notation, ",".join(_valid_notations)))

    if notation != "chem":
        raise NotImplementedError("Only Chemists' notation is implemented")
    blck = EriBlock(block, notation=notation)
    perms, trans = blck.build_permutations()
    return trans
