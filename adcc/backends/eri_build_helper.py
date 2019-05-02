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
EriPermutationSymm = namedtuple('EriPermutation', ['pref', 'transposition'])

_valid_notations = ["chem", "phys", "phys_asym"]

# ERIs in Chemists' notation
# (ij|kl) = (kl|ij) = (ji|lk) = (lk|ji)
# = (ji|kl) = (lk|ij) = (ij|lk) = (kl|ji)
ii, jj, kk, ll = 0, 1, 2, 3
_chem_allowed = [EriPermutationSymm(1, [ii, jj, kk, ll]),  # (ij|kl)
                 EriPermutationSymm(1, [kk, ll, ii, jj]),  # (kl|ij)
                 EriPermutationSymm(1, [jj, ii, ll, kk]),  # (ji|lk)
                 EriPermutationSymm(1, [ll, kk, jj, ii]),  # (lk|ji)
                 EriPermutationSymm(1, [jj, ii, kk, ll]),  # (ji|kl)
                 EriPermutationSymm(1, [ll, kk, ii, jj]),  # (lk|ij)
                 EriPermutationSymm(1, [ii, jj, ll, kk]),  # (ij|lk)
                 EriPermutationSymm(1, [kk, ll, jj, ii])]  # (kl|ji)

# Spin symmetry helper
# provide allowed spin block transposition
# together with the prefactors that are needed to form the
# antisymmetrized integral from Chemists' notation ERIs.
# TODO: further documentation
#   Example: Consider the <ab||ab> block in Physicists' notation
#   <ab||ab> = <ab|ab> - <ab|ba> = (aa|bb) - (ab|ba)
#   Here, the last term vanished, so this must be respected when
#   computing the final integral. The first prefactor (pref1) in
#   this case is 1, whereas the second prefactor (pref2) is 0 due to
#   vanishing block of the antisymmetrized integral.
EriPermutationSpinAntiSymm = namedtuple('EriPermutationSpinAntiSymm',
                                        ['pref1', 'pref2', 'transposition'])
_eri_phys_asymm_spin_allowed_prefactors = [
    EriPermutationSpinAntiSymm(1, 1, "aaaa"),
    EriPermutationSpinAntiSymm(1, 1, "bbbb"),
    EriPermutationSpinAntiSymm(1, 0, "abab"),
    EriPermutationSpinAntiSymm(1, 0, "baba"),
    EriPermutationSpinAntiSymm(0, 1, "baab"),
    EriPermutationSpinAntiSymm(0, 1, "abba"),
]


def is_spin_allowed(spin_block, notation="phys_asym"):
    if notation != "phys_asym":
        raise NotImplementedError("Only implemented for phys_asym")
    for symm in _eri_phys_asymm_spin_allowed_prefactors:
        if spin_block == symm.transposition:
            return True, symm
    return False, 0.0


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


class BlockSliceMappingHelper:
    """
    """
    def __init__(self, aro, bro,
                 arv, brv):
        self.block2slice = {
            "oa": aro,
            "ob": bro,
            "va": arv,
            "vb": brv,
        }

    def map_slices_to_blocks_and_spins(self, slices):
        requested_blocks = []
        requested_spins = []
        comp_block_slices = []
        for s in slices:
            slice_range = range(s.start, s.stop, 1)
            for block in self.block2slice:
                test_slice = self.block2slice[block]
                test_slice_range = range(test_slice.start,
                                         test_slice.stop, 1)
                if all(i in test_slice_range for i in slice_range):
                    requested_blocks.append(block[0].upper())
                    requested_spins.append(block[1])
                    start_offset = s.start - test_slice.start
                    comp_block_slice = slice(start_offset,
                                             s.stop - s.start + start_offset,
                                             1)
                    comp_block_slices.append(comp_block_slice)
        return requested_blocks, requested_spins, tuple(comp_block_slices)
