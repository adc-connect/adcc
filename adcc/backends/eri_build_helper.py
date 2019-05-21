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


class EriBuilder:
    """
    Parent class for building ERIs with different backends
    Note: Currently, only RHF references are available

    Implementation of the following functions in a derived class
    is necessary:
        coeffs_occ_alpha: provide occupied alpha coefficients
            (return np.ndarray)
        coeffs_virt_alpha: provide virtual alpha coefficients
            (return np.ndarray)
        compute_mo_eri: compute a block of integrals (Chemists')
            using the provided coefficients and retrieve block from
            cache if possible

    Depending on the host program, other functions like
        build_eri_phys_asym_block also need to be re-implemented.

    """
    def __init__(self, n_orbs, n_orbs_alpha, n_alpha, n_beta):
        self.block_slice_mapping = None
        self.n_orbs = n_orbs
        self.n_orbs_alpha = n_orbs_alpha
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.prepare_block_slice_mapping()

        self.eri_cache = {}
        self.eri_asymm_cache = {}
        self.last_block = None

    def prepare_block_slice_mapping(self):
        aro = slice(0, self.n_alpha, 1)
        bro = slice(self.n_orbs_alpha, self.n_orbs_alpha + self.n_beta, 1)
        arv = slice(self.n_alpha, self.n_orbs_alpha, 1)
        brv = slice(self.n_orbs_alpha + self.n_beta, self.n_orbs, 1)
        self.block_slice_mapping = BlockSliceMappingHelper(aro, bro,
                                                           arv, brv)

    @property
    def coeffs_occ_alpha(self):
        raise NotImplementedError("Implement coeffs_occ_alpha")

    @property
    def coeffs_virt_alpha(self):
        raise NotImplementedError("Implement coeffs_virt_alpha")

    @property
    def coeffs_occ_beta(self):
        raise NotImplementedError("Implement coeffs_occ_beta")

    @property
    def coeffs_virt_beta(self):
        raise NotImplementedError("Implement coeffs_virt_beta")

    def compute_mo_eri(self, block, coeffs, use_cache=True):
        raise NotImplementedError("Implement compute_mo_eri")

    def compute_mo_eri_slice(self, coeffs):
        return self.compute_mo_eri(None, coeffs, use_cache=False)

    def compute_mo_asym_eri(self, asym_block, spin_block):
        raise NotImplementedError("Implement compute_mo_asym_eri")

    @property
    def has_mo_asym_eri(self):
        return False

    def fill_slice(self, slices, out):
        mo_spaces, \
            spin_block, \
            comp_block_slices \
            = self.block_slice_mapping.map_slices_to_blocks_and_spins(slices)
        if len(mo_spaces) != 4:
            raise RuntimeError("Could not assign MO spaces from slice"
                               " {},"
                               " found {} and spin {}".format(slices,
                                                              mo_spaces,
                                                              spin_block))
        if mo_spaces != self.last_block:
            self.last_block = mo_spaces
            self.flush_cache()
        # if a host program allows to compute anti-symmetrized
        # integrals directly, check if this feature was enabled and
        # subsequently use compute_mo_asym_eri
        if self.has_mo_asym_eri:
            mo_spaces = "".join(mo_spaces)
            spin_block = "".join(spin_block)
            print("<{}||{}>, {}".format(mo_spaces[:2], mo_spaces[2:],
                                        spin_block))
            out[:] = self.compute_mo_asym_eri(mo_spaces,
                                              spin_block)[comp_block_slices]
        else:
            mo_spaces_chem = "".join(
                np.take(np.array(mo_spaces), [0, 2, 1, 3])
            )
            spin_block_str = "".join(spin_block)
            print(mo_spaces, mo_spaces_chem, spin_block_str)
            allowed, spin_symm = is_spin_allowed(spin_block_str)
            if allowed:
                eri = self.build_eri_phys_asym_block(
                    can_block=mo_spaces_chem, spin_symm=spin_symm
                )
                out[:] = eri[comp_block_slices]
            else:
                out[:] = 0

    def build_eri_phys_asym_block(self, can_block=None, spin_symm=None):
        co = self.coeffs_occ_alpha
        cv = self.coeffs_virt_alpha
        block = can_block
        asym_block = "".join([block[i] for i in [0, 3, 2, 1]])
        both_blocks = "{}-{}-{}-{}".format(
            block, asym_block, str(spin_symm.pref1), str(spin_symm.pref2)
        )
        if both_blocks in self.eri_asymm_cache.keys():
            return self.eri_asymm_cache[both_blocks]

        # TODO: avoid caching of (VV|VV)
        # because we don't need it anymore -> <VV||VV> is cached anyways

        coeffs_transform = tuple(co if x == "O" else cv for x in block)
        if spin_symm.pref1 != 0 and spin_symm.pref2 != 0:
            can_block_integrals = self.compute_mo_eri(block, coeffs_transform)
            eri_phys = can_block_integrals.transpose(0, 2, 1, 3)
            # (ik|jl) - (il|jk)
            chem_asym = tuple(coeffs_transform[i] for i in [0, 3, 2, 1])
            asymm = self.compute_mo_eri(
                asym_block, chem_asym
            ).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
            eris = spin_symm.pref1 * eri_phys - spin_symm.pref2 * asymm
        elif spin_symm.pref1 != 0 and spin_symm.pref2 == 0:
            can_block_integrals = self.compute_mo_eri(block, coeffs_transform)
            eris = spin_symm.pref1 * can_block_integrals.transpose(0, 2, 1, 3)
        elif spin_symm.pref1 == 0 and spin_symm.pref2 != 0:
            chem_asym = tuple(coeffs_transform[i] for i in [0, 3, 2, 1])
            asymm = self.compute_mo_eri(
                asym_block, chem_asym
            ).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
            eris = - spin_symm.pref2 * asymm

        self.eri_asymm_cache[both_blocks] = eris
        self.print_cache_memory()
        return eris

    def print_cache_memory(self):
        def cache_memory_gb(dict):
            return sum(dict[f].nbytes * 1e-9 for f in dict)

        print("Cached ERI asymm: {:.2f} Gb".format(
              cache_memory_gb(self.eri_asymm_cache)
              ))
        print("Cached ERI chem: {:.2f} Gb".format(
              cache_memory_gb(self.eri_cache)
              ))

    def build_full_eri_ffff(self):
        n_orbs = self.n_orbs
        n_alpha = self.n_alpha
        n_beta = self.n_beta
        n_orbs_alpha = self.n_orbs_alpha

        aro = slice(0, n_alpha)
        bro = slice(n_orbs_alpha, n_orbs_alpha + n_beta)
        arv = slice(n_alpha, n_orbs_alpha)
        brv = slice(n_orbs_alpha + n_beta, n_orbs)
        eri = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))

        co = self.coeffs_occ_alpha
        cv = self.coeffs_virt_alpha
        blocks = ["OOVV", "OVOV", "OOOV", "OOOO", "OVVV", "VVVV"]
        # TODO: needs to be refactored to support UHF
        for b in blocks:
            slices_alpha = [aro if x == "O" else arv for x in b]
            slices_beta = [bro if x == "O" else brv for x in b]
            coeffs_transform = tuple(co if x == "O" else cv for x in b)
            # make canonical integral block
            can_block_integrals = self.compute_mo_eri(b, coeffs_transform)

            # automatically set ERI tensor's symmetry-equivalent blocks
            trans_sym_blocks = get_symmetry_equivalent_transpositions_for_block(b)

            # Slices for the spin-allowed blocks
            aaaa = SpinBlockSlice("aaaa", (slices_alpha[0], slices_alpha[1],
                                           slices_alpha[2], slices_alpha[3]))
            bbbb = SpinBlockSlice("bbbb", (slices_beta[0], slices_beta[1],
                                           slices_beta[2], slices_beta[3]))
            aabb = SpinBlockSlice("aabb", (slices_alpha[0], slices_alpha[1],
                                           slices_beta[2], slices_beta[3]))
            bbaa = SpinBlockSlice("bbaa", (slices_beta[0], slices_beta[1],
                                           slices_alpha[2], slices_alpha[3]))
            non_zero_spin_block_slice_list = [aaaa, bbbb, aabb, bbaa]
            for tsym_block in trans_sym_blocks:
                sym_block_eri = can_block_integrals.transpose(tsym_block)
                for non_zero_spin_block in non_zero_spin_block_slice_list:
                    transposed_spin_slices = tuple(non_zero_spin_block.slices[i] for i in tsym_block)
                    eri[transposed_spin_slices] = sym_block_eri
        return eri

    def flush_cache(self):
        self.eri_asymm_cache = {}
        self.eri_cache = {}


class BlockSliceMappingHelper:
    """
    BlockSliceMappingHelper can map the slices coming from libtensor
    to the respective orbital and spin space, such that
    EriBuilder knows which block needs to be computed and provided
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
