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

from collections import namedtuple

# Helper namedtuple for slices of spin blocks
SpinBlockSlice = namedtuple('SpinBlockSlice', ['spins', 'slices'])

# Helper class to contain a specific ERI permutation with the resp. sign
EriPermutationSymm = namedtuple('EriPermutation', ['pref', 'transposition'])

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

_phys_allowed = [EriPermutationSymm(1, [ii, jj, kk, ll]),  # <ij|kl>
                 EriPermutationSymm(1, [jj, ii, ll, kk]),  # <ji|lk>
                 EriPermutationSymm(1, [kk, ll, ii, jj]),  # <kl|ij>
                 EriPermutationSymm(1, [ll, kk, jj, ii]),  # <lk|ji>
                 EriPermutationSymm(1, [kk, jj, ii, ll]),  # <kj|il>
                 EriPermutationSymm(1, [ll, ii, jj, kk]),  # <li|jk>
                 EriPermutationSymm(1, [ii, ll, kk, jj]),  # <il|kj>
                 EriPermutationSymm(1, [jj, kk, ll, ii])]  # <jk|li>

eri_permutations = {
    "chem": _chem_allowed,
    "phys": _phys_allowed,
}

# Spin symmetry helper
# provide allowed spin block transposition
# together with the prefactors that are needed to form the
# antisymmetrized integral from Chemists' notation ERIs.
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


# TODO: Use Symmetry object feature in the future
def is_spin_allowed(spin_block, notation="phys_asym"):
    if notation != "phys_asym":
        raise NotImplementedError("Only implemented for phys_asym")
    for symm in _eri_phys_asymm_spin_allowed_prefactors:
        if spin_block == symm.transposition:
            return True, symm
    return False, None


class EriBlock:
    """
    Helper class for ERI Blocks and their permutational symmetries
    Note: This class is only used to build the FULL ERI tensor
            for test purposes
    """
    def __init__(self, block_name, notation="chem"):
        self.block_name = block_name
        if notation not in eri_permutations.keys():
            raise ValueError("Invalid notation {}.".format(notation))
        self.notation = notation

    def build_permutations(self):
        bsp = np.array(tuple(self.block_name))
        allowed_permutations_strings = []
        allowed_transpositions = []
        for permutation in eri_permutations[self.notation]:
            t = np.take(bsp, permutation.transposition)
            tstring = "".join(t.tolist())
            if tstring not in allowed_permutations_strings:
                allowed_permutations_strings.append(tstring)
                allowed_transpositions.append(tuple(permutation.transposition))
        return (allowed_permutations_strings, allowed_transpositions)


def get_symm_equivalent_transpositions_for_block(block, notation="chem"):
    """
    Obtains the symmetry-equivalent blocks of a canonical ERI tensor
    block in the given notation.
    """
    if not isinstance(block, str):
        raise ValueError("Please specify the block as string!")
    blck = EriBlock(block, notation=notation)
    perms, trans = blck.build_permutations()
    return trans


# TODO: This class could be simplified by
# a) Symmetry/MoIndexTranslation
# b) doing the anti-symmetrization in libtensor
class EriBuilder:
    """
    Parent class for building ERIs with different backends
    Note: Currently, only RHF references are available

    Implementation of the following functions in a derived class
    is necessary:
        - ``coefficients``: Return a dict from a key describing the block
          to the MO coefficients as an ``np.ndarray``. The expected keys
          are ``Oa`` (occupied-alpha), ``Ob`` (occupied-beta),
          ``Va`` (virtual-alpha), ``Vb`` (virtual-beta).
        - ``compute_mo_eri``: compute a block of integrals (Chemists')
          using the provided coefficients and retrieve block from cache
          if possible

    Depending on the host program, other functions like
    ``build_eri_phys_asym_block`` also need to be re-implemented.

    """
    def __init__(self, n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted):
        self.block_slice_mapping = None
        self.n_orbs = n_orbs
        self.n_orbs_alpha = n_orbs_alpha
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.restricted = restricted

        self.eri_cache = {}
        self.eri_asymm_cache = {}
        self.last_block = None

        aro = slice(0, self.n_alpha, 1)
        bro = slice(self.n_orbs_alpha, self.n_orbs_alpha + self.n_beta, 1)
        arv = slice(self.n_alpha, self.n_orbs_alpha, 1)
        brv = slice(self.n_orbs_alpha + self.n_beta, self.n_orbs, 1)
        self.block_slice_mapping = BlockSliceMappingHelper(aro, bro, arv, brv)

    @property
    def coefficients(self):
        raise NotImplementedError("Implement coefficients")

    def compute_mo_eri(self, block, coeffs, use_cache=True):
        raise NotImplementedError("Implement compute_mo_eri")

    def compute_mo_eri_slice(self, coeffs):
        return self.compute_mo_eri(None, coeffs, use_cache=False)

    @property
    def has_mo_asym_eri(self):
        return False

    @property
    def eri_notation(self):
        return "chem"

    def fill_slice(self, slices, out):
        """
        slices  requested slice of ERIs from libtensor
        out     view to libtensor memory
        """
        # map index slices to orbital and spin space
        # because we do not have access to the MoSpaces object inside the
        # HartreeFockProvider implementation
        mapping = self.block_slice_mapping.map_slices_to_blocks_and_spins(
            slices
        )

        # mo_spaces (antisymm. physicists' notation),
        # e.g. an index slice from <oo||oo> maps to ['O', 'O', 'O', 'O']
        #
        # spin_block is the equivalent for the spin block
        #
        # comp_block_slices gives you the slices inside the computed block
        # required. E.g., if you need a part of o2, the full occupied
        # integral block is computed in the host program, but only
        # parts of it are written to libtensor memory
        mo_spaces, spin_block, comp_block_slices = mapping
        if len(mo_spaces) != 4:
            raise RuntimeError(
                "Could not assign MO spaces from slice {},"
                " found {} and spin {}".format(slices, mo_spaces, spin_block)
            )
        if mo_spaces != self.last_block:
            self.last_block = mo_spaces
            self.flush_cache()
        # transform the incoming MO spaces to Chemists' notation
        # if the backend provides Chemists' ERIs
        # <ij|kl> -> (ik|jl)
        if self.eri_notation == "chem":
            mo_spaces = "".join(np.take(np.array(mo_spaces), [0, 2, 1, 3]))
        spin_block_str = "".join(spin_block)
        allowed, spin_symm = is_spin_allowed(spin_block_str)
        if allowed:
            eri = self.build_eri_phys_asym_block(can_block=mo_spaces,
                                                 spin_symm=spin_symm)
            out[:] = eri[comp_block_slices]
        else:
            out[:] = 0

    def build_eri_phys_asym_block(self, can_block=None, spin_symm=None):
        block = can_block
        asym_block = "".join([block[i] for i in [0, 3, 2, 1]])
        both_blocks = f"{block}-{asym_block}-{spin_symm.pref1}-{spin_symm.pref2}"
        if not self.restricted:
            # add the spin block to the caching string
            both_blocks += "-" + spin_symm.transposition
        if both_blocks in self.eri_asymm_cache.keys():
            return self.eri_asymm_cache[both_blocks]

        # TODO: avoid caching of (VV|VV)
        # because we don't need it anymore -> <VV||VV> is cached anyways

        spin_chem = list(spin_symm.transposition[i] for i in [0, 2, 1, 3])
        spin_key = "".join(spin_chem)
        coeffs_transform = tuple(self.coefficients[x + y]
                                 for x, y in zip(block, spin_chem))

        if not self.restricted:
            block += spin_key
            asym_block += "".join([spin_key[i] for i in [0, 2, 1, 3]])

        # For the given spin block, check which individual blocks
        # in Chemists' notation are actually needed to form the antisymmetrized
        # integral in Physicists' notation.
        # Some of them are zero and don't need to be computed.
        # Example:
        # <ov||ov> = <ov|ov> - <ov|vo> = (oo|vv) - (ov|vo)
        # the last term is zero in the case of (ab|ba)

        # both terms in the antisymm. are non-zero
        if spin_symm.pref1 != 0 and spin_symm.pref2 != 0:
            can_block_integrals = self.compute_mo_eri(block, coeffs_transform)
            eri_phys = can_block_integrals.transpose(0, 2, 1, 3)
            # (ik|jl) - (il|jk)
            chem_asym = tuple(coeffs_transform[i] for i in [0, 3, 2, 1])
            asymm = self.compute_mo_eri(
                asym_block, chem_asym
            ).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
            eris = spin_symm.pref1 * eri_phys - spin_symm.pref2 * asymm
        # only the second term is zero
        elif spin_symm.pref1 != 0 and spin_symm.pref2 == 0:
            can_block_integrals = self.compute_mo_eri(block, coeffs_transform)
            eris = spin_symm.pref1 * can_block_integrals.transpose(0, 2, 1, 3)
        # only the first term is zero
        elif spin_symm.pref1 == 0 and spin_symm.pref2 != 0:
            chem_asym = tuple(coeffs_transform[i] for i in [0, 3, 2, 1])
            asymm = self.compute_mo_eri(
                asym_block, chem_asym
            ).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
            eris = - spin_symm.pref2 * asymm

        self.eri_asymm_cache[both_blocks] = eris
        return eris

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

        blocks = ["OOVV", "OVOV", "OOOV", "OOOO", "OVVV", "VVVV"]
        for b in blocks:
            slices_alpha = [aro if x == "O" else arv for x in b]
            slices_beta = [bro if x == "O" else brv for x in b]

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
            trans_sym_blocks = get_symm_equivalent_transpositions_for_block(b)

            # automatically set ERI tensor's symmetry-equivalent blocks
            for spin_block in non_zero_spin_block_slice_list:
                coeffs_transform = tuple(self.coefficients[x + y]
                                         for x, y in zip(b, spin_block.spins))
                can_block_integrals = self.compute_mo_eri(
                    b + "".join(spin_block.spins), coeffs_transform
                )
                for tsym_block in trans_sym_blocks:
                    sym_block_eri = can_block_integrals.transpose(tsym_block)
                    transposed_spin_slices = tuple(spin_block.slices[i]
                                                   for i in tsym_block)
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
            "oa": aro, "ob": bro, "va": arv, "vb": brv,
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
                    comp_block_slice = slice(
                        start_offset, s.stop - s.start + start_offset, 1
                    )
                    comp_block_slices.append(comp_block_slice)
        return requested_blocks, requested_spins, tuple(comp_block_slices)
