#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from adcc import AmplitudeVector, Symmetry, Tensor

from ..AdcMatrix import AdcMatrixlike
from ..AdcMethod import AdcType


def guess_zero(matrix, spin_change=0, spin_block_symmetrisation="none"):
    """
    Return an AmplitudeVector object filled with zeros, but where the symmetry
    has been properly set up to meet the specified requirements on the guesses.

    matrix       The matrix for which guesses are to be constructed
    spin_change  The spin change to enforce in an excitation.
                 Typical values are 0 (singlet/triplet/any) and -1 (spin-flip).
    spin_block_symmetrisation
                 Symmetrisation to enforce between equivalent spin blocks,
                 which all yield the desired spin_change. E.g. if
                 spin_change == 0, then both the alpha->alpha and beta->beta
                 blocks of the singles part of the excitation vector achieve a
                 spin change of 0. The symmetry specified with this parameter
                 will then be imposed between the a-a and b-b blocks. Valid
                 values are "none", "symmetric" and "antisymmetric", where
                 "none" enforces no particular symmetry.
    """
    return AmplitudeVector(**{
        block: Tensor(sym) for block, sym in guess_symmetries(
            matrix, spin_change=spin_change,
            spin_block_symmetrisation=spin_block_symmetrisation
        ).items()
    })


def guess_symmetries(matrix, spin_change=0, spin_block_symmetrisation="none"):
    """
    Return guess symmetry objects (one for each AmplitudeVector block) such
    that the specified requirements on the guesses are satisfied.

    matrix       The matrix for which guesses are to be constructed
    spin_change  The spin change to enforce in an excitation.
                 Typical values are 0 (singlet/triplet/any) and -1 (spin-flip).
    spin_block_symmetrisation
                 Symmetrisation to enforce between equivalent spin blocks,
                 which all yield the desired spin_change. E.g. if
                 spin_change == 0, then both the alpha->alpha and beta->beta
                 blocks of the singles part of the excitation vector achieve a
                 spin change of 0. The symmetry specified with this parameter
                 will then be imposed between the a-a and b-b blocks. Valid
                 values are "none", "symmetric" and "antisymmetric", where
                 "none" enforces no particular symmetry.
    """
    if not isinstance(matrix, AdcMatrixlike):
        raise TypeError("matrix needs to be of type AdcMatrixlike")
    if spin_block_symmetrisation not in ["none", "symmetric", "antisymmetric"]:
        raise ValueError("Invalid value for spin_block_symmetrisation: "
                         "{}".format(spin_block_symmetrisation))
    if spin_block_symmetrisation != "none" and \
       not matrix.reference_state.restricted:
        raise ValueError("spin_block_symmetrisation != none is only valid for "
                         "ADC calculations on top of restricted reference "
                         "states.")
    if int(spin_change * 2) / 2 != spin_change:
        raise ValueError("Only integer or half-integer spin_change is allowed. "
                         "You passed {}".format(spin_change))

    max_spin_change = 0.5 * len(matrix.axis_blocks[-1])
    valid_spin_changes = [
        max_spin_change - i for i in range(int(2 * max_spin_change + 1))]

    if spin_change not in valid_spin_changes:
        raise ValueError("spin_change may only be one of "
                         f"{valid_spin_changes}, and not {spin_change}.")

    symmetries = {}
    singles = matrix.axis_blocks[0]
    symmetries[singles] = guess_symmetry_singles(
        matrix, spin_change=spin_change,
        spin_block_symmetrisation=spin_block_symmetrisation, block=singles
    )
    if len(matrix.axis_blocks) >= 2:
        doubles = matrix.axis_blocks[1]
        symmetries[doubles] = guess_symmetry_doubles(
            matrix, spin_change=spin_change,
            spin_block_symmetrisation=spin_block_symmetrisation, block=doubles
        )
    if "ppphhh" in matrix.axis_blocks:
        symmetries["ppphhh"] = guess_symmetry_triples(
            matrix, spin_change=spin_change,
            spin_block_symmetrisation=spin_block_symmetrisation
        )
    return symmetries


def guess_symmetry_singles(matrix, spin_change=0,
                           spin_block_symmetrisation="none", block="ph"):
    symmetry = Symmetry(matrix.mospaces, "".join(matrix.axis_spaces[block]))
    symmetry.irreps_allowed = ["A"]

    if spin_change != 0 and spin_block_symmetrisation != "none":
        raise NotImplementedError("spin_symmetrisation != 'none' only "
                                  "implemented for spin_change == 0")

    if spin_block_symmetrisation in ("symmetric", "antisymmetric"):
        # PP-ADC
        fac = 1 if spin_block_symmetrisation == "symmetric" else -1
        symmetry.spin_block_maps = [("aa", "bb", fac)]
        symmetry.spin_blocks_forbidden = ["ab", "ba"]

    elif (
        matrix.method.adc_type is not AdcType.PP
        and matrix.reference_state.restricted
    ):
        # IP- and EA-ADC
        # attach/detach alpha electron
        # forbidden beta blocks (["a"]) not needed because for a
        # restricted reference only alpha states will be computed
        symmetry.spin_blocks_forbidden = ["b"]

    return symmetry


def guess_symmetry_doubles(matrix, spin_change=0,
                           spin_block_symmetrisation="none", block="pphh"):
    spaces_d = matrix.axis_spaces[block]
    symmetry = Symmetry(matrix.mospaces, "".join(spaces_d))
    symmetry.irreps_allowed = ["A"]

    if spin_change != 0 and spin_block_symmetrisation != "none":
        raise NotImplementedError("spin_symmetrisation != 'none' only "
                                  "implemented for spin_change == 0")

    if matrix.method.adc_type is AdcType.PP:
        # PP-ADC
        if spin_block_symmetrisation in ("symmetric", "antisymmetric"):
            fac = 1 if spin_block_symmetrisation == "symmetric" else -1
            # Spin mapping between blocks where alpha and beta are mirrored
            symmetry.spin_block_maps = [("aaaa", "bbbb", fac),
                                        ("abab", "baba", fac),
                                        ("abba", "baab", fac)]

            # Mark blocks which change spin as forbidden
            symmetry.spin_blocks_forbidden = ["aabb",  # spin_change -2
                                              "bbaa",  # spin_change +2
                                              "aaab",  # spin_change -1
                                              "aaba",  # spin_change -1
                                              "abaa",  # spin_change +1
                                              "baaa",  # spin_change +1
                                              "abbb",  # spin_change -1
                                              "babb",  # spin_change -1
                                              "bbab",  # spin_change +1
                                              "bbba"]  # spin_change +1

        # Add index permutation symmetry:
        permutations = ["ijab"]
        if spaces_d[0] == spaces_d[1]:
            permutations.append("-jiab")
        if spaces_d[2] == spaces_d[3]:
            permutations.append("-ijba")
        if len(permutations) > 1:
            symmetry.permutations = permutations

    elif matrix.method.adc_type is AdcType.IP:
        # IP-ADC
        # No spin mapping between blocks
        symmetry.spin_block_maps = []

        # Mark blocks which change spin incorrectly as forbidden
        if matrix.reference_state.restricted:  # kind=doublet
            # detach alpha electron
            # forbidden beta ionization blocks not needed because for a
            # restricted reference only alpha states will be computed
            symmetry.spin_blocks_forbidden = ["aab",  # spin_change -3/2
                                              "bba",  # spin_change +3/2
                                              "aba",  # spin_change +1/2
                                              "baa",  # spin_change +1/2
                                              "bbb"]  # spin_change +1/2

        # Add index permutation symmetry:
        permutations = ["ija"]
        if spaces_d[0] == spaces_d[1]:
            permutations.append("-jia")
        if len(permutations) > 1:
            symmetry.permutations = permutations

    elif matrix.method.adc_type is AdcType.EA:
        # EA-ADC
        # No spin mapping between blocks
        symmetry.spin_block_maps = []

        # Mark blocks which change spin incorrectly as forbidden
        if matrix.reference_state.restricted:  # kind=doublet
            # attach alpha electron
            # forbidden beta attachment blocks not needed because for a
            # restricted reference only alpha states will be computed
            symmetry.spin_blocks_forbidden = ["baa",  # spin_change +3/2
                                              "abb",  # spin_change -3/2
                                              "aab",  # spin_change -1/2
                                              "aba",  # spin_change -1/2
                                              "bbb"]  # spin_change -1/2

        # Add index permutation symmetry:
        permutations = ["iab"]
        if spaces_d[1] == spaces_d[2]:
            permutations.append("-iba")
        if len(permutations) > 1:
            symmetry.permutations = permutations

    return symmetry


def guess_symmetry_triples(matrix, spin_change=0,
                           spin_block_symmetrisation="none"):
    spaces_t = matrix.axis_spaces["ppphhh"]
    symmetry = Symmetry(matrix.mospaces, "".join(spaces_t))
    symmetry.irreps_allowed = ["A"]

    if spin_change != 0 and spin_block_symmetrisation != "none":
        raise NotImplementedError("spin_symmetrisation != 'none' only "
                                  "implemented for spin_change == 0")

    if spin_change == 0 \
       and spin_block_symmetrisation in ("symmetric", "antisymmetric"):
        fac = 1 if spin_block_symmetrisation == "symmetric" else -1
        # Spin mapping between blocks where alpha and beta are just mirrored
        symmetry.spin_block_maps = [("aaaaaa", "bbbbbb", fac),
                                    ("aabaab", "bbabba", fac),
                                    ("aababa", "bbabab", fac),
                                    ("aabbaa", "bbaabb", fac),
                                    ("abaaab", "babbba", fac),
                                    ("abaaba", "babbab", fac),
                                    ("ababaa", "bababb", fac),
                                    ("baaaab", "abbbba", fac),
                                    ("baaaba", "abbbab", fac),
                                    ("baabaa", "abbabb", fac)]

        # Mark blocks which change spin as forbidden
        symmetry.spin_blocks_forbidden = [
            "aaaaab", "aaaaba", "aaabaa",  # 0/1
            "aabaaa", "abaaaa", "baaaaa",  # 1/0
            "aaaabb", "aaabab", "aaabba",  # 0/2
            "abbaaa", "babaaa", "bbaaaa",  # 2/0
            "aaabbb",                      # 0/3
            "bbbaaa",                      # 3/0
            "aababb", "aabbab", "aabbba",  # 1/2
            "abaabb", "ababab", "ababba",  # 1/2
            "baaabb", "baabab", "baabba",  # 1/2
            "abbaab", "abbaba", "abbbaa",  # 2/1
            "babaab", "bababa", "babbaa",  # 2/1
            "bbaaab", "bbaaba", "bbabaa",  # 2/1
            "aabbbb", "ababbb", "baabbb",  # 1/3
            "bbbaab", "bbbaba", "bbbbaa",  # 3/1
            "abbbbb", "babbbb", "bbabbb",  # 2/3
            "bbbabb", "bbbbab", "bbbbba",  # 3/2
        ]

    # Add index permutation symmetry:
    permutations = ["ijkabc"]
    if spaces_t[0] == spaces_t[1]:
        permutations.append("-jikabc")
    if spaces_t[1] == spaces_t[2]:
        permutations.append("-ikjabc")
    if spaces_t[3] == spaces_t[4]:
        permutations.append("-ijkbac")
    if spaces_t[4] == spaces_t[5]:
        permutations.append("-ijkacb")
    if len(permutations) > 1:
        symmetry.permutations = permutations
    return symmetry
