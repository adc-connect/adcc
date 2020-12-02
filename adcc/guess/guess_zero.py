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


def guess_zero(matrix, spin_change=0, spin_block_symmetrisation="none"):
    """
    Return an AmplitudeVector object filled with zeros, but where the symmetry
    has been properly set up to meet the specified requirements on the guesses.

    matrix       The matrix for which guesses are to be constructed
    spin_change  The spin change to enforce in an excitation.
                 Typical values are 0 (singlet/triplet/any) and -1 (spin-flip).
    spin_block_symmetrisation
                 Symmetrisation to enforce between equivalent spin blocks, which
                 all yield the desired spin_change. E.g. if spin_change == 0,
                 then both the alpha->alpha and beta->beta blocks of the singles
                 part of the excitation vector achieve a spin change of 0.
                 The symmetry specified with this parameter will then be imposed
                 between the a-a and b-b blocks. Valid values are "none",
                 "symmetric" and "antisymmetric", where "none" enforces
                 no particular symmetry.
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
                 Symmetrisation to enforce between equivalent spin blocks, which
                 all yield the desired spin_change. E.g. if spin_change == 0,
                 then both the alpha->alpha and beta->beta blocks of the singles
                 part of the excitation vector achieve a spin change of 0.
                 The symmetry specified with this parameter will then be imposed
                 between the a-a and b-b blocks. Valid values are "none",
                 "symmetric" and "antisymmetric", where "none" enforces
                 no particular symmetry.
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

    max_spin_change = 0
    if "ph" in matrix.axis_blocks:
        max_spin_change = 1
    if "pphh" in matrix.axis_blocks:
        max_spin_change = 2
    if spin_change > max_spin_change:
        raise ValueError("spin_change for singles guesses may only be in the "
                         f"range [{-max_spin_change}, {max_spin_change}] and "
                         f"not {spin_change}.")

    symmetries = {}
    if "ph" in matrix.axis_blocks:
        symmetries["ph"] = guess_symmetry_singles(
            matrix, spin_change=spin_change,
            spin_block_symmetrisation=spin_block_symmetrisation
        )
    if "pphh" in matrix.axis_blocks:
        symmetries["pphh"] = guess_symmetry_doubles(
            matrix, spin_change=spin_change,
            spin_block_symmetrisation=spin_block_symmetrisation
        )
    return symmetries


def guess_symmetry_singles(matrix, spin_change=0,
                           spin_block_symmetrisation="none"):
    symmetry = Symmetry(matrix.mospaces, "".join(matrix.axis_spaces["ph"]))
    symmetry.irreps_allowed = ["A"]
    if spin_change != 0 and spin_block_symmetrisation != "none":
        raise NotImplementedError("spin_symmetrisation != 'none' only "
                                  "implemented for spin_change == 0")
    elif spin_block_symmetrisation == "symmetric":
        symmetry.spin_block_maps = [("aa", "bb", 1)]
        symmetry.spin_blocks_forbidden = ["ab", "ba"]
    elif spin_block_symmetrisation == "antisymmetric":
        symmetry.spin_block_maps = [("aa", "bb", -1)]
        symmetry.spin_blocks_forbidden = ["ab", "ba"]
    return symmetry


def guess_symmetry_doubles(matrix, spin_change=0,
                           spin_block_symmetrisation="none"):
    spaces_d = matrix.axis_spaces["pphh"]
    symmetry = Symmetry(matrix.mospaces, "".join(spaces_d))
    symmetry.irreps_allowed = ["A"]

    if spin_change != 0 and spin_block_symmetrisation != "none":
        raise NotImplementedError("spin_symmetrisation != 'none' only "
                                  "implemented for spin_change == 0")

    if spin_change == 0 \
       and spin_block_symmetrisation in ("symmetric", "antisymmetric"):
        fac = 1 if spin_block_symmetrisation == "symmetric" else -1
        # Spin mapping between blocks where alpha and beta are just mirrored
        symmetry.spin_block_maps = [("aaaa", "bbbb", fac),
                                    ("abab", "baba", fac),
                                    ("abba", "baab", fac)]

        # Mark blocks which change spin as forbidden
        symmetry.spin_blocks_forbidden = ["aabb",  # spin_change +2
                                          "bbaa",  # spin_change -2
                                          "aaab",  # spin_change +1
                                          "aaba",  # spin_change +1
                                          "abaa",  # spin_change -1
                                          "baaa",  # spin_change -1
                                          "abbb",  # spin_change +1
                                          "babb",  # spin_change +1
                                          "bbab",  # spin_change -1
                                          "bbba"]  # spin_change -1

    # Add index permutation symmetry:
    permutations = ["ijab"]
    if spaces_d[0] == spaces_d[1]:
        permutations.append("-jiab")
    if spaces_d[2] == spaces_d[3]:
        permutations.append("-ijba")
    if len(permutations) > 1:
        symmetry.permutations = permutations
    return symmetry
