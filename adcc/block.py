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
from .MoSpaces import split_spaces
from .NParticleOperator import OperatorSymmetry


def __getattr__(attr):
    """
    Return a multi-dimensional block string like 'o1o2v1v1'
    when requesting the attribute 'ocvv'.
    """
    if any(c not in ["o", "v", "c"] for c in attr):
        raise AttributeError
    mapping = {"o": "o1", "v": "v1", "c": "o2"}
    return "".join(mapping[c] for c in attr)


def get_canonical_block(block: str,
                        operator_symmetry: OperatorSymmetry
                        ) -> tuple[str, int, tuple]:
    """
    Returns the canonical form of the block and the factor needed to
    reconstruct the original block from it.
    """
    def sort(lst: list[str]) -> list[int]:
        """Returns indices that would sort the list."""
        return sorted(range(len(lst)), key=lambda i: lst[i])

    spaces = split_spaces(block)
    factor = 1
    assert not len(spaces) % 2
    nparticleop = len(spaces) // 2
    bra, ket = spaces[:len(spaces) // 2], spaces[len(spaces) // 2:]

    # sort them
    bra_sorted, ket_sorted = sorted(bra), sorted(ket)
    bra_perm = sort(bra)
    ket_perm = [x + nparticleop for x in sort(ket)]
    if bra_sorted != bra:
        factor *= -1
    if ket_sorted != ket:
        factor *= -1
    bra, ket = bra_sorted, ket_sorted

    # get prefactor
    if operator_symmetry in (OperatorSymmetry.HERMITIAN,
                             OperatorSymmetry.ANTIHERMITIAN) and bra > ket:
        bra, ket = ket, bra
        bra_perm, ket_perm = ket_perm, bra_perm

        if operator_symmetry == OperatorSymmetry.ANTIHERMITIAN:
            factor *= -1

    canonical_block = "".join(["".join(bra) + "".join(ket)])

    # correct permutational still needs to be fixed.
    perm = tuple(bra_perm[::-1] + ket_perm[::-1])
    print(block, canonical_block, perm)
    return (canonical_block, factor, perm)
