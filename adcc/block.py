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
from collections.abc import Callable
from typing import TypeVar, Any, Union


def __getattr__(attr):
    """
    Return a multi-dimensional block string like 'o1o2v1v1'
    when requesting the attribute 'ocvv'.
    """
    if any(c not in ["o", "v", "c"] for c in attr):
        raise AttributeError
    mapping = {"o": "o1", "v": "v1", "c": "o2"}
    return "".join(mapping[c] for c in attr)


T = TypeVar("T")


# adapted from
# https://github.com/sympy/sympy/blob/master/sympy/physics/secondquant.py
def _sort_anticommuting(to_sort: list[T],
                        key: Union[Callable[[T], Any], None] = None
                        ) -> tuple[list[T], int]:
    """
    Sort a list of mutually anticommuting operators into canonical order.

    All elements in ``to_sort`` are assumed to anticommute pairwise.  The
    sorting is performed via a bidirectional bubble sort.

    Parameters
    ----------
    to_sort : list[T]
        List of operators (or any comparable objects) to be sorted.
    key : callable, optional
        Optional function mapping each element of ``to_sort`` to a comparison
        key.  If ``None`` (default), the elements themselves are compared.

    Returns
    -------
    to_sort : list[T]
        The operators sorted into canonical order.
    sign : int
        The sign accumulated from the reordering, i.e. ``+1`` or ``-1``.
    """
    verified = False
    sign = 1
    rng = tuple(range(len(to_sort) - 1))
    rev = tuple(range(len(to_sort) - 3, -1, -1))

    def _identity(x):
        return x

    if key is None:
        key = _identity

    keys = list(map(key, to_sort))
    to_sort = list(to_sort)

    while not verified:
        verified = True
        for i in rng:
            left = keys[i]
            right = keys[i + 1]
            if left > right:
                verified = False
                keys[i], keys[i + 1] = right, left
                to_sort[i], to_sort[i + 1] = to_sort[i + 1], to_sort[i]
                sign *= -1
        if verified:
            break
        for i in rev:
            left = keys[i]
            right = keys[i + 1]
            if left > right:
                verified = False
                keys[i], keys[i + 1] = right, left
                to_sort[i], to_sort[i + 1] = to_sort[i + 1], to_sort[i]
                sign *= -1
    return (to_sort, sign)


def get_canonical_block(bra: str, ket: str,
                        operator_symmetry: OperatorSymmetry
                        ) -> tuple[str, int, tuple]:
    """
    Return the canonical representation of an operator block, together with the
    factor required to recover the desired block and the transpose tuple that
    encodes the transformation from the canonical form back to the requested block.

    Parameters
    ----------
    bra : str
        The bra index string of the block.
    ket : str
        The ket index string of the block.
    operator_symmetry : OperatorSymmetry
        Symmetry of the operator.

    Returns
    -------
    canonical_block : str
        The canonicalised block.
    factor : int
        The integer factor needed to recover the desired block
        from the canonical block, i.e. ``+1`` or ``-1``.
    transform : tuple
        A tuple encoding the transformation applied to obtain the desired block
        from the canonical block.
    """
    def invert_transpose_tuple(p: tuple[int, ...]) -> tuple[int, ...]:
        """
        Invert transpose tuple containing permutation that maps a block to its
        canonical form. Returns the tuple that maps from the canonical block
        back to the original block.
        """
        q = [None] * len(p)
        for i, pi in enumerate(p):
            q[pi] = i
        return tuple(q)

    factor = 1

    bra = split_spaces(bra) if bra else []
    bra_sorted, bra_factor = _sort_anticommuting(
        list(enumerate(bra)), key=lambda tpl: tpl[1]
    )
    bra_transpose = [val for val, _ in bra_sorted]
    bra_space_sorted = [val for _, val in bra_sorted]
    factor *= bra_factor

    ket = split_spaces(ket) if ket else []
    ket_sorted, ket_factor = _sort_anticommuting(
        list(enumerate(ket)), key=lambda tpl: tpl[1]
    )
    ket_transpose = [val for val, _ in ket_sorted]
    ket_transpose = tuple(x + len(bra) for x in ket_transpose)
    ket_space_sorted = [val for _, val in ket_sorted]
    factor *= ket_factor

    transpose = (*bra_transpose, *ket_transpose)
    canonical_block = "".join(bra_space_sorted) + "".join(ket_space_sorted)

    if operator_symmetry != OperatorSymmetry.NOSYMMETRY:
        if len(bra) != len(ket):
            raise ValueError("Invalid blocks strings.")

        if ket_space_sorted < bra_space_sorted:
            transpose = (*ket_transpose, *bra_transpose)
            canonical_block = "".join(ket_space_sorted) + "".join(bra_space_sorted)
            if operator_symmetry == OperatorSymmetry.ANTIHERMITIAN:
                factor *= -1
    transpose = invert_transpose_tuple(transpose)
    return (canonical_block, factor, transpose)
