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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

# Avoid top level imports from other adcc modules at runtime (circular imports)
if TYPE_CHECKING:
    from _typeshed import SupportsDunderGT

    from .NParticleOperator import OperatorSymmetry

# available spaces and the subset of available spaces with a short hand notation
_shorthands: dict[str, str] = {"o": "o1", "c": "o2", "v": "v1"}
_valid_spaces: tuple[str, ...] = ("o1", "o2", "o3", "v1", "v2")
assert all(v in _valid_spaces for v in _shorthands.values())


def __getattr__(spaces: str) -> str:
    """
    Return a multi-dimensional block string like 'o1o2v1v1' when requesting the attribute 'ocvv'.
    """
    if not spaces or any(c not in _shorthands for c in spaces):
        raise AttributeError(
            f"module {__name__!r} has no attribute {spaces!r} "
            f"(available space shorthands are {', '.join(_shorthands)})."
        )
    return "".join(_shorthands[c] for c in spaces)


def split_spaces(spaces: str) -> list[str]:
    """
    Splits a string of spaces (``spaces``) into the individual space string components.
    """
    splitted = [spaces[i : i + 2] for i in range(0, len(spaces), 2)]
    if any(sp not in _valid_spaces for sp in splitted):
        raise ValueError(
            f"Invalid space string '{spaces}'. Valid spaces are {', '.join(_valid_spaces)}."
        )
    return splitted


_T = TypeVar("_T")
_ComparableT = TypeVar("_ComparableT", bound="SupportsDunderGT[Any]")


@overload
def _sort_anticommuting(
    to_sort: Sequence[_T], key: Callable[[_T], _ComparableT]
) -> tuple[list[_T], int]: ...
@overload
def _sort_anticommuting(
    to_sort: Sequence[_ComparableT], key: None = None
) -> tuple[list[_ComparableT], int]: ...


# adapted from
# https://github.com/sympy/sympy/blob/master/sympy/physics/secondquant.py
def _sort_anticommuting(
    to_sort: Sequence[Any],
    key: "Callable[[Any], SupportsDunderGT[Any]] | None" = None,
) -> tuple[list[Any], int]:
    """
    Sort a list of mutually anticommuting operators into canonical order.

    All elements in ``to_sort`` are assumed to anticommute pairwise.  The
    sorting is performed via a bidirectional bubble sort.

    Parameters
    ----------
    to_sort : Sequence[T]
        Sequence of operators (or any comparable objects) to be sorted.
    key : Callable[[T], SupportsDunderGT], optional
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


def get_canonical_block(
    bra: str, ket: str, operator_symmetry: "OperatorSymmetry"
) -> tuple[str, int, tuple[int, ...]]:
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
    transform : tuple[int, ...]
        A tuple encoding the transformation applied to obtain the desired block
        from the canonical block.
    """
    from .NParticleOperator import OperatorSymmetry

    def invert_transpose_tuple(transpose: tuple[int, ...]) -> tuple[int, ...]:
        """
        Invert transpose tuple containing permutation that maps a block to its
        canonical form. Returns the tuple that maps from the canonical block
        back to the original block.
        """
        inverted: list[int] = [-1] * len(transpose)
        for i, pi in enumerate(transpose):
            inverted[pi] = i
        assert -1 not in inverted, f"Invalid transpose {transpose}"
        return tuple(inverted)

    factor = 1

    bra_splitted = split_spaces(bra)
    bra_sorted, bra_factor = _sort_anticommuting(
        list(enumerate(bra_splitted)), key=lambda tpl: tpl[1]
    )
    bra_transpose = tuple(val for val, _ in bra_sorted)
    bra_space_sorted = [val for _, val in bra_sorted]
    factor *= bra_factor

    ket_splitted = split_spaces(ket)
    ket_sorted, ket_factor = _sort_anticommuting(
        list(enumerate(ket_splitted)), key=lambda tpl: tpl[1]
    )
    ket_transpose = tuple(val + len(bra_splitted) for val, _ in ket_sorted)
    ket_space_sorted = [val for _, val in ket_sorted]
    factor *= ket_factor

    transpose = (*bra_transpose, *ket_transpose)
    canonical_block = "".join(bra_space_sorted) + "".join(ket_space_sorted)

    if operator_symmetry is not OperatorSymmetry.NOSYMMETRY:
        if len(bra_splitted) != len(ket_splitted):
            raise ValueError("Invalid blocks strings.")

        if ket_space_sorted < bra_space_sorted:
            transpose = (*ket_transpose, *bra_transpose)
            canonical_block = "".join(ket_space_sorted) + "".join(bra_space_sorted)
            if operator_symmetry is OperatorSymmetry.ANTIHERMITIAN:
                factor *= -1
            else:
                assert operator_symmetry is OperatorSymmetry.HERMITIAN
    transpose = invert_transpose_tuple(transpose)
    return (canonical_block, factor, transpose)
