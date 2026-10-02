## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import re
from collections.abc import Sequence
from typing import Any, Protocol, TypeGuard, TypeVar, overload

import opt_einsum
from opt_einsum.typing import OptimizeKind

import libadcc

from .AmplitudeVector import AmplitudeVector
from .typing import Array1D

_TensorT = TypeVar("_TensorT")
_TensorT_contra = TypeVar("_TensorT_contra", contravariant=True)


class SupportsDot(Protocol[_TensorT_contra]):
    @overload
    def dot(self, other: _TensorT_contra, /) -> float: ...
    @overload
    def dot(self, other: Sequence[_TensorT_contra], /) -> Array1D: ...


@overload
def dot(a: SupportsDot[_TensorT], b: _TensorT) -> float: ...
@overload
def dot(a: SupportsDot[_TensorT], b: Sequence[_TensorT]) -> Array1D: ...
def dot(a: SupportsDot[_TensorT], b: _TensorT | Sequence[_TensorT]) -> float | Array1D:
    """
    Form the scalar product between two tensors.
    """
    return a.dot(b)


# once we drop python 3.10:
# use typing.Self instead of _TensorT
class SupportsCopy(Protocol):
    def copy(self: _TensorT) -> _TensorT: ...  # noqa: PYI019


_SupportsCopyT = TypeVar("_SupportsCopyT", bound=SupportsCopy)


def copy(a: _SupportsCopyT) -> _SupportsCopyT:
    """
    Return a copy of the input tensor.
    """
    return a.copy()


class SupportsTranspose(Protocol):
    @overload
    def transpose(self: _TensorT) -> _TensorT: ...
    @overload
    def transpose(self: _TensorT, axes: Sequence[int], /) -> _TensorT: ...


_SupportsTransposeT = TypeVar("_SupportsTransposeT", bound=SupportsTranspose)


def transpose(a: _SupportsTransposeT, axes: Sequence[int] | None = None) -> _SupportsTransposeT:
    """
    Return the transpose of a tensor as a *copy*. If axes is not given all axes are reversed.
    Else the axes are expect as a tuple of indices, e.g. (1,0,2,3) will permute first two axes
    in the returned tensor.
    """
    if axes is None:
        return a.transpose()
    return a.transpose(axes)


class SupportsEmptyLike(Protocol):
    def empty_like(self: _TensorT) -> _TensorT: ...  # noqa: PYI019


_SupportsEmptyLikeT = TypeVar("_SupportsEmptyLikeT", bound=SupportsEmptyLike)


def empty_like(a: _SupportsEmptyLikeT) -> _SupportsEmptyLikeT:
    """
    Return an empty tensor of the same shape and symmetry as the input tensor.
    """
    return a.empty_like()


class SupportsZerosLike(Protocol):
    def zeros_like(self: _TensorT) -> _TensorT: ...  # noqa: PYI019


_SupportsZerosLikeT = TypeVar("_SupportsZerosLikeT", bound=SupportsZerosLike)


def zeros_like(a: _SupportsZerosLikeT) -> _SupportsZerosLikeT:
    """
    Return a zero tensor of the same shape and symmetry as the input tensor.
    """
    return a.zeros_like()


class SupportsOnesLike(Protocol):
    def ones_like(self: _TensorT) -> _TensorT: ...  # noqa: PYI019


_SupportsOnesLikeT = TypeVar("_SupportsOnesLikeT", bound=SupportsOnesLike)


def ones_like(a: _SupportsOnesLikeT) -> _SupportsOnesLikeT:
    """
    Return tensor of the same shape and symmetry as the input tensor, but initialised to 1,
    that is the canonical blocks are 1 and the other ones are symmetry-equivalent (-1 or 0).
    """
    return a.ones_like()


class SupportsNosymLike(Protocol):
    def nosym_like(self: _TensorT) -> _TensorT: ...  # noqa: PYI019


_SupportsNosymLikeT = TypeVar("_SupportsNosymLikeT", bound=SupportsNosymLike)


def nosym_like(a: _SupportsNosymLikeT) -> _SupportsNosymLikeT:
    """
    Return tensor of the same shape, but without the symmetry setup of the input tensor.
    """
    return a.nosym_like()


def is_amplitude_vector_sequence(value: Any) -> TypeGuard[Sequence[AmplitudeVector]]:
    return isinstance(value, Sequence) and all(isinstance(v, AmplitudeVector) for v in value)


def is_tensor_sequence(value: Any) -> TypeGuard[Sequence[libadcc.Tensor]]:
    return isinstance(value, Sequence) and all(isinstance(v, libadcc.Tensor) for v in value)


@overload
def lincomb(
    coefficients: Sequence[float] | Array1D,
    tensors: Sequence[libadcc.Tensor],
    evaluate: bool = False,
) -> libadcc.Tensor: ...
@overload
def lincomb(
    coefficients: Sequence[float] | Array1D,
    tensors: Sequence[AmplitudeVector],
    evaluate: bool = False,
) -> AmplitudeVector: ...
def lincomb(
    coefficients: Sequence[float] | Array1D,
    tensors: Sequence[AmplitudeVector] | Sequence[libadcc.Tensor],
    evaluate: bool = False,
) -> AmplitudeVector | libadcc.Tensor:
    """
    Form a linear combination from a list of tensors.

    Parameters
    ----------
    coefficients : Sequence[float] | Array1D
        Coefficients for the linear combination
    tensors : Sequence[AmplitudeVector] | Sequence[libadcc.Tensor]
        Tensors for the linear combination
    evaluate : bool
        Should the linear combination be evaluated (True) or should just
        a lazy expression be formed (False). Notice that `lincomb(..., evaluate=True)`
        is identical to `lincomb(..., evaluate=False).evaluate()`,
        but the former is generally faster.
    """
    if len(tensors) == 0:
        raise ValueError("Sequence of tensors cannot be empty.")
    if len(tensors) != len(coefficients):
        raise ValueError("Number of coefficient values does not match number of tensors.")

    if is_amplitude_vector_sequence(tensors):
        # the amplitude vectors might contain different blocks: loop over the union of blocks
        # only considering tensors which have the corresponding block (treating missing blocks
        # as zero blocks)
        ret: dict[str, libadcc.Tensor] = {}
        for block in {block for vec in tensors for block in vec}:
            relevant = tuple(i for i, vec in enumerate(tensors) if block in vec)
            ret[block] = lincomb(
                tuple(coefficients[i] for i in relevant),
                tuple(tensors[i][block] for i in relevant),
                evaluate=evaluate,
            )
        return AmplitudeVector(**ret)
    elif is_tensor_sequence(tensors):
        if evaluate:
            # Perform strict evaluation on this linear combination
            return libadcc.linear_combination_strict(coefficients, tensors)
        # Perform lazy evaluation on this linear combination
        start = float(coefficients[0]) * tensors[0]
        return sum(
            (float(c) * t for (c, t) in zip(coefficients[1:], tensors[1:], strict=True)), start
        )
    raise TypeError(
        "Invalid tensor type. Supported are a sequence of 'AmplitudeVector' or a sequence of "
        "'libadcc.Tensor'."
    )


class SupportsEvaluate(Protocol):
    def evaluate(self: _TensorT) -> _TensorT: ...  # noqa: PYI019


_SupportsEvaluateT = TypeVar("_SupportsEvaluateT", bound=SupportsEvaluate)


@overload
def evaluate(a: _SupportsEvaluateT) -> _SupportsEvaluateT: ...
@overload
def evaluate(a: Sequence[_SupportsEvaluateT]) -> list[_SupportsEvaluateT]: ...
def evaluate(
    a: _SupportsEvaluateT | Sequence[_SupportsEvaluateT],
) -> _SupportsEvaluateT | list[_SupportsEvaluateT]:
    """Force full evaluation of a tensor expression"""
    if isinstance(a, Sequence):
        return [evaluate(elem) for elem in a]
    return a.evaluate()


def direct_sum(subscripts: str, *operands: libadcc.Tensor) -> libadcc.Tensor:
    """
    Form the direct sum of tensors, e.g. ``direct_sum("ia+jb->ijab", a, b)`` computes
    res[i, j, a, b] = a[i, a] + b[j, b].

    Parameters
    ----------
    subscripts : str
        Specifies the index labels of each operand, separated by "+", "-" or ","
        (equivalent to "+"). A "-" negates the following operand. Labels are single letters
        (case-sensitive) and must be unique across all operands. An optional
        "->" followed by a permutation of all labels specifies the axis order of
        the result tensor. Otherwise the axes follow the order of the operands.
        Whitespaces are ignored.
    *operands : libadcc.Tensor
        At least two tensors. The number of labels of each term has to match the
        dimension of the corresponding tensor.

    Examples
    --------
    >>> direct_sum("a-i->ia", fvv_diag, foo_diag)  # res[i, a] = fvv[a] - foo[i]
    >>> direct_sum("-i-j+a->aij", eps_o, eps_o, eps_v)
    """
    subscripts = re.sub(r"\s", "", subscripts)
    # split in src and dest
    src, arrow, dest = subscripts.partition("->")
    # src has to be a sequence of the form: [sign/comma][indices]
    # with the first sign being optional
    # dest has to be a sequence of [indices]
    if not re.fullmatch(r"[-+]?[a-zA-Z]+(?:[-+,][a-zA-Z]+)*", src) or (
        arrow and not re.fullmatch(r"[a-zA-Z]+", dest)
    ):
        raise ValueError(f"Invalid 'direct_sum' subscripts '{subscripts}'.")
    # split src into the individual terms: The first term might not hold a sign
    terms = re.findall(r"([-+,]?)([a-zA-Z]+)", src)
    if len(operands) < 2:
        raise ValueError(f"'direct_sum' requires at least two operands, got {len(operands)}.")
    if len(terms) != len(operands):
        raise ValueError(
            f"Number of operands (= {len(operands)}) does not agree with the number "
            f"of subscript terms (= {len(terms)}, parsed from {subscripts})."
        )
    # verify that the dimensions match
    for i, ((_, idcs), op) in enumerate(zip(terms, operands, strict=True)):
        if len(idcs) != op.ndim:
            raise ValueError(
                f"Subscripts (= {idcs}) do not match the dimension "
                f"of {i + 1}-th tensor (= {op.ndim})."
            )
    # verify that we have no repeating index in the source subscripts
    # direct_sum('i+i', a, b) computes a_i + b_j, while the code implies a_i + b_i
    src_indices = "".join(idcs for _, idcs in terms)
    if len(set(src_indices)) != len(src_indices):
        raise ValueError(
            f"Repeated index detected in source subscripts '{src_indices}' "
            f"parsed from '{subscripts}'."
        )
    # compute the result
    res: libadcc.Tensor = -operands[0] if terms[0][0] == "-" else operands[0]
    for (sign, _), op in zip(terms[1:], operands[1:], strict=True):
        res = libadcc.direct_sum(res, -op if sign == "-" else op)
    # determine whether we have to transpose the result tensor
    if arrow:
        if sorted(src_indices) != sorted(dest):
            raise ValueError(
                f"Target subscripts '{dest}' must be a permutation of the "
                f"source subscripts '{src_indices}'."
            )
        res = res.transpose(tuple(src_indices.index(c) for c in dest))
    return res


def einsum(
    subscripts: str, *operands: libadcc.Tensor, optimise: OptimizeKind = "auto"
) -> libadcc.Tensor | float:
    """
    Evaluate Einstein summation convention for the operands similar
    to numpy's einsum function. Uses opt_einsum and libadcc to
    perform the contractions.

    Using this function does not evaluate, but returns a contraction
    expression tree where contractions are queued in optimal order.

    Parameters
    ----------

    subscripts : str
        Specifies the subscripts for summation.
    *operands : libadcc.Tensor
        These are the arrays for the operation.
    optimise : OptimizeKind, optional (default: ``auto``)
        Choose the type of the path optimisation, see
        opt_einsum.contract for details.
    """
    # The union return type is unfortunately correct,
    # but that means that callers have to narrow the type.
    return opt_einsum.contract(
        subscripts,
        *operands,
        optimize=optimise,
        backend="libadcc",  # not registered in opt_einsums BackendType # type: ignore
    )
