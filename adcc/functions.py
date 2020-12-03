#!/usr/bin/env python3
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
import libadcc

import opt_einsum

from .AmplitudeVector import AmplitudeVector


def dot(a, b):
    """
    Form the scalar product between two tensors.
    """
    return a.dot(b)


def copy(a):
    """
    Return a copy of the input tensor.
    """
    return a.copy()


def transpose(a, axes=None):
    """
    Return the transpose of a tensor as a *copy*.
    If axes is not given all axes are reversed.
    Else the axes are expect as a tuple of indices,
    e.g. (1,0,2,3) will permute first two axes in the
    returned tensor.
    """
    if axes:
        return a.transpose(axes)
    else:
        return a.transpose()


def empty_like(a):
    """
    Return an empty tensor of the same shape and symmetry as
    the input tensor.
    """
    return a.empty_like()


def zeros_like(a):
    """
    Return a zero tensor of the same shape and symmetry as
    the input tensor.
    """
    return a.zeros_like()


def ones_like(a):
    """
    Return tensor of the same shape and symmetry as
    the input tensor, but initialised to 1,
    that is the canonical blocks are 1 and the
    other ones are symmetry-equivalent (-1 or 0)
    """
    return a.ones_like()


def nosym_like(a):
    """
    Return tensor of the same shape, but without the
    symmetry setup of the input tensor.
    """
    return a.nosym_like()


def lincomb(coefficients, tensors, evaluate=False):
    """
    Form a linear combination from a list of tensors.

    If coefficients is a 1D array, just form a single
    linear combination, else return a list of vectors
    representing the linear combination by reading
    the coefficients row-by-row.

    Parameters
    ----------
    coefficients : list
        Coefficients for the linear combination
    tensors : list
        Tensors for the linear combination
    evaluate : bool
        Should the linear combination be evaluated (True) or should just
        a lazy expression be formed (False). Notice that
        `lincomb(..., evaluate=True)`
        is identical to `lincomb(..., evaluate=False).evaluate()`,
        but the former is generally faster.
    """
    if len(tensors) == 0:
        raise ValueError("List of tensors cannot be empty")
    if len(tensors) != len(coefficients):
        raise ValueError("Number of coefficient values does not match "
                         "number of tensors.")
    if isinstance(tensors[0], AmplitudeVector):
        return AmplitudeVector(**{
            block: lincomb(coefficients, [ten[block] for ten in tensors],
                           evaluate=evaluate)
            for block in tensors[0].blocks_ph
        })
    elif not isinstance(tensors[0], libadcc.Tensor):
        raise TypeError("Tensor type not supported")

    if evaluate:
        # Perform strict evaluation on this linear combination
        return libadcc.linear_combination_strict(coefficients, tensors)
    else:
        # Perform lazy evaluation on this linear combination
        start = float(coefficients[0]) * tensors[0]
        return sum((float(c) * t
                    for (c, t) in zip(coefficients[1:], tensors[1:])), start)


def linear_combination(*args, **kwargs):
    import warnings

    warnings.warn(DeprecationWarning("linear_combination is deprecated and will "
                                     "be removed in 0.17. Use lincomb."))
    return lincomb(*args, **kwargs)


def evaluate(a):
    """Force full evaluation of a tensor expression"""
    if isinstance(a, list):
        return [evaluate(elem) for elem in a]
    elif hasattr(a, "evaluate"):
        return a.evaluate()
    else:
        return libadcc.evaluate(a)


def direct_sum(subscripts, *operands):
    subscripts = subscripts.replace(" ", "")

    def split_signs_symbols(subscripts):
        subscripts = subscripts.replace(",", "+")
        if subscripts[0] not in "+-":
            subscripts = "+" + subscripts
        signs = [x for x in subscripts if x in "+-"]
        symbols = subscripts[1:].replace("-", "+").split("+")
        return signs, symbols

    if "->" in subscripts:
        src, dest = subscripts.split("->")
        signs, src = split_signs_symbols(src)
        # permutation = tuple(dest.index(c) for c in "".join(src))
        permutation = tuple("".join(src).index(c) for c in dest)
    else:
        signs, src = split_signs_symbols(subscripts)
        permutation = None

    if len(src) != len(operands):
        raise ValueError("Number of contraction subscripts does not agree with "
                         "number of operands")
    for i, idcs in enumerate(src):
        if len(idcs) != operands[i].ndim:
            raise ValueError(f"Number of subscripts of {i}-th tensor (== {idcs}) "
                             "does not match dimension of tensor "
                             f"(== {operands[i].ndim}).")

    if signs[0] == "-":
        res = -operands[0]
    else:
        res = operands[0]
    for i, op in enumerate(operands[1:]):
        if signs[i + 1] == "-":
            op = -op
        res = libadcc.direct_sum(res, op)
    if permutation is not None:
        res = res.transpose(permutation)
    return res


def einsum(subscripts, *operands, optimise="auto"):
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
    *operands : list of tensors
        These are the arrays for the operation.
    optimise : str, list or bool, optional (default: ``auto``)
        Choose the type of the path optimisation, see
        opt_einsum.contract for details.
    """
    return opt_einsum.contract(subscripts, *operands, optimize=optimise,
                               backend="libadcc")


def contract(subscripts, a, b):
    """
    Form a single, einsum-like contraction, that is contract
    tensor a and be to form out via a contraction defined
    by the first argument string, e.g. "ab,bc->ac"
    or "abc,bcd->ad".

    Note: The contract function is deprecated. It will disappear in 0.16.
    """
    import warnings

    warnings.warn(DeprecationWarning("contract is deprecated and will "
                                     "be removed in 0.16. Use einsum."))
    return einsum(subscripts, a, b)
