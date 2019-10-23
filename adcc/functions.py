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
from .AmplitudeVector import AmplitudeVector

import libadcc


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


def contract(contraction, a, b, out=None):
    """
    Form a single, einsum-like contraction, that is contract
    tensor a and be to form out via a contraction defined
    by the first argument string, e.g. "ab,bc->ac"
    or "abc,bcd->ad".

    Note: The contract function is experimental. Its interface can change
          and the function may disappear in the future.
    """
    if out is None:
        return libadcc.contract(contraction, a, b)
    else:
        return libadcc.contract_to(contraction, a, b, out)


def add(a, b, out=None):
    """
    Return the elementwise sum of two objects
    If out is given the result will be written to the
    latter tensor.
    """
    if out is None:
        return a + b
    if isinstance(a, AmplitudeVector):
        for block in a.blocks:
            add(a[block], b[block], out[block])
    else:
        return libadcc.add(a, b, out)


def subtract(a, b, out=None):
    """
    Return the elementwise difference of two objects
    If out is given the result will be written to the
    latter tensor.
    """
    if out is None:
        return a - b
    if isinstance(a, AmplitudeVector):
        for block in a.blocks:
            subtract(a[block], b[block], out[block])
    else:
        return libadcc.subtract(a, b, out)


def multiply(a, b, out=None):
    """
    Return the elementwise product of two objects
    If out is given the result will be written to the
    latter tensor.

    Note: If out is not given, the symmetry of the
    contained objects will be destroyed!
    """
    if out is None:
        return a * b
    if isinstance(a, AmplitudeVector):
        for block in a.blocks:
            multiply(a[block], b[block], out[block])
    else:
        return libadcc.multiply(a, b, out)


def divide(a, b, out=None):
    """
    Return the elementwise division of two objects
    If out is given the result will be written to the
    latter tensor.

    Note: If out is not given, the symmetry of the
    contained objects will be destroyed!
    """
    if out is None:
        return a / b
    if isinstance(a, AmplitudeVector):
        for block in a.blocks:
            divide(a[block], b[block], out[block])
    else:
        return libadcc.divide(a, b, out)


def linear_combination(coefficients, tensors):
    """
    Form a linear combination from a list of tensors.

    If coefficients is a 1D array, just form a single
    linear combination, else return a list of vectors
    representing the linear combination by reading
    the coefficients row-by-row.
    """
    if len(tensors) == 0:
        raise ValueError("List of tensors cannot be empty")
    if len(tensors) != len(coefficients):
        raise ValueError("Number of coefficient values does not match "
                         "number of tensors.")
    ret = zeros_like(tensors[0])
    return ret.add_linear_combination(coefficients, tensors)
