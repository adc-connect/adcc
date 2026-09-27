## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, TypeVar

import numpy as np

# This module has to stay completely independent of other adcc modules
# (at runtime) to avoid import circles!
if TYPE_CHECKING:
    from .OneParticleOperator import OneParticleOperator

ShapeT = TypeVar("ShapeT", bound=tuple[int, ...])
FloatArray = np.ndarray[ShapeT, np.dtype[np.float64]]

Array1D = FloatArray[tuple[int]]
Array2D = FloatArray[tuple[int, int]]
Array4D = FloatArray[tuple[int, int, int, int]]
DipoleLikeArray = tuple[Array2D, Array2D, Array2D]
# quotes: cannot evaluate this type at runtime
DipoleLike: TypeAlias = "tuple[OneParticleOperator, OneParticleOperator, OneParticleOperator]"
# Once we drop python 3.10 we can write
# QuadrupoleLikeArray = tuple[*DipoleLikeArray, *DipoleLikeArray, *DipoleLikeArray]
QuadrupoleLikeArray = tuple[
    Array2D,
    Array2D,
    Array2D,
    Array2D,
    Array2D,
    Array2D,
    Array2D,
    Array2D,
    Array2D,
]
QuadrupoleLike = tuple[DipoleLike, DipoleLike, DipoleLike]
Coordinate = tuple[float, float, float]


def is_float_array(value: Any, shape: ShapeT) -> TypeGuard[FloatArray[ShapeT]]:
    """
    Whether ``value`` is a float64 numpy array with as many dimensions as ``shape``.
    Note that only the number of dimensions is checked, not the actual shape.
    """
    return isinstance(value, np.ndarray) and value.ndim == len(shape) and value.dtype == np.float64


def is_array_2d(value: Any) -> TypeGuard[Array2D]:
    """Whether ``value`` is a two-dimensional float64 numpy array."""
    return isinstance(value, np.ndarray) and value.ndim == 2 and value.dtype == np.float64


def is_dipole_like_array(value: Any) -> TypeGuard[DipoleLikeArray]:
    """Whether ``value`` is a tuple of three two-dimensional float64 arrays (x, y, z)."""
    return isinstance(value, tuple) and len(value) == 3 and all(is_array_2d(v) for v in value)


def is_quadrupole_like_array(value: Any) -> TypeGuard[QuadrupoleLikeArray]:
    """
    Whether ``value`` is a tuple of nine two-dimensional float64 arrays
    (xx, xy, xz, yx, yy, yz, zx, zy, zz).
    """
    return isinstance(value, tuple) and len(value) == 9 and all(is_array_2d(v) for v in value)


def is_dipole_like(value: Any) -> TypeGuard[DipoleLike]:
    """Whether ``value`` is a tuple of three OneParticleOperators (x, y, z)."""
    from .OneParticleOperator import OneParticleOperator

    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(v, OneParticleOperator) for v in value)
    )


def is_quadrupole_like(value: Any) -> TypeGuard[QuadrupoleLike]:
    """
    Whether ``value`` is a tuple of three dipole-like tuples, i.e., a 3x3 nested
    tuple of OneParticleOperators ((xx, xy, xz), (yx, yy, yz), (zx, zy, zz)).
    """
    return isinstance(value, tuple) and len(value) == 3 and all(is_dipole_like(v) for v in value)
