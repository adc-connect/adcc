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
from typing import Any, TypeGuard

import numpy as np

from .OneParticleOperator import OneParticleOperator

Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Array4D = np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]
DipoleLikeArray = tuple[Array2D, Array2D, Array2D]
DipoleLike = tuple[OneParticleOperator, OneParticleOperator, OneParticleOperator]
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


def is_array_2d(value: Any) -> TypeGuard[Array2D]:
    return isinstance(value, np.ndarray) and value.ndim == 2 and value.dtype == np.float64


def is_quadrupole_like_array(value: Any) -> TypeGuard[QuadrupoleLikeArray]:
    return isinstance(value, tuple) and len(value) == 9 and all(is_array_2d(v) for v in value)


def is_dipole_like(value: Any) -> TypeGuard[DipoleLike]:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(v, OneParticleOperator) for v in value)
    )


def is_quadruple_like(value: Any) -> TypeGuard[QuadrupoleLike]:
    return isinstance(value, tuple) and len(value) == 3 and all(is_dipole_like(v) for v in value)
