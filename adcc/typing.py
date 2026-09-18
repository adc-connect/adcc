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
import numpy as np

Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
DipoleLikeArray = tuple[Array2D, Array2D, Array2D]
# Once we drop python 3.10 we can write
# QuadrupoleLike = tuple[*DipoleLike, *DipoleLike, *DipoleLike]
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
Coordinate = tuple[float, float, float]
