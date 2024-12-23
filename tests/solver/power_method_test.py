#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse.linalg import aslinearoperator

from adcc.solver.power_method import default_print, power_method


sizes = ["0004", "0050", "0200", "1000"]


@pytest.mark.parametrize("size", sizes)
class TestPowerMethod:
    def test_random_matrix(self, size: str):
        np.random.seed(42)
        size = int(size)
        conv_tol = 1e-10
        ev = np.random.randn(size)
        ev[0] = abs(ev[0]) + 5

        start = np.random.randn(len(ev))
        start[0] += 0.001
        res = power_method(aslinearoperator(np.diag(ev)), start,
                           conv_tol=conv_tol, callback=default_print,
                           explicit_symmetrisation=None,
                           max_iter=100)

        ones = np.zeros(size)
        ones[0] = 1 * np.sign(res.eigenvectors[0][0])
        extrafac = 1
        if size > 100:
            extrafac = 3
        assert_allclose(res.eigenvectors[0], ones, atol=conv_tol * 10 * extrafac)
        assert pytest.approx(res.eigenvalues[0]) == ev[0]
