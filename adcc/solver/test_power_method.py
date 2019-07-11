#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import unittest
import numpy as np

from numpy.testing import assert_allclose
from scipy.sparse.linalg import aslinearoperator

from pytest import approx

from adcc.misc import expand_test_templates
from adcc.solver.power_method import default_print, power_method

sizes = ["0004", "0050", "0200", "1000"]


@expand_test_templates(sizes)
class TestPowerMethod(unittest.TestCase):
    def template_random_matrix(self, size):
        size = int(size)
        conv_tol = 1e-10
        ev = np.random.randn(size)
        ev[0] = abs(ev[0]) + 5

        start = np.random.randn(len(ev))
        start[0] += 0.001
        res = power_method(aslinearoperator(np.diag(ev)), start,
                           conv_tol=conv_tol, callback=default_print,
                           explicit_symmetrisation=None)

        ones = np.zeros(size)
        ones[0] = 1 * np.sign(res.eigenvectors[0][0])
        assert_allclose(res.eigenvectors[0], ones, atol=conv_tol * 10)
        assert approx(res.eigenvalues[0]) == ev[0]
