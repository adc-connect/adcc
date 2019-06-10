#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import adcc
import unittest
import numpy as np

from adcc import LazyMp
from adcc.solver.adcman import jacobi_davidson
from adcc.testdata.cache import cache

import pytest
import libadcc

from pytest import approx


@pytest.mark.skipif("adcman" not in libadcc.__features__,
                    reason="adcc -> adcman interface not enabled.")
class TestSolverAdcman(unittest.TestCase):
    def test_adc2s(self):
        refdata = cache.reference_data["h2o_sto3g"]

        # Setup the matrix
        matrix = adcc.AdcMatrix("adc2", LazyMp(cache.refstate["h2o_sto3g"]))

        # Solve for singlets and triplets
        res = jacobi_davidson(matrix, n_singlets=9, n_triplets=10)
        res_singlets, res_triplets = res

        ref_singlets = refdata["adc2"]["singlet"]["eigenvalues"]
        ref_triplets = refdata["adc2"]["triplet"]["eigenvalues"]

        assert np.all(res_singlets.residuals_converged)
        assert np.all(res_triplets.residuals_converged)
        assert res_singlets.eigenvalues == approx(ref_singlets)
        assert res_triplets.eigenvalues == approx(ref_triplets)
