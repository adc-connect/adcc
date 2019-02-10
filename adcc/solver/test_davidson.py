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

from adcc.solver.davidson import jacobi_davidson
from pytest import approx
from adcc.testdata.cache import cache
import adcc
import unittest


class TestSolverDavidson(unittest.TestCase):
    def test_adc2_singlets(self):
        refdata = cache.reference_data["h2o_sto3g"]

        matrix = adcc.AdcMatrix("adc2", cache.prelim["h2o_sto3g"].ground_state)

        # Solve for singlets
        res = jacobi_davidson(matrix, cache.prelim["h2o_sto3g"].guesses_singlet,
                              n_ep=10)

        ref_singlets = refdata["adc2"]["singlet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_singlets)

    def test_adc2_triplets(self):
        refdata = cache.reference_data["h2o_sto3g"]

        matrix = adcc.AdcMatrix("adc2", cache.prelim["h2o_sto3g"].ground_state)

        # Solve for triplets
        res = jacobi_davidson(matrix, cache.prelim["h2o_sto3g"].guesses_triplet,
                              n_ep=10)

        ref_triplets = refdata["adc2"]["triplet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_triplets)
