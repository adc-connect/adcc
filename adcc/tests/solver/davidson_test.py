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
import adcc
import unittest
import pytest

from adcc import LazyMp
from adcc.solver.davidson import jacobi_davidson

from ..testdata_cache import testdata_cache


class TestSolverDavidson(unittest.TestCase):
    def test_adc2_singlets(self):
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["singlet"]

        matrix = adcc.AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

        # Solve for singlets
        guesses = adcc.guesses_singlet(matrix, n_guesses=9, block="ph")
        res = jacobi_davidson(matrix, guesses, n_ep=9)

        ref_singlets = refdata["eigenvalues"]
        n_states = min(len(ref_singlets), len(res.eigenvalues))
        assert n_states > 1
        assert res.converged
        assert res.eigenvalues[:n_states] == pytest.approx(ref_singlets[:n_states])

    def test_adc2_triplets(self):
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["triplet"]
        matrix = adcc.AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

        # Solve for triplets
        guesses = adcc.guesses_triplet(matrix, n_guesses=10, block="ph")
        res = jacobi_davidson(matrix, guesses, n_ep=10)

        ref_triplets = refdata["eigenvalues"]
        n_states = min(len(ref_triplets), len(res.eigenvalues))
        assert n_states > 1
        assert res.converged
        assert res.eigenvalues[:n_states] == pytest.approx(ref_triplets[:n_states])
