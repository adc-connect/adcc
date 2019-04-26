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
import unittest

import adcc

from pytest import approx

from adcc import LazyMp
from adcc.testdata.cache import cache
from adcc.solver.davidson import jacobi_davidson


class TestSolverDavidson(unittest.TestCase):
    def test_adc2_singlets(self):
        refdata = cache.reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("adc2", LazyMp(cache.refstate["h2o_sto3g"]))

        # Solve for singlets
        guesses = adcc.guesses_singlet(matrix, n_guesses=10, block="s")
        res = jacobi_davidson(matrix, guesses, n_ep=10)

        ref_singlets = refdata["adc2"]["singlet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_singlets)

    def test_adc2_triplets(self):
        refdata = cache.reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("adc2", LazyMp(cache.refstate["h2o_sto3g"]))

        # Solve for triplets
        guesses = adcc.guesses_triplet(matrix, n_guesses=10, block="s")
        res = jacobi_davidson(matrix, guesses, n_ep=10)

        ref_triplets = refdata["adc2"]["triplet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_triplets)
