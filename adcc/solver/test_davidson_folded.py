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

from pytest import approx

from adcc import LazyMp
from adcc.testdata.cache import cache
from adcc.solver.davidson import davidson_folded_DIIS


class TestSolverDavidson(unittest.TestCase):
    def test_adc2_singlets_folded(self):
        refdata = cache.reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("adc2", LazyMp(cache.refstate["h2o_sto3g"]))

        # Solve for singlets
        guesses = adcc.guesses_singlet(matrix, n_guesses=8, block="ph")
        res = davidson_folded_DIIS(matrix, guesses, n_ep=8)

        ref_singlets = refdata["adc2"]["singlet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_singlets[:8])

    def test_adc2_triplets_folded(self):
        refdata = cache.reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("adc2", LazyMp(cache.refstate["h2o_sto3g"]))

        # Solve for triplets
        guesses = adcc.guesses_triplet(matrix, n_guesses=8, block="ph")
        res = davidson_folded_DIIS(matrix, guesses, n_ep=8)

        ref_triplets = refdata["adc2"]["triplet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_triplets[:8])

    def test_adc2_singlets_folded_adc1Guesses(self):
        from adcc.workflow import run_adc

        refdata = cache.reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("adc2", LazyMp(cache.refstate["h2o_sto3g"]))

        # run adc1 for initial guesses
        matrix_adc1 = adcc.AdcMatrix("adc1", LazyMp(cache.refstate["h2o_sto3g"]))
        adc1 = run_adc(matrix_adc1, method="adc1", n_singlets=8)
        omegas = adc1.excitation_energy_uncorrected
        guesses = adc1.excitation_vector
        # Solve for singlets
        res = davidson_folded_DIIS(matrix, guesses, omegas=omegas, n_ep=8)

        ref_singlets = refdata["adc2"]["singlet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_singlets[:8])

    def test_adc2_triplets_folded_adc1(self):
        from adcc.workflow import run_adc

        refdata = cache.reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("adc2", LazyMp(cache.refstate["h2o_sto3g"]))

        # run adc1 for initial guesses
        matrix_adc1 = adcc.AdcMatrix("adc1", LazyMp(cache.refstate["h2o_sto3g"]))
        adc1 = run_adc(matrix_adc1, method="adc1", n_triplets=8)
        omegas = adc1.excitation_energy_uncorrected
        guesses = adc1.excitation_vector
        # Solve for triplets
        res = davidson_folded_DIIS(matrix, guesses=guesses, omegas=omegas, n_ep=8)

        ref_triplets = refdata["adc2"]["triplet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_triplets[:8])
