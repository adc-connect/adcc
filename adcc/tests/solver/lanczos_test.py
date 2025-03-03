#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from adcc.AdcMatrix import AdcMatrixShifted
from adcc.solver.lanczos import default_print as la_print, lanczos
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver.conjugate_gradient import (IterativeInverse,
                                            default_print as cg_print)
from adcc.solver.explicit_symmetrisation import IndexSpinSymmetrisation

from ..testdata_cache import testdata_cache


class TestSolverLanczos(unittest.TestCase):
    def test_adc2_singlets(self):
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["singlet"]
        matrix = adcc.AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

        # Solve for singlets
        guesses = adcc.guesses_singlet(matrix, n_guesses=5, block="ph")
        res = lanczos(matrix, guesses, n_ep=5, which="SM")

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
        guesses = adcc.guesses_triplet(matrix, n_guesses=6, block="ph")
        res = lanczos(matrix, guesses, n_ep=6, which="SM")

        ref_triplets = refdata["eigenvalues"]
        n_states = min(len(ref_triplets), len(res.eigenvalues))
        assert n_states > 1
        assert res.converged
        assert res.eigenvalues[:n_states] == pytest.approx(ref_triplets[:n_states])

    def test_adc2_shift_invert_singlets(self):
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["singlet"]
        matrix = adcc.AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

        conv_tol = 1e-5
        shift = -0.5

        # Construct shift and inverted matrix:
        shinv = IterativeInverse(AdcMatrixShifted(matrix, shift),
                                 conv_tol=conv_tol / 10,
                                 Pinv=JacobiPreconditioner,
                                 callback=cg_print)

        # Solve for singlets
        guesses = adcc.guesses_singlet(matrix, n_guesses=5, block="ph")
        symm = IndexSpinSymmetrisation(matrix, enforce_spin_kind="singlet")
        res = lanczos(shinv, guesses, n_ep=5, callback=la_print,
                      explicit_symmetrisation=symm)
        assert res.converged

        # Undo spectral transformation and compare
        eigenvalues = sorted(1 / res.eigenvalues - shift)
        ref_singlets = refdata["eigenvalues"]
        n_states = min(len(ref_singlets), len(res.eigenvalues))
        assert n_states > 1
        assert eigenvalues[:n_states] == pytest.approx(ref_singlets[:n_states])

    def test_adc2_shift_invert_triplets(self):
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["triplet"]
        matrix = adcc.AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

        conv_tol = 1e-5
        shift = -0.5

        # Construct shift and inverted matrix:
        shinv = IterativeInverse(AdcMatrixShifted(matrix, shift),
                                 conv_tol=conv_tol / 10,
                                 Pinv=JacobiPreconditioner,
                                 callback=cg_print)

        # Solve for triplets
        guesses = adcc.guesses_triplet(matrix, n_guesses=5, block="ph")
        symm = IndexSpinSymmetrisation(matrix, enforce_spin_kind="triplet")
        res = lanczos(shinv, guesses, n_ep=5, callback=la_print,
                      explicit_symmetrisation=symm)
        assert res.converged

        # Undo spectral transformation and compare
        eigenvalues = sorted(1 / res.eigenvalues - shift)
        ref_triplets = refdata["eigenvalues"]
        n_states = min(len(ref_triplets), len(res.eigenvalues))
        assert n_states > 1
        assert eigenvalues[:n_states] == pytest.approx(ref_triplets[:n_states])
