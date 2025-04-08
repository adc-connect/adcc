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
import adcc
import pytest
import numpy as np

from adcc.solver import IndexSpinSymmetrisation
from adcc.solver.power_method import default_print as powprint, power_method
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver.conjugate_gradient import (
    IterativeInverse, conjugate_gradient, default_print, guess_from_previous
)

from ..testdata_cache import testdata_cache


class TestConjugateGradient:
    def base_adc2(self, kind: str, guess_function: callable):
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )[kind]
        matrix = adcc.AdcMatrix(
            "adc2", testdata_cache.refstate("h2o_sto3g", case="gen")
        )

        conv_tol = 1e-6
        guesses = guess_function(matrix, n_guesses=1)
        symm = IndexSpinSymmetrisation(matrix, enforce_spin_kind=kind)
        inverse = IterativeInverse(
            matrix, Pinv=JacobiPreconditioner, conv_tol=conv_tol / 10,
            construct_guess=guess_from_previous
        )
        res = power_method(
            inverse, guesses[0], conv_tol=conv_tol, explicit_symmetrisation=symm,
            callback=powprint, max_iter=100
        )

        ref_singlets = refdata["eigenvalues"]
        assert res.converged
        assert 1 / res.eigenvalues[0] == pytest.approx(ref_singlets[0])

    def test_adc2_singlet(self):
        self.base_adc2("singlet", adcc.guesses_singlet)

    def test_adc2_triplet(self):
        self.base_adc2("triplet", adcc.guesses_triplet)

    def test_adc2_triplet_random(self):
        def guess_random(matrix, n_guesses):
            guess = adcc.guess_zero(
                matrix, spin_block_symmetrisation="antisymmetric"
            )
            guess.set_random()
            return [guess]
        self.base_adc2("triplet", guess_random)

    def test_adc1_linear_solve(self):
        conv_tol = 1e-9
        matrix = adcc.AdcMatrix(
            "adc1", testdata_cache.refstate("h2o_sto3g", case="gen")
        )
        rhs = adcc.guess_zero(matrix)
        rhs.set_random()

        guess = rhs.copy()
        guess.set_random()
        res = conjugate_gradient(matrix, rhs, guess, callback=default_print,
                                 conv_tol=conv_tol)
        residual = matrix @ res.solution - rhs
        assert np.sqrt(residual @ residual) < conv_tol

    def test_adc2x_linear_solve(self):
        conv_tol = 1e-9
        matrix = adcc.AdcMatrix(
            "adc2x", testdata_cache.refstate("h2o_sto3g", case="gen")
        )
        rhs = adcc.guess_zero(matrix)
        rhs.set_random()

        guess = rhs.copy()
        guess.set_random()
        res = conjugate_gradient(matrix, rhs, guess, callback=default_print,
                                 conv_tol=conv_tol, Pinv=JacobiPreconditioner)
        residual = matrix @ res.solution - rhs
        assert np.sqrt(residual @ residual) < conv_tol
