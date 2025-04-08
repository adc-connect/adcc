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
from adcc.solver.davidson import jacobi_davidson, eigsh
from adcc.misc import cached_property

from ..testdata_cache import testdata_cache


class TestSolverDavidson(unittest.TestCase):
    @cached_property
    def matrix(self):
        return adcc.AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

    def test_n_guesses(self):
        # we have to have a guess for each state
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=1, block="ph")
        with pytest.raises(ValueError):
            eigsh(self.matrix, guesses, n_ep=2)
        res = eigsh(self.matrix, guesses, n_ep=1, max_iter=1)
        assert len(res.eigenvalues) == 1
        # by default: construct 1 state for each guess
        res = eigsh(self.matrix, guesses, max_iter=1)
        assert len(res.eigenvalues) == 1

    def test_n_block(self):
        # has to be: n_ep <= n_block <= n_guesses
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=3, block="ph")
        with pytest.raises(ValueError):
            eigsh(self.matrix, guesses, n_ep=2, n_block=1)
        with pytest.raises(ValueError):
            eigsh(self.matrix, guesses, n_ep=2, n_block=4)
        # defaults to n_ep
        res = eigsh(self.matrix, guesses, n_ep=2, max_iter=2)
        assert len(res.eigenvalues) == 2
        assert res.n_applies == 5

    def test_max_subspace(self):
        # max_subspace >= 2 * n_block
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=3, block="ph")
        with pytest.raises(ValueError):
            eigsh(self.matrix, guesses, n_ep=1, n_block=2, max_subspace=3)
        res = eigsh(self.matrix, guesses, n_ep=1, n_block=2, max_subspace=4)
        assert len(res.eigenvalues) == 1
        # max_subspace >= n_guesses
        with pytest.raises(ValueError):
            eigsh(self.matrix, guesses, n_ep=1, n_block=1, max_subspace=2)
        res = eigsh(self.matrix, guesses, n_ep=1, n_block=1, max_subspace=3)
        assert len(res.eigenvalues) == 1

    def test_adc0_singlet(self):
        # ensure that a diagonal matrix (with guesses from the diagonal)
        # converges in a single iteration to the correct eigenvalues
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc0", case="gen"
        )["singlet"]

        matrix = adcc.AdcMatrix(
            "adc0", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

        # Solve for singlets
        guesses = adcc.guesses_singlet(matrix, n_guesses=2, block="ph")
        res = jacobi_davidson(matrix, guesses, n_ep=2)

        assert res.converged
        assert res.n_iter == 1
        assert res.n_applies == 2

        ref_singlets = refdata["eigenvalues"]
        n_states = min(len(ref_singlets), len(res.eigenvalues))
        assert n_states > 1
        assert res.eigenvalues[:n_states] == pytest.approx(ref_singlets[:n_states])

    def test_adc2_singlets(self):
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["singlet"]

        # Solve for singlets
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=9, block="ph")
        res = jacobi_davidson(self.matrix, guesses, n_ep=9)

        ref_singlets = refdata["eigenvalues"]
        n_states = min(len(ref_singlets), len(res.eigenvalues))
        assert n_states > 1
        assert res.converged
        assert res.eigenvalues[:n_states] == pytest.approx(ref_singlets[:n_states])

    def test_adc2_triplets(self):
        refdata = testdata_cache.adcman_data(
            system="h2o_sto3g", method="adc2", case="gen"
        )["triplet"]

        # Solve for triplets
        guesses = adcc.guesses_triplet(self.matrix, n_guesses=10, block="ph")
        res = jacobi_davidson(self.matrix, guesses, n_ep=10)

        ref_triplets = refdata["eigenvalues"]
        n_states = min(len(ref_triplets), len(res.eigenvalues))
        assert n_states > 1
        assert res.converged
        assert res.eigenvalues[:n_states] == pytest.approx(ref_triplets[:n_states])
