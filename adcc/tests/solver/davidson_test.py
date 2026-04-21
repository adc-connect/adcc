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
from adcc.AdcMatrix import Adc2MatrixFolded

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
        

class TestSolverDavidsonFolded(unittest.TestCase):
    @cached_property
    def matrix(self):
        return adcc.AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

    @cached_property
    def matrix_folded(self):
        return Adc2MatrixFolded(self.matrix)

    def test_adc2_singlets(self):
        import numpy as np
        # Solve for singlets
        n_states = 2
        guesses = adcc.guesses_singlet(self.matrix, n_guesses=n_states, block="ph")
        res = jacobi_davidson(self.matrix, guesses, n_ep=n_states)
        for n in range(n_states):
            print(np.sum(res.eigenvectors[n].ph.to_ndarray()*res.eigenvectors[n].ph.to_ndarray()))
        matrix_adc1 = adcc.AdcMatrix("adc1", self.matrix.ground_state)
        guesses_adc1 = adcc.guesses_singlet(matrix_adc1, n_guesses=n_states, block="ph")
        res_adc1 = jacobi_davidson(matrix_adc1, guesses_adc1, n_ep=n_states)
        assert res_adc1.converged
        guesses_folded = res_adc1.eigenvectors
        guesses_omegas_folded = res_adc1.eigenvalues
        res_folded = jacobi_davidson(self.matrix_folded, guesses_folded, n_ep=n_states, guess_omegas=guesses_omegas_folded)
        print(res.eigenvalues[:n_states])
        print(res_folded.eigenvalues[:n_states])
        assert res_folded.eigenvalues[:n_states] == pytest.approx(res.eigenvalues[:n_states], rel=1e-9)
        for n in range(n_states):
            print(f"====================== {n} ======================")
            v1 = res.eigenvectors[n].ph.to_ndarray()
            v2 = res.eigenvectors[n].pphh.to_ndarray()
            self.matrix_folded.omega = res_folded.eigenvalues[n]
            v_folded = self.matrix_folded.unfold(res_folded.eigenvectors[n])
            v1_folded = v_folded.ph.to_ndarray()
            v2_folded = v_folded.pphh.to_ndarray()
            print(v1)
            print(v1_folded)
            np.testing.assert_allclose(v1, -1.0 * v1_folded, atol=1e-9)
            np.testing.assert_allclose(v2, -1.0 * v2_folded, atol=1e-9)
            assert res.eigenvectors[n].ph.describe_symmetry() == v_folded.ph.describe_symmetry()
            print(res.eigenvectors[n].pphh.describe_symmetry())
            print(v_folded.pphh.describe_symmetry())
            assert res.eigenvectors[n].pphh.describe_symmetry() == v_folded.pphh.describe_symmetry()
