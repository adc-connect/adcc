#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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
import unittest

import adcc
import numpy as np
import pytest
from adcc import AdcMatrix, AmplitudeVector, LazyMp
from adcc.functions import direct_sum, dot
from adcc.misc import assert_allclose_signfix, cached_property
from adcc.solver.davidson import jacobi_davidson
from adcc.solver.explicit_symmetrisation import IndexSymmetrisation
from adcc.solver.modified_davidson_diis import modified_davidson_diis
from adcc.solver.preconditioner import JacobiPreconditioner

from ..projection_test import assert_equal_symmetry
from ..testdata_cache import testdata_cache


class Adc2MatrixFolded(AdcMatrix):
    def __init__(self, matrix, omega=None):
        assert matrix.method.name == "adc2"
        super().__init__(matrix.method, matrix.ground_state,
                         block_orders=matrix.block_orders,
                         intermediates=matrix.intermediates,
                         diagonal_precomputed=matrix.diagonal())
        self.omega = omega
        self.isymm = IndexSymmetrisation(matrix)

    def get_doubles_amplitudes(self, v_1):
        diag = super().diagonal().pphh
        e = diag.ones_like()
        return (
            self.block_apply("pphh_ph", v_1) / (e * self.omega - diag)
        )  # .antisymmetrise(0, 1).antisymmetrise(2, 3).symmetrise([(0, 1), (2, 3)])

    def matvec(self, v):
        v_2 = self.get_doubles_amplitudes(v.ph)
        sigma_1 = self.block_apply("ph_ph", v.ph) + self.block_apply("ph_pphh", v_2)
        return AmplitudeVector(ph=sigma_1)

    def diagonal(self):
        # approxmiated as ADC(0) diagonal
        diag = AmplitudeVector(ph=direct_sum("a-i->ia",
                                             self.reference_state.fvv.diagonal(),
                                             self.reference_state.foo.diagonal()))
        return diag.evaluate()

    def block_view(self, block):
        raise NotImplementedError("Block-view not yet implemented for "
                                  "folded ADC(2) matrices.")

    def unfold(self, v):
        v_1_norm2 = dot(v.ph, v.ph)
        v_2 = self.get_doubles_amplitudes(v.ph)
        v_2_norm2 = dot(v_2, v_2)
        renorm_factor = np.sqrt(1 / (v_1_norm2 + v_2_norm2))
        v1_renorm = v.ph * renorm_factor
        v2_renorm = v_2 * renorm_factor
        return self.isymm.symmetrise(AmplitudeVector(ph=v1_renorm, pphh=v2_renorm))


class TestSolverModifiedDavidsonDiis(unittest.TestCase):
    @cached_property
    def matrix(self):
        return AdcMatrix(
            "adc2", LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        )

    @cached_property
    def matrix_folded(self):
        return Adc2MatrixFolded(self.matrix)

    def test_adc2_singlets(self):
        n_states = 8
        conv_tol = 1e-6
        atol = 1e-6

        guesses = adcc.guesses_singlet(self.matrix, n_guesses=n_states, block="ph")
        res = jacobi_davidson(self.matrix, guesses, n_ep=n_states,
                              conv_tol=conv_tol)

        matrix_adc1 = AdcMatrix("adc1", self.matrix.ground_state)
        guesses_adc1 = adcc.guesses_singlet(matrix_adc1, n_guesses=n_states,
                                            block="ph")
        res_adc1 = jacobi_davidson(matrix_adc1, guesses_adc1, n_ep=n_states)
        assert res_adc1.converged
        guess_vectors_folded = res_adc1.eigenvectors
        guess_omegas_folded = res_adc1.eigenvalues
        res_folded = modified_davidson_diis(self.matrix_folded,
                                            guess_vectors_folded,
                                            guess_omegas_folded, n_ep=n_states,
                                            conv_tol=conv_tol,
                                            conv_tol_davidson=1e-3,
                                            preconditioner=JacobiPreconditioner,
                                            max_iter_diis=500)
        assert res_folded.eigenvalues == pytest.approx(res.eigenvalues, abs=atol)
        for n in range(n_states):
            v1 = res.eigenvectors[n].ph.to_ndarray()
            v2 = res.eigenvectors[n].pphh.to_ndarray()
            self.matrix_folded.omega = res_folded.eigenvalues[n]
            v_folded = self.matrix_folded.unfold(res_folded.eigenvectors[n])
            v1_folded = v_folded.ph.to_ndarray()
            v2_folded = v_folded.pphh.to_ndarray()
            assert_allclose_signfix(v1, v1_folded, atol=atol)
            assert_allclose_signfix(v2, v2_folded, atol=atol)
            assert_equal_symmetry(res.eigenvectors[n].ph, v_folded.ph)
            assert_equal_symmetry(res.eigenvectors[n].pphh, v_folded.pphh)
