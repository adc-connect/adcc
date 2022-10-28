#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2022 by the adcc authors
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
import itertools
import numpy as np

from adcc.IsrMatrix import IsrMatrix
from adcc.testdata.cache import cache
from adcc.OneParticleOperator import product_trace
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.misc import expand_test_templates


testcases = [("h2o_sto3g", "singlet"), ("cn_sto3g", "state")]
methods = ["adc0", "adc1", "adc2"]
operator_kinds = ["electric", "magnetic"]


@expand_test_templates(list(itertools.product(testcases, methods, operator_kinds)))
class TestIsrMatrix(unittest.TestCase):
    def template_matrix_vector_product(self, case, method, op_kind):
        (system, kind) = case
        state = cache.adc_states[system][method][kind]
        mp = state.ground_state
        if op_kind == "electric":  # example of a symmetric operator
            dips = state.reference_state.operators.electric_dipole
        elif op_kind == "magnetic":  # example of an asymmetric operator
            dips = state.reference_state.operators.magnetic_dipole
        else:
            raise NotImplementedError(
                "Tests are only implemented for"
                "electric and magnetic dipole operators."
            )

        # computing Y_m @ B @ Y_n yields the state-to-state
        # transition dipole moments (n->m) (for n not equal to m)
        # they can either be obtained using the matvec method of the IsrMatrix
        # class or via the state-to-state transition density matrices
        # (the second method serves as a reference here)

        matrix = IsrMatrix(method, mp, dips)
        for excitation1 in state.excitations:
            resv = matrix @ excitation1.excitation_vector
            for excitation2 in state.excitations:
                s2s_tdm = [excitation2.excitation_vector @ v for v in resv]
                tdm = state2state_transition_dm(
                    state.property_method, mp,
                    excitation1.excitation_vector,
                    excitation2.excitation_vector,
                    state.matrix.intermediates
                )
                s2s_tdm_ref = np.array([product_trace(tdm, dip) for dip in dips])
                np.testing.assert_allclose(s2s_tdm, s2s_tdm_ref, atol=1e-12)


class TestIsrMatrixInterface(unittest.TestCase):
    def test_matvec(self):
        system = "h2o_sto3g"
        method = "adc2"
        kind = "singlet"

        state = cache.adc_states[system][method][kind]
        mp = state.ground_state
        dips = state.reference_state.operators.electric_dipole
        magdips = state.reference_state.operators.magnetic_dipole
        vecs = [exc.excitation_vector for exc in state.excitations[:2]]

        matrix_ref = IsrMatrix(method, mp, dips)
        refv1, refv2 = matrix_ref @ vecs

        # multiple vectors
        for i, dip in enumerate(dips):
            matrix = IsrMatrix(method, mp, dip)
            resv1, resv2 = matrix @ vecs
            diffs = [refv1[i] - resv1, refv2[i] - resv2]
            for j in range(2):
                assert diffs[j].ph.dot(diffs[j].ph) < 1e-12
                assert diffs[j].pphh.dot(diffs[j].pphh) < 1e-12

        # compute Y_n @ B @ Y_n with matvec and rmatvec
        # symmetric operators
        for vec1 in vecs:
            for vec2 in vecs:
                resl = [vec1 @ mvprod for mvprod in matrix_ref.matvec(vec2)]
                resr = [mvprod @ vec2 for mvprod in matrix_ref.rmatvec(vec1)]
                np.testing.assert_allclose(
                    np.array(resl), np.array(resr), atol=1e-12
                )

        # anti-symmetric operators
        magmatrix = IsrMatrix(method, mp, magdips)
        for vec1 in vecs:
            for vec2 in vecs:
                resl = [vec1 @ mvprod for mvprod in magmatrix.matvec(vec2)]
                resr = [mvprod @ vec2 for mvprod in magmatrix.rmatvec(vec1)]
                np.testing.assert_allclose(
                    np.array(resl), np.array(resr), atol=1e-12
                )
