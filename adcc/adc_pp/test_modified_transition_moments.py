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
import unittest
import itertools
import numpy as np

from .modified_transition_moments import modified_transition_moments

from adcc.misc import expand_test_templates
from adcc.testdata.cache import cache

from pytest import skip, approx

# The methods to test
basemethods = ["adc0", "adc1", "adc2"]
methods = [m for bm in basemethods for m in [bm, "cvs-" + bm]]

operator_kinds = ["electric", "magnetic"]


@expand_test_templates(list(itertools.product(methods, operator_kinds)))
class TestModifiedTransitionMoments(unittest.TestCase):
    def base_test(self, system, method, kind, op_kind):
        state = cache.adcc_states[system][method][kind]
        ref = cache.adcc_reference_data[system][method][kind]
        n_ref = len(state.excitation_vector)

        if op_kind == "electric":
            dips = state.reference_state.operators.electric_dipole
            ref_tdm = ref["transition_dipole_moments"]
        elif op_kind == "magnetic":
            dips = state.reference_state.operators.magnetic_dipole
            ref_tdm = ref["transition_magnetic_dipole_moments"]
        else:
            skip("Tests are only implemented for electric "
                 "and magnetic dipole operators.")

        mtms = modified_transition_moments(method, state.ground_state, dips)

        for i in range(n_ref):
            # Computing the scalar product of the eigenvector
            # and the modified transition moments yields
            # the transition dipole moment (doi.org/10.1063/1.1752875)
            excivec = state.excitation_vector[i]
            res_tdm = np.array([excivec @ mtms[i] for i in range(3)])

            # Test norm and actual values
            res_tdm_norm = np.sum(res_tdm * res_tdm)
            ref_tdm_norm = np.sum(ref_tdm[i] * ref_tdm[i])
            assert res_tdm_norm == approx(ref_tdm_norm, abs=1e-8)
            np.testing.assert_allclose(res_tdm, ref_tdm[i], atol=1e-8)

    #
    # General
    #
    def template_h2o_sto3g_singlets(self, method, op_kind):
        self.base_test("h2o_sto3g", method, "singlet", op_kind)

    def template_h2o_def2tzvp_singlets(self, method, op_kind):
        self.base_test("h2o_def2tzvp", method, "singlet", op_kind)

    def template_h2o_sto3g_triplets(self, method, op_kind):
        self.base_test("h2o_sto3g", method, "triplet", op_kind)

    def template_h2o_def2tzvp_triplets(self, method, op_kind):
        self.base_test("h2o_def2tzvp", method, "triplet", op_kind)

    def template_cn_sto3g(self, method, op_kind):
        self.base_test("cn_sto3g", method, "any", op_kind)

    def template_cn_ccpvdz(self, method, op_kind):
        self.base_test("cn_ccpvdz", method, "any", op_kind)
