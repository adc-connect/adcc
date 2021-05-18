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
import numpy as np

from .modified_transition_moments import modified_transition_moments

from adcc.misc import expand_test_templates
from adcc.testdata.cache import cache

from pytest import approx

# The methods to test
basemethods = ["adc0", "adc1", "adc2"]
methods = [m for bm in basemethods for m in [bm, "cvs-" + bm]]


@expand_test_templates(methods)
class TestModifiedTransitionMoments(unittest.TestCase):
    def base_test(self, system, method, kind):
        state = cache.adc_states[system][method][kind]
        ref = cache.reference_data[system][method][kind]
        n_ref = len(state.excitation_vector)

        mtms = modified_transition_moments(method, state.ground_state)
        for i in range(n_ref):
            # computing the scalar product of the eigenvector
            # and the modified transition moments yields
            # the transition dipole moment (doi.org/10.1063/1.1752875)
            excivec = state.excitation_vector[i]
            res_tdm = -1.0 * np.array([excivec @ mtms[i] for i in range(3)])
            ref_tdm = ref["transition_dipole_moments"][i]

            # Test norm and actual values
            res_tdm_norm = np.sum(res_tdm * res_tdm)
            ref_tdm_norm = np.sum(ref_tdm * ref_tdm)
            assert res_tdm_norm == approx(ref_tdm_norm, abs=1e-8)
            np.testing.assert_allclose(res_tdm, ref_tdm, atol=1e-8)

    #
    # General
    #
    def template_h2o_sto3g_singlets(self, method):
        self.base_test("h2o_sto3g", method, "singlet")

    def template_h2o_def2tzvp_singlets(self, method):
        self.base_test("h2o_def2tzvp", method, "singlet")

    def template_h2o_sto3g_triplets(self, method):
        self.base_test("h2o_sto3g", method, "triplet")

    def template_h2o_def2tzvp_triplets(self, method):
        self.base_test("h2o_def2tzvp", method, "triplet")

    def template_cn_sto3g(self, method):
        self.base_test("cn_sto3g", method, "state")

    def template_cn_ccpvdz(self, method):
        self.base_test("cn_ccpvdz", method, "state")
