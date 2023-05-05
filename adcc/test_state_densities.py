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
import unittest
import numpy as np

from adcc.State2States import State2States
from adcc.testdata.cache import cache

from .misc import expand_test_templates
from pytest import approx, skip

# The methods to test
basemethods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]
methods = [m for bm in basemethods for m in [bm, "cvs_" + bm]]


@expand_test_templates(methods)
class Runners():
    def base_test(self, *args, **kwargs):
        raise NotImplementedError

    def template_h2o_sto3g_singlet(self, method):
        self.base_test("h2o_sto3g", method, "singlet")

    def template_h2o_def2tzvp_singlet(self, method):
        self.base_test("h2o_def2tzvp", method, "singlet")

    def template_h2o_sto3g_triplet(self, method):
        self.base_test("h2o_sto3g", method, "triplet")

    def template_h2o_def2tzvp_triplet(self, method):
        self.base_test("h2o_def2tzvp", method, "triplet")

    def template_cn_sto3g(self, method):
        self.base_test("cn_sto3g", method, "state")

    def template_cn_ccpvdz(self, method):
        self.base_test("cn_ccpvdz", method, "state")

    def template_hf3_631g_spin_flip(self, method):
        self.base_test("hf3_631g", method, "spin_flip")

    #
    # Other runners (to test that FC and FV work as they should)
    #
    def test_h2o_sto3g_fc_adc2_singlets(self):
        self.base_test("h2o_sto3g", "fc_adc2", "singlet")

    def test_h2o_sto3g_fv_adc2x_singlets(self):
        self.base_test("h2o_sto3g", "fv_adc2x", "singlet")

    def test_cn_sto3g_fc_adc2_states(self):
        self.base_test("cn_sto3g", "fc_adc2", "state")

    def test_cn_sto3g_fv_adc2x_states(self):
        self.base_test("cn_sto3g", "fv_adc2x", "state")

    def test_cn_sto3g_fv_cvs_adc2x_states(self):
        self.base_test("cn_sto3g", "fv_cvs_adc2x", "state")

    def test_h2s_sto3g_fc_cvs_adc2_singlets(self):
        self.base_test("h2s_sto3g", "fc_cvs_adc2", "singlet")

    def test_h2s_6311g_fc_adc2_singlets(self):
        self.base_test("h2s_6311g", "fc_adc2", "singlet")

    def test_h2s_6311g_fv_adc2_singlets(self):
        self.base_test("h2s_6311g", "fv_adc2", "singlet")

    def test_h2s_6311g_fc_cvs_adc2x_singlets(self):
        self.base_test("h2s_6311g", "fc_cvs_adc2x", "singlet")

    def test_h2s_6311g_fv_cvs_adc2x_singlets(self):
        self.base_test("h2s_6311g", "fv_cvs_adc2x", "singlet")


# Return combinations not tested so far:
#     The rationale is that cvs-spin-flip as a method do not make
#     that much sense and probably the routines are anyway covered
#     by the other testing we do.
delattr(Runners, "test_hf3_631g_spin_flip_cvs_adc0")
delattr(Runners, "test_hf3_631g_spin_flip_cvs_adc1")
delattr(Runners, "test_hf3_631g_spin_flip_cvs_adc2")
delattr(Runners, "test_hf3_631g_spin_flip_cvs_adc2x")
delattr(Runners, "test_hf3_631g_spin_flip_cvs_adc3")


class TestStateDiffDm(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        refdens_a = refdata[method][kind]["state_diffdm_bb_a"]
        refdens_b = refdata[method][kind]["state_diffdm_bb_b"]
        refevals = refdata[method][kind]["eigenvalues"]
        for i in range(len(state.excitation_vector)):
            # Check that we are talking about the same state when
            # comparing reference and computed
            assert state.excitation_energy[i] == refevals[i]

            dm_ao_a, dm_ao_b = state.state_diffdm[i].to_ao_basis()
            assert dm_ao_a.to_ndarray() == approx(refdens_a[i])
            assert dm_ao_b.to_ndarray() == approx(refdens_b[i])


# For the ground-to-excited state tdm, only consistency tests are
# performed due to the changes in transition_dm.py
class TestStateGroundToExcitedTdm(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        kind = "any" if kind == "state" else kind
        refdata = cache.adcc_reference_data[system]
        state = cache.adcc_states[system][method][kind]

        refdens_a = refdata[method][kind]["ground_to_excited_tdm_bb_a"]
        refdens_b = refdata[method][kind]["ground_to_excited_tdm_bb_b"]
        refevals = refdata[method][kind]["eigenvalues"]
        for i in range(len(state.excitation_vector)):
            # Check that we are talking about the same state when
            # comparing reference and computed
            assert state.excitation_energy[i] == refevals[i]

            dm_ao_a, dm_ao_b = state.transition_dm[i].to_ao_basis()
            assert dm_ao_a.to_ndarray() == approx(refdens_a[i])
            assert dm_ao_b.to_ndarray() == approx(refdens_b[i])


class TestStateExcitedToExcitedTdm(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")
        if "cvs" in method:
            skip("State-to-state transition dms not yet implemented for CVS.")
        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]
        state_to_state = refdata[method][kind]["state_to_state"]
        refevals = refdata[method][kind]["eigenvalues"]

        for i, exci in enumerate(state.excitations):
            # Check that we are talking about the same state when
            # comparing reference and computed
            assert exci.excitation_energy == refevals[i]
            fromi_ref_a = state_to_state[f"from_{i}"]["state_to_excited_tdm_bb_a"]
            fromi_ref_b = state_to_state[f"from_{i}"]["state_to_excited_tdm_bb_b"]

            state2state = State2States(state, initial=i)
            for ii, j in enumerate(range(i + 1, state.size)):
                assert state.excitation_energy[j] == refevals[j]
                ee_ref = refevals[j] - refevals[i]
                assert state2state.excitation_energy[ii] == ee_ref
                dm_ao_a, dm_ao_b = state2state.transition_dm[ii].to_ao_basis()
                np.testing.assert_allclose(fromi_ref_a[ii],
                                           dm_ao_a.to_ndarray(), atol=1e-4)
                np.testing.assert_allclose(fromi_ref_b[ii],
                                           dm_ao_b.to_ndarray(), atol=1e-4)
