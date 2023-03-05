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
# import adcc

from numpy.testing import assert_allclose

from adcc.State2States import State2States
from adcc.testdata.cache import cache

from .misc import assert_allclose_signfix
from .test_state_densities import Runners

from pytest import approx, skip


class TestTransitionDipoleMoments(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        # if method == 'adc3':
        #    state._property_method = state.method.at_level(3)

        res_tdms = state.transition_dipole_moment
        ref_tdms = refdata[method][kind]["transition_dipole_moments"]
        refevals = refdata[method][kind]["eigenvalues"]
        n_ref = len(state.excitation_vector)
        for i in range(n_ref):
            res_tdm = res_tdms[i]
            ref_tdm = ref_tdms[i]
            assert state.excitation_energy[i] == refevals[i]
            res_tdm_norm = np.sum(res_tdm * res_tdm)
            ref_tdm_norm = np.sum(ref_tdm * ref_tdm)
            assert res_tdm_norm == approx(ref_tdm_norm, abs=1e-5)
            assert_allclose_signfix(res_tdm, ref_tdm, atol=1e-5)


class TestOscillatorStrengths(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        # if method == 'adc3':
        #    n_ref = len(state.excitation_vector)
        #    refstate = adcc.ReferenceState(cache.hfdata[system])
        #    if kind == 'singlet':
        #        state2 = adcc.adc3(refstate, n_singlets = n_ref,
        #                           properties_level='adc3', conv_tol = 1e-8)
        #    elif kind == 'triplet':
        #        state2 = adcc.adc3(refstate, n_triplets = n_ref,
        #                           properties_level='adc3', conv_tol = 1e-8)
        #    elif kind == 'state':
        #        state2 = adcc.adc3(refstate, n_states = n_ref,
        #                           properties_level='adc3', conv_tol = 1e-8)
        #    elif kind == 'spin_flip':
        #        state2 = adcc.adc3(refstate, n_spin_flip = n_ref,
        #                           properties_level='adc3', conv_tol = 1e-8)

        #    res_oscs = state2.oscillator_strength
        #    ref_tdms = refdata[method][kind]["transition_dipole_moments"]
        #    refevals = refdata[method][kind]["eigenvalues"]
        #    n_ref = len(state2.excitation_vector)
        #    for i in range(n_ref):
        #        assert state2.excitation_energy[i] == refevals[i]
        #        ref_tdm_norm = np.sum(ref_tdms[i] * ref_tdms[i])
        #        assert res_oscs[i] == approx(2. / 3. * ref_tdm_norm * refevals[i])
        # else:
        res_oscs = state.oscillator_strength
        ref_tdms = refdata[method][kind]["transition_dipole_moments"]
        refevals = refdata[method][kind]["eigenvalues"]
        n_ref = len(state.excitation_vector)
        for i in range(n_ref):
            assert state.excitation_energy[i] == refevals[i]
            ref_tdm_norm = np.sum(ref_tdms[i] * ref_tdms[i])
            assert res_oscs[i] == approx(2. / 3. * ref_tdm_norm * refevals[i])


class TestStateDipoleMoments(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        # if method == 'adc3':
        #    state._property_method = state.method.at_level(3)

        res_dms = state.state_dipole_moment
        ref = refdata[method][kind]
        n_ref = len(state.excitation_vector)
        assert_allclose(res_dms, ref["state_dipole_moments"][:n_ref], atol=1e-4)


class TestState2StateTransitionDipoleMoments(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")
        if "cvs" in method:
            skip("State-to-state transition dms not yet implemented for CVS.")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        # if method == 'adc3':
        #    state._property_method = state.method.at_level(3)

        state_to_state = refdata[method][kind]["state_to_state"]
        refevals = refdata[method][kind]["eigenvalues"]
        for i, exci in enumerate(state.excitations):
            assert exci.excitation_energy == refevals[i]
            fromi_ref = state_to_state[f"from_{i}"]["transition_dipole_moments"]

            state2state = State2States(state, initial=i)
            for ii, j in enumerate(range(i + 1, state.size)):
                assert state.excitation_energy[j] == refevals[j]
                assert_allclose_signfix(state2state.transition_dipole_moment[ii],
                                        fromi_ref[ii], atol=1e-4)
