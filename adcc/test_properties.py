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

from numpy.testing import assert_allclose

from adcc.State2States import State2States
from adcc.testdata.cache import cache
from adcc.backends import run_hf
from adcc import run_adc

from .misc import assert_allclose_signfix, expand_test_templates
from .test_state_densities import Runners

from pytest import approx, skip


class TestTransitionDipoleMoments(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        # For CVS, only consistency tests are performed because
        # the reference data are erroneous
        if "cvs" in method:
            kind = "any" if kind == "state" else kind
            refdata = cache.adcc_reference_data[system]
            state = cache.adcc_states[system][method][kind]

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

        # For CVS, only consistency tests are performed because
        # the reference data are erroneous
        if "cvs" in method:
            kind = "any" if kind == "state" else kind
            refdata = cache.adcc_reference_data[system]
            state = cache.adcc_states[system][method][kind]

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


basemethods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]
methods = [m for bm in basemethods for m in [bm, "cvs_" + bm]]


@expand_test_templates(methods)
class TestMagneticTransitionDipoleMoments(unittest.TestCase):
    def template_linear_molecule(self, method):
        method = method.replace("_", "-")
        backend = ""
        xyz = """
            C 0 0 0
            O 0 0 2.7023
        """
        basis = "sto-3g"
        scfres = run_hf(backend, xyz, basis)

        if "cvs" in method:
            state = run_adc(scfres, method=method, n_singlets=5, core_orbitals=2)
        else:
            state = run_adc(scfres, method=method, n_singlets=10)
        tdms = state.transition_magnetic_dipole_moment

        # For linear molecules lying on the z-axis, the z-component must be zero
        for tdm in tdms:
            assert tdm[2] < 1e-10
