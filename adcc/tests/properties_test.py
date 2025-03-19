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
import pytest
import numpy as np
from numpy.testing import assert_allclose

from adcc.State2States import State2States
from adcc.backends import run_hf
from adcc.misc import assert_allclose_signfix
from adcc import run_adc, AdcMethod

from .testdata_cache import testdata_cache
from . import testcases


# The density matrices are already tested in state_densities_test.py
# -> here we only want to test the contraction of the densities with
#    the operator matrices
# -> independent of method, case (gen/cvs/fc/fv) and kind (singlet/triplet)
# Actually, the tests should also be independent of the systems, because
# we only load some already tested density and contract it with some operator.
methods = ["adc2"]
generators = ["adcman", "adcc"]

test_cases = testcases.get_by_filename(
    "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz", "hf_631g"
)
cases = [(case.file_name, "gen", kind)
         for case in test_cases
         for kind in ["singlet", "any", "spin_flip"] if kind in case.kinds.pp]
gauge_origins = ["origin", "mass_center", "charge_center"]


@pytest.mark.parametrize("method", methods)
class TestProperties:
    @pytest.mark.parametrize("generator", generators)
    @pytest.mark.parametrize("system,case,kind", cases)
    def test_transition_dipole_moments(self, system: str, case: str, kind: str,
                                       method: str, generator: str):
        if "cvs" in case and AdcMethod(method).level == 0 and generator == "adcman":
            pytest.skip("No CVS-ADC(0) adcman reference data available.")

        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source=generator
        )[kind]
        state = testdata_cache._make_mock_adc_state(
            system=system, method=method, case=case, kind=kind, source=generator
        )

        res_tdms = state.transition_dipole_moment
        refevals = refdata["eigenvalues"]
        # should be zero by spin symmetry. adcman does not compute the transition
        # dipoles in this case
        if generator == "adcman" and kind == "triplet":
            ref_tdms = [np.array([0., 0., 0.]) for _ in range(len(refevals))]
        else:
            ref_tdms = refdata["transition_dipole_moments"]

        n_ref = len(state.excitation_vector)
        for i in range(n_ref):
            res_tdm = res_tdms[i]
            ref_tdm = ref_tdms[i]
            assert state.excitation_energy[i] == refevals[i]
            res_tdm_norm = np.sum(res_tdm * res_tdm)
            ref_tdm_norm = np.sum(ref_tdm * ref_tdm)
            assert res_tdm_norm == pytest.approx(ref_tdm_norm, abs=1e-5)
            assert_allclose_signfix(res_tdm, ref_tdm, atol=1e-5)

    @pytest.mark.parametrize("generator", generators)
    @pytest.mark.parametrize("system,case,kind", cases)
    def test_oscillator_strengths(self, system: str, case: str, kind: str,
                                  method: str, generator: str):
        if "cvs" in case and AdcMethod(method).level == 0 and generator == "adcman":
            pytest.skip("No CVS-ADC(0) adcman reference data available.")

        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source=generator
        )[kind]
        state = testdata_cache._make_mock_adc_state(
            system=system, method=method, case=case, kind=kind, source=generator
        )

        res_oscs = state.oscillator_strength
        refevals = refdata["eigenvalues"]
        # should be zero by spin symmetry. adcman does not compute the oscillator
        # strength in this case
        if generator == "adcman" and kind == "triplet":
            ref_tdms = [0 for _ in range(len(refevals))]
        else:
            ref_tdms = refdata["transition_dipole_moments"]

        n_ref = len(state.excitation_vector)
        for i in range(n_ref):
            assert state.excitation_energy[i] == refevals[i]
            ref_tdm_norm = np.sum(ref_tdms[i] * ref_tdms[i])
            assert (
                res_oscs[i] == pytest.approx(2. / 3. * ref_tdm_norm * refevals[i])
            )

    @pytest.mark.parametrize("generator", generators)
    @pytest.mark.parametrize("system,case,kind", cases)
    def test_state_dipole_moments(self, system: str, case: str, kind: str,
                                  method: str, generator: str):
        if "cvs" in case and AdcMethod(method).level == 0 and generator == "adcman":
            pytest.skip("No CVS-ADC(0) adcman reference data available.")

        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source=generator
        )[kind]
        state = testdata_cache._make_mock_adc_state(
            system=system, method=method, case=case, kind=kind, source=generator
        )

        res_dms = state.state_dipole_moment
        n_ref = len(state.excitation_vector)
        assert_allclose(res_dms, refdata["state_dipole_moments"][:n_ref], atol=1e-4)

    # CVS-ADC state2state tdm not implemented
    @pytest.mark.parametrize("generator", generators)
    @pytest.mark.parametrize("system,case,kind", [c for c in cases
                                                  if "cvs" not in c[1]])
    def test_state2state_transition_dipole_moments(self, system: str, case: str,
                                                   kind: str, method: str,
                                                   generator: str):
        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source=generator
        )[kind]
        state = testdata_cache._make_mock_adc_state(
            system=system, method=method, case=case, kind=kind, source=generator
        )

        refevals = refdata["eigenvalues"]
        if len(refevals) < 2:
            pytest.skip("Less than two states available.")

        state_to_state = refdata["state_to_state"]
        for i in range(len(state.excitation_vector) - 1):
            assert state.excitation_energy[i] == refevals[i]
            fromi_ref = state_to_state[f"from_{i}"]["transition_dipole_moments"]

            state2state = State2States(state, initial=i)
            for ii, j in enumerate(range(i + 1, state.size)):
                assert state.excitation_energy[j] == refevals[j]
                assert_allclose_signfix(state2state.transition_dipole_moment[ii],
                                        fromi_ref[ii], atol=1e-4)

    @pytest.mark.parametrize("case", ["gen", "cvs"])
    def test_magnetic_transition_dipole_moments_z_component(self, method: str,
                                                            case: str):
        backend = ""
        xyz = """
            C 0 0 0
            O 0 0 2.7023
        """
        basis = "sto-3g"
        scfres = run_hf(backend, xyz, basis)

        if "cvs" in case:
            if "cvs" not in method:
                method = f"cvs-{method}"
            state = run_adc(scfres, method=method, n_singlets=5, core_orbitals=2)
        else:
            state = run_adc(scfres, method=method, n_singlets=10)
        tdms = state.transition_magnetic_dipole_moment("origin")

        # For linear molecules lying on the z-axis, the z-component must be zero
        for tdm in tdms:
            assert tdm[2] < 1e-10

    # Only adcc reference data available.
    @pytest.mark.parametrize("system,case,kind", cases)
    def test_magnetic_transition_dipole_moments(self, system: str, case: str,
                                                kind: str, method: str):
        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source="adcc"
        )[kind]
        state = testdata_cache._make_mock_adc_state(
            system=system, method=method, case=case, kind=kind, source="adcc"
        )

        n_ref = len(state.excitation_vector)
        for g_origin in gauge_origins:
            res_dms = state.transition_magnetic_dipole_moment(g_origin)
            for i in range(n_ref):
                assert_allclose_signfix(
                    res_dms[i],
                    refdata[f"transition_magnetic_dipole_moments_{g_origin}"][i],
                    atol=1e-4
                )

    # Only adcc reference data available.
    @pytest.mark.parametrize("system,case,kind", cases)
    def test_transition_dipole_moments_velocity(self, system: str, case: str,
                                                kind: str, method: str):
        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source="adcc"
        )[kind]
        state = testdata_cache._make_mock_adc_state(
            system=system, method=method, case=case, kind=kind, source="adcc"
        )

        res_dms = state.transition_dipole_moment_velocity
        n_ref = len(state.excitation_vector)
        for i in range(n_ref):
            assert_allclose_signfix(
                res_dms[i], refdata["transition_dipole_moments_velocity"][i],
                atol=1e-4
            )

    # Only adcc reference data available.
    @pytest.mark.parametrize("system,case,kind", cases)
    def test_transition_quadrupole_moments(self, system: str, case: str,
                                           kind: str, method: str):
        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source="adcc"
        )[kind]
        state = testdata_cache._make_mock_adc_state(
            system=system, method=method, case=case, kind=kind, source="adcc"
        )

        n_ref = len(state.excitation_vector)
        for g_origin in gauge_origins:
            res_dms = state.transition_quadrupole_moment(g_origin)
            for i in range(n_ref):
                assert_allclose_signfix(
                    res_dms[i],
                    refdata[f"transition_quadrupole_moments_{g_origin}"][i],
                    atol=1e-4
                )

    # Only adcc reference data available.
    @pytest.mark.parametrize("system,case,kind", cases)
    def test_rotatory_strengths(self, system: str, case: str, kind: str,
                                method: str):
        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source="adcc"
        )[kind]
        state = testdata_cache._make_mock_adc_state(
            system=system, method=method, case=case, kind=kind, source="adcc"
        )

        res_rots = state.rotatory_strength
        ref_tmdm = refdata["transition_magnetic_dipole_moments_origin"]
        ref_tdmvel = refdata["transition_dipole_moments_velocity"]
        refevals = refdata["eigenvalues"]
        n_ref = len(state.excitation_vector)
        for i in range(n_ref):
            assert state.excitation_energy[i] == refevals[i]
            ref_dot = np.dot(ref_tmdm[i], ref_tdmvel[i])
            assert res_rots[i] == pytest.approx(ref_dot / refevals[i])
