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
from pytest import approx

from adcc import ExcitedStates, AdcMethod
from adcc.State2States import State2States

from .testdata_cache import testdata_cache
from . import testcases


methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]
generators = ["adcman", "adcc"]


# There are only distinct density matrix implementations for the
# "gen" and "cvs" cases
# -> no need to test other cases (fc/fv)
# -> independent of state kind (singlet/triplet)
test_cases = testcases.get_by_filename(
    "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz", "hf_631g"
)
cases = [(case.file_name, c, kind)
         for case in test_cases
         for c in ["gen", "cvs"] if c in case.cases
         for kind in ["singlet", "any", "spin_flip"] if kind in case.kinds.pp]


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("generator", generators)
class TestStateDensities:
    @pytest.mark.parametrize("system,case,kind", cases)
    def test_state_diffdm(self, system: str, case: str, kind: str, method: str,
                          generator: str):
        if "cvs" in case and AdcMethod(method).level == 0 and generator == "adcman":
            pytest.skip("No CVS-ADC(0) adcman reference data available.")
        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source=generator
        )[kind]
        # construct a ExcitedStates instance using the eigenvalues and eigenstates
        # from the reference data.
        state: ExcitedStates = getattr(testdata_cache, f"{generator}_states")(
            system=system, method=method, case=case, kind=kind
        )
        # since refdata was used to build state we have to have the same amount
        # of states
        for i in range(len(state.excitation_vector)):
            # Check that we are talking about the same state when
            # comparing reference and computed
            assert state.excitation_energy[i] == refdata["eigenvalues"][i]

            dm_ao_a, dm_ao_b = state.state_diffdm[i].to_ao_basis()
            assert dm_ao_a.to_ndarray() == approx(refdata["state_diffdm_bb_a"][i])
            assert dm_ao_b.to_ndarray() == approx(refdata["state_diffdm_bb_b"][i])

    # adcman does not compute the tdm for singlet -> triplet transitions,
    # because the transition dipole moment should be zero anyway
    # -> remove triplet tests
    @pytest.mark.parametrize("system,case,kind",
                             [c for c in cases if c[2] != "triplet"])
    def test_ground_to_excited_tdm(self, system: str, case: str, kind: str,
                                   method: str, generator: str):
        if "cvs" in case and AdcMethod(method).level == 0 and generator == "adcman":
            pytest.skip("No CVS-ADC(0) adcman reference data available.")
        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source=generator
        )[kind]
        # construct a ExcitedStates instance using the eigenvalues and eigenstates
        # from the reference data.
        state: ExcitedStates = getattr(testdata_cache, f"{generator}_states")(
            system=system, method=method, case=case, kind=kind
        )
        # since refdata was used to build state we have to have the same amount
        # of states
        for i in range(len(state.excitation_vector)):
            # Check that we are talking about the same state when
            # comparing reference and computed
            assert state.excitation_energy[i] == refdata["eigenvalues"][i]

            dm_ao_a, dm_ao_b = state.transition_dm[i].to_ao_basis()
            dm_ao_a, dm_ao_b = dm_ao_a.to_ndarray(), dm_ao_b.to_ndarray()
            ref_dm_a = refdata["ground_to_excited_tdm_bb_a"][i]
            ref_dm_b = refdata["ground_to_excited_tdm_bb_b"][i]
            assert (dm_ao_a == approx(ref_dm_a))
            assert (dm_ao_b == approx(ref_dm_b))

    # CVS state-to-state TDM is not implemented in adcc
    @pytest.mark.parametrize("system,case,kind",
                             [c for c in cases if "cvs" not in c[1]])
    def test_state_to_state_tdm(self, system: str, case: str, kind: str,
                                method: str, generator: str):

        refdata = testdata_cache._load_data(
            system=system, method=method, case=case, source=generator
        )[kind]
        if len(refdata["eigenvalues"]) < 2:
            pytest.skip("Less than two states available.")
        s2s_data = refdata["state_to_state"]

        # construct a ExcitedStates instance using the eigenvalues and eigenstates
        # from the reference data.
        state: ExcitedStates = getattr(testdata_cache, f"{generator}_states")(
            system=system, method=method, case=case, kind=kind
        )
        # since refdata was used to build state we have to have the same amount
        # of states
        for i in range(len(state.excitation_vector) - 1):
            # Check that we are talking about the same state when
            # comparing reference and computed
            assert state.excitation_energy[i] == refdata["eigenvalues"][i]
            fromi_ref_a = s2s_data[f"from_{i}"]["state_to_excited_tdm_bb_a"]
            fromi_ref_b = s2s_data[f"from_{i}"]["state_to_excited_tdm_bb_b"]

            state_to_state = State2States(state, initial=i)
            for j, (ref_a, ref_b) in enumerate(zip(fromi_ref_a, fromi_ref_b)):
                ito = i + j + 1
                assert state.excitation_energy[ito] == refdata["eigenvalues"][ito]
                ref_energy = refdata["eigenvalues"][ito] - refdata["eigenvalues"][i]
                assert state_to_state.excitation_energy[j] == ref_energy
                dm_ao_a, dm_ao_b = state_to_state.transition_dm[j].to_ao_basis()
                np.testing.assert_allclose(dm_ao_a.to_ndarray(), ref_a, atol=1e-4)
                np.testing.assert_allclose(dm_ao_b.to_ndarray(), ref_b, atol=1e-4)
