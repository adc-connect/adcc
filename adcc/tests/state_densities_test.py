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
import itertools
from collections.abc import Sequence

import numpy as np
import pytest
from pytest import approx

from adcc import ExcitedStates
from adcc.State2States import State2States

from . import testcases
from .testdata_cache import testdata_cache

methods = [
    ("adc0", None),
    ("adc1", None),
    ("adc2", None),
    ("adc2", 1),
    ("adc2", "1s"),
    ("adc2x", None),
    ("adc3", None),
    ("adc4", None),
]
methods_isr3: list[tuple[str, int | str | None]] = [
    ("adc3", 3),
    ("adc4", "3d"),
]
generators = ["adcman", "adcc"]

# There are only distinct density matrix implementations for the
# "gen" and "cvs" cases
# -> no need to test other cases (fc/fv)
# -> independent of state kind (singlet/triplet)
test_cases = testcases.get_by_filename(
    "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz", "hf_631g"
)
cases = [
    (case.file_name, c, kind)
    for case in test_cases
    for c in ["gen", "cvs"]
    if c in case.cases
    for kind in ["singlet", "any", "spin_flip"]
    if kind in case.kinds.pp
]
cases_no_cvs = [c for c in cases if "cvs" not in c[1]]
small_cases_no_cvs = [
    c for c in cases_no_cvs if not testcases.get_by_filename(c[0]).pop().only_full_mode
]


def _available(
    system: str, case: str, adc_method: str, isr_order: int | str | None, generator: str
) -> bool:
    """Whether the method is implemented and reference data available"""
    # - cvs-adc(0) is not available for adcman
    # - adc4: only small test cases
    # - adc4: no cvs-adc4 implemented in adcc and adcman
    # - isr3: no cvs-isr3 implemented in adcc and adcman
    # - adcman only implements isr1s and only calculates it for adc1
    return (
        not (adc_method == "adc0" and "cvs" in case and generator == "adcman")
        and not (adc_method == "adc4" and testcases.get_by_filename(system).pop().only_full_mode)
        and not (adc_method == "adc4" and "cvs" in case)
        and not (isr_order in (3, "3d") and "cvs" in case)
        and not (isr_order in (1, "1s") and adc_method != "adc1" and generator == "adcman")
    )


def _parametrize(
    cases: Sequence[tuple[str, str, str]],
    generators: Sequence[str],
    methods: Sequence[tuple[str, int | str | None]],
) -> pytest.MarkDecorator:
    return pytest.mark.parametrize(
        "system,case,kind,generator,adc_method,isr_order",
        [
            (system, case, kind, generator, adc_method, isr_order)
            for (system, case, kind), generator, (adc_method, isr_order) in itertools.product(
                cases, generators, methods
            )
            if _available(
                system=system,
                case=case,
                generator=generator,
                adc_method=adc_method,
                isr_order=isr_order,
            )
        ],
    )


class TestStateDensities:
    @_parametrize(cases=cases, generators=generators, methods=methods + methods_isr3)
    def test_state_diffdm(
        self,
        system: str,
        case: str,
        kind: str,
        adc_method: str,
        isr_order: int | str | None,
        generator: str,
    ):

        refdata = testdata_cache._load_data(
            system=system, method=adc_method, case=case, source=generator
        )[str(isr_order)][kind]

        # construct a ExcitedStates instance using the eigenvalues and eigenstates
        # from the reference data.
        state: ExcitedStates = getattr(testdata_cache, f"{generator}_states")(
            system=system, method=adc_method, case=case, kind=kind, isr_order=isr_order
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

    @_parametrize(cases=cases, generators=generators, methods=methods + methods_isr3)
    def test_ground_to_excited_tdm(
        self,
        system: str,
        case: str,
        kind: str,
        adc_method: str,
        isr_order: int | str | None,
        generator: str,
    ):

        refdata = testdata_cache._load_data(
            system=system, method=adc_method, case=case, source=generator
        )[str(isr_order)][kind]
        # construct a ExcitedStates instance using the eigenvalues and eigenstates
        # from the reference data.
        state: ExcitedStates = getattr(testdata_cache, f"{generator}_states")(
            system=system, method=adc_method, case=case, kind=kind, isr_order=isr_order
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
            assert dm_ao_a == approx(ref_dm_a)
            assert dm_ao_b == approx(ref_dm_b)

    # CVS state-to-state TDM is not implemented in adcc
    @_parametrize(cases=cases_no_cvs, generators=generators, methods=methods + methods_isr3)
    def test_state_to_state_tdm(
        self,
        system: str,
        case: str,
        kind: str,
        adc_method: str,
        isr_order: int | str | None,
        generator: str,
    ):

        refdata = testdata_cache._load_data(
            system=system, method=adc_method, case=case, source=generator
        )[str(isr_order)][kind]
        if len(refdata["eigenvalues"]) < 2:
            pytest.skip("Less than two states available.")
        s2s_data = refdata["state_to_state"]
        # construct a ExcitedStates instance using the eigenvalues and eigenstates
        # from the reference data.
        state: ExcitedStates = getattr(testdata_cache, f"{generator}_states")(
            system=system, method=adc_method, case=case, kind=kind, isr_order=isr_order
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

    @_parametrize(cases=small_cases_no_cvs, generators=["adcc"], methods=methods)
    def test_state_diffdm_mo_2p(
        self,
        system: str,
        case: str,
        kind: str,
        adc_method: str,
        isr_order: int | str | None,
        generator: str,
    ):
        # consistency test for the 2p diffdm in the MO basis
        refdata = testdata_cache._load_data(
            system=system, method=adc_method, case=case, source=generator
        )[str(isr_order)][kind]
        state: ExcitedStates = getattr(testdata_cache, f"{generator}_states")(
            system=system, method=adc_method, case=case, kind=kind, isr_order=isr_order
        )
        for i, ddm_2p in enumerate(state.state_diffdm_2p):
            assert state.excitation_energy[i] == refdata["eigenvalues"][i]
            ref = refdata["state_diffdm_2p"][f"state_{i}"]
            # ensure that correct blocks are populated
            assert sorted(ddm_2p.blocks_nonzero) == sorted(ref)
            for block in ddm_2p.blocks_nonzero:
                np.testing.assert_allclose(ddm_2p[block].to_ndarray(), ref[block], atol=1e-14)
