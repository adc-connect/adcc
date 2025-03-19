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
import itertools
import pytest
import numpy as np
from numpy.testing import assert_allclose

import adcc
import adcc.backends
from adcc.misc import assert_allclose_signfix

from .testing import cached_backend_hf
from .. import testcases

# molsturm is super slow
backends = [b for b in adcc.backends.available() if b != "molsturm"]
backends_with_gauge_origin = ["pyscf", "veloxchem"]

h2o = testcases.get_by_filename("h2o_sto3g", "h2o_def2tzvp")
h2o_cases = [(case.file_name, c) for case in h2o for c in case.cases]

methox = testcases.get_by_filename(
    "r2methyloxirane_sto3g", "r2methyloxirane_ccpvdz"
)
methox_cases = [(case.file_name, c) for case in methox for c in case.cases]

ch2nh2 = testcases.get_by_filename("ch2nh2_sto3g", "ch2nh2_ccpvdz")
ch2nh2_cases = [(case.file_name, c) for case in ch2nh2 for c in case.cases]


@pytest.mark.skipif(len(backends) < 2,
                    reason="Need at least two available backends for cross "
                    "reference test.")
class TestCrossReferenceBackends:

    @pytest.mark.parametrize("system,case", h2o_cases)
    def test_adc2_h2o(self, system, case):
        system = testcases.get_by_filename(system).pop()
        # Veloxchem does not support f-functions.
        # Define local variable to track which backends should be tested.
        if "veloxchem" in backends and system.basis == "def2-tzvp":
            backends_test = [b for b in backends if b != "veloxchem"]
        else:
            backends_test = [b for b in backends]
        if len(backends_test) < 2:
            pytest.skip("Veloxchem does not support f-functions. "
                        "Not enough backends available.")

        kwargs = {"n_singlets": 5}
        # fewer states available for fc-fv-cvs (4) and fv-cvs (5)
        if "fv" in case and "cvs" in case:
            kwargs["n_singlets"] = 3
            kwargs["n_guesses"] = 3
        elif "cvs" in case:
            # state 5 and 6 are degenerate -> can't compare the eigenvectors
            kwargs["n_singlets"] = 4

        method = "cvs-adc2" if "cvs" in case else "adc2"
        core_orbitals = system.core_orbitals if "cvs" in case else None
        frozen_core = system.frozen_core if "fc" in case else None
        frozen_virtual = system.frozen_virtual if "fv" in case else None

        results = {}
        for b in backends_test:
            scfres = cached_backend_hf(b, system, conv_tol=1e-10)
            results[b] = adcc.run_adc(
                scfres, method=method, conv_tol=1e-9,
                core_orbitals=core_orbitals, frozen_core=frozen_core,
                frozen_virtual=frozen_virtual, **kwargs
            )
            assert results[b].converged
        compare_adc_results(results, 5e-8)

    @pytest.mark.parametrize("system,case", methox_cases)
    def test_adc2_r2methyloxirane(self, system, case):
        system = testcases.get_by_filename(system).pop()
        # Veloxchem not available for (R)-2-Methyloxirane.
        # Define local variable to track which backends should be tested.
        if "veloxchem" in backends:
            backends_test = [b for b in backends if b != "veloxchem"]
        else:
            backends_test = [b for b in backends]
        if len(backends_test) < 2:
            pytest.skip("Veloxchem not available for (R)-2-Methyloxirane. "
                        "Not enough backends available.")
        method = "cvs-adc2" if "cvs" in case else "adc2"
        core_orbitals = system.core_orbitals if "cvs" in case else None
        frozen_core = system.frozen_core if "fc" in case else None
        frozen_virtual = system.frozen_virtual if "fv" in case else None

        results = {}
        for b in backends_test:
            scfres = cached_backend_hf(b, system, conv_tol=1e-10)
            results[b] = adcc.run_adc(
                scfres, method=method, n_singlets=3, conv_tol=1e-8,
                core_orbitals=core_orbitals, frozen_core=frozen_core,
                frozen_virtual=frozen_virtual
            )
            assert results[b].converged
        compare_adc_results(results, 5e-7)

    @pytest.mark.parametrize("system,case", ch2nh2_cases)
    def test_adc2_uhf_ch2nh2(self, system, case):
        system = testcases.get_by_filename(system).pop()
        method = "cvs-adc2" if "cvs" in case else "adc2"
        core_orbitals = system.core_orbitals if "cvs" in case else None
        frozen_core = system.frozen_core if "fc" in case else None
        frozen_virtual = system.frozen_virtual if "fv" in case else None

        results = {}
        for b in backends:
            scfres = cached_backend_hf(b, system, conv_tol=1e-10)
            results[b] = adcc.run_adc(
                scfres, method=method, n_states=5, conv_tol=1e-9,
                core_orbitals=core_orbitals, frozen_core=frozen_core,
                frozen_virtual=frozen_virtual
            )
            assert results[b].converged
        compare_adc_results(results, 5e-7)

    @pytest.mark.parametrize("system", h2o, ids=[case.file_name for case in h2o])
    def test_hf_properties(self, system: testcases.TestCase):
        # Veloxchem does not support f-functions.
        # Define local variable to track which backends should be tested.
        if "veloxchem" in backends and system.basis == "def2-tzvp":
            backends_test = [b for b in backends if b != "veloxchem"]
        else:
            backends_test = [b for b in backends]
        if len(backends_test) < 2:
            pytest.skip("Veloxchem does not support f-functions. "
                        "Not enough backends available.")
        results = {}
        for b in backends_test:
            results[b] = adcc.ReferenceState(cached_backend_hf(b, system))
        compare_hf_properties(results, 5e-9)


def compare_hf_properties(results, atol):
    for comb in itertools.combinations(results, r=2):
        refstate1 = results[comb[0]]
        refstate2 = results[comb[1]]

        # assert same gauge origins
        if len(set(comb) & set(backends_with_gauge_origin)) == 2:
            gauge_origins = ["origin", "mass_center", "charge_center"]
            for gauge_origin in gauge_origins:
                assert_allclose(refstate1.gauge_origin_to_xyz(gauge_origin),
                                refstate2.gauge_origin_to_xyz(gauge_origin),
                                atol=1e-5)

        assert_allclose(refstate1.nuclear_total_charge,
                        refstate2.nuclear_total_charge, atol=atol)
        assert_allclose(refstate1.nuclear_dipole,
                        refstate2.nuclear_dipole, atol=atol)

        if "electric_dipole" in refstate1.operators.available and \
           "electric_dipole" in refstate2.operators.available:
            assert_allclose(refstate1.dipole_moment,
                            refstate2.dipole_moment, atol=atol)


def compare_adc_results(adc_results, atol):
    for comb in itertools.combinations(adc_results, r=2):
        state1 = adc_results[comb[0]]
        state2 = adc_results[comb[1]]
        assert_allclose(
            state1.excitation_energy, state2.excitation_energy
        )
        # allow a deviation of 1 davidson iteration
        assert abs(state1.n_iter - state2.n_iter) <= 1

        blocks1 = state1.excitation_vector[0].blocks
        blocks2 = state2.excitation_vector[0].blocks
        assert blocks1 == blocks2
        for v1, v2 in zip(state1.excitation_vector, state2.excitation_vector):
            for block in blocks1:
                v1np = v1[block].to_ndarray()
                v2np = v2[block].to_ndarray()
                nonz_count1 = np.count_nonzero(np.abs(v1np) >= atol)
                if nonz_count1 == 0:
                    # Only zero elements in block.
                    continue

                assert_allclose(
                    np.abs(v1np), np.abs(v2np), atol=10 * atol,
                    err_msg="ADC vectors are not equal"
                            "in block {}".format(block)
                )

        def fix_signs(actual, desired, atol):
            fixed_signs = (
                np.sign(actual * (np.absolute(actual) > atol))
                * np.sign(desired * (np.absolute(desired) > atol))
            )
            for i in range(len(fixed_signs)):
                if 1 in fixed_signs[i]:
                    subs = 1
                elif -1 in fixed_signs[i]:
                    subs = -1
                else:
                    subs = np.nan
                # zero components are given the same sign as the others
                # if they are all zero, the sign is nan
                fixed_signs[i][fixed_signs[i] == 0] = subs
                assert (
                    np.all(fixed_signs[i] == 1)
                    or np.all(fixed_signs[i] == -1)
                    or np.all(np.isnan(fixed_signs[i]))
                )
            return fixed_signs

        def assert_allclose_signs(actual, desired):
            shape = actual.shape[1:]
            axis = tuple(range(1, len(shape) + 1))
            # the nan entries should not be compared
            idx_a = np.where(np.all(np.isnan(actual), axis=axis))
            actual_mod = actual.copy()
            for i in idx_a:
                actual_mod[i] = desired[i]
            idx_d = np.where(np.all(np.isnan(desired), axis=axis))
            desired_mod = desired.copy()
            for i in idx_d:
                desired_mod[i] = actual[i]
            np.testing.assert_allclose(actual_mod, desired_mod)

        fixed_signs = None
        # test properties
        # the signs of the transition moments can differ between different ADC
        # calculations (due to the eigenvectors), but this should be uniform for
        # all properties
        if "electric_dipole" in state1.operators.available and \
                "electric_dipole" in state2.operators.available:
            assert_allclose(state1.oscillator_strength,
                            state2.oscillator_strength, atol=atol)
            assert_allclose(state1.state_dipole_moment,
                            state2.state_dipole_moment, atol=atol)

            for i in range(len(state1.transition_dipole_moment)):
                assert_allclose_signfix(
                    state1.transition_dipole_moment[i],
                    state2.transition_dipole_moment[i],
                    atol=atol)
            fixed_signs_new = fix_signs(
                state1.transition_dipole_moment,
                state2.transition_dipole_moment,
                atol=atol
            )
            if fixed_signs is None:
                fixed_signs = fixed_signs_new
            else:
                assert_allclose_signs(fixed_signs_new, fixed_signs)

        if "electric_dipole_velocity" in state1.operators.available and \
                "electric_dipole_velocity" in state2.operators.available:
            assert_allclose(state1.oscillator_strength_velocity,
                            state2.oscillator_strength_velocity, atol=atol)
            for i in range(len(state1.transition_dipole_moment_velocity)):
                assert_allclose_signfix(
                    state1.transition_dipole_moment_velocity[i],
                    state2.transition_dipole_moment_velocity[i],
                    atol=atol)
            fixed_signs_new = fix_signs(
                state1.transition_dipole_moment_velocity,
                state2.transition_dipole_moment_velocity,
                atol=atol
            )
            if fixed_signs is None:
                fixed_signs = fixed_signs_new
            else:
                assert_allclose_signs(fixed_signs_new, fixed_signs)

        if "magnetic_dipole" in state1.operators.available and \
                "magnetic_dipole" in state2.operators.available:
            for i in range(len(state1.transition_magnetic_dipole_moment("origin"))):
                assert_allclose_signfix(
                    state1.transition_magnetic_dipole_moment("origin")[i],
                    state2.transition_magnetic_dipole_moment("origin")[i],
                    atol=atol)
            fixed_signs_new = fix_signs(
                state1.transition_magnetic_dipole_moment("origin"),
                state2.transition_magnetic_dipole_moment("origin"),
                atol=atol
            )
            if fixed_signs is None:
                fixed_signs = fixed_signs_new
            else:
                assert_allclose_signs(fixed_signs_new, fixed_signs)

        # Only in two backends the gauge origin selection is implemented.
        if len(set(comb) & set(backends_with_gauge_origin)) == 2:
            gauge_origins = ["origin", "mass_center", "charge_center"]
        else:
            gauge_origins = ["origin"]
        has_rotatory1 = all(op in state1.operators.available
                            for op
                            in ["magnetic_dipole", "electric_dipole_velocity"])
        has_rotatory2 = all(op in state2.operators.available
                            for op
                            in ["magnetic_dipole", "electric_dipole_velocity"])
        if has_rotatory1 and has_rotatory2:
            assert_allclose(state1.rotatory_strength,
                            state2.rotatory_strength,
                            atol=atol)

        has_rotatory_len1 = all(op in state1.operators.available
                                for op
                                in ["electric_dipole", "magnetic_dipole"])
        has_rotatory_len2 = all(op in state2.operators.available
                                for op
                                in ["electric_dipole", "magnetic_dipole"])
        if has_rotatory_len1 and has_rotatory_len2:
            for gauge_origin in gauge_origins:
                # reduce the tolerance criteria for mass_center as there are diff.
                # isotropic mass values implemented in veloxchem and pyscf.
                if gauge_origin == "mass_center":
                    atol_tdm_mag = 2e-4
                else:
                    atol_tdm_mag = atol
                assert_allclose(state1.rotatory_strength_length(gauge_origin),
                                state2.rotatory_strength_length(gauge_origin),
                                atol=atol_tdm_mag)
