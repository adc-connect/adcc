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
import adcc
import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import approx

from adcc.misc import assert_allclose_signfix
from adcc import ExcitedStates

from . import testcases
from .testdata_cache import testdata_cache

# The methods to test
methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]

# The different reference data that we test against
generators = ["adcman", "adcc"]

test_cases = testcases.get_by_filename(
    "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz", "hf_631g"
)
cases = [
    (case.file_name, c, kind)
    for case in test_cases for c in case.cases for kind in case.kinds.pp
]


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("generator", generators)
@pytest.mark.parametrize("system,case,kind", cases)
class TestFunctionality:
    def base_test(self, system: testcases.TestCase, case: str, method: str,
                  kind: str, generator: str, **args):
        # build a ReferenceState that is already aware of the case (cvs/...)
        hf = testdata_cache.refstate(system, case=case)
        # load the adc refdata
        refdata = getattr(testdata_cache, f"{generator}_data")(
            system=system, method=method, case=case
        )
        ref: dict = refdata[kind]
        n_ref = len(ref["eigenvalues"])

        if "cvs" in case and "cvs" not in method:
            method = f"cvs-{method}"

        # Run ADC and properties
        res = getattr(adcc, method.replace("-", "_"))(hf, **args)

        # Checks
        assert isinstance(res, ExcitedStates)
        assert res.converged
        assert_allclose(res.excitation_energy[:n_ref],
                        ref["eigenvalues"], atol=1e-7)

        # load the mp refdata and compare mp2/3 energies
        refmp = getattr(testdata_cache, f"{generator}_data")(
            system=system, method="mp", case=case
        )
        if res.method.level >= 2:
            assert res.ground_state.energy_correction(2) == \
                approx(refmp["mp2"]["energy"])
        if res.method.level >= 3:
            if not res.method.is_core_valence_separated:
                # TODO The latter check can be removed once CVS-MP3 energies
                #      are implemented
                assert res.ground_state.energy_correction(3) == \
                    approx(refmp["mp3"]["energy"])

        for i in range(n_ref):
            # Computing the dipole moment implies a lot of cancelling in the
            # contraction, which has quite an impact on the accuracy.
            res_tdm = res.transition_dipole_moment[i]
            # adcman does not compute the spin forbidden singlet -> triplet
            # transition dipole moments (they have to be zero anyway)
            if generator == "adcman" and kind == "triplet":
                ref_tdm = np.array([0., 0., 0.])
            else:
                ref_tdm = ref["transition_dipole_moments"][i]

            # Test norm and actual values
            res_tdm_norm = np.sum(res_tdm * res_tdm)
            ref_tdm_norm = np.sum(ref_tdm * ref_tdm)
            assert res_tdm_norm == approx(ref_tdm_norm, abs=1e-5)

            # If the eigenpair is degenerate, then some rotation
            # in the eigenspace is possible, which reflects as a
            # rotation inside the dipole moments. This is the case
            # for example for the CN test system. For simplicity,
            # we only compare the norm of the transition dipole moment
            # in such cases and skip the test for the exact values.
            if system.name != "cn":
                assert_allclose_signfix(res_tdm, ref_tdm, atol=1e-5)

        # Computing the dipole moment implies a lot of cancelling in the
        # contraction, which has quite an impact on the accuracy.
        assert_allclose(res.state_dipole_moment[:n_ref],
                        ref["state_dipole_moments"], atol=1e-4)

        # Test we do not use too many iterations
        # removed explicit numbers per method, because they were not system and case
        # dependent anyway.
        assert res.n_iter <= 1 if method in ["adc0", "cvs-adc0"] else 40

    def test_functionality(self, system: str, case: str, method: str, kind: str,
                           generator: str):
        method: adcc.AdcMethod = adcc.AdcMethod(method)
        if generator == "adcman" and "cvs" in case and method.level == 0:
            pytest.skip("CVS-ADC(0) adcman data is not available")

        system: testcases.TestCase = testcases.get_by_filename(system).pop()
        n_states = testcases.kinds_to_nstates([kind]).pop()

        kwargs = {n_states: 3}
        # only few states available for h2o sto3g
        if system.name == "h2o" and system.basis == "sto-3g":
            if method.level < 2:  # adc0/adc1
                if "cvs" in case and "fv" in case:
                    kwargs[n_states] = 1
                elif "cvs" in case:
                    kwargs[n_states] = 2
            elif method.level < 4:  # adc2/adc3
                if "cvs" in case and "fv" in case:  # only 5 states available
                    kwargs["n_guesses"] = 3

        self.base_test(
            system=system, case=case, method=method.name, kind=kind,
            generator=generator, **kwargs
        )
