#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from numpy.testing import assert_allclose
import pytest

import adcc
import adcc.backends
from adcc.exceptions import InputError
from adcc.AdcMatrix import AdcExtraTerm
from adcc.adc_pp.environment import block_ph_ph_0_pe

from .testing import cached_backend_hf
from ..testdata_cache import testdata_cache, tmole_data
from .. import testcases

try:
    import cppe  # noqa: F401

    has_cppe = True
except ImportError:
    has_cppe = False


backends = [b for b in adcc.backends.available()
            if b not in ["molsturm", "veloxchem"]]

methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]


@pytest.mark.skipif(not has_cppe, reason="CPPE not found")
@pytest.mark.skipif(len(backends) == 0, reason="No backend found.")
@pytest.mark.parametrize("system", ["formaldehyde_sto3g", "formaldehyde_ccpvdz"])
@pytest.mark.parametrize("backend", backends)
class TestPolarizableEmbedding:
    @pytest.mark.parametrize("method", methods)
    def test_perturbative(self, system: str, method: str, backend: str):
        test_case = testcases.get_by_filename(system).pop()

        refdata = testdata_cache.adcman_data(
            system=system, method=method, case="gen"
        )["singlet"]

        assert test_case.pe_potfile is not None
        pe_options = {"potfile": test_case.pe_potfile}
        scfres = cached_backend_hf(
            backend=backend, system=system, pe_options=pe_options
        )
        state = adcc.run_adc(
            scfres, method=method, n_singlets=5, conv_tol=1e-7,
            environment=["ptlr", "ptss"]
        )
        assert state.converged

        n_states = min(len(refdata["eigenvalues"]), len(state.excitation_energy))
        assert n_states > 1

        assert_allclose(
            refdata["eigenvalues"][:n_states],
            state.excitation_energy_uncorrected[:n_states],
            atol=1e-5
        )
        assert_allclose(
            + refdata["eigenvalues"][:n_states]
            + refdata["pe_ptss_correction"][:n_states]
            + refdata["pe_ptlr_correction"][:n_states],
            state.excitation_energy[:n_states],
            atol=1e-5
        )
        assert_allclose(
            refdata["pe_ptss_correction"][:n_states],
            state.pe_ptss_correction[:n_states],
            atol=1e-5
        )
        assert_allclose(
            refdata["pe_ptlr_correction"][:n_states],
            state.pe_ptlr_correction[:n_states],
            atol=1e-5
        )

    @pytest.mark.parametrize("method", ["adc2"])
    def test_linear_response(self, system: str, method: str, backend: str):
        system: testcases.TestCase = testcases.get_by_filename(system).pop()

        basename = f"{system.file_name}_pe_{method}"
        tm_result = tmole_data[basename]

        assert system.pe_potfile is not None
        pe_options = {"potfile": system.pe_potfile}
        scfres = cached_backend_hf(
            backend=backend, system=system.file_name, pe_options=pe_options
        )
        assert_allclose(scfres.energy_scf, tm_result["energy_scf"], atol=1e-8)

        matrix = adcc.AdcMatrix(method, scfres)
        solvent = AdcExtraTerm(matrix, {'ph_ph': block_ph_ph_0_pe})

        # manually add the coupling term
        matrix += solvent
        assert len(matrix.extra_terms)

        assert_allclose(
            matrix.ground_state.energy(2),
            tm_result["energy_mp2"],
            atol=1e-8
        )
        state = adcc.run_adc(matrix, n_singlets=5, conv_tol=1e-7,
                             environment=False)
        assert state.converged
        assert_allclose(
            state.excitation_energy_uncorrected,
            tm_result["excitation_energy"],
            atol=1e-6
        )

        # invalid combination
        with pytest.raises(InputError):
            adcc.run_adc(scfres, method=method, n_singlets=5,
                         environment={"linear_response": True, "ptlr": True})
        # no scheme specified
        with pytest.raises(InputError):
            adcc.run_adc(scfres, method=method, n_singlets=5)

        # automatically add coupling term
        state = adcc.run_adc(scfres, method=method, n_singlets=5,
                             conv_tol=1e-7, environment="linear_response")
        assert state.converged
        assert_allclose(
            state.excitation_energy_uncorrected,
            tm_result["excitation_energy"],
            atol=1e-6
        )
