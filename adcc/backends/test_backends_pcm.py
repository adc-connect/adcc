import pytest
import unittest
import itertools
import adcc
import adcc.backends
import os

from adcc.misc import expand_test_templates
from adcc.testdata.cache import psi_data
from .testing import cached_backend_hf
from numpy.testing import assert_allclose
from adcc.exceptions import InputError

from adcc.adc_pp.environment import block_ph_ph_0_pcm
from adcc.AdcMatrix import AdcExtraTerm

# remove pyscf until implemented
backends = [b for b in adcc.backends.available()
            if b not in ["molsturm", "veloxchem", "pyscf"]]
basissets = ["sto3g", "ccpvdz"]
methods = ["adc1", "adc2", "adc3"]


@pytest.mark.skipif(len(backends) == 0, reason="No backend found.")
@expand_test_templates(list(itertools.product(basissets, methods, backends)))
class TestPCM(unittest.TestCase):
    def template_pcm_linear_response_formaldehyde(self, basis, method, backend):
        if method != "adc1":
            pytest.skip("Reference only exists for adc1.")
        basename = f"formaldehyde_{basis}_pcm_{method}"
        psi_result = psi_data[basename]
        scfres = cached_backend_hf(backend, "formaldehyde", basis,
                                   pcm=True)
        assert_allclose(scfres.energy_scf, psi_result["energy_scf"], atol=1e-8)

        matrix = adcc.AdcMatrix(method, scfres)
        solvent = AdcExtraTerm(matrix, {'ph_ph': block_ph_ph_0_pcm})

        matrix += solvent
        assert len(matrix.extra_terms)

        state = adcc.run_adc(matrix, n_singlets=5, conv_tol=1e-7,
                             environment=False)
        assert_allclose(
            state.excitation_energy_uncorrected,
            psi_result["excitation_energy"],
            atol=1e-5
        )

        # invalid combination
        with pytest.raises(InputError):
            adcc.run_adc(scfres, method=method, n_singlets=5,
                         environment={"linear_response": True, "ptlr": True})

        # no environment specified
        with pytest.raises(InputError):
            adcc.run_adc(scfres, method=method, n_singlets=5)

        # automatically add coupling term
        state = adcc.run_adc(scfres, method=method, n_singlets=5,
                             conv_tol=1e-7, environment="linear_response")
        assert_allclose(
            state.excitation_energy_uncorrected,
            psi_result["excitation_energy"],
            atol=1e-5
        )

        # remove cavity files from PSI4 PCM calculations
        for cavityfile in os.listdir(os.getcwd()):
            if cavityfile.startswith(("cavity.off_", "PEDRA.OUT_")):
                os.remove(cavityfile)
