import pytest
import unittest
import itertools
import adcc
import adcc.backends
import os

from adcc.misc import expand_test_templates
from adcc.testdata import static_data
from adcc.testdata.cache import psi4_data
from numpy.testing import assert_allclose
from adcc.exceptions import InputError

from adcc.adc_pp.environment import block_ph_ph_0_pcm
from adcc.AdcMatrix import AdcExtraTerm

basissets = ["sto-3g", "cc-pvdz"]
methods = ["adc1", "adc2", "adc3"]


@pytest.mark.skipif("psi4" not in adcc.backends.available(),
                    reason="Psi4 backend not found.")
@expand_test_templates(list(itertools.product(basissets, methods)))
class TestPCM(unittest.TestCase):
    def template_pcm_psi4_ptlr_formaldehyde(self, basis, method):
        if method != "adc1":
            pytest.skip("Data only available for adc1.")
        basename = f"formaldehyde_{basis}_pcm_{method}"
        psi4_pcm_options = {"weight": 0.3, "pcm_method": "IEFPCM", "neq": True,
                            "solvent": "Water"}
        scfres = psi4_run_pcm_hf(static_data.xyz["formaldehyde"], basis, charge=0,
                                 multiplicity=1, conv_tol=1e-12,
                                 conv_tol_grad=1e-11, max_iter=150,
                                 pcm_options=psi4_pcm_options)

        psi4_result = psi4_data[basename]
        assert_allclose(scfres.energy_scf, psi4_result["energy_scf"], atol=1e-8)

        state = adcc.run_adc(scfres, method=method, n_singlets=5,
                             conv_tol=1e-7, environment="ptlr")

        # compare ptLR result to LR data
        assert_allclose(state.excitation_energy,
                        psi4_result["lr_excitation_energy"], atol=5 * 1e-3)

        # Consistency check with values obtained with ADCc
        assert_allclose(state.excitation_energy,
                        psi4_result["ptlr_adcc_excitation_energy"], atol=1e-6)

        # remove cavity files from PSI4 PCM calculations
        for cavityfile in os.listdir(os.getcwd()):
            if cavityfile.startswith(("cavity.off_", "PEDRA.OUT_")):
                os.remove(cavityfile)

    def template_pcm_psi4_linear_response_formaldehyde(self, basis, method):
        if method != "adc1":
            pytest.skip("Reference only exists for adc1.")
        basename = f"formaldehyde_{basis}_pcm_{method}"
        psi4_result = psi4_data[basename]

        psi4_pcm_options = {"weight": 0.3, "pcm_method": "IEFPCM", "neq": True,
                            "solvent": "Water"}

        scfres = psi4_run_pcm_hf(static_data.xyz["formaldehyde"], basis, charge=0,
                                 multiplicity=1, conv_tol=1e-12,
                                 conv_tol_grad=1e-11, max_iter=150,
                                 pcm_options=psi4_pcm_options)

        assert_allclose(scfres.energy_scf, psi4_result["energy_scf"], atol=1e-8)

        matrix = adcc.AdcMatrix(method, scfres)
        solvent = AdcExtraTerm(matrix, {'ph_ph': block_ph_ph_0_pcm})

        matrix += solvent
        assert len(matrix.extra_terms)

        state = adcc.run_adc(matrix, n_singlets=5, conv_tol=1e-7,
                             environment=False)
        assert_allclose(
            state.excitation_energy_uncorrected,
            psi4_result["lr_excitation_energy"],
            atol=1e-5
        )

        state_cis = adcc.ExcitedStates(state, property_method="adc0")
        assert_allclose(
            state_cis.oscillator_strength,
            psi4_result["lr_osc_strength"], atol=1e-3
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
            psi4_result["lr_excitation_energy"],
            atol=1e-5
        )

        # remove cavity files from PSI4 PCM calculations
        for cavityfile in os.listdir(os.getcwd()):
            if cavityfile.startswith(("cavity.off_", "PEDRA.OUT_")):
                os.remove(cavityfile)


def psi4_run_pcm_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-12,
                    conv_tol_grad=1e-11, max_iter=150, pcm_options=None):
    import psi4

    # needed for PE and PCM tests
    psi4.core.clean_options()
    mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        symmetry c1
        units au
        no_reorient
        no_com
    """)

    psi4.core.be_quiet()
    psi4.set_options({
        'basis': basis,
        'scf_type': 'pk',
        'e_convergence': conv_tol,
        'd_convergence': conv_tol_grad,
        'maxiter': max_iter,
        'reference': "RHF",
        "pcm": True,
        "pcm_scf_type": "total",
    })
    psi4.pcm_helper(f"""
        Units = AU
        Cavity {{Type = Gepol
                Area = {pcm_options.get("weight", 0.3)}}}
        Medium {{Solvertype = {pcm_options.get("pcm_method", "IEFPCM")}
                Nonequilibrium = {pcm_options.get("neq", True)}
                Solvent = {pcm_options.get("solvent", "Water")}}}""")

    if multiplicity != 1:
        psi4.set_options({
            'reference': "UHF",
            'maxiter': max_iter + 500,
            'soscf': 'true'
        })

    _, wfn = psi4.energy('SCF', return_wfn=True, molecule=mol)
    psi4.core.clean()
    return adcc.backends.import_scf_results(wfn)
