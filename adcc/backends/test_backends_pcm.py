import os
import pytest
import unittest
import itertools
import numpy as np
import adcc
import adcc.backends

from scipy import constants
from numpy.testing import assert_allclose

from adcc.misc import expand_test_templates
from adcc.testdata import static_data
from adcc.testdata.cache import psi4_data, pyscf_data
from adcc.AdcMatrix import AdcExtraTerm
from adcc.exceptions import InputError
from adcc.adc_pp.environment import block_ph_ph_0_pcm

backends = [b for b in ["psi4", "pyscf"] if b in adcc.backends.available()]
basissets = ["sto3g", "ccpvdz"]
methods = ["adc1"]


@pytest.mark.skipif(len(backends) == 0, reason="No backend for PCM available.")
@expand_test_templates(list(itertools.product(basissets, methods, backends)))
class TestPCM(unittest.TestCase):
    def template_pcm_ptlr_formaldehyde(self, basis, method, backend):
        if method != "adc1":
            pytest.skip("Data only available for adc1.")

        c = config[backend]
        basename = f"formaldehyde_{basis}_pcm_{method}"
        result = c["data"][basename]

        run_hf = c["run_hf"]
        scfres = run_hf(static_data.xyz["formaldehyde"], basis, charge=0,
                        multiplicity=1, conv_tol=1e-12, conv_tol_grad=1e-11,
                        max_iter=150, pcm_options=c["pcm_options"],
                        options=c["options"])

        assert_allclose(scfres.energy_scf, result["energy_scf"], atol=1e-8)

        state = adcc.run_adc(scfres, method=method, n_singlets=5,
                             conv_tol=1e-7, environment="ptlr")

        # compare ptLR result to LR data
        assert_allclose(state.excitation_energy,
                        result["lr_excitation_energy"], atol=5e-3)

        # Consistency check with values obtained with ADCc
        assert_allclose(state.excitation_energy,
                        result["ptlr_adcc_excitation_energy"], atol=1e-5)

        if backend == "psi4":
            # remove cavity files from PSI4 PCM calculations
            remove_cavity_psi4()

    def template_pcm_linear_response_formaldehyde(self, basis, method, backend):
        if method != "adc1":
            pytest.skip("Reference only exists for adc1.")

        c = config[backend]
        basename = f"formaldehyde_{basis}_pcm_{method}"
        result = c["data"][basename]

        run_hf = c["run_hf"]
        scfres = run_hf(static_data.xyz["formaldehyde"], basis, charge=0,
                        multiplicity=1, conv_tol=1e-12, conv_tol_grad=1e-11,
                        max_iter=150, pcm_options=c["pcm_options"],
                        options=c["options"])

        assert_allclose(scfres.energy_scf, result["energy_scf"], atol=1e-8)

        matrix = adcc.AdcMatrix(method, scfres)
        solvent = AdcExtraTerm(matrix, {'ph_ph': block_ph_ph_0_pcm})

        matrix += solvent
        assert len(matrix.extra_terms)

        state = adcc.run_adc(matrix, n_singlets=5, conv_tol=1e-7,
                             environment=False)
        assert_allclose(
            state.excitation_energy_uncorrected,
            result["lr_excitation_energy"],
            atol=1e-5
        )

        state_cis = adcc.ExcitedStates(state, property_method="adc0")
        assert_allclose(
            state_cis.oscillator_strength,
            result["lr_osc_strength"], atol=1e-3
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
            result["lr_excitation_energy"],
            atol=1e-5
        )

        if backend == "psi4":
            # remove cavity files from PSI4 PCM calculations
            remove_cavity_psi4()


@pytest.mark.skipif("psi4" not in adcc.backends.available(),
                    reason="Psi4 not available")
class TestPCMcomparison(unittest.TestCase):
    def test_comparison_dd_pcmsolver(self):
        scfargs = dict(
            xyz=static_data.xyz["nh3"],
            basis="3-21g",
            charge=0, multiplicity=1,
            conv_tol=1e-12, conv_tol_grad=1e-11,
            max_iter=150,
        )
        psiopts = dict(scf_type="direct")
        adcopts = dict(method="adc2", n_singlets=5, conv_tol=1e-7,
                       environment="linear_response")

        scfres_ief = psi4_run_pcm_hf(**scfargs, options=dict(pcm=True, **psiopts),
                                     pcm_options=dict(pcm_method="IEFPCM",
                                                      solvent="Water",
                                                      radiiset="UFF"))
        state_ief = adcc.run_adc(scfres_ief, **adcopts)

        scfres_dd = psi4_run_pcm_hf(**scfargs, options=dict(
            ddx=True, ddx_model="pcm", ddx_solvent="water",
            ddx_radii_set="uff", ddx_radii_scaling=1.2,
            ddx_lmax=10, **psiopts,
        ))
        state_dd = adcc.run_adc(scfres_dd, **adcopts)

        assert_allclose(scfres_ief.energy_scf, scfres_dd.energy_scf, atol=1e-4)
        assert_allclose(state_ief.excitation_energy, state_dd.excitation_energy,
                        atol=5e-4)
        remove_cavity_psi4()

    def test_comparison_gaussian(self):
        scfres = psi4_run_pcm_hf(
            xyz=static_data.xyz["nh3"],
            basis="3-21g",
            charge=0, multiplicity=1,
            conv_tol=1e-12, conv_tol_grad=1e-11,
            max_iter=150,
            options=dict(
                ddx=True, ddx_model="cosmo", ddx_solvent_epsilon=2.0,
                ddx_solvent_epsilon_optical=2.0, ddx_radii_set="uff",
                ddx_radii_scaling=1.1, ddx_lmax=10, ddx_n_lebedev=590,
                scf_type="direct"
            )
        )
        state = adcc.cis(scfres, n_singlets=7, conv_tol=1e-7,
                         environment="linear_response")

        eV = constants.value("Hartree energy in eV")
        ref_gaussian = np.array([9.2735, 11.2083, 11.2083, 15.2852, 15.2852,
                                 17.6378, 17.9519]) / eV
        assert_allclose(state.excitation_energy, ref_gaussian, atol=5e-4)


def remove_cavity_psi4():
    # removes cavity files from PSI4 PCM calculations
    for cavityfile in os.listdir(os.getcwd()):
        if cavityfile.startswith(("cavity.off_", "PEDRA.OUT_")):
            os.remove(cavityfile)


def psi4_run_pcm_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-12,
                    conv_tol_grad=1e-11, max_iter=150, options=None,
                    pcm_options=None):
    basis_map = {
        "sto3g": "sto-3g",
        "ccpvdz": "cc-pvdz",
    }

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
        'basis': basis_map.get(basis, basis),
        'scf_type': 'pk',
        'e_convergence': conv_tol,
        'd_convergence': conv_tol_grad,
        'maxiter': max_iter,
        'reference': "RHF",
    })
    psi4.set_options(options)

    if options.get("pcm", False):
        psi4.set_options({"pcm_scf_type": "total"})
        pcm_options = pcm_options.copy()

        psi4.pcm_helper(f"""
            Units = AU
            Cavity {{Type = Gepol
                    Area = {pcm_options.get("area", 0.3)}
                    RadiiSet = {pcm_options.get("radiiset", "Bondi")}}}
            Medium {{
                Solvertype = {pcm_options.get("pcm_method", "IEFPCM")}
                Nonequilibrium = {pcm_options.get("nonequilibrium", True)}
                Solvent = {pcm_options.get("solvent", "Water")}
            }}""")

    if multiplicity != 1:
        psi4.set_options({
            'reference': "UHF",
            'maxiter': max_iter + 500,
            'soscf': 'true'
        })

    _, wfn = psi4.energy('SCF', return_wfn=True, molecule=mol)
    psi4.core.clean()
    return adcc.backends.import_scf_results(wfn)


def pyscf_run_pcm_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-11,
                     conv_tol_grad=1e-10, max_iter=150, pcm_options=None,
                     options=None):

    from pyscf import gto, scf
    from pyscf.solvent import ddCOSMO

    mol = gto.M(
        atom=xyz,
        unit="Bohr",
        basis=basis,
        # spin in the pyscf world is 2S
        spin=multiplicity - 1,
        charge=charge,
        # Disable commandline argument parsing in pyscf
        parse_arg=False,
        dump_input=False,
        verbose=0,
    )

    mf = ddCOSMO(scf.RHF(mol))
    # default eps
    mf.with_solvent.eps = pcm_options.get("eps")
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_iter

    mf.kernel()
    # replace eps with eps_opt for the ADC calculation
    mf.with_solvent.eps = pcm_options.get("eps_opt")
    return adcc.backends.import_scf_results(mf)


config = {
    "psi4": {"data": psi4_data, "run_hf": psi4_run_pcm_hf,
             "options": {"pcm": True, },
             "pcm_options": {"pcm_method": "IEFPCM", "solvent": "Water"}},
    "pyscf": {"data": pyscf_data, "run_hf": pyscf_run_pcm_hf,
              "options": {},
              "pcm_options": {"eps": 78.3553, "eps_opt": 1.78}}
}
