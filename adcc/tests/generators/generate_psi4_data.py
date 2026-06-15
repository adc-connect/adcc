import psi4
import adcc
from psi4.driver.procrouting.response.scf_response import tdscf_excitations

from adcc.tests import testcases

from pathlib import Path
import json


_testdata_dirname = "data"


def run_psi4_tdscf(xyz: str, basis: str, charge: int = 0, multiplicity: int = 1,
                   conv_tol: float = 1e-12, conv_tol_grad: float = 1e-11,
                   max_iter: int = 150, pcm_options: dict | None = None):
    """
    Performs a psi4 CIS calculation and returns the SCF and the CIS results.
    """
    mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        units au
        symmetry c1
    """)
    psi4.set_options({
        'basis': basis,
        'scf_type': "pK",
        'e_convergence': conv_tol,
        'd_convergence': conv_tol_grad,
        'maxiter': max_iter,
        'reference': "RHF",
        'save_jk': True,
    })
    if pcm_options:
        # think its better to use a dict for the pcm options, because there
        # are already enough function arguments. Of course one could just
        # fix the options... would be sufficient for now too.
        psi4.set_options({'pcm': True, 'pcm_scf_type': "total"})
        psi4.pcm_helper(f"""
            Units = AU
            Cavity {{
                Type = Gepol
                Area = {pcm_options.get("weight", 0.3)}
            }}
            Medium {{
                Solvertype = {pcm_options.get("pcm_method", "IEFPCM")}
                Solvent = {pcm_options.get("solvent", "Water")}
                Nonequilibrium = {pcm_options.get("neq", True)}
            }}
        """)
    psi4.core.be_quiet()
    _, wfn = psi4.energy('scf', return_wfn=True, molecule=mol)  # type: ignore

    res = tdscf_excitations(wfn, states=5, triplets="NONE", tda=True,
                            r_convergence=1e-7)

    # remove cavity files from PSI4 PCM calculations
    if pcm_options:
        for cavityfile in Path.cwd().iterdir():
            if cavityfile.name.startswith(("cavity.off_", "cavity.npz",
                                           "PEDRA.OUT_")):
                cavityfile.unlink()
    return wfn, res


def run_adcc_ptlr(wfn) -> adcc.ExcitedStates:
    """
    Perform a adcc ptlr-pcm adc1 calculation on top of the given psi4 SCF
    calculation.
    """
    # only converges with a small subspace, such that the subspace is collapsed
    # "at the right moment"... If the subspsace is too large, the davidson
    # will converge to zero eigenvalues. 30 was too large already...
    return adcc.run_adc(wfn, method="adc1", n_singlets=5, max_subspace=25,
                        max_iter=100, conv_tol=1e-7, environment="ptlr")


def dump_results(test_case: testcases.TestCase, pcm_options: dict | None = None
                 ) -> tuple[str, dict]:
    """
    Performs a psi4 CIS calculation (potentially with PCM) and collects the
    the result in a dict. For PCM calculations additionally the adcc ADC(1)
    excitation energies are collected.
    """
    name = f"{test_case.file_name}"
    if pcm_options is not None:
        name += "_pcm"
    name += "_adc1"

    assert test_case.unit == "Bohr"  # run_psi4_tdscf hardcodes au
    kwargs = test_case.asdict("xyz", "basis", "charge", "multiplicity")
    wfn, res = run_psi4_tdscf(**kwargs, pcm_options=pcm_options)

    ret = {"basis": test_case.basis,
           "method": "adc1",
           "molecule": test_case.name,
           "energy_scf": wfn.energy()}
    # can't dump numpy types
    ret["lr_excitation_energy"] = [
        round(float(r["EXCITATION ENERGY"]), 9) for r in res
    ]
    ret["lr_osc_strength"] = [
        round(float(r["OSCILLATOR STRENGTH (LEN)"]), 5) for r in res
    ]

    if pcm_options:
        state = run_adcc_ptlr(wfn)
        ret["ptlr_adcc_excitation_energy"] = [
            round(float(e), 9) for e in state.excitation_energy
        ]
    return name, ret


def main():
    cases = testcases.get(n_expected_cases=2, name="formaldehyde")
    pcm_options = {"weight": 0.3, "pcm_method": "IEFPCM", "neq": True,
                   "solvent": "Water"}
    psi4_results = {}
    for case in cases:
        key, res = dump_results(test_case=case, pcm_options=pcm_options)
        psi4_results[key] = res
        print(f"Dumped {key}.")
    psi4_results["psi4_version"] = psi4.__version__

    dump_file = Path(__file__).parent.parent / _testdata_dirname / "psi4_data.json"
    json.dump(psi4_results, open(dump_file, "w"), indent=2)


if __name__ == "__main__":
    main()
