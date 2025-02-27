import adcc
import pyscf
from pyscf import gto, scf, tdscf
from pyscf.solvent import ddCOSMO

from adcc.tests import testcases

from pathlib import Path
import json


_testdata_dirname = "data"


def run_pyscf_tdscf(xyz: str, basis: str, unit: str = "Bohr", charge: int = 0,
                    multiplicity: int = 1, conv_tol: float = 1e-12,
                    conv_tol_grad: float = 1e-10, max_iter: int = 150,
                    pcm_options: dict = None):
    """
    Performs a pyscf CIS calculation. Returns the SCF and the CIS objects.
    """
    mol = gto.M(
        atom=xyz,
        basis=basis,
        unit=unit,
        charge=charge,
        # spin in the pyscf world is 2S
        spin=multiplicity - 1,
        verbose=0,
        # Disable commandline argument parsing in pyscf
        parse_arg=False,
        dump_input=False,
    )

    if pcm_options:
        mf = ddCOSMO(scf.RHF(mol))
        mf.with_solvent.eps = pcm_options.get("eps")
    else:
        mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_iter

    mf.kernel()

    if pcm_options:
        mf.with_solvent.eps = pcm_options.get("eps_opt")
        # for n_eq solvation only PTE implemented
        mf.with_solvent.equilibrium_solvation = True
        cis = ddCOSMO(tdscf.TDA(mf))
    else:
        cis = tdscf.TDA(mf)
    cis.nstates = 5
    cis.conv_tol = 1e-7
    cis.kernel()
    return mf, cis


def run_adcc_ptlr(wfn) -> adcc.ExcitedStates:
    """Performs a adcc ptlr-pcm adc1 calculation."""
    # only converges with a small subspace, such that the subspace is collapsed
    # "at the right moment"... If the subspsace is too large, the davidson
    # will converge to zero eigenvalues. 30 was too large already...
    return adcc.run_adc(wfn, method="adc1", n_singlets=5, max_subspace=25,
                        max_iter=250, conv_tol=1e-7, environment="ptlr")


def dump_results(test_case: testcases.TestCase, pcm_options: dict = None
                 ) -> tuple[str, dict]:
    name = test_case.file_name
    if pcm_options is not None:
        name += "_pcm"
    name += "_adc1"

    kwargs = test_case.asdict("xyz", "basis", "unit", "charge", "multiplicity")
    hf, cis = run_pyscf_tdscf(**kwargs, pcm_options=pcm_options)

    ret = {"basis": test_case.basis,
           "method": "adc1",
           "molecule": test_case.name,
           "energy_scf": float(hf.e_tot)}
    # dump does not like numpy types
    ret["lr_excitation_energy"] = [float(round(s, 9)) for s in cis.e]
    ret["lr_osc_strength"] = [
        float(round(s, 5)) for s in cis.oscillator_strength()
    ]

    if pcm_options:
        state = run_adcc_ptlr(hf)
        ret["ptlr_adcc_excitation_energy"] = [
            round(float(e), 9) for e in state.excitation_energy
        ]
    return name, ret


def main():
    cases = testcases.get(n_expected_cases=2, name="formaldehyde")

    pcm_options = {"eps": 78.3553, "eps_opt": 1.78}
    pyscf_results = {}
    for case in cases:
        key, res = dump_results(case, pcm_options=pcm_options)
        pyscf_results[key] = res
        print(f"Dumped {key}.")
    pyscf_results["pyscf_version"] = pyscf.__version__

    dump_file = Path(__file__).parent.parent / _testdata_dirname / "pyscf_data.json"
    json.dump(pyscf_results, open(dump_file, "w"), indent=2)


if __name__ == "__main__":
    main()
