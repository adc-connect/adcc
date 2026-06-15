from adcc.tests.generators.dump_pyscf import dump_pyscf
from adcc.tests import testcases

from pyscf import gto, scf

from pathlib import Path
import h5py
import os
import tempfile


_testdata_dirname = "data"


def run_pyscf(test_case: testcases.TestCase, frac_occ: bool):
    """
    Runs the pyscf calculation for the given testcase.
    """
    # create a temporary directory for the temporary pyscf files:
    # on a cluster the /tmp folder might not be cleaned as often.
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        old_tmpdir = os.environ.get("PYSCF_TMPDIR", None)
        os.environ["PYSCF_TMPDIR"] = tmpdir
        # Run SCF in pyscf and converge super-tight using an EDIIS
        mol = gto.M(
            atom=test_case.xyz,
            basis=test_case.basis,
            unit=test_case.unit,
            spin=test_case.multiplicity - 1,  # =2S
            verbose=0
        )
        mf = scf.RHF(mol) if test_case.restricted else scf.UHF(mol)
        mf.diis = scf.EDIIS()
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-11
        mf.diis_space = 6
        mf.max_cycle = 600
        if frac_occ:
            mf = scf.addons.frac_occ(mf)
        mf.kernel()
        # restore the original state of the env variable
        if old_tmpdir is None:
            del os.environ["PYSCF_TMPDIR"]
        else:
            os.environ["PYSCF_TMPDIR"] = old_tmpdir
        assert mf.converged  # ensure that the SCF is converged
    return mf


def generate(test_case: testcases.TestCase,
             frac_occ: bool) -> None:
    """
    Run Pyscf for the given test case and dump the result in the hdf5 file
    if the file does not already exist.
    """
    hdf5_file = Path(__file__).resolve().parent.parent / _testdata_dirname
    hdf5_file /= test_case.hfdata_file_name
    if hdf5_file.exists():
        return None
    print(f"Generating data for {test_case.file_name}.")
    mf = run_pyscf(test_case, frac_occ)
    with h5py.File(hdf5_file, "w") as hdf5_file:
        dump_pyscf(mf, hdf5_file)


def generate_ch2nh2():
    cases = testcases.get(n_expected_cases=1, name="ch2nh2", basis="sto-3g").pop()
    generate(cases, frac_occ=True)


def generate_cn():
    cases = testcases.get(n_expected_cases=2, name="cn")
    for case in cases:
        generate(case, frac_occ=True)


def generate_h2o():
    cases = testcases.get_by_filename("h2o_sto3g", "h2o_def2tzvp")
    for case in cases:
        generate(case, frac_occ=False)


def generate_hf():
    case = testcases.get(n_expected_cases=1, name="hf").pop()
    generate(case, frac_occ=False)


def generate_methox():
    case = testcases.get(
        n_expected_cases=1, name="r2methyloxirane", basis="sto-3g"
    ).pop()
    generate(case, frac_occ=False)


def main():
    generate_ch2nh2()
    generate_cn()
    generate_h2o()
    generate_hf()
    generate_methox()


if __name__ == "__main__":
    main()
