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


def generate(test_case: testcases.TestCase, frac_occ: bool) -> h5py.File:
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
    hdf5_file = h5py.File(hdf5_file, "w")
    dump_pyscf(mf, hdf5_file)
    return hdf5_file


def generate_ch2nh2():
    cases = testcases.get(n_expected_cases=1, name="ch2nh2", basis="sto-3g").pop()
    generate(cases, frac_occ=True)


def generate_cn():
    cases = testcases.get(n_expected_cases=2, name="cn")
    for case in cases:
        hdf5_file = generate(case, frac_occ=True)
        if hdf5_file is None:
            continue
        # TODO: is this still needed? -> comment out for now and see.
        # This only modifies the orbital energies which can be obtained as
        # ReferenceState.orbital_energies(space)
        # However, with the exception of some tests, the orbital energies are
        # usually obtained as
        # ReferenceState.fock(space).diagonal()
        # which is not modified here, since DataHFProvider reads the fock matrix
        # from "fock_ff".
        # In fact, TestReferenceStateCounterData should be the only test that may be
        # affected by this.

        # Since CN has some symmetry some energy levels are degenerate,
        # which can lead to all sort of inconsistencies. This code
        # adds a fudge value of 1e-14 to make them numerically distinguishable
        # orben_f = hdf5_file["orben_f"]
        # for i in range(1, len(orben_f)):
        #     if np.abs(orben_f[i - 1] - orben_f[i]) < 1e-14:
        #         orben_f[i - 1] -= 1e-14
        #         orben_f[i] += 1e-14


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
