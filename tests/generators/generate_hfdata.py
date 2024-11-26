from dump_pyscf import dump_pyscf
import test_cases

from pyscf import gto, scf

from pathlib import Path
import numpy as np
import h5py


_testdata_dirname = "data"


def hdf5_filename(test_case: test_cases.TestCase) -> Path:
    """
    Build the absolute file name for the test case.
    """
    hdf5_file = Path(__file__).resolve().parent.parent / _testdata_dirname
    hdf5_file /= f"{test_case.file_name}_hfdata.hdf5"
    return hdf5_file


def run_pyscf(test_case: test_cases.TestCase, restricted: bool, frac_occ: bool):
    """
    Runs the pyscf calculation for the given testcase.
    """
    # Run SCF in pyscf and converge super-tight using an EDIIS
    mol = gto.M(
        atom=test_case.xyz,
        basis=test_case.basis,
        unit=test_case.unit,
        spin=test_case.multiplicity - 1,  # =2S
        verbose=4
    )
    mf = scf.RHF(mol) if restricted else scf.UHF(mol)
    mf.diis = scf.EDIIS()
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-12
    mf.diis_space = 6
    mf.max_cycle = 600
    if frac_occ:
        mf = scf.addons.frac_occ(mf)
    mf.kernel()
    assert mf.converged  # ensure that the SCF is converged
    return mf


def generate(test_case: test_cases.TestCase, restricted: bool,
             frac_occ: bool) -> h5py.File:
    """
    Run Pyscf for the given test case and dump the result in the hdf5 file
    if the file does not already exist.
    """
    hdf5_file = hdf5_filename(test_case)
    if hdf5_file.exists():
        print(f"Skipping {test_case.file_name}. hdata already exists.")
        return None
    mf = run_pyscf(test_case, restricted, frac_occ)
    return dump_pyscf(mf, str(hdf5_file))


def generate_ch2nh2():
    cases = test_cases.get(n_expected_cases=1, name="ch2nh2").pop()
    generate(cases, restricted=False, frac_occ=True)


def generate_cn():
    cases = test_cases.get(n_expected_cases=2, name="cn")
    for case in cases:
        hdf5_file = generate(case, restricted=False, frac_occ=True)
        if hdf5_file is None:
            continue
        # Since CN has some symmetry some energy levels are degenerate,
        # which can lead to all sort of inconsistencies. This code
        # adds a fudge value of 1e-14 to make them numerically distinguishable
        orben_f = hdf5_file["orben_f"]
        for i in range(1, len(orben_f)):
            if np.abs(orben_f[i - 1] - orben_f[i]) < 1e-14:
                orben_f[i - 1] -= 1e-14
                orben_f[i] += 1e-14


def generate_h2o():
    cases = test_cases.get(n_expected_cases=2, name="h2o")
    for case in cases:
        generate(case, restricted=True, frac_occ=False)


def generate_h2s():
    cases = test_cases.get(n_expected_cases=2, name="h2s")
    for case in cases:
        generate(case, restricted=True, frac_occ=False)


def generate_hf():
    case = test_cases.get(n_expected_cases=1, name="hf").pop()
    generate(case, restricted=False, frac_occ=False)


def generate_methox():
    case = test_cases.get(n_expected_cases=1, name="r2methyloxirane").pop()
    generate(case, restricted=True, frac_occ=False)


def main():
    generate_ch2nh2()
    generate_cn()
    generate_h2o()
    generate_h2s()
    generate_hf()
    generate_methox()


if __name__ == "__main__":
    main()