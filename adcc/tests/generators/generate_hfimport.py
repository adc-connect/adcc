from adcc.tests.testdata_cache import testdata_cache
from adcc.tests import testcases

import adcc
from adcc.hdf5io import emplace_dict

from pathlib import Path
import numpy as np
import h5py
import itertools


_testdata_dirname = "data"


def dump_imported(test_case: testcases.TestCase):
    hdf5_file = Path(__file__).resolve().parent.parent / _testdata_dirname
    hdf5_file /= test_case.hfimport_file_name
    hdf5_file = h5py.File(hdf5_file, "a")  # Read/write if exists, create otherwise
    # go through the reference cases
    for case in test_case.cases:
        if case in hdf5_file:
            continue
        # build the reference state for the given reference case
        print(f"Generating hfimport data for {case} {test_case.file_name}.")
        refstate = testdata_cache.refstate(test_case, case)
        # extract the relevant data to dictionary
        data = collect_data(refstate)
        # and write in the hdf5 file
        case_group = hdf5_file.create_group(case)
        emplace_dict(data, case_group, compression="gzip")
        case_group.attrs["adcc_version"] = adcc.__version__


def collect_data(refstate: adcc.ReferenceState) -> dict:
    subspaces = refstate.mospaces.subspaces

    # dump the subspaces
    ret = {"subspaces": np.array(subspaces, dtype="S")}  # dtype S: char bytes
    # and the orbital energies + orbital coefficients
    for space in subspaces:
        ret[f"orbital_energies/{space}"] = (
            refstate.orbital_energies(space).to_ndarray()
        )
        ret[f"orbital_coefficients/{space}b"] = (
            refstate.orbital_coefficients(f"{space}b").to_ndarray()
        )
    # dump the fock matrix
    canonical_pairs = []
    for space1, space2 in itertools.combinations_with_replacement(subspaces, 2):
        canonical_pairs.append(space1 + space2)
        ret[f"fock/{space1}{space2}"] = refstate.fock(space1 + space2).to_ndarray()
    # dump the eri
    for bra, ket in itertools.combinations_with_replacement(canonical_pairs, 2):
        ret[f"eri/{bra}{ket}"] = refstate.eri(bra + ket).to_ndarray()
    return ret


def main():
    test_cases = testcases.get_by_filename(
        "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz", "ch2nh2_sto3g"
    )
    for tcase in test_cases:
        dump_imported(tcase)


if __name__ == "__main__":
    main()
