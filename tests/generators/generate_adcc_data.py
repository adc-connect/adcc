from dump_adcc import dump_groundstate, dump_excited_states, dump_matrix_testdata
from testdata_cache import testdata_cache
import testcases

from adcc.AdcMethod import AdcMethod
from adcc.LazyMp import LazyMp
from adcc.workflow import run_adc, validate_state_parameters
from adcc import copy as adcc_copy

from pathlib import Path
import h5py


_testdata_dirname = "data"


def generate_adc(test_case: testcases.TestCase, method: AdcMethod, case: str,
                 n_states: int = None, n_singlets: int = None,
                 n_triplets: int = None, n_spin_flip: int = None) -> None:
    """
    Generate and dump the excited states reference data for the given reference case
    of the given test case if the data is not available already.
    """
    # TODO: only import the first n excited states
    datadir = Path(__file__).parent.parent / _testdata_dirname
    datafile = datadir / test_case.adcdata_file_name("adcc", method.name)
    hdf5_file = h5py.File(datafile, "a")  # Read/write if exists, create otherwise
    # we might have reference data for singlet and triplet states
    # which are located on the next level of the hdf5 file. So we need to predict
    # the kind at this point to check if we need to perform a calculation
    # for this purpose we need the reference state
    hf = testdata_cache.refstate(test_case=test_case, case=case)
    _, kind = validate_state_parameters(
        hf, n_states=n_states, n_singlets=n_singlets, n_triplets=n_triplets,
        n_spin_flip=n_spin_flip
    )
    if f"{case}/{kind}" in hdf5_file:
        return None
    print(f"Generating {method.name} data for {case} {test_case.file_name}.")
    # prepend cvs to the method if needed (otherwise we will get an error)
    if "cvs" in case and not method.is_core_valence_separated:
        method = AdcMethod(f"cvs-{method.name}")
    states = run_adc(
        hf, method=method, n_states=n_states, n_singlets=n_singlets,
        n_triplets=n_triplets, n_spin_flip=n_spin_flip
    )
    assert states.kind == kind  # maybe we predicted wrong?
    if f"{case}/matrix" not in hdf5_file:
        # the matrix data is only dumped once for each case. I think it does not
        # make sense to dump the data once for a singlet and once for a triplet
        # trial vector.
        matrix_group = hdf5_file.create_group(f"{case}/matrix")
        trial_vec = adcc_copy(states.excitation_vector[0]).set_random()
        dump_matrix_testdata(states.matrix, trial_vec, matrix_group)
    # dump the excited states data
    kind_group = hdf5_file.create_group(f"{case}/{states.kind}")
    dump_excited_states(states, kind_group)


def generate_adc_all(test_case: testcases.TestCase, method: AdcMethod,
                     n_states: int = None, n_singlets: int = None,
                     n_triplets: int = None, n_spin_flip: int = None,
                     states_per_case: dict[str, dict[str, int]] = None) -> None:
    """
    Generate and dump the excited states reference data for all reference cases
    of the given test case.
    """
    for case in test_case.filter_cases(method.adc_type):
        if states_per_case is not None and case in states_per_case:
            n_singlets = states_per_case[case].get("n_singlets", 0)
            n_triplets = states_per_case[case].get("n_triplets", 0)
            n_spin_flip = states_per_case[case].get("n_spin_flip", 0)
        generate_adc(
            test_case, method, case, n_states=n_states, n_singlets=n_singlets,
            n_triplets=n_triplets, n_spin_flip=n_spin_flip
        )


def generate_groundstate(test_case: testcases.TestCase) -> None:
    """Generate and dump the ground state reference data for the given test case."""
    datadir = Path(__file__).parent.parent / _testdata_dirname
    datafile = datadir / test_case.mpdata_file_name("adcc")
    hdf5_file = h5py.File(datafile, "a")  # Read/write if exists, create otherwise
    # go through all test cases
    for case in test_case.cases:
        if case in hdf5_file:
            continue
        print(f"Generating MP data for {case} {test_case.file_name}.")
        hf = testdata_cache.refstate(test_case, case)
        mp = LazyMp(hf)
        case_group = hdf5_file.create_group(case)
        dump_groundstate(mp, case_group)


def generate_h2o():
    cases = testcases.get(n_expected_cases=2, name="h2o")
    for case in cases:
        generate_groundstate(case)
        generate_adc_all(case, method=AdcMethod("adc1"), n_singlets=5)
        generate_adc_all(case, method=AdcMethod("adc1"), n_triplets=5)


def main():
    generate_h2o()


if __name__ == "__main__":
    main()
