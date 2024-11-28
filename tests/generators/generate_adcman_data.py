from run_qchem import run_qchem
import testcases

from adcc.AdcMethod import AdcMethod
from adcc.hdf5io import emplace_dict

from pathlib import Path
import h5py


_testdata_dirname = "data"

# the base methods for each adc_type for which to generate data
# the different cases (cvs, fc, ...) are handled in the generate functions.
_methods = {
    "pp": ("adc0", "adc1", "adc2", "adc2x", "adc3")
}
# Since it seems not possible to only perform an adcman MPn calculation,
# the ground state data has to be extracted from an adc(n) calculation.
# The method given below is used for this. The pt order should rather high
# to ensure that all desired MP properties are generated.
_gs_data_method = "adc3"


def generate_adc(test_case: testcases.TestCase, method: AdcMethod, case: str,
                 n_singlets: int = 0, n_triplets: int = 0) -> None:
    """
    Generate and dump the excited state reference data for the given reference case
    of the given test case if the data doesn't exist already.
    """
    # TODO: maybe we only want to import the first n excited states?
    # or we stay consistent with the existing adcc strategy to always dump
    # all energies, etc, but only dump the dms, tms, ... for the first n states.
    datadir = Path(__file__).parent.parent / _testdata_dirname
    datafile = datadir / test_case.adcdata_file_name("adcman", method.name)
    hdf5_file = h5py.File(datafile, "a")  # Read/write if exists, create otherwise
    if case in hdf5_file:
        return None
    # skip cvs-adc(0), since it is not available in qchem.
    if case == "cvs" and method.level == 0:
        return None
    print(f"Generating {method.name} data for {case} {test_case.file_name}.")
    state_data, _ = run_qchem(
        test_case, method, case, import_states=True, import_gs=False,
        n_singlets=n_singlets, n_triplets=n_triplets
    )
    # the data returned from run_qchem should have already been imported
    # using the correct keys -> just dump them
    case_group = hdf5_file.create_group(case)
    emplace_dict(state_data, case_group, compression="gzip")


def generate_adc_all(test_case: testcases.TestCase, method: AdcMethod,
                     n_singlets: int = 0, n_triplets: int = 0,
                     states_per_case: dict[str, dict[str, int]] = None) -> None:
    """
    Generate and dump the excited state reference data for all relevant
    reference cases of the given test case.
    """
    # go through all cases and generate the data
    for case in test_case.filter_cases(method.adc_type):
        if states_per_case is not None and case in states_per_case:
            n_singlets = states_per_case[case].get(case, 0)
            n_triplets = states_per_case[case].get(case, 0)
        generate_adc(
            test_case, method, case, n_singlets=n_singlets, n_triplets=n_triplets
        )


def generate_groundstate(test_case: testcases.TestCase) -> None:
    """
    Generate and dump the ground state data for the given test case.
    """
    datadir = Path(__file__).parent.parent / _testdata_dirname
    datafile = datadir / test_case.mpdata_file_name("adcman")
    hdf5_file = h5py.File(datafile, "a")  # Read/write if exists, create otherwise
    # go through all cases and generate the data
    # but only if the data for the case does not already exist in the file.
    for case in test_case.cases:
        if case in hdf5_file:
            continue
        # run a adc calculation for a single singlet state:
        # we have to ask for a state for adcman to do sth...
        print(f"Generating MP data for {case} {test_case.file_name}.")
        method = f"cvs-{_gs_data_method}" if "cvs" in case else _gs_data_method
        method = AdcMethod(method)
        _, gs_data = run_qchem(
            test_case, method, case, import_states=False, import_gs=True,
            n_singlets=1
        )
        # the data returned from run_qchem should already have been imported
        # such that the correct keys are used -> directly dump them.
        case_group = hdf5_file.create_group(case)
        emplace_dict(gs_data, case_group, compression="gzip")


def generate_ch2nh2():  # UHF, doublet
    case = testcases.get(n_expected_cases=1, name="ch2nh2").pop()
    generate_groundstate(case)
    for method in _methods["pp"]:
        generate_adc_all(case, AdcMethod(method), n_singlets=3)


def generate_cn():  # UHF, doublet
    cases = testcases.get(n_expected_cases=2, name="cn")
    for case in cases:
        generate_groundstate(case)
        for method in _methods["pp"]:
            generate_adc_all(case, AdcMethod(method), n_singlets=6)


def generate_h2o():
    cases = testcases.get(n_expected_cases=2, name="h2o")
    for case in cases:
        generate_groundstate(case)


def generate_h2s():
    cases = testcases.get(n_expected_cases=2, name="h2s")
    for case in cases:
        generate_groundstate(case)


def generate_hf():
    case = testcases.get(n_expected_cases=1, name="hf").pop()
    generate_groundstate(case)


def generate_methox():
    case = testcases.get(n_expected_cases=1, name="r2methyloxirane").pop()
    generate_groundstate(case)


def main():
    # TODO: decide which data to generate and how many states for which case
    generate_ch2nh2()
    generate_cn()
    generate_h2o()
    generate_h2s()
    generate_hf()
    generate_methox()


if __name__ == "__main__":
    main()
