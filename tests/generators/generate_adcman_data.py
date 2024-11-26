from run_qchem import run_qchem
import test_cases

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


def generate(test_case: test_cases.TestCase,
             method: AdcMethod) -> None:
    states, _ = run_qchem(test_case, method, import_states=True, import_gs=False)


def generate_groundstate(test_case: test_cases.TestCase) -> None:
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


def generate_ch2nh2():
    case = test_cases.get(n_expected_cases=1, name="ch2nh2").pop()
    generate_groundstate(case)


def generate_cn():
    cases = test_cases.get(n_expected_cases=2, name="cn")
    for case in cases:
        generate_groundstate(case)


def generate_h2o():
    cases = test_cases.get(n_expected_cases=2, name="h2o")
    for case in cases:
        generate_groundstate(case)


def generate_h2s():
    cases = test_cases.get(n_expected_cases=2, name="h2s")
    for case in cases:
        generate_groundstate(case)


def generate_hf():
    case = test_cases.get(n_expected_cases=1, name="hf").pop()
    generate_groundstate(case)


def generate_methox():
    case = test_cases.get(n_expected_cases=1, name="r2methyloxirane").pop()
    generate_groundstate(case)


def main():
    generate_ch2nh2()
    generate_cn()
    generate_h2o()
    generate_h2s()
    generate_hf()
    generate_methox()


if __name__ == "__main__":
    main()
