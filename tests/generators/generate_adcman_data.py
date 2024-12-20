from tests.generators.run_qchem import run_qchem
from tests import testcases

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
                 n_singlets: int = 0, n_triplets: int = 0,
                 n_spin_flip: int = 0, n_states: int = 0,
                 dump_nstates: int = None, **kwargs) -> None:
    """
    Generate and dump the excited state reference data for the given reference case
    of the given test case if the data doesn't exist already.
    The function currently assumes that all states kinds are generated
    simultaneously, e.g., singlet and triplet states in a single calculation!
    """
    datadir = Path(__file__).parent.parent / _testdata_dirname
    datafile = datadir / test_case.adcdata_file_name("adcman", method.name)
    hdf5_file = h5py.File(datafile, "a")  # Read/write if exists, create otherwise
    if case in hdf5_file:
        return None
    # skip cvs-adc(0), since it is not available in qchem.
    if "cvs" in case and method.level == 0:
        return None
    print(f"Generating {method.name} data for {case} {test_case.file_name}.")
    # add a cvs prefix to the method if necessary
    if "cvs" in case and not method.is_core_valence_separated:
        method = AdcMethod(f"cvs-{method.name}")
    state_data, _ = run_qchem(
        test_case, method, case, import_states=True, import_gs=False,
        n_singlets=n_singlets, n_triplets=n_triplets, n_spin_flip=n_spin_flip,
        n_states=n_states, import_nstates=dump_nstates, **kwargs
    )
    # the data returned from run_qchem should have already been imported
    # using the correct keys -> just dump them
    case_group = hdf5_file.create_group(case)
    emplace_dict(state_data, case_group, compression="gzip")


def generate_adc_all(test_case: testcases.TestCase, method: AdcMethod,
                     n_singlets: int = 0, n_triplets: int = 0,
                     n_spin_flip: int = 0, n_states: int = 0,
                     dump_nstates: int = None,
                     states_per_case: dict[str, dict[str, int]] = None,
                     **kwargs) -> None:
    """
    Generate and dump the excited state reference data for all relevant
    reference cases of the given test case.
    """
    # go through all cases and generate the data
    for case in test_case.filter_cases(method.adc_type):
        if states_per_case is not None and case in states_per_case:
            n_singlets = states_per_case[case].get("n_singlets", 0)
            n_triplets = states_per_case[case].get("n_triplets", 0)
            n_spin_flip = states_per_case[case].get("n_spin_flip", 0)
            n_states = states_per_case[case].get("n_states", 0)
        generate_adc(
            test_case, method, case, n_singlets=n_singlets, n_triplets=n_triplets,
            n_spin_flip=n_spin_flip, n_states=n_states, dump_nstates=dump_nstates,
            **kwargs
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
            n_states=1
        )
        # the data returned from run_qchem should already have been imported
        # such that the correct keys are used -> directly dump them.
        case_group = hdf5_file.create_group(case)
        emplace_dict(gs_data, case_group, compression="gzip")


def generate_h2o_sto3g():
    # RHF, Singlet, 7 basis functions: 5 occ, 2 virt.
    states = {
        # only define for adc1, because we skip adc0 cvs calculations
        # (they are not implented in adcman)
        "adc1": {
            # 1 core and 2 virtual orbitals
            "cvs": {"n_singlets": 2, "n_triplets": 2},
            # 1 core and 1 virtual orbital
            "fv-cvs": {"n_singlets": 1, "n_triplets": 1}
        }
    }
    test_case = testcases.get(n_expected_cases=1, name="h2o", basis="sto-3g").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        # get the number of states for all kinds -> Generate the data in a single
        # adcman calculation
        n_states = {kind: 3 for kind in
                    testcases.kinds_to_nstates(test_case.pp_kinds)}
        generate_adc_all(
            test_case, method=method, dump_nstates=2,
            states_per_case=states.get(method.name, None), **n_states
        )


def generate_h2o_def2tzvp():
    # RHF, Singlet, 43 basis functions: 5 occ, 38 virt.
    test_case = testcases.get(
        n_expected_cases=1, name="h2o", basis="def2-tzvp"
    ).pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        n_states = {kind: 3 for kind in
                    testcases.kinds_to_nstates(test_case.pp_kinds)}
        generate_adc_all(
            test_case, method=method, dump_nstates=2, **n_states
        )


def generate_cn_sto3g():
    # UHF, Doublet, 10 basis functions: (7a, 6b) occ, (3a, 4b) virt
    test_case = testcases.get(n_expected_cases=1, name="cn", basis="sto-3g").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        n_states = {kind: 3 for kind in
                    testcases.kinds_to_nstates(test_case.pp_kinds)}
        generate_adc_all(
            test_case, method=method, dump_nstates=2, **n_states
        )


def generate_cn_ccpvdz():
    # UHF, Doublet
    test_case = testcases.get(n_expected_cases=1, name="cn", basis="cc-pvdz").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        n_states = {kind: 3 for kind in
                    testcases.kinds_to_nstates(test_case.pp_kinds)}
        generate_adc_all(
            test_case, method=method, dump_nstates=2, **n_states
        )


def generate_hf_631g():
    # UHF, Triplet
    test_case = testcases.get(n_expected_cases=1, name="hf").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        n_states = {kind: 3 for kind in
                    testcases.kinds_to_nstates(test_case.pp_kinds)}
        generate_adc_all(
            test_case, method=method, dump_nstates=2, **n_states
        )


def generate_h2s_sto3g():
    # RHF, Singlet
    states = {
        # only define for adc1, because we skip adc0 cvs calculations
        # (they are not implented in adcman)
        "adc1": {
            # 1 core and 2 virtual orbitals
            "cvs": {"n_singlets": 2, "n_triplets": 2},
            # 1 core and 1 virtual orbital
            "fv-cvs": {"n_singlets": 1, "n_triplets": 1}
        }
    }
    test_case = testcases.get(n_expected_cases=1, name="h2s", basis="sto-3g").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        n_states = {kind: 3 for kind in
                    testcases.kinds_to_nstates(test_case.pp_kinds)}
        generate_adc_all(
            test_case, method=method, dump_nstates=2,
            states_per_case=states.get(method.name, None), **n_states
        )


def generate_h2s_6311g():
    # RHF, Singlet
    test_case = testcases.get(
        n_expected_cases=1, name="h2s", basis="6-311+g**"
    ).pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        for case in test_case.cases:
            kwargs = {kind: 3 for kind in
                      testcases.kinds_to_nstates(test_case.pp_kinds)}
            # davidson did not converge for fc-cvs adc1 triplets
            if "cvs" in case and "fc" in case:
                kwargs["max_ss"] = 21
            generate_adc(
                test_case, method=method, case=case, dump_nstates=2, **kwargs
            )


def main():
    generate_h2o_sto3g()
    generate_h2o_def2tzvp()
    generate_cn_sto3g()
    generate_cn_ccpvdz()
    generate_hf_631g()
    generate_h2s_sto3g()
    generate_h2s_6311g()


if __name__ == "__main__":
    main()
