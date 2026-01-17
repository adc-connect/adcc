from adcc.tests.generators.dump_adcc import (
    dump_groundstate, dump_matrix_testdata, dump_excited_states
)
from adcc.tests.testdata_cache import testdata_cache
from adcc.tests import testcases

from adcc.AdcMethod import AdcMethod
from adcc.LazyMp import LazyMp
from adcc.workflow import run_adc, validate_state_parameters
from adcc import copy as adcc_copy

from pathlib import Path
import h5py


_testdata_dirname = "data"

# the base methods for each adc_type for which to generate data
# the different cases (cvs, fc, ...) are handled in the generate functions.
_methods = {
    "pp": ("adc0", "adc1", "adc2", "adc2x", "adc3")
}


def generate_adc(test_case: testcases.TestCase, method: AdcMethod, case: str,
                 gs_density_order: int | None = None,
                 n_states: int | None = None, n_singlets: int | None = None,
                 n_triplets: int | None = None, n_spin_flip: int | None = None,
                 dump_nstates: int | None = None, **kwargs) -> None:
    """
    Generate and dump the excited states reference data for the given reference case
    of the given test case if the data is not available already.
    """
    datadir = Path(__file__).parent.parent / _testdata_dirname
    datafile = datadir / test_case.adcdata_file_name("adcc", method.name)
    hdf5_file = h5py.File(datafile, "a")  # Read/write if exists, create otherwise
    # we might have reference data for singlet and triplet states
    # which are located on the next level of the hdf5 file. So we need to predict
    # the kind at this point to check if we need to perform a calculation
    # for this purpose we need the reference state
    hf = testdata_cache.refstate(system=test_case, case=case)
    _, kind = validate_state_parameters(
        hf, n_states=n_states, n_singlets=n_singlets, n_triplets=n_triplets,
        n_spin_flip=n_spin_flip
    )
    key = f"{case}/{gs_density_order}"
    if f"{key}/{kind}" in hdf5_file:
        return None
    print(f"Generating {method.name} data for {case} {test_case.file_name}.")
    # prepend cvs to the method if needed (otherwise we will get an error)
    if "cvs" in case and not method.is_core_valence_separated:
        method = AdcMethod(f"cvs-{method.name}")
    # TODO: add the gs_density_order to the run_adc call once implemented
    # For now just use it to enforce the correct data structure in the hdf5 file
    assert gs_density_order is None
    states = run_adc(
        hf, method=method, n_states=n_states, n_singlets=n_singlets,
        n_triplets=n_triplets, n_spin_flip=n_spin_flip, **kwargs
    )
    assert states.kind == kind  # maybe we predicted wrong?  # type: ignore
    if f"{key}/matrix" not in hdf5_file:
        # the matrix data is only dumped once for each case. I think it does not
        # make sense to dump the data once for a singlet and once for a triplet
        # trial vector.
        matrix_group = hdf5_file.create_group(f"{key}/matrix")
        trial_vec = adcc_copy(states.excitation_vector[0]).set_random()  # type: ignore # noqa: E501
        dump_matrix_testdata(states.matrix, trial_vec, matrix_group)
    # dump the excited states data
    kind_group = hdf5_file.create_group(f"{key}/{states.kind}")  # type: ignore
    dump_excited_states(states, kind_group, dump_nstates=dump_nstates)


def generate_adc_all(test_case: testcases.TestCase, method: AdcMethod,
                     n_states: int | None = None, n_singlets: int | None = None,
                     n_triplets: int | None = None, n_spin_flip: int | None = None,
                     dump_nstates: int | None = None,
                     states_per_case: dict[str, dict[str, int]] | None = None,
                     **kwargs) -> None:
    """
    Generate and dump the excited states reference data for all reference cases
    of the given test case.
    """
    for case in test_case.filter_cases(method.adc_type):
        if states_per_case is not None and case in states_per_case:
            n_states = states_per_case[case].get("n_states", None)
            n_singlets = states_per_case[case].get("n_singlets", None)
            n_triplets = states_per_case[case].get("n_triplets", None)
            n_spin_flip = states_per_case[case].get("n_spin_flip", None)
        for density_order in test_case.gs_density_orders:
            generate_adc(
                test_case, method, case, n_states=n_states, n_singlets=n_singlets,
                n_triplets=n_triplets, n_spin_flip=n_spin_flip,
                dump_nstates=dump_nstates, gs_density_order=density_order, **kwargs
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
        dump_groundstate(mp, case_group, test_case.only_full_mode)


def generate_h2o_sto3g():
    # RHF, Singlet, 7 basis functions: 5 occ, 2 virt.
    # cvs: 1 core orbital and 2 virtual orbitals
    # fv-cvs: 1 core and 1 virtual orbital
    # fc-cvs: 1 core and 2 virtual
    # fc-fv-cvs: 1 core and 1 virtual
    states_per_case = {
        "cvs": 2, "fc-cvs": 2, "fv-cvs": 1, "fc-fv-cvs": 1
    }
    test_case = testcases.get(n_expected_cases=1, name="h2o", basis="sto-3g").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        for n_states in \
                testcases.kinds_to_nstates(test_case.kinds[method.adc_type]):
            per_case = None
            if method.level < 2:  # adc0/adc1
                per_case = {
                    case: {n_states: n} for case, n in states_per_case.items()
                }
            n_states = {n_states: 3}
            # for fv-fc-cvs ADC(2) + ADC(3) we apparently only have
            # 4 states available
            # -> reduce n_guesses for all ADC(2) and ADC(3) calculations
            kwargs = {}
            if method.level >= 2:
                kwargs = {"n_guesses": 4}
            generate_adc_all(
                test_case, method=method, dump_nstates=2, states_per_case=per_case,
                **n_states, **kwargs
            )


def generate_h2o_def2tzvp():
    # RHF, Singlet, 43 basis functions: 5 occ, 38 virt.
    test_case = testcases.get(
        n_expected_cases=1, name="h2o", basis="def2-tzvp"
    ).pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        for n_states in \
                testcases.kinds_to_nstates(test_case.kinds[method.adc_type]):
            n_states = {n_states: 3}
            generate_adc_all(
                test_case, method=method, dump_nstates=2, states_per_case=None,
                **n_states
            )


def generate_cn_sto3g():
    # UHF, Doublet, 10 basis functions: (7a, 6b) occ, (3a, 4b) virt
    test_case = testcases.get(n_expected_cases=1, name="cn", basis="sto-3g").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        for n_states in \
                testcases.kinds_to_nstates(test_case.kinds[method.adc_type]):
            kwargs = {n_states: 3}
            generate_adc_all(
                test_case=test_case, method=method, dump_nstates=2,
                states_per_case=None, **kwargs
            )


def generate_cn_ccpvdz():
    # UHF, Doublet, 10 basis functions: (7a, 6b) occ, (3a, 4b) virt
    test_case = testcases.get(n_expected_cases=1, name="cn", basis="cc-pvdz").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        for n_states in \
                testcases.kinds_to_nstates(test_case.kinds[method.adc_type]):
            n_states = {n_states: 3}
            generate_adc_all(
                test_case, method=method, dump_nstates=2, states_per_case=None,
                **n_states
            )


def generate_hf_631g():
    # UHF, Triplet
    test_case = testcases.get(n_expected_cases=1, name="hf").pop()
    generate_groundstate(test_case)
    for method in _methods["pp"]:
        method = AdcMethod(method)
        for n_states in \
                testcases.kinds_to_nstates(test_case.kinds[method.adc_type]):
            n_states = {n_states: 3}
            generate_adc_all(
                test_case, method=method, dump_nstates=2, states_per_case=None,
                **n_states
            )


def main():
    generate_h2o_sto3g()
    generate_h2o_def2tzvp()
    generate_cn_sto3g()
    generate_cn_ccpvdz()
    generate_hf_631g()


if __name__ == "__main__":
    main()
