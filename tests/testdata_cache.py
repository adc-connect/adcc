from . import testcases

from adcc.AdcMatrix import AdcMatrix
from adcc.ExcitedStates import ExcitedStates
from adcc.LazyMp import LazyMp
from adcc.misc import cached_member_function
from adcc.ReferenceState import ReferenceState
from adcc.solver import EigenSolverStateBase
from adcc import hdf5io, guess_zero

from pathlib import Path
import numpy as np
import h5py
import json


_testdata_dirname = "data"


class AdcMockState(EigenSolverStateBase):
    def __init__(self, matrix):
        super().__init__(matrix)


class TestdataCache:
    @cached_member_function
    def _load_hfdata(self, system: str) -> dict:
        """Load the HF data for the given test case."""
        if isinstance(system, str):
            # avoid loading data twice for str and TestCase. Instead store a
            # reference to the same object in both cases.
            system = testcases.get_by_filename(system).pop()
            return self._load_hfdata(system)
        assert isinstance(system, testcases.TestCase)
        datadir = Path(__file__).parent / _testdata_dirname
        fname = datadir / system.hfdata_file_name
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing hfdata file for {system.file_name}. Was looking for "
                f"{fname}."
            )
        return hdf5io.load(fname)

    @cached_member_function
    def refstate(self, system: str, case: str) -> ReferenceState:
        """
        Build the adcc.ReferenceState.

        Parameters
        ----------
        system: str
            File name of the test case, e.g., "h2o_sto3g". It is also possible
            to pass the TestCase directly.
        case: str
            The reference case for which to construct the ReferenceState, e.g.,
            "gen" for generic or "fv-cvs" for a frozen virtual, core valence
            separated reference state.
        """
        if isinstance(system, str):
            # avoid building ReferenceState twice for str and TestCase.
            # Instead store a reference to the same object in both cases.
            system = testcases.get_by_filename(system).pop()
            return self.refstate(system, case=case)
        assert isinstance(system, testcases.TestCase)
        # ensure that the case is valid for the testcase
        assert case in system.cases
        # load the pyscf hf data and initialize a ReferenceState depending
        # on the case (cvs, fc, ...)
        hfdata = self._load_hfdata(system)
        core_orbitals = system.core_orbitals if "cvs" in case else None
        frozen_core = system.frozen_core if "fc" in case else None
        frozen_virtual = system.frozen_virtual if "fv" in case else None
        refstate = ReferenceState(
            hfdata, core_orbitals=core_orbitals, frozen_core=frozen_core,
            frozen_virtual=frozen_virtual
        )
        refstate.import_all()
        return refstate

    @cached_member_function
    def hfimport(self, system: str, case: str) -> dict:
        """
        Load HF data that was dumped after an import with ReferenceState.

        Parameters
        ----------
        system: str
            File name of the test case, e.g., "h2o_sto3g". It is also possible
            to pass the TestCase directly.
        case: str
            The reference case for which to load the data, e.g.,
            "gen" for generic or "fv-cvs" for frozen virtual core valence
            separated.
        """
        if isinstance(system, str):
            # avoid loading data twice for str and TestCase. Instead store a
            # reference to the same object in both cases.
            system = testcases.get_by_filename(system).pop()
            return self.hfimport(system, case=case)
        assert isinstance(system, testcases.TestCase)
        # ensure that the case is valid for the testcase
        assert case in system.cases
        # ensure that the file exists
        datadir = Path(__file__).parent / _testdata_dirname
        fname = datadir / system.hfimport_file_name
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing hfimport data file for {system.file_name}."
            )
        data = hdf5io.load(fname).get(case, None)
        if data is None:
            raise ValueError(f"No data available for case {case} in file {fname}.")
        return data

    @cached_member_function
    def _load_data(self, system: str, method: str, case: str, source: str) -> dict:
        """
        Read the reference data for the given test case and method.
        Source defines the source which generated the reference data, i.e.,
        either adcman or adcc.
        """
        if isinstance(system, str):
            # avoid loading data twice for str and TestCase. Instead store a
            # reference to the same object in both cases.
            system = testcases.get_by_filename(system).pop()
            return self._load_data(system, method=method, case=case, source=source)
        assert isinstance(system, testcases.TestCase)
        datadir = Path(__file__).parent / _testdata_dirname
        if method == "mp":
            datafile = datadir / system.mpdata_file_name(source)
        else:  # adc data
            datafile = datadir / system.adcdata_file_name(source, method)
        if not datafile.exists():
            raise FileNotFoundError(f"Missing reference data file {datafile}.")
        with h5py.File(datafile, "r") as hdf5_file:
            if case not in hdf5_file:
                raise ValueError(f"No data available for case {case} in file "
                                 f"{datafile}.")
            data = hdf5io.extract_group(hdf5_file[case])
        return data

    def adcc_data(self, system: str, method: str, case: str) -> dict:
        """
        Load the adcc reference data for the given system and method, where
        method might refer to mp or adcn.
        """
        return self._load_data(
            system=system, method=method, case=case, source="adcc"
        )

    def adcman_data(self, system: str, method: str, case: str) -> dict:
        """
        Load the adcman reference data for the given system and method, where
        method might refer to mp or adcn.
        """
        return self._load_data(
            system=system, method=method, case=case, source="adcman"
        )

    @cached_member_function
    def _make_mock_adc_state(self, system: str, method: str, case: str,
                             kind: str, source: str):
        """
        Create an ExcitedStates instance for the given test case, method (adcn),
        reference case (gen/cvs/fc/...) and state kind (singlet/triplet/any/...).
        Source refers to the source with which the data were generated
        (adcman/adcc).
        """
        if isinstance(system, str):
            # avoid building an ExcitedStates object from the same data twice for
            # str and TestCase. Instead store a reference to the same object in
            # both cases.
            system = testcases.get_by_filename(system).pop()
            return self._make_mock_adc_state(
                system, method=method, case=case, kind=kind, source=source
            )
        assert isinstance(system, testcases.TestCase)
        # load the adc data
        data = self._load_data(system, method=method, case=case, source=source)
        adc_data = data.get(kind, None)
        if adc_data is None:
            raise ValueError(f"No data available for kind {kind} in {case} "
                             f"{method} {system}.")
        # load the reference state and build a matrix on top
        if "cvs" in case and "cvs" not in method:
            method = f"cvs-{method}"
        refstate = self.refstate(system, case)
        ground_state = LazyMp(refstate)
        matrix = AdcMatrix(method, ground_state)

        states = AdcMockState(matrix)
        states.method = matrix.method
        states.ground_state = ground_state
        states.reference_state = refstate
        states.kind = kind
        states.eigenvalues = adc_data["eigenvalues"]

        if refstate.restricted and kind == "singlet":
            symm = "symmetric"
            spin_change = 0
        elif refstate.restricted and kind == "triplet":
            symm = "antisymmetric"
            spin_change = 0
        elif kind in ["spin_flip", "any"]:
            symm = "none"
            spin_change = 0 if kind == "any" else -1
        else:
            raise ValueError(f"Unknown kind: {kind}")

        n_states = len(adc_data["eigenvalues"])
        states.eigenvectors = [guess_zero(matrix, spin_change=spin_change,
                                          spin_block_symmetrisation=symm)
                               for _ in range(n_states)]

        blocks = matrix.axis_blocks
        for i, evec in enumerate(states.eigenvectors):
            evec[blocks[0]].set_from_ndarray(adc_data["eigenvectors_singles"][i])
            if len(blocks) > 1:
                evec[blocks[1]].set_from_ndarray(
                    adc_data["eigenvectors_doubles"][i], 1e-14
                )
        return ExcitedStates(states)

    def adcc_states(self, system: str, method: str, kind: str,
                    case: str) -> ExcitedStates:
        """
        Construct an ExcitedStates object for the given test case,
        method (adcn), state kind (singlet/triplet/any/...) and
        reference case (gen/cvs/fc/...) using the adcc reference data
        (the eigenvalues and eigenstates).
        """
        return self._make_mock_adc_state(
            system, method=method, case=case, kind=kind, source="adcc"
        )

    def adcman_states(self, system: str, method: str, kind: str,
                      case: str) -> ExcitedStates:
        """
        Construct an ExcitedStates object for the given test case,
        method (adcn), state kind (singlet/triplet/any/...) and
        reference case (gen/cvs/fc/...) using the adcman reference data
        (the eigenvalues and eigenstates).
        """
        return self._make_mock_adc_state(
            system, method=method, case=case, kind=kind, source="adcman"
        )


testdata_cache = TestdataCache()


def read_json_data(name: str) -> dict:
    """Import the json file from the data directory."""
    jsonfile = Path(__file__).parent / _testdata_dirname / name
    if not jsonfile.exists():
        raise FileNotFoundError(f"Missing json data file {jsonfile}.")
    return json.load(open(jsonfile, "r"), object_hook=_import_hook)


def _import_hook(data: dict):
    return {key: np.array(val) if isinstance(val, list) else val
            for key, val in data.items()}


psi4_data = read_json_data("psi4_data.json")
pyscf_data = read_json_data("pyscf_data.json")
