from . import testcases

from adcc.AdcMatrix import AdcMatrix
from adcc.ExcitedStates import ExcitedStates
from adcc.LazyMp import LazyMp
from adcc.misc import cached_member_function
from adcc.ReferenceState import ReferenceState
from adcc.solver import EigenSolverStateBase
from adcc import hdf5io, guess_zero

from pathlib import Path


_testdata_dirname = "data"


class AdcMockState(EigenSolverStateBase):
    def __init__(self, matrix):
        super().__init__(matrix)


class TestdataCache:
    @cached_member_function
    def _load_hfdata(self, test_case: testcases.TestCase) -> dict:
        """Load the HF data for the given test case."""
        datadir = Path(__file__).parent / _testdata_dirname
        fname = datadir / test_case.hfdata_file_name
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing hfdata file for {test_case.file_name}. Was looking for "
                f"{fname}."
            )
        return hdf5io.load(fname)

    @cached_member_function
    def refstate(self, system: str, case: str) -> ReferenceState:
        """
        Build the adcc.ReferenceState.

        Parameters
        ----------
        test_case: str
            File name of the test case, e.g., "h2o_sto3g". It is also possible
            to pass the TestCase directly.
        case: str
            The reference case for which to construct the ReferenceState, e.g.,
            "gen" for generic or "fv-cvs" for a frozen virtual, core valence
            separated reference state.
        """
        if isinstance(system, str):
            system = testcases.get_by_filename(system).pop()
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
    def _load_data(self, system: str, method: str, case: str, source: str) -> dict:
        """
        Read the reference data for the given test case and method.
        Source defines the source which generated the reference data, i.e.,
        either adcman or adcc.
        """
        if isinstance(system, str):
            system: testcases.TestCase = (
                testcases.get_by_filename(system).pop()
            )
        datadir = Path(__file__).parent / _testdata_dirname
        if method == "mp":
            datafile = datadir / system.mpdata_file_name(source)
        else:  # adc data
            datafile = datadir / system.adcdata_file_name(source, method)
        if not datafile.exists():
            raise FileNotFoundError(f"Missing reference data file {datafile}.")
        data = hdf5io.load(datafile).get(case, None)
        if data is None:
            raise ValueError(f"No data available for case {case} in file "
                             f"{datafile}.")
        return data

    def adcc_data(self, system: str, method: str, case: str) -> dict:
        """
        Load the adcc reference data for the given system and method, where
        method might refer to mp or adcn.
        """
        return self._load_data(
            system=system, method=method, case=case, source="adcc"
        )

    def adcman_data(self):
        raise NotImplementedError

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
            system: testcases.TestCase = (
                testcases.get_by_filename(system).pop()
            )
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
            raise ValueError("Unknown kind: {}".format(kind))

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
        reference case (gen/cvs/fc/...) using the adcc reference data.
        """
        return self._make_mock_adc_state(
            system, method=method, case=case, kind=kind, source="adcc"
        )

    def adcman_states(self, test_case: str, method: str, kind: str,
                      case: str) -> ExcitedStates:
        """
        Construct an ExcitedStates object for the given test case,
        method (adcn), state kind (singlet/triplet/any/...) and
        reference case (gen/cvs/fc/...) using the adcman reference data.
        """
        return self._make_mock_adc_state(
            test_case, method=method, case=case, kind=kind, source="adcman"
        )


testdata_cache = TestdataCache()
