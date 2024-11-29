import testcases

from adcc.misc import cached_member_function
from adcc.ReferenceState import ReferenceState
from adcc import hdf5io

from pathlib import Path


_testdata_dirname = "data"


class TestdataCache:
    @cached_member_function
    def _load_hfdata(self, test_case: testcases.TestCase) -> dict:
        """Load the HF data for the given test case."""
        datadir = Path(__file__).parent.parent / _testdata_dirname
        fname = datadir / test_case.hfdata_file_name
        if not fname.exists():
            raise FileNotFoundError(
                f"Missing hfdata file for {test_case.file_name}. Was looking for "
                f"{fname}."
            )
        return hdf5io.load(fname)

    @cached_member_function
    def refstate(self, test_case: testcases.TestCase, case: str) -> ReferenceState:
        """Build the adcc.ReferenceState for the given test case"""
        # ensure that the case is valid for the testcase
        assert case in test_case.cases
        # load the pyscf hf data and initialize a ReferenceState depending
        # on the case (cvs, fc, ...)
        hfdata = self._load_hfdata(test_case)
        core_orbitals = test_case.core_orbitals if "cvs" in case else 0
        frozen_core = test_case.frozen_core if "fc" in case else 0
        frozen_virtual = test_case.frozen_virtual if "fv" in case else 0
        refstate = ReferenceState(
            hfdata, core_orbitals=core_orbitals, frozen_core=frozen_core,
            frozen_virtual=frozen_virtual
        )
        refstate.import_all()
        return refstate


testdata_cache = TestdataCache()
