from . import testcases

from adcc.AdcMatrix import AdcMatrix
from adcc.AdcMethod import AdcMethod
from adcc.ExcitedStates import ExcitedStates
from adcc.ChargedExcitations import AttachedStates, DetachedStates
from adcc.LazyMp import LazyMp
from adcc.misc import cached_member_function
from adcc.ReferenceState import ReferenceState
from adcc.solver import EigenSolverStateBase
from adcc import hdf5io
from adcc.guess import guess_zero, determine_spin_change

from pathlib import Path
from typing import Optional, Union
import numpy as np
import h5py
import json


_testdata_dirname = "data"


class AdcMockState(EigenSolverStateBase):
    def __init__(self, matrix):
        super().__init__(matrix)


class TestdataCache:
    @cached_member_function()
    def _load_hfdata(self, system: Union[str, testcases.TestCase]) -> dict:
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

    @cached_member_function()
    def refstate(self, system: Union[str, testcases.TestCase],
                 case: str) -> ReferenceState:
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

    @cached_member_function()
    def hfimport(self, system: Union[str, testcases.TestCase],
                 case: str) -> dict:
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

    @cached_member_function()
    def _load_data(self, system: Union[str, testcases.TestCase],
                   method: str, case: str, source: str,
                   gs_density_order: Optional[int] = None,
                   is_alpha: Optional[bool] = None) -> dict:
        """
        Load the reference data for the given system, method (mpn / adcn),
        reference case (cvs, fc, fv-cvs, ...) and optionally gs_density_order
        (2, 3, sigma4+, ...) and if it is an alpha process for IP/EA.
        Source defines the source which generated the reference data, i.e.,
        either adcman or adcc.
        """
        if isinstance(system, str):
            # avoid loading data twice for str and TestCase. Instead store a
            # reference to the same object in both cases.
            system = testcases.get_by_filename(system).pop()
            return self._load_data(
                system, method=method, case=case, source=source,
                gs_density_order=gs_density_order, is_alpha=is_alpha
            )
        assert isinstance(system, testcases.TestCase)
        assert case in system.cases
        assert gs_density_order in system.gs_density_orders

        datadir = Path(__file__).parent / _testdata_dirname
        if method == "mp":
            datafile = datadir / system.mpdata_file_name(source)
            key = case
            assert gs_density_order is None  # not considered -> should not be set
        else:  # adc data is one level deeper than mpdata: gs_density_order
            datafile = datadir / system.adcdata_file_name(source, method)
            key = f"{case}/{gs_density_order}"
            if AdcMethod(method).adc_type in ("ip", "ea"):
                assert isinstance(is_alpha, bool)
                spin = "alpha" if is_alpha else "beta"
                key = f"{case}/{gs_density_order}/{spin}"
        if not datafile.exists():
            raise FileNotFoundError(f"Missing reference data file {datafile}.")
        with h5py.File(datafile, "r") as hdf5_file:
            if key not in hdf5_file:
                if is_alpha is None:
                    raise ValueError(
                        f"No data available for case {case} and "
                        f"gs_density_order {gs_density_order} in file {datafile}."
                    )
                else:
                    raise ValueError(
                        f"No data available for case {case}, gs_density_order "
                        f"{gs_density_order} and spin {spin} in file {datafile}."
                    )
            data = hdf5io.extract_group(hdf5_file[key])
        return data

    def adcc_data(self, system: str, method: str, case: str,
                  gs_density_order: Optional[int] = None,
                  is_alpha: Optional[bool] = None) -> dict:
        """
        Load the adcc reference data for the given system, method (mpn / adcn),
        reference case (cvs, fc, fv-cvs, ...) and optionally gs_density_order
        (2, 3, sigma4+, ...) and optionally is_alpha for IP/EA data.
        """
        if ("ip" in method or "ea" in method) and is_alpha is None:
            is_alpha = True
        return self._load_data(
            system=system, method=method, case=case,
            gs_density_order=gs_density_order, source="adcc", is_alpha=is_alpha
        )

    def adcman_data(self, system: str, method: str, case: str,
                    gs_density_order: Optional[int] = None,
                    is_alpha: Optional[bool] = None) -> dict:
        """
        Load the adcman reference data for the given system, method (mpn / adcn),
        reference case (cvs, fc, fv-cvs, ...) and optionally gs_density_order
        (2, 3, sigma4+, ...) and optionally is_alpha for IP/EA data.
        """
        if ("ip" in method or "ea" in method) and is_alpha is None:
            is_alpha = True
        return self._load_data(
            system=system, method=method, case=case,
            gs_density_order=gs_density_order, source="adcman",
            is_alpha=is_alpha
        )

    @cached_member_function()
    def _make_mock_adc_state(
        self, system: Union[str, testcases.TestCase],
        method: str, case: str,
        kind: str, source: str,
        gs_density_order: Optional[int] = None,
        is_alpha: Optional[bool] = None
        ) -> ExcitedStates | AttachedStates | DetachedStates:
        """
        Create an ExcitedStates/AttachedStates/DetachedStates instance for the 
        given test case, method (adcn), reference case (gen/cvs/fc/...), 
        state kind (singlet/triplet/any/...) and optionally gs_density_order 
        (2/3/sigma4+) and optionally is_alpha for IP/EA. Source refers to the
        source with which the data were generated (adcman/adcc).
        The states object is build on top of the loaded HF data and
        contains the eigenstates and eigenvalues of the loaded ADC data.
        """
        if isinstance(system, str):
            # avoid building an ExcitedStates object from the same data twice for
            # str and TestCase. Instead store a reference to the same object in
            # both cases.
            system = testcases.get_by_filename(system).pop()
            return self._make_mock_adc_state(
                system, method=method, case=case, kind=kind, source=source,
                gs_density_order=gs_density_order, is_alpha=is_alpha
            )
        assert isinstance(system, testcases.TestCase)
        assert case in system.cases
        assert gs_density_order in system.gs_density_orders
        # load the adc data
        data = self._load_data(
            system, method=method, case=case, source=source,
            gs_density_order=gs_density_order, is_alpha=is_alpha
        )
        adc_data = data.get(kind, None)
        if adc_data is None:
            raise ValueError(f"No data available for kind {kind} in {case} "
                             f"{method} {system}.")
        # load the reference state and build a matrix on top
        if "cvs" in case and "cvs" not in method:
            method = f"cvs-{method}"
        refstate = self.refstate(system, case)
        # TODO: here we need to pass gs_density_order to LazyMp once implemented
        assert gs_density_order is None
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
        elif refstate.restricted and kind == "doublet":
            symm = "none"
        elif kind in ["spin_flip", "any"]:
            symm = "none"
        else:
            raise ValueError(f"Unknown kind: {kind}")

        spin_change = determine_spin_change(matrix.method, kind, is_alpha)

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

        if matrix.method.adc_type == "pp":
            return ExcitedStates(states)
        elif matrix.method.adc_type == "ip":
            return DetachedStates(states, is_alpha)
        elif matrix.method.adc_type == "ea":
            return AttachedStates(states, is_alpha)
        else:
            raise ValueError(f"Unknown ADC method: {method.name}")

    def adcc_states(self, system: str, method: str, kind: str,
                    case: str, gs_density_order: Optional[int] = None,
                    is_alpha: Optional[bool] = None
                    ) -> ExcitedStates | AttachedStates | DetachedStates:
        """
        Create an ExcitedStates/AttachedStates/DetachedStates instance for the  
        given test case, method (adcn), reference case (gen/cvs/fc/...), 
        state kind (singlet/triplet/any/...) and optionally gs_density_order 
        (2/3/sigma4+) using the adcc eigenstates and eigenvalues.
        """
        if ("ip" in method or "ea" in method) and is_alpha is None:
            is_alpha = True
        return self._make_mock_adc_state(
            system, method=method, case=case, kind=kind,
            gs_density_order=gs_density_order, source="adcc", is_alpha=is_alpha
        )

    def adcman_states(self, system: str, method: str, kind: str,
                      case: str, gs_density_order: Optional[int] = None,
                      is_alpha: Optional[bool] = None
                      ) -> ExcitedStates | AttachedStates | DetachedStates:
        """
        Create an ExcitedStates/AttachedStates/DetachedStates instance for the  
        given test case, method (adcn), reference case (gen/cvs/fc/...), 
        state kind (singlet/triplet/any/...) and optionally gs_density_order 
        (2/3/sigma4+) using the adcman eigenstates and eigenvalues.
        """
        if ("ip" in method or "ea" in method) and is_alpha is None:
            is_alpha = True
        return self._make_mock_adc_state(
            system, method=method, case=case, kind=kind,
            gs_density_order=gs_density_order, source="adcman",
            is_alpha=is_alpha
        )


testdata_cache = TestdataCache()


def read_json_data(name: str) -> dict:
    """Import the json file from the data directory."""
    jsonfile = Path(__file__).parent / _testdata_dirname / name
    if not jsonfile.exists():
        raise FileNotFoundError(f"Missing json data file {jsonfile}.")
    return json.load(open(jsonfile, "r"), object_hook=_import_hook)


def _import_hook(data: dict) -> dict:
    return {key: np.array(val) if isinstance(val, list) else val
            for key, val in data.items()}


psi4_data = read_json_data("psi4_data.json")
pyscf_data = read_json_data("pyscf_data.json")
tmole_data = read_json_data("tmole_data.json")
