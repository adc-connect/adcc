from import_qchem_data import _mp_data, _excited_state_data
import test_cases

from adcc.AdcMethod import AdcMethod
from adcc.hdf5io import emplace_dict

from collections import defaultdict
from pathlib import Path
import h5py

_testdata_dirname = "data"


def dump_excited_states(test_case: test_cases.TestCase, method: AdcMethod,
                        state_data: dict) -> None:
    """
    Dump the excited states data (excitation energies, amplitude vectors, ...)
    to a HDF5 file.
    This function assumes that the data has been imported with
    "import_qchem_data.import_excited_states", i.e.,
    state data should be organized as
    {state_kind: {state_n: data}},
    where data is itself a dict that already uses the correct keys to store
    the data. Therefore, this function only collects the data from the
    different states in a list before dumping.
    """
    valid_keys = tuple(
        key for treemap in _excited_state_data.values()
        for key in treemap.values()
    )
    datadir = Path(__file__).parent.parent / _testdata_dirname
    fname = datadir / f"{test_case.file_name}_adcman_{method.name}.hdf5"
    hdf5_file = h5py.File(fname, "w")
    for kind, kind_data in state_data.items():
        data_to_dump = defaultdict(list)
        # sort the kind data to start with the lowest excited state
        # this way we can just append the values to the corresponding list
        for _, state_data in sorted(kind_data.items()):
            assert all(key in valid_keys for key in state_data)
            for key, value in state_data.items():
                data_to_dump[key].append(value)
        kind_group = hdf5_file.create_group(kind)
        emplace_dict(
            data_to_dump, kind_group, compression="gzip", compression_opts=8
        )
