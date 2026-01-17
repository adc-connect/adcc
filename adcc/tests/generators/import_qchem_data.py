from adcc.AdcMethod import AdcMethod
from adcc.hdf5io import _extract_dataset

from collections.abc import Hashable
from typing import Any
import numpy as np
import h5py


class DataImportError(ValueError):
    pass


def import_groundstate(context: h5py.File, only_full_mode: bool,
                       dims_pref: str = "dims/") -> dict:
    """
    Import the MP ground state.

    Parameters
    ----------
    context: h5py.File
        The hdf5 to import from.
    only_full_mode: bool
        Indicates whether the data to read is for a test case that is only
        run in full mode. Since these systems are typically quite large
        one might not want to import all data for these cases.
    dims_pref: str, optional
        Since tensors are exported as flattened array, the dimensions of the
        tensors are exported too. The dimensions can be found by adding the given
        prefix to the context tree of the object.
    """
    # import all available ground state data
    data_to_read = {
        path: key for path, key in _mp_data.items() if path in context
    }
    if not only_full_mode:
        addition = {
            path: key for path, key in _mp_data_large.items() if path in context
        }
        assert not addition.keys() & data_to_read.keys()
        data_to_read.update(addition)
        del addition
    return import_data(context, dims_pref=dims_pref, **data_to_read)


def import_excited_states(context: h5py.File, method: AdcMethod,
                          only_full_mode: bool,
                          is_spin_flip: bool = False,
                          import_nstates: int | None = None,
                          dims_pref: str = "dims/") -> dict:
    """
    Import the excited states data (excitation energies, amplitude vectors, ...)
    from the context.

    Parameters
    ----------
    context: h5py.File
        The hdf5 file to import from.
    method: AdcMethod
        The adc method, e.g., adc2 or adc3
    only_full_mode: bool
        Indicates whether the data to read is for a test case that only
        runs in full mode. Since these systems are typically quite large
        one might not want to import all data for these cases.
    is_spin_flip: bool, optional
        Indicates whether the calculation was a spin flip calculation.
    import_nstates: int, optional
        Only import the first n states from the context.
    dims_pref: str, optional
        Since tensors are exported as flattened array, the dimensions of the
        tensors are exported too. The dimensions can be found by adding the given
        prefix to the context tree of the object.
    """
    # define the possible state kinds to import for each adc variant.
    state_kinds = {
        "pp": [
            ("singlets", True), ("triplets", True),  # restricted
            # uhf and spin flip are located in the same ".../uhf/..." subtree
            ("any_or_spinflip", False),  # unrestricted
        ]
    }
    # Of course the kinds have to have a slightly different name in adcc....
    kind_map = {"singlets": "singlet", "triplets": "triplet"}
    # also the adcc methods have to be translated
    method_name: str = method.name.replace("-", "_")  # cvs-adcn -> cvs_adcn
    if method_name.endswith("adc2"):  # adc2 -> adc2s
        method_name += "s"
    # go through the different possible state kinds and import the states.
    data = {}
    for kind, restricted in state_kinds[method.adc_type]:
        states = _import_excited_states(
            context, method=method_name, only_full_mode=only_full_mode,
            adc_type=method.adc_type, import_nstates=import_nstates,
            state_kind=kind, restricted=restricted, dims_pref=dims_pref
        )
        if states is None:  # no states of the given kind available
            continue
        # since we have states, state-to-state data has to be available
        # -> if we have more than 1 state!
        if len(states["eigenvalues"]) > 1:
            state_to_state = _import_state_to_state_data(
                context, method=method_name, only_full_mode=only_full_mode,
                adc_type=method.adc_type, import_nstates=import_nstates,
                state_kind=kind, restricted=restricted, dims_pref=dims_pref
            )
            states["state_to_state"] = state_to_state

        if kind == "any_or_spinflip":
            kind = "spin_flip" if is_spin_flip else "any"
        data[kind_map.get(kind, kind)] = states
    if not data:
        raise RuntimeError(f"Could not find any states for {method.name} in "
                           f"{context.filename}.")
    return data


def _import_excited_states(context: h5py.File, method: str, only_full_mode: bool,
                           adc_type: str = "pp",
                           import_nstates: int | None = None,
                           state_kind: str | None = None, restricted: bool = True,
                           dims_pref: str = "dims/") -> None | dict[str, Any]:
    """
    Import the excited states data (excitation energies, amplitude vectors, ...)
    from the context.
    The data is restructured during import such that we have a list containing
    the data for all states per imported property.

    Parameters
    ----------
    context: h5py.File
        The hdf5 file to import from.
    method: str
        The adc method, e.g., adc2 or adc3
    only_full_mode: bool
        Indicates whether the data to read is for a test case that only
        runs in full mode. Since these systems are typically quite large
        one might not want to import all data for these cases.
    adc_type: str, optional
        Which type of adc calculation has been performed, e.g., pp.
    import_nstates: int, optional
        Only import the first n states from the context.
    state_kind: str, optional
        The multiplicity of the states, e.g., singlet or triplet for restricted
        pp-adc calculations.
    restricted: bool, optional
        Whether the adc calculation is based on a restricted reference state.
    dims_pref: str, optional
        Since tensors are exported as flattened array, the dimensions of the
        tensors are exported too. The dimensions can be found by adding the given
        prefix to the context tree of the object.
    """
    # build the path under which to find the exicted states
    tree = [f"adc_{adc_type}", method]
    if restricted:
        assert state_kind is not None  # needs to be defined for restricted calcs
        tree.extend(["rhf", state_kind])
    else:
        tree.append("uhf")
    tree.append("0")  # we assume that we only have a single irrep!!
    tree = "/".join(tree)
    # check that we have states to read and return if not
    if f"{tree}/nstates" not in context:
        return None
    _, n_states = _extract_dataset(context[f"{tree}/nstates"])
    if not isinstance(n_states, int):
        n_states = int(n_states)
    if import_nstates is not None:
        n_states = min(n_states, import_nstates)
    # go through the states and gather the tree paths to read and import from the
    # context.
    data_to_read = {}
    for n in range(n_states):
        state_tree = tree + f"/es{n}"
        # ensure that the state is converged
        _, converged = _extract_dataset(context[f"{state_tree}/converged"])
        if not converged:
            raise DataImportError(f"State {n} of kind {state_kind} in file "
                                  f"{context.filename} is not converged.")
        data_to_read.update({
            f"{state_tree}/{path}": (n, key)
            for path, key in _excited_state_data["required"].items()
        })
        data_to_read.update({
            f"{state_tree}/{path}": (n, key)
            for path, key in _excited_state_data["optional"].items()
            if f"{state_tree}/{path}" in context
        })
        if not only_full_mode:
            # update data_to_read depending on the required structure of
            # the dict
            assert not _excited_state_data_large
    # read and import the objects
    raw_data = import_data(context, dims_pref=dims_pref, **data_to_read)
    # collect the data for each property in a list
    # sort the raw_data to ensure that we always start with the lowest state.
    data: dict[str, list] = {}
    for (_, key), val in sorted(raw_data.items()):
        if key not in data:
            data[key] = []
        data[key].append(val)
    # ensure that we did converge on a nonzero result
    if any(abs(e) < 1e-12 for e in data["eigenvalues"]):
        raise DataImportError("Eigenvalue < 1e-12 detected. Calculation converged "
                              "towards zero.")
    # convert to numpy array!
    return {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()}


def _import_state_to_state_data(context: h5py.File, method: str,
                                only_full_mode: bool,
                                adc_type: str = "pp",
                                import_nstates: int | None = None,
                                state_kind: str | None = None,
                                restricted: bool = True,
                                dims_pref: str = "dims/"
                                ) -> dict[str, dict[str, Any]]:
    """
    Import the state-to-state data (tdms, transition dipole moments, ...)
    from the context.
    The data is restructured during import such that we have a list containing
    the state-to-state data for each state per imported property.

    Parameters
    ----------
    context: h5py.File
        The hdf5 file to import from.
    method: str
        The adc method, e.g., adc2 or adc3
    only_full_mode: bool
        Indicates whether the data to read is for a test case that only
        runs in full mode. Since these systems are typically quite large
        one might not want to import all data for these cases.
    adc_type: str, optional
        Which type of adc calculation has been performed, e.g., pp.
    import_nstates: int, optional
        Only import the first n states from the context.
    state_kind: str, optional
        The multiplicity of the states, e.g., singlet or triplet for restricted
        pp-adc calculations.
    restricted: bool, optional
        Whether the adc calculation is based on a restricted reference state.
    dims_pref: str, optional
        Since tensors are exported as flattened array, the dimensions of the
        tensors are exported too. The dimensions can be found by adding the given
        prefix to the context tree of the object.
    """
    # build the path under which to look for the state-to-state data.
    tree = [f"adc_{adc_type}", method]
    if restricted:
        assert state_kind is not None  # needs to be defined for restricted calcs
        tree.extend(["rhf", "isr", state_kind])
    else:
        tree.extend(["uhf", "isr"])
    tree.append("0-0")  # we assume that we only have a single irrep!
    tree = "/".join(tree)
    if tree not in context:
        raise DataImportError("No state-to-state ISR data available.")
    # adcman stores the state-to-state data for a state pair using keys of the
    # form: ito-from. For instance,
    # '1-0', '2-0', '2-1'
    # for 3 available states.
    # -> determine the number of available states from the keys
    s2s_data = context[tree]
    assert isinstance(s2s_data, h5py.Group)
    n_states = max(int(key.split("-")[0]) for key in s2s_data.keys()) + 1
    if import_nstates is not None:
        n_states = min(n_states, import_nstates)
    del s2s_data

    data: dict[str, dict[str, Any]] = {}
    for ifrom in range(n_states - 1):
        data_to_read = {
            f"{tree}/{ito}-{ifrom}/{path}": (ito, key)
            for ito in range(ifrom + 1, n_states)
            for path, key in _state_to_state_data.items()
        }
        if not only_full_mode:
            # update data to read depending on the required structure
            # of the dict
            assert not _state_to_state_data_large
        raw_data = import_data(context, dims_pref=dims_pref, **data_to_read)
        # collect the data in a list
        # sort the data to ensure that we start with the lowest ito
        ifrom_data: dict[str, list] = {}
        for (_, key), val in sorted(raw_data.items()):
            if key not in ifrom_data:
                ifrom_data[key] = []
            ifrom_data[key].append(val)
        # convert the lists to numpy array
        data[f"from_{ifrom}"] = {k: np.array(v) if isinstance(v, list) else v
                                 for k, v in ifrom_data.items()}
    return data


def import_data(context: h5py.File, dims_pref: str = "dims/",
                **kwargs: Hashable) -> dict:
    """
    Read and import data from the dumped libctx context.
    The data to import can be defined through kwargs in the form
    "mp1/df_o1v1" = "mp1/df",
    where "mp1/df_o1v1" defines the tree in the libctx context, while "mp1/df"
    is the key under which the imported object is placed in the returned dict.
    If a object can not be found in the context, an exception is raised.
    Since arrays are exported as flattened vectors, the dimensions of the array
    are assumed to be available under "dims/mp1/df_o1v1", where "dims/" is a prefix
    that is added to the context path of the object.
    """
    data = {}
    for context_tree, key in kwargs.items():
        # load the value from the context
        raw_value = context.get(context_tree, None)
        if raw_value is None:
            raise KeyError(f"Missing required context entry: {context_tree}.")
        # import the object
        assert isinstance(raw_value, h5py.Dataset)
        _, value = _extract_dataset(raw_value)
        # for numpy arrays we might have to reshape when we find a matching
        # dimension object
        dim_path = dims_pref + context_tree
        if isinstance(value, np.ndarray) and dim_path in context:
            # import the dimensions object
            _, dims = _extract_dataset(context[dim_path])
            value = value.reshape(dims)
        if key in data:
            raise DataImportError(f"The key {key} is not unique. Overwriting "
                                  "data during import.")
        data[key] = value
    return data


###################################################################################
# Definition of pairs of a tree path to read from the context and a key under which
# the imported objects are placed after the import.
# The tree paths are separated in required keys that have to be available and
# optional keys that may not be available, because the objects are only
# computed for certain methods.
_excited_state_data = {
    "required": {
        "energy": "eigenvalues",
        # diff dm in the AO basis
        "opdm/dm_bb_a": "state_diffdm_bb_a",
        "opdm/dm_bb_b": "state_diffdm_bb_b",
        # excited state dipole moment (vector)
        "prop/dipole": "state_dipole_moments",
        # the singles part of the amplitude vector
        "u1": "eigenvectors_singles"
    },
    "optional": {
        # transition dm in the AO basis:
        # not computed for Singlet -> Triplet excitations
        "optdm/dm_bb_a": "ground_to_excited_tdm_bb_a",
        "optdm/dm_bb_b": "ground_to_excited_tdm_bb_b",
        # transition dipole moment (vector): only when we have a optdm
        "tprop/dipole": "transition_dipole_moments",
        # doubles and triples part of the amplitude vector
        "u2": "eigenvectors_doubles",
        "u3": "eigenvectors_triples",
        # PE data
        "prop/e_pe_ptSS": "pe_ptss_correction",
        "tprop/e_pe_ptLR": "pe_ptlr_correction",
    }
}
# The large excited state data that is only imported for small test cases
# that not only run in full mode.
_excited_state_data_large = {}

# The state-to-state ISR data to import for each pair of states.
# All keys are required.
_state_to_state_data = {
    "dipole": "transition_dipole_moments",
    "optdm/dm_bb_a": "state_to_excited_tdm_bb_a",
    "optdm/dm_bb_b": "state_to_excited_tdm_bb_b",
}
# The large state-to-state ISR data that is only imported for small
# test cases that not only run in full mode
_state_to_state_data_large = {}

# The available MP data depends on the adc method and order
# -> treat all MP data as optional and import everything that is available
_mp_data = {
    # MP1
    "mp1/df_o1v1": "mp1/df_o1v1",
    "mp1/t_o1o1v1v1": "mp1/t_o1o1v1v1",
    # CVS-MP1
    "mp1/df_o2v1": "mp1/df_o2v1",
    "mp1/t_o1o2v1v1": "mp1/t_o1o2v1v1",
    "mp1/t_o2o2v1v1": "mp1/t_o2o2v1v1",
    # MP2
    "mp2/energy": "mp2/energy",
    # MP2 density in the AO basis
    "mp2/opdm/dm_bb_a": "mp2/dm_bb_a",
    "mp2/opdm/dm_bb_b": "mp2/dm_bb_b",
    # MP2 density in the MO basis
    "mp2/opdm/dm_o1o1": "mp2/dm_o1o1",
    "mp2/opdm/dm_o1v1": "mp2/dm_o1v1",
    "mp2/opdm/dm_v1v1": "mp2/dm_v1v1",
    # CVS-MP2 density in the MO basis (additional blocks)
    "mp2/opdm/dm_o2o1": "mp2/dm_o2o1",
    "mp2/opdm/dm_o2o2": "mp2/dm_o2o2",
    "mp2/opdm/dm_o2v1": "mp2/dm_o2v1",
    # MP2 dipole vector
    "mp2/prop/dipole": "mp2/dipole",
    # MP2 doubles amplitudes
    "mp2/td_o1o1v1v1": "mp2/td_o1o1v1v1",
    # MP3
    "mp3/energy": "mp3/energy",
    # MP3 density in the AO basis
    "mp3/opdm/dm_bb_a": "mp3/dm_bb_a",
    "mp3/opdm/dm_bb_b": "mp3/dm_bb_b",
    # MP3 density in the MO basis
    "mp3/opdm/dm_o1o1": "mp3/dm_o1o1",
    "mp3/opdm/dm_o1v1": "mp3/dm_o1v1",
    "mp3/opdm/dm_v1v1": "mp3/dm_v1v1",
    # MP3 dipole vector
    "mp3/prop/dipole": "mp3/dipole",
    # sigma4+ density in the AO basis
    "sigma4+/opdm/dm_bb_a": "sigma4+/dm_bb_a",
    "sigma4+/opdm/dm_bb_b": "sigma4+/dm_bb_b",
    # sigma4+ density in the MO basis
    "sigma4+/opdm/dm_o1o1": "sigma4+/dm_o1o1",
    "sigma4+/opdm/dm_o1v1": "sigma4+/dm_o1v1",
    "sigma4+/opdm/dm_v1v1": "sigma4+/dm_v1v1",
    # sigma4+ dipole vector
    "sigma4+/prop/dipole": "sigma4+/dipole",
}
# the following quantities are only imported for small test cases that
# not only run in full mode
_mp_data_large = {
    # MP2 triples amplitudes
    "mp2/tt2_o1o1o1v1v1v1": "mp2/tt_o1o1o1v1v1v1",
}
