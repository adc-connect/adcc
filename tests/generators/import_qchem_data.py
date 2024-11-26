from adcc.AdcMethod import AdcMethod
from adcc.hdf5io import _extract_dataset

from collections import defaultdict
import numpy as np
import h5py


class NotConvergedError(ValueError):
    pass


def import_groundstate(context: h5py.File, dims_pref: str = "dims/") -> dict:
    """
    Import the MP ground state.

    Parameters
    ----------
    context: h5py.File
        The hdf5 to import from.
    dims_pref: str, optional
        Since tensors are exported as flattened array, the dimensions of the
        tensors are exported too. The dimensions can be found by adding the given
        prefix to the context tree of the object.
    """
    # import all available ground state data
    data_to_read = {
        path: key for path, key in _mp_data.items() if path in context
    }
    return import_data(context, dims_pref=dims_pref, **data_to_read)


def import_excited_states(context: h5py.File, method: AdcMethod,
                          import_nstates: int = None, dims_pref: str = "dims/"
                          ) -> dict:
    """
    Import the excited states data (excitation energies, amplitude vectors, ...)
    from the context.

    Parameters
    ----------
    context: h5py.File
        The hdf5 file to import from.
    method: AdcMethod
        The adc method, e.g., adc2 or adc3
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
    method_name: str = method.name.replace("-", "_")  # cvs-adcn -> cvs_adcn
    if method_name.endswith("adc2"):  # adc2 -> adc2s
        method_name += "s"
    # go through the different possible state kinds and import the states.
    data = {}
    for kind, restricted in state_kinds[method.adc_type]:
        states = _import_excited_states(
            context, method=method_name, adc_type=method.adc_type,
            import_nstates=import_nstates, state_kind=kind, restricted=restricted,
            dims_pref=dims_pref
        )
        if states is None:  # no states of the given kind available
            continue
        data[kind] = states
    if not data:
        raise RuntimeError(f"Could not find any states for {method.name} in "
                           f"{context.filename}.")
    return data


def _import_excited_states(context: h5py.File, method: str, adc_type: str = "pp",
                           import_nstates: int = None, state_kind: str = None,
                           restricted: bool = True, dims_pref: str = "dims/"
                           ) -> None | dict[int, dict]:
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
            raise NotConvergedError(f"State {n} of file {context.filename} is not "
                                    "converged.")
        data_to_read.update({
            f"{state_tree}/{path}": (n, key)
            for path, key in _excited_state_data["required"].items()
        })
        data_to_read.update({
            f"{state_tree}/{path}": (n, key)
            for path, key in _excited_state_data["optional"].items()
            if f"{state_tree}/{path}" in context
        })
    # read and import the objects
    raw_data = import_data(context, dims_pref=dims_pref, **data_to_read)
    # collect the data for each property in a list
    # sort the raw_data to ensure that we always start with the lowest state.
    data = defaultdict(list)
    for (_, key), val in sorted(raw_data.items()):
        data[key].append(val)
    return dict(data)


def import_data(context: h5py.File, dims_pref: str = "dims/",
                **kwargs: str) -> dict:
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
        # transition dm in the AO basis
        "optdm/dm_bb_a": "ground_to_excited_tdm_bb_a",
        "optdm/dm_bb_b": "ground_to_excited_tdm_bb_b",
        # excited state dipole moment (vector)
        "prop/dipole": "state_dipole_moments",
        # transition dipole moment (vector)
        "tprop/dipole": "transition_dipole_moments",
        # the singles part of the amplitude vector
        "u1": "eigenvectors_singles"
    },
    "optional": {
        # doubles and triples part of the amplitude vector
        "u2": "eigenvectors_doubles",
        "u3": "eigenvectors_triples"
    }
}

# The available MP data depends on the adc method and order
# -> treat all MP data as optional and import everything that is available
_mp_data = {
    # MP1
    "mp1/df_o1v1": "mp1/df_o1v1",
    "mp1/t_o1o1v1v1": "mp1/t_o1o1v1v1",
    # MP2
    "mp2/energy": "mp2/energy",
    # MP2 density in the AO basis
    "mp2/opdm/dm_bb_a": "mp2/dm_bb_a",
    "mp2/opdm/dm_bb_b": "mp2/dm_bb_b",
    # MP2 density in the MO basis
    "mp2/opdm/dm_o1o1": "mp2/dm_o1o1",
    "mp2/opdm/dm_o1v1": "mp2/dm_o1v1",
    "mp2/opdm/dm_v1v1": "mp2/dm_v1v1",
    # MP2 dipole vector
    "mp2/prop/dipole": "mp2/dipole",
    # MP2 doubles amplitudes
    "mp2/td_o1o1v1v1": "mp2/td_o1o1v1v1",
    # MP3
    "mp3/energy": "mp3/energy",
}
