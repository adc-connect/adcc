from adcc.ExcitedStates import ExcitedStates
from adcc.LazyMp import LazyMp
from adcc.hdf5io import emplace_dict

import h5py


def dump_groundstate(ground_state: LazyMp, hdf5_file: h5py.Group) -> None:
    """
    Dump the MP data to the given hdf5 file/group.
    """
    gs = "mp"

    gs_data = {}
    # MP1 data
    gs_data[f"{gs}1/df_o1v1"] = ground_state.df("o1v1").to_ndarray()
    gs_data[f"{gs}1/t_o1o1v1v1"] = ground_state.t2("o1o1v1v1").to_ndarray()
    # CVS-MP1 data
    if ground_state.has_core_occupied_space:
        gs_data[f"{gs}1/df_o2v1"] = ground_state.df("o2v1").to_ndarray()
        gs_data[f"{gs}1/t_o1o2v1v1"] = ground_state.t2("o1o2v1v1").to_ndarray()
        gs_data[f"{gs}1/t_o2o2v1v1"] = ground_state.t2("o2o2v1v1").to_ndarray()
    # MP2 data
    gs_data[f"{gs}2/energy"] = ground_state.energy_correction(2)
    gs_data[f"{gs}2/dipole"] = ground_state.dipole_moment(2)
    if not ground_state.has_core_occupied_space:
        gs_data[f"{gs}2/td_o1o1v1v1"] = ground_state.td2("o1o1v1v1").to_ndarray()
    # MP3 data
    if not ground_state.has_core_occupied_space:
        gs_data[f"{gs}3/energy"] = ground_state.energy_correction(3)
    # MP2 density: MO basis
    dm_blocks = ["dm_o1o1", "dm_o1v1", "dm_v1v1"]
    if ground_state.has_core_occupied_space:
        dm_blocks.extend(["dm_o2o1", "dm_o2o2", "dm_o2v1"])
    for block in dm_blocks:
        blk = block.split("_")[-1]
        gs_data[f"{gs}2/{block}"] = ground_state.mp2_diffdm[blk].to_ndarray()
    # MP2 density: AO basis
    dm_bb_a, dm_bb_b = ground_state.mp2_diffdm.to_ao_basis(
        ground_state.reference_state
    )
    gs_data[f"{gs}2/dm_bb_a"] = dm_bb_a
    gs_data[f"{gs}2/dm_bb_b"] = dm_bb_b
    # write the data to hdf5
    emplace_dict(gs_data, hdf5_file, compression="gzip")


def dump_excited_states(states: ExcitedStates, hdf5_file: h5py.Group) -> None:
    """
    Dump the excited states data to the given hdf5 file/group.
    """
    raise NotImplementedError()
