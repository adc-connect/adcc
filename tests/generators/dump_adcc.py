from adcc.AdcMatrix import AdcMatrix
from adcc.AmplitudeVector import AmplitudeVector
from adcc.ExcitedStates import ExcitedStates
from adcc.hdf5io import emplace_dict
from adcc.LazyMp import LazyMp
from adcc.State2States import State2States

import numpy as np
import h5py


def dump_groundstate(ground_state: LazyMp, hdf5_file: h5py.Group) -> None:
    """
    Dump the MP data to the given hdf5 file/group. Data is dumped sorted by the
    perturbation theoretical orders of the quantity, e.g., mp1/t_o1o1v1v1 for
    the first order doubles amplitudes.
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
    Dump the excited states data to the given hdf5 file/group. The excited state
    data is dumped into a kind (singlet, triplet, any, ...) subgroup.
    """
    # TODO: add functionality to only import the first n states. In the original
    # verion we had n_import_full: import all energies but only the first n
    # statedms, tdms, ...
    # TODO: add data for matrix tests (see original dump_reference_adcc.py)
    assert states.converged
    n_states = len(states.excitation_energy)
    dm_bb_a = []  # State diffdm AO basis alpha part
    dm_bb_b = []  # State diffdm AO basis beta part.
    tdm_bb_a = []  # Ground to Excited state tdm AO basis alpha part
    tdm_bb_b = []  # Ground to Excited state tdm AO basis beta part
    # split the eigenvectors according to their excitation degree for all states
    eigenvectors: dict[int, list] = {}
    for n in range(n_states):
        # densities
        bb_a, bb_b = states.state_diffdm[n].to_ao_basis(states.reference_state)
        dm_bb_a.append(bb_a.to_ndarray())
        dm_bb_b.append(bb_b.to_ndarray())
        bb_a, bb_b = states.transition_dm[n].to_ao_basis(states.reference_state)
        tdm_bb_a.append(bb_a.to_ndarray())
        tdm_bb_b.append(bb_b.to_ndarray())
        # eigenvectors
        for exdegree, block in enumerate(states.matrix.axis_blocks):
            if exdegree + 1 not in eigenvectors:
                eigenvectors[exdegree + 1] = []
            eigenvectors[exdegree + 1].append(
                getattr(states.excitation_vector[n], block).to_ndarray()
            )
    kind_data = {}
    # eigenvalues
    kind_data["eigenvalues"] = states.excitation_energy
    # state and transition dipole moments
    kind_data["state_dipole_moments"] = states.state_dipole_moment
    kind_data["transition_dipole_moments"] = states.transition_dipole_moment
    kind_data["transition_dipole_moments_velocity"] = (
        states.transition_dipole_moment_velocity
    )
    kind_data["transition_magnetic_dipole_moments"] = (
        states.transition_magnetic_dipole_moment
    )
    # state diffdm and ground to excited state tdm
    kind_data["state_diffdm_bb_a"] = np.asarray(dm_bb_a)
    kind_data["state_diffdm_bb_b"] = np.asarray(dm_bb_b)
    kind_data["ground_to_excited_tdm_bb_a"] = np.asarray(tdm_bb_a)
    kind_data["ground_to_excited_tdm_bb_b"] = np.asarray(tdm_bb_b)
    # dump the eigenvectors
    kind_data["eigenvectors_singles"] = np.asarray(eigenvectors[1])
    if 2 in eigenvectors:
        kind_data["eigenvectors_doubles"] = np.asarray(eigenvectors[2])
    # state to state tdm: not implemented for CVS
    if not states.method.is_core_valence_separated:
        for ifrom in range(n_states - 1):
            state2state = State2States(states, initial=ifrom)
            # extract the tdms
            tdm_bb_a = []
            tdm_bb_b = []
            for tdm in state2state.transition_dm:
                bb_a, bb_b = tdm.to_ao_basis(states.reference_state)
                tdm_bb_a.append(bb_a.to_ndarray())
                tdm_bb_b.append(bb_b.to_ndarray())
            kind_data[f"state_to_state/from_{ifrom}/transition_dipole_moments"] = (
                state2state.transition_dipole_moment
            )
            kind_data[f"state_to_state/from_{ifrom}/state_to_excited_tdm_bb_a"] = (
                np.asarray(tdm_bb_a)
            )
            kind_data[f"state_to_state/from_{ifrom}/state_to_excited_tdm_bb_b"] = (
                np.asarray(tdm_bb_b)
            )
    # write the data to hdf5
    kind_group = hdf5_file.create_group(states.kind)
    emplace_dict(kind_data, kind_group, compression="gzip")


def dump_matrix_testdata(matrix: AdcMatrix, trial_vec: AmplitudeVector,
                         hdf5_file: h5py.Group) -> None:
    """
    Dump the testdata to test the adcmatrix equations.
    trial_vec is a random amplitude vector.
    """
    blocks = matrix.axis_blocks  # [singles, doubles, ...]
    singles_singles = "_".join(blocks[0], blocks[0])
    data = {}
    # compute the MVP for individual blocks of the secular matrix.
    data["result_ss"] = matrix.block_apply(singles_singles, trial_vec[blocks[0]])
    if len(blocks) > 1:  # we have doubles
        assert blocks[1] in trial_vec
        singles_doubles = "_".join(blocks[0], blocks[1])
        data["result_sd"] = matrix.block_apply(
            singles_doubles, trial_vec[blocks[1]]
        )[blocks[0]].to_ndarray()
        doubles_singles = "_".join(blocks[1], blocks[0])
        data["result_ds"] = matrix.block_apply(
            doubles_singles, trial_vec[blocks[0]]
        )[blocks[1]].to_ndarray()
        doubles_doubles = "_".join(blocks[1], blocks[1])
        data["result_dd"] = matrix.block_apply(
            doubles_doubles, trial_vec[blocks[1]]
        )[blocks[1]].to_ndarray()
    # compute the full mvp
    matvec = matrix.matvec(trial_vec)
    data["matvec_singles"] = matvec[blocks[0]].to_ndarray()
    if len(blocks) > 1:
        data["matvec_doubles"] = matvec[blocks[1]].to_ndarray()
    # compute the diagonal
    data["diagonal_singles"] = matrix.diagonal()[blocks[0]].to_ndarray()
    if len(blocks) > 1:
        data["diagonal_doubles"] = matrix.diagonal()[blocks[1]].to_ndarray()
    # dump the trial vector
    data["random_singles"] = trial_vec[blocks[0]].to_ndarray()
    if len(blocks) > 1:
        data["random_doubles"] = trial_vec[blocks[1]].to_ndarray()
    # write the data to hdf5
    emplace_dict(data, hdf5_file, compression="gzip")
