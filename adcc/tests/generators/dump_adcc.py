import adcc
from adcc.AdcMatrix import AdcMatrix
from adcc.AmplitudeVector import AmplitudeVector
from adcc.ExcitedStates import ExcitedStates
from adcc.hdf5io import emplace_dict
from adcc.LazyMp import LazyMp
from adcc.State2States import State2States

import numpy as np
import h5py


def dump_groundstate(ground_state: LazyMp, hdf5_file: h5py.Group,
                     only_full_mode: bool) -> None:
    """
    Dump the MP data to the given hdf5 file/group. Data is dumped sorted by the
    perturbation theoretical orders of the quantity, e.g., mp1/t_o1o1v1v1 for
    the first order doubles amplitudes.
    The only_full_mode flag indicates whether the underlying test case is only
    run in full mode. Typically tests that run in full mode are larger
    and therefore not all test data might be dumped in that case.
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
    gs_data[f"{gs}2/td_o1o1v1v1"] = ground_state.td2("o1o1v1v1").to_ndarray()
    if not only_full_mode:
        # triples take a lot of memory for the larger test cases
        gs_data[f"{gs}2/tt_o1o1o1v1v1v1"] = (
            ground_state.tt2("o1o1o1v1v1v1").to_ndarray()
        )
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
    gs_data[f"{gs}2/dm_bb_a"] = dm_bb_a.to_ndarray()
    gs_data[f"{gs}2/dm_bb_b"] = dm_bb_b.to_ndarray()
    # write the data to hdf5
    emplace_dict(gs_data, hdf5_file, compression="gzip")
    hdf5_file.attrs["adcc_version"] = adcc.__version__


def dump_excited_states(states: ExcitedStates, hdf5_file: h5py.Group,
                        dump_nstates: int | None = None) -> None:
    """
    Dump the excited states data to the given hdf5 file/group.
    The number of states to dump can be given by dump_nstates. By default all states
    are dumped.
    """
    # ensure that the calculation converged on a nonzero result
    assert states.converged  # type: ignore
    assert all(abs(e) > 1e-12 for e in states.excitation_energy)

    n_states = len(states.excitation_energy)
    if dump_nstates is not None:
        n_states = min(n_states, dump_nstates)

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
            eigenvectors[exdegree + 1].append(getattr(
                states.excitation_vector[n], block  # type: ignore
            ).to_ndarray())
    kind_data = {}
    # eigenvalues
    kind_data["eigenvalues"] = states.excitation_energy[:n_states]
    # state and transition dipole moments
    kind_data["state_dipole_moments"] = states.state_dipole_moment[:n_states]
    kind_data["transition_dipole_moments"] = (
        states.transition_dipole_moment[:n_states]
    )
    kind_data["transition_dipole_moments_velocity"] = (
        states.transition_dipole_moment_velocity[:n_states]
    )

    gauge_origins = ["origin", "mass_center", "charge_center"]
    for g_origin in gauge_origins:
        kind_data[f"transition_magnetic_dipole_moments_{g_origin}"] = (
            states.transition_magnetic_dipole_moment(g_origin)[:n_states]
        )
        kind_data[f"transition_quadrupole_moments_{g_origin}"] = (
            states.transition_quadrupole_moment(g_origin)[:n_states]
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
            # extract the tdms for the desired states
            tdm_bb_a = []
            tdm_bb_b = []
            for j, tdm in enumerate(state2state.transition_dm):
                if ifrom + j + 2 > n_states:  # ifrom + j + 2 = ito
                    break
                bb_a, bb_b = tdm.to_ao_basis(states.reference_state)
                tdm_bb_a.append(bb_a.to_ndarray())
                tdm_bb_b.append(bb_b.to_ndarray())
            kind_data[f"state_to_state/from_{ifrom}/transition_dipole_moments"] = (
                state2state.transition_dipole_moment[:n_states - ifrom - 1]
            )
            kind_data[f"state_to_state/from_{ifrom}/state_to_excited_tdm_bb_a"] = (
                np.asarray(tdm_bb_a)
            )
            kind_data[f"state_to_state/from_{ifrom}/state_to_excited_tdm_bb_b"] = (
                np.asarray(tdm_bb_b)
            )
    # write the data to hdf5
    emplace_dict(kind_data, hdf5_file, compression="gzip")
    hdf5_file.attrs["adcc_version"] = adcc.__version__


def dump_matrix_testdata(matrix: AdcMatrix, trial_vec: AmplitudeVector,
                         hdf5_file: h5py.Group) -> None:
    """
    Dump the testdata to test the adcmatrix equations.
    trial_vec is a random amplitude vector.
    """
    blocks = matrix.axis_blocks  # [singles, doubles, ...]
    singles_singles = f"{blocks[0]}_{blocks[0]}"
    data = {}
    # compute the MVP for individual blocks of the secular matrix.
    assert blocks[0] in trial_vec
    data["result_ss"] = matrix.block_apply(
        singles_singles, trial_vec[blocks[0]]
    ).to_ndarray()
    if len(blocks) > 1:  # we have doubles
        assert blocks[1] in trial_vec
        singles_doubles = f"{blocks[0]}_{blocks[1]}"
        data["result_sd"] = matrix.block_apply(
            singles_doubles, trial_vec[blocks[1]]
        ).to_ndarray()
        doubles_singles = f"{blocks[1]}_{blocks[0]}"
        data["result_ds"] = matrix.block_apply(
            doubles_singles, trial_vec[blocks[0]]
        ).to_ndarray()
        doubles_doubles = f"{blocks[1]}_{blocks[1]}"
        data["result_dd"] = matrix.block_apply(
            doubles_doubles, trial_vec[blocks[1]]
        ).to_ndarray()
    # compute the full mvp
    matvec = matrix.matvec(trial_vec)
    data["matvec_singles"] = matvec[blocks[0]].to_ndarray()  # type: ignore
    if len(blocks) > 1:
        data["matvec_doubles"] = matvec[blocks[1]].to_ndarray()  # type: ignore
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
    hdf5_file.attrs["adcc_version"] = adcc.__version__
