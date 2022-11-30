#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import adcc
from adcc.State2States import State2States
import numpy as np

import h5py


def dump_reference_adcc(data, method, dumpfile, mp_tree="mp", adc_tree="adc",
                        n_states_full=None, n_guess_singles=None, **kwargs):
    if isinstance(dumpfile, h5py.File):
        out = dumpfile
    elif isinstance(dumpfile, str):
        out = h5py.File(dumpfile, "w")
    else:
        raise TypeError("Unknown type for out, only HDF5 file and str supported.")

    states = []
    if "n_states" in kwargs or "n_spin_flip" in kwargs:
        state = adcc.run_adc(data, method=method, **kwargs)
        states.append(state)
    else:
        n_singlets = kwargs.pop("n_singlets", 0)
        n_triplets = kwargs.pop("n_triplets", 0)
        if n_singlets:
            state = adcc.run_adc(data, method=method, n_singlets=n_singlets,
                                 n_guesses=n_guess_singles, **kwargs)
            states.append(state)
        if n_triplets:
            state = adcc.run_adc(data, method=method, n_triplets=n_triplets,
                                 n_guesses=n_guess_singles, **kwargs)
            states.append(state)

    if not len(states):
        raise ValueError("No excited states obtained.")

    #
    # MP
    #
    mp = out.create_group(mp_tree)

    # obtain ground state
    ground_state = states[0].ground_state

    mp["mp2/energy"] = ground_state.energy_correction(2)
    if "cvs" not in method:
        # TODO: MP3 energy correction missing in adcc for cvs
        mp["mp3/energy"] = ground_state.energy_correction(3)

    mp["mp2/dipole"] = ground_state.dipole_moment(level=2)
    mp.create_dataset("mp1/t_o1o1v1v1",
                      data=ground_state.t2("o1o1v1v1").to_ndarray(),
                      compression=8)
    mp.create_dataset("mp1/df_o1v1", data=ground_state.df("o1v1").to_ndarray(),
                      compression=8)
    if "cvs" not in method:
        # TODO: missing in adcc for cvs
        mp.create_dataset("mp2/td_o1o1v1v1",
                          data=ground_state.td2("o1o1v1v1").to_ndarray(),
                          compression=8)
    if ground_state.has_core_occupied_space:
        mp.create_dataset("mp1/t_o2o2v1v1",
                          data=ground_state.t2("o2o2v1v1").to_ndarray(),
                          compression=8)
        mp.create_dataset("mp1/t_o1o2v1v1",
                          data=ground_state.t2("o1o2v1v1").to_ndarray(),
                          compression=8)
        mp.create_dataset("mp1/df_o2v1",
                          data=ground_state.df("o2v1").to_ndarray(),
                          compression=8)

    for block in ["dm_o1o1", "dm_o1v1", "dm_v1v1", "dm_o2o1", "dm_o2o2", "dm_o2v1"]:
        blk = block.split("_")[-1]
        if blk in ground_state.mp2_diffdm.blocks:
            mp.create_dataset("mp2/" + block, compression=8,
                              data=ground_state.mp2_diffdm[blk].to_ndarray())
    dm_bb_a, dm_bb_b = ground_state.mp2_diffdm.to_ao_basis(
        ground_state.reference_state
    )
    mp.create_dataset("mp2/dm_bb_a", compression=8, data=dm_bb_a.to_ndarray())
    mp.create_dataset("mp2/dm_bb_b", compression=8, data=dm_bb_b.to_ndarray())

    #
    # ADC
    #
    adc = out.create_group(adc_tree)

    # Data for matrix tests
    gmatrix = adc.create_group("matrix")
    random_vector = adcc.copy(states[0].excitation_vector[0]).set_random()

    # Compute matvec and block-wise apply
    matvec = states[0].matrix.matvec(random_vector)
    result = {}
    result["ss"] = states[0].matrix.block_apply("ph_ph", random_vector.ph)
    if "pphh" in random_vector.keys():
        result["ds"] = states[0].matrix.block_apply("pphh_ph",
                                                    random_vector.ph)
        result["sd"] = states[0].matrix.block_apply("ph_pphh",
                                                    random_vector.pphh)
        result["dd"] = states[0].matrix.block_apply("pphh_pphh",
                                                    random_vector.pphh)

    gmatrix["random_singles"] = random_vector.ph.to_ndarray()
    gmatrix["diagonal_singles"] = states[0].matrix.diagonal().ph.to_ndarray()
    gmatrix["matvec_singles"] = matvec.ph.to_ndarray()
    gmatrix["result_ss"] = result["ss"].to_ndarray()
    if "pphh" in random_vector.keys():
        gmatrix.create_dataset("random_doubles", compression=8,
                               data=random_vector.pphh.to_ndarray())
        gmatrix.create_dataset("diagonal_doubles", compression=8,
                               data=states[0].matrix.diagonal().pphh.to_ndarray())
        gmatrix.create_dataset("matvec_doubles", compression=8,
                               data=matvec.pphh.to_ndarray())
        if "ds" in result:
            gmatrix.create_dataset("result_ds", compression=8,
                                   data=result["ds"].to_ndarray())
        if "sd" in result:
            gmatrix["result_sd"] = result["sd"].to_ndarray()
        if "dd" in result:
            gmatrix.create_dataset("result_dd", compression=8,
                                   data=result["dd"].to_ndarray())

    available_kinds = []
    for state in states:
        assert state.converged
        available_kinds.append(state.kind)
        dm_bb_a = []
        dm_bb_b = []
        tdm_bb_a = []
        tdm_bb_b = []
        eigenvectors_singles = []
        eigenvectors_doubles = []
        n_states = state.excitation_energy.size

        # Up to n_states_extract states we save everything
        if n_states_full is not None:
            n_states_extract = min(n_states_full, n_states)
        else:
            n_states_extract = n_states

        for i in range(n_states_extract):
            bb_a, bb_b = state.state_diffdm[i].to_ao_basis(state.reference_state)
            dm_bb_a.append(bb_a.to_ndarray())
            dm_bb_b.append(bb_b.to_ndarray())
            bb_a, bb_b = state.transition_dm[i].to_ao_basis(state.reference_state)
            tdm_bb_a.append(bb_a.to_ndarray())
            tdm_bb_b.append(bb_b.to_ndarray())

            eigenvectors_singles.append(
                state.excitation_vector[i].ph.to_ndarray()
            )
            if "pphh" in state.excitation_vector[i].keys():
                eigenvectors_doubles.append(
                    state.excitation_vector[i].pphh.to_ndarray()
                )
            else:
                eigenvectors_doubles.clear()

        kind = state.kind
        # Transform to numpy array
        adc[kind + "/state_diffdm_bb_a"] = np.asarray(dm_bb_a)
        adc[kind + "/state_diffdm_bb_b"] = np.asarray(dm_bb_b)
        adc[kind + "/ground_to_excited_tdm_bb_a"] = np.asarray(tdm_bb_a)
        adc[kind + "/ground_to_excited_tdm_bb_b"] = np.asarray(tdm_bb_b)
        adc[kind + "/state_dipole_moments"] = state.state_dipole_moment
        adc[kind + "/transition_dipole_moments"] = state.transition_dipole_moment
        adc[kind + "/transition_dipole_moments_velocity"] = \
            state.transition_dipole_moment_velocity
        adc[kind + "/transition_magnetic_dipole_moments"] = \
            state.transition_magnetic_dipole_moment
        adc[kind + "/eigenvalues"] = state.excitation_energy
        adc[kind + "/eigenvectors_singles"] = np.asarray(
            eigenvectors_singles)

        if eigenvectors_doubles:  # For ADC(0) and ADC(1) there are no doubles
            adc.create_dataset(kind + "/eigenvectors_doubles",
                               compression=8,
                               data=np.asarray(eigenvectors_doubles))

        # TODO State-to-state properties not implemented for CVS
        if "cvs" not in method:
            s2s = adc.create_group(kind + "/state_to_state")
            for ifrom in range(n_states - 1):
                state2state = State2States(state, initial=ifrom)
                tdm_bb_a = []
                tdm_bb_b = []
                transition_dipoles = []

                for j, ito in enumerate(range(ifrom + 1, n_states)):
                    if ito <= n_states_extract and ifrom <= n_states_extract:
                        bb_a, bb_b = state2state.transition_dm[j].to_ao_basis(
                            state.reference_state
                        )
                        tdm_bb_a.append(bb_a.to_ndarray())
                        tdm_bb_b.append(bb_b.to_ndarray())
                    transition_dipoles.append(
                        state2state.transition_dipole_moment[j]
                    )

                s2s_from = s2s.create_group(f"from_{ifrom}")
                s2s_from["transition_dipole_moments"] = \
                    np.asarray(transition_dipoles)
                if tdm_bb_a and tdm_bb_b:
                    s2s_from["state_to_excited_tdm_bb_a"] = np.asarray(tdm_bb_a)
                    s2s_from["state_to_excited_tdm_bb_b"] = np.asarray(tdm_bb_b)

    # Store which kinds are available
    out.create_dataset("available_kinds", shape=(len(available_kinds), ),
                       data=np.array(available_kinds,
                                     dtype=h5py.special_dtype(vlen=str)))

    return out
