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
import numpy as np

import h5py


def dump_reference_adcc(data, method, dumpfile, mp_tree="mp", adc_tree="adc",
                        n_states_full=None, **kwargs):
    if isinstance(dumpfile, h5py.File):
        out = dumpfile
    elif isinstance(dumpfile, str):
        out = h5py.File(dumpfile, "w")
    else:
        raise TypeError("Unknown type for out, only HDF5 file and str supported.")

    # TODO: spin-flip, etc.
    states = []
    if "n_states" in kwargs:
        state = adcc.run_adc(data, method=method, **kwargs)
        states.append(state)
    else:
        n_singlets = kwargs.pop("n_singlets", 0)
        n_triplets = kwargs.pop("n_triplets", 0)
        if n_singlets:
            state = adcc.run_adc(data, method=method, n_singlets=n_singlets,
                                 **kwargs)
            states.append(state)
        if n_triplets:
            state = adcc.run_adc(data, method=method, n_triplets=n_triplets,
                                 **kwargs)
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

    for block in ["dm_o1o1", "dm_o1v1", "dm_v1v1", "dm_bb_a", "dm_bb_b",
                  "dm_o2o1", "dm_o2o2", "dm_o2v1"]:
        blk = block.split("_")[-1]
        if blk in ground_state.mp2_diffdm.blocks:
            mp.create_dataset("mp2/" + block, compression=8,
                              data=ground_state.mp2_diffdm[blk].to_ndarray())

    #
    # ADC
    #
    adc = out.create_group(adc_tree)

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
        n_states = state.excitation_energies.size

        # Up to n_states_extract states we save everything
        if n_states_full is not None:
            n_states_extract = min(n_states_full, n_states)
        else:
            n_states_extract = n_states

        for i in range(n_states_extract):
            bb_a, bb_b = state.state_diffdms[i].transform_to_ao_basis(
                state.reference_state
            )
            dm_bb_a.append(bb_a.to_ndarray())
            dm_bb_b.append(bb_b.to_ndarray())
            bb_a, bb_b = state.transition_dms[i].transform_to_ao_basis(
                state.reference_state
            )
            tdm_bb_a.append(bb_a.to_ndarray())
            tdm_bb_b.append(bb_b.to_ndarray())

            eigenvectors_singles.append(
                state.excitation_vectors[i]['s'].to_ndarray()
            )
            if 'd' in state.excitation_vectors[i].blocks:
                eigenvectors_doubles.append(
                    state.excitation_vectors[i]['d'].to_ndarray()
                )
            else:
                eigenvectors_doubles.clear()

        kind = state.kind
        # Transform to numpy array
        adc[kind + "/state_diffdm_bb_a"] = np.asarray(dm_bb_a)
        adc[kind + "/state_diffdm_bb_b"] = np.asarray(dm_bb_b)
        adc[kind + "/ground_to_excited_tdm_bb_a"] = np.asarray(tdm_bb_a)
        adc[kind + "/ground_to_excited_tdm_bb_b"] = np.asarray(tdm_bb_b)
        adc[kind + "/state_dipole_moments"] = state.state_dipole_moments
        adc[kind + "/transition_dipole_moments"] = state.transition_dipole_moments
        adc[kind + "/transition_dipole_moments_velocity"] = \
            state.transition_dipole_moments_velocity
        adc[kind + "/transition_magnetic_dipole_moments"] = \
            state.transition_magnetic_dipole_moments
        adc[kind + "/eigenvalues"] = state.excitation_energies
        adc[kind + "/eigenvectors_singles"] = np.asarray(
            eigenvectors_singles)

        if eigenvectors_doubles:  # For ADC(0) and ADC(1) there are no doubles
            adc.create_dataset(kind + "/eigenvectors_doubles",
                               compression=8,
                               data=np.asarray(eigenvectors_doubles))

    # Store which kinds are available
    out.create_dataset("available_kinds", shape=(len(available_kinds), ),
                       data=np.array(available_kinds,
                                     dtype=h5py.special_dtype(vlen=str)))

    # TODO: dump state2state once in master
    return out
