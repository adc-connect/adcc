#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import os
import ast
import sys

from os.path import dirname, join
from adcc.MoSpaces import expand_spaceargs

sys.path.insert(0, join(dirname(__file__), "adcc-testdata"))

import adcctestdata as atd  # noqa: E402

#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by Michael F. Herbst
##
## This file is part of adcc-testdata.
##
## adcc-testdata is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc-testdata is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc-testdata. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import numpy as np

import h5py
import adcc


def dump_reference_adcc(data, method, dumpfile, mp_tree="mp", adc_tree="adc",
                        n_states_full=None, **kwargs):
    if isinstance(dumpfile, h5py.File):
        out = dumpfile
    elif isinstance(dumpfile, str):
        out = h5py.File(dumpfile, "w")
    else:
        raise TypeError("Unknown type for out, only HDF5 file and str supported.")

    # TODO: splin-flip, etc...
    states = []
    if "n_states" in kwargs:
        state = adcc.run_adc(data, method=method, **kwargs)
        states.append(state)
    else:
        n_singlets = kwargs.get("n_singlets", 0)
        n_triplets = kwargs.get("n_triplets", 0)
        kwargs.pop("n_singlets", None)
        kwargs.pop("n_triplets", None)
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
    # TODO: is this the total energy or energy correction?
    mp["mp2/energy"] = ground_state.energy(2)
    mp["mp3/energy"] = ground_state.energy(3)

    mp["mp2/dipole"] = ground_state.dipole_moment(level=2)

    # for key in ["mp1/t_o1o1v1v1", "mp1/t_o2o2v1v1", "mp1/t_o1o2v1v1",
    #             "mp1/df_o1v1", "mp1/df_o2v1", "mp2/td_o1o1v1v1"]:
    mp.create_dataset("mp1/t_o1o1v1v1",
                      data=ground_state.t2("o1o1v1v1").to_ndarray(),
                      compression=8)
    mp.create_dataset("mp1/df_o1v1", data=ground_state.df("o1v1").to_ndarray(),
                      compression=8)
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

    # TODO: take this into account above when creating adcc results
    available_kinds = []
    # kind_trees = {
    #     "singlet": method_tree + "/rhf/singlets/0",
    #     "triplet": method_tree + "/rhf/triplets/0",
    #     "state": method_tree + "/uhf/0",
    #     "spin_flip": method_tree + "/uhf/0",
    # }
    # if "n_spin_flip" not in kwargs:
    #     del kind_trees["spin_flip"]
    # if "n_states" not in kwargs:
    #     del kind_trees["state"]

    for state in states:
        assert state.converged
        kind = state.kind
        available_kinds.append(kind)
        dm_bb_a = []
        dm_bb_b = []
        tdm_bb_a = []
        tdm_bb_b = []
        state_dipoles = []
        transition_dipoles = []
        transition_dipoles_vel = []
        # TODO: add new properties
        transition_magnetic_dipoles = []
        eigenvalues = []
        eigenvectors_singles = []
        eigenvectors_doubles = []
        n_states = state.excitation_energies.size

        # Up to n_states_extract states we save everything
        if n_states_full is not None:
            n_states_extract = min(n_states_full, n_states)
        else:
            n_states_extract = n_states

        for i in range(n_states_extract):
            bb_a, bb_b = state.state_diffdms[i].transform_to_ao_basis(state.reference_state)
            dm_bb_a.append(bb_a.to_ndarray())
            dm_bb_b.append(bb_b.to_ndarray())
            bb_a, bb_b = state.transition_dms[i].transform_to_ao_basis(state.reference_state)
            tdm_bb_a.append(bb_a.to_ndarray())
            tdm_bb_b.append(bb_b.to_ndarray())

            state_dipoles.append(state.state_dipole_moments[i])
            transition_dipoles.append(state.transition_dipole_moments[i])
            transition_dipoles_vel.append(
                state.transition_dipole_moments_velocity[i]
            )
            transition_magnetic_dipoles.append(
                state.transition_magnetic_dipole_moments[i]
            )

            eigenvalues.append(state.excitation_energies[i])
            eigenvectors_singles.append(
                state.excitation_vectors[i]['s'].to_ndarray()
            )
            if 'd' in state.excitation_vectors[i].blocks:
                eigenvectors_doubles.append(
                    state.excitation_vectors[i]['d'].to_ndarray()
                )
            else:
                eigenvectors_doubles.clear()

        # from the others only the energy and dipoles
        for i in range(n_states_extract, n_states):
            state_dipoles.append(state.state_dipole_moments[i])
            transition_dipoles.append(state.transition_dipole_moments[i])
            transition_magnetic_dipoles.append(
                state.transition_magnetic_dipole_moments[i]
            )
            transition_dipoles_vel.append(
                state.transition_dipole_moments_velocity[i]
            )
            eigenvalues.append(state.excitation_energies[i])

        # Transform to numpy array
        adc[kind + "/state_diffdm_bb_a"] = np.asarray(dm_bb_a)
        adc[kind + "/state_diffdm_bb_b"] = np.asarray(dm_bb_b)
        adc[kind + "/ground_to_excited_tdm_bb_a"] = np.asarray(tdm_bb_a)
        adc[kind + "/ground_to_excited_tdm_bb_b"] = np.asarray(tdm_bb_b)
        adc[kind + "/state_dipole_moments"] = np.asarray(state_dipoles)
        adc[kind + "/transition_dipole_moments"] = np.asarray(
            transition_dipoles)
        adc[kind + "/transition_dipole_moments_velocity"] = np.asarray(
            transition_dipoles_vel)
        adc[kind + "/transition_magnetic_dipole_moments"] = np.asarray(
            transition_magnetic_dipoles)
        adc[kind + "/eigenvalues"] = np.array(eigenvalues)
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


def dump_all_adcc(case, kwargs, kwargs_overwrite={}, spec="gen"):
    assert spec in ["gen", "cvs"]
    for method in ["adc0", "adc1", "adc2", "adc2x", "adc3"]:
        kw = kwargs_overwrite.get(method, kwargs)
        dump_method_adcc(case, method, kw, spec)


def dump_method_adcc(case, method, kwargs, spec):
    h5file = case + "_hfdata.hdf5"
    if not os.path.isfile(h5file):
        raise ValueError("HfData not found: " + h5file)
    hfdata = atd.HdfProvider(h5file)

    # Get dictionary of parameters for the reference cases.
    refcases = ast.literal_eval(hfdata.data["reference_cases"][()])
    kwargs = dict(kwargs)
    kwargs.update(expand_spaceargs(hfdata, **refcases[spec]))

    fullmethod = method
    if "cvs" in spec:
        fullmethod = "cvs-" + method

    prefix = ""
    if spec != "gen":
        prefix = spec.replace("-", "_") + "_"
    adc_tree = prefix.replace("_", "-") + method
    mp_tree = prefix.replace("_", "-") + "mp"

    dumpfile = "{}_reference_{}{}.hdf5".format(case, prefix, method)
    print(dumpfile)
    if not os.path.isfile(dumpfile):
        print(kwargs)
        dump_reference_adcc(h5file, fullmethod, dumpfile, mp_tree=mp_tree,
                            adc_tree=adc_tree, n_states_full=2, **kwargs)


def dump_methox_sto3g():  # (R)-2-methyloxirane
    kwargs = {"n_singlets": 2} # "n_singlets": 5, "n_triplets": 5}
    # overwrite = {"adc2": {"n_singlets": 9, "n_triplets": 10}, }
    overwrite = {}
    dump_all_adcc("methox_sto3g", kwargs, overwrite, spec="gen")
    # TODO: not working
    # dump_all_adcc("methox_sto3g", kwargs, overwrite, spec="cvs")


# def dump_h2o_def2tzvp():  # H2O restricted
#     kwargs = {"n_singlets": 3, "n_triplets": 3, "n_guess_singles": 6,
#               "max_subspace": 24}
#     dump_all("h2o_def2tzvp", kwargs, spec="gen")


def main():
    dump_methox_sto3g()


if __name__ == "__main__":
    main()
