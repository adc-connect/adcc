#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

from adcc import hdf5io
from adcc.testdata.cache import cache
import adcc
import adcc.solver.adcman as adcman
import numpy as np
import os


def dump_all_methods(key, kwargs_cvs, kwargs_general):
    data = cache.hfdata[key]

    for method in ["adc2", "adc2x", "adc3"]:
        tmethod = method
        if method == "adc2":
            tmethod = "adc2s"

        if method not in ["adc3"]:
            # CVS
            dumpfile = "{}_reference_cvs_{}.hdf5".format(key, method)
            if not os.path.isfile(dumpfile):
                res = run(data, method="cvs-" + method, **kwargs_cvs,
                          core_valence_separation=True)
                dictionary = build_dict(res,
                                        method_tree="adc_pp/cvs_" + tmethod,
                                        out_tree="cvs-" + method)
                hdf5io.save(dumpfile, dictionary)

        # General
        dumpfile = "{}_reference_{}.hdf5".format(key, method)
        if not os.path.isfile(dumpfile):
            res = run(data, method=method, **kwargs_general)
            dictionary = build_dict(res, method_tree="adc_pp/" + tmethod,
                                    out_tree=method)
            hdf5io.save(dumpfile, dictionary)


def run(data, method, n_singlets=None, n_triplets=None,
        n_states=None, core_valence_separation=False):
    n_core_orbitals = None
    if core_valence_separation:
        n_core_orbitals = data["n_core_orbitals"]

    if n_singlets and n_triplets:
        n_guesses = max(n_singlets, n_triplets)
    else:
        n_guesses = n_states

    # Run preliminary calculation
    res = adcc.tmp_run_prelim(data, method, n_guess_singles=n_guesses,
                              n_core_orbitals=n_core_orbitals)

    # Setup the matrix and run adcman solver
    matrix = adcc.AdcMatrix(res.method, res.ground_state)
    states = adcman.jacobi_davidson(matrix, print_level=100,
                                    n_singlets=n_singlets,
                                    n_triplets=n_triplets,
                                    n_states=n_states, conv_tol=1e-9)

    # return the full adcman context
    return states[0].ctx


def build_dict(ctx, method_tree, out_tree, n_states_full=2):
    ret = {}
    ret["mp1/t_o1o1v1v1"] = ctx.at_tensor("/mp1/t_o1o1v1v1").to_ndarray()
    ret["mp1/df_o1v1"] = ctx.at_tensor("/mp1/df_o1v1").to_ndarray()
    ret["mp2/td_o1o1v1v1"] = ctx.at_tensor("/mp2/td_o1o1v1v1").to_ndarray()

    mp_dm = "/mp2/opdm/"
    ret["mp2/dm_o1o1"] = ctx.at_tensor(mp_dm + "dm_o1o1").to_ndarray()
    ret["mp2/dm_o1v1"] = ctx.at_tensor(mp_dm + "dm_o1v1").to_ndarray()
    ret["mp2/dm_v1v1"] = ctx.at_tensor(mp_dm + "dm_v1v1").to_ndarray()
    ret["mp2/dm_bb_a"] = ctx.at_tensor(mp_dm + "dm_bb_a").to_ndarray()
    ret["mp2/dm_bb_b"] = ctx.at_tensor(mp_dm + "dm_bb_b").to_ndarray()

    kind_trees = {
        "singlet": method_tree + "/rhf/singlets/0",
        "triplet": method_tree + "/rhf/triplets/0",
        "state": method_tree + "/uhf/0",
    }

    available_kinds = []
    for kind, tree in kind_trees.items():
        dm_bb_a = []
        dm_bb_b = []
        tdm_bb_a = []
        tdm_bb_b = []
        eigenvalues = []
        eigenvectors_singles = []
        eigenvectors_doubles = []
        n_states = ctx.at_unsigned_long(tree + "/nstates", 0)
        if n_states == 0:
            continue

        available_kinds.append(kind)

        # From 2 states we save everything
        n_states_full = max(n_states_full, n_states)

        for i in range(n_states_full):
            state_tree = tree + "/es" + str(i)

            dm_bb_a.append(
                ctx.at_tensor(state_tree + "/opdm/dm_bb_a").to_ndarray()
            )
            dm_bb_b.append(
                ctx.at_tensor(state_tree + "/opdm/dm_bb_b").to_ndarray()
            )
            tdm_bb_a.append(
                ctx.at_tensor(state_tree + "/optdm/dm_bb_a").to_ndarray()
            )
            tdm_bb_b.append(
                ctx.at_tensor(state_tree + "/optdm/dm_bb_b").to_ndarray()
            )

            eigenvalues.append(ctx.at_scalar(state_tree + "/energy"))
            eigenvectors_singles.append(
                ctx.at_tensor(state_tree + "/u1").to_ndarray()
            )
            eigenvectors_doubles.append(
                ctx.at_tensor(state_tree + "/u2").to_ndarray()
            )

        # from the others only the energy
        for i in range(n_states_full, n_states):
            state_tree = tree + "/es" + str(i)
            eigenvalues.append(ctx.at_scalar(state_tree + "/energy"))

        # Transform to numpy array
        pfx = out_tree + "/" + kind
        ret[pfx + "/state_diffdm_bb_a"] = np.asarray(dm_bb_a)
        ret[pfx + "/state_diffdm_bb_b"] = np.asarray(dm_bb_b)
        ret[pfx + "/ground_to_excited_tdm_bb_a"] = np.asarray(tdm_bb_a)
        ret[pfx + "/ground_to_excited_tdm_bb_b"] = np.asarray(tdm_bb_b)
        ret[pfx + "/eigenvalues"] = np.array(eigenvalues)
        ret[pfx + "/eigenvectors_singles"] = np.asarray(eigenvectors_singles)
        ret[pfx + "/eigenvectors_doubles"] = np.asarray(eigenvectors_doubles)
    # for kind

    ret["available_kinds"] = available_kinds
    return ret


def main():
    #
    # H2O restricted
    #
    kwargs_cvs = {"n_singlets": 3, "n_triplets": 3}
    kwargs_general = {"n_singlets": 10, "n_triplets": 10}
    dump_all_methods("h2o_sto3g", kwargs_cvs, kwargs_general)

    #
    # CN unrestricted
    #
    kwargs_cvs = {"n_states": 6}
    kwargs_general = {"n_states": 8}
    dump_all_methods("cn_sto3g", kwargs_cvs, kwargs_general)


if __name__ == "__main__":
    main()