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
import os
import numpy as np
import adcc
import adcc.solver.adcman as adcman

from adcc import hdf5io
from adcc.testdata.cache import cache

from libadcc import CtxMap, Symmetry


def dump_all_methods(key, kwargs_cvs, kwargs_general, kwargs_overwrite={}):
    data = cache.hfdata[key]

    for method in ["adc0", "adc1", "adc2", "adc2x", "adc3"]:
        tmethod = method
        if method == "adc2":
            tmethod = "adc2s"

        if method in kwargs_overwrite:
            overwrite = kwargs_overwrite[method]
            kw_general = overwrite.get("general", kwargs_general)
            kw_cvs = overwrite.get("cvs", kwargs_cvs)
        else:
            kw_general = kwargs_general
            kw_cvs = kwargs_cvs

        if method not in ["adc3"] and kwargs_cvs is not None:
            # do CVS
            dumpfile = "{}_reference_cvs_{}.hdf5".format(key, method)
            if not os.path.isfile(dumpfile):
                method_tree = "adc_pp/cvs_" + tmethod
                res = run(data, method="cvs-" + method, method_tree=method_tree,
                          **kw_cvs, core_valence_separation=True)
                dictionary = build_dict(kw_cvs, res, mp_tree="mp_cvs",
                                        method_tree=method_tree,
                                        out_tree="cvs-" + method)
                hdf5io.save(dumpfile, dictionary)

        # General
        dumpfile = "{}_reference_{}.hdf5".format(key, method)
        if not os.path.isfile(dumpfile):
            method_tree = "adc_pp/" + tmethod
            res = run(data, method=method, method_tree=method_tree,
                      **kw_general)
            dictionary = build_dict(kw_general, res, mp_tree="mp",
                                    method_tree=method_tree,
                                    out_tree=method)
            hdf5io.save(dumpfile, dictionary)


def build_params(refstate, method_tree):
    params = CtxMap()

    kind_trees = []
    if refstate.restricted:
        kind_trees.append(method_tree + "/rhf/singlets/0")
        kind_trees.append(method_tree + "/rhf/triplets/0")
    else:
        kind_trees.append(method_tree + "/uhf/0")

    # Properties
    prop_trees = ["hf/prop", "mp2/prop"]
    for ktree in kind_trees:
        prop_trees.append(ktree + "/prop")
        prop_trees.append(ktree + "/tprop")
    for ptree in prop_trees:
        prop_template = {".": "1", "dipole": "1", "rsq": "0"}
        for key, value in prop_template.items():
            params.update_string(ptree + "/" + key, value)
    return params


def as_tensor_bb(mospaces, array, symmetric=True):
    assert array.ndim == 2
    assert array.shape[0] == array.shape[1]
    sym = Symmetry(mospaces, "bb",
                   {"b": (array.shape[0], 0)})
    if symmetric:
        sym.permutations = ["ij", "ji"]
    tensor = adcc.Tensor(sym)
    tensor.set_from_ndarray(array)
    return tensor


def build_ctx(refstate):
    ctx = CtxMap()

    # Nuclear dipole moment
    nucmm = [refstate.nuclear_total_charge] + refstate.nuclear_dipole
    nucmm += 6 * [0.0]
    ctx.update_scalar_list("ao/nucmm", nucmm)

    # AO integrals for properties
    integrals_ao = refstate.operators.provider_ao
    if hasattr(integrals_ao, "electric_dipole"):
        for i, comp in enumerate(["x", "y", "z"]):
            dip_bb = as_tensor_bb(refstate.mospaces,
                                  integrals_ao.electric_dipole[i],
                                  symmetric=True)
            ctx.update_tensor("ao/d{}_bb".format(comp), dip_bb)

    return ctx


def run(data, method, method_tree, n_singlets=None, n_triplets=None,
        n_states=None, n_spin_flip=None,
        core_valence_separation=False, n_guess_singles=0,
        max_subspace=0):
    n_core_orbitals = None
    if core_valence_separation:
        n_core_orbitals = data["n_core_orbitals"]
    refstate = adcc.ReferenceState(data, core_orbitals=n_core_orbitals)

    # Gather extra parameters
    extra_params = build_params(refstate, method_tree)
    extra_ctx = build_ctx(refstate)

    # Setup the matrix and run adcman solver
    matrix = adcc.AdcMatrix(method, adcc.LazyMp(refstate))
    states = adcman.jacobi_davidson(
        matrix, print_level=100, n_singlets=n_singlets, n_triplets=n_triplets,
        n_spin_flip=n_spin_flip, n_states=n_states, conv_tol=1e-9,
        n_guess_singles=n_guess_singles, max_iter=100,
        max_subspace=max_subspace, extra_ctx=extra_ctx,
        extra_params=extra_params
    )

    # return the full adcman context
    return states[0].ctx


def build_dict(kwrun, ctx, mp_tree, method_tree, out_tree, n_states_full=2):
    """
    kwrun     Kwargs passed on the adcman run
    ctx       Context produced from the adcman run
    """
    ret = {}

    #
    # MP
    #
    mp = mp_tree
    if len(mp) > 0 and mp[-1] != "/":
        mp = mp + "/"

    if ctx.exists("/mp2/energy"):
        ret[mp + "mp2/energy"] = ctx["/mp2/energy"]
    if ctx.exists("/mp3/energy"):
        ret[mp + "mp3/energy"] = ctx["/mp3/energy"]
    if ctx.exists("/mp2/prop/dipole"):
        ret[mp + "mp2/dipole"] = np.array(ctx["/mp2/prop/dipole"])

    if ctx.exists("/mp1/t_o1o1v1v1"):  # For generic
        ret[mp + "mp1/t_o1o1v1v1"] = ctx["/mp1/t_o1o1v1v1"].to_ndarray()
    if ctx.exists("/mp1/t_o2o2v1v1"):  # For CVS
        ret[mp + "mp1/t_o2o2v1v1"] = ctx["/mp1/t_o2o2v1v1"].to_ndarray()
        ret[mp + "mp1/t_o1o2v1v1"] = ctx["/mp1/t_o1o2v1v1"].to_ndarray()

    if ctx.exists("/mp1/df_o1v1"):  # For generic
        ret[mp + "mp1/df_o1v1"] = ctx["/mp1/df_o1v1"].to_ndarray()
    if ctx.exists("/mp1/df_o2v1"):  # For CVS
        ret[mp + "mp1/df_o2v1"] = ctx["/mp1/df_o2v1"].to_ndarray()

    if ctx.exists("/mp2/td_o1o1v1v1"):  # For generic
        ret[mp + "mp2/td_o1o1v1v1"] = ctx["/mp2/td_o1o1v1v1"].to_ndarray()

    mp_dm = "/mp2/opdm/"
    if ctx.exists(mp_dm + "dm_o1o1"):  # For generic
        ret[mp + "mp2/dm_o1o1"] = ctx[mp_dm + "dm_o1o1"].to_ndarray()
        ret[mp + "mp2/dm_o1v1"] = ctx[mp_dm + "dm_o1v1"].to_ndarray()
        ret[mp + "mp2/dm_v1v1"] = ctx[mp_dm + "dm_v1v1"].to_ndarray()
        ret[mp + "mp2/dm_bb_a"] = ctx[mp_dm + "dm_bb_a"].to_ndarray()
        ret[mp + "mp2/dm_bb_b"] = ctx[mp_dm + "dm_bb_b"].to_ndarray()

    if ctx.exists(mp_dm + "dm_o2o2"):  # For CVS
        ret[mp + "mp2/dm_o2o1"] = ctx[mp_dm + "dm_o2o1"].to_ndarray()
        ret[mp + "mp2/dm_o2o2"] = ctx[mp_dm + "dm_o2o2"].to_ndarray()
        ret[mp + "mp2/dm_o2v1"] = ctx[mp_dm + "dm_o2v1"].to_ndarray()

    #
    # ADC
    #
    kind_trees = {
        "singlet": method_tree + "/rhf/singlets/0",
        "triplet": method_tree + "/rhf/triplets/0",
        "state": method_tree + "/uhf/0",
        "spin_flip": method_tree + "/uhf/0",
    }
    if "n_spin_flip" not in kwrun:
        del kind_trees["spin_flip"]
    if "n_states" not in kwrun:
        del kind_trees["state"]

    available_kinds = []
    for kind, tree in kind_trees.items():
        dm_bb_a = []
        dm_bb_b = []
        tdm_bb_a = []
        tdm_bb_b = []
        state_dipoles = []
        transition_dipoles = []
        eigenvalues = []
        eigenvectors_singles = []
        eigenvectors_doubles = []
        n_states = ctx.at(tree + "/nstates", 0)
        if n_states == 0:
            continue
        available_kinds.append(kind)

        # From 2 states we save everything
        n_states_full = max(n_states_full, n_states)

        for i in range(n_states_full):
            state_tree = tree + "/es" + str(i)

            dm_bb_a.append(ctx[state_tree + "/opdm/dm_bb_a"].to_ndarray())
            dm_bb_b.append(ctx[state_tree + "/opdm/dm_bb_b"].to_ndarray())
            tdm_bb_a.append(ctx[state_tree + "/optdm/dm_bb_a"].to_ndarray())
            tdm_bb_b.append(ctx[state_tree + "/optdm/dm_bb_b"].to_ndarray())

            state_dipoles.append(ctx[state_tree + "/prop/dipole"])
            transition_dipoles.append(ctx[state_tree + "/tprop/dipole"])

            eigenvalues.append(ctx.at(state_tree + "/energy"))
            eigenvectors_singles.append(
                ctx.at(state_tree + "/u1").to_ndarray()
            )
            if ctx.exists(state_tree + "/u2"):
                eigenvectors_doubles.append(
                    ctx.at(state_tree + "/u2").to_ndarray()
                )
            else:
                eigenvectors_doubles.clear()

        # from the others only the energy and dipoles
        for i in range(n_states_full, n_states):
            state_tree = tree + "/es" + str(i)
            state_dipoles.append(ctx[state_tree + "/prop/dipole"])
            transition_dipoles.append(ctx[state_tree + "/tprop/dipole"])
            eigenvalues.append(ctx.at(state_tree + "/energy"))

        # Transform to numpy array
        pfx = out_tree + "/" + kind
        ret[pfx + "/state_diffdm_bb_a"] = np.asarray(dm_bb_a)
        ret[pfx + "/state_diffdm_bb_b"] = np.asarray(dm_bb_b)
        ret[pfx + "/ground_to_excited_tdm_bb_a"] = np.asarray(tdm_bb_a)
        ret[pfx + "/ground_to_excited_tdm_bb_b"] = np.asarray(tdm_bb_b)
        ret[pfx + "/state_dipole_moments"] = np.asarray(state_dipoles)
        ret[pfx + "/transition_dipole_moments"] = np.asarray(transition_dipoles)
        ret[pfx + "/eigenvalues"] = np.array(eigenvalues)
        ret[pfx + "/eigenvectors_singles"] = np.asarray(eigenvectors_singles)

        if eigenvectors_doubles:  # For ADC(0) and ADC(1) there are no doubles
            ret[pfx + "/eigenvectors_doubles"] = \
                np.asarray(eigenvectors_doubles)
    # for kind

    ret["available_kinds"] = available_kinds
    return ret


def main():
    #
    # H2O restricted
    #
    kwargs_cvs = {"n_singlets": 3, "n_triplets": 3}
    kwargs_general = {"n_singlets": 10, "n_triplets": 10}
    kwargs_overwrite = {
        "adc0": {"cvs": {"n_singlets": 2, "n_triplets": 2}, },
        "adc1": {"cvs": {"n_singlets": 2, "n_triplets": 2}, },
        "adc2": {"general": {"n_singlets": 9, "n_triplets": 10, }, }
    }
    dump_all_methods("h2o_sto3g", kwargs_cvs, kwargs_general, kwargs_overwrite)

    #
    # H2O restricted (TZVP)
    #
    kwargs_cvs = {"n_singlets": 3, "n_triplets": 3, "n_guess_singles": 6,
                  "max_subspace": 24}
    kwargs_general = {"n_singlets": 3, "n_triplets": 3, "n_guess_singles": 6,
                      "max_subspace": 24}
    dump_all_methods("h2o_def2tzvp", kwargs_cvs, kwargs_general)

    #
    # CN unrestricted
    #
    kwargs_cvs = {"n_states": 6}
    kwargs_general = {"n_states": 8}
    dump_all_methods("cn_sto3g", kwargs_cvs, kwargs_general)

    #
    # CN unrestricted (cc-pVDZ)
    #
    kwargs_cvs = {"n_states": 5, "n_guess_singles": 7}
    kwargs_general = {"n_states": 5, "n_guess_singles": 7}
    kwargs_overwrite = {
        "adc1": {"general": {"n_states": 4, "n_guess_singles": 8}, },
    }
    dump_all_methods("cn_ccpvdz", kwargs_cvs, kwargs_general, kwargs_overwrite)

    #
    # HF triplet unrestricted (for spin-flip)
    #
    kwargs_cvs = None
    kwargs_general = {"n_spin_flip": 9}
    dump_all_methods("hf3_631g", kwargs_cvs, kwargs_general)


if __name__ == "__main__":
    main()
