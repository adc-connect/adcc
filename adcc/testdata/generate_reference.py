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

from libadcc import CtxMap, Symmetry

from adcc import hdf5io
from adcc.testdata.cache import cache


def dump_all(case, kwargs, kwargs_overwrite={}, spec="gen"):
    assert spec in ["gen", "cvs"]
    for method in ["adc0", "adc1", "adc2", "adc2x", "adc3"]:
        kw = kwargs_overwrite.get(method, kwargs)
        if spec == "gen":
            dump_general(case, method, kw, spec=spec)
        else:
            dump_cvs(case, method, kw, spec=spec)


def dump_general(case, method, kwargs, spec="gen"):
    data = cache.hfdata[case]
    kwargs = dict(kwargs)
    kwargs.update(data["reference_cases"][spec])

    method_tree = "adc_pp/" + method
    if method == "adc2":
        method_tree = "adc_pp/adc2s"

    prefix = spec.replace("-", "_") + "_"
    if spec == "gen":
        prefix = ""
    out_tree = prefix.replace("_", "-") + method
    dumpfile = "{}_reference_{}{}.hdf5".format(case, prefix, method)

    if spec == "gen":
        mp_tree = "mp"
    else:
        mp_tree = prefix + "mp"

    if not os.path.isfile(dumpfile):
        res = run(data, method=method, method_tree=method_tree, **kwargs)
        dictionary = build_dict(kwargs, res, mp_tree=mp_tree, out_tree=out_tree,
                                method_tree=method_tree)
        hdf5io.save(dumpfile, dictionary)


def dump_cvs(case, method, kwargs, spec="cvs"):
    data = cache.hfdata[case]
    kwargs = dict(kwargs)
    kwargs.update(data["reference_cases"][spec])

    method_tree = "adc_pp/cvs_" + method
    if method == "adc2":
        method_tree = "adc_pp/cvs_adc2s"

    assert spec != "gen"
    prefix = spec.replace("-", "_") + "_"
    out_tree = prefix.replace("_", "-") + method
    dumpfile = "{}_reference_{}{}.hdf5".format(case, prefix, method)

    if spec == "cvs":
        mp_tree = "mp_cvs"
    else:
        mp_tree = prefix + "mp"

    if not os.path.isfile(dumpfile):
        res = run(data, method="cvs-" + method, **kwargs,
                  method_tree=method_tree)
        if method == "adc3":
            # For CVS-ADC(3) the MP3 energy in adcman is wrong
            del res["/mp3/energy"]
        dictionary = build_dict(kwargs, res, mp_tree=mp_tree,
                                method_tree=method_tree, out_tree=out_tree)
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
        n_states=None, n_spin_flip=None, n_guess_singles=0,
        max_subspace=0, core_orbitals=None, frozen_core=None,
        frozen_virtual=None):
    print("#")
    print("#-- {}    n_singlets={} n_triplets={} n_states={} n_sf={}".format(
        method, n_singlets, n_triplets, n_states, n_spin_flip))
    print("#--          core_orbitals={} frozen_core={} frozen_virtual={}"
          "".format(core_orbitals, frozen_core, frozen_virtual))
    print("#")
    refstate = adcc.ReferenceState(data, core_orbitals=core_orbitals,
                                   frozen_virtual=frozen_virtual,
                                   frozen_core=frozen_core)

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
        n_states = ctx.get(tree + "/nstates", 0)
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

            eigenvalues.append(ctx[state_tree + "/energy"])
            eigenvectors_singles.append(
                ctx[state_tree + "/u1"].to_ndarray()
            )
            if ctx.exists(state_tree + "/u2"):
                eigenvectors_doubles.append(
                    ctx[state_tree + "/u2"].to_ndarray()
                )
            else:
                eigenvectors_doubles.clear()

        # from the others only the energy and dipoles
        for i in range(n_states_full, n_states):
            state_tree = tree + "/es" + str(i)
            state_dipoles.append(ctx[state_tree + "/prop/dipole"])
            transition_dipoles.append(ctx[state_tree + "/tprop/dipole"])
            eigenvalues.append(ctx[state_tree + "/energy"])

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


def dump_h2o_sto3g():  # H2O restricted
    # All methods for general and CVS
    kwargs = {"n_singlets": 10, "n_triplets": 10}
    overwrite = {"adc2": {"n_singlets": 9, "n_triplets": 10}, }
    dump_all("h2o_sto3g", kwargs, overwrite, spec="gen")

    kwargs = {"n_singlets": 3, "n_triplets": 3}
    overwrite = {
        "adc0": {"n_singlets": 2, "n_triplets": 2},
        "adc1": {"n_singlets": 2, "n_triplets": 2},
    }
    dump_all("h2o_sto3g", kwargs, overwrite, spec="cvs")

    case = "h2o_sto3g"  # Just ADC(2) and ADC(2)-x
    kwargs = {"n_singlets": 3, "n_triplets": 3}
    dump_general(case, "adc2", kwargs, spec="fc")
    dump_general(case, "adc2", kwargs, spec="fc-fv")
    dump_general(case, "adc2x", kwargs, spec="fv")
    dump_cvs(case, "adc2x", kwargs, spec="fv-cvs")


def dump_h2o_def2tzvp():  # H2O restricted
    kwargs = {"n_singlets": 3, "n_triplets": 3, "n_guess_singles": 6,
              "max_subspace": 24}
    dump_all("h2o_def2tzvp", kwargs, spec="gen")
    dump_all("h2o_def2tzvp", kwargs, spec="cvs")


def dump_cn_sto3g():  # CN unrestricted
    dump_all("cn_sto3g", {"n_states": 8}, spec="gen")
    dump_all("cn_sto3g", {"n_states": 6}, spec="cvs")

    # Just ADC(2) and ADC(2)-x for the other methods
    case = "cn_sto3g"
    dump_general(case, "adc2", {"n_states": 4}, spec="fc")
    dump_general(case, "adc2", {"n_states": 4, "n_guess_singles": 8},
                 spec="fc-fv")
    dump_general(case, "adc2x", {"n_states": 4}, spec="fv")
    dump_cvs(case, "adc2x", {"n_states": 4}, spec="fv-cvs")


def dump_cn_ccpvdz():  # CN unrestricted
    kwargs = {"n_states": 5, "n_guess_singles": 7}
    overwrite = {"adc1": {"n_states": 4, "n_guess_singles": 8}, }
    dump_all("cn_ccpvdz", kwargs, overwrite, spec="gen")
    dump_all("cn_ccpvdz", kwargs, spec="cvs")


def dump_hf3_631g():  # HF triplet unrestricted (spin-flip)
    dump_all("hf3_631g", {"n_spin_flip": 9}, spec="gen")


def dump_h2s_sto3g():
    case = "h2s_sto3g"
    kwargs = {"n_singlets": 3, "n_triplets": 3}
    dump_cvs(case, "adc2", kwargs, spec="fc-cvs")
    dump_cvs(case, "adc2x", kwargs, spec="fc-fv-cvs")


def dump_h2s_6311g():
    case = "h2s_6311g"
    kwargs = {"n_singlets": 3, "n_triplets": 3}
    for spec in ["gen", "fc", "fv", "fc-fv"]:
        dump_general(case, "adc2", kwargs, spec=spec)

    kwargs = {"n_singlets": 3, "n_triplets": 3, "n_guess_singles": 6,
              "max_subspace": 60}
    for spec in ["cvs", "fv-cvs", "fc-cvs", "fc-fv-cvs"]:
        dump_cvs(case, "adc2x", kwargs, spec=spec)


def main():
    dump_h2o_sto3g()
    dump_h2o_def2tzvp()
    dump_cn_sto3g()
    dump_cn_ccpvdz()
    dump_hf3_631g()
    dump_h2s_sto3g()
    dump_h2s_6311g()


if __name__ == "__main__":
    main()
