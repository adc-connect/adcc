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

from libadcc import HfData
from molsturm.State import State


def convert_scf_to_dict(scfres):
    if not isinstance(scfres, State):
        raise TypeError("Unsupported type for backends.molsturm.import_scf.")

    n_alpha = scfres["n_alpha"]
    n_beta = scfres["n_beta"]
    data = {}

    # Keys to include verbatim from the hf res dictionary to the adcc input
    verbatim_keys = [
        "n_alpha", "n_beta", "n_orbs_alpha", "n_orbs_beta",
        "n_bas", "restricted",
        "orben_f", "eri_ffff", "fock_ff",
        "energy_nuclear_repulsion", "energy_nuclear_attraction",
        "energy_coulomb", "energy_exchange", "energy_kinetic",
    ]
    for k in verbatim_keys:
        data[k] = scfres[k]

    if scfres["restricted"]:
        data["spin_multiplicity"] = 2 * (n_alpha - n_beta) + 1
    else:
        data["spin_multiplicity"] = 0

    data["energy_scf"] = scfres["energy_ground_state"]
    data["threshold"] = 10 * scfres["final_error_norm"]
    data["orbcoeff_fb"] = scfres["orbcoeff_bf"].transpose().copy()
    return data


def import_scf(scfres):
    data = convert_scf_to_dict(scfres)
    ret = HfData.from_dict(data)
    ret.backend = "molsturm"

    # TODO temporary hack to make sure the data dict lives longer
    #      than the Hfdata object. Don't rely on this object for
    #      your code.
    ret._original_dict = data
    return ret
