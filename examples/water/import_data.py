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
import numpy as np

from data import coeff_data, eri_data, orben_data


def import_data():
    n_bas = 7
    data = {
        "n_alpha": 5,
        "n_beta": 5,
        "n_orbs_alpha": 7,
        "n_orbs_beta": 7,
        "n_bas": n_bas,
        "energy_scf": -7.4959319286025718e+01,
        "energy_nuclear_repulsion": 9.251479269240862,
        "restricted": True,
        "threshold": 1e-12,
        "spin_multiplicity": 1,
    }
    n_orbs = data["n_orbs_alpha"] + data["n_orbs_beta"]

    data["occupation_f"] = np.array(5 * [1] + [0, 0] + 5 * [1] + [0, 0.])
    data["orbcoeff_fb"] = np.array(coeff_data).reshape((n_orbs, n_bas))
    data["orben_f"] = np.array(orben_data).reshape((n_orbs))
    data["eri_ffff"] = np.array(eri_data).reshape((n_orbs, n_orbs,
                                                   n_orbs, n_orbs))

    data["fock_ff"] = np.zeros((n_orbs, n_orbs))
    for i in range(n_orbs):
        data["fock_ff"][i, i] = orben_data[i]
    return data
