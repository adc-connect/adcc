#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import numpy as np

from data import coeff_data, dip_data, eri_data, orben_data


def import_data():
    n_orbs_alpha = n_bas = 7
    n_orbs = 2 * n_orbs_alpha
    data = {
        "n_orbs_alpha": n_orbs_alpha,
        "n_bas": n_bas,
        "energy_scf": -7.4959319286025718e+01,
        "restricted": True,
        "threshold": 1e-12,
        "spin_multiplicity": 1,
        "multipoles": {
            "elec_0": -10,
            "nuclear_0": 10,
            "nuclear_1": np.array([1.693194615993441, 0.,
                                   1.196196642772152]),
            "elec_1": np.array(dip_data).reshape(3, n_bas, n_bas)
        },
    }
    data["occupation_f"] = np.array(5 * [1] + [0, 0] + 5 * [1] + [0, 0.])
    data["orbcoeff_fb"] = np.array(coeff_data).reshape((n_orbs, n_bas))
    data["orben_f"] = np.array(orben_data).reshape((n_orbs))
    data["eri_ffff"] = np.array(eri_data).reshape((n_orbs, n_orbs,
                                                   n_orbs, n_orbs))

    data["fock_ff"] = np.zeros((n_orbs, n_orbs))
    for i in range(n_orbs):
        data["fock_ff"][i, i] = orben_data[i]
    return data
