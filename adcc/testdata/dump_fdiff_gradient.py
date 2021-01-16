#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2021 by the adcc authors
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
import itertools
import adcc
import numpy as np
import yaml
from tqdm import tqdm

from pyscf import gto

from static_data import xyz


prefactors_5p = np.array([1.0, -8.0, 8.0, -1.0]) / 12.0
multipliers_5p = [-2, -1, 1, 2]
coords_label = ["x", "y", "z"]


def _molstring(elems, coords):
    s = ""
    for kk, (i, c) in enumerate(zip(elems, coords)):
        s += f"{i} {c[0]} {c[1]} {c[2]}"
        if kk != len(elems) - 1:
            s += "\n"
    return s


def adc_energy(scfres, method, **kwargs):
    state = adcc.run_adc(method=method, data_or_matrix=scfres,
                         output=None, **kwargs)
    return state.total_energy


def mp_energy(scfres, method, **kwargs):
    level = {
        "mp2": 2,
        "mp3": 3,
    }
    refstate = adcc.ReferenceState(scfres)
    return adcc.LazyMp(refstate).energy(level[method])


def fdiff_gradient(molstring, method, basis, step=1e-4, **kwargs):
    m = gto.M(atom=molstring, unit='Bohr', basis=basis)
    coords = m.atom_coords().copy()
    elements = m.elements.copy()

    n_grads = kwargs.get("n_singlets", 1)
    conv_tol = kwargs.get("conv_tol", 1e-10) / 10

    # run unperturbed system
    scfres = adcc.backends.run_hf(
        'pyscf', molstring, basis, conv_tol=conv_tol, conv_tol_grad=conv_tol
    )
    if "adc" in method:
        en = adc_energy(scfres, method, **kwargs)
    else:
        en = mp_energy(scfres, method, **kwargs)

    natoms = len(elements)
    grad = np.zeros((n_grads, natoms, 3))
    at_c = list(itertools.product(range(natoms), range(3)))
    for i, c in tqdm(at_c):
        for f, p in zip(multipliers_5p, prefactors_5p):
            coords_p = coords.copy()
            coords_p[i, c] += f * step
            geom_p = _molstring(elements, coords_p)
            scfres = adcc.backends.run_hf(
                'pyscf', geom_p, basis, conv_tol=conv_tol, conv_tol_grad=conv_tol
            )
            if "adc" in method:
                en_pert = adc_energy(scfres, method, **kwargs)
            else:
                en_pert = mp_energy(scfres, method, **kwargs)
            grad[:, i, c] += p * en_pert / step
    return en, grad


def main():
    config_excited = {
        "n_singlets": 5,
    }
    basissets = [
        "sto3g",
        "ccpvdz",
    ]
    methods = [
        "mp2",
        "adc1",
        "adc2",
    ]
    molecules = ["h2o", "hf", "formaldehyde"]
    ret = {}
    for molecule in molecules:
        ret[molecule] = {}
        for basis in basissets:
            ret[molecule][basis] = {}
            for method in methods:
                kwargs = {
                    "conv_tol": 1e-8,
                }
                if "adc" in method:
                    kwargs.update(config_excited)
                basename = f"{molecule}_{basis}_{method}"
                print(f"Evaluating finite difference gradient for {basename}.")
                en, grad = fdiff_gradient(xyz[molecule], method, basis, **kwargs)
                if isinstance(en, np.ndarray):
                    en = en.tolist()
                cont = {
                    "energy": en,
                    "gradient": np.squeeze(grad).tolist(),
                }
                ret[molecule][basis][method] = cont
    with open("grad_dump.yml", "w") as yamlout:
        yaml.safe_dump(ret, yamlout)


if __name__ == "__main__":
    main()
