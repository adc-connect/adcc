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
from dataclasses import dataclass
import itertools
import adcc
import numpy as np
import yaml
from tqdm import tqdm

from pyscf import gto, lib

adcc.set_n_threads(8)
lib.num_threads(8)


prefactors_5p = np.array([1.0, -8.0, 8.0, -1.0]) / 12.0
multipliers_5p = [-2, -1, 1, 2]
# prefactors_9p = [1. / 280., -4. / 105., 1. / 5., -4. / 5.,
#                  4. / 5., -1. / 5., 4. / 105., -1. / 280.]
# multipliers_9p = [-4., -3., - 2., -1., 1., 2., 3., 4.]
coords_label = ["x", "y", "z"]


@dataclass
class Molecule:
    name: str
    charge: int = 0
    multiplicity: int = 1
    core_orbitals: int = 1

    @property
    def xyz(self):
        return xyz[self.name]


molecules = [
    Molecule("h2o", 0, 1),
]

# all coordinates in Bohr (perturbed)
xyz = {
    "h2o": """
    O 0 0 0
    H 0 0 1.895239827225189
    H 1.693194615993441 0 -0.599043184453037
    """,
}


def _molstring(elems, coords):
    s = ""
    for kk, (i, c) in enumerate(zip(elems, coords)):
        s += f"{i} {c[0]} {c[1]} {c[2]}"
        if kk != len(elems) - 1:
            s += "\n"
    return s


def adc_energy(scfres, method, **kwargs):
    state = adcc.run_adc(method=method, data_or_matrix=scfres,
                         output=None,
                         **kwargs)
    return state.total_energy


def mp_energy(scfres, method, **kwargs):
    level = int(method[-1])
    refstate = adcc.ReferenceState(scfres)
    return adcc.LazyMp(refstate).energy(level)


def fdiff_gradient(molecule, method, basis, step=1e-4, **kwargs):
    m = gto.M(atom=molecule.xyz, unit='Bohr', basis=basis,
              spin=molecule.multiplicity - 1, charge=molecule.charge)
    coords = m.atom_coords().copy()
    elements = m.elements.copy()

    conv_tol = kwargs.get("conv_tol", 1e-10) / 100

    # run unperturbed system
    scfres = adcc.backends.run_hf(
        'pyscf', molecule.xyz, basis, conv_tol=conv_tol, conv_tol_grad=conv_tol,
        charge=molecule.charge, multiplicity=molecule.multiplicity
    )
    if "adc" in method:
        en = adc_energy(scfres, method, **kwargs)
        ngrad = len(en)
    else:
        en = mp_energy(scfres, method, **kwargs)
        ngrad = 1

    natoms = len(elements)
    grad = np.zeros((ngrad, natoms, 3))
    at_c = list(itertools.product(range(natoms), range(3)))
    for i, c in tqdm(at_c):
        for f, p in zip(multipliers_5p, prefactors_5p):
            coords_p = coords.copy()
            coords_p[i, c] += f * step
            geom_p = _molstring(elements, coords_p)
            scfres = adcc.backends.run_hf(
                'pyscf', geom_p, basis, conv_tol=conv_tol,
                conv_tol_grad=conv_tol * 10,
                charge=molecule.charge, multiplicity=molecule.multiplicity,
                max_iter=500,
            )
            if "adc" in method:
                en_pert = adc_energy(scfres, method, **kwargs)
            else:
                en_pert = mp_energy(scfres, method, **kwargs)
            grad[:, i, c] += p * en_pert / step
    return en, grad


def main():
    basissets = [
        "sto3g",
        "ccpvdz",
    ]
    methods = [
        "mp2",
        "adc0",
        "adc1",
        "adc2",
        "cvs-adc0",
        "cvs-adc1",
        "cvs-adc2",
        # "cvs-adc2x",  # TODO: broken
    ]
    molnames = [
        "h2o",
    ]
    mols = [m for m in molecules if m.name in molnames]
    ret = {
        "basissets": basissets,
        "methods": methods,
        "molecules": molnames,
    }
    config_excited = {"n_singlets": 3}
    for molecule in mols:
        molname = molecule.name
        ret[molname] = {}
        for basis in basissets:
            ret[molname][basis] = {}
            for method in methods:
                kwargs = {
                    "conv_tol": 1e-10,
                }
                if "adc" in method:
                    kwargs.update(config_excited)
                basename = f"{molname}_{basis}_{method}"
                print(f"Evaluating finite difference gradient for {basename}.")

                core_orbitals = None
                if "cvs" in method:
                    core_orbitals = molecule.core_orbitals
                kwargs["core_orbitals"] = core_orbitals
                en, grad = fdiff_gradient(molecule, method, basis, **kwargs)
                if isinstance(en, np.ndarray):
                    en = en.tolist()
                cont = {
                    "energy": en,
                    "gradient": np.squeeze(grad).tolist(),
                    "config": kwargs,
                }
                ret[molname][basis][method] = cont
        ret[molname]["xyz"] = molecule.xyz
    with open("grad_dump.yml", "w") as yamlout:
        yaml.safe_dump(ret, yamlout)


if __name__ == "__main__":
    main()
