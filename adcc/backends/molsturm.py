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
import numpy as np

from molsturm.State import State
from adcc.DataHfProvider import DataHfProvider


def convert_scf_to_dict(scfres):
    if not isinstance(scfres, State):
        raise TypeError("Unsupported type for backends.molsturm.import_scf.")

    n_alpha = scfres["n_alpha"]
    n_beta = scfres["n_beta"]
    data = {}
    if scfres["n_orbs_alpha"] != scfres["n_orbs_beta"]:
        raise ValueError("n_orbs_alpha != n_orbs_beta not supported.")

    # Keys to include verbatim from the hf res dictionary to the adcc input
    verbatim_keys = [
        "n_orbs_alpha", "restricted", "orben_f", "eri_ffff", "fock_ff",
    ]
    for k in verbatim_keys:
        data[k] = scfres[k]

    if scfres["restricted"]:
        data["spin_multiplicity"] = 2 * (n_alpha - n_beta) + 1
    else:
        data["spin_multiplicity"] = 0

    n_oa = data["n_orbs_alpha"]
    data["occupation_f"] = np.zeros(2 * n_oa)
    data["occupation_f"][:n_alpha] = 1.
    data["occupation_f"][n_oa:n_oa + n_beta] = 1.

    data["energy_scf"] = scfres["energy_ground_state"]
    data["conv_tol"] = 10 * scfres["final_error_norm"]
    data["orbcoeff_fb"] = scfres["orbcoeff_bf"].transpose().copy()

    # Compute electric and nuclear multipole moments
    data["multipoles"] = {"elec_0": -int(n_alpha + n_beta), }
    if "input_parameters" in scfres:
        params = scfres["input_parameters"]
        coords = np.asarray(params["system"]["coords"])
        charges = np.asarray(params["system"]["atom_numbers"])
        data["multipoles"]["nuclear_0"] = int(np.sum(charges)),
        data["multipoles"]["nuclear_1"] = np.einsum('i,ix->x', charges, coords)
    else:
        import warnings

        # We have no information about this, so we can just provide dummies
        data["multipoles"]["nuclear_0"] = -1
        data["multipoles"]["nuclear_1"] = np.zeros(3)
        warnings.warn("The passed molsturm scfres has no information about "
                      "nuclear multipoles, such that dummy data are used. "
                      "Results involving nuclear multipoles could be wrong.")

    data["backend"] = "molsturm"
    return data


def import_scf(scfres):
    return DataHfProvider(convert_scf_to_dict(scfres))


basis_remap = {
    "sto3g": "sto-3g",
    "def2tzvp": "def2-tzvp",
    "ccpvdz": "cc-pvdz",
}


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-12,
           conv_tol_grad=1e-8, max_iter=150):

    import molsturm

    # Quick-and-dirty xyz parser:
    geom = xyz.split()
    n_atom = len(geom) // 4
    assert n_atom * 4 == len(geom)
    atoms = [geom[i * 4] for i in range(n_atom)]
    coords = [[float(geom[i * 4 + 1]),
               float(geom[i * 4 + 2]),
               float(geom[i * 4 + 3])] for i in range(n_atom)]

    mol = molsturm.System(atoms, coords)
    mol.charge = charge
    mol.multiplicity = multiplicity

    return molsturm.hartree_fock(mol, basis_type="gaussian",
                                 basis_set_name=basis_remap.get(basis, basis),
                                 conv_tol=conv_tol_grad, max_iter=max_iter)
