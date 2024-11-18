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

from pyscf import ao2mo, scf

import h5py

def get_qchem_formatted_basis(mol):
    L = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}
    cgtos_by_atom = {}
    for cgto_number in range(mol.nbas):
        atom_number = mol.bas_atom(cgto_number)
        atom_name = mol.elements[atom_number]
        cgto_exps = mol.bas_exp(cgto_number)
        cgto_coeffs = mol.bas_ctr_coeff(cgto_number)
        cgto_angular_momentum = mol.bas_angular(cgto_number)
        cgto_angular_momentum_name = L[cgto_angular_momentum]
        if not atom_number in cgtos_by_atom:
            cgtos_by_atom[atom_number] = []
        cur_basis_function = (cgto_angular_momentum_name,
                              list(zip(cgto_exps, cgto_coeffs)))
        cgtos_by_atom[atom_number].append(cur_basis_function)
    qchem_formatted_basis = []
    for atom_number in sorted(cgtos_by_atom):
        atom_name = mol.elements[atom_number]
        qchem_formatted_basis.append("{: <2s}    {: >3d}".format(
                                     atom_name, atom_number+1))
        for cgto in cgtos_by_atom[atom_number]:
            angular_momentum_name, primitive_gtos = cgto
            n_primitive_gtos = len(primitive_gtos)
            n_cgtos = len(primitive_gtos[0][1])
            for i in range(n_cgtos):
                qchem_formatted_basis.append("{:s}   {: >2d}   1.00".format(
                                             angular_momentum_name,
                                             n_primitive_gtos))
                for exp, coeffs in primitive_gtos:
                    coeff = coeffs[i]
                    basis_line = "{: >20.8E}{: >20.8E}".format(exp, coeff)
                    qchem_formatted_basis.append(basis_line.replace("E", "D"))
        qchem_formatted_basis.append("****")
    return qchem_formatted_basis

def dump_pyscf(scfres, out):
    """
    Convert pyscf SCF result to HDF5 file in adcc format
    """
    if not isinstance(scfres, scf.hf.SCF):
        raise TypeError("Unsupported type for dump_pyscf.")

    if not scfres.converged:
        raise ValueError(
            "Cannot dump a pyscf calculation, "
            "which is not yet converged. Did you forget to run "
            "the kernel() or the scf() function of the pyscf scf "
            "object?"
        )

    if isinstance(out, h5py.File):
        data = out
    elif isinstance(out, str):
        data = h5py.File(out, "w")
    else:
        raise TypeError("Unknown type for out, only HDF5 file and str supported.")

    # Try to determine whether we are restricted
    if isinstance(scfres.mo_occ, list):
        restricted = len(scfres.mo_occ) < 2
    elif isinstance(scfres.mo_occ, np.ndarray):
        restricted = scfres.mo_occ.ndim < 2
    else:
        raise ValueError(
            "Unusual pyscf SCF class encountered. Could not "
            "determine restricted / unrestricted."
        )

    mo_occ = scfres.mo_occ
    mo_energy = scfres.mo_energy
    mo_coeff = scfres.mo_coeff
    fock_bb = scfres.get_fock()

    # pyscf only keeps occupation and mo energies once if restriced,
    # so we unfold it here in order to unify the treatment in the rest
    # of the code
    if restricted:
        mo_occ = np.asarray((mo_occ / 2, mo_occ / 2))
        mo_energy = (mo_energy, mo_energy)
        mo_coeff = (mo_coeff, mo_coeff)
        fock_bb = (fock_bb, fock_bb)

    # Transform fock matrix to MOs
    fock = tuple(mo_coeff[i].transpose().conj() @ fock_bb[i] @ mo_coeff[i]
                 for i in range(2))

    # Determine number of orbitals
    n_orbs_alpha = mo_coeff[0].shape[1]
    n_orbs_beta = mo_coeff[1].shape[1]
    n_orbs = n_orbs_alpha + n_orbs_beta
    if n_orbs_alpha != n_orbs_beta:
        raise ValueError(
            "adcc cannot deal with different number of alpha and "
            "beta orbitals like in a restricted "
            "open-shell reference at the moment."
        )

    # Determine number of electrons
    n_alpha = np.sum(mo_occ[0] > 0)
    n_beta = np.sum(mo_occ[1] > 0)
    if n_alpha != np.sum(mo_occ[0]) or n_beta != np.sum(mo_occ[1]):
        raise ValueError("Fractional occupation numbers are not supported "
                         "in adcc.")

    # conv_tol is energy convergence, conv_tol_grad is gradient convergence
    if scfres.conv_tol_grad is None:
        conv_tol_grad = np.sqrt(scfres.conv_tol)
    else:
        conv_tol_grad = scfres.conv_tol_grad
    threshold = max(10 * scfres.conv_tol, conv_tol_grad)

    basis = get_qchem_formatted_basis(scfres.mol)

    #
    # Put basic data into HDF5 file
    #
    data.create_dataset("qchem_formatted_basis", shape=len(basis), data=basis,
                        dtype=h5py.string_dtype())
    data.create_dataset("n_orbs_alpha", shape=(), data=int(n_orbs_alpha))
    data.create_dataset("energy_scf", shape=(), data=float(scfres.e_tot))
    data.create_dataset("restricted", shape=(), data=restricted)
    data.create_dataset("conv_tol", shape=(), data=float(threshold))
    data.create_dataset("cartesian_angular_functions", shape=(),
                        data=scfres.mol.cart)
    data.create_dataset("charge", shape=(), data=int(scfres.mol.charge))

    # Note: In the pyscf world spin is 2S, so the multiplicity
    #       is spin + 1
    data.create_dataset(
        "spin_multiplicity", shape=(), data=int(scfres.mol.spin) + 1
    )

    #
    # Orbital reordering
    #
    # TODO This should not be needed any more
    # adcc assumes that the occupied orbitals are specified first,
    # followed by the virtual orbitals. Pyscf does this by means of the
    # mo_occ numpy arrays, so we need to reorder in order to agree
    # with what is expected in adcc.
    #
    # First build a structured numpy array with the negative occupation
    # in the primary field and the energy in the secondary
    # for each alpha and beta
    order_array = (
        np.array(list(zip(-mo_occ[0], mo_energy[0])),
                 dtype=np.dtype("float,float")),
        np.array(list(zip(-mo_occ[1], mo_energy[1])),
                 dtype=np.dtype("float,float")),
    )
    sort_indices = tuple(np.argsort(ary) for ary in order_array)

    # Use the indices which sort order_array (== stort_indices) to reorder
    mo_occ = tuple(mo_occ[i][sort_indices[i]] for i in range(2))
    mo_energy = tuple(mo_energy[i][sort_indices[i]] for i in range(2))
    mo_coeff = tuple(mo_coeff[i][:, sort_indices[i]] for i in range(2))
    fock = tuple(fock[i][sort_indices[i]][:, sort_indices[i]] for i in range(2))

    #
    # SCF orbitals and SCF results
    #
    data.create_dataset("occupation_f", data=np.hstack((mo_occ[0], mo_occ[1])))
    data.create_dataset("orben_f", data=np.hstack((mo_energy[0], mo_energy[1])))
    fullfock_ff = np.zeros((n_orbs, n_orbs))
    fullfock_ff[:n_orbs_alpha, :n_orbs_alpha] = fock[0]
    fullfock_ff[n_orbs_alpha:, n_orbs_alpha:] = fock[1]
    data.create_dataset("fock_ff", data=fullfock_ff, compression=8)

    fock_bf = np.hstack((fock_bb[0], fock_bb[1]))
    data.create_dataset("fock_fb", data=fock_bf.transpose(), compression=8)

    non_canonical = np.max(np.abs(data["fock_ff"] - np.diag(data["orben_f"])))
    if non_canonical > data["conv_tol"][()]:
        raise ValueError("Running adcc on top of a non-canonical fock "
                         "matrix is not implemented.")

    cf_bf = np.hstack((mo_coeff[0], mo_coeff[1]))
    data.create_dataset("orbcoeff_fb", data=cf_bf.transpose(), compression=8)

    #
    # ERI AO to MO transformation
    #
    if hasattr(scfres, "_eri") and scfres._eri is not None:
        # eri is stored ... use it directly
        eri_ao = scfres._eri
    else:
        # eri is not stored ... generate it now.
        eri_ao = scfres.mol.intor("int2e", aosym="s8")

    aro = slice(0, n_alpha)
    bro = slice(n_orbs_alpha, n_orbs_alpha + n_beta)
    arv = slice(n_alpha, n_orbs_alpha)
    brv = slice(n_orbs_alpha + n_beta, n_orbs)

    # compute full ERI tensor (with really everything)
    eri = ao2mo.general(eri_ao, (cf_bf, cf_bf, cf_bf, cf_bf), compact=False)
    eri = eri.reshape(n_orbs, n_orbs, n_orbs, n_orbs)
    del eri_ao

    # Adjust spin-forbidden blocks to be exactly zero
    eri[aro, bro, :, :] = 0
    eri[aro, brv, :, :] = 0
    eri[arv, bro, :, :] = 0
    eri[arv, brv, :, :] = 0

    eri[bro, aro, :, :] = 0
    eri[bro, arv, :, :] = 0
    eri[brv, aro, :, :] = 0
    eri[brv, arv, :, :] = 0

    eri[:, :, aro, bro] = 0
    eri[:, :, aro, brv] = 0
    eri[:, :, arv, bro] = 0
    eri[:, :, arv, brv] = 0

    eri[:, :, bro, aro] = 0
    eri[:, :, bro, arv] = 0
    eri[:, :, brv, aro] = 0
    eri[:, :, brv, arv] = 0
    data.create_dataset("eri_ffff", data=eri, compression=8)

    # Compute electric and nuclear multipole moments
    charges = scfres.mol.atom_charges()
    coords = scfres.mol.atom_coords()
    mmp = data.create_group("multipoles")
    mmp.create_dataset("nuclear_0", shape=(), data=int(np.sum(charges)))
    mmp.create_dataset("nuclear_1", data=np.einsum("i,ix->x", charges, coords))
    mmp.create_dataset("elec_0", shape=(), data=-int(n_alpha + n_beta))
    mmp.create_dataset("elec_1",
                       data=scfres.mol.intor_symmetric("int1e_r", comp=3))

    magm = data.create_group("magnetic_moments")
    derivs = data.create_group("derivatives")
    with scfres.mol.with_common_orig([0.0, 0.0, 0.0]):
        magm.create_dataset(
            "mag_1",
            data=0.5 * scfres.mol.intor('int1e_cg_irxp', comp=3, hermi=2)
        )
        derivs.create_dataset(
            "nabla",
            data=-1.0 * scfres.mol.intor('int1e_ipovlp', comp=3, hermi=2)
        )

    data.attrs["backend"] = "pyscf"
    return data
