## Taken from adcc-testdata
##
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
    del fock_bb

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

    #
    # Put basic data into HDF5 file
    #
    data.create_dataset("n_orbs_alpha", shape=(), data=int(n_orbs_alpha))
    data.create_dataset("energy_scf", shape=(), data=float(scfres.e_tot))
    data.create_dataset("restricted", shape=(), data=restricted)
    data.create_dataset("conv_tol", shape=(), data=float(threshold))

    if restricted:
        # Note: In the pyscf world spin is 2S, so the multiplicity
        #       is spin + 1
        data.create_dataset(
            "spin_multiplicity", shape=(), data=int(scfres.mol.spin) + 1
        )
    else:
        data.create_dataset("spin_multiplicity", shape=(), data=0)

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

    # Calculate mass center and charge center
    mass = scfres.mol.atom_mass_list()
    charges = scfres.mol.atom_charges()
    coords = scfres.mol.atom_coords()
    mass_center = np.einsum('i,ij->j', mass, coords) / mass.sum()
    print(mass_center)
    charge_center = np.einsum('i,ij->j', charges, coords) / charges.sum()

    # Compute electric and nuclear multipole moments
    mmp = data.create_group("multipoles")
    mmp.create_dataset("nuclear_0", shape=(), data=int(np.sum(charges)))
    mmp.create_dataset("nuclear_1", data=np.einsum("i,ix->x", charges, coords))
    mmp.create_dataset("elec_0", shape=(), data=-int(n_alpha + n_beta))
    mmp.create_dataset("elec_1",
                       data=scfres.mol.intor_symmetric("int1e_r", comp=3))
    with scfres.mol.with_common_orig([0.0, 0.0, 0.0]):
        mmp.create_dataset("elec_2",
                           data=scfres.mol.intor_symmetric('int1e_rr', comp=9))
    with scfres.mol.with_common_orig(mass_center):
        mmp.create_dataset("elec_2_mass_center",
                           data=scfres.mol.intor_symmetric('int1e_rr', comp=9))
    with scfres.mol.with_common_orig(charge_center):
        mmp.create_dataset("elec_2_charge_center",
                           data=scfres.mol.intor_symmetric('int1e_rr', comp=9))

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
    with scfres.mol.with_common_orig(mass_center):
        magm.create_dataset(
            "mag_1_mass_center",
            data=0.5 * scfres.mol.intor('int1e_cg_irxp', comp=3, hermi=2)
        )
        derivs.create_dataset(
            "nabla_mass_center",
            data=-1.0 * scfres.mol.intor('int1e_ipovlp', comp=3, hermi=2)
        )
    with scfres.mol.with_common_orig(charge_center):
        magm.create_dataset(
            "mag_1_charge_center",
            data=0.5 * scfres.mol.intor('int1e_cg_irxp', comp=3, hermi=2)
        )
        derivs.create_dataset(
            "nabla_charge_center",
            data=-1.0 * scfres.mol.intor('int1e_ipovlp', comp=3, hermi=2)
        )

    data.attrs["backend"] = "pyscf"
    return data
