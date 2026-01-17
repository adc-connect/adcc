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
import h5py

import pyscf
from pyscf import ao2mo, scf, gto

from adcc.backends.pyscf import import_scf, PyScfHFProvider
from adcc import hdf5io


def dump_pyscf(scfres: scf.hf.SCF, hdf5_file: h5py.Group):
    """
    Convert pyscf SCF result to HDF5 file in adcc format
    """
    data = {}
    # build a HFProvider isntance to obtain meta data about the SCF
    # like restricted, convergence tolerance, ...
    # Tensors like MO coefficients, not antisymmetrised ERI, ... are imported
    # directly from the pyscf object.
    hfprovider: PyScfHFProvider = import_scf(scfres)

    data["energy_scf"] = hfprovider.get_energy_scf()

    # Determine the number of orbitals
    n_orbs = hfprovider.n_orbs
    n_orbs_alpha = hfprovider.get_n_orbs_alpha()
    assert n_orbs == 2 * n_orbs_alpha  # equal number of alpha and beta orbitals

    data["n_orbs_alpha"] = n_orbs_alpha

    # dump the scf convergence tolerance
    data["conv_tol"] = hfprovider.get_conv_tol()

    restricted = hfprovider.get_restricted()
    data["restricted"] = restricted

    # NOTE: The following parameters are needed to perform a QChem calculation
    # on top of the reference data.
    # - format the basis set in qchem format
    data["qchem_formatted_basis"] = get_qchem_formatted_basis(scfres.mol)
    # - dump whether cartesian angular functions have been used
    data["cartesian_angular_functions"] = scfres.mol.cart
    # - dump the xyz geometry and the unit
    geom, unit = get_xyz_geometry(scfres.mol)
    data["xyz"] = geom
    data["xyz_unit"] = unit
    # - the charge of the system
    data["charge"] = scfres.mol.charge
    # - the multiplicity of the system
    data["spin_multiplicity"] = hfprovider.get_spin_multiplicity()

    # get the MO coefsfs, MO energies, Fock matrix in the AO basis and the
    # occupation numbers.
    mo_occ = scfres.mo_occ
    assert isinstance(mo_occ, np.ndarray)
    mo_energy = scfres.mo_energy
    assert isinstance(mo_energy, np.ndarray)
    mo_coeff = scfres.mo_coeff
    assert isinstance(mo_coeff, np.ndarray)
    fock_bb = scfres.get_fock()

    # pyscf only keeps occupation and mo energies once if restriced,
    # so we unfold it here in order to unify the treatment in the rest
    # of the code
    if restricted:
        mo_occ = np.asarray((mo_occ / 2, mo_occ / 2))
        mo_energy = (mo_energy, mo_energy)
        mo_coeff = (mo_coeff, mo_coeff)
        fock_bb = (fock_bb, fock_bb)

    # stacked fock matrix in ao basis (needed for the qchem calculation)
    # a ( )
    # b ( )
    data["fock_bb"] = np.vstack((fock_bb[0], fock_bb[1]))

    # Determine number of electrons
    n_alpha = np.sum(mo_occ[0] > 0)
    n_beta = np.sum(mo_occ[1] > 0)
    if n_alpha != np.sum(mo_occ[0]) or n_beta != np.sum(mo_occ[1]):
        raise ValueError("Fractional occupation numbers are not supported "
                         "in adcc.")

    # NOTE: orbitals should already be sorted correctly:
    # occupied below virtual and according to their energy within the space.
    # assert this behaviour
    for spin in range(2):
        raw_order = list(zip(-mo_occ[spin], mo_energy[spin]))
        assert raw_order == sorted(raw_order)

    # Dump occupation numbers, orbital energies
    data["occupation_f"] = np.hstack((mo_occ[0], mo_occ[1]))
    data["orben_f"] = np.hstack((mo_energy[0], mo_energy[1]))
    # Transform fock matrix to MOs and build the full, block-diagonal matrix
    fock = tuple(
        mo_coeff[spin].transpose().conj() @ fock_bb[spin] @ mo_coeff[spin]
        for spin in range(2)
    )
    full_fock_ff = np.zeros((n_orbs, n_orbs))
    full_fock_ff[:n_orbs_alpha, :n_orbs_alpha] = fock[0]
    full_fock_ff[n_orbs_alpha:, n_orbs_alpha:] = fock[1]
    data["fock_ff"] = full_fock_ff

    non_canonical = np.max(np.abs(data["fock_ff"] - np.diag(data["orben_f"])))
    if non_canonical > max(1e-11, data["conv_tol"]):
        raise ValueError("Running adcc on top of a non-canonical fock "
                         "matrix is not implemented.")

    # Dump the stacked orbital coefficients
    #    b
    # a ( )
    # b ( )
    cf_bf = np.hstack((mo_coeff[0], mo_coeff[1]))
    data["orbcoeff_fb"] = cf_bf.transpose()

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
    assert isinstance(eri, np.ndarray)
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
    data["eri_ffff"] = eri

    # Calculate mass center and charge center
    masses = scfres.mol.atom_mass_list(isotope_avg=True)
    charges = scfres.mol.atom_charges()
    coords = scfres.mol.atom_coords()
    mass_center = np.einsum('i,ij->j', masses, coords) / masses.sum()
    charge_center = np.einsum('i,ij->j', charges, coords) / charges.sum()

    def calculate_nuclear_quadrupole(charges, coordinates, gauge_origin):
        coords = coordinates - gauge_origin
        r_r = np.einsum("ij,ik->ijk", coords, coords)
        res = np.einsum("i,ijk->jk", charges, r_r)
        res = [res[0, 0], res[0, 1], res[0, 2], res[1, 1], res[1, 2], res[2, 2]]
        return np.array(res)

    # Compute electric and nuclear multipole moments
    data["multipoles"] = {}
    data["multipoles"]["nuclear_0"] = int(np.sum(charges))
    data["multipoles"]["nuclear_1"] = np.einsum("i,ix->x", charges, coords)
    data["multipoles"]["nuclear_2_origin"] = \
        calculate_nuclear_quadrupole(charges, coords, (0, 0, 0))
    data["multipoles"]["nuclear_2_mass_center"] = \
        calculate_nuclear_quadrupole(charges, coords, mass_center)
    data["multipoles"]["nuclear_2_charge_center"] = \
        calculate_nuclear_quadrupole(charges, coords, charge_center)
    data["multipoles"]["elec_0"] = -int(n_alpha + n_beta)
    data["multipoles"]["elec_1"] = (
        -1.0 * scfres.mol.intor_symmetric("int1e_r", comp=3)
    )

    with scfres.mol.with_common_orig([0.0, 0.0, 0.0]):
        data["multipoles"]["elec_2_origin"] = (
            -1.0 * scfres.mol.intor_symmetric('int1e_rr', comp=9)
        )
    with scfres.mol.with_common_orig(mass_center):
        data["multipoles"]["elec_2_mass_center"] = (
            -1.0 * scfres.mol.intor_symmetric('int1e_rr', comp=9)
        )
    with scfres.mol.with_common_orig(charge_center):
        data["multipoles"]["elec_2_charge_center"] = (
            -1.0 * scfres.mol.intor_symmetric('int1e_rr', comp=9)
        )

    data["derivatives"] = {}
    data["derivatives"]["elec_vel_1"] = (
        scfres.mol.intor('int1e_ipovlp', comp=3, hermi=2)
    )

    data["magnetic_moments"] = {}
    with scfres.mol.with_common_orig([0.0, 0.0, 0.0]):
        data["magnetic_moments"]["mag_1_origin"] = (
            -0.5 * scfres.mol.intor('int1e_cg_irxp', comp=3, hermi=2)
        )
    with scfres.mol.with_common_orig(mass_center):
        data["magnetic_moments"]["mag_1_mass_center"] = (
            -0.5 * scfres.mol.intor('int1e_cg_irxp', comp=3, hermi=2)
        )
    with scfres.mol.with_common_orig(charge_center):
        data["magnetic_moments"]["mag_1_charge_center"] = (
            -0.5 * scfres.mol.intor('int1e_cg_irxp', comp=3, hermi=2)
        )

    hdf5io.emplace_dict(data, hdf5_file, compression="gzip")
    hdf5_file.attrs["backend"] = "pyscf"
    hdf5_file.attrs["pyscf_version"] = pyscf.__version__


def get_qchem_formatted_basis(mol: gto.Mole) -> str:
    """
    Extracts the basis information from the pyscf SCF result and converts the
    data to string using the QChem basis set format.
    """
    # translate the angular momentums
    angular_momentum_map = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}
    # iterate over the contracted gaussians and sort them according to their
    # atom id (the index of the atom in the geometry).
    cgtos_by_atom: dict[int, list] = {}
    for cgto_number in range(mol.nbas):
        atom_number: int = mol.bas_atom(cgto_number)
        atom_name: str = mol.elements[atom_number]
        cgto_exps: np.ndarray = mol.bas_exp(cgto_number)
        cgto_coeffs: np.ndarray = mol.bas_ctr_coeff(cgto_number)
        cgto_angular_momentum: int = mol.bas_angular(cgto_number)
        cgto_angular_momentum_name = angular_momentum_map[cgto_angular_momentum]
        if atom_number not in cgtos_by_atom:
            cgtos_by_atom[atom_number] = []
        cur_basis_function = (cgto_angular_momentum_name,
                              list(zip(cgto_exps, cgto_coeffs)))
        cgtos_by_atom[atom_number].append(cur_basis_function)
    qchem_formatted_basis: list[str] = []
    # sort the data to go through the atoms in ascending order
    # (the order they are given in the geomtry)
    for atom_number in sorted(cgtos_by_atom):
        atom_name: str = mol.elements[atom_number]
        qchem_formatted_basis.append(
            "{: <2s}    {: >3d}".format(atom_name, atom_number + 1)
        )
        for cgto in cgtos_by_atom[atom_number]:
            angular_momentum_name, primitive_gtos = cgto
            n_primitive_gtos = len(primitive_gtos)
            # each primitive GTO might have more than one coefficient, i.e.,
            # each pgto might be used in more than one cgto.
            n_cgtos_per_pgto = len(primitive_gtos[0][1])
            assert all(len(coeffs) == n_cgtos_per_pgto
                       for _, coeffs in primitive_gtos)
            for coeff_i in range(n_cgtos_per_pgto):
                qchem_formatted_basis.append(
                    "{:s}   {: >2d}   1.00".format(angular_momentum_name,
                                                   n_primitive_gtos)
                )
                for exp, coeffs in primitive_gtos:
                    coeff = coeffs[coeff_i]
                    basis_line = "{: >20.8E}{: >20.8E}".format(exp, coeff)
                    # qchem expects the fortran scientific format
                    qchem_formatted_basis.append(basis_line.replace("E", "D"))
        qchem_formatted_basis.append("****")
    return "\n".join(qchem_formatted_basis)


def get_xyz_geometry(mol: gto.Mole) -> tuple[str, str]:
    """
    Extracts the geometry from the pyscf Mole object as xyz coordinates in Bohr.
    """
    out = []
    coordinates = mol.atom_coords()
    for i in range(mol.natm):
        symbol = mol.atom_pure_symbol(i)
        x, y, z = coordinates[i]
        out.append(f"{symbol} {x:>20.15f} {y:>20.15f} {z:>20.15f}")
    return "\n".join(out), "bohr"
