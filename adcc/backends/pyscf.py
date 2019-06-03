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
import warnings
import numpy as np

from adcc.DictHfProvider import DictHfProvider

from pyscf import ao2mo, gto, scf
from libadcc import HartreeFockProvider
from .eri_build_helper import EriBuilder


class PyScfOperatorIntegralProvider:
    def __init__(self, scfres):
        self.scfres = scfres
        self.backend = "pyscf"

    def electric_dipole(self, component="x"):
        # TODO avoid re-computation
        ao_dip = self.scfres.mol.intor_symmetric('int1e_r', comp=3)
        ao_dip = {k: ao_dip[i] for i, k in enumerate(['x', 'y', 'z'])}
        return ao_dip[component]

    def nuclear_dipole(self):
        # compute nuclear dipole
        charges = self.scfres.mol.atom_charges()
        coords = self.scfres.mol.atom_coords()
        return np.einsum('i,ix->x', charges, coords)

    def fock(self):
        return self.scfres.get_fock()


# TODO: refactor ERI builder to be more general
# IntegralBuilder would be good
class PyScfEriBuilder(EriBuilder):
    def __init__(self, scfres, n_orbs, n_orbs_alpha, n_alpha, n_beta):
        self.scfres = scfres
        super().__init__(n_orbs, n_orbs_alpha, n_alpha, n_beta)

    @property
    def coeffs_occ_alpha(self):
        return self.scfres.mo_coeff[:, :self.n_alpha]

    @property
    def coeffs_virt_alpha(self):
        return self.scfres.mo_coeff[:, self.n_alpha:]

    def compute_mo_eri(self, block, coeffs, use_cache=True):
        if block in self.eri_cache and use_cache:
            return self.eri_cache[block]
        sizes = [i.shape[1] for i in coeffs]
        eri = ao2mo.general(self.scfres.mol, coeffs,
                            compact=False).reshape(sizes[0], sizes[1],
                                                   sizes[2], sizes[3])
        self.eri_cache[block] = eri
        return eri


class PyScfHFProvider(HartreeFockProvider):
    """
        This implementation is only valid for RHF
    """
    def __init__(self, scfres):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.scfres = scfres
        self.eri_ffff = None
        self.eri_builder = PyScfEriBuilder(
            self.scfres, self.n_orbs, self.n_orbs_alpha,
            self.n_alpha, self.n_beta
        )

        self.operator_integral_provider = PyScfOperatorIntegralProvider(
            self.scfres
        )

    def get_backend(self):
        return "pyscf"

    def get_n_alpha(self):
        return np.sum(self.scfres.mo_occ > 0)

    def get_n_beta(self):
        return self.get_n_alpha()

    def get_conv_tol(self):
        if self.scfres.conv_tol_grad is None:
            conv_tol_grad = np.sqrt(self.scfres.conv_tol)
        else:
            conv_tol_grad = self.scfres.conv_tol_grad
        conv_tol = max(10 * self.scfres.conv_tol, conv_tol_grad)
        return conv_tol

    def get_restricted(self):
        if isinstance(self.scfres.mo_occ, list):
            restricted = len(self.scfres.mo_occ) < 2
        elif isinstance(self.scfres.mo_occ, np.ndarray):
            restricted = self.scfres.mo_occ.ndim < 2
        else:
            raise ValueError("Unusual pyscf SCF class encountered. Could not "
                             "determine restricted / unrestricted.")
        return restricted

    def get_energy_scf(self):
        return float(self.scfres.e_tot)

    def get_spin_multiplicity(self):
        # Note: In the pyscf world spin is 2S, so the multiplicity
        #       is spin + 1
        return int(self.scfres.mol.spin) + 1

    def get_n_orbs_alpha(self):
        if self.restricted:
            return self.scfres.mo_coeff.shape[1]
        else:
            return self.scfres.mo_coeff[0].shape[1]

    def get_n_orbs_beta(self):
        if self.restricted:
            return self.get_n_orbs_alpha()
        else:
            return self.scfres.mo_coeff[1].shape[1]

    def get_n_bas(self):
        return int(self.scfres.mol.nao_nr())

    def fill_occupation_f(self, out):
        if self.restricted:
            out[:] = np.hstack((self.scfres.mo_occ / 2,
                                self.scfres.mo_occ / 2))
        else:
            out[:] = np.hstack((self.scfres.mo_occ[0],
                                self.scfres.mo_occ[1]))

    def fill_orbcoeff_fb(self, out):
        if self.restricted:
            mo_coeff = (self.scfres.mo_coeff,
                        self.scfres.mo_coeff)
        else:
            mo_coeff = self.scfres.mo_coeff
        out[:] = np.transpose(
            np.hstack((mo_coeff[0], mo_coeff[1]))
        )

    def fill_orben_f(self, out):
        if self.restricted:
            out[:] = np.hstack((self.scfres.mo_energy,
                                self.scfres.mo_energy))
        else:
            out[:] = np.hstack((self.scfres.mo_energy[0],
                                self.scfres.mo_energy[1]))

    def fill_fock_ff(self, slices, out):
        diagonal = np.empty(self.n_orbs)
        self.fill_orben_f(diagonal)
        out[:] = np.diag(diagonal)[slices]

    def fill_eri_ffff(self, slices, out):
        if self.eri_ffff is None:
            self.eri_ffff = self.eri_builder.build_full_eri_ffff()
        out[:] = self.eri_ffff[slices]

    def fill_eri_phys_asym_ffff(self, slices, out):
        self.eri_builder.fill_slice(slices, out)

    def has_eri_phys_asym_ffff(self):
        return True

    def flush_cache(self):
        self.eri_builder.flush_cache()
        self.eri_ffff = None


def convert_scf_to_dict(scfres):
    if not isinstance(scfres, scf.hf.SCF):
        raise TypeError("Unsupported type for backends.pyscf.import_scf.")

    if not scfres.converged:
        raise ValueError("Cannot start an adc calculation on top of an SCF, "
                         "which is not yet converged. Did you forget to run "
                         "the kernel() or the scf() function of the pyscf scf "
                         "object?")

    # Try to determine whether we are restricted
    if isinstance(scfres.mo_occ, list):
        restricted = len(scfres.mo_occ) < 2
    elif isinstance(scfres.mo_occ, np.ndarray):
        restricted = scfres.mo_occ.ndim < 2
    else:
        raise ValueError("Unusual pyscf SCF class encountered. Could not "
                         "determine restricted / unrestricted.")

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
        raise ValueError("adcc cannot deal with different number of alpha and "
                         "beta orbitals like in a restricted "
                         "open-shell reference at the moment.")

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
    # Put basic data into a dictionary
    #
    data = {
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "n_orbs_alpha": int(n_orbs_alpha),
        "n_orbs_beta": int(n_orbs_beta),
        "n_bas": int(scfres.mol.nao_nr()),
        "energy_scf": float(scfres.e_tot),
        "energy_nuclear_repulsion": float(scfres.mol.energy_nuc()),
        "restricted": restricted,
        "threshold": float(threshold),
        "spin_multiplicity": 0,
    }
    if restricted:
        # Note: In the pyscf world spin is 2S, so the multiplicity
        #       is spin + 1
        data["spin_multiplicity"] = int(scfres.mol.spin) + 1

    #
    # Orbital reordering
    #
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
    data["occupation_f"] = np.hstack((mo_occ[0], mo_occ[1]))
    data["orben_f"] = np.hstack((mo_energy[0], mo_energy[1]))
    fullfock_ff = np.zeros((n_orbs, n_orbs))
    fullfock_ff[:n_orbs_alpha, :n_orbs_alpha] = fock[0]
    fullfock_ff[n_orbs_alpha:, n_orbs_alpha:] = fock[1]
    data["fock_ff"] = fullfock_ff

    non_canonical = np.max(np.abs(data["fock_ff"] - np.diag(data["orben_f"])))
    if non_canonical > data["threshold"]:
        raise ValueError("Running adcc on top of a non-canonical fock "
                         "matrix is not implemented.")

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
        eri_ao = scfres.mol.intor('int2e', aosym='s8')

    aro = slice(0, n_alpha)
    bro = slice(n_orbs_alpha, n_orbs_alpha + n_beta)
    arv = slice(n_alpha, n_orbs_alpha)
    brv = slice(n_orbs_alpha + n_beta, n_orbs)

    # compute full ERI tensor (with really everything)
    eri = ao2mo.general(
        eri_ao, (cf_bf, cf_bf, cf_bf, cf_bf), compact=False
    )
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
    data["backend"] = "pyscf"
    return data


def import_scf(scfres):
    if not isinstance(scfres, scf.hf.SCF):
        raise TypeError("Unsupported type for backends.pyscf.import_scf.")

    if not scfres.converged:
        raise ValueError("Cannot start an adc calculation on top of an SCF, "
                         "which is not yet converged. Did you forget to run "
                         "the kernel() or the scf() function of the pyscf scf "
                         "object?")

    # Try to determine whether we are restricted
    if isinstance(scfres.mo_occ, list):
        restricted = len(scfres.mo_occ) < 2
    elif isinstance(scfres.mo_occ, np.ndarray):
        restricted = scfres.mo_occ.ndim < 2
    else:
        raise ValueError("Unusual pyscf SCF class encountered. Could not "
                         "determine restricted / unrestricted.")

    if restricted:
        return PyScfHFProvider(scfres)
    else:
        warnings.warn("Falling back to slow import for UHF result.")
        return DictHfProvider(convert_scf_to_dict(scfres))


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-12,
           conv_tol_grad=1e-8, max_iter=150):
    mol = gto.M(
        atom=xyz,
        basis=basis,
        unit="Bohr",
        # spin in the pyscf world is 2S
        spin=multiplicity - 1,
        charge=charge,
        # Disable commandline argument parsing in pyscf
        parse_arg=False,
    )
    mf = scf.HF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_iter
    mf.kernel()
    return mf
