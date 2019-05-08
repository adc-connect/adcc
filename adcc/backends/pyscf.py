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

from pyscf import ao2mo, scf

from libadcc import HfData, HartreeFockProvider

from .eri_build_helper import (SpinBlockSlice,
                               get_symmetry_equivalent_transpositions_for_block,
                               is_spin_allowed,
                               BlockSliceMappingHelper)


class PySCFHFProvider(HartreeFockProvider):
    """
        This implementation is only valid for RHF
    """
    def __init__(self, scfres):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.backend = "pyscf"
        self.scfres = scfres
        self.energy_terms = {
            "nuclear_repulsion": self.scfres.mol.energy_nuc()
        }
        self.eri_ffff = None
        self.eri_ffff_asymm = None
        self.block_slice_mapping = None

        # TODO: do caching in a clever way
        self.eri_cache = {}

    def build_full_eri_ffff(self):
        n_orbs = self.n_orbs
        n_alpha = self.n_alpha
        n_beta = self.n_beta
        n_orbs_alpha = self.n_orbs_alpha

        aro = slice(0, n_alpha)
        bro = slice(n_orbs_alpha, n_orbs_alpha + n_beta)
        arv = slice(n_alpha, n_orbs_alpha)
        brv = slice(n_orbs_alpha + n_beta, n_orbs)
        eri = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))

        nocc = n_alpha
        co = self.scfres.mo_coeff[:, :nocc]
        cv = self.scfres.mo_coeff[:, nocc:]
        blocks = ["OOVV", "OVOV", "OOOV", "OOOO", "OVVV", "VVVV"]
        for b in blocks:
            indices = [n_alpha if x == "O" else n_orbs_alpha - n_alpha
                       for x in b]
            slices_alpha = [aro if x == "O" else arv for x in b]
            slices_beta = [bro if x == "O" else brv for x in b]
            coeffs_transform = tuple(co if x == "O" else cv for x in b)
            # make canonical integral block
            can_block_integrals = ao2mo.general(self.scfres.mol,
                                                coeffs_transform,
                                                compact=False).reshape(indices[0],
                                                                       indices[1],
                                                                       indices[2],
                                                                       indices[3])
            # automatically set ERI tensor's symmetry-equivalent blocks
            trans_sym_blocks = get_symmetry_equivalent_transpositions_for_block(b)

            # Slices for the spin-allowed blocks
            aaaa = SpinBlockSlice("aaaa", (slices_alpha[0], slices_alpha[1],
                                           slices_alpha[2], slices_alpha[3]))
            bbbb = SpinBlockSlice("bbbb", (slices_beta[0], slices_beta[1],
                                           slices_beta[2], slices_beta[3]))
            aabb = SpinBlockSlice("aabb", (slices_alpha[0], slices_alpha[1],
                                           slices_beta[2], slices_beta[3]))
            bbaa = SpinBlockSlice("bbaa", (slices_beta[0], slices_beta[1],
                                           slices_alpha[2], slices_alpha[3]))
            non_zero_spin_block_slice_list = [aaaa, bbbb, aabb, bbaa]
            for tsym_block in trans_sym_blocks:
                sym_block_eri = can_block_integrals.transpose(tsym_block)
                for non_zero_spin_block in non_zero_spin_block_slice_list:
                    transposed_spin_slices = tuple(non_zero_spin_block.slices[i] for i in tsym_block)
                    eri[transposed_spin_slices] = sym_block_eri
        return eri

    def compute_mo_eri(self, block, coeffs):
        if block in self.eri_cache:
            return self.eri_cache[block]
        sizes = [i.shape[1] for i in coeffs]
        eri = ao2mo.general(self.scfres.mol, coeffs,
                            compact=False).reshape(sizes[0], sizes[1],
                                                   sizes[2], sizes[3])
        self.eri_cache[block] = eri
        return eri

    def build_eri_phys_asym_block(self, can_block=None, spin_block=None,
                                  spin_symm=None):
        nocc = self.n_alpha
        co = self.scfres.mo_coeff[:, :nocc]
        cv = self.scfres.mo_coeff[:, nocc:]
        block = can_block
        asym_block = "".join([block[i] for i in [0, 3, 2, 1]])
        coeffs_transform = tuple(co if x == "O" else cv for x in block)
        # TODO: cleanup a bit
        if spin_symm.pref1 != 0 and spin_symm.pref2 != 0:
            can_block_integrals = self.compute_mo_eri(block, coeffs_transform)
            eri_phys = can_block_integrals.transpose(0, 2, 1, 3)
            # (ik|jl) - (il|jk)
            chem_asym = tuple(coeffs_transform[i] for i in [0, 3, 2, 1])
            asymm = self.compute_mo_eri(asym_block,
                                        chem_asym).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
            eris = spin_symm.pref1 * eri_phys - spin_symm.pref2 * asymm
        elif spin_symm.pref1 != 0 and spin_symm.pref2 == 0:
            can_block_integrals = self.compute_mo_eri(block, coeffs_transform)
            eris = spin_symm.pref1 * can_block_integrals.transpose(0, 2, 1, 3)
        elif spin_symm.pref1 == 0 and spin_symm.pref2 != 0:
            chem_asym = tuple(coeffs_transform[i] for i in [0, 3, 2, 1])
            asymm = self.compute_mo_eri(asym_block,
                                        chem_asym).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
            eris = - spin_symm.pref2 * asymm
        return eris

    def prepare_block_slice_mapping(self):
        n_orbs = self.n_orbs
        n_alpha = self.n_alpha
        n_beta = self.n_beta
        n_orbs_alpha = self.n_orbs_alpha

        aro = slice(0, n_alpha, 1)
        bro = slice(n_orbs_alpha, n_orbs_alpha + n_beta, 1)
        arv = slice(n_alpha, n_orbs_alpha, 1)
        brv = slice(n_orbs_alpha + n_beta, n_orbs, 1)
        self.block_slice_mapping = BlockSliceMappingHelper(aro, bro,
                                                           arv, brv)

    def get_n_alpha(self):
        return np.sum(self.scfres.mo_occ > 0)

    def get_n_beta(self):
        return self.get_n_alpha()

    def get_threshold(self):
        if self.scfres.conv_tol_grad is None:
            conv_tol_grad = np.sqrt(self.scfres.conv_tol)
        else:
            conv_tol_grad = self.scfres.conv_tol_grad
        threshold = max(10 * self.scfres.conv_tol, conv_tol_grad)
        return threshold

    def get_restricted(self):
        return True

    def get_energy_term(self, term):
        return self.energy_terms[term]

    def get_energy_scf(self):
        return float(self.scfres.e_tot)

    def get_spin_multiplicity(self):
        # Note: In the pyscf world spin is 2S, so the multiplicity
        #       is spin + 1
        return int(self.scfres.mol.spin) + 1

    def get_n_orbs_alpha(self):
        return self.scfres.mo_coeff.shape[1]

    def get_n_orbs_beta(self):
        return self.get_n_orbs_alpha()

    def get_n_bas(self):
        return int(self.scfres.mol.nao_nr())

    def fill_orbcoeff_fb(self, out):
        mo_coeff_a = self.scfres.mo_coeff
        mo_coeff = (mo_coeff_a, mo_coeff_a)
        out[:] = np.transpose(
            np.hstack((mo_coeff[0].T, mo_coeff[1].T))
        )

    def fill_orben_f(self, out):
        orben_a = self.scfres.mo_energy
        out[:] = np.hstack((orben_a, orben_a))

    def fill_fock_ff(self, slices, out):
        diagonal = np.empty(self.n_orbs)
        self.fill_orben_f(diagonal)
        out[:] = np.diag(diagonal)[slices]

    # TODO: obsolete code, just used for testing
    def fill_eri_ffff(self, slices, out):
        if self.eri_ffff is None:
            self.eri_ffff = self.build_full_eri_ffff()
        out[:] = self.eri_ffff[slices]

    def fill_eri_phys_asym_ffff(self, slices, out):
        if not self.block_slice_mapping:
            self.prepare_block_slice_mapping()
        mo_spaces, \
            spin_block, \
            comp_block_slices \
            = self.block_slice_mapping.map_slices_to_blocks_and_spins(slices)
        if len(mo_spaces) != 4:
            raise RuntimeError("Could not assign MO spaces from slice"
                               " {},"
                               " found {} and spin {}".format(slices,
                                                              mo_spaces,
                                                              spin_block))
        mo_spaces_chem = "".join(np.take(np.array(mo_spaces), [0, 2, 1, 3]))
        spin_block_str = "".join(spin_block)
        print(mo_spaces, mo_spaces_chem, spin_block_str)
        allowed, spin_symm = is_spin_allowed(spin_block_str)
        if allowed:
            # TODO: cache ERIs nicely
            eri = self.build_eri_phys_asym_block(can_block=mo_spaces_chem,
                                                 spin_symm=spin_symm)
            out[:] = eri[comp_block_slices]
        else:
            out[:] = 0

    def has_eri_phys_asym_ffff(self):
        return True

    def get_energy_term_keys(self):
        # TODO: implement full set of keys
        return ["nuclear_repulsion"]

    def flush_cache(self):
        self.eri_ffff = None
        self.eri_cache = None
        self.c_all = None


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
    # TODO: new integral import also for UHF!
    if restricted:
        eri = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))
        nocc = n_alpha
        co = scfres.mo_coeff[:, :nocc]
        cv = scfres.mo_coeff[:, nocc:]
        blocks = ["OOVV", "OVOV", "OOOV", "OOOO", "OVVV", "VVVV"]
        for b in blocks:
            indices = [n_alpha if x == "O" else n_orbs_alpha - n_alpha
                       for x in b]
            slices_alpha = [aro if x == "O" else arv for x in b]
            slices_beta = [bro if x == "O" else brv for x in b]
            coeffs_transform = tuple(co if x == "O" else cv for x in b)
            # make canonical integral block
            can_block_integrals = ao2mo.general(eri_ao, coeffs_transform,
                                                compact=False).reshape(indices[0],
                                                                       indices[1],
                                                                       indices[2],
                                                                       indices[3])
            # automatically set ERI tensor's symmetry-equivalent blocks
            trans_sym_blocks = get_symmetry_equivalent_transpositions_for_block(b)

            # Slices for the spin-allowed blocks
            aaaa = SpinBlockSlice("aaaa", (slices_alpha[0], slices_alpha[1],
                                           slices_alpha[2], slices_alpha[3]))
            bbbb = SpinBlockSlice("bbbb", (slices_beta[0], slices_beta[1],
                                           slices_beta[2], slices_beta[3]))
            aabb = SpinBlockSlice("aabb", (slices_alpha[0], slices_alpha[1],
                                           slices_beta[2], slices_beta[3]))
            bbaa = SpinBlockSlice("bbaa", (slices_beta[0], slices_beta[1],
                                           slices_alpha[2], slices_alpha[3]))
            non_zero_spin_block_slice_list = [aaaa, bbbb, aabb, bbaa]
            for tsym_block in trans_sym_blocks:
                sym_block_eri = can_block_integrals.transpose(tsym_block)
                for non_zero_spin_block in non_zero_spin_block_slice_list:
                    transposed_spin_slices = tuple(non_zero_spin_block.slices[i] for i in tsym_block)
                    eri[transposed_spin_slices] = sym_block_eri
    else:
        # compute full ERI tensor (with really everything)
        eri = ao2mo.general(eri_ao,
                            (cf_bf, cf_bf, cf_bf, cf_bf), compact=False)
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
        provider = PySCFHFProvider(scfres)
        return provider
    else:
        # fallback
        print("WARNING: falling back to slow import for UHF result.")

        data = convert_scf_to_dict(scfres)
        ret = HfData.from_dict(data)
        ret.backend = "pyscf"

        # TODO temporary hack to make sure the data dict lives longer
        #      than the Hfdata object. Don't rely on this object for
        #      your code.
        ret._original_dict = data
        return ret
