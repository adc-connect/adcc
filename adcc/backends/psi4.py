#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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

from libadcc import HartreeFockProvider
import numpy as np
import psi4

from .eri_build_helper import (SpinBlockSlice,
                               get_symmetry_equivalent_transpositions_for_block,
                               is_spin_allowed,
                               BlockSliceMappingHelper)


class Psi4HFProvider(HartreeFockProvider):
    """
        This implementation is only valid for RHF
    """
    def __init__(self, wfn):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.backend = "psi4"
        self.wfn = wfn
        self.energy_terms = {
            "nuclear_repulsion": self.wfn.molecule().nuclear_repulsion_energy()
        }
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        self.eri_ffff = None
        self.eri_ffff_asymm = None
        self.c_all = None
        self.block_slice_mapping = None

        # TODO: do caching in a clever way
        self.eri_cache = {}
        self.transform_on_the_fly = False

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

        eri = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))
        co = self.wfn.Ca_subset("AO", "OCC")
        cv = self.wfn.Ca_subset("AO", "VIR")
        blocks = ["OOVV", "OVOV", "OOOV", "OOOO", "OVVV", "VVVV"]
        for b in blocks:
            # indices = [n_alpha if x == "O" else n_orbs_alpha - n_alpha
            #            for x in b]
            slices_alpha = [aro if x == "O" else arv for x in b]
            slices_beta = [bro if x == "O" else brv for x in b]
            coeffs_transform = tuple(co if x == "O" else cv for x in b)
            # make canonical integral block
            can_block_integrals = np.asarray(self.mints.mo_eri(*coeffs_transform))
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
        eri = np.asarray(self.mints.mo_eri(*coeffs))
        self.eri_cache[block] = eri
        return eri

    def build_eri_phys_asym_block(self, can_block=None, spin_block=None,
                                  spin_symm=None):
        co = self.wfn.Ca_subset("AO", "OCC")
        cv = self.wfn.Ca_subset("AO", "VIR")
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

    def transform_block_from_slices(self, slices, blocks, spin_symm):
        coeffs_transform = []
        for slice in slices:
            bla = psi4.core.Matrix.from_array(self.c_all[:, slice])
            coeffs_transform.append(bla)
        coeffs_transform = tuple(coeffs_transform[i] for i in [0, 2, 1, 3])
        if spin_symm.pref1 != 0 and spin_symm.pref2 != 0:
            can_block_integrals = np.asarray(self.mints.mo_eri(*coeffs_transform))
            eri_phys = can_block_integrals.transpose(0, 2, 1, 3)
            # (ik|jl) - (il|jk)
            chem_asym = tuple(coeffs_transform[i] for i in [0, 3, 2, 1])
            asymm = np.asarray(self.mints.mo_eri(*chem_asym)).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
            eris = spin_symm.pref1 * eri_phys - spin_symm.pref2 * asymm
        elif spin_symm.pref1 != 0 and spin_symm.pref2 == 0:
            can_block_integrals = np.asarray(self.mints.mo_eri(*coeffs_transform))
            eris = spin_symm.pref1 * can_block_integrals.transpose(0, 2, 1, 3)
        elif spin_symm.pref1 == 0 and spin_symm.pref2 != 0:
            chem_asym = tuple(coeffs_transform[i] for i in [0, 3, 2, 1])
            asymm = np.asarray(self.mints.mo_eri(*chem_asym)).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
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
        return self.wfn.nalpha()

    def get_n_beta(self):
        return self.get_n_alpha()

    def get_threshold(self):
        # psi4.core.get_option("SCF", "E_CONVERGENCE")
        # RMS value of the orbital gradient
        return psi4.core.get_option("SCF", "D_CONVERGENCE")

    def get_restricted(self):
        return True

    def get_energy_term(self, term):
        return self.energy_terms[term]

    def get_energy_scf(self):
        return self.wfn.energy()

    def get_spin_multiplicity(self):
        return self.wfn.molecule().multiplicity()

    def get_n_orbs_alpha(self):
        return self.wfn.nmo()

    def get_n_orbs_beta(self):
        return self.get_n_orbs_alpha()

    def get_n_bas(self):
        return self.wfn.basisset().nbf()

    def fill_orbcoeff_fb(self, out):
        mo_coeff_a = np.asarray(self.wfn.Ca())
        mo_coeff = (mo_coeff_a, mo_coeff_a)
        out[:] = np.transpose(
            np.hstack((mo_coeff[0].T, mo_coeff[1].T))
        )

    def fill_orben_f(self, out):
        orben_a = np.asarray(self.wfn.epsilon_a())
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
        if self.c_all is None and self.transform_on_the_fly:
            co_a = np.asarray(self.wfn.Ca_subset("AO", "OCC"))
            co_b = np.asarray(self.wfn.Cb_subset("AO", "OCC"))
            cv_a = np.asarray(self.wfn.Ca_subset("AO", "VIR"))
            cv_b = np.asarray(self.wfn.Cb_subset("AO", "VIR"))
            self.c_all = np.hstack((co_a, cv_a, co_b, cv_b))
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
            if self.transform_on_the_fly:
                print("OTF transformation")
                out[:] = self.transform_block_from_slices(slices=slices,
                                                          blocks=mo_spaces_chem,
                                                          spin_symm=spin_symm)
            else:
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


def import_scf(scfdrv):
    if not isinstance(scfdrv, psi4.core.RHF):
        raise TypeError("Unsupported type for backends.psi4.import_scf.")

    # TODO
    # if not scfdrv.is_converged:
    #     raise ValueError("Cannot start an adc calculation on top of an SCF, "
    #                      "which is not converged.")

    provider = Psi4HFProvider(scfdrv)
    return provider
