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
from mpi4py import MPI
import veloxchem as vlx

from .eri_build_helper import (SpinBlockSlice,
                               get_symmetry_equivalent_transpositions_for_block,
                               is_spin_allowed,
                               BlockSliceMappingHelper)


class VeloxChemHFProvider(HartreeFockProvider):
    """
        This implementation is only valid for RHF
    """
    def __init__(self, scfdrv):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.backend = "veloxchem"
        self.scfdrv = scfdrv
        self.mol_orbs = self.scfdrv.mol_orbs
        self.molecule = self.scfdrv.task.molecule
        self.ao_basis = self.scfdrv.task.ao_basis
        self.mpi_comm = self.scfdrv.task.mpi_comm
        self.ostream = self.scfdrv.task.ostream
        self.energy_terms = {
            "nuclear_repulsion": self.molecule.nuclear_repulsion_energy()
        }
        self.moints_drv = vlx.MOIntegralsDriver()
        self.eri_ffff = None
        self.block_slice_mapping = None

        # TODO: do caching in a clever way
        self.eri_cache = {}

    def compute_eri_block_vlx(self, block):
        return self.moints_drv.compute(self.molecule, self.ao_basis,
                                       self.mol_orbs, block,
                                       self.mpi_comm, self.ostream)

    def compute_mo_eri(self, block):
        # if block in self.eri_cache:
        #     return self.eri_cache[block]
        n_alpha = self.n_alpha
        n_orbs_alpha = self.n_orbs_alpha

        blck = self.moints_drv.compute(self.molecule, self.ao_basis,
                                       self.mol_orbs, block,
                                       self.mpi_comm, self.ostream)
        indices = [n_alpha if x == "O" else n_orbs_alpha - n_alpha
                   for x in block]
        if block != "OOVV":
            indices = [indices[i] for i in [0, 2, 1, 3]]

        # make canonical integral block
        can_block_integrals = np.zeros((indices[0], indices[1],
                                        indices[2], indices[3]))
        # offset to retrieve integrals
        ioffset = n_alpha if block == "VVVV" else 0
        joffset = n_alpha if (block == "OVVV" or block == "VVVV") else 0
        # fill canonical block
        for i in range(ioffset, indices[0] + ioffset):
            for j in range(joffset, indices[1] + joffset):
                ij = blck.to_numpy(vlx.TwoIndexes(i, j))
                can_block_integrals[i - ioffset,
                                    j - joffset, :, :] = ij[:, :]
        self.eri_cache[block] = can_block_integrals
        return can_block_integrals

    def build_eri_phys_asym_block(self, can_block=None, spin_block=None,
                                  spin_symm=None):
        block = can_block
        asym_block = "".join([block[i] for i in [0, 3, 2, 1]])
        print(block, asym_block)
        # TODO: cleanup a bit
        if spin_symm.pref1 != 0 and spin_symm.pref2 != 0:
            can_block_integrals = self.compute_mo_eri(block)
            eri_phys = can_block_integrals.transpose(0, 2, 1, 3)
            # (ik|jl) - (il|jk)
            asymm = self.compute_mo_eri(asym_block).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
            eris = spin_symm.pref1 * eri_phys - spin_symm.pref2 * asymm
        elif spin_symm.pref1 != 0 and spin_symm.pref2 == 0:
            can_block_integrals = self.compute_mo_eri(block)
            eris = spin_symm.pref1 * can_block_integrals.transpose(0, 2, 1, 3)
        elif spin_symm.pref1 == 0 and spin_symm.pref2 != 0:
            asymm = self.compute_mo_eri(asym_block).transpose(0, 3, 2, 1).transpose(0, 2, 1, 3)
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

        blocks = ["OOVV", "OVOV", "OOOV", "OOOO", "OVVV", "VVVV"]
        for b in blocks:
            blck = self.compute_eri_block_vlx(b)
            indices = [n_alpha if x == "O" else n_orbs_alpha - n_alpha
                       for x in b]
            slices_alpha = [aro if x == "O" else arv for x in b]
            slices_beta = [bro if x == "O" else brv for x in b]

            # TODO: improve this, very annoying code
            slices_alpha = [slices_alpha[i] for i in [0, 2, 1, 3]]
            slices_beta = [slices_beta[i] for i in [0, 2, 1, 3]]
            if b != "OOVV":
                indices = [indices[i] for i in [0, 2, 1, 3]]

            # make canonical integral block
            can_block_integrals = np.zeros((indices[0], indices[1],
                                            indices[2], indices[3]))
            # offset to retrieve integrals
            ioffset = n_alpha if b == "VVVV" else 0
            joffset = n_alpha if (b == "OVVV" or b == "VVVV") else 0
            # fill canonical block
            for i in range(ioffset, indices[0] + ioffset):
                for j in range(joffset, indices[1] + joffset):
                    ij = blck.to_numpy(vlx.TwoIndexes(i, j))
                    can_block_integrals[i - ioffset,
                                        j - joffset, :, :] = ij[:, :]
            # the only block where we get stuff in physicists' notation from vlx
            # convert to chemists' notation and rename block
            if b == "OOVV":
                can_block_integrals = can_block_integrals.swapaxes(1, 2)
                b = "OVOV"
            else:
                b = "".join([b[i] for i in [0, 2, 1, 3]])
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

    def get_n_alpha(self):
        return self.molecule.number_of_alpha_electrons()

    def get_n_beta(self):
        return self.get_n_alpha()

    def get_threshold(self):
        return self.scfdrv.conv_thresh

    def get_restricted(self):
        return True

    def get_energy_term(self, term):
        return self.energy_terms[term]

    def get_energy_scf(self):
        return self.scfdrv.get_scf_energy()

    def get_spin_multiplicity(self):
        return self.molecule.get_multiplicity()

    def get_n_orbs_alpha(self):
        return self.mol_orbs.number_mos()

    def get_n_orbs_beta(self):
        return self.mol_orbs.number_mos()

    def get_n_bas(self):
        return self.mol_orbs.number_aos()

    def fill_orbcoeff_fb(self, out):
        mo_coeff_a = self.mol_orbs.alpha_to_numpy()
        mo_coeff = (mo_coeff_a, mo_coeff_a)
        out[:] = np.transpose(
            np.hstack((mo_coeff[0].T, mo_coeff[1].T))
        )

    def fill_orben_f(self, out):
        orben_a = self.mol_orbs.ea_to_numpy()
        out[:] = np.hstack((orben_a, orben_a))

    def fill_fock_ff(self, slices, out):
        # TODO: get Fock matrix from Veloxchem properly
        diagonal = np.empty(self.n_orbs)
        self.fill_orben_f(diagonal)
        out[:] = np.diag(diagonal)[slices]

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

    # TODO: overload eri_asym_ffff, has_eri_asym_ffff
    def has_eri_phys_asym_ffff(self):
        return True

    def get_energy_term_keys(self):
        # TODO: implement full set of keys
        return ["nuclear_repulsion"]

    def flush_cache(self):
        self.eri_ffff = None


def import_scf(scfdrv):
    if not isinstance(scfdrv, vlx.scfrestdriver.ScfRestrictedDriver):
        raise TypeError("Unsupported type for backends.veloxchem.import_scf.")

    if not hasattr(scfdrv, "task"):
        raise TypeError("Please attach the VeloxChem task to "
                        "the VeloxChem SCF driver")

    # TODO
    # if not scfdrv.is_converged:
    #     raise ValueError("Cannot start an adc calculation on top of an SCF, "
    #                      "which is not converged.")

    provider = VeloxChemHFProvider(scfdrv)
    return provider
