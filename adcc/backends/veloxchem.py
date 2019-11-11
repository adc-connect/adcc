#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import os
import tempfile
import numpy as np
import veloxchem as vlx

from .InvalidReference import InvalidReference
from .eri_build_helper import (EriBuilder, SpinBlockSlice,
                               get_symm_equivalent_transpositions_for_block)

from mpi4py import MPI
from adcc.misc import cached_property

from libadcc import HartreeFockProvider


class VeloxChemOperatorIntegralProvider:
    def __init__(self, scfdrv):
        self.scfdrv = scfdrv
        self.backend = "veloxchem"

    @cached_property
    def electric_dipole(self, component="x"):
        dipole_drv = vlx.ElectricDipoleIntegralsDriver(
            self.scfdrv.task.mpi_comm
        )
        dipole_mats = dipole_drv.compute(
            self.scfdrv.task.molecule, self.scfdrv.task.ao_basis
        )
        integrals = (dipole_mats.x_to_numpy(), dipole_mats.y_to_numpy(),
                     dipole_mats.z_to_numpy())
        return list(integrals)


# VeloxChem is a special case... not using coefficients at all
# so we need this boilerplate code to make it work...
class VeloxChemEriBuilder(EriBuilder):
    def __init__(self, task, mol_orbs,
                 n_orbs, n_orbs_alpha, n_alpha, n_beta):
        self.task = task
        self.molecule = self.task.molecule
        self.ao_basis = self.task.ao_basis
        self.mpi_comm = self.task.mpi_comm
        self.ostream = self.task.ostream
        self.mol_orbs = mol_orbs
        self.moints_drv = vlx.MOIntegralsDriver(self.mpi_comm, self.ostream)
        super().__init__(n_orbs, n_orbs_alpha, n_alpha, n_beta)

    def compute_mo_eri(self, block, coeffs=None, use_cache=True):
        # compute_in_mem return Physicists' integrals
        if isinstance(block, list):
            block = "".join(block)
        if block in self.eri_cache and use_cache:
            return self.eri_cache[block]
        eri = self.moints_drv.compute_in_mem(
            self.molecule, self.ao_basis, self.mol_orbs, block
        )
        self.eri_cache[block] = eri
        return eri

    @property
    def eri_notation(self):
        return "phys"

    @property
    def has_mo_asym_eri(self):
        return False

    def flush_cache(self):
        self.eri_asymm_cache = {}
        self.eri_cache = {}

    def build_eri_phys_asym_block(self, can_block=None, spin_symm=None):
        block = can_block
        asym_block = "".join([block[i] for i in [0, 1, 3, 2]])
        both_blocks = "{}-{}-{}-{}".format(
            block, asym_block, str(spin_symm.pref1), str(spin_symm.pref2)
        )
        if both_blocks in self.eri_asymm_cache.keys():
            return self.eri_asymm_cache[both_blocks]

        if spin_symm.pref1 != 0 and spin_symm.pref2 != 0:
            eri_phys = self.compute_mo_eri(block)
            # <ij|kl> - <ij|lk>
            asymm = self.compute_mo_eri(
                asym_block
            ).transpose(0, 1, 3, 2)
            eris = spin_symm.pref1 * eri_phys - spin_symm.pref2 * asymm
        elif spin_symm.pref1 != 0 and spin_symm.pref2 == 0:
            eri_phys = self.compute_mo_eri(block)
            eris = spin_symm.pref1 * eri_phys
        elif spin_symm.pref1 == 0 and spin_symm.pref2 != 0:
            asymm = self.compute_mo_eri(
                asym_block
            ).transpose(0, 1, 3, 2)
            eris = - spin_symm.pref2 * asymm

        self.eri_asymm_cache[both_blocks] = eris
        return eris

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
        for bl in blocks:
            # compute_in_mem return Physicists' integrals
            # we convert to Chemists' notation because we're building the
            # full ERI tensor for adcc, which requires Chemists' integrals
            can_block_integrals = self.compute_mo_eri(bl)
            b = bl[0] + bl[2] + bl[1] + bl[3]
            can_block_integrals = can_block_integrals.transpose(0, 2, 1, 3)
            slices_alpha = [aro if x == "O" else arv for x in b]
            slices_beta = [bro if x == "O" else brv for x in b]

            # automatically set ERI tensor's symmetry-equivalent blocks
            trans_sym_blocks = get_symm_equivalent_transpositions_for_block(
                b, notation="chem"
            )

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
                    transposed_spin_slices = tuple(
                        non_zero_spin_block.slices[i] for i in tsym_block
                    )
                    eri[transposed_spin_slices] = sym_block_eri
        return eri


class VeloxChemHFProvider(HartreeFockProvider):
    """
        This implementation is only valid for RHF
    """
    def __init__(self, scfdrv):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()

        if not isinstance(scfdrv, vlx.scfrestdriver.ScfRestrictedDriver):
            raise TypeError(
                "Only restricted references (RHF) are supported."
            )

        self.scfdrv = scfdrv
        self.mol_orbs = self.scfdrv.mol_orbs
        self.molecule = self.scfdrv.task.molecule
        self.ao_basis = self.scfdrv.task.ao_basis
        self.mpi_comm = self.scfdrv.task.mpi_comm
        self.ostream = self.scfdrv.task.ostream
        self.eri_ffff = None
        n_alpha = self.molecule.number_of_alpha_electrons()
        n_beta = self.molecule.number_of_beta_electrons()
        self.eri_builder = VeloxChemEriBuilder(
            self.scfdrv.task, self.mol_orbs, self.n_orbs, self.n_orbs_alpha,
            n_alpha, n_beta
        )

        self.operator_integral_provider = VeloxChemOperatorIntegralProvider(
            self.scfdrv
        )

    def get_backend(self):
        return "veloxchem"

    def get_conv_tol(self):
        return self.scfdrv.conv_thresh

    def get_restricted(self):
        return True  # The only one supported for now

    def get_energy_scf(self):
        return self.scfdrv.get_scf_energy()

    def get_spin_multiplicity(self):
        return self.molecule.get_multiplicity()

    def get_n_orbs_alpha(self):
        return self.mol_orbs.number_mos()

    def get_n_bas(self):
        return self.mol_orbs.number_aos()

    def get_nuclear_multipole(self, order):
        mol = self.scfdrv.task.molecule
        nuc_charges = mol.elem_ids_to_numpy()
        if order == 0:
            # The function interface needs to be a np.array on return
            return np.array([np.sum(nuc_charges)])
        elif order == 1:
            coords = np.vstack(
                (mol.x_to_numpy(), mol.y_to_numpy(), mol.z_to_numpy())
            ).transpose()
            return np.einsum('i,ix->x', nuc_charges, coords)
        else:
            raise NotImplementedError("get_nuclear_multipole with order > 1")

    def fill_orbcoeff_fb(self, out):
        mo_coeff_a = self.mol_orbs.alpha_to_numpy()
        mo_coeff_b = self.mol_orbs.beta_to_numpy()
        out[:] = np.transpose(
            np.hstack((mo_coeff_a, mo_coeff_b))
        )

    def fill_orben_f(self, out):
        orben_a = self.mol_orbs.ea_to_numpy()
        orben_b = self.mol_orbs.ea_to_numpy()
        out[:] = np.hstack((orben_a, orben_b))

    def fill_occupation_f(self, out):
        # TODO I (mfh) have no better idea than the fallback implementation
        noa = self.mol_orbs.number_mos()
        na = self.molecule.number_of_alpha_electrons()
        nb = self.molecule.number_of_beta_electrons()
        out[:] = np.zeros(2 * noa)
        out[noa:noa + nb] = out[:na] = 1.

    def fill_fock_ff(self, slices, out):
        mo_coeff_a = self.mol_orbs.alpha_to_numpy()
        mo_coeff = (mo_coeff_a, mo_coeff_a)
        fock_bb = self.scfdrv.scf_tensors['F']

        fock = tuple(mo_coeff[i].transpose().conj() @ fock_bb[i] @ mo_coeff[i]
                     for i in range(2))
        fullfock_ff = np.zeros((self.n_orbs, self.n_orbs))
        fullfock_ff[:self.n_orbs_alpha, :self.n_orbs_alpha] = fock[0]
        fullfock_ff[self.n_orbs_alpha:, self.n_orbs_alpha:] = fock[1]
        out[:] = fullfock_ff[slices]

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


def import_scf(scfdrv):
    # TODO The error messages in here could be a little more informative

    if not isinstance(scfdrv, vlx.scfrestdriver.ScfRestrictedDriver):
        raise InvalidReference("Unsupported type for "
                               "backends.veloxchem.import_scf.")

    if not hasattr(scfdrv, "task"):
        raise InvalidReference("Please attach the VeloxChem task to "
                               "the VeloxChem SCF driver")

    if not scfdrv.is_converged:
        raise InvalidReference("Cannot start an adc calculation on top "
                               "of an SCF, which is not converged.")

    provider = VeloxChemHFProvider(scfdrv)
    return provider


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=None,
           conv_tol_grad=1e-8, max_iter=150):
    basis_remap = {
        "sto3g": "sto-3g",
        "def2tzvp": "def2-tzvp",
        "ccpvdz": "cc-pvdz",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        infile = os.path.join(tmpdir, "vlx.in")
        outfile = os.path.join(tmpdir, "vlx.out")
        with open(infile, "w") as fp:
            lines = ["@jobs", "task: hf", "@end", ""]
            lines += ["@method settings",
                      "basis: {}".format(basis_remap.get(basis, basis)),
                      "@end"]
            lines += ["@molecule",
                      "charge: {}".format(charge),
                      "multiplicity: {}".format(multiplicity),
                      "units: bohr",
                      "xyz:\n{}".format("\n".join(xyz.split(";"))),
                      "@end"]
            fp.write("\n".join(lines))
        task = vlx.MpiTask([infile, outfile], MPI.COMM_WORLD)

        scfdrv = vlx.ScfRestrictedDriver(task.mpi_comm, task.ostream)
        # elec. gradient norm
        scfdrv.conv_thresh = conv_tol_grad
        scfdrv.max_iter = max_iter
        scfdrv.compute(task.molecule, task.ao_basis, task.min_basis)
        scfdrv.task = task
    return scfdrv
