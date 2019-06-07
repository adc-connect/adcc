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
import os
import tempfile
import numpy as np

from mpi4py import MPI

import veloxchem as vlx

from libadcc import HartreeFockProvider
from .eri_build_helper import (EriBuilder, SpinBlockSlice,
                               get_symmetry_equivalent_transpositions_for_block)
from adcc.misc import cached_property


class VeloxChemOperatorIntegralProvider:
    def __init__(self, scfdrv):
        self.scfdrv = scfdrv
        self.backend = "veloxchem"

    @cached_property
    def electric_dipole(self, component="x"):
        return list(self.scfdrv.scf_tensors['Mu'])


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

    def compute_eri_block_vlx(self, block):
        grps = [p for p in range(self.task.mpi_size)]
        return self.moints_drv.compute(self.molecule, self.ao_basis,
                                       self.mol_orbs, block,
                                       grps)

    # TODO: cleanup
    def compute_mo_asym_eri(self, asym_block, spin_block):
        cache_key = asym_block + spin_block
        if cache_key in self.eri_asymm_cache:
            return self.eri_asymm_cache[cache_key]

        n_alpha = self.n_alpha
        n_orbs_alpha = self.n_orbs_alpha

        indices = [n_alpha if x == "O" else n_orbs_alpha - n_alpha
                   for x in asym_block]
        eris = np.zeros((indices[0], indices[1],
                         indices[2], indices[3]))

        grps = [p for p in range(self.task.mpi_size)]

        vlx_block = "ASYM_" + asym_block.upper()
        # VeloxChem cannot compute this block, but it might be
        # needed in some CVS cases
        if asym_block == "OVOO":
            vlx_block = "ASYM_OOOV"
            eris = eris.transpose(2, 3, 0, 1)

        if vlx_block in self.eri_cache.keys():
            blck = self.eri_cache[vlx_block]
        else:
            blck = self.moints_drv.compute(self.molecule, self.ao_basis,
                                           self.mol_orbs, vlx_block,
                                           grps)
            self.eri_cache[vlx_block] = blck

        ioffset = n_alpha if asym_block == "VVVV" else 0
        joffset = n_alpha if (asym_block == "OVVV" or asym_block == "OVOV"
                              or asym_block == "VVVV") else 0

        if asym_block != "OVOV" and asym_block != "OVVV":
            for idx, pair in enumerate(blck.get_gen_pairs()):
                i = pair.first()
                j = pair.second()
                fxy = blck.xy_to_numpy(idx)
                fyx = blck.yx_to_numpy(idx)
                if spin_block == "aaaa" or spin_block == "bbbb":
                    eris[i - ioffset, j - joffset, :, :] = fxy - fyx.T
                    eri_ij = eris[i - ioffset, j - joffset, :, :]
                    eris[j - joffset, i - ioffset, :, :] = - eri_ij
                elif spin_block == "abab" or spin_block == "baba":
                    eris[i - ioffset, j - joffset, :, :] = fxy
                    eris[j - joffset, i - ioffset, :, :] = fyx.T
                elif spin_block == "abba" or spin_block == "baab":
                    eris[i - ioffset, j - joffset, :, :] = -fyx.T
                    eris[j - joffset, i - ioffset, :, :] = -fxy
        else:
            for idx, pair in enumerate(blck.get_gen_pairs()):
                i = pair.first()
                j = pair.second()
                if spin_block == "aaaa" or spin_block == "bbbb":
                    fxy = blck.xy_to_numpy(idx)
                    fyx = blck.yx_to_numpy(idx)
                    eris[i - ioffset, j - joffset, :, :] = fxy - fyx.T
                elif spin_block == "abab" or spin_block == "baba":
                    fxy = blck.xy_to_numpy(idx)
                    eris[i - ioffset, j - joffset, :, :] = fxy
                elif spin_block == "abba":
                    fyx = blck.yx_to_numpy(idx)
                    eris[i - ioffset, j - joffset, :, :] = -fyx.T
        if asym_block == "OVOO":
            eris = eris.transpose(2, 3, 0, 1)
        self.eri_asymm_cache[cache_key] = eris
        return eris

    @property
    def has_mo_asym_eri(self):
        return True

    def flush_cache(self):
        self.eri_asymm_cache = {}
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
        self.eri_builder = VeloxChemEriBuilder(
            self.scfdrv.task, self.mol_orbs, self.n_orbs, self.n_orbs_alpha,
            self.n_alpha, self.n_beta
        )

        self.operator_integral_provider = VeloxChemOperatorIntegralProvider(
            self.scfdrv
        )

    def get_backend(self):
        return "veloxchem"

    def get_n_alpha(self):
        return self.molecule.number_of_alpha_electrons()

    def get_n_beta(self):
        return self.get_n_alpha()

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

    def get_n_orbs_beta(self):
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
        n_oa = self.mol_orbs.number_mos()
        out[:] = np.zeros(2 * n_oa)
        out[:self.get_n_alpha()] = 1.
        out[n_oa:n_oa + self.get_n_beta()] = 1.

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
    if not isinstance(scfdrv, vlx.scfrestdriver.ScfRestrictedDriver):
        raise TypeError("Unsupported type for backends.veloxchem.import_scf.")

    if not hasattr(scfdrv, "task"):
        raise TypeError("Please attach the VeloxChem task to "
                        "the VeloxChem SCF driver")

    if not scfdrv.is_converged:
        raise ValueError("Cannot start an adc calculation on top of an SCF, "
                         "which is not converged.")

    provider = VeloxChemHFProvider(scfdrv)
    return provider


basis_remap = {
    "sto3g": "sto-3g",
    "def2tzvp": "def2-tzvp",
    "ccpvdz": "cc-pvdz",
}


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=None,
           conv_tol_grad=1e-8, max_iter=150):
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
