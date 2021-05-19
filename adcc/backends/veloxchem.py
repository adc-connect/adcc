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

from mpi4py import MPI
from libadcc import HartreeFockProvider
from adcc.misc import cached_property

import veloxchem as vlx

from .EriBuilder import EriBuilder
from ..exceptions import InvalidReference
from ..ExcitedStates import EnergyCorrection

from veloxchem.mpitask import MpiTask
from veloxchem.veloxchemlib import (AngularMomentumIntegralsDriver,
                                    LinearMomentumIntegralsDriver)


class VeloxChemOperatorIntegralProvider:
    def __init__(self, scfdrv):
        self.scfdrv = scfdrv
        self.backend = "veloxchem"

    @cached_property
    def electric_dipole(self):
        task = self.scfdrv.task
        dipole_drv = vlx.ElectricDipoleIntegralsDriver(task.mpi_comm)
        dipole_mats = dipole_drv.compute(task.molecule, task.ao_basis)
        return [dipole_mats.x_to_numpy(), dipole_mats.y_to_numpy(),
                dipole_mats.z_to_numpy()]

    @cached_property
    def magnetic_dipole(self):
        # TODO: Gauge origin?
        task = self.scfdrv.task
        angmom_drv = AngularMomentumIntegralsDriver(task.mpi_comm)
        angmom_mats = angmom_drv.compute(task.molecule, task.ao_basis)
        return (0.5 * angmom_mats.x_to_numpy(), 0.5 * angmom_mats.y_to_numpy(),
                0.5 * angmom_mats.z_to_numpy())

    @cached_property
    def nabla(self):
        task = self.scfdrv.task
        linmom_drv = LinearMomentumIntegralsDriver(task.mpi_comm)
        linmom_mats = linmom_drv.compute(task.molecule, task.ao_basis)
        return (-1.0 * linmom_mats.x_to_numpy(), -1.0 * linmom_mats.y_to_numpy(),
                -1.0 * linmom_mats.z_to_numpy())


# VeloxChem is a special case... not using coefficients at all
# so we need this boilerplate code to make it work...
class VeloxChemEriBuilder(EriBuilder):
    def __init__(self, task, mol_orbs, n_orbs, n_orbs_alpha, n_alpha,
                 n_beta, restricted):
        self.moints_drv = vlx.MOIntegralsDriver(task.mpi_comm, task.ostream)
        self.compute_args = (task.molecule, task.ao_basis, mol_orbs)
        super().__init__(n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted)

    def compute_mo_eri(self, blocks, spins):
        assert self.restricted   # Veloxchem cannot do unrestricted
        assert spins == "aaaa"

        # Compute_in_mem return Physicists' integrals, so we first transform
        # the block specification from chemists' convention
        # to physicists' convention
        blocks_phys = blocks[0] + blocks[2] + blocks[1] + blocks[3]
        eri = self.moints_drv.compute_in_mem(*self.compute_args, blocks_phys)

        # Transform slice back to chemists' indexing convention and return
        return eri.transpose((0, 2, 1, 3))


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
        n_alpha = self.molecule.number_of_alpha_electrons()
        n_beta = self.molecule.number_of_beta_electrons()
        self.eri_builder = VeloxChemEriBuilder(
            self.scfdrv.task, self.mol_orbs, self.n_orbs, self.n_orbs_alpha,
            n_alpha, n_beta, self.restricted
        )

        self.operator_integral_provider = VeloxChemOperatorIntegralProvider(
            self.scfdrv
        )

    def pe_energy(self, dm, elec_only=True):
        e_pe, _ = self.scfdrv.pe_drv.get_pe_contribution(dm.to_ndarray(),
                                                         elec_only=elec_only)
        return e_pe

    @property
    def excitation_energy_corrections(self):
        ret = []
        if hasattr(self.scfdrv, "pe_drv"):
            ptlr = EnergyCorrection(
                "pe_ptlr_correction",
                lambda view: 2.0 * self.pe_energy(view.transition_dm_ao,
                                                  elec_only=True)
            )
            ptss = EnergyCorrection(
                "pe_ptss_correction",
                lambda view: self.pe_energy(view.state_diffdm_ao,
                                            elec_only=True)
            )
            ret.extend([ptlr, ptss])
        return {ec.name: ec for ec in ret}

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
        self.eri_builder.fill_slice_symm(slices, out)

    def fill_eri_phys_asym_ffff(self, slices, out):
        raise NotImplementedError("fill_eri_phys_asym_ffff not implemented.")

    def has_eri_phys_asym_ffff(self):
        return False

    def flush_cache(self):
        self.eri_builder.flush_cache()


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


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol_grad=1e-8,
           max_iter=150, pe_options=None):
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
                      "basis: {}".format(basis_remap.get(basis, basis))]
            # TODO: PE results in VeloxChem are currently wrong, because
            # polarizabilities are always made isotropic
            if pe_options:
                potfile = pe_options["potfile"]
                lines += ["pe: yes",
                          f"potfile: {potfile}"]
            lines += ["@end"]
            lines += ["@molecule",
                      "charge: {}".format(charge),
                      "multiplicity: {}".format(multiplicity),
                      "units: bohr",
                      "xyz:\n{}".format("\n".join(xyz.split(";"))),
                      "@end"]
            fp.write("\n".join(lines))
        task = MpiTask([infile, outfile], MPI.COMM_WORLD)

        scfdrv = vlx.ScfRestrictedDriver(task.mpi_comm, task.ostream)
        scfdrv.update_settings(task.input_dict['scf'],
                               task.input_dict['method_settings'])
        # elec. gradient norm
        scfdrv.conv_thresh = conv_tol_grad
        scfdrv.max_iter = max_iter
        scfdrv.compute(task.molecule, task.ao_basis, task.min_basis)
        scfdrv.task = task
    return scfdrv
