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

import veloxchem as vlx

from .EriBuilder import EriBuilder
from ..exceptions import InvalidReference
from ..ElectronicStates import EnergyCorrection

from veloxchem.mpitask import MpiTask
from veloxchem.veloxchemlib import (AngularMomentumIntegralsDriver,
                                    LinearMomentumIntegralsDriver)


class VeloxChemOperatorIntegralProvider:
    available: tuple[str, ...] = (
        "electric_dipole", "electric_dipole_velocity", "magnetic_dipole",
    )

    def __init__(self, scfdrv):
        self.scfdrv = scfdrv
        self.backend = "veloxchem"

    @property
    def electric_dipole(self) -> tuple[np.ndarray, ...]:
        """-sum_i r_i"""
        task = self.scfdrv.task
        dipole_drv = vlx.ElectricDipoleIntegralsDriver(task.mpi_comm)
        # define the origin for electric dipole integrals
        dipole_drv.origin = tuple(np.zeros(3))
        dipole_mats = dipole_drv.compute(task.molecule, task.ao_basis)
        return (-1.0 * dipole_mats.x_to_numpy(),
                -1.0 * dipole_mats.y_to_numpy(),
                -1.0 * dipole_mats.z_to_numpy())

    def magnetic_dipole(self, gauge_origin="origin") -> tuple[np.ndarray, ...]:
        """
        The imaginary part of the integral is returned.
        -0.5 * sum_i r_i x p_i
        """
        gauge_origin = _transform_gauge_origin_to_xyz(self.scfdrv, gauge_origin)
        task = self.scfdrv.task
        angmom_drv = AngularMomentumIntegralsDriver(task.mpi_comm)
        angmom_drv.origin = tuple(gauge_origin)
        angmom_mats = angmom_drv.compute(task.molecule, task.ao_basis)
        return (0.5 * angmom_mats.x_to_numpy(),
                0.5 * angmom_mats.y_to_numpy(),
                0.5 * angmom_mats.z_to_numpy())

    @property
    def electric_dipole_velocity(self) -> tuple[np.ndarray, ...]:
        """
        The imaginary part of the integral is returned.
        -sum_i p_i
        """
        task = self.scfdrv.task
        linmom_drv = LinearMomentumIntegralsDriver(task.mpi_comm)
        linmom_mats = linmom_drv.compute(task.molecule, task.ao_basis)
        return (-1.0 * linmom_mats.x_to_numpy(),
                -1.0 * linmom_mats.y_to_numpy(),
                -1.0 * linmom_mats.z_to_numpy())


class VeloxChemEriBuilder(EriBuilder):
    def __init__(self, task, mol_orbs, n_orbs, n_orbs_alpha, n_alpha,
                 n_beta, restricted):
        self.moints_drv = vlx.MOIntegralsDriver(task.mpi_comm, task.ostream)
        self.compute_args = (task.molecule, task.ao_basis, mol_orbs)
        super().__init__(n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted)

    def compute_mo_eri(self, blocks, spins):
        eri = self.moints_drv.compute_in_memory(*self.compute_args,
                                                moints_name="chem_" + blocks,
                                                moints_spin=spins)
        return eri


class VeloxChemHFProvider(HartreeFockProvider):
    """
        This implementation is valid for RHF and UHF
    """
    def __init__(self, scfdrv):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()

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
        if isinstance(self.scfdrv, vlx.scfrestdriver.ScfRestrictedDriver):
            restricted = True
        else:
            restricted = False
        return restricted

    def get_energy_scf(self):
        return self.scfdrv.get_scf_energy()

    def get_spin_multiplicity(self):
        return self.molecule.get_multiplicity()

    def get_n_orbs_alpha(self):
        return self.mol_orbs.alpha_to_numpy().shape[1]

    def get_n_bas(self):
        return self.mol_orbs.number_aos()

    def get_nuclear_multipole(self, order, gauge_origin=(0, 0, 0)):
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
        elif order == 2:
            coords = mol.get_coordinates_in_bohr() - gauge_origin
            r_r = np.einsum("ij,ik->ijk", coords, coords)
            res = np.einsum("i,ijk->jk", nuc_charges, r_r)
            res = [res[0, 0], res[0, 1], res[0, 2], res[1, 1], res[1, 2], res[2, 2]]
            return np.array(res)
        else:
            raise NotImplementedError("get_nuclear_multipole with order > 2")

    def transform_gauge_origin_to_xyz(self, gauge_origin):
        return _transform_gauge_origin_to_xyz(self.scfdrv, gauge_origin)

    def fill_occupation_f(self, out):
        n_mo = self.mol_orbs.number_mos()
        occ_a = self.molecule.get_aufbau_alpha_occupation(n_mo)
        occ_b = self.molecule.get_aufbau_beta_occupation(n_mo)
        out[:] = np.hstack((occ_a, occ_b))

    def fill_orbcoeff_fb(self, out):
        mo_coeff_a = self.mol_orbs.alpha_to_numpy()
        mo_coeff_b = self.mol_orbs.beta_to_numpy()
        out[:] = np.transpose(
            np.hstack((mo_coeff_a, mo_coeff_b))
        )

    def fill_orben_f(self, out):
        orben_a = self.mol_orbs.ea_to_numpy()
        orben_b = self.mol_orbs.eb_to_numpy()
        out[:] = np.hstack((orben_a, orben_b))

    def fill_fock_ff(self, slices, out):
        diagonal = np.empty(self.n_orbs)
        self.fill_orben_f(diagonal)
        out[:] = np.diag(diagonal)[slices]

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

    if not hasattr(scfdrv, "task"):
        raise InvalidReference("Please attach the VeloxChem task to "
                               "the VeloxChem SCF driver")

    if not scfdrv.is_converged:
        raise InvalidReference("Cannot start an adc calculation on top "
                               "of an SCF, which is not converged.")

    provider = VeloxChemHFProvider(scfdrv)
    return provider


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=None, conv_tol_grad=1e-9,
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
            lines += ["@scf",
                      "conv_thresh: {}".format(conv_tol_grad),
                      "max_iter: {}".format(max_iter),
                      "@end", ""]
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
        if multiplicity == 1:
            scfdrv = vlx.ScfRestrictedDriver(task.mpi_comm, task.ostream)
        else:
            scfdrv = vlx.ScfUnrestrictedDriver(task.mpi_comm, task.ostream)
        scfdrv.update_settings(task.input_dict['scf'],
                               task.input_dict['method_settings'])
        scfdrv.compute(task.molecule, task.ao_basis, task.min_basis)
        scfdrv.task = task
    return scfdrv


def _transform_gauge_origin_to_xyz(scfdrv, gauge_origin):
    """
    Determines the gauge origin. If the gauge origin is defined as a tuple
    the coordinates need to be given in atomic units!
    """
    molecule = scfdrv.task.molecule
    coords = molecule.get_coordinates_in_bohr()
    charges = molecule.elem_ids_to_numpy()
    masses = molecule.masses_to_numpy()

    if gauge_origin == "mass_center":
        gauge_origin = tuple(np.einsum("i,ij->j", masses, coords) / masses.sum())
    elif gauge_origin == "charge_center":
        gauge_origin = tuple(np.einsum("i,ij->j", charges, coords)
                             / charges.sum())
    elif gauge_origin == "origin":
        gauge_origin = (0.0, 0.0, 0.0)
    elif isinstance(gauge_origin, tuple):
        gauge_origin = gauge_origin
    else:
        raise NotImplementedError("The gauge origin can be defined either by a "
                                  "keyword (origin, mass_center or charge_center) "
                                  "or by a tuple defining the Cartesian components "
                                  "e.g. (x, y, z)."
                                  )
    return gauge_origin
