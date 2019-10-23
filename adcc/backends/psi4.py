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
import psi4
import numpy as np

from .eri_build_helper import EriBuilder
from adcc.misc import cached_property

from libadcc import HartreeFockProvider


class Psi4OperatorIntegralProvider:
    def __init__(self, wfn):
        self.wfn = wfn
        self.backend = "psi4"
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())

    @cached_property
    def electric_dipole(self):
        return [-np.asarray(comp) for comp in self.mints.ao_dipole()]


class Psi4EriBuilder(EriBuilder):
    def __init__(self, wfn, n_orbs, n_orbs_alpha, n_alpha, n_beta):
        self.wfn = wfn
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        super().__init__(n_orbs, n_orbs_alpha, n_alpha, n_beta)

    @property
    def coeffs_occ_alpha(self):
        return self.wfn.Ca_subset("AO", "OCC")

    @property
    def coeffs_virt_alpha(self):
        return self.wfn.Ca_subset("AO", "VIR")

    def compute_mo_eri(self, block, coeffs, use_cache=True):
        if block in self.eri_cache and use_cache:
            return self.eri_cache[block]
        eri = np.asarray(self.mints.mo_eri(*coeffs))
        self.eri_cache[block] = eri
        return eri


class Psi4HFProvider(HartreeFockProvider):
    """
        This implementation is only valid for RHF
    """
    def __init__(self, wfn):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()

        if not isinstance(wfn, psi4.core.RHF):
            raise TypeError("Only restricted references (RHF) are supported.")

        self.wfn = wfn
        self.eri_ffff = None
        self.eri_builder = Psi4EriBuilder(self.wfn, self.n_orbs, self.wfn.nmo(),
                                          wfn.nalpha(), wfn.nbeta())
        self.operator_integral_provider = Psi4OperatorIntegralProvider(self.wfn)

    def get_backend(self):
        return "psi4"

    def get_conv_tol(self):
        conv_tol = psi4.core.get_option("SCF", "E_CONVERGENCE")
        # RMS value of the orbital gradient
        conv_tol_grad = psi4.core.get_option("SCF", "D_CONVERGENCE")
        threshold = max(10 * conv_tol, conv_tol_grad)
        return threshold

    def get_restricted(self):
        return True  # TODO Hard-coded for now.

    def get_energy_scf(self):
        return self.wfn.energy()

    def get_spin_multiplicity(self):
        return self.wfn.molecule().multiplicity()

    def get_n_orbs_alpha(self):
        return self.wfn.nmo()

    def get_n_bas(self):
        return self.wfn.basisset().nbf()

    def get_nuclear_multipole(self, order):
        molecule = self.wfn.molecule()
        if order == 0:
            # The function interface needs to be a np.array on return
            return np.array([sum(molecule.charge(i)
                                 for i in range(molecule.natom()))])
        elif order == 1:
            dip_nuclear = molecule.nuclear_dipole()
            return np.array([dip_nuclear[0], dip_nuclear[1], dip_nuclear[2]])
        else:
            raise NotImplementedError("get_nuclear_multipole with order > 1")

    def fill_orbcoeff_fb(self, out):
        mo_coeff_a = np.asarray(self.wfn.Ca())
        mo_coeff = (mo_coeff_a, mo_coeff_a)
        out[:] = np.transpose(
            np.hstack((mo_coeff[0], mo_coeff[1]))
        )

    def fill_occupation_f(self, out):
        out[:] = np.hstack((
            np.asarray(self.wfn.occupation_a()),
            np.asarray(self.wfn.occupation_b())
        ))

    def fill_orben_f(self, out):
        orben_a = np.asarray(self.wfn.epsilon_a())
        orben_b = np.asarray(self.wfn.epsilon_b())
        out[:] = np.hstack((orben_a, orben_b))

    def fill_fock_ff(self, slices, out):
        diagonal = np.empty(self.n_orbs)
        self.fill_orben_f(diagonal)
        out[:] = np.diag(diagonal)[slices]

    # TODO: obsolete code, just used for testing
    def fill_eri_ffff(self, slices, out):
        if self.eri_ffff is None:
            self.eri_ffff = self.eri_builder.build_full_eri_ffff()
        out[:] = self.eri_ffff[slices]

    def fill_eri_phys_asym_ffff(self, slices, out):
        self.eri_builder.fill_slice(slices, out)

    def has_eri_phys_asym_ffff(self):
        return True

    def flush_cache(self):
        self.eri_ffff = None
        self.eri_cache = None


def import_scf(scfdrv):
    if not isinstance(scfdrv, psi4.core.HF):
        raise TypeError("Unsupported type for backends.psi4.import_scf.")

    if not isinstance(scfdrv, psi4.core.RHF):
        raise TypeError("Only restricted references (RHF) are supported.")

    # TODO: Psi4 throws an exception if SCF is not converged
    # and there is, to the best of my knowledge, no `is_converged` property
    # of the psi4 wavefunction
    # if not scfdrv.is_converged:
    #     raise ValueError("Cannot start an adc calculation on top of an SCF, "
    #                      "which is not converged.")

    provider = Psi4HFProvider(scfdrv)
    return provider


basissets = {
    "sto3g": "sto-3g",
    "def2tzvp": "def2-tzvp",
    "ccpvdz": "cc-pvdz",
}


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-12,
           conv_tol_grad=1e-8, max_iter=150):
    mol = psi4.geometry("""
        {charge} {multiplicity}
        {xyz}
        symmetry c1
        units au
        no_reorient
        no_com
        """.format(xyz=xyz, charge=charge, multiplicity=multiplicity))
    psi4.core.be_quiet()
    reference = "RHF"
    if multiplicity != 1:
        reference = "UHF"
    psi4.set_options({'basis': basissets.get(basis, basis),
                      'scf_type': 'pk',
                      'e_convergence': conv_tol,
                      'd_convergence': conv_tol_grad,
                      'maxiter': max_iter,
                      'reference': reference})
    _, wfn = psi4.energy('SCF', return_wfn=True, molecule=mol)
    psi4.core.clean()
    return wfn
