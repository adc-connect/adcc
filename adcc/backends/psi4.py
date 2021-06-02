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
import numpy as np

from libadcc import HartreeFockProvider
from adcc.misc import cached_property

import psi4

from .EriBuilder import EriBuilder
from ..exceptions import InvalidReference
from ..ExcitedStates import EnergyCorrection


class Psi4OperatorIntegralProvider:
    def __init__(self, wfn):
        self.wfn = wfn
        self.backend = "psi4"
        self.mints = psi4.core.MintsHelper(self.wfn)

    @cached_property
    def electric_dipole(self):
        return [-1.0 * np.asarray(comp) for comp in self.mints.ao_dipole()]

    @cached_property
    def magnetic_dipole(self):
        # TODO: Gauge origin?
        return [
            0.5 * np.asarray(comp)
            for comp in self.mints.ao_angular_momentum()
        ]

    @cached_property
    def nabla(self):
        return [-1.0 * np.asarray(comp) for comp in self.mints.ao_nabla()]

    @property
    def pe_induction_elec(self):
        if hasattr(self.wfn, "pe_state"):
            def pe_induction_elec_ao(dm):
                return self.wfn.pe_state.get_pe_contribution(
                    psi4.core.Matrix.from_array(dm.to_ndarray()),
                    elec_only=True
                )[1]
            return pe_induction_elec_ao


class Psi4EriBuilder(EriBuilder):
    def __init__(self, wfn, n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted):
        self.wfn = wfn
        self.mints = psi4.core.MintsHelper(self.wfn)
        super().__init__(n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted)

    @property
    def coefficients(self):
        return {
            "Oa": self.wfn.Ca_subset("AO", "OCC"),
            "Ob": self.wfn.Cb_subset("AO", "OCC"),
            "Va": self.wfn.Ca_subset("AO", "VIR"),
            "Vb": self.wfn.Cb_subset("AO", "VIR"),
        }

    def compute_mo_eri(self, blocks, spins):
        coeffs = tuple(self.coefficients[blocks[i] + spins[i]] for i in range(4))
        return np.asarray(self.mints.mo_eri(*coeffs))


class Psi4HFProvider(HartreeFockProvider):
    """
        This implementation is only valid
        if no orbital reordering is required.
    """
    def __init__(self, wfn):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.wfn = wfn
        self.eri_builder = Psi4EriBuilder(self.wfn, self.n_orbs, self.wfn.nmo(),
                                          wfn.nalpha(), wfn.nbeta(),
                                          self.restricted)
        self.operator_integral_provider = Psi4OperatorIntegralProvider(self.wfn)

    def pe_energy(self, dm, elec_only=True):
        density_psi = psi4.core.Matrix.from_array(dm.to_ndarray())
        e_pe, _ = self.wfn.pe_state.get_pe_contribution(density_psi,
                                                        elec_only=elec_only)
        return e_pe

    @property
    def excitation_energy_corrections(self):
        ret = []
        if self.environment == "pe":
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

    @property
    def environment(self):
        ret = None
        if hasattr(self.wfn, "pe_state"):
            ret = "pe"
        return ret

    def get_backend(self):
        return "psi4"

    def get_conv_tol(self):
        conv_tol = psi4.core.get_option("SCF", "E_CONVERGENCE")
        # RMS value of the orbital gradient
        conv_tol_grad = psi4.core.get_option("SCF", "D_CONVERGENCE")
        threshold = max(10 * conv_tol, conv_tol_grad)
        return threshold

    def get_restricted(self):
        return isinstance(self.wfn, (psi4.core.RHF, psi4.core.ROHF))

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
        mo_coeff_b = np.asarray(self.wfn.Cb())
        mo_coeff = (mo_coeff_a, mo_coeff_b)
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

    def fill_eri_ffff(self, slices, out):
        self.eri_builder.fill_slice_symm(slices, out)

    def fill_eri_phys_asym_ffff(self, slices, out):
        raise NotImplementedError("fill_eri_phys_asym_ffff not implemented.")

    def has_eri_phys_asym_ffff(self):
        return False

    def flush_cache(self):
        self.eri_builder.flush_cache()


def import_scf(wfn):
    if not isinstance(wfn, psi4.core.HF):
        raise InvalidReference(
            "Only psi4.core.HF and its subtypes are supported references in "
            "backends.psi4.import_scf. This indicates that you passed an "
            "unsupported SCF reference. Make sure you did a restricted or "
            "unrestricted HF calculation."
        )

    if not isinstance(wfn, (psi4.core.RHF, psi4.core.UHF)):
        raise InvalidReference("Right now only RHF and UHF references are "
                               "supported for Psi4.")

    # TODO This is not fully correct, because the core.Wavefunction object
    #      has an internal, but py-invisible Options structure, which contains
    #      the actual set of options ... theoretically they could differ
    scf_type = psi4.core.get_global_option('SCF_TYPE')
    # CD = Choleski, DF = density-fitting
    unsupported_scf_types = ["CD", "DISK_DF", "MEM_DF"]
    if scf_type in unsupported_scf_types:
        raise InvalidReference("Unsupported Psi4 SCF_TYPE, should not be one "
                               f"of {unsupported_scf_types}")

    if wfn.nirrep() > 1:
        raise InvalidReference("The passed Psi4 wave function object needs to "
                               "have exactly one irrep, i.e. be of C1 symmetry.")

    # Psi4 throws an exception if SCF is not converged, so there is no need
    # to assert that here.
    provider = Psi4HFProvider(wfn)
    return provider


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-11,
           conv_tol_grad=1e-8, max_iter=150, pe_options=None):
    basissets = {
        "sto3g": "sto-3g",
        "def2tzvp": "def2-tzvp",
        "ccpvdz": "cc-pvdz",
    }

    mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        symmetry c1
        units au
        no_reorient
        no_com
    """)

    psi4.core.be_quiet()
    psi4.set_options({
        'basis': basissets.get(basis, basis),
        'scf_type': 'pk',
        'e_convergence': conv_tol,
        'd_convergence': conv_tol_grad,
        'maxiter': max_iter,
        'reference': "RHF",
    })
    if pe_options:
        psi4.set_options({"pe": "true"})
        psi4.set_module_options("pe", {"potfile": pe_options["potfile"]})

    if multiplicity != 1:
        psi4.set_options({
            'reference': "UHF",
            'maxiter': max_iter + 500,
            'soscf': 'true'
        })

    _, wfn = psi4.energy('SCF', return_wfn=True, molecule=mol)
    psi4.core.clean()
    return wfn
