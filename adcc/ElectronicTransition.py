#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
import warnings

from .ElectronicStates import ElectronicStates
from .misc import cached_member_function
from .OneParticleOperator import OneParticleOperator, product_trace


timer_name = "_property_timer"


class ElectronicTransition(ElectronicStates):
    @property
    def transition_dm(self) -> list[OneParticleOperator]:
        """List of transition density matrices of all computed states"""
        return [self._transition_dm(i) for i in range(self.size)]

    @cached_member_function(timer=timer_name, separate_timings_by_args=False)
    def _transition_dm(self, state_n: int) -> OneParticleOperator:
        """Computes the tansition density matrix for a single state"""
        evec = self.excitation_vector[state_n]
        return self._module.transition_dm(
            self.property_method, self.ground_state, evec, self.matrix.intermediates
        )

    @property
    def transition_dipole_moment(self) -> np.ndarray:
        """Array of transition dipole moments of all computed states"""
        return np.array([
            self._transition_dipole_moment(i) for i in range(self.size)
        ])

    @cached_member_function(timer=timer_name, separate_timings_by_args=False)
    def _transition_dipole_moment(self, state_n: int) -> np.ndarray:
        """Computes the transition dipole moment for a single state"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition dipole moments are known to be "
                          "faulty in some cases.")
        dipole_integrals = self.operators.electric_dipole
        tdm = self._transition_dm(state_n)
        return np.array(
            [product_trace(comp, tdm) for comp in dipole_integrals]
        )

    @property
    def oscillator_strength(self) -> np.ndarray:
        """Array of oscillator strengths of all computed states"""
        return np.array([self._oscillator_strength(i) for i in range(self.size)])

    @cached_member_function(timer=timer_name, separate_timings_by_args=False)
    def _oscillator_strength(self, state_n: int) -> np.float64:
        """Computes the oscillator strengths for a single state"""
        tdm = self._transition_dipole_moment(state_n)
        ev = self.excitation_energy[state_n]
        return 2. / 3. * np.linalg.norm(tdm)**2 * np.abs(ev)

    # @cached_property
    # @mark_excitation_property()
    # @timed_member_call(timer="_property_timer")
    # def transition_dipole_moment_velocity(self):
    #     """List of transition dipole moments in the
    #     velocity gauge of all computed states"""
    #     if self.property_method.level == 0:
    #         warnings.warn("ADC(0) transition velocity dipole moments "
    #                       "are known to be faulty in some cases.")
    #     dipole_integrals = self.operators.electric_dipole_velocity
    #     return np.array([
    #         [product_trace(comp, tdm) for comp in dipole_integrals]
    #         for tdm in self.transition_dm
    #     ])

    # @property
    # @mark_excitation_property()
    # @timed_member_call(timer="_property_timer")
    # def transition_magnetic_dipole_moment(self):
    #     """List of transition magnetic dipole moments of all computed states"""
    #     if self.property_method.level == 0:
    #         warnings.warn("ADC(0) transition magnetic dipole moments "
    #                       "are known to be faulty in some cases.")

    #     def g_origin_dep_trans_magdip_moment(gauge_origin="origin"):
    #         mag_dipole_integrals = self.operators.magnetic_dipole(gauge_origin)
    #         return np.array([
    #             [product_trace(comp, tdm) for comp in mag_dipole_integrals]
    #             for tdm in self.transition_dm
    #         ])
    #     return g_origin_dep_trans_magdip_moment

    # @property
    # @mark_excitation_property()
    # @timed_member_call(timer="_property_timer")
    # def transition_quadrupole_moment(self):
    #     """List of transition quadrupole moments of all computed states"""
    #     if self.property_method.level == 0:
    #         warnings.warn("ADC(0) transition quadrupole moments are known to be "
    #                       "faulty in some cases.")

    #     def g_origin_dep_trans_el_quad_moment(gauge_origin="origin"):
    #         quadrupole_integrals = self.operators.electric_quadrupole(gauge_origin)
    #         return np.array([[
    #             [product_trace(quad1, tdm) for quad1 in quad]
    #             for quad in quadrupole_integrals]
    #             for tdm in self.transition_dm
    #         ])
    #     return g_origin_dep_trans_el_quad_moment

    # @property
    # @mark_excitation_property()
    # @timed_member_call(timer="_property_timer")
    # def transition_quadrupole_moment_velocity(self):
    #     """List of transition quadrupole moments of all computed states"""
    #     if self.property_method.level == 0:
    #         warnings.warn("ADC(0) transition velocity quadrupole moments are known "
    #                       "to be faulty in some cases.")

    #     def g_origin_dep_trans_el_quad_vel_moment(gauge_origin="origin"):
    #         quadrupole_integrals = \
    #             self.operators.electric_quadrupole_velocity(gauge_origin)
    #         return np.array([[
    #             [product_trace(quad1, tdm) for quad1 in quad]
    #             for quad in quadrupole_integrals]
    #             for tdm in self.transition_dm
    #         ])
    #     return g_origin_dep_trans_el_quad_vel_moment

    # @cached_property
    # @mark_excitation_property()
    # def oscillator_strength_velocity(self):
    #     """List of oscillator strengths in velocity gauge of all computed states"""
    #     return 2. / 3. * np.array([
    #         np.linalg.norm(tdm)**2 / np.abs(ev)
    #         for tdm, ev in zip(self.transition_dipole_moment_velocity,
    #                            self.excitation_energy)
    #     ])

    # @cached_property
    # @mark_excitation_property()
    # def rotatory_strength(self):
    #     """List of rotatory strengths (in velocity gauge) of all computed states.
    #     This property is gauge-origin invariant, thus, it is not possible to
    #     select a gauge origin."""
    #     return np.array([
    #         np.dot(tdm, magmom) / ee
    #         for tdm, magmom, ee in zip(
    #             self.transition_dipole_moment_velocity,
    #             self.transition_magnetic_dipole_moment("origin"),
    #             self.excitation_energy)
    #     ])

    # @property
    # @mark_excitation_property()
    # def rotatory_strength_length(self):
    #     """List of rotatory strengths in length gauge of all computed states"""
    #     def g_origin_dep_rot_str_len(gauge_origin="origin"):
    #         return np.array([
    #             -1.0 * np.dot(tdm, magmom)
    #             for tdm, magmom in zip(
    #                 self.transition_dipole_moment,
    #                 self.transition_magnetic_dipole_moment(gauge_origin))
    #         ])
    #     return g_origin_dep_rot_str_len

    # @property
    # @mark_excitation_property()
    # def cross_section(self):
    #     """List of one-photon absorption cross sections of all computed states"""
    #     # TODO Source?
    #     fine_structure = constants.fine_structure
    #     fine_structure_au = 1 / fine_structure
    #     prefac = 2.0 * np.pi ** 2 / fine_structure_au
    #     return prefac * self.oscillator_strength

    # @requires_module("matplotlib")
    # def plot_spectrum(self, broadening="lorentzian", xaxis="eV",
    #                   yaxis="cross_section", width=0.01, **kwargs):
    #     """One-shot plotting function for the spectrum generated by all states
    #     known to this class.

    #     Makes use of the :class:`adcc.visualisation.ExcitationSpectrum` class
    #     in order to generate and format the spectrum to be plotted, using
    #     many sensible defaults.

    #     Parameters
    #     ----------
    #     broadening : str or None or callable, optional
    #         The broadening type to used for the computed excitations.
    #         A value of None disables broadening any other value is passed
    #         straight to
    #         :func:`adcc.visualisation.ExcitationSpectrum.broaden_lines`.
    #     xaxis : str
    #         Energy unit to be used on the x-Axis. Options:
    #         ["eV", "au", "nm", "cm-1"]
    #     yaxis : str
    #         Quantity to plot on the y-Axis. Options are "cross_section",
    #         "osc_strength", "dipole" (plots norm of transition dipole),
    #         "rotational_strength" (ECD spectrum with rotational strength)
    #     width : float, optional
    #         Gaussian broadening standard deviation or Lorentzian broadening
    #         gamma parameter. The value should be given in atomic units
    #         and will be converted to the unit of the energy axis.
    #     """
    #     from matplotlib import pyplot as plt
    #     if xaxis == "eV":
    #         eV = constants.value("Hartree energy in eV")
    #         energies = self.excitation_energy * eV
    #         width = width * eV
    #         xlabel = "Energy (eV)"
    #     elif xaxis in ["au", "Hartree", "a.u."]:
    #         energies = self.excitation_energy
    #         xlabel = "Energy (au)"
    #     elif xaxis == "nm":
    #         hc = constants.h * constants.c
    #         Eh = constants.value("Hartree energy")
    #         energies = hc / (self.excitation_energy * Eh) * 1e9
    #         xlabel = "Wavelength (nm)"
    #         if broadening is not None and not callable(broadening):
    #             raise ValueError("xaxis=nm and broadening enabled is "
    #                              "not supported.")
    #     elif xaxis in ["cm-1", "cm^-1", "cm^{-1}"]:
    #         towvn = constants.value("hartree-inverse meter relationship") / 100
    #         energies = self.excitation_energy * towvn
    #         width = width * towvn
    #         xlabel = "Wavenumbers (cm^{-1})"
    #     else:
    #         raise ValueError("Unknown xaxis specifier: {}".format(xaxis))

    #     if yaxis in ["osc", "osc_strength", "oscillator_strength", "f"]:
    #         absorption = self.oscillator_strength
    #         ylabel = "Oscillator strengths (au)"
    #     elif yaxis in ["dipole", "dipole_norm", "μ"]:
    #         absorption = np.linalg.norm(self.transition_dipole_moment, axis=1)
    #         ylabel = "Modulus of transition dipole (au)"
    #     elif yaxis in ["cross_section", "σ"]:
    #         absorption = self.cross_section
    #         ylabel = "Cross section (au)"
    #     elif yaxis in ["rot", "rotational_strength", "rotatory_strength"]:
    #         absorption = self.rotatory_strength
    #         ylabel = "Rotatory strength (au)"
    #     else:
    #         raise ValueError("Unknown yaxis specifier: {}".format(yaxis))

    #     sp = ExcitationSpectrum(energies, absorption)
    #     sp.xlabel = xlabel
    #     sp.ylabel = ylabel
    #     if not broadening:
    #         plots = sp.plot(style="discrete", **kwargs)
    #     else:
    #         kwdisc = kwargs.copy()
    #         kwdisc.pop("label", "")
    #         plots = sp.plot(style="discrete", **kwdisc)

    #         kwargs.pop("color", "")
    #         sp_broad = sp.broaden_lines(width, shape=broadening)
    #         plots.extend(sp_broad.plot(color=plots[0].get_color(),
    #                                    style="continuous", **kwargs))

    #     if xaxis in ["nm"]:
    #         # Invert x axis
    #         plt.xlim(plt.xlim()[::-1])
    #     return plots
