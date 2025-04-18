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
from scipy import constants
import warnings

from .ElectronicStates import ElectronicStates, _timer_name
from .Excitation import Excitation
from .misc import cached_member_function
from .OneParticleOperator import OneParticleOperator, product_trace


class ElectronicTransition(ElectronicStates):
    # The child classes S2S and ExcitedStates currently
    # both share the Excitation class for the state view.
    # This might change in the future once e.g. ExcitedStates
    # specific methods are implemented.
    _state_view_cls = Excitation

    @property
    def excitations(self) -> list[Excitation]:
        """
        Provides a list of Excitations, i.e., a view to all individual
        excitations and their properties.
        """
        return [self._state_view(i) for i in range(self.size)]

    @property
    def transition_dm(self) -> list[OneParticleOperator]:
        """List of transition density matrices of all computed states"""
        return [self._transition_dm(i) for i in range(self.size)]

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
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

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
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
    def transition_dipole_moment_velocity(self) -> np.ndarray:
        """
        Array of transition dipole moments in the velocity gauge of all
        computed states
        """
        return np.array([
            self._transition_dipole_moment_velocity(i) for i in range(self.size)
        ])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _transition_dipole_moment_velocity(self, state_n: int) -> np.ndarray:
        """
        Computes the transition dipole moments in the velocity gauge for a
        single state
        """
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition velocity dipole moments "
                          "are known to be faulty in some cases.")
        dipole_integrals = self.operators.electric_dipole_velocity
        tdm = self._transition_dm(state_n)
        return np.array(
            [product_trace(comp, tdm) for comp in dipole_integrals]
        )

    def transition_magnetic_dipole_moment(self,
                                          gauge_origin="origin") -> np.ndarray:
        """Array of transition magnetic dipole moments of all computed states"""
        return np.array([
            self._transition_magnetic_dipole_moment(state_n=i,
                                                    gauge_origin=gauge_origin)
            for i in range(self.size)
        ])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _transition_magnetic_dipole_moment(self, state_n: int,
                                           gauge_origin="origin") -> np.ndarray:
        """
        Computes the transition magnetic dipole moments for a single state
        """
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition magnetic dipole moments "
                          "are known to be faulty in some cases.")
        mag_dipole_integrals = self.operators.magnetic_dipole(gauge_origin)
        tdm = self._transition_dm(state_n)
        return np.array([
            product_trace(comp, tdm) for comp in mag_dipole_integrals
        ])

    def transition_quadrupole_moment(self, gauge_origin="origin") -> np.ndarray:
        """Array of transition quadrupole moments of all computed states"""
        return np.array([
            self._transition_quadrupole_moment(state_n=i, gauge_origin=gauge_origin)
            for i in range(self.size)
        ])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _transition_quadrupole_moment(self, state_n: int, gauge_origin="origin"):
        """Computes the transition quadrupole moments for a single state"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition quadrupole moments are known to be "
                          "faulty in some cases.")
        quadrupole_integrals = self.operators.electric_quadrupole(gauge_origin)
        tdm = self._transition_dm(state_n)
        return np.array([
            [product_trace(q, tdm) for q in quad] for quad in quadrupole_integrals
        ])

    def transition_quadrupole_moment_velocity(self,
                                              gauge_origin="origin") -> np.ndarray:
        """Array of transition quadrupole moments of all computed states"""
        return np.array([
            self._transition_quadrupole_moment_velocity(state_n=i,
                                                        gauge_origin=gauge_origin)
            for i in range(self.size)
        ])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _transition_quadrupole_moment_velocity(self, state_n: int,
                                               gauge_origin="origin") -> np.ndarray:
        """Compute the transition quadrupole moments for a single state"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition velocity quadrupole moments are known "
                          "to be faulty in some cases.")
        quadrupole_integrals = (
            self.operators.electric_quadrupole_velocity(gauge_origin)
        )
        tdm = self._transition_dm(state_n)
        return np.array([
            [product_trace(q, tdm) for q in quad] for quad in quadrupole_integrals
        ])

    @property
    def oscillator_strength(self) -> np.ndarray:
        """Array of oscillator strengths of all computed states"""
        return np.array([self._oscillator_strength(i) for i in range(self.size)])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _oscillator_strength(self, state_n: int) -> np.float64:
        """Computes the oscillator strengths for a single state"""
        tdm = self._transition_dipole_moment(state_n)
        ev = self.excitation_energy[state_n]
        return 2. / 3. * np.linalg.norm(tdm)**2 * np.abs(ev)

    @property
    def oscillator_strength_velocity(self) -> np.ndarray:
        """Array of oscillator strengths in velocity gauge of all computed states"""
        return np.array([
            self._oscillator_strength_velocity(i) for i in range(self.size)
        ])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _oscillator_strength_velocity(self, state_n: int) -> np.float64:
        """Computes the oscillator strength in velocity gauge for a single state"""
        tdm = self._transition_dipole_moment_velocity(state_n)
        ev = self.excitation_energy[state_n]
        return 2. / 3. * np.linalg.norm(tdm)**2 / np.abs(ev)

    @property
    def rotatory_strength(self) -> np.ndarray:
        """
        Array of rotatory strengths (in velocity gauge) of all computed states.
        This property is gauge-origin invariant, thus, it is not possible to
        select a gauge origin.
        """
        return np.array([
            self._rotatory_strength(i) for i in range(self.size)
        ])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _rotatory_strength(self, state_n: int) -> np.float64:
        """
        Computes the rotatory strength (in velocity gauge) for a single state.
        This property is gauge-origin invariant, thus, it is not possible to
        select a gauge origin.
        """
        tdm = self._transition_dipole_moment_velocity(state_n)
        magmom = self._transition_magnetic_dipole_moment(
            state_n=state_n, gauge_origin="origin"
        )
        ee = self.excitation_energy[state_n]
        return np.dot(tdm, magmom) / ee

    def rotatory_strength_length(self, gauge_origin="origin") -> np.ndarray:
        """Array of rotatory strengths in length gauge of all computed states"""
        return np.array([
            self._rotatory_strength_length(state_n=i, gauge_origin=gauge_origin)
            for i in range(self.size)
        ])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _rotatory_strength_length(self, state_n: int,
                                  gauge_origin="origin") -> np.float64:
        """Computes the rotatory strength in length gauge for a single state"""
        tdm = self._transition_dipole_moment(state_n)
        magmom = self._transition_magnetic_dipole_moment(state_n=state_n,
                                                         gauge_origin=gauge_origin)
        return -1.0 * np.dot(tdm, magmom)

    @property
    def cross_section(self) -> np.ndarray:
        """Array of one-photon absorption cross sections of all computed states"""
        return np.array([self._cross_section(i) for i in range(self.size)])

    def _cross_section(self, state_n: int) -> np.float64:
        """Computes the one-photon absorption cross sections for a single state"""
        # TODO Source?
        fine_structure = constants.fine_structure
        fine_structure_au = 1 / fine_structure
        prefac = 2.0 * np.pi ** 2 / fine_structure_au
        return prefac * self._oscillator_strength(state_n)

    def plot_spectrum(self, broadening="lorentzian", xaxis="eV",
                      yaxis="cross_section", width=0.01,
                      width_unit: str = "au", **kwargs):
        """
        One-shot plotting function for the spectrum generated by all states
        known to this class.

        Makes use of the :class:`adcc.visualisation.Spectrum` class
        in order to generate and format the spectrum to be plotted, using
        many sensible defaults.

        Parameters
        ----------
        broadening : str or None or callable, optional
            The broadening type to used for the computed excitations.
            A value of None disables broadening any other value is passed
            straight to
            :func:`adcc.visualisation.Spectrum.broaden_lines`.
        xaxis : str
            Energy unit to be used on the x-Axis. Options:
            ["eV", "au", "nm", "cm-1"]
        yaxis : str
            Quantity to plot on the y-Axis. Options are "cross_section",
            "osc_strength", "dipole" (plots norm of transition dipole),
            "rotational_strength" (ECD spectrum with rotational strength)
        width : float, optional
            Gaussian broadening standard deviation or Lorentzian broadening
            gamma parameter. The value should be given in atomic units
            and will be converted to the unit of the energy axis.
        width_unit: str, optional
            The unit the width is given in. All xaxis options except "nm" are
            possible.
        """
        if yaxis in ["osc", "osc_strength", "oscillator_strength", "f"]:
            yvalues = self.oscillator_strength
            ylabel = "Oscillator strengths (au)"
        elif yaxis in ["dipole", "dipole_norm", "μ"]:
            yvalues = np.linalg.norm(self.transition_dipole_moment, axis=1)
            ylabel = "Modulus of transition dipole (au)"
        elif yaxis in ["cross_section", "σ"]:
            yvalues = self.cross_section
            ylabel = "Cross section (au)"
        elif yaxis in ["rot", "rotational_strength", "rotatory_strength"]:
            yvalues = self.rotatory_strength
            ylabel = "Rotatory strength (au)"
        else:
            raise ValueError(f"Unknown yaxis specifier: {yaxis}")

        return self._plot_spectrum(
            yvalues=yvalues, ylabel=ylabel, xaxis=xaxis, broadening=broadening,
            width=width, width_unit=width_unit, **kwargs
        )
