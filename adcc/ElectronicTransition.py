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
import warnings
import numpy as np

from .misc import cached_property, requires_module
from .timings import Timer, timed_member_call
from .visualisation import ExcitationSpectrum
from .OneParticleOperator import product_trace
from .AdcMethod import AdcMethod

from scipy import constants
from .Excitation import mark_excitation_property
from .solver.SolverStateBase import EigenSolverStateBase


class ElectronicTransition:
    def __init__(self, data, method=None, property_method=None):
        """Construct an ElectronicTransition class from some data obtained
        from an interative solver or another :class:`ElectronicTransition`
        object.

        Parameters
        ----------
        data
            Any kind of iterative solver state. Typically derived off
            a :class:`solver.EigenSolverStateBase`.
        method : str, optional
            Provide an explicit method parameter if data contains none.
        property_method : str, optional
            Provide an explicit method for property calculations to
            override the automatic selection.
        """
        self.matrix = data.matrix
        self.ground_state = self.matrix.ground_state
        self.reference_state = self.matrix.ground_state.reference_state
        self.operators = self.reference_state.operators

        # List of all the objects which have timers (do not yet collect
        # timers, since new times might be added implicitly at a later point)
        self._property_timer = Timer()
        self._timed_objects = [("", self.reference_state),
                               ("adcmatrix", self.matrix),
                               ("mp", self.ground_state),
                               ("intermediates", self.matrix.intermediates)]
        if hasattr(data, "timer"):
            datakey = getattr(data, "algorithm", data.__class__.__name__)
            self._timed_objects.append((datakey, data))

        # Copy some optional attributes
        for optattr in ["converged", "spin_change", "kind", "n_iter"]:
            if hasattr(data, optattr):
                setattr(self, optattr, getattr(data, optattr))

        self.method = getattr(data, "method", method)
        if self.method is None:
            self.method = self.matrix.method
        if not isinstance(self.method, AdcMethod):
            self.method = AdcMethod(self.method)
        if property_method is None:
            if self.method.level < 3:
                property_method = self.method
            else:
                # Auto-select ADC(2) properties for ADC(3) calc
                property_method = self.method.at_level(2)
        elif not isinstance(property_method, AdcMethod):
            property_method = AdcMethod(property_method)
        self._property_method = property_method

        # Special stuff for special solvers
        if isinstance(data, EigenSolverStateBase):
            self._excitation_vector = data.eigenvectors
            self._excitation_energy_uncorrected = data.eigenvalues
            self.residual_norm = data.residual_norms
        else:
            if hasattr(data, "eigenvalues"):
                self._excitation_energy_uncorrected = data.eigenvalues
            if hasattr(data, "eigenvectors"):
                self._excitation_vector = data.eigenvectors
            # if both excitation_energy and excitation_energy_uncorrected
            # are present, the latter one has priority
            if hasattr(data, "excitation_energy"):
                self._excitation_energy_uncorrected = \
                    data.excitation_energy.copy()
            if hasattr(data, "excitation_energy_uncorrected"):
                self._excitation_energy_uncorrected =\
                    data.excitation_energy_uncorrected.copy()
            if hasattr(data, "excitation_vector"):
                self._excitation_vector = data.excitation_vector

        # Collect all excitation energy corrections
        self._excitation_energy = self._excitation_energy_uncorrected.copy()

    def __len__(self):
        return self.size

    @property
    def size(self):
        return self._excitation_energy.size

    @property
    def timer(self):
        """Return a cumulative timer collecting timings from the calculation"""
        ret = Timer()
        for key, obj in self._timed_objects:
            ret.attach(obj.timer, subtree=key)
        ret.attach(self._property_timer, subtree="properties")
        ret.time_construction = self.reference_state.timer.time_construction
        return ret

    @property
    def property_method(self):
        """The method used to evaluate ADC properties"""
        return self._property_method

    @property
    @mark_excitation_property()
    def excitation_energy(self):
        """Excitation energies including all corrections in atomic units"""
        return self._excitation_energy

    @property
    @mark_excitation_property()
    def excitation_energy_uncorrected(self):
        """Excitation energies without any corrections in atomic units"""
        return self._excitation_energy_uncorrected

    @property
    @mark_excitation_property()
    def excitation_vector(self):
        """List of excitation vectors"""
        return self._excitation_vector

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_dipole_moment(self):
        """List of transition dipole moments of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition dipole moments are known to be "
                          "faulty in some cases.")
        dipole_integrals = self.operators.electric_dipole
        return np.array([
            [product_trace(comp, tdm) for comp in dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_dipole_moment_velocity(self):
        """List of transition dipole moments in the
        velocity gauge of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition velocity dipole moments "
                          "are known to be faulty in some cases.")
        dipole_integrals = self.operators.nabla
        return np.array([
            [product_trace(comp, tdm) for comp in dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_magnetic_dipole_moment(self):
        """List of transition magnetic dipole moments of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition magnetic dipole moments "
                          "are known to be faulty in some cases.")
        mag_dipole_integrals = self.operators.magnetic_dipole
        return np.array([
            [product_trace(comp, tdm) for comp in mag_dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    def oscillator_strength(self):
        """List of oscillator strengths of all computed states"""
        return 2. / 3. * np.array([
            np.linalg.norm(tdm)**2 * np.abs(ev)
            for tdm, ev in zip(self.transition_dipole_moment,
                               self.excitation_energy)
        ])

    @cached_property
    @mark_excitation_property()
    def oscillator_strength_velocity(self):
        """List of oscillator strengths in
        velocity gauge of all computed states"""
        return 2. / 3. * np.array([
            np.linalg.norm(tdm)**2 / np.abs(ev)
            for tdm, ev in zip(self.transition_dipole_moment_velocity,
                               self.excitation_energy)
        ])

    @cached_property
    @mark_excitation_property()
    def rotatory_strength(self):
        """List of rotatory strengths of all computed states"""
        return np.array([
            np.dot(tdm, magmom) / ee
            for tdm, magmom, ee in zip(self.transition_dipole_moment_velocity,
                                       self.transition_magnetic_dipole_moment,
                                       self.excitation_energy)
        ])

    @property
    @mark_excitation_property()
    def cross_section(self):
        """List of one-photon absorption cross sections of all computed states"""
        # TODO Source?
        fine_structure = constants.fine_structure
        fine_structure_au = 1 / fine_structure
        prefac = 2.0 * np.pi ** 2 / fine_structure_au
        return prefac * self.oscillator_strength

    @requires_module("matplotlib")
    def plot_spectrum(self, broadening="lorentzian", xaxis="eV",
                      yaxis="cross_section", width=0.01, **kwargs):
        """One-shot plotting function for the spectrum generated by all states
        known to this class.

        Makes use of the :class:`adcc.visualisation.ExcitationSpectrum` class
        in order to generate and format the spectrum to be plotted, using
        many sensible defaults.

        Parameters
        ----------
        broadening : str or None or callable, optional
            The broadening type to used for the computed excitations.
            A value of None disables broadening any other value is passed
            straight to
            :func:`adcc.visualisation.ExcitationSpectrum.broaden_lines`.
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
        """
        from matplotlib import pyplot as plt
        if xaxis == "eV":
            eV = constants.value("Hartree energy in eV")
            energies = self.excitation_energy * eV
            width = width * eV
            xlabel = "Energy (eV)"
        elif xaxis in ["au", "Hartree", "a.u."]:
            energies = self.excitation_energy
            xlabel = "Energy (au)"
        elif xaxis == "nm":
            hc = constants.h * constants.c
            Eh = constants.value("Hartree energy")
            energies = hc / (self.excitation_energy * Eh) * 1e9
            xlabel = "Wavelength (nm)"
            if broadening is not None and not callable(broadening):
                raise ValueError("xaxis=nm and broadening enabled is "
                                 "not supported.")
        elif xaxis in ["cm-1", "cm^-1", "cm^{-1}"]:
            towvn = constants.value("hartree-inverse meter relationship") / 100
            energies = self.excitation_energy * towvn
            width = width * towvn
            xlabel = "Wavenumbers (cm^{-1})"
        else:
            raise ValueError("Unknown xaxis specifier: {}".format(xaxis))

        if yaxis in ["osc", "osc_strength", "oscillator_strength", "f"]:
            absorption = self.oscillator_strength
            ylabel = "Oscillator strengths (au)"
        elif yaxis in ["dipole", "dipole_norm", "μ"]:
            absorption = np.linalg.norm(self.transition_dipole_moment, axis=1)
            ylabel = "Modulus of transition dipole (au)"
        elif yaxis in ["cross_section", "σ"]:
            absorption = self.cross_section
            ylabel = "Cross section (au)"
        elif yaxis in ["rot", "rotational_strength", "rotatory_strength"]:
            absorption = self.rotatory_strength
            ylabel = "Rotatory strength (au)"
        else:
            raise ValueError("Unknown yaxis specifier: {}".format(yaxis))

        sp = ExcitationSpectrum(energies, absorption)
        sp.xlabel = xlabel
        sp.ylabel = ylabel
        if not broadening:
            plots = sp.plot(style="discrete", **kwargs)
        else:
            kwdisc = kwargs.copy()
            kwdisc.pop("label", "")
            plots = sp.plot(style="discrete", **kwdisc)

            kwargs.pop("color", "")
            sp_broad = sp.broaden_lines(width, shape=broadening)
            plots.extend(sp_broad.plot(color=plots[0].get_color(),
                                       style="continuous", **kwargs))

        if xaxis in ["nm"]:
            # Invert x axis
            plt.xlim(plt.xlim()[::-1])
        return plots
