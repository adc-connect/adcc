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
import warnings
import numpy as np

from adcc import dot
from matplotlib import pyplot as plt

from .misc import cached_property
from .timings import Timer
from .AdcMethod import AdcMethod
from .visualisation import ExcitationSpectrum
from .state_densities import compute_gs2state_optdm, compute_state_diffdm
from .OneParticleOperator import product_trace
from .solver.SolverStateBase import EigenSolverStateBase

from scipy import constants


class ExcitedStates:
    def __init__(self, data, method=None, property_method=None):
        """Construct an ExcitedStates class from some data obtained
        from an interative solver.

        The class provides access to the results from an ADC calculation
        as well as derived properties. Properties are computed lazily
        on the fly as requested by the user.

        By default the ADC method is extracted from the data object
        and the property method in property_method is set equal to
        this method, except ADC(3) where property_method=="adc2".
        This can be overwritten using the parameters.

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

        if not hasattr(data, "timer"):
            self._solver_timer = Timer()
        else:
            self._solver_timer = data.timer

        # Copy some optional attributes
        for optattr in ["converged", "spin_change", "kind", "n_iter"]:
            if hasattr(data, optattr):
                setattr(self, optattr, getattr(data, optattr))

        if method is None:
            self.method = self.matrix.method
        elif not isinstance(method, AdcMethod):
            method = AdcMethod(method)
        if property_method is None:
            if self.method.level < 3:
                property_method = self.method
            else:
                # Auto-select ADC(2) properties for ADC(3) calc
                property_method = self.method.at_level(2)
        elif not isinstance(property_method, AdcMethod):
            property_method = AdcMethod(method)
        self.__property_method = property_method

        # Special stuff for special solvers
        if isinstance(data, EigenSolverStateBase):
            self.excitation_vectors = data.eigenvectors
            self.excitation_energies = data.eigenvalues
        else:
            if hasattr(data, "eigenvalues"):
                self.excitation_energies = data.eigenvalues
            if hasattr(data, "eigenvectors"):
                self.excitation_vectors = data.eigenvectors

    @property
    def timer(self):
        """Return a cumulative timer collecting timings from the calculation"""
        ret = Timer()
        ret.attach(self._solver_timer)
        ret.attach(self.reference_state.timer)
        ret.time_construction = self.reference_state.timer.time_construction
        return ret

    @property
    def property_method(self):
        """The method used to evaluate ADC properties"""
        return self.__property_method

    @cached_property
    def transition_dms(self):
        """List of transition density matrices of all computed states"""
        return [compute_gs2state_optdm(self.property_method, self.ground_state,
                                       evec, self.matrix.intermediates)
                for evec in self.excitation_vectors]

    @cached_property
    def transition_dipole_moments(self):
        """List of transition dipole moments of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition dipole moments are known to be "
                          "faulty in some cases.")
        dipole_integrals = self.operators.electric_dipole
        return np.array([
            [product_trace(comp, tdm) for comp in dipole_integrals]
            for tdm in self.transition_dms
        ])

    @cached_property
    def oscillator_strengths(self):
        """List of oscillator strengths of all computed states"""
        return 2. / 3. * np.array([
            np.linalg.norm(tdm)**2 * np.abs(ev)
            for tdm, ev in zip(self.transition_dipole_moments,
                               self.excitation_energies)
        ])

    @cached_property
    def state_diffdms(self):
        """List of difference density matrices of all computed states"""
        return [compute_state_diffdm(self.property_method, self.ground_state,
                                     evec, self.matrix.intermediates)
                for evec in self.excitation_vectors]

    @property
    def state_dms(self):
        """List of state density matrices of all computed states"""
        mp_density = self.ground_state.density(self.property_method.level)
        return [mp_density + diffdm for diffdm in self.state_diffdms]

    @cached_property
    def state_dipole_moments(self):
        """List of state dipole moments"""
        pmethod = self.property_method
        if pmethod.level == 0:
            gs_dip_moment = self.reference_state.dipole_moment
        else:
            gs_dip_moment = self.ground_state.dipole_moment(pmethod.level)

        dipole_integrals = self.operators.electric_dipole
        return gs_dip_moment - np.array([
            [product_trace(comp, ddm) for comp in dipole_integrals]
            for ddm in self.state_diffdms
        ])

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
            "osc_strength", "dipole" (plots norm of transition dipole).
        width : float, optional
            Gaussian broadening standard deviation or Lorentzian broadening
            gamma parameter. The value should be given in atomic units
            and will be converted to the unit of the energy axis.
        """
        if xaxis == "eV":
            eV = constants.value("Hartree energy in eV")
            energies = self.excitation_energies * eV
            width = width * eV
            xlabel = "Energy (eV)"
        elif xaxis in ["au", "Hartree", "a.u."]:
            energies = self.excitation_energies
            xlabel = "Energy (au)"
        elif xaxis == "nm":
            hc = constants.h * constants.c
            Eh = constants.value("Hartree energy")
            energies = hc / (self.excitation_energies * Eh) * 1e9
            xlabel = "Wavelength (nm)"
            if broadening is not None and not callable(broadening):
                raise ValueError("xaxis=nm and broadening enabled is "
                                 "not supported.")
        elif xaxis in ["cm-1", "cm^-1", "cm^{-1}"]:
            towvn = constants.value("hartree-inverse meter relationship") / 100
            energies = self.excitation_energies * towvn
            width = width * towvn
            xlabel = "Wavenumbers (cm^{-1})"
        else:
            raise ValueError("Unknown xaxis specifier: {}".format(xaxis))

        if yaxis in ["osc", "osc_strength", "oscillator_strength", "f"]:
            absorption = self.oscillator_strengths
            ylabel = "Oscillator strengths (au)"
        elif yaxis in ["dipole", "dipole_norm", "μ"]:
            absorption = np.linalg.norm(self.transition_dipole_moments, axis=1)
            ylabel = "Modulus of transition dipole (au)"
        elif yaxis in ["cross_section", "σ"]:
            # TODO Source?
            fine_structure = constants.fine_structure
            fine_structure_au = 1 / fine_structure
            prefac = 2.0 * np.pi**2 / fine_structure_au
            absorption = prefac * self.oscillator_strengths
            ylabel = "Cross section (au)"
        else:
            raise ValueError("Unknown yaxis specifier: {}".format(yaxis))

        sp = ExcitationSpectrum(energies, absorption)
        sp.xlabel = xlabel
        sp.ylabel = ylabel
        if not broadening:
            plots = sp.plot(style="discrete", **kwargs)
        else:
            kwdisc = kwargs.copy()
            kwdisc["label"] = ""
            plots = sp.plot(style="discrete", **kwdisc)

            sp_broad = sp.broaden_lines(width, shape=broadening)
            plots.extend(sp_broad.plot(color=plots[0].get_color(),
                                       style="continuous", **kwargs))

        if xaxis in ["nm"]:
            # Invert x axis
            plt.xlim(plt.xlim()[::-1])
        return plots

    def describe(self):
        """Return a string providing a human-readable description of
        the class"""
        eV = constants.value("Hartree energy in eV")

        # Build information about the optional columns
        opt_thead = ""
        opt_body = ""
        opt = {}
        if "electric_dipole" in self.operators.available:
            opt_body += "{osc:8.4f} "
            opt_thead += " osc str "
            opt["osc"] = lambda i, vec: self.oscillator_strengths[i]
        if "s" in self.matrix.blocks:
            opt_body += "{v1:9.4g} "
            opt_thead += "   |v1|^2 "
            opt["v1"] = lambda i, vec: dot(vec["s"], vec["s"])
        if "d" in self.matrix.blocks:
            opt_body += "{v2:9.4g} "
            opt_thead += "   |v2|^2 "
            opt["v2"] = lambda i, vec: dot(vec["d"], vec["d"])

        # Heading of the table
        kind = ""
        if hasattr(self, "kind") and self.kind \
           and self.kind not in [" ", ""]:
            kind = self.kind + " "

        spin_change = ""
        if kind == "spin_flip" and hasattr(self, "spin_change") \
           and self.spin_change and self.spin_change != -1:
            spin_change = "(ΔMS={:+2d})".format(self.spin_change)

        conv = ""
        if hasattr(self, "converged"):
            conv = "NOT CONVERGED"
            if self.converged:
                conv = "converged"

        propname = ""
        if self.property_method != self.method:
            propname = " (" + self.property_method.name + ")"

        text = ""
        separator = "+" + 33 * "-" + len(opt_thead) * "-" + "+"
        text += separator + "\n"
        head = "| {0:18s}  {1:>" + str(11 + len(opt_thead)) + "s} |\n"
        delim = ",  " if kind else ""
        text += head.format(self.method.name + propname,
                            kind + spin_change + delim + conv)
        # TODO Print property method if it differs!
        text += separator + "\n"

        # Body of the table
        body = "| {i:2d} {ene:13.7g} {ev:13.7g} " + opt_body + " |\n"
        text += "|  #        excitation energy    " + opt_thead + " |\n"
        text += "|          (au)           (eV)   "
        text += len(opt_thead) * " " + " |\n"
        for i, vec in enumerate(self.excitation_vectors):
            fields = {}
            for k, compute in opt.items():
                fields[k] = compute(i, vec)
            text += body.format(i=i, ene=self.excitation_energies[i],
                                ev=self.excitation_energies[i] * eV,
                                **fields)
            # TODO Add dominant amplitudes
        text += separator + "\n"
        return text

    def _repr_pretty_(self, pp, cycle):
        if cycle:
            pp.text("ExcitedStates(...)")
        else:
            pp.text(self.describe())
