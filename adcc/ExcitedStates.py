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

from .misc import cached_property
from .AdcMethod import AdcMethod
from .visualisation import ExcitationSpectrum
from .state_densities import compute_gs2state_optdm, compute_state_diffdm
from .OneParticleOperator import product_trace

import scipy.constants

from scipy import constants
from .solver.SolverStateBase import EigenSolverStateBase


# TODO This class needs docstrings (also for some of the attributes -> make them properties)
class ExcitedStates:
    def __init__(self, data, method=None, property_method=None):
        # property_method=None auto-selects (ADC(2) for ADC(3))

        self.matrix = data.matrix
        self._solver_timer = data.timer
        self.ground_state = self.matrix.ground_state
        self.reference_state = self.matrix.ground_state.reference_state
        self.operators = self.reference_state.operators

        # Copy some optional attributes
        for optattr in ["converged", "spin_change", "kind"]:
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

    @property
    def timer(self):
        # Collect all timer data we have from all the classes
        # right here and return it.
        return self._solver_timer

    @property
    def property_method(self):
        """The method used to evaluate ADC properties"""
        return self.__property_method

    @cached_property
    def transition_dms(self):
        return [compute_gs2state_optdm(self.property_method, self.ground_state,
                                       evec, self.matrix.intermediates)
                for evec in self.excitation_vectors]

    @cached_property
    def transition_dipole_moments(self):
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
        return 2. / 3. * np.array([
            np.linalg.norm(tdm)**2 * np.abs(ev)
            for tdm, ev in zip(self.transition_dipole_moments,
                               self.excitation_energies)
        ])

    @cached_property
    def state_diffdms(self):
        return [compute_state_diffdm(self.property_method, self.ground_state,
                                     evec, self.matrix.intermediates)
                for evec in self.excitation_vectors]

    @property
    def state_dms(self):
        mp_density = self.ground_state.density(self.property_method.level)
        return [mp_density + diffdm for diffdm in self.state_diffdms]

    @cached_property
    def state_dipole_moments(self):
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

    def plot_spectrum(self, broadening="lorentzian", **kwargs):
        """
        broadening=None disables
        """
        eV = constants.value("Hartree energy in eV")
        sp = ExcitationSpectrum(self.excitation_energies * eV,
                                self.oscillator_strengths)
        plots = sp.plot("x", **kwargs)
        if broadening:
            sp_broad = sp.broaden_lines(shape=broadening)
            plots.extend(sp_broad.plot("-", color=plots[0].get_color(),
                                       **kwargs))
        return plots

    def describe(self):
        text = ""
        toeV = scipy.constants.value("Hartree energy in eV")

        if hasattr(self, "kind") and self.kind \
           and self.kind not in [" ", ""]:
            kind = self.kind + " "
        else:
            kind = ""

        if kind == "spin_flip" and hasattr(self, "spin_change") \
           and self.spin_change and self.spin_change != -1:
            spin_change = "(ΔMS={:+2d})".format(self.spin_change)
        else:
            spin_change = ""

        if self.converged:
            conv = "converged"
        else:
            conv = "NOT CONVERGED"

        text += "+" + 53 * "-" + "+\n"
        head = "| {0:15s}  {1:>34s} |\n"
        delim = ",  " if kind else ""
        text += head.format(self.method.name,
                            kind + spin_change + delim + conv)
        text += "+" + 53 * "-" + "+\n"

        # TODO Certain methods such as ADC(0), ADC(1) do not have
        #      a doubles part and it does not really make sense to
        #      display it here.

        # TODO Add dominant amplitudes
        body = "| {0:2d} {1:13.7g} {2:13.7g} {3:9.4g} {4:9.4g}  |\n"
        text += "|  #        excitation energy       |v1|^2    |v2|^2  |\n"
        text += "|          (au)           (eV)                        |\n"
        for i, vec in enumerate(self.excitation_vectors):
            v1_norm = vec["s"].dot(vec["s"])
            if "d" in vec.blocks:
                v2_norm = vec["d"].dot(vec["d"])
            else:
                v2_norm = 0
            text += body.format(i, self.excitation_energies[i],
                                self.excitation_energies[i] * toeV,
                                v1_norm, v2_norm)
        text += "+" + 53 * "-" + "+"
        return text

    def _repr_pretty_(self, pp, cycle):
        if cycle:
            pp.text("ExcitedStates(...)")
        else:
            pp.text(self.describe())
