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
import warnings
import numpy as np

from .misc import cached_property
from .timings import Timer, timed_member_call
from .AdcMethod import AdcMethod
from .FormatIndex import (FormatIndexAdcc, FormatIndexBase,
                          FormatIndexHfProvider, FormatIndexHomoLumo)
from .visualisation import ExcitationSpectrum
from .state_densities import compute_gs2state_optdm, compute_state_diffdm
from .OneParticleOperator import product_trace
from .FormatDominantElements import FormatDominantElements

from adcc import dot
from matplotlib import pyplot as plt

from scipy import constants
from .solver.SolverStateBase import EigenSolverStateBase


class FormatExcitationVector:
    def __init__(self, matrix, tolerance=0.01, index_format=None):
        """
        Set up a formatter class for formatting excitation vectors.

        Parameters
        ----------
        tolerance : float, optional
            Minimal absolute value of the excitation amplitudes considered

        index_format : NoneType or str or FormatIndexBase, optional
            Formatter to use for displaying tensor indices.
            Valid are ``"adcc"`` to keep the adcc-internal indexing,
            ``"hf"`` to select the HFProvider indexing, ``"homolumo"``
            to index relative on the HOMO / LUMO / HOCO orbitals.
            If ``None`` an automatic selection will be made.
        """
        self.matrix = matrix
        refstate = matrix.reference_state
        if index_format is None:
            closed_shell = refstate.n_alpha == refstate.n_beta
            if closed_shell and refstate.is_aufbau_occupation:
                index_format = "homolumo"
            else:
                index_format = "hf"
        if index_format in ["adcc"]:
            index_format = FormatIndexAdcc(refstate)
        elif index_format in ["hf"]:
            index_format = FormatIndexHfProvider(refstate)
        elif index_format in ["homolumo"]:
            index_format = FormatIndexHomoLumo(refstate)
        elif not isinstance(index_format, FormatIndexBase):
            raise ValueError("Unsupported value for index_format: "
                             + str(index_format))
        self.tensor_format = FormatDominantElements(matrix.mospaces, tolerance,
                                                    index_format)
        self.index_format = self.tensor_format.index_format
        self.value_format = "{:+8.3g}"  # Formatting used for the values

    def optimise_formatting(self, vectors):
        if not isinstance(vectors, list):
            return self.optimise_formatting([vectors])
        for vector in vectors:
            for block in self.matrix.blocks:
                spaces = self.matrix.block_spaces(block)
                self.tensor_format.optimise_formatting((spaces, vector[block]))

    @property
    def linewidth(self):
        """
        The width of an amplitude line if a tensor is formatted with this class
        """
        # TODO This assumes a PP ADC matrix
        if self.matrix.blocks == ["s"]:
            nblk = 2
        elif self.matrix.blocks == ["s", "d"]:
            nblk = 4
        else:
            raise NotImplementedError("Unknown ADC matrix structure")
        width_indices = nblk * (self.index_format.max_n_characters + 1) + 2
        width_spins = nblk + 2
        width_value = len(self.value_format.format(0))
        return width_indices + width_spins + width_value + 5

    def format(self, vector):
        idxgap = self.index_format.max_n_characters * " "
        # TODO This assumes a PP ADC matrix
        if self.matrix.blocks == ["s"]:
            formats = {"ov": "{0} -> {1}  {2}->{3}", }
        elif self.matrix.blocks == ["s", "d"]:
            formats = {
                "ov":   "{0} " + idxgap + " -> {1} " + idxgap + "  {2} ->{3} ",
                "oovv": "{0} {1} -> {2} {3}  {4}{5}->{6}{7}",
            }
        else:
            raise NotImplementedError("Unknown ADC matrix structure")

        ret = []
        for block in self.matrix.blocks:
            # Strip numbers for the lookup into formats above
            spaces = self.matrix.block_spaces(block)
            stripped = "".join(c for c in "".join(spaces) if c.isalpha())

            formatted = self.tensor_format.format_as_list(spaces, vector[block])
            for indices, spins, value in formatted:
                ret.append(formats[stripped].format(*indices, *spins)
                           + "   " + self.value_format.format(value))
        return "\n".join(ret)


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
        for key, obj in self._timed_objects:
            ret.attach(obj.timer, subtree=key)
        ret.attach(self._property_timer, subtree="properties")
        ret.time_construction = self.reference_state.timer.time_construction
        return ret

    @property
    def property_method(self):
        """The method used to evaluate ADC properties"""
        return self.__property_method

    @cached_property
    @timed_member_call(timer="_property_timer")
    def transition_dms(self):
        """List of transition density matrices of all computed states"""
        return [compute_gs2state_optdm(self.property_method, self.ground_state,
                                       evec, self.matrix.intermediates)
                for evec in self.excitation_vectors]

    @cached_property
    @timed_member_call(timer="_property_timer")
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
    @timed_member_call(timer="_property_timer")
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
    @timed_member_call(timer="_property_timer")
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

    def describe(self, oscillator_strengths=True, state_dipole_moments=False,
                 transition_dipole_moments=False, block_norms=True):
        """
        Return a string providing a human-readable description of the class

        Parameters
        ----------
        oscillator_strengths : bool optional
            Show oscillator strengths, by default ``True``.

        state_dipole_moments : bool, optional
            Show state dipole moments, by default ``False``.

        transition_dipole_moments : bool, optional
            Show state dipole moments, by default ``False``.

        block_norms : bool, optional
            Show the norms of the (1p1h, 2p2h, ...) blocks of the excited states,
            by default ``True``.
        """
        # TODO This function is quite horrible and definitely needs some
        #      refactoring

        eV = constants.value("Hartree energy in eV")
        has_dipole = "electric_dipole" in self.operators.available

        # Build information about the optional columns
        opt_thead = ""
        opt_body = ""
        opt = {}
        if has_dipole and transition_dipole_moments:
            opt_body += " {tdmx:8.4f} {tdmy:8.4f} {tdmz:8.4f}"
            opt_thead += "  transition dipole moment "
            opt["tdmx"] = lambda i, vec: self.transition_dipole_moments[i][0]
            opt["tdmy"] = lambda i, vec: self.transition_dipole_moments[i][1]
            opt["tdmz"] = lambda i, vec: self.transition_dipole_moments[i][2]
        if has_dipole and oscillator_strengths:
            opt_body += "{osc:8.4f} "
            opt_thead += " osc str "
            opt["osc"] = lambda i, vec: self.oscillator_strengths[i]
        if "s" in self.matrix.blocks and block_norms:
            opt_body += "{v1:9.4g} "
            opt_thead += "   |v1|^2 "
            opt["v1"] = lambda i, vec: dot(vec["s"], vec["s"])
        if "d" in self.matrix.blocks and block_norms:
            opt_body += "{v2:9.4g} "
            opt_thead += "   |v2|^2 "
            opt["v2"] = lambda i, vec: dot(vec["d"], vec["d"])
        if has_dipole and state_dipole_moments:
            opt_body += " {dmx:8.4f} {dmy:8.4f} {dmz:8.4f}"
            opt_thead += "     state dipole moment   "
            opt["dmx"] = lambda i, vec: self.state_dipole_moments[i][0]
            opt["dmy"] = lambda i, vec: self.state_dipole_moments[i][1]
            opt["dmz"] = lambda i, vec: self.state_dipole_moments[i][2]

        # Heading of the table
        kind = ""
        if hasattr(self, "kind") and self.kind \
           and self.kind not in [" ", ""]:
            kind = self.kind + " "

        spin_change = ""
        if kind.strip() == "spin_flip" and hasattr(self, "spin_change") and \
                self.spin_change is not None and self.spin_change != -1:
            spin_change = "(ΔMS={:+2d})".format(self.spin_change)

        conv = ""
        if hasattr(self, "converged"):
            conv = "NOT CONVERGED"
            if self.converged:
                conv = "converged"

        propname = ""
        if self.property_method != self.method:
            propname = " (" + self.property_method.name + ")"

        head = "| {0:18s}  {1:>" + str(11 + len(opt_thead)) + "s} |\n"
        delim = ",  " if kind else ""
        headtext = head.format(self.method.name + propname,
                               kind + spin_change + delim + conv)

        extra = len(headtext) - len(opt_thead) - 36
        text = ""
        separator = "+" + 33 * "-" + extra * "-" + len(opt_thead) * "-" + "+"
        text += separator + "\n"
        text += headtext
        text += separator + "\n"

        if extra > 0:
            opt["space"] = lambda i, vec: ""
            opt_body += "{space:" + str(extra) + "s}"
            opt_thead += (extra * " ")

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
                                ev=self.excitation_energies[i] * eV, **fields)
        text += separator + "\n"
        return text

    def _repr_pretty_(self, pp, cycle):
        if cycle:
            pp.text("ExcitedStates(...)")
        else:
            pp.text(self.describe())

    def describe_amplitudes(self, tolerance=0.01, index_format=None):
        """
        Return a string describing the dominant amplitudes of each
        excitation vector in human-readable form. The ``kwargs``
        are for :py:class:`FormatExcitationVector`.

        Parameters
        ----------
        tolerance : float, optional
            Minimal absolute value of the excitation amplitudes considered.

        index_format : NoneType or str or FormatIndexBase, optional
            Formatter to use for displaying tensor indices.
            Valid are ``"adcc"`` to keep the adcc-internal indexing,
            ``"hf"`` to select the HFProvider indexing, ``"homolumo"``
            to index relative on the HOMO / LUMO / HOCO orbitals.
            If ``None`` an automatic selection will be made.
        """
        eV = constants.value("Hartree energy in eV")
        vector_format = FormatExcitationVector(self.matrix, tolerance=tolerance,
                                               index_format=index_format)

        # Optimise the formatting by pre-inspecting all tensors
        for tensor in self.excitation_vectors:
            vector_format.optimise_formatting(tensor)

        # Determine width of a line
        lw = 2 + vector_format.linewidth
        separator = "+" + lw * "-" + "+\n"

        ret = separator
        for i, vec in enumerate(self.excitation_vectors):
            ene = self.excitation_energies[i]
            eev = ene * eV
            head = f"State {i:3d} , {ene:13.7g} au"
            if lw > 47:
                head += f", {eev:13.7} eV"
            ret += "| " + head + (lw - len(head) - 2) * " " + " |\n"
            ret += separator
            formatted = vector_format.format(vec).replace("\n", " |\n| ")
            ret += "| " + formatted + " |\n"
            if i != len(self.excitation_vectors) - 1:
                ret += "\n"
                ret += separator
        return ret[:-1]
