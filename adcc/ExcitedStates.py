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

from adcc import dot
from scipy import constants

from . import adc_pp
from .misc import cached_property, requires_module
from .timings import timed_member_call
from .Excitation import Excitation, mark_excitation_property
from .FormatIndex import (FormatIndexAdcc, FormatIndexBase,
                          FormatIndexHfProvider, FormatIndexHomoLumo)
from .OneParticleOperator import product_trace
from .ElectronicTransition import ElectronicTransition
from .FormatDominantElements import FormatDominantElements


class EnergyCorrection:
    def __init__(self, name, function):
        """A helper class to represent excitation energy
        corrections.

        Parameters
        ----------
        name : str
            descriptive name of the energy correction
        function : callable
            function that takes a :py:class:`Excitation`
            as single argument and returns the energy
            correction as a float
        """
        if not isinstance(name, str):
            raise TypeError("name needs to be a string.")
        if not callable(function):
            raise TypeError("function needs to be callable.")
        self.name = name
        self.function = function

    def __call__(self, excitation):
        assert isinstance(excitation, Excitation)
        return self.function(excitation)


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
            for block, spaces in self.matrix.axis_spaces.items():
                self.tensor_format.optimise_formatting((spaces, vector[block]))

    @property
    def linewidth(self):
        """
        The width of an amplitude line if a tensor is formatted with this class
        """
        # TODO This assumes a PP ADC matrix
        if self.matrix.axis_blocks == ["ph"]:
            nblk = 2
        elif self.matrix.axis_blocks == ["ph", "pphh"]:
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
        if self.matrix.axis_blocks == ["ph"]:
            formats = {"ov": "{0} -> {1}  {2}->{3}", }
        elif self.matrix.axis_blocks == ["ph", "pphh"]:
            formats = {
                "ov":   "{0} " + idxgap + " -> {1} " + idxgap + "  {2} ->{3} ",
                "oovv": "{0} {1} -> {2} {3}  {4}{5}->{6}{7}",
            }
        else:
            raise NotImplementedError("Unknown ADC matrix structure")

        ret = []
        for block, spaces in self.matrix.axis_spaces.items():
            # Strip numbers for the lookup into formats above
            stripped = "".join(c for c in "".join(spaces) if c.isalpha())

            formatted = self.tensor_format.format_as_list(spaces, vector[block])
            for indices, spins, value in formatted:
                ret.append(formats[stripped].format(*indices, *spins)
                           + "   " + self.value_format.format(value))
        return "\n".join(ret)


class ExcitedStates(ElectronicTransition):
    def __init__(self, data, method=None, property_method=None):
        """Construct an ExcitedStates class from some data obtained
        from an interative solver or another :class:`ExcitedStates`
        object.

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
            a :class:`solver.EigenSolverStateBase`. Can also be an
            :class:`ExcitedStates` object.
        method : str, optional
            Provide an explicit method parameter if data contains none.
        property_method : str, optional
            Provide an explicit method for property calculations to
            override the automatic selection.
        """
        super().__init__(data, method, property_method)
        self._excitation_energy_corrections = []
        # copy energy corrections if possible
        # and avoids re-computation of the corrections
        if hasattr(data, "_excitation_energy_corrections"):
            for eec in data._excitation_energy_corrections:
                correction_energy = getattr(data, eec.name)
                setattr(self, eec.name, correction_energy)
                self._excitation_energy += correction_energy
                self._excitation_energy_corrections.append(eec)

    def __add_energy_correction(self, correction):
        assert isinstance(correction, EnergyCorrection)
        if hasattr(self, correction.name):
            raise ValueError("ExcitedStates already has a correction"
                             f" with the name '{correction.name}'")
        correction_energy = np.array([correction(exci)
                                      for exci in self.excitations])
        setattr(self, correction.name, correction_energy)
        self._excitation_energy += correction_energy
        self._excitation_energy_corrections.append(correction)

    def __iadd__(self, other):
        if isinstance(other, EnergyCorrection):
            self.__add_energy_correction(other)
        elif isinstance(other, list):
            for k in other:
                self += k
        else:
            return NotImplemented
        return self

    def __add__(self, other):
        if not isinstance(other, (EnergyCorrection, list)):
            return NotImplemented
        ret = ExcitedStates(self, self.method, self.property_method)
        ret += other
        return ret

    @cached_property
    @mark_excitation_property(transform_to_ao=True)
    @timed_member_call(timer="_property_timer")
    def transition_dm(self):
        """List of transition density matrices of all computed states"""
        return [adc_pp.transition_dm(self.property_method, self.ground_state,
                                     evec, self.matrix.intermediates)
                for evec in self.excitation_vector]

    @cached_property
    @mark_excitation_property(transform_to_ao=True)
    @timed_member_call(timer="_property_timer")
    def state_diffdm(self):
        """List of difference density matrices of all computed states"""
        return [adc_pp.state_diffdm(self.property_method, self.ground_state,
                                    evec, self.matrix.intermediates)
                for evec in self.excitation_vector]

    @property
    @mark_excitation_property(transform_to_ao=True)
    def state_dm(self):
        """List of state density matrices of all computed states"""
        mp_density = self.ground_state.density(self.property_method.level)
        return [mp_density + diffdm for diffdm in self.state_diffdm]

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def state_dipole_moment(self):
        """List of state dipole moments"""
        pmethod = self.property_method
        if pmethod.level == 0:
            gs_dip_moment = self.reference_state.dipole_moment
        else:
            gs_dip_moment = self.ground_state.dipole_moment(pmethod.level)

        dipole_integrals = self.operators.electric_dipole
        return gs_dip_moment - np.array([
            [product_trace(comp, ddm) for comp in dipole_integrals]
            for ddm in self.state_diffdm
        ])

    def describe(self, oscillator_strengths=True, rotatory_strengths=False,
                 state_dipole_moments=False, transition_dipole_moments=False,
                 block_norms=True):
        """
        Return a string providing a human-readable description of the class

        Parameters
        ----------
        oscillator_strengths : bool optional
            Show oscillator strengths, by default ``True``.

        rotatory_strengths : bool optional
           Show rotatory strengths, by default ``False``.

        state_dipole_moments : bool, optional
            Show state dipole moments, by default ``False``.

        transition_dipole_moments : bool, optional
            Show state dipole moments, by default ``False``.

        block_norms : bool, optional
            Show the norms of the (1p1h, 2p2h, ...) blocks of the excited states,
            by default ``True``.
        """
        # TODO This function is quite horrible and definitely needs some
        #      refactoring, also it assumes ADC-PP everywhere

        eV = constants.value("Hartree energy in eV")
        has_dipole = "electric_dipole" in self.operators.available
        has_rotatory = all(op in self.operators.available
                           for op in ["magnetic_dipole", "nabla"])

        # Build information about the optional columns
        opt_thead = ""
        opt_body = ""
        opt = {}
        if has_dipole and transition_dipole_moments:
            opt_body += " {tdmx:8.4f} {tdmy:8.4f} {tdmz:8.4f}"
            opt_thead += "  transition dipole moment "
            opt["tdmx"] = lambda i, vec: self.transition_dipole_moment[i][0]
            opt["tdmy"] = lambda i, vec: self.transition_dipole_moment[i][1]
            opt["tdmz"] = lambda i, vec: self.transition_dipole_moment[i][2]
        if has_dipole and oscillator_strengths:
            opt_body += "{osc:8.4f} "
            opt_thead += " osc str "
            opt["osc"] = lambda i, vec: self.oscillator_strength[i]
        if has_rotatory and rotatory_strengths:
            opt_body += "{rot:8.4f} "
            opt_thead += " rot str "
            opt["rot"] = lambda i, vec: self.rotatory_strength[i]
        if "ph" in self.matrix.axis_blocks and block_norms:
            opt_body += "{v1:9.4g} "
            opt_thead += "   |v1|^2 "
            opt["v1"] = lambda i, vec: dot(vec.ph, vec.ph)
        if "pphh" in self.matrix.axis_blocks and block_norms:
            opt_body += "{v2:9.4g} "
            opt_thead += "   |v2|^2 "
            opt["v2"] = lambda i, vec: dot(vec.pphh, vec.pphh)
        if has_dipole and state_dipole_moments:
            opt_body += " {dmx:8.4f} {dmy:8.4f} {dmz:8.4f}"
            opt_thead += "     state dipole moment   "
            opt["dmx"] = lambda i, vec: self.state_dipole_moment[i][0]
            opt["dmy"] = lambda i, vec: self.state_dipole_moment[i][1]
            opt["dmz"] = lambda i, vec: self.state_dipole_moment[i][2]

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
        for i, vec in enumerate(self.excitation_vector):
            fields = {}
            for k, compute in opt.items():
                fields[k] = compute(i, vec)
            text += body.format(i=i, ene=self.excitation_energy[i],
                                ev=self.excitation_energy[i] * eV, **fields)
        text += separator + "\n"
        if len(self._excitation_energy_corrections):
            head_corr = "|  Excitation energy includes these corrections:"
            text += head_corr
            nspace = len(separator) - len(head_corr) - 1
            text += nspace * " " + "|\n"
            maxlen = len(separator) - 8
            for eec in self._excitation_energy_corrections:
                label = f"|    - {eec.name:{maxlen}.{maxlen}}|\n"
                text += label
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
        for tensor in self.excitation_vector:
            vector_format.optimise_formatting(tensor)

        # Determine width of a line
        lw = 2 + vector_format.linewidth
        separator = "+" + lw * "-" + "+\n"

        ret = separator
        for i, vec in enumerate(self.excitation_vector):
            ene = self.excitation_energy[i]
            eev = ene * eV
            head = f"State {i:3d} , {ene:13.7g} au"
            if lw > 47:
                head += f", {eev:13.7} eV"
            ret += "| " + head + (lw - len(head) - 2) * " " + " |\n"
            ret += separator
            formatted = vector_format.format(vec).replace("\n", " |\n| ")
            ret += "| " + formatted + " |\n"
            if i != len(self.excitation_vector) - 1:
                ret += "\n"
                ret += separator
        return ret[:-1]

    @requires_module("pandas")
    def to_dataframe(self):
        """
        Exports the ExcitedStates object as :class:`pandas.DataFrame`.
        Atomic units are used for all values.
        """
        import pandas as pd
        propkeys = self.excitation_property_keys
        propkeys.extend([k.name for k in self._excitation_energy_corrections])
        data = {
            "excitation": np.arange(0, self.size, dtype=int),
            "kind": np.tile(self.kind, self.size)
        }
        for key in propkeys:
            try:
                d = getattr(self, key)
            except NotImplementedError:
                # some properties are not available for every backend
                continue
            if not isinstance(d, np.ndarray):
                continue
            if not np.issubdtype(d.dtype, np.number):
                continue
            if d.ndim == 1:
                data[key] = d
            elif d.ndim == 2 and d.shape[1] == 3:
                for i, p in enumerate(["x", "y", "z"]):
                    data[f"{key}_{p}"] = d[:, i]
            elif d.ndim > 2:
                warnings.warn(f"Exporting NumPy array for property {key}"
                              f" with shape {d.shape} not supported.")
                continue
        df = pd.DataFrame(data=data)
        df.set_index("excitation")
        return df

    @property
    def excitation_property_keys(self):
        """
        Extracts the property keys which are marked
        as excitation property with :func:`mark_excitation_property`.
        """
        ret = []
        for key in dir(self):
            if key == "excitations":
                continue
            if "__" in key or key.startswith("_"):
                continue  # skip "private" fields
            if not hasattr(type(self), key):
                continue
            if not isinstance(getattr(type(self), key), property):
                continue
            fget = getattr(type(self), key).fget
            if hasattr(fget, "__excitation_property"):
                ret.append(key)
        return ret

    def to_qcvars(self, properties=False, recurse=False):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        name = self.method.name.upper()

        qcvars = {
            "EXCITATION KIND": self.kind.upper(),
            "NUMBER OF EXCITED STATES": len(self.excitation_energy),
            f"{name} ITERATIONS": self.n_iter,
            f"{name} EXCITATION ENERGIES": self.excitation_energy,
        }

        if properties:
            qcvars.update({
                # Transition properties
                f"{name} TRANSITION DIPOLES (LEN)": self.transition_dipole_moment,
                f"{name} TRANSITION DIPOLES (VEL)": self.transition_dipole_moment_velocity,  # noqa: E501
                f"{name} OSCILLATOR STRENGTHS (LEN)": self.oscillator_strength,
                f"{name} OSCILLATOR STRENGTHS (VEL)": self.oscillator_strength_velocity,  # noqa: E501
                f"{name} ROTATIONAL STRENGTHS (VEL)": self.rotatory_strength,
                #
                # State properties
                f"{name} STATE DIPOLES": self.state_dipole_moment
            })

        if recurse:
            mpvars = self.ground_state.to_qcvars(properties, recurse=True,
                                                 maxlevel=self.method.level)
            qcvars.update(mpvars)
        return qcvars

    @property
    def excitations(self):
        """
        Provides a list of Excitations, i.e., a view to all individual
        excited states and their properties. Still under heavy development.
        """
        excitations = [Excitation(self, index=i, method=self.method)
                       for i in range(self.size)]
        return excitations


# deprecated property names of ExcitedStates
deprecated = {
    "excitation_energies": "excitation_energy",
    "transition_dipole_moments": "transition_dipole_moment",
    "transition_dms": "transition_dm",
    "transition_dipole_moments_velocity":
        "transition_dipole_moment_velocity",
    "transition_magnetic_dipole_moments":
        "transition_magnetic_dipole_moment",
    "state_dipole_moments": "state_dipole_moment",
    "state_dms": "state_dm",
    "state_diffdms": "state_diffdm",
    "oscillator_strengths": "oscillator_strength",
    "oscillator_stenths_velocity": "oscillator_strength_velocity",
    "rotatory_strengths": "rotatory_strength",
    "excitation_vectors": "excitation_vector",
}

for dep_property in deprecated:
    new_key = deprecated[dep_property]

    def deprecated_property(self, key=new_key, old_key=dep_property):
        warnings.warn(f"Property '{old_key}' is deprecated "
                      " and will be removed in version 0.16.0."
                      f" Please use '{key}' instead.")
        return getattr(self, key)
    setattr(ExcitedStates, dep_property, property(deprecated_property))
