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
from scipy import constants

from .import adc_pp
from .ElectronicStates import TableColumn
from .ElectronicTransition import ElectronicTransition
from .FormatDominantElements import FormatDominantElements
from .FormatIndex import (FormatIndexAdcc, FormatIndexBase,
                          FormatIndexHfProvider, FormatIndexHomoLumo)
from .functions import dot


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
    _module = adc_pp

    def __init__(self, data, method: str = None, property_method: str = None):
        super().__init__(data, method, property_method)

        if self.method.adc_type != "pp":
            raise ValueError("ExcitedStates computes excited state properties "
                             "for PP-ADC. Got the non-PP-ADC method "
                             f"{self.method.name}")

    def _repr_pretty_(self, pp, cycle):
        if cycle:
            pp.text("ExcitedStates(...)")
        else:
            pp.text(self.describe())

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
        has_dipole = "electric_dipole" in self.operators.available
        has_rotatory = all(
            op in self.operators.available for op in
            ["magnetic_dipole", "electric_dipole_velocity"]
        )
        # Collect the columns to print
        columns: list[TableColumn] = []
        values: list[str] = []
        # count the number of states
        values.extend(str(i) for i in range(self.size))
        columns.append(TableColumn(header="#", values=values.copy(), unit=""))
        values.clear()
        # excitation energy in a.u. and eV
        eV = constants.value("Hartree energy in eV")
        values.extend(f"{e:^13.7g} {e * eV:^13.7g}" for e in self.excitation_energy)
        columns.append(TableColumn(
            header="excitation energy", values=values.copy(),
            unit="(au)         (eV)"
        ))
        values.clear()
        # the transition dipole moments
        if transition_dipole_moments and has_dipole:
            for tdm in self.transition_dipole_moment:
                tdmx, tdmy, tdmz = tdm
                values.append(
                    f"{tdmx:^8.4f} {tdmy:^8.4f} {tdmz:^8.4f}"
                    f"{np.linalg.norm(tdm):^8.4f}"
                )
            columns.append(TableColumn(
                header="transition dipole moment", values=values.copy(),
                unit="x(au)    y(au)    z(au)    abs(au)"
            ))
            values.clear()
        # the oscillator strengths
        if oscillator_strengths and has_dipole:
            values.extend(f"{osc:^8.4f}" for osc in self.oscillator_strength)
            columns.append(TableColumn(
                header="osc str", values=values.copy(), unit="(au)"
            ))
            values.clear()
        if rotatory_strengths and has_rotatory:
            values.extend(f"{rot:^8.4f}" for rot in self.rotatory_strength)
            columns.append(TableColumn(
                header="rot str", values=values.copy(), unit="(au)"
            ))
            values.clear()
        # vector norm
        if block_norms and "ph" in self.matrix.axis_blocks:
            values.extend(f"{dot(vec.ph, vec.ph):^9.4f}"
                          for vec in self.excitation_vector)
            columns.append(TableColumn(
                header="|v1|^2", values=values.copy(), unit=""
            ))
            values.clear()
        if block_norms and "pphh" in self.matrix.axis_blocks:
            values.extend(f"{dot(vec.pphh, vec.pphh):^9.4f}"
                          for vec in self.excitation_vector)
            columns.append(TableColumn(
                header="|v2|^2", values=values.copy(), unit=""
            ))
            values.clear()
        # the state dipole moment
        if state_dipole_moments and has_dipole:
            for dm in self.state_dipole_moment:
                dmx, dmy, dmz = dm
                values.append(
                    f"{dmx:^8.4f} {dmy:^8.4f} {dmz:^8.4f}"
                    f"{np.linalg.norm(dm):^8.4f}"
                )
            columns.append(TableColumn(
                header="state dipole moment", values=values.copy(),
                unit="x(au)    y(au)    z(au)    abs(au)"
            ))
            values.clear()
            values.clear()
        return self._describe(columns)

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
