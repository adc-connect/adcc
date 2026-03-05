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
import warnings

from .import adc_ip, adc_ea
from .ElectronicStates import TableColumn, ElectronicStates, _timer_name
from .functions import dot
from .misc import cached_member_function


class ChargedExcitation(ElectronicStates):

    @property
    def pole_strength(self) -> np.ndarray:
        """Array of pole strengths of all computed states"""
        return np.array([
            self._pole_strength(i) for i in range(self.size)
        ])


    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _pole_strength(self, state_n: int) -> np.ndarray:
        """Computes the pole strength for a single state"""
        evec = self.excitation_vector[state_n]
        return self._module.pole_strength(
            self.property_method, self.ground_state, evec,
            self.matrix.intermediates)

    def describe_helper(self, pole_strengths=True, state_dipole_moments=False,
                 block_norms=True, excitation_type="energy", ssq=False):
        """
        Creates and returns the to be printed columns

        Parameters
        ----------
        pole_strengths : bool optional
            Show oscillator strengths, by default ``True``.

        state_dipole_moments : bool, optional
            Show state dipole moments, by default ``False``.

        block_norms : bool, optional
            Show the norms of the n particle (n-1) hole blocks of the charged
            excited states, by default ``True``.

        excitation_type : str, optional
            Defines the name of the energy property. 
            'ionization potential'/'electron affinity' for IP/EA
        ssq : bool, optional
            Show the <S^2> values of the excited states,  by default ``False``.
        """
        has_dipole = "electric_dipole" in self.operators.available
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
            header=excitation_type, values=values.copy(),
            unit="(au)         (eV)"
        ))
        values.clear()
        # the pole strengths
        if pole_strengths:
            values.extend(f"{pstr:^8.4f}" for pstr in self.pole_strength)
            columns.append(TableColumn(
                header="pole str", values=values.copy(), unit="(au)"
            ))
            values.clear()
        # vector norm
        blocks = self.matrix.axis_blocks
        if block_norms and len(blocks) > 0:
            values.extend(f"{dot(vec.get(blocks[0]), vec.get(blocks[0])):^9.4f}"
                          for vec in self.excitation_vector)
            columns.append(TableColumn(
                header="|v1|^2", values=values.copy(), unit=""
            ))
            values.clear()
        if block_norms and len(blocks) > 1:
            values.extend(f"{dot(vec.get(blocks[1]), vec.get(blocks[1])):^9.4f}"
                          for vec in self.excitation_vector)
            columns.append(TableColumn(
                header="|v2|^2", values=values.copy(), unit=""
            ))
            values.clear()
        # the state dipole moment
        if state_dipole_moments and has_dipole:
            warnings.warn("Dipole moments of charged species are gauge "
                          "dependent.")
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
        # <S^2> values
        if ssq and not self.reference_state.restricted:
            values.extend(f"{ssq:^9.4f}"
                          for ssq in self.state_ssq)
            columns.append(TableColumn(
                header="<S^2>", values=values.copy(), unit="(au)"
            ))
            values.clear()

        return columns


class DetachedStates(ChargedExcitation):
    _module = adc_ip

    def __init__(self, data, is_alpha: bool, method: str = None,
                 property_method: str = None):
        self.is_alpha = is_alpha
        super().__init__(data, method, property_method)

        if self.method.adc_type != "ip":
            raise ValueError("DetachedStates computes excited state properties"
                             " for IP-ADC. Got the non-IP-ADC method "
                             f"{self.method.name}")

    def describe(self, pole_strengths=True, state_dipole_moments=False,
                 block_norms=True, ssq=False):
        """
        Return a string providing a human-readable description of the class

        Parameters
        ----------
        pole_strengths : bool optional
            Show oscillator strengths, by default ``True``.

        state_dipole_moments : bool, optional
            Show state dipole moments, by default ``False``.

        block_norms : bool, optional
            Show the norms of the n particle (n+1) hole blocks of the charged
            excited states, by default ``True``.
        """
        assert (self.matrix.axis_blocks == ["h"] 
                or self.matrix.axis_blocks == ["h", "phh"])
        columns = self.describe_helper(
            pole_strengths=pole_strengths,
            state_dipole_moments=state_dipole_moments,
            block_norms=block_norms,
            excitation_type="ionization potential",
            ssq=ssq)

        # Format the state information: kind, spin_change,
        # alpha/beta detachment, and convergence
        state_info = []
        if hasattr(self, "kind") and self.kind:
            state_info.append(self.kind)
        spin_type = "alpha" if self.is_alpha else "beta"
        state_info.append(f"(ΔMS={self.spin_change:+.1f}), "
                                  f"{spin_type} detachment")
        if hasattr(self, "converged"):
            conv = "converged" if self.converged else "NOT CONVERGED"
            if state_info:  # add separator to previous entry
                state_info[-1] += ","
            state_info.append(conv)
        state_info = " ".join(state_info)
        return self._describe(columns, state_info)

    def to_qcvars(self, properties=False, recurse=False):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        name = self.method.name.upper()

        qcvars = {
            "EXCITATION KIND": self.kind.upper(),
            "NUMBER OF IONIZED STATES": len(self.excitation_energy),
            f"{name} ITERATIONS": self.n_iter,
            f"{name} IONIZATION POTENTIALS": self.excitation_energy,
        }

        if properties:
            qcvars.update({
                # Transition properties
                f"{name} POLE STRENGTHS": self.pole_strength,
                #
                # State properties
                f"{name} STATE DIPOLES": self.state_dipole_moment
            })

        if recurse:
            mpvars = self.ground_state.to_qcvars(properties, recurse=True,
                                                 maxlevel=self.method.level)
            qcvars.update(mpvars)
        return qcvars   


class AttachedStates(ChargedExcitation):
    _module = adc_ea

    def __init__(self, data, is_alpha: bool, method: str = None,
                 property_method: str = None):
        self.is_alpha = is_alpha
        super().__init__(data, method, property_method)

        if self.method.adc_type != "ea":
            raise ValueError("DetachedStates computes excited state properties"
                             " for EA-ADC. Got the non-EA-ADC method "
                             f"{self.method.name}")

    def describe(self, pole_strengths=True, state_dipole_moments=False,
                 block_norms=True, ssq=False):
        """
        Return a string providing a human-readable description of the class

        Parameters
        ----------
        pole_strengths : bool optional
            Show oscillator strengths, by default ``True``.

        state_dipole_moments : bool, optional
            Show state dipole moments, by default ``False``.

        block_norms : bool, optional
            Show the norms of the n particle (n+1) hole blocks of the charged
            excited states, by default ``True``.
        """
        assert (self.matrix.axis_blocks == ["p"] 
                or self.matrix.axis_blocks == ["p", "pph"])
        columns = self.describe_helper(
            pole_strengths=pole_strengths,
            state_dipole_moments=state_dipole_moments,
            block_norms=block_norms,
            excitation_type="electron affinity",
            ssq=ssq)

        # Format the state information: kind, spin_change,
        # alpha/beta detachment, and convergence
        state_info = []
        if hasattr(self, "kind") and self.kind:
            state_info.append(self.kind)
        spin_type = "alpha" if self.is_alpha else "beta"
        state_info.append(f"(ΔMS={self.spin_change:+.1f}), "
                                  f"{spin_type} attachment")
        if hasattr(self, "converged"):
            conv = "converged" if self.converged else "NOT CONVERGED"
            if state_info:  # add separator to previous entry
                state_info[-1] += ","
            state_info.append(conv)
        state_info = " ".join(state_info)
        return self._describe(columns, state_info)

    def to_qcvars(self, properties=False, recurse=False):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        name = self.method.name.upper()

        qcvars = {
            "EXCITATION KIND": self.kind.upper(),
            "NUMBER OF ELECTRON ATTACHED STATES": len(self.excitation_energy),
            f"{name} ITERATIONS": self.n_iter,
            f"{name} ELECTRON AFFINITIES": self.excitation_energy,
        }

        if properties:
            qcvars.update({
                # Transition properties
                f"{name} POLE STRENGTHS": self.pole_strength,
                #
                # State properties
                f"{name} STATE DIPOLES": self.state_dipole_moment
            })

        if recurse:
            mpvars = self.ground_state.to_qcvars(properties, recurse=True,
                                                 maxlevel=self.method.level)
            qcvars.update(mpvars)
        return qcvars   