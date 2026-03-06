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
        pass


    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _pole_strength(self, state_n: int) -> np.ndarray:
        """Computes the pole strength for a single state"""
        pass


    def describe_helper(self, block_norms=True, excitation_type_name="energy"):
        """
        Creates and returns the to be printed columns

        Parameters
        ----------
        block_norms : bool, optional
            Show the norms of the n particle (n-1) hole blocks of the charged
            excited states, by default ``True``.

        excitation_name : str, optional
            Defines the name of the energy property. 
            'ionization potential'/'electron affinity' for IP/EA
        ssq : bool, optional
            Show the <S^2> values of the excited states,  by default ``False``.
        """
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
            header=excitation_type_name, values=values.copy(),
            unit="(au)         (eV)"
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

    def describe(self, block_norms=True):
        """
        Return a string providing a human-readable description of the class

        Parameters
        ----------
        block_norms : bool, optional
            Show the norms of the n particle (n+1) hole blocks of the charged
            excited states, by default ``True``.
        """
        assert (self.matrix.axis_blocks == ["h"] 
                or self.matrix.axis_blocks == ["h", "phh"])
        columns = self.describe_helper(
            block_norms=block_norms,
            excitation_type_name="ionization potential")

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

    def describe(self, block_norms=True):
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
            block_norms=block_norms,
            excitation_type_name="electron affinity")

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