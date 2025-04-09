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

from . import adc_pp
from .ElectronicStates import _timer_name
from .ElectronicTransition import ElectronicTransition
from .misc import cached_member_function
from .OneParticleOperator import OneParticleOperator


class State2States(ElectronicTransition):
    def __init__(self, data, method=None, property_method=None, initial=0):
        """Construct a State2States class from some data obtained
            from an interative solver or an :class:`ExcitedStates` object.

            The class provides access to ADC transition properties between
            excited states, i.e., from the `initial` state to all higher-lying
            excited states obtained from an ADC calculation.

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
            initial : int, optional
                Provide the index of the excited state from which transitions
                to all other higher-lying states are to be computed.
        """
        super().__init__(data, method, property_method)
        self.initial = initial

        # Since this class should be used for all adc_types we have to determine
        # the module according to the method.
        if self.method.adc_type == "pp":
            self._module = adc_pp
        else:
            raise ValueError(f"Unknown adc_type {self.method.adc_type}.")

    @property
    def size(self) -> int:
        # the number of states "above" the initial state to correctly index
        # into the transition_dm array
        return super().size - self.initial - 1

    @property
    def excitation_energy(self) -> np.ndarray:
        """
        Excitation energies from the inital state to energetically higher lying
        states in atomic units
        """
        return np.array([
            self._excitation_energy[final]
            - self._excitation_energy[self.initial]
            for final in range(self.initial + 1, super().size)
        ])

    @property
    def excitation_energy_uncorrected(self) -> np.ndarray:
        """
        Excitation energies without any corrections from the inital state to
        energetically higher lying states in atomic units
        """
        return np.array([
            self._excitation_energy_uncorrected[final]
            - self._excitation_energy_uncorrected[self.initial]
            for final in range(self.initial + 1, super().size)
        ])

    @property
    def excitation_vector(self):
        """List of excitation vectors"""
        return self._excitation_vector[self.initial + 1:]

    @property
    def transition_dm(self) -> list[OneParticleOperator]:
        """
        List of transition density matrices from initial state to final state/s
        """
        return [self._transition_dm(final) for final in range(self.size)]

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _transition_dm(self, state_n: int) -> OneParticleOperator:
        """
        Computes the transition density matrices from initial state to a single
        final state
        """
        # NOTE: state_n is relative to the initial state, i.e.,
        # 0 refers to initial_state + 1.
        # This is necessary to enable the use of the implementations on
        # the parent class.
        state_n = self.initial + state_n + 1
        return self._module.state2state_transition_dm(
            self.property_method, self.ground_state,
            super().excitation_vector[self.initial],
            super().excitation_vector[state_n],
            self.matrix.intermediates
        )

    def _state_diffdm(self, state_n: int) -> OneParticleOperator:
        """
        The difference density matrix are not available through the
        :class:`State2States`
        class. Please use e.g. :class:`ExcitedStates` (for PP-ADC) instead
        """
        # alternatively one could create an independent StateProperties class
        # and additionally inherit from that class on ExcitedStates and the
        # future IP/EA class.
        raise NotImplementedError("Difference density matrices and excited state "
                                  "properties are not available through "
                                  f"'{self.__class__.__name__}'. Please use e.g. "
                                  "'ExcitedStates' for PP-ADC.")
