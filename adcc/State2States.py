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
from .misc import cached_property
from .timings import timed_member_call

from .Excitation import mark_excitation_property
from .ElectronicTransition import ElectronicTransition


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

    @property
    def excitation_energy(self):
        return np.array([
            self._excitation_energy[final]
            - self._excitation_energy[self.initial]
            for final in range(self.size) if final > self.initial
        ])

    @cached_property
    @mark_excitation_property(transform_to_ao=True)
    @timed_member_call(timer="_property_timer")
    def transition_dm(self):
        """
        List of transition density matrices from
        initial state to final state/s
        """
        return [
            adc_pp.state2state_transition_dm(
                self.property_method, self.ground_state,
                self.excitation_vector[self.initial],
                self.excitation_vector[final],
                self.matrix.intermediates
            )
            for final in range(self.size) if final > self.initial
        ]
