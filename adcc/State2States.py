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
    """
    Documentation
    """
    def __init__(self, parent_state, initial=0):
        self.parent_state = parent_state
        self.reference_state = self.parent_state.reference_state
        self.ground_state = self.parent_state.ground_state
        self.matrix = self.parent_state.matrix
        self.property_method = self.parent_state.property_method
        self.operators = self.parent_state.operators
        self.initial = initial
    # TODO: describe?!

    @property
    def excitation_energy(self):
        return np.array([
            e.excitation_energy
            - self.parent_state.excitation_energy[self.initial]
            for e in self.parent_state.excitations if e.index > self.initial
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
                self.parent_state.excitation_vector[self.initial],
                e.excitation_vector, self.matrix.intermediates
            )
            for e in self.parent_state.excitations
            if e.index > self.initial
        ]
