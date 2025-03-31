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

from .OneParticleOperator import OneParticleOperator
from .StateView import StateView


def mark_excitation_property(**kwargs):
    """
    Decorator to mark properties of :class:`adcc.ExcitedStates` which
    can be transferred to :class:`Excitation` defined below

    Parameters
    ----------
    transform_to_ao : bool, optional
        If set to True, the marked matrix property
        (e.g., :func:`adcc.ExcitedStates.transition_dm`)
        transformed to the AO basis will be added to :class:`Excitation`.
        In the given example, this would result in `transition_dm` and
        `transition_dm_ao`. The AO matrix is returned as a numpy array.
    """
    def inner(f, kwargs=kwargs):
        f.__excitation_property = kwargs
        return f
    return inner


class Excitation(StateView):
    def __init__(self, parent_state, index: int):
        """
        Construct an Excitation instance from an :class:`adcc.ExcitedStates`
        parent object.

        The class provides access to the properties of a single
        excited state, dynamically constructed inside ExcitedStates.excitations.

        Parameters
        ----------
        parent_state
            :class:`adcc.ExcitedStates` object from which the Excitation
            is derived
        index : int
            Index of the excited state the constructed :class:`adcc.Excitation`
            should refer to (0-based)
        """
        from .ExcitedStates import ElectronicTransition
        # NOTE: This should also work with S2S. But then index
        # would need to be relative to S2S.initial, i.e., 0 for S2S.initial + 1
        # which is kind of weird. But I think we can allow it.
        if not isinstance(parent_state, ElectronicTransition):
            raise TypeError("parent_state needs to be an ExcitedStates object. "
                            f"Got: {type(parent_state)}.")
        super().__init__(parent_state, index)
        self._parent_state: ElectronicTransition

    @property
    def transition_dm(self) -> OneParticleOperator:
        """The transition density matrix"""
        return self._parent_state._transition_dm(self.index)

    @property
    def transition_dm_ao(self) -> OneParticleOperator:
        """The transition density matrix in the AO basis"""
        return sum(self.transition_dm.to_ao_basis())

    @property
    def transition_dipole_moment(self) -> np.ndarray:
        """The transition dipole moment"""
        return self._parent_state._transition_dipole_moment(self.index)

    @property
    def oscillator_strength(self) -> np.float64:
        """The oscillator strength"""
        return self._parent_state._oscillator_strength(self.index)
