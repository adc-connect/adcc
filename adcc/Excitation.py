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


class Excitation:
    def __init__(self, parent_state, index, method):
        """Construct an Excitation from an :class:`adcc.ExcitedStates`
        parent object.

        The class provides access to the properties of a single
        excited state, dynamically constructed inside ExcitedStates.excitations
        All properties marked with :func:`mark_excitation_property` are
        set as properties of :class:`adcc.Excitation`.

        Parameters
        ----------
        parent_state
            :class:`adcc.ExcitedStates` object from which the Excitation
            is derived
        index : int
            Index of the excited state the constructed :class:`adcc.Excitation`
            should refer to (0-based)
        method : AdcMethod
            ADC method of the parent :class:`adcc.ExcitedStates` object
        """
        self.__parent_state = parent_state
        self.index = index
        self.method = method
        for key in self.parent_state.excitation_property_keys:
            fget = getattr(type(self.parent_state), key).fget
            # Extract the kwargs passed to mark_excitation_property
            kwargs = getattr(fget, "__excitation_property").copy()

            def get_parent_property(self, key=key, kwargs=kwargs):
                return getattr(self.parent_state, key)[self.index]

            setattr(Excitation, key, property(get_parent_property))

            transform = kwargs.pop("transform_to_ao", False)
            if transform:
                def get_parent_property_transform(self, key=key):
                    matrix = getattr(self.parent_state, key)[self.index]
                    return sum(matrix.to_ao_basis())

                setattr(Excitation, key + "_ao",
                        property(get_parent_property_transform))

    @property
    def parent_state(self):
        return self.__parent_state
