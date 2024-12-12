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
from .timings import Timer
from .functions import evaluate


class Intermediates():
    """
    Class offering to return a number of intermediate tensors.
    The tensors are cached internally for later reuse
    """
    generators = {}      # Registered generator functions

    def __init__(self, ground_state):
        self.ground_state = ground_state
        self.reference_state = ground_state.reference_state
        self.timer = Timer()
        self.cached_tensors = {}  # Cached tensors
        # TODO Make caching configurable ??

    def __getattr__(self, key):
        if key in self.cached_tensors:
            return self.cached_tensors[key]
        elif key in self.generators:
            # Evaluate the tensor, all generators take (hf, mp, intermediates)
            generator = self.generators[key]
            with self.timer.record(key):
                tensor = generator(self.reference_state, self.ground_state, self)
                self.cached_tensors[key] = evaluate(tensor)
            return self.cached_tensors[key]
        else:
            raise AttributeError

    def clear(self):
        """Clear all cached tensors to free storage"""
        self.cached_tensors.clear()

    def __repr__(self):
        return (
            "AdcIntermediates(contains="
            + list(self.cached_tensors.keys()).join(",")
            + ")"
        )


def register_as_intermediate(function):
    """
    This decorator allows to register a function such that it can
    be used to produce intermediates for storage in Intermediates.
    The rule of thumb is that this should only be used on expressions
    which are used in multiple places of the ADC code (e.g. properties
    and matrix, multiple ADC variants etc.)
    """
    Intermediates.generators[function.__name__] = function
    return function
