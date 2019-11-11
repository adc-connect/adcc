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
import copy

from .Tensor import _tensor_select_below_absmax
from .FormatIndex import FormatIndexAdcc, FormatIndexBase

from libadcc import Tensor


class FormatDominantElements:
    def __init__(self, mospaces, tolerance=0.01, index_format=FormatIndexAdcc):
        self.mospaces = mospaces
        self.tolerance = tolerance
        self.value_format = "{:+8.3g}"  # Formatting used for the values

        if isinstance(index_format, type):
            self.index_format = index_format(self.mospaces)
        elif isinstance(index_format, FormatIndexBase):
            self.index_format = copy.copy(index_format)
        else:
            raise TypeError("index_format needs to be of type FormatIndexBase")

    def optimise_formatting(self, spaces_tensor_pairs):
        """
        Optimise the formatting parameters of this class and the index_format
        class in order to be able to nicely produce equivalently formatted tensor
        format strings for all the passed spaces-tensor pairs.

        This function can be called multiple times.
        """
        if not isinstance(spaces_tensor_pairs, list):
            return self.optimise_formatting([spaces_tensor_pairs])

        for spaces, tensor in spaces_tensor_pairs:
            if not isinstance(spaces, (tuple, list)) or \
               not isinstance(tensor, Tensor):
                raise TypeError("spaces_tensor_pairs should be a list of "
                                "(spaces tuples, Tensor) tuples")
            for indices, _ in _tensor_select_below_absmax(tensor, self.tolerance):
                self.index_format.optimise_formatting(
                    [(spaces[j], idx) for j, idx in enumerate(indices)])

    def format_as_list(self, spaces, tensor):
        """Raw-format the dominant tensor elements as a list of tuples with one
        tuple for each element. Each tuple has three entries, the formatted
        indices, the formatted spins and the value"""
        ret = []
        for indices, value in _tensor_select_below_absmax(tensor, self.tolerance):
            formatted = tuple(self.index_format.format(spaces[j], idx,
                                                       concat_spin=False)
                              for j, idx in enumerate(indices))
            ret.append(tuple(zip(*formatted)) + (value, ))
        return ret

    def format(self, spaces, tensor):
        """
        Return a multiline string representing the dominant tensor elements.
        The tensor index is formatted according to the index format passed
        upon class construction.
        """
        ret = []
        for indices, spins, value in self.format_as_list(spaces, tensor):
            ret.append(" ".join(indices) + "  " + "".join(spins)
                       + "   " + self.value_format.format(value))
        return "\n".join(ret)
