#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import libadcc

BLOCK_LABELS = ["s", "d", "t"]


class AmplitudeVector:
    def __init__(self, *tensors):
        """Initialise an AmplitudeVector from some blocks"""
        self.tensors = list(tensors)

    # TODO Attach some information about this Amplitude, e.g.
    #      is it CVS?

    def to_cpp(self):
        """
        Return the C++ equivalent of this object.
        This is needed at the interface to the C++ code.
        """
        return libadcc.AmplitudeVector(tuple(self.tensors))

    @property
    def blocks(self):
        return [BLOCK_LABELS[i] for i in range(len(self.tensors))]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.tensors[index]
        elif isinstance(index, str):
            if index not in BLOCK_LABELS:
                raise ValueError("Invalid index, either a block string "
                                 "like s,d,t, ... or an integer index "
                                 "are expected.")
            return self.__getitem__(BLOCK_LABELS.index(index))

    def __setitem__(self, index, item):
        if isinstance(index, int):
            self.tensors[index] = item
        elif isinstance(index, str):
            if index not in BLOCK_LABELS:
                raise ValueError("Invalid index, either a block string "
                                 "like s,d,t, ... or an integer index "
                                 "are expected.")
            return self.__setitem__(BLOCK_LABELS.index(index), item)

    def copy(self):
        """Return a copy of the AmplitudeVector"""
        return AmplitudeVector(*tuple(t.copy() for t in self.tensors))

    def ones_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(*tuple(t.ones_like() for t in self.tensors))

    def empty_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(*tuple(t.empty_like() for t in self.tensors))

    def nosym_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(*tuple(t.nosym_like() for t in self.tensors))

    def zeros_like(self):
        """Return an AmplitudeVector of the same shape and symmetry with
           all elements set to zero"""
        return AmplitudeVector(*tuple(t.zeros_like() for t in self.tensors))

    def add_linear_combination(self, scalars, others):
        """Return an AmplitudeVector of the same shape and symmetry with
           all elements set to zero"""
        if not isinstance(others, list):
            raise TypeError("Other is expected to be a list")
        if len(others) != len(scalars):
            raise ValueError("Length of scalars and others lists do not agree.")
        alltensors = [[av[b] for av in others] for b in self.blocks]
        return AmplitudeVector(*tuple(
            t.add_linear_combination(scalars, ot)
            for t, ot in zip(self.tensors, alltensors)
        ))

    def dot(self, other):
        """Return the dot product with another AmplitudeVector
        or the dot products with a list of AmplitudeVectors.
        In the latter case a np.ndarray is returned.
        """
        if isinstance(other, list):
            # Make a list where the first index is all singles parts,
            # the second is all doubles parts and so on
            alltensors = [[av[b] for av in other] for b in self.blocks]
            return sum(t.dot(ots) for t, ots in zip(self.tensors, alltensors))
        else:
            return sum(t.dot(ot) for t, ot in zip(self.tensors, other.tensors))

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.dot(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return self.dot(other)
        return NotImplemented

    def __forward_to_blocks(self, fname, other):
        if isinstance(other, AmplitudeVector):
            ret = tuple(getattr(t, fname)(ot)
                        for t, ot in zip(self.tensors, other.tensors))
        else:
            ret = tuple(getattr(t, fname)(other) for t in self.tensors)

        if any(r == NotImplemented for r in ret):
            return NotImplemented
        else:
            return AmplitudeVector(*ret)

    def __mul__(self, other):
        return self.__forward_to_blocks("__mul__", other)

    def __rmul__(self, other):
        return self.__forward_to_blocks("__rmul__", other)

    def __add__(self, other):
        return self.__forward_to_blocks("__add__", other)

    def __sub__(self, other):
        return self.__forward_to_blocks("__sub__", other)

    def __truediv__(self, other):
        return self.__forward_to_blocks("__truediv__", other)

    def __imul__(self, other):
        return self.__forward_to_blocks("__imul__", other)

    def __iadd__(self, other):
        return self.__forward_to_blocks("__iadd__", other)

    def __isub__(self, other):
        return self.__forward_to_blocks("__isub__", other)

    def __itruediv__(self, other):
        return self.__forward_to_blocks("__itruediv__", other)

    def __repr__(self):
        return "AmplitudeVector(blocks=" + ",".join(self.blocks) + ")"
