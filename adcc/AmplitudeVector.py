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


class AmplitudeVector(dict):
    def __init__(self, **kwargs):
        """
        Construct an AmplitudeVector. Typical use cases are
        ``AmplitudeVector(ph=tensor_singles, pphh=tensor_doubles)``.
        """
        super().__init__(**kwargs)

    def __getattr__(self, key):
        if self.__contains__(key):
            return self.__getitem__(key)
        raise AttributeError

    def __setattr__(self, key, item):
        if self.__contains__(key):
            return self.__setitem__(key, item)
        raise AttributeError

    @property
    def blocks(self):
        """
        Return the blocks which are used inside the vector.
        """
        return sorted(self.keys())

    def copy(self):
        """Return a copy of the AmplitudeVector"""
        return AmplitudeVector(**{k: t.copy() for k, t in self.items()})

    def evaluate(self):
        for t in self.values():
            t.evaluate()
        return self

    @property
    def needs_evaluation(self):
        return any(t.needs_evaluation for _, t in self.items())

    def ones_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.ones_like() for k, t in self.items()})

    def empty_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.empty_like() for k, t in self.items()})

    def nosym_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.nosym_like() for k, t in self.items()})

    def zeros_like(self):
        """Return an AmplitudeVector of the same shape and symmetry with
           all elements set to zero"""
        return AmplitudeVector(**{k: t.zeros_like() for k, t in self.items()})

    def set_random(self):
        for t in self.values():
            t.set_random()
        return self

    def dot(self, other):
        """Return the dot product with another AmplitudeVector
        or the dot products with a list of AmplitudeVectors.
        In the latter case a np.ndarray is returned.
        """
        if isinstance(other, list):
            # Make a list where the first index is all singles parts,
            # the second is all doubles parts and so on
            return sum(self[b].dot([av[b] for av in other]) for b in self.keys())
        else:
            return sum(self[b].dot(other[b]) for b in self.keys())

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.dot(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return self.dot(other)
        return NotImplemented

    def __forward_to_blocks(self, fname, other):
        if isinstance(other, AmplitudeVector):
            if sorted(other.blocks) != sorted(self.blocks):
                raise ValueError("Blocks of both AmplitudeVector objects "
                                 f"need to agree to perform {fname}")
            ret = {k: getattr(tensor, fname)(other[k])
                   for k, tensor in self.items()}
        else:
            ret = {k: getattr(tensor, fname)(other) for k, tensor in self.items()}
        if any(r == NotImplemented for r in ret.values()):
            return NotImplemented
        else:
            return AmplitudeVector(**ret)

    def __mul__(self, other):
        return self.__forward_to_blocks("__mul__", other)

    def __rmul__(self, other):
        return self.__forward_to_blocks("__rmul__", other)

    def __sub__(self, other):
        return self.__forward_to_blocks("__sub__", other)

    def __rsub__(self, other):
        return self.__forward_to_blocks("__rsub__", other)

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
        return "AmplitudeVector(" + "=..., ".join(self.blocks) + "=...)"

    # __add__ is special because we want to be able to add AmplitudeVectors
    # with missing blocks
    def __add__(self, other):
        if isinstance(other, AmplitudeVector):
            allblocks = sorted(set(self.blocks).union(other.blocks))
            ret = {k: self.get(k, 0) + other.get(k, 0) for k in allblocks}
            ret = {k: v for k, v in ret.items() if v != 0}
        else:
            ret = {k: tensor + other for k, tensor in self.items()}
        return AmplitudeVector(**ret)

    def __radd__(self, other):
        if isinstance(other, AmplitudeVector):
            return other.__add__(self)
        else:
            ret = {k: other + tensor for k, tensor in self.items()}
            return AmplitudeVector(**ret)
