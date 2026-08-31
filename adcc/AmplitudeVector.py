#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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
from collections.abc import Sequence
from typing import overload

import numpy as np

import libadcc


class AmplitudeVector(dict[str, libadcc.Tensor]):
    def __init__(self, **kwargs: libadcc.Tensor):
        """
        Construct an AmplitudeVector. Typical use cases are
        ``AmplitudeVector(ph=tensor_singles, pphh=tensor_doubles)``.
        """
        super().__init__(**kwargs)

    def __getattr__(self, key: str) -> libadcc.Tensor:
        if self.__contains__(key):
            return self.__getitem__(key)
        raise AttributeError

    def __setattr__(self, key: str, item: libadcc.Tensor) -> None:
        if self.__contains__(key):
            return self.__setitem__(key, item)
        raise AttributeError

    @property
    def blocks(self) -> list[str]:
        """
        Return the sorted blocks which are used inside the vector, e.g.,
        ['ph', 'pphh'].
        """
        return sorted(self.keys())

    def copy(self) -> "AmplitudeVector":
        """Return a copy of the AmplitudeVector"""
        return AmplitudeVector(**{k: t.copy() for k, t in self.items()})

    def evaluate(self) -> "AmplitudeVector":
        """
        Ensures all blocks of the AmplitudeVector are evaluated and
        resilient in memory.
        """
        for t in self.values():
            t.evaluate()
        return self

    @property
    def needs_evaluation(self) -> bool:
        return any(t.needs_evaluation for _, t in self.items())

    def ones_like(self) -> "AmplitudeVector":
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.ones_like() for k, t in self.items()})

    def empty_like(self) -> "AmplitudeVector":
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.empty_like() for k, t in self.items()})

    def nosym_like(self) -> "AmplitudeVector":
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.nosym_like() for k, t in self.items()})

    def zeros_like(self) -> "AmplitudeVector":
        """Return an AmplitudeVector of the same shape and symmetry with
        all elements set to zero"""
        return AmplitudeVector(**{k: t.zeros_like() for k, t in self.items()})

    def set_random(self) -> "AmplitudeVector":
        """
        Fills all blocks of the AmplitudeVector with random elements.
        """
        for t in self.values():
            t.set_random()
        return self

    @overload
    def dot(self, other: "AmplitudeVector") -> float:
        ...

    @overload
    def dot(
        self, other: Sequence["AmplitudeVector"]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        ...

    def dot(
        self, other: "Sequence[AmplitudeVector] | AmplitudeVector"
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | float:
        """
        Return the dot product with another AmplitudeVector
        or the dot products with a list of AmplitudeVectors.
        In the latter case a np.ndarray is returned.
        Only blocks common to self and other are considered for the dot product.
        """
        if isinstance(other, Sequence):
            ret = np.zeros(len(other))
            for block, tensor in self.items():
                idx = [i for i, vec in enumerate(other) if block in vec]
                if idx:
                    # use "fast path" that computes the overlap with
                    # multiple tensors at once in the backend
                    ret[idx] += tensor.dot([other[i][block] for i in idx])
            return ret
        return sum(
            (self[b].dot(other[b]) for b in sorted(self.keys() & other.keys())),
            0.0  # so we always return a float even when the intersection is empty
        )

    @overload
    def __matmul__(self, other: "AmplitudeVector") -> float:
        ...

    @overload
    def __matmul__(
        self, other: Sequence["AmplitudeVector"]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        ...

    def __matmul__(
        self, other: "Sequence[AmplitudeVector] | AmplitudeVector"
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | float:
        if isinstance(other, AmplitudeVector) or isinstance(other, Sequence) and all(
            isinstance(t, AmplitudeVector) for t in other
        ):
            return self.dot(other)
        return NotImplemented

    def __mul__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        # missing blocks can be treated as multiplying by zero
        # and omited in the result Vector
        if isinstance(other, AmplitudeVector):
            ret = {
                block: self[block].__mul__(other[block])
                for block in self.keys() & other.keys()
            }
        else:
            ret = {block: self[block].__mul__(other) for block in self.keys()}
        return AmplitudeVector(**ret)

    def __rmul__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        if isinstance(other, AmplitudeVector):
            return other.__mul__(self)
        # This path is not reachable currently, because the libadcc backend
        # raises a TypeError instead of returning NotImplemented.
        # Thus, the reflective operations are not called in this case.
        elif isinstance(other, libadcc.Tensor):
            ret = {block: other.__mul__(tensor) for block, tensor in self.items()}
        else:  # float
            ret = {block: tensor.__rmul__(other) for block, tensor in self.items()}
        return AmplitudeVector(**ret)

    def __imul__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        # tensor operations in the backend are currently not really in-place
        # erase all blocks that are missing in other and update the common blocks
        if isinstance(other, AmplitudeVector):
            for block in self.keys() | other.keys():
                if block in self and block in other:
                    self[block] = self[block].__mul__(other[block])
                elif block in self:
                    del self[block]
        elif isinstance(other, libadcc.Tensor):
            for block, tensor in self.items():
                self[block] = tensor.__mul__(other)
        else:  # float
            for block, tensor in self.items():
                self[block] = tensor.__imul__(other)
        return self

    def __add__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        if isinstance(other, AmplitudeVector):
            ret = {}
            for block in self.keys() | other.keys():
                if block in self and block in other:
                    ret[block] = self[block].__add__(other[block])
                elif block in self:  # x + 0
                    ret[block] = self[block].copy()
                else:  # 0 + x
                    ret[block] = other[block].copy()
        else:
            ret = {block: tensor.__add__(other) for block, tensor in self.items()}
        return AmplitudeVector(**ret)

    def __radd__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        if isinstance(other, AmplitudeVector):
            return other.__add__(self)
        # This path is not reachable currently, because the libadcc backend
        # raises a TypeError instead of returning NotImplemented.
        # Thus, the reflective operations are not called in this case.
        elif isinstance(other, libadcc.Tensor):
            ret = {block: other.__add__(tensor) for block, tensor in self.items()}
        else:  # float
            ret = {block: tensor.__radd__(other) for block, tensor in self.items()}
        return AmplitudeVector(**ret)

    def __iadd__(
        self, other: "AmplitudeVector | libadcc.Tensor"
    ) -> "AmplitudeVector":
        # not really in-place in the backend!
        # missing block in self: 0 + x = x -> copy the other block
        # missing block in other: x + 0 = x -> nothing to do
        if isinstance(other, AmplitudeVector):
            for block, tensor in other.items():
                if block in self:
                    self[block] = self[block].__iadd__(tensor)
                else:
                    self[block] = tensor.copy()
        else:
            for block, tensor in self.items():
                self[block] = tensor.__iadd__(other)
        return self

    def __sub__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        if isinstance(other, AmplitudeVector):
            ret = {}
            for block in self.keys() | other.keys():
                if block in self and block in other:
                    ret[block] = self[block].__sub__(other[block])
                elif block in self:  # x - 0
                    ret[block] = self[block].copy()
                else:  # 0 - x
                    ret[block] = other[block].__neg__()
        else:
            ret = {block: self[block].__sub__(other) for block in self.keys()}
        return AmplitudeVector(**ret)

    def __rsub__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        if isinstance(other, AmplitudeVector):
            return other.__sub__(self)
        # This path is not reachable currently, because the libadcc backend
        # raises a TypeError instead of returning NotImplemented.
        # Thus, the reflective operations are not called in this case.
        elif isinstance(other, libadcc.Tensor):
            ret = {block: other.__sub__(tensor) for block, tensor in self.items()}
        else:  # float
            ret = {block: tensor.__rsub__(other) for block, tensor in self.items()}
        return AmplitudeVector(**ret)

    def __isub__(
        self, other: "AmplitudeVector | libadcc.Tensor"
    ) -> "AmplitudeVector":
        # not really in-place in the backend!
        # missing block in self: 0 - x = -x
        # missing block in other: x - 0 = x -> nothing to do
        if isinstance(other, AmplitudeVector):
            for block, tensor in other.items():
                if block in self:
                    self[block] = self[block].__isub__(tensor)
                else:
                    self[block] = tensor.__neg__()
        else:
            for block, tensor in self.items():
                self[block] = tensor.__isub__(other)
        return self

    def __truediv__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        # here all blocks of other have to be in self present, too!
        # block missing in other: x / 0 -> division by zero
        # block missing in self: 0 / x = 0 -> fine but not in result
        if isinstance(other, AmplitudeVector):
            if any(block not in other for block in self.keys()):
                raise ZeroDivisionError(
                    "Division by zero, since missing blocks are treated as "
                    f"zero blocks. Self contains {self.blocks}. Other contains "
                    f"{other.blocks}."
                )
            ret = {
                block: tensor.__truediv__(other[block])
                for block, tensor in self.items()
            }
        else:
            ret = {block: self[block].__truediv__(other) for block in self.keys()}
        return AmplitudeVector(**ret)

    def __itruediv__(
        self, other: "AmplitudeVector | libadcc.Tensor | float"
    ) -> "AmplitudeVector":
        # not really in-place in the backend!
        # missing block in other: x / 0 -> division by zero
        # missing block in self: 0 / x = 0 -> fine but not in result
        if isinstance(other, AmplitudeVector):
            if any(block not in other for block in self.keys()):
                raise ZeroDivisionError(
                    "Division by zero, since missing blocks are treated as "
                    f"zero blocks. Self contains {self.blocks}. Other contains "
                    f"{other.blocks}."
                )
            for block, tensor in self.items():
                self[block] = tensor.__truediv__(other[block])
        elif isinstance(other, libadcc.Tensor):
            for block, tensor in self.items():
                self[block] = tensor.__truediv__(other)
        else:  # float
            for block, tensor in self.items():
                self[block] = tensor.__itruediv__(other)
        return self

    def __repr__(self):
        if self:
            return f"AmplitudeVector({'=..., '.join(self.blocks)}=...)"
        else:  # empty vector
            return "AmplitudeVector()"
