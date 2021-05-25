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
import warnings
from .AmplitudeVector import AmplitudeVector


class DipAmplitudeVector(AmplitudeVector):
    def __init__(self, *args, **kwargs):
        """
        Construct an AmplitudeVector for the Dip scheme. Typical use cases are
        ``AmplitudeVector(hh=tensor_singles, phhh=tensor_doubles)``.
        """
    @property
    def blocks(self):
        warnings.warn("The blocks attribute will change behaviour in 0.16.0.")
        if sorted(self.blocks_hh) == ["hh", "phhh"]:
            return ["s", "d"]
        if sorted(self.blocks_hh) == ["phhh"]:
            return ["d"]
        elif sorted(self.blocks_hh) == ["hh"]:
            return ["s"]
        elif sorted(self.blocks_hh) == []:
            return []
        else:
            raise NotImplementedError(self.blocks_hh)

    @property
    def blocks_hh(self):
        """
        Return the blocks which are used inside the vector.
        Note: This is a temporary name. The attribute will be removed in 0.16.0.
        """
        return sorted(self.keys())

    def copy(self):
        """Return a copy of the AmplitudeVector"""
        return DipAmplitudeVector(**{k: t.copy() for k, t in self.items()})

    def ones_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return DipAmplitudeVector(**{k: t.ones_like() for k, t in self.items()})

    def empty_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return DipAmplitudeVector(**{k: t.empty_like() for k, t in self.items()})

    def nosym_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return DipAmplitudeVector(**{k: t.nosym_like() for k, t in self.items()})

    def zeros_like(self):
        """Return an AmplitudeVector of the same shape and symmetry with
           all elements set to zero"""
        return DipAmplitudeVector(**{k: t.zeros_like() for k, t in self.items()})

    def __forward_to_blocks(self, fname, other):
        if isinstance(other, DipAmplitudeVector):
            if sorted(other.blocks_hh) != sorted(self.blocks_hh):
                raise ValueError("Blocks of both AmplitudeVector objects "
                                 f"need to agree to perform {fname}")
            ret = {k: getattr(tensor, fname)(other[k])
                   for k, tensor in self.items()}
        else:
            ret = {k: getattr(tensor, fname)(other) for k, tensor in self.items()}
        if any(r == NotImplemented for r in ret.values()):
            return NotImplemented
        else:
            return DipAmplitudeVector(**ret)

    def __repr__(self):
        return "DipAmplitudeVector(" + "=..., ".join(self.blocks_hh) + "=...)"

    # __add__ is special because we want to be able to add AmplitudeVectors
    # with missing blocks
    def __add__(self, other):
        if isinstance(other, AmplitudeVector):
            allblocks = sorted(set(self.blocks_hh).union(other.blocks_hh))
            ret = {k: self.get(k, 0) + other.get(k, 0) for k in allblocks}
            ret = {k: v for k, v in ret.items() if v != 0}
        else:
            ret = {k: tensor + other for k, tensor in self.items()}
        return DipAmplitudeVector(**ret)

    def __radd__(self, other):
        if isinstance(other, AmplitudeVector):
            return other.__add__(self)
        else:
            ret = {k: other + tensor for k, tensor in self.items()}
            return DipAmplitudeVector(**ret)
