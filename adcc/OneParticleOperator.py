#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

import numpy as np

import libadcc

from .Tensor import Tensor
from .functions import empty_like


class OneParticleOperator(libadcc.OneParticleOperator):
    def __init__(self, mospaces, is_symmetric=True, cartesian_transform="1"):
        n_occ_ss = len(mospaces.subspaces_occupied)
        n_virt_ss = len(mospaces.subspaces_virtual)
        super().__init__(n_occ_ss, n_virt_ss, is_symmetric)
        # set all blocks to zero with correct symmetry
        for b in self.blocks:
            sym = libadcc.make_symmetry_operator(
                mospaces, b, is_symmetric, cartesian_transform
            )
            self.set_block(b, Tensor(sym))

    def __add__(self, other):
        if isinstance(other, libadcc.OneParticleOperator):
            assert self.blocks == other.blocks
            ret = empty_like(self)
            for b in ret.blocks:
                ret.set_block(b, self[b] + other[b])
            return ret
        else:
            raise TypeError("Cannot add OneParticleOperator"
                            " and {}".format(type(other)))

    def expectation_value(self, dm):
        non_zero_blocks = [b for b in dm.blocks if not dm.is_zero_block(b)]
        if self.is_symmetric and dm.is_symmetric:
            ret = 0
            assert self.blocks == dm.blocks
            for b in non_zero_blocks:
                # transposed block string
                tb = b[2:] + b[:2]
                if b == tb:
                    ret += self[b].dot(dm[b])
                else:
                    ret += 2.0 * self[b].dot(dm[b])
            return ret
        elif self.is_symmetric and not dm.is_symmetric:
            ret = 0
            for b in non_zero_blocks:
                # transposed block string
                tb = b[2:] + b[:2]
                if b in self.blocks:
                    ret += self[b].dot(dm[b])
                elif tb in self.blocks:
                    ret += self[tb].transpose().dot(dm[b])
            return ret
        elif not self.is_symmetric and dm.is_symmetric:
            ret = 0
            for b in self.blocks:
                # transposed block string
                tb = b[2:] + b[:2]
                if b in dm.blocks and not dm.is_zero_block(b):
                    ret += self[b].dot(dm[b])
                elif tb in dm.blocks and not dm.is_zero_block(tb):
                    ret += self[b].dot(dm[tb].transpose())
            return ret
        else:
            ret = 0
            assert self.blocks == dm.blocks
            for b in non_zero_blocks:
                ret += self[b].dot(dm[b])
            return ret


class HfDensityMatrix(OneParticleOperator):
    def __init__(self, mospaces):
        super().__init__(mospaces, True)
        for b in self.blocks:
            sym = libadcc.make_symmetry_operator(
                mospaces, b, True, "1"
            )
            self.set_block(b, Tensor(sym))
        diag = np.eye(self["o1o1"].shape[0])
        self["o1o1"].set_from_ndarray(diag)
        if self.has_block("o2o2"):
            diag = np.eye(self["o2o2"].shape[0])
            self["o2o2"].set_from_ndarray(diag)
