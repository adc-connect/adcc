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

from .AdcMatrix import AdcMatrixlike
from .functions import empty_like
from .AmplitudeVector import AmplitudeVector

import libadcc


class AdcBlockView(AdcMatrixlike):
    def __init__(self, fullmatrix, block):
        """
        Object which acts as a single block of the ADC matrix.

        Parameters
        ----------
        fullmatrix : AdcMatrixlike
            The full matrix of which one wants to view a block
        block : str
            The block to view. Valid choices include `s` (view the
            singles-singles block)or `d` (view the doubles-doubles block).
        """
        if block not in fullmatrix.blocks:
            raise ValueError(f"Block {block} not a valid block of passed "
                             "matrix {self.__fullmatrix}")
        self.__block = block
        self.__fullmatrix = fullmatrix
        super().__init__(fullmatrix)

    def __repr__(self):
        return f"AdcBlockView({self.__fullmatrix},{self.__block})"

    @property
    def blocks(self):
        return [self.__block]

    def has_block(self, block):
        return self.__block == block

    def __len__(self):
        return np.prod([
            self.mospaces.n_orbs(sp) for sp in self.block_spaces(self.__block)
        ])

    @property
    def shape(self):
        return (len(self), len(self))

    def __assert_block(self, block):
        if not all(b == self.__block for b in block):
            raise ValueError(
                f"The AdcBlockView onto the {self.__block} block of the matrix "
                "{self.__fullmatrix} does not have block {block}."
            )

    def diagonal(self, block):
        self.__assert_block(block)
        return super().diagonal(block)

    def block_spaces(self, block):
        self.__assert_block(block)
        return super().block_spaces(self.__block)

    def compute_matvec(self, in_ampl, out_ampl=None):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result
        in the out_ampl if it is given, else the result
        will be returned.
        """
        # Unwrap input object, making sure we have two adcc.Tensors: one
        # for the input and one for the result
        return_bare_tensors = True
        if isinstance(in_ampl, AmplitudeVector):
            return_bare_tensors = False
            if not in_ampl.blocks == [self.__block]:
                raise ValueError("in_ampl does not consist of the correct blocks")
            in_ampl = in_ampl[self.__block]
        elif not isinstance(in_ampl, libadcc.Tensor):
            raise TypeError("in_ampl has to be of type AmplitudeVector "
                            "or Tensor.")
        if out_ampl is None:
            out_ampl = empty_like(in_ampl)
        elif isinstance(out_ampl, AmplitudeVector):
            if not out_ampl.blocks == [self.__block]:
                raise ValueError("out_ampl does not consist of the "
                                 "correct blocks")
            out_ampl = out_ampl[self.__block]
        elif not isinstance(out_ampl, libadcc.Tensor):
            raise TypeError("out_ampl has to be of type AmplitudeVector "
                            "or Tensor.")

        # Compute the matrix-vector product
        super().compute_apply(2 * self.__block, in_ampl, out_ampl)

        if return_bare_tensors:
            return out_ampl
        else:
            # enwrap again in AmplitudeVector
            if self.__block != "s":
                raise NotImplementedError
            return AmplitudeVector(out_ampl)

    def compute_apply(self, block, in_ampl, out_ampl):
        self.__assert_block(block)
        super().compute_apply(block, in_ampl, out_ampl)
