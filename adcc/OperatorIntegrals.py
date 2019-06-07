#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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

from .Tensor import Tensor
from .OneParticleOperator import OneParticleOperator

import libadcc


def transform_operator_ao2mo(tensor_bb, tensor_ff, coefficients,
                             conv_tol=1e-14):
    """
    Take a block-diagonal tensor in the atomic orbital basis
    and transform it into the molecular orbital basis in the
    convention used by adcc.

    @param tensor_bb  Block-diagonal tensor in the atomic orbital basis
    @param tensor_ff  Output tensor with the symmetry set-up to contain
                      the operator in the molecular orbital representation
    @param coefficients    Function providing coefficient blocks
    @param conv_tol   SCF convergence tolerance
    """
    for blk in tensor_ff.blocks:
        assert len(blk) == 4
        cleft = coefficients(blk[:2] + "b")
        cright = coefficients(blk[2:] + "b")
        temp = cleft @ tensor_bb @ cright.transpose()

        # TODO: once the permutational symmetry is correct:
        # tensor_ff.set_block(blk, tensor_ff)
        tensor_ff[blk].set_from_ndarray(temp.to_ndarray(), conv_tol)


def replicate_ao_block(mospaces, tensor, is_symmetric=True):
    """
    transform_operator_ao2mo requires the operator in AO to be
    replicated in a block-diagonal fashion (i.e. like [A 0
                                                       0 A].
    This is achieved using this function.
    """
    sym = libadcc.make_symmetry_operator_basis(
        mospaces, tensor.shape[0], is_symmetric
    )
    result = Tensor(sym)

    zerobk = np.zeros_like(tensor)
    result.set_from_ndarray(np.block([
        [tensor, zerobk],
        [zerobk, tensor],
    ]))
    return result


class OperatorIntegrals:
    def __init__(self, provider, mospaces, coefficients, conv_tol):
        self.__provider_ao = provider
        self.mospaces = mospaces
        self.__coefficients = coefficients
        self.__conv_tol = conv_tol

    @property
    def provider_ao(self):
        """
        The data structure which provides the integral data in the
        atomic orbital basis from the backend.
        """
        return self.__provider_ao

    @property
    def electric_dipole(self):
        """
        Return the electric dipole integrals in the molecular orbital basis.
        """
        if not hasattr(self.provider_ao, "electric_dipole"):
            raise NotImplementedError(
                "Electric dipole operator not implemented in "
                "{} backend.".format(self.provider_ao.backend)
            )

        dipoles = []
        for i, component in enumerate(["x", "y", "z"]):
            dip_backend = self.provider_ao.electric_dipole[i]
            dip_bb = replicate_ao_block(self.mospaces, dip_backend,
                                        is_symmetric=True)
            dip_ff = OneParticleOperator(self.mospaces, is_symmetric=True,
                                         cartesian_transform=component)
            transform_operator_ao2mo(dip_bb, dip_ff, self.__coefficients,
                                     self.__conv_tol)
            dipoles.append(dip_ff)
        return dipoles
