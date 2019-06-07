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


class OperatorIntegrals:
    def __init__(self, provider, mospaces, coefficients, conv_tol):
        self.provider = provider
        self.mospaces = mospaces
        self.coefficients = coefficients
        self.conv_tol = conv_tol

    def transform_backend_to_adcc(self, op_backend, op_adcc):
        """
        Transform an operator from the backend (given in the atomic
        orbital basis) to the molecular orbital basis used in adcc.
        """
        sym = libadcc.make_symmetry_operator_basis(
            self.mospaces, op_backend.shape[0], op_adcc.is_symmetric
        )
        tensor_bb = Tensor(sym)

        A = op_backend
        Z = np.zeros_like(A)
        tensor_bb.set_from_ndarray(np.block([
            [A, Z],
            [Z, A],
        ]), self.conv_tol)

        for blk in op_adcc.blocks:
            assert len(blk) == 4
            cleft = self.coefficients(blk[:2] + "b")
            cright = self.coefficients(blk[2:] + "b")
            tensor_ff = cleft @ tensor_bb @ cright.transpose()

            # TODO: once the permutational symmetry is correct:
            # op_adcc.set_block(blk, tensor_ff)
            op_adcc[blk].set_from_ndarray(tensor_ff.to_ndarray(), self.conv_tol)

    @property
    def electric_dipole(self):
        if not hasattr(self.provider, "electric_dipole"):
            raise NotImplementedError(
                "Electric dipole operator not implemented in "
                "{} backend.".format(self.provider.backend)
            )

        dipoles = []
        for i, component in enumerate(["x", "y", "z"]):
            dip = OneParticleOperator(self.mospaces, is_symmetric=True,
                                      cartesian_transform=component)
            dip_backend = self.provider.electric_dipole[i]
            self.transform_backend_to_adcc(dip_backend, dip)
            dipoles.append(dip)
        return dipoles
