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

import libadcc

from .OneParticleOperator import OneParticleOperator
from .Tensor import Tensor


def make_block_diagonal_tensor(tensor):
    return np.vstack((
        np.hstack((tensor, np.zeros_like(tensor))),
        np.hstack((np.zeros_like(tensor), tensor))
    ))


class OperatorIntegrals:
    def __init__(self, provider, mospaces, coefficients,
                 conv_tol):
        self.provider = provider
        self.mospaces = mospaces
        self.coefficients = coefficients
        self.conv_tol = conv_tol

    def transform_bb_to_ff(self, block, tensor_bb):
        c1 = self.coefficients(block[:2] + "b")
        c2 = self.coefficients(block[2:] + "b")
        return c1 @ tensor_bb @ c2.transpose()

    def transform_backend_to_adcc(self, op_backend, op_adcc, is_symmetric=True):
        sym = libadcc.make_symmetry_operator_basis(
            self.mospaces, op_backend.shape[0], is_symmetric
        )
        tensor_bb = Tensor(sym)
        tensor_bb.set_from_ndarray(
            make_block_diagonal_tensor(op_backend), self.conv_tol
        )
        for b in op_adcc.blocks:
            op_adcc.set_block(b, self.transform_bb_to_ff(b, tensor_bb))

    def electric_dipole(self, component="x"):
        if not hasattr(self.provider, "electric_dipole"):
            raise NotImplementedError("Electric dipole operator not implemented"
                                      " in {}.".format(self.provider.backend))
        elec_dip = OneParticleOperator(self.mospaces, is_symmetric=True,
                                       cartesian_transform=component)
        ao_dip = self.provider.electric_dipole(component)
        self.transform_backend_to_adcc(ao_dip, elec_dip, is_symmetric=True)
        return elec_dip

    def fock(self):
        if not hasattr(self.provider, "fock"):
            raise NotImplementedError("Fock operator not implemented"
                                      " in {}.".format(self.provider.backend))
        fock = OneParticleOperator(self.mospaces, is_symmetric=True,
                                   cartesian_transform="1")
        ao_fock = self.provider.fock()
        self.transform_backend_to_adcc(ao_fock, fock, is_symmetric=True)
        return fock
