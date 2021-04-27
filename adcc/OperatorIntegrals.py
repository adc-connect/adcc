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
import numpy as np

from .misc import cached_property
from .Tensor import Tensor
from .timings import Timer, timed_member_call
from .OneParticleOperator import OneParticleOperator

import libadcc


def transform_operator_ao2mo(tensor_bb, tensor_ff, coefficients,
                             conv_tol=1e-14):
    """Take a block-diagonal tensor in the atomic orbital basis
    and transform it into the molecular orbital basis in the
    convention used by adcc.

    Parameters
    ----------
    tensor_bb : Tensor
        Block-diagonal tensor in the atomic orbital basis
    tensor_ff : Tensor
        Output tensor with the symmetry set-up to contain
        the operator in the molecular orbital representation
    coefficients : callable
        Function providing coefficient blocks
    conv_tol : float, optional
        SCF convergence tolerance, by default 1e-14
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
    ]), 1e-14)
    return result


class OperatorIntegrals:
    def __init__(self, provider, mospaces, coefficients, conv_tol):
        self.__provider_ao = provider
        self.mospaces = mospaces
        self.__coefficients = coefficients
        self.__conv_tol = conv_tol
        self._import_timer = Timer()

    @property
    def provider_ao(self):
        """
        The data structure which provides the integral data in the
        atomic orbital basis from the backend.
        """
        return self.__provider_ao

    @cached_property
    def available(self):
        """Which integrals are available in the underlying backend"""
        integrals = (
            "electric_dipole",
            "magnetic_dipole",
            "nabla",
            "pe_induction_elec"
        )
        return [integral for integral in integrals
                if hasattr(self.provider_ao, integral)]

    def import_dipole_like_operator(self, integral, is_symmetric=True):
        if integral not in self.available:
            raise NotImplementedError(f"{integral.replace('_', ' ')} operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        dipoles = []
        for i, component in enumerate(["x", "y", "z"]):
            dip_backend = getattr(self.provider_ao, integral)[i]
            dip_bb = replicate_ao_block(self.mospaces, dip_backend,
                                        is_symmetric=is_symmetric)
            dip_ff = OneParticleOperator(self.mospaces, is_symmetric=is_symmetric)
            transform_operator_ao2mo(dip_bb, dip_ff, self.__coefficients,
                                     self.__conv_tol)
            dipoles.append(dip_ff)
        return dipoles

    @property
    @timed_member_call("_import_timer")
    def electric_dipole(self):
        """Return the electric dipole integrals in the molecular orbital basis."""
        return self.import_dipole_like_operator("electric_dipole",
                                                is_symmetric=True)

    @property
    @timed_member_call("_import_timer")
    def magnetic_dipole(self):
        """Return the magnetic dipole integrals in the molecular orbital basis."""
        return self.import_dipole_like_operator("magnetic_dipole",
                                                is_symmetric=False)

    @property
    @timed_member_call("_import_timer")
    def nabla(self):
        """
        Return the momentum (nabla operator) integrals
        in the molecular orbital basis.
        """
        return self.import_dipole_like_operator("nabla", is_symmetric=False)

    def __import_density_dependent_operator(self, ao_callback, is_symmetric=True):
        """Returns a function that imports a density-dependent operator.
        The returned function imports the operator and transforms it to the
        molecular orbital basis.

        Parameters
        ----------
        ao_callback : callable
            Function that computes the operator in atomic orbitals using
            a :py:class:`OneParticleOperator` (the density matrix
            in atomic orbitals) as single argument
        is_symmetric : bool, optional
            if the imported operator is symmetric, by default True
        """
        def process_operator(dm, callback=ao_callback, is_symmetric=is_symmetric):
            dm_ao = sum(dm.to_ao_basis())
            v_ao = callback(dm_ao)
            v_bb = replicate_ao_block(
                self.mospaces, v_ao, is_symmetric=is_symmetric
            )
            v_ff = OneParticleOperator(self.mospaces, is_symmetric=is_symmetric)
            transform_operator_ao2mo(
                v_bb, v_ff, self.__coefficients, self.__conv_tol
            )
            return v_ff
        return process_operator

    @property
    def pe_induction_elec(self):
        """
        Returns a function to obtain the (density-dependent)
        PE electronic induction operator in the molecular orbital basis
        """
        if "pe_induction_elec" not in self.available:
            raise NotImplementedError("PE electronic induction operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")
        callback = self.provider_ao.pe_induction_elec
        return self.__import_density_dependent_operator(callback)

    @property
    def timer(self):
        ret = Timer()
        ret.attach(self._import_timer, subtree="import")
        return ret
