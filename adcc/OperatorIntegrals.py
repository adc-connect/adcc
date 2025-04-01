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

from .misc import cached_property, cached_member_function
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
        self._provider_ao = provider
        self.mospaces = mospaces
        self._coefficients = coefficients
        self._conv_tol = conv_tol
        self._import_timer = Timer()

    @property
    def provider_ao(self):
        """
        The data structure which provides the integral data in the
        atomic orbital basis from the backend.
        """
        return self._provider_ao

    @property
    def available(self) -> tuple[str]:
        """Which integrals are available in the underlying backend"""
        return self.provider_ao.available

    def _import_dipole_like_operator(self, integral: str, is_symmetric: bool = True
                                     ) -> tuple[OneParticleOperator]:
        if integral not in self.available:
            raise NotImplementedError(f"{integral.replace('_', ' ')} operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        ao_operator = getattr(self.provider_ao, integral)
        assert len(ao_operator) == 3  # has to have a x, y and z component

        dipoles = []
        for comp in range(3):  # [x, y, z]
            dip_bb = replicate_ao_block(self.mospaces, ao_operator[comp],
                                        is_symmetric=is_symmetric)
            dip_ff = OneParticleOperator(self.mospaces, is_symmetric=is_symmetric)
            transform_operator_ao2mo(dip_bb, dip_ff, self._coefficients,
                                     self._conv_tol)
            dipoles.append(dip_ff)
        return tuple(dipoles)

    @cached_property
    @timed_member_call("_import_timer")
    def electric_dipole(self) -> tuple[OneParticleOperator]:
        """Return the electric dipole integrals in the molecular orbital basis."""
        return self._import_dipole_like_operator("electric_dipole",
                                                 is_symmetric=True)

    @cached_property
    @timed_member_call("_import_timer")
    def electric_dipole_velocity(self) -> tuple[OneParticleOperator]:
        """
        Return the electric dipole integrals (in the velocity gauge)
        in the molecular orbital basis.
        """
        return self._import_dipole_like_operator("electric_dipole_velocity",
                                                 is_symmetric=False)

    def _import_g_origin_dep_dip_like_operator(self, integral: str,
                                               gauge_origin="origin",
                                               is_symmetric: bool = True
                                               ) -> tuple[OneParticleOperator]:
        """
        Imports the operator and transforms it to the molecular orbital basis.

        Parameters
        ----------
        integral : str
            The dipole like gauge dependent integral to import: an integral
            that consists of 3 components (x, y, z) and whose AO import function
            takes the gauge origin as argument.
        gauge_origin: str or tuple[str]
            The gauge origin used for the generation of the AO integrals.
        is_symmetric : bool, optional
            If the imported operator is symmetric, by default True
        """
        if integral not in self.available:
            raise NotImplementedError(f"{integral} operator is not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        ao_operator = getattr(self.provider_ao, integral)(gauge_origin)
        assert len(ao_operator) == 3  # has to have a x, y and z component

        dipoles = []
        for comp in range(3):  # [x, y, z]
            dip_bb = replicate_ao_block(self.mospaces, ao_operator[comp],
                                        is_symmetric=is_symmetric)
            dip_ff = OneParticleOperator(self.mospaces, is_symmetric=is_symmetric)
            transform_operator_ao2mo(dip_bb, dip_ff, self._coefficients,
                                     self._conv_tol)
            dipoles.append(dip_ff)
        return tuple(dipoles)

    # separate the timings, so one can easily see in the timings how many different
    # gauge_origins were used throughout the calculation
    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def magnetic_dipole(self, gauge_origin="origin") -> tuple[OneParticleOperator]:
        """
        Returns the magnetic dipole intergrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_dip_like_operator(
            integral="magnetic_dipole", gauge_origin=gauge_origin,
            is_symmetric=False
        )

    def _import_g_origin_dep_quad_like_operator(self, integral: str,
                                                gauge_origin="origin",
                                                is_symmetric: bool = True
                                                ) -> tuple[tuple[OneParticleOperator]]:  # noqa E501
        """
        Imports the operator and transforms it to the molecular orbital basis.

        Parameters
        ----------
        integral : str
            The quadrupole like gauge dependent integral to import: an integral
            that consists of 9 components (xx, xy, xz, ... zz)
            and whose AO import function takes the gauge origin as single argument.
        gauge_origin: str or tuple[str]
            The gauge origin used for the generation of the AO integrals.
        is_symmetric : bool, optional
            if the imported operator is symmetric, by default True
        """
        if integral not in self.available:
            raise NotImplementedError(f"{integral} operator is not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        ao_operator = getattr(self.provider_ao, integral)(gauge_origin)
        assert len(ao_operator) == 9

        flattened = []
        for comp in range(9):  # [xx, xy, xz, yx, yy, yz, zx, zy, zz]
            quad_bb = replicate_ao_block(self.mospaces, ao_operator[comp],
                                         is_symmetric=is_symmetric)
            quad_ff = OneParticleOperator(self.mospaces, is_symmetric=is_symmetric)
            transform_operator_ao2mo(quad_bb, quad_ff, self._coefficients,
                                     self._conv_tol)
            flattened.append(quad_ff)
        return (tuple(flattened[:3]), tuple(flattened[3:6]), tuple(flattened[6:]))

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole(self, gauge_origin="origin"
                            ) -> tuple[tuple[OneParticleOperator]]:
        """
        Returns the electric quadrupole integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_quad_like_operator(
            integral="electric_quadrupole", gauge_origin=gauge_origin,
            is_symmetric=True
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole_traceless(self, gauge_origin="origin"
                                      ) -> tuple[tuple[OneParticleOperator]]:
        """
        Returns the traceless electric quadrupole integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_quad_like_operator(
            integral="electric_quadrupole_traceless", gauge_origin=gauge_origin,
            is_symmetric=True
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole_velocity(self, gauge_origin="origin"
                                     ) -> tuple[tuple[OneParticleOperator]]:
        """
        Returns the electric quadrupole integrals in velocity gauge
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_quad_like_operator(
            integral="electric_quadrupole_velocity", gauge_origin=gauge_origin,
            is_symmetric=False
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def diamagnetic_magnetizability(self, gauge_origin="origin"
                                    ) -> tuple[tuple[OneParticleOperator]]:
        """
        Returns the diamagnetic magnetizability integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_quad_like_operator(
            integral="diamagnetic_magnetizability", gauge_origin=gauge_origin,
            is_symmetric=True
        )

    def _import_density_dependent_operator(self, operator: str,
                                           density_mo: OneParticleOperator,
                                           is_symmetric: bool = True
                                           ) -> OneParticleOperator:
        """
        Import the density-dependent operator and transform it to the
        molecular orbital basis.

        Parameters
        ----------
        integral : str
            The density-dependent operator to import: an operator
            whose AO import function takes a density matrix as single argument.
        density_mo: OneParticleOperator
            The density in the MO basis for which to compute the operator.
        is_symmetric : bool, optional
            if the imported operator is symmetric, by default True
        """
        dm_ao = sum(density_mo.to_ao_basis())
        v_ao = getattr(self.provider_ao, operator)(dm_ao)
        if v_ao is None:
            raise NotImplementedError("Could not compute the density dependent "
                                      f"operator {operator} in backend"
                                      f"{self.provider_ao.backend}.")
        v_bb = replicate_ao_block(
            self.mospaces, v_ao, is_symmetric=is_symmetric
        )
        v_ff = OneParticleOperator(self.mospaces, is_symmetric=is_symmetric)
        transform_operator_ao2mo(
            v_bb, v_ff, self._coefficients, self._conv_tol
        )
        return v_ff

    def pe_induction_elec(self,
                          density_mo: OneParticleOperator) -> OneParticleOperator:
        """
        Returns the (density-dependent) PE electronic induction operator in the
        molecular orbital basis.
        """
        if "pe_induction_elec" not in self.available:
            raise NotImplementedError("PE electronic induction operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")
        return self._import_density_dependent_operator(
            operator="pe_induction_elec", density_mo=density_mo, is_symmetric=True
        )

    def pcm_potential_elec(self,
                           density_mo: OneParticleOperator) -> OneParticleOperator:
        """
        Returns the (density-dependent) electronic PCM potential operator in the
        molecular orbital basis
        """
        if "pcm_potential_elec" not in self.available:
            raise NotImplementedError("Electronic PCM potential operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")
        return self._import_density_dependent_operator(
            operator="pcm_potential_elec", density_mo=density_mo, is_symmetric=True
        )

    @property
    def timer(self):
        ret = Timer()
        ret.attach(self._import_timer, subtree="import")
        return ret
