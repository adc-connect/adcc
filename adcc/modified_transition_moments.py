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
import libadcc

from .AdcMethod import AdcMethod
from .AmplitudeVector import AmplitudeVector


def compute_modified_transition_moments(ground_state, method,
                                        dipole_operator):
    """
    Compute the modified transition moments (MTM) for the provided
    ADC method with reference to the passed ground state and
    the appropriate dipole integrals in the MO basis.

    It is expected that dipole_integrals is a dictionary with the
    keys oo, ov, vv containing the occupied-occupied, ocupied-virtual
    and virtual-virtual blocks of the dipole integrals

    Note: This interface is not ideal and is likely to change.

    The MTM are returned as an AmplitudeVector.
    """
    # TODO Think about how this generalises to higher-order methods
    #      than ADC(2): Is this interface then still working?

    # TODO Also: how about using the ADC matrix as the first argument
    #      instead of the ground state and the method. I feel that's
    #      more likely to hold for other ADC cases, too

    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, libadcc.LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(dipole_operator, libadcc.OneParticleOperator):
        raise TypeError("dipole_operator should be "
                        "libadcc.OneParticleOperator.")

    if method.level != 2:
        raise NotImplementedError("compute_modified_transition_moments "
                                  "only implemented for ADC(2).")

    if method.is_core_valence_separated:
        raise NotImplementedError("compute_modified_transition_moments "
                                  "not implemented for CVS")

    mtm_cpp = libadcc.compute_modified_transition_moments(
        method.name, ground_state, dipole_operator
    )
    return AmplitudeVector(*mtm_cpp.to_tuple())
