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
from .AdcMethod import AdcMethod
from .AmplitudeVector import AmplitudeVector
from .OneParticleOperator import OneParticleOperator

import libadcc


def compute_state_diffdm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle difference density matrix of an excited state
    in the MO basis.

    @param method        The method to use for the computation (e.g. "adc2")
    @param ground_state  The ground state upon which the excitation was based
    @param amplitude     The amplitude vector
    @param intermediates   Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, libadcc.LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = libadcc.AdcIntermediates(ground_state)

    ret = OneParticleOperator.from_cpp(libadcc.compute_state_diffdm(
        method.property_method, ground_state, amplitude.to_cpp(), intermediates
    ))
    ret.reference_state = ground_state.reference_state
    return ret


def compute_gs2state_optdm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle transition density matrix from ground to excited
    state in the MO basis.

    @param method        The method to use for the computation (e.g. "adc2")
    @param ground_state  The ground state upon which the excitation was based
    @param amplitude     The amplitude vector
    @param intermediates   Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, libadcc.LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = libadcc.AdcIntermediates(ground_state)

    ret = OneParticleOperator.from_cpp(libadcc.compute_gs2state_optdm(
        method.property_method, ground_state, amplitude.to_cpp(), intermediates
    ))
    ret.reference_state = ground_state.reference_state
    return ret


def compute_state2state_optdm(method, ground_state, amplitude_from,
                              amplitude_to, intermediates=None):
    """
    Compute the state to state transition density matrix
    state in the MO basis using the intermediate-states representation.

    @param method          The method to use for the computation (e.g. "adc2")
    @param ground_state    The ground state upon which the excitation was based
    @param amplitude_from  The amplitude vector of the state to start from
    @param amplitude_to    The amplitude vector of the state to excite to
    @param intermediates   Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, libadcc.LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude_from, AmplitudeVector):
        raise TypeError("amplitude_from should be an AmplitudeVector object.")
    if not isinstance(amplitude_to, AmplitudeVector):
        raise TypeError("amplitude_to should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = libadcc.AdcIntermediates(ground_state)

    ret = OneParticleOperator.from_cpp(libadcc.compute_state2state_optdm(
        method.property_method, ground_state, amplitude_from.to_cpp(),
        amplitude_to.to_cpp(), intermediates
    ))
    ret.reference_state = ground_state.reference_state
    return ret
