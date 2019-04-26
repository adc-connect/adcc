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
from .AdcMethod import AdcMethod
from .AmplitudeVector import AmplitudeVector

from copy import copy

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
    return libadcc.compute_state_diffdm(method.property_method, ground_state,
                                        amplitude.to_cpp(), intermediates)


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
    return libadcc.compute_gs2state_optdm(method.property_method, ground_state,
                                          amplitude.to_cpp(), intermediates)


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
    return libadcc.compute_state2state_optdm(method.property_method,
                                             ground_state,
                                             amplitude_from.to_cpp(),
                                             amplitude_to.to_cpp(),
                                             intermediates)


def attach_state_densities(solver_state, ground_state=None,
                           method=None, state_diffdm=True,
                           ground_to_excited_tdm=True,
                           state_to_state_tdm=False,
                           overwrite_existing=False,
                           intermediates=None):
    """
    Compute one-particle state density matrices and
    one-particle transition density matrices for all eigenvectors
    in the passed solver state and attach the results to the
    state. The function returns the result and modifies the
    state object.

    @param solver_state    The solver state to work on.

    @param ground_state
    The ground state upon which the excited states calculation is based.
    By default the ground_state stored inside the solver_state is used.

    @param method
    The method to use for the density calculation. By default the method
    stored inside the solver_state is used. For ADC(3) where the ISR is not yet
    implemented one probably wants to explicitly change this to "adc2".

    @param state_diffdm
    Flag whether to compute the excited state one-particle difference density
    matrices. These will be stored inside the attribute 'state_diffdms'
    of the state as a list in the same order as the eigenvectors.

    @param ground_to_excited_tdm
    Flag whether to compute the ground to excited state one-particle
    transition density matrices. These will be stored inside
    the attribute 'ground_to_excited_tdms' as a list in the same order
    as the eigenvectors.

    @param state_to_state_tdm
    Flag wether to compute the state to state one-particle transition
    density matrices. These will be stored inside the attribute
    'state_to_state_tdm' of the state as a dict. The dict maps
    the tuple (index_from, index_to) to the resulting transition
    density matrix. In this index_from and index_to are the indices
    of the respective states in the 'eigenvectors'
    list of the solver_state object.

    @param overwrite_existing
    Whether to silently overwrite existing
    attributes inside the solver_state object (if the flag is True)
    or to alternatively throw an ArgumentError (default, if the
    flag is False).

    @param intermediates
    Intermediates from the ADC calculation to reuse for the property
    calculation. If not passed required intermediates will be recomputed.
    """
    solver_state = copy(solver_state)

    if not hasattr(solver_state, "eigenvectors"):
        raise TypeError("The solver_state objects is invalid since it has no "
                        "'eigenvectors' attribute.")

    if method is None:
        if not hasattr(solver_state, "method"):
            raise TypeError("The solver_state objects is invalid since it has "
                            "no 'method' attribute.")
        method = solver_state.method

    if ground_state is None:
        if not hasattr(solver_state, "ground_state"):
            raise TypeError("The solver_state objects is invalid since it has "
                            "no 'ground_state' attribute.")
        ground_state = solver_state.ground_state

    if intermediates is None:
        intermediates = libadcc.AdcIntermediates(ground_state)

    if state_diffdm:
        if hasattr(solver_state, "state_diffdms") and not overwrite_existing:
            raise ValueError("Bailing out of computing state densities: "
                             "The solver_state object already has a 'state_dm' "
                             "attribute and overwrite_existing is False")
        solver_state.state_diffdms = [
            compute_state_diffdm(method, ground_state, amplitude, intermediates)
            for amplitude in solver_state.eigenvectors
        ]

    if ground_to_excited_tdm:
        if hasattr(solver_state, "ground_to_excited_tdms") and \
           not overwrite_existing:
            raise ValueError("Bailing out of computing ground to excited state "
                             "transition densities: The solver_state object "
                             "already has a 'ground_to_excited_tdms' "
                             "attribute and overwrite_existing is False")
        solver_state.ground_to_excited_tdms = [
            compute_gs2state_optdm(method, ground_state, amplitude,
                                   intermediates)
            for amplitude in solver_state.eigenvectors
        ]

    if state_to_state_tdm:
        # Store into state.state_to_state_tdms as map as described above
        raise NotImplementedError("state_to_state_tdm")

    return solver_state
