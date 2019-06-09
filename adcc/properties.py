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
import warnings
import numpy as np

from copy import copy

from .AdcMethod import AdcMethod
from .state_densities import attach_state_densities
from .OneParticleOperator import HfDensityMatrix, product_trace

__all__ = ["attach_properties"]


def transition_dipole_moments(state, method=None):
    """
    Compute the ground to excited state transition dipole moments
    from a solver state (in a.u.)
    """
    if not hasattr(state, "ground_to_excited_tdms"):
        raise ValueError("Cannot compute transition dipole moment without"
                         " transition densities.")
    if not hasattr(state.reference_state.operator_integrals, "electric_dipole"):
        raise ValueError("Reference state for backend "
                         + state.reference_state.backend + "cannot provide "
                         "electric dipole integrals.")
    if method is None:
        method = state.method
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if method.level == 0:
        warnings.warn("ADC(0) transition dipole moments are known to be faulty "
                      "in some cases.")

    dipole_integrals = state.reference_state.operator_integrals.electric_dipole
    tdip_moments = np.array([
        [product_trace(comp, tdm) for comp in dipole_integrals]
        for tdm in state.ground_to_excited_tdms
    ])
    return tdip_moments


def oscillator_strengths(transition_dipole_moments, energies):
    """
    General routine to compute the oscillator strengths from a list of
    transition dipole moments and energies.

    f = 2/3 * <tdipmom>^2 * E

    Can be used for ground->excited state and excited state->excited state
    transitions.
    """
    oscs = 2. / 3. * np.array([
        np.linalg.norm(tdm)**2 * np.abs(ev)
        for tdm, ev in zip(transition_dipole_moments, energies)
    ])
    return oscs


def state_dipole_moments(state, method=None):
    """
    Compute the dipole moment of each excited state
    from a solver state (in a.u.), including contributions from
    the ground state dipole moment.
    """
    if not hasattr(state, "state_diffdms"):
        raise ValueError("Cannot compute transition dipole moment without"
                         " state densities.")
    if not hasattr(state.reference_state.operator_integrals, "electric_dipole"):
        raise ValueError("Reference state for backend "
                         + state.reference_state.backend + "cannot provide "
                         "electric dipole integrals.")
    if method is None:
        method = state.method
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)

    dipole_integrals = state.reference_state.operator_integrals.electric_dipole

    # TODO Have a total density up to MP level 2 function in LazyMp
    # Compute ground-state dipole moment (TODO Put this into LazyMp?),
    # noting that the nuclear contribution carries the opposite sign
    # as the electronic contribution (due to the opposite
    # charge of the nuclei)
    gsdm = HfDensityMatrix(state.reference_state)
    if method.level > 1:
        gsdm += state.ground_state.mp2_diffdm
    # end changes implied by TODO

    gs_dip_moment = state.reference_state.nuclear_dipole - np.array([
        product_trace(comp, gsdm) for comp in dipole_integrals
    ])
    return gs_dip_moment - np.array([
        [product_trace(comp, ddm) for comp in dipole_integrals]
        for ddm in state.state_diffdms
    ])


def attach_state_properties(state, method=None):
    if not hasattr(state, "state_diffdms"):
        state = attach_state_densities(state, state_diffdm=True,
                                       ground_to_excited_tdm=False,
                                       state_to_state_tdm=False,
                                       method=method)
    state.state_dipole_moments = state_dipole_moments(state, method=method)
    return state


def attach_transition_properties(state, method=None):
    if not hasattr(state, "ground_to_excited_tdms"):
        state = attach_state_densities(state, state_diffdm=False,
                                       ground_to_excited_tdm=True,
                                       state_to_state_tdm=False, method=method)
    state.transition_dipole_moments = transition_dipole_moments(state, method)
    state.oscillator_strengths = oscillator_strengths(
        state.transition_dipole_moments, state.eigenvalues
    )
    return state


def attach_properties(state, transition_properties=True,
                      state_properties=True, method=None):
    """
    Compute state and transition properties from a solver
    state and attach them to the state

    @param state     The solver state to work on.
    @transition_properties  Compute transition properties
    @state_properties       Compute properties of the excited states
    @method                 The method to use for property calculations
    """
    state = copy(state)
    if state_properties:
        state = attach_state_properties(state, method)
    if transition_properties:
        state = attach_transition_properties(state, method)
    return state
