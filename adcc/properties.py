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
import numpy as np

from .state_densities import attach_state_densities
from .OneParticleOperator import HfDensityMatrix, product_trace

from copy import copy


def transition_dipole_moments(state):
    """
    Compute the ground to excited state transition dipole moments
    from a solver state (in a.u.)
    """
    if not hasattr(state, "ground_to_excited_tdms"):
        raise ValueError("Cannot compute transition dipole moment without"
                         " transition densities.")
    if not hasattr(state.reference_state.operator_integrals, "electric_dipole"):
        raise ValueError("Reference state cannot provide electric dipole"
                         " integrals.")
    comps = ['x', 'y', 'z']
    dips = state.reference_state.operator_integrals.electric_dipole
    tdip_moments = np.array([
        [product_trace(dips(k), tdm) for k in comps]
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


def dipole_moments(state):
    """
    Compute the dipole moment of each excited state
    from a solver state (in a.u.), including contributions from
    the ground state dipole moment.
    """
    if not hasattr(state, "state_diffdms"):
        raise ValueError("Cannot compute transition dipole moment without"
                         " state densities.")
    if not hasattr(state.reference_state.operator_integrals, "electric_dipole"):
        raise ValueError("Reference state cannot provide electric dipole"
                         " integrals.")
    comps = ['x', 'y', 'z']
    dips = state.reference_state.operator_integrals.electric_dipole
    # TODO: what about ADC(1)?
    mp2_diffdm = state.ground_state.mp2_diffdm
    hf_dm = HfDensityMatrix(mospaces=state.reference_state.mospaces)
    hf_dip_moment = np.array([
        product_trace(dips(k), hf_dm) for k in comps
    ])
    mp_dip_moment = np.array([
        product_trace(dips(k), mp2_diffdm) for k in comps
    ])
    gs_dip_moment = hf_dip_moment + mp_dip_moment
    state_dip_moments = np.array([
        [product_trace(dips(k), ddm) for k in comps]
        for ddm in state.state_diffdms
    ])
    nuc_dip = state.reference_state.operator_integrals.nuclear_dipole()
    state_dip_moments += gs_dip_moment - nuc_dip
    return state_dip_moments


def attach_properties(solver_state, transition_properties=True,
                      state_properties=True, method=None):
    """
    Compute state and transition properties from a solver
    state and attach them to the state

    @param solver_state     The solver state to work on.

    @transition_properties  Compute transition properties

    @state_properties       Compute properties of the excited states
    """
    solver_state = copy(solver_state)
    if state_properties:
        if not hasattr(solver_state, "state_diffdms"):
            solver_state = attach_state_densities(
                solver_state, state_diffdm=True, ground_to_excited_tdm=False,
                method=method
            )
        solver_state.state_dipole_moments = dipole_moments(
            solver_state
        )

    if transition_properties:
        if not hasattr(solver_state, "ground_to_excited_tdms"):
            solver_state = attach_state_densities(
                solver_state, state_diffdm=False, ground_to_excited_tdm=True,
                method=method
            )
        solver_state.transition_dipole_moments = transition_dipole_moments(
            solver_state
        )
        solver_state.oscillator_strengths = oscillator_strengths(
            solver_state.transition_dipole_moments, solver_state.eigenvalues
        )
    return solver_state
