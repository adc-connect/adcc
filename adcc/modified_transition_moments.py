#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import libadcc

from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod
from .AmplitudeVector import AmplitudeVector

from libadcc import AdcIntermediates, LazyMp

from adcc import einsum

from math import sqrt


def compute_modified_transition_moments(gs_or_matrix, dipole_operator,
                                        method=None):
    """Compute the modified transition moments (MTM) for the provided
    ADC method with reference to the passed ground state and
    the appropriate dipole integrals in the MO basis.

    Parameters
    ----------
    gs_or_matrix
        The MP ground state or ADC matrix for which level of theory
        the modified transition moments are to be computed

    dipole_operator: OneParticleOperator
        Electric dipole operator

    method: optional
        Provide an explicit method to override the automatic selection.
        Only required if LazyMp is provided as gs_or_matrix

    Returns
    -------
    adcc.AmplitudeVector
    """
    if isinstance(gs_or_matrix, AdcMatrix):
        ground_state = gs_or_matrix.ground_state
        if method is None:
            method = gs_or_matrix.method
        elif not isinstance(method, AdcMethod):
            method = AdcMethod(method)
    elif isinstance(gs_or_matrix, LazyMp):
        ground_state = gs_or_matrix
        if method is None:
            raise ValueError("A method must be provided if only LazyMp object"
                             " is supplied.")
        elif not isinstance(method, AdcMethod):
            method = AdcMethod(method)
    else:
        raise TypeError("gs_or_matrix should be a LazyMp or AdcMatrix object.")
    if not isinstance(dipole_operator, libadcc.OneParticleOperator):
        raise TypeError("dipole_operator should be "
                        "libadcc.OneParticleOperator.")

    if method.is_core_valence_separated:
        return modified_transition_moments_cvs(gs_or_matrix,
                                               dipole_operator, method)
    if method.name == "adc2":
        mtm_cpp = libadcc.compute_modified_transition_moments(
            method.name, ground_state, dipole_operator
        )
        return AmplitudeVector(*mtm_cpp.to_tuple())
    else:
        return modified_transition_moments(gs_or_matrix,
                                           dipole_operator, method)


def modified_transition_moments_cvs(gs_or_matrix, dipole_operator,
                                    method=None):
    if method.name == "cvs-adc0" or method.name == "cvs-adc1":
        return AmplitudeVector(dipole_operator['o2v1'])

    elif method.name == "cvs-adc2":
        if isinstance(gs_or_matrix, AdcMatrix):
            ground_state = gs_or_matrix.ground_state
            intermediates = gs_or_matrix.intermediates
        elif isinstance(gs_or_matrix, LazyMp):
            ground_state = gs_or_matrix
            intermediates = AdcIntermediates(ground_state)

        rho_ov = intermediates.cv_p_ov
        rho_vv = intermediates.cv_p_vv
        d_co = dipole_operator['o2o1']
        d_cv = dipole_operator['o2v1']

        mp_t2_oovv = ground_state.t2("o1o1v1v1")

        f1 = (
            + d_cv
            - einsum("Ib,ba->Ia", d_cv, rho_vv)
            - einsum("Ij,ja->Ia", d_co, rho_ov)
        )

        f2 = + 1.0 / sqrt(2) * einsum("Ik,kjab->jIab", d_co, mp_t2_oovv)
        return AmplitudeVector(f1, f2)
    else:
        raise NotImplementedError("compute_modified_transition_moments "
                                  "not implemented for", method.name)


def modified_transition_moments(gs_or_matrix, dipole_operator,
                                method=None):
    if method.name == "adc0":
        return AmplitudeVector(dipole_operator['o1v1'])

    elif method.name == "adc1":
        if isinstance(gs_or_matrix, AdcMatrix):
            ground_state = gs_or_matrix.ground_state
        elif isinstance(gs_or_matrix, LazyMp):
            ground_state = gs_or_matrix

        d_ov = dipole_operator['o1v1']
        mp_t2_oovv = ground_state.t2("o1o1v1v1")

        return AmplitudeVector(
            d_ov - einsum("ijab,jb->ia", mp_t2_oovv, d_ov)
        )
    else:
        raise NotImplementedError("compute_modified_transition_moments "
                                  "not implemented for", method.name)
