#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from math import sqrt

from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

from .util import check_doubles_amplitudes, check_singles_amplitudes


def diffdm_ip_adc0(mp, amplitude, intermediates):
    check_singles_amplitudes([b.o], amplitude)
    u1 = amplitude.h

    dm = OneParticleOperator(mp, is_symmetric=True)
    dm.oo = -einsum("j,i->ij", u1, u1)
    return dm


def diffdm_ip_adc2(mp, amplitude, intermediates):
    dm = diffdm_ip_adc0(mp, amplitude, intermediates)  # Get ADC(0/1) result
    check_doubles_amplitudes([b.o, b.o, b.v], amplitude)
    u1, u2 = amplitude.h, amplitude.phh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_oo = dm.oo.evaluate()  # ADC(1) diffdm

    # Zeroth order doubles contributions
    p2_oo = 2 * einsum("kja,ika->ij", u2, u2)
    p2_vv = einsum("ija,ijb->ab", u2, u2)
    p_ov = sqrt(2) * einsum("j,ija->ia", u1, u2)

    # ADC(2) ISR intermediate (TODO Move to intermediates)
    # ru1 = einsum("i,ijab->jab", u1, t2).evaluate()

    # Compute second-order contributions to the density matrix
    dm.oo = (  # ip_adc2_p_oo
        + p1_oo + p2_oo
        - 0.5 * einsum("k,jk,i->ij", u1, p0.oo, u1)
        - 0.5 * einsum("j,ki,k->ij", u1, p0.oo, u1)
        + 0.5 * einsum("iab,jab->ij", einsum("k,kiab->iab", u1, t2),
                       einsum("l,ljab->jab", u1, t2))
    )

    dm.vv = (  # ip_adc2_p_vv
        + p2_vv
        - einsum("kcb,kca->ab", einsum("i,kicb->kcb", u1, t2),
                 einsum("j,kjca->kca", u1, t2))
    )

    dm.ov = (  # ip_adc2_p_ov
        + p_ov
        + 1/sqrt(2) * (
            + einsum("klb,klba,i->ia", u2, t2, u1)
            + 2 * einsum("kb,ikba->ia", einsum("klb,l->kb", u2, u1), t2))
        - einsum("k,ka,i->ia", u1, p0.ov, u1)
    )
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "ip_adc0": diffdm_ip_adc0,
    "ip_adc1": diffdm_ip_adc0,       # same as ADC(0)
    "ip_adc2": diffdm_ip_adc2,
    "ip_adc2x": diffdm_ip_adc2,      # same as ADC(2)
}


def state_diffdm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle difference density matrix of an excited state
    in the MO basis.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("state_diffdm is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
